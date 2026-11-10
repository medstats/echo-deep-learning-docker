"""Offline, CPU-only inference worker. Run via sh runtime/python."""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, os, resource, subprocess, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parent

def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def check_finite(value):
    if isinstance(value, dict):
        return all(check_finite(v) for v in value.values())
    return bool(torch.isfinite(torch.as_tensor(value)).all())

def checkpoint(path):
    # Official checkpoints only, fetched from the URLs in weights/manifest.json.
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError as e:
        if 'mmap' not in str(e):
            raise
        return torch.load(path, map_location='cpu', weights_only=False)

def view_label(path):
    import re
    for v in ['A4C', 'A2C', 'A3C', 'PLAX', 'PSAX', 'A5C']:
        if re.search(r'(?:^|_)'+v+r'(?:_|\.|$)', path.name, flags=re.I):
            return v
    return 'UNKNOWN'

def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        return {'file': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                'view': view_label(path), 'view_source': 'filename',
                'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()

def read_selected(path, indices, transform):
    """Decode incrementally, keep only selected resized/preprocessed frames."""
    indices = [int(i) for i in indices]
    wanted = set(indices)
    cap = cv2.VideoCapture(str(path))
    found = {}
    try:
        for i in range(max(indices)+1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Truncated video at frame {i}, expected through {max(indices)}')
            if i in wanted:
                found[i] = transform(frame)
    finally:
        cap.release()
    return [found[i] for i in indices]

def echonet_frame(frame):
    # Official ConvertDICOMToAVI notebook: first near-black row, center square,
    # 10% border crop, cubic resize and fixed scan-sector mask. Inputs already
    # have overlays removed; use the same spatial operations without re-encoding.
    rows = frame.mean(axis=(1, 2))
    blank = np.flatnonzero(rows < 1)
    # The official DICOM heuristic assumes a black header row before the sector.
    # These sector-only exports may first turn black BELOW the heart; never
    # interpret that bottom border as a header to crop away.
    if len(blank) and int(blank[0]) < frame.shape[0] * 0.15:
        frame = frame[int(blank[0]):]
    h, w = frame.shape[:2]
    bias = abs(w-h)//2
    if bias:
        frame = frame[:, bias:-bias] if h < w else frame[bias:-bias]
    h = frame.shape[0]
    border = h//10
    frame = frame[border:h-border, border:h-border, 0]
    frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_CUBIC)
    xx, yy = np.meshgrid(np.arange(112), np.arange(112))
    mask = (((xx+yy) > (112//2+112//10)) & ((xx-yy) < (112//2+112//10))).astype(np.uint8)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    return np.repeat(frame[..., None], 3, axis=-1)

def echonet_video(path):
    meta = video_info(path)
    frames = read_selected(path, range(meta['frames']), echonet_frame)
    return np.stack(frames), meta

def load_echonet():
    model = torchvision.models.video.r2plus1d_18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    cp = checkpoint(ROOT/'weights/echonet.pt')
    sd = {k.removeprefix('module.'): v for k, v in cp['state_dict'].items()}
    model.load_state_dict(sd, strict=True)
    del cp, sd
    return model.eval(), {'architecture': 'r2plus1d_18', 'frames': 32, 'period': 2}

def load_panecho():
    import pandas as pd
    source = ROOT/'repos/PanEcho'
    mod = module_at('panecho_official_models', source/'src/models.py')
    tasks = pd.read_pickle(source/'content/tasks.pkl')
    task_list = [SimpleNamespace(task_name=k, task_type=v['task_type'],
                                class_names=v['class_names'], mean=v['mean']) for k,v in tasks.items()]
    # The final PanEcho checkpoint supplies every learned backbone parameter.
    # Suppress only the redundant generic ImageNet initialization/download.
    original = torchvision.models.convnext_tiny
    def no_imagenet(*a, **kw):
        kw['weights'] = None
        return original(*a, **kw)
    torchvision.models.convnext_tiny = no_imagenet
    try:
        enc = mod.FrameTransformer('convnext_tiny', 8, 4, 0., 'mean', 16)
    finally:
        torchvision.models.convnext_tiny = original
    model = mod.MultiTaskModel(enc, enc.encoder.n_features, task_list, 0.25, True)
    cp = checkpoint(ROOT/'weights/panecho.pt')
    sd = cp['weights']
    # Official hubconf discards the fixed positional encoding for variable clips.
    sd.pop('encoder.time_encoder.pe', None)
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys != ['encoder.time_encoder.pe'] or result.unexpected_keys:
        raise RuntimeError(f'Unexpected checkpoint keys: {result}')
    del cp, sd
    return model.eval(), {'architecture': 'convnext_tiny_frame_transformer', 'tasks': tasks,
                          'checkpoint_missing_keys': result.missing_keys,
                          'preprocessing': 'OpenCV BGR retained as in released loader; INTER_AREA resize 256x256; center crop224; ImageNet normalize; released B,C,T,H,W layout'}

def load_echoclip():
    import open_clip
    config = json.loads((ROOT/'weights/echoclip/open_clip_config.json').read_text())
    local = ROOT/'runtime/echo_clip_cpu.json'
    local.write_text(json.dumps(config['model_cfg']))
    open_clip.add_model_config(local)
    model, _, preprocess = open_clip.create_model_and_transforms(
        'echo_clip_cpu', pretrained=str(ROOT/'weights/echoclip/open_clip_pytorch_model.bin'),
        precision='fp32', device='cpu', image_mean=tuple(config['preprocess_cfg']['mean']),
        image_std=tuple(config['preprocess_cfg']['std']))
    utils = module_at('echoclip_official_utils', ROOT/'repos/echo_CLIP/utils.py')
    return model.eval(), {'architecture': 'ConvNeXt-Base_CLIP', 'preprocess': preprocess,
                          'tokenize': open_clip.get_tokenizer('echo_clip_cpu'), 'utils': utils}

def smoke(model, name, aux):
    start = time.monotonic()
    with torch.inference_mode():
        if name == 'echonet':
            out = model(torch.zeros(1,3,32,112,112))
        elif name == 'panecho':
            out = model(torch.zeros(1,3,16,224,224))
        else:
            out = {'image': model.encode_image(torch.zeros(1,3,224,224)),
                   'text': model.encode_text(aux['tokenize'](['A normal echocardiogram.']))}
        if not check_finite(out):
            raise ValueError('Non-finite synthetic forward output')
    return {'passed': True, 'seconds': time.monotonic()-start,
            'output_shapes': {k: list(v.shape) for k,v in out.items()} if isinstance(out,dict) else list(out.shape),
            'peak_rss_mib': peak_mb()}

def infer_echonet(model, files, aux, args):
    selected = [p for p in files if view_label(p)=='A4C']
    if not selected:
        raise ValueError('No filename-labelled A4C loop; cannot apply EchoNet EF to other views')
    videos = []
    # Follow official external-test notebook's input-cohort channel normalization.
    s1 = np.zeros(3); s2 = np.zeros(3); n = 0
    for p in selected:
        arr, meta = echonet_video(p)
        vals = arr.astype(np.float64)
        s1 += vals.sum(axis=(0,1,2)); s2 += (vals*vals).sum(axis=(0,1,2)); n += np.prod(arr.shape[:3])
        del vals
    mean = s1/n; std = np.sqrt(np.maximum(s2/n-mean**2,1e-12))
    for p in selected:
        started = time.monotonic()
        arr, meta = echonet_video(p)
        x = (arr.astype(np.float32)-mean.astype(np.float32))/std.astype(np.float32)
        if len(x)<64:
            x = np.concatenate([x,np.zeros((64-len(x),112,112,3),np.float32)])
        max_start = len(x)-63
        starts = np.unique(np.linspace(0, max_start-1, min(args.clips,max_start)).astype(int))
        preds = []
        for start in starts:
            clip = torch.from_numpy(x[start+2*np.arange(32)].transpose(3,0,1,2).copy()).unsqueeze(0)
            with torch.inference_mode():
                pred = model(clip).item()
            if not np.isfinite(pred):raise ValueError('Non-finite EF')
            preds.append(pred)
        videos.append({**meta,'ef_percent':float(np.mean(preds)),'clip_predictions':preds,
                       'clip_start_frames':starts.tolist(),'clip_frame_span_seconds':62/meta['fps'] if meta['fps'] else None,'seconds':time.monotonic()-started})
        del arr,x
    return {'predictions':{'ef_percent':float(np.mean([v['ef_percent'] for v in videos]))},
            'videos':videos, 'normalization':{'method':'input-cohort channel mean/std, following official external-test InitializationNotebook; all supplied A4C frames used deterministically',
                                            'mean_0_255':mean.tolist(),'std_0_255':std.tolist()},
            'preprocessing': 'Notebook square/10% crop, sector mask and cubic112; header trim only if first black row is in top15% (sector-only MP4 adaptation); grayscale repeated3;32 frames,stride2; evenly spread clips',
            'limitations':['Input-derived normalization from a single patient is not the original training-cohort calibration.',
                           'Clip-average EF, not segmentation-based beat-to-beat analysis; no clinical ground truth available.']}

def infer_panecho(model, files, aux, args):
    tasks = aux['tasks']; videos=[]; errors=[]
    norm_mean=torch.tensor([.485,.456,.406]).reshape(1,3,1,1)
    norm_std=torch.tensor([.229,.224,.225]).reshape(1,3,1,1)
    for p in files:
        started=time.monotonic()
        try:
            meta=video_info(p)
            starts=np.unique(np.linspace(0,max(0,meta['frames']-16),args.clips).astype(int))
            outs=[]
            for start in starts:
                ids=np.minimum(start+np.arange(16),meta['frames']-1)
                frames=read_selected(p,ids,lambda f:cv2.resize(f,(256,256),interpolation=cv2.INTER_AREA)[16:240,16:240].copy())
                x=torch.from_numpy(np.stack(frames).transpose(0,3,1,2).copy()).float()/255
                x=(x-norm_mean)/norm_std
                x=x.permute(1,0,2,3).unsqueeze(0).contiguous()
                with torch.inference_mode():out=model(x)
                if not check_finite(out):raise ValueError('Non-finite clinical outputs')
                outs.append({k:v[0].tolist() for k,v in out.items()})
            raw={k:np.mean([v[k] for v in outs],axis=0).tolist() for k in outs[0]}
            videos.append({**meta,'raw_predictions':raw,'clip_start_frames':starts.tolist(),'clip_frame_span_seconds':15/meta['fps'] if meta['fps'] else None,'seconds':time.monotonic()-started})
        except Exception as e:
            errors.append({'file':p.name,'error':str(e)})
    if not videos:raise ValueError(f'All video predictions failed: {errors}')
    predictions={}
    for key,task in tasks.items():
        value=np.mean([v['raw_predictions'][key] for v in videos],axis=0)
        if task['task_type']=='regression':
            predictions[key]={'value':float(value[0])*(-1 if key=='GLS' else 1),'type':'regression'}
            if key=='GLS':predictions[key]['note']='Negative of released model output, as official task documentation specifies'
        elif task['task_type']=='multi-class_classification':
            names=[str(x) for x in task['class_names']]
            predictions[key]={'probabilities':dict(zip(names,value.tolist())),'type':'multiclass'}
        else:
            predictions[key]={'probability':float(value[0]),'class_names':[str(x) for x in task['class_names']], 'positive_class':str(task['class_names'][0 if key in {'MVStenosis','AVStructure','RASize','RVSystolicFunction','LVWallThickness-increased-modsev','LVWallThickness-increased-any','pericardial-effusion'} else -1]), 'type':'binary'}
    return {'predictions':predictions,'videos':videos,'errors':errors,'preprocessing':aux['preprocessing'],
            'aggregation':'Mean activated clip predictions within video, then mean video predictions across study',
            'limitations':['Limited 2D-only study; all task outputs are model estimates, not acquired Doppler or calibrated anatomical measurements.',
                           'Wall-motion output is study-level probability, not a regional segment map.',
                           'Exact released forward layout preserved, including its direct reshape; no architecture correction applied.',
                           'Original frame rate retained; at68fps a16-frame clip spans only0.22s. No undocumented temporal resampling applied.']}

def infer_echoclip(model, files, aux, args):
    utils=aux['utils'];tok=aux['tokenize'];preprocess=aux['preprocess']
    prompts=[];values=[]
    for template in utils.zero_shot_prompts['ejection_fraction']:
        for value in range(101):prompts.append(template.replace('<#>',str(value)));values.append(value)
    device_prompts=utils.zero_shot_prompts['pacemaker']
    text=[]
    t=time.monotonic()
    with torch.inference_mode():
        for i,prompt in enumerate(prompts+device_prompts):
            text.append(torch.nn.functional.normalize(model.encode_text(tok([prompt])),dim=-1))
            if i%50==0:print('EchoCLIP text prompts',i,flush=True)
    text=torch.cat(text)
    text_seconds=time.monotonic()-t
    videos=[];all_emb=[];errors=[]
    for p in files:
        started=time.monotonic()
        try:
            meta=video_info(p)
            ids=list(range(0,min(40,meta['frames']),2))
            # Official read_avi crops/scales OpenCV BGR arrays; preserve this color convention.
            frames=read_selected(p,ids,lambda f:utils.crop_and_scale(f,(224,224)))
            emb=[]
            with torch.inference_mode():
                for frame in frames:
                    x=preprocess(Image.fromarray(frame)).unsqueeze(0)
                    emb.append(torch.nn.functional.normalize(model.encode_image(x),dim=-1))
            emb=torch.cat(emb)
            if not check_finite(emb):raise ValueError('Non-finite embeddings')
            ef=utils.compute_regression_metric(emb.unsqueeze(0),text[:len(prompts)],values).item()
            score=utils.compute_binary_metric(emb.unsqueeze(0),text[len(prompts):]).item()
            videos.append({**meta,'ef_percent':ef,'pacemaker_similarity_score':score,
                           'sampled_frames':ids,'seconds':time.monotonic()-started})
            all_emb.append(emb)
        except Exception as e:errors.append({'file':p.name,'error':str(e)})
    if not videos:raise ValueError(f'All EchoCLIP videos failed: {errors}')
    emb=torch.cat(all_emb).unsqueeze(0)
    with torch.inference_mode():
        ef=utils.compute_regression_metric(emb,text[:len(prompts)],values).item()
        score=utils.compute_binary_metric(emb,text[len(prompts):]).item()
    return {'predictions':{'ef_percent':ef,'pacemaker_similarity_score':score},'videos':videos,'errors':errors,
            'text_prompt_encoding_seconds':text_seconds,
            'preprocessing':'Official crop_and_scale224 (square,10% zoom), OpenCV BGR convention; OpenCLIP checkpoint preprocessing; frames0:40:2, image batch1; CPU FP32',
            'aggregation':'Official EF prompt-ranking/mean-rank/median-top20% method across all sampled frames; exploratory multi-view frame pooling',
            'limitations':['Zero-shot EF; not a supervised EF regression head.',
                           'Pacemaker cosine similarity is not a probability; no locally calibrated binary threshold applied.',
                           'No verified RWMA output; sampled first40 frames only, matching official example.']}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['echonet','panecho','echoclip'],required=True)
    ap.add_argument('--patient',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--clips',type=int,default=4)
    ap.add_argument('--threads',type=int,default=3)
    ap.add_argument('--smoke-only',action='store_true')
    args=ap.parse_args()
    if args.clips<1:ap.error('--clips must be positive')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(0);np.random.seed(0);cv2.setNumThreads(1)
    started=time.monotonic()
    report={'model':args.model,'patient':args.patient.name,'device':'cpu','precision':'float32',
            'threads':args.threads,'batch_size':1,'status':'error','errors':[],
            'model_version':{},'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    try:
        manifest=json.loads((ROOT/'weights/manifest.json').read_text())
        report['checkpoint']=manifest[args.model]
        repo={'echonet':'dynamic','panecho':'PanEcho','echoclip':'echo_CLIP'}[args.model]
        report['model_version']={'repo':repo,'commit':subprocess.check_output(['git','-c','safe.directory='+str(ROOT/'repos'/repo),'-C',str(ROOT/'repos'/repo),'rev-parse','HEAD'],text=True).strip()}
        t=time.monotonic()
        model,aux={'echonet':load_echonet,'panecho':load_panecho,'echoclip':load_echoclip}[args.model]()
        report['load_seconds']=time.monotonic()-t
        report['parameter_count']=sum(p.numel() for p in model.parameters())
        report['synthetic_test']=smoke(model,args.model,aux)
        print(args.model,'loaded, synthetic forward passed','RSS MiB',round(peak_mb(),1),flush=True)
        if not args.smoke_only:
            files=sorted(args.patient.resolve().glob('*.mp4'))
            if not files:raise ValueError('No MP4 files in patient folder')
            t=time.monotonic()
            result={'echonet':infer_echonet,'panecho':infer_panecho,'echoclip':infer_echoclip}[args.model](model,files,aux,args)
            report.update(result);report['inference_seconds']=time.monotonic()-t
        report['status']='partial' if report['errors'] else 'ok'
    except Exception as e:
        report['errors'].append({'error':str(e),'traceback':traceback.format_exc()})
        traceback.print_exc()
    finally:
        report['total_seconds']=time.monotonic()-started
        report['peak_rss_mib']=peak_mb()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('SAVED',args.output,'status',report['status'],'seconds',round(report['total_seconds'],2),flush=True)
    return 0 if report['status']!='error' else 1

if __name__=='__main__':sys.exit(main())
