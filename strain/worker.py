"""Local CPU adapters for released Unity-GLS and EchoNet-Strain; no training."""
from pathlib import Path
import argparse, ast, csv, gc, hashlib, json, logging, os, resource, subprocess, sys, time, traceback, types
import cv2
import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {'unity_gls': ROOT/'weights/strain/unity-211-weights-401.pt',
           'echonet_strain': ROOT/'weights/strain/deeplabv3_resnet50_random.pt'}
REPOS = {'unity_gls':ROOT/'repos/unity-gls', 'echonet_strain':ROOT/'repos/strain'}
LAYERS = ['curve-lv-endo','lv-apex-endo','mv-ant-wall-hinge','mv-inf-wall-hinge']

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x,np.generic): x=x.item()
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def save_json(p,x):
    Path(p).write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def trusted_load(p):
    try: return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
    except RuntimeError as e:
        if 'mmap' not in str(e): raise
        return torch.load(p,map_location='cpu',weights_only=False)


def read_input(path, max_frames=None):
    """Retain native frame order and spacing; no temporal resampling."""
    path=Path(path)
    if path.suffix.lower() in {'.dcm','.dicom'}:
        import pydicom
        d=pydicom.dcmread(path)
        a=d.pixel_array
        n=int(getattr(d,'NumberOfFrames',1))
        if n<2: raise ValueError('A multiframe cine is required')
        if a.ndim==3: a=np.repeat(a[...,None],3,axis=-1)
        if a.ndim!=4 or a.shape[-1]!=3: raise ValueError('Unsupported DICOM pixel layout')
        if a.dtype!=np.uint8: raise ValueError('Adapter requires uint8 ultrasound pixels; no silent intensity conversion')
        if getattr(d,'PhotometricInterpretation','')=='MONOCHROME1':a=255-a
        frames=a[...,::-1].copy()  # standard RGB pixel_array -> OpenCV BGR
        ft=float(getattr(d,'FrameTime',0));fps=1000/ft if ft>0 else float(getattr(d,'CineRate',0))
        meta={'kind':'DICOM','frames_available':len(frames),'fps':fps or None,
              'height':frames.shape[1],'width':frames.shape[2],
              'calibration_note':'Lengths exported in pixels; no assumed centimetre calibration. Native pixel aspect ratio must be reviewed.'}
    else:
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():raise ValueError('Cannot open cine')
        meta={'kind':'video','frames_available':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
              'fps':cap.get(cv2.CAP_PROP_FPS),'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              'calibration_note':'MP4/AVI lacks ultrasound physical calibration; lengths are pixels, ratio assumes stable isotropic image geometry. Native acquisition timing may differ from playback.'}
        frames=[]
        while True:
            ok,f=cap.read()
            if not ok:break
            frames.append(f)
            if max_frames and len(frames)>=max_frames:break
        cap.release()
        if not frames:raise ValueError('No decoded frames')
        frames=np.stack(frames)
    if max_frames:frames=frames[:max_frames]
    if len(frames)<7:raise ValueError('At least seven frames are needed; a complete multi-cycle cine is preferable')
    meta['frames_processed']=len(frames)
    return frames,meta

def write_avi(path, frames, fps):
    h,w=frames.shape[1:3]
    v=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),float(fps or 30),(w,h))
    if not v.isOpened():raise RuntimeError('AVI MJPG encoder unavailable')
    for f in frames:v.write(np.ascontiguousarray(f,dtype=np.uint8))
    v.release()

def save_montage(path,frames):
    indices=np.linspace(0,len(frames)-1,min(6,len(frames))).astype(int)
    tiles=[]
    for i in indices:
        f=frames[i]
        if f.shape[-1]==1:f=np.repeat(f,3,axis=-1)
        tile=cv2.resize(f,(304,304),interpolation=cv2.INTER_AREA)
        cv2.putText(tile,f'Frame {i}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
        tiles.append(tile)
    cv2.imwrite(str(path),np.concatenate(tiles,axis=1))

def load_unity():
    sys.path.insert(0,str(REPOS['unity_gls']))
    from scanpilot.models import BMode
    # Preserve upstream source on disk. Route only device literals to CPU.
    text=Path(BMode.__file__).read_text().replace('USE_CUDA = True','USE_CUDA = False')
    text=text.replace('"cuda"','"cpu"').replace("'cuda'","'cpu'")
    text=text.replace('for i in range(num_frames):', 'for i in range(num_frames):\n            if i % 10 == 0: print("Unity heatmap frame", i, "of", num_frames, flush=True)')
    mod=types.ModuleType('scanpilot.models.strain_cpu_adapter');mod.__package__='scanpilot.models'
    exec(compile(text,str(BMode.__file__)+'[CPU adapter]','exec'),mod.__dict__)
    # Only the official local checkpoint, no request/cache download in inference.
    import ScantensusPT.utils as u
    u.load_checkpoint=lambda checkpoint_path,device='cpu':torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    mod.load_checkpoint=u.load_checkpoint
    loaded=mod.bmode_seg_init(str(WEIGHTS['unity_gls']),cache_name=str(ROOT/'runtime/strain/unity-cache'))
    count=sum(x['params'][k].size for x in loaded.tensors for k in x['params'])
    begin=time.monotonic()
    test=mod.bmode_seg_process(loaded,np.zeros((1,608,608,1),np.uint8),output_layer_list=LAYERS)
    if not np.isfinite(test[0]).all():raise ValueError('Nonfinite synthetic heatmaps')
    return (mod,loaded),{'passed':True,'seconds':time.monotonic()-begin,'shape':list(test[0].shape),'parameters':count,
                         'architecture':'HRNetV2M10','context_frames':loaded.config['pre_post_list']}

def run_unity(loaded,frames,meta,out):
    import scipy.signal
    from Scantensus.labels.bmode import calc_bmode_labels
    from Scantensus.measure.lv import fix_lv_endo_points,find_peaks_cm
    from Scantensus.utils.labels import labels_calc_len
    mod,model=loaded
    rgb=frames[...,::-1].copy()
    heat,hs,ws,images=mod.bmode_seg_process(model,rgb,output_layer_list=LAYERS,debug_mode=True)
    keys=[f'frame-{i:04}' for i in range(len(frames))]
    labels=calc_bmode_labels((heat,hs,ws),LAYERS,keys,calc_lv_gls=True)
    labels=labels_calc_len(fix_lv_endo_points(labels))
    lengths=[];valid=[]
    overlay=images.copy()
    if overlay.shape[-1]==1:overlay=np.repeat(overlay,3,axis=-1)
    for i,k in enumerate(keys):
        instance=labels[k]['labels'].get('curve-lv-endo',[{}])[0]
        value=instance.get('curve_len',np.nan)
        lengths.append(value)
        ok=instance.get('type')=='curve' and np.isfinite(value) and value>0
        valid.append(bool(ok))
        if ok:
            pts=np.stack([instance['x'],instance['y']],axis=-1)+np.array([ws,hs])
            if np.isfinite(pts).all():cv2.polylines(overlay[i],[np.rint(pts).astype(np.int32)],False,(0,255,0),1)
    lengths=np.asarray(lengths,float)
    smooth=scipy.signal.savgol_filter(lengths,7,3)
    # Upstream function's selection uses quantiles/distance only: constant unit scaling cancels in percent shortening.
    peaks=find_peaks_cm(smooth,min_name='lv_len_es',max_name='lv_len_ed')
    for seq in peaks.values():
        for x in seq:x['unit']='pixel'
    ed=[x['value'] for x in peaks['lv_len_ed']];es=[x['value'] for x in peaks['lv_len_es']]
    value=100*(np.mean(es)-np.mean(ed))/np.mean(ed) if ed and es and np.mean(ed)>0 else None
    if value is not None and not np.isfinite(value):value=None
    with (out/'lengths.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['frame','length_pixels','smoothed_length_pixels','valid_contour']);w.writerows(zip(range(len(frames)),lengths,smooth,valid))
    save_json(out/'contours.json',labels)
    save_json(out/'peaks.json',peaks)
    np.savez_compressed(out/'landmark_heatmaps_first_frame.npz',heatmaps=heat[0],layers=LAYERS)
    cv2.imwrite(str(out/'input_preview.png'),images[0])
    cv2.imwrite(str(out/'contour_preview.png'),overlay[0])
    write_avi(out/'contours.avi',overlay,meta['fps'])
    save_montage(out/'contour_montage.png',overlay)
    plot_lengths(out,lengths,smooth)
    return {'status':'ok' if value is not None else 'no_valid_strain',
            'signed_longitudinal_contour_shortening_percent':value,
            'ed_mean_length_pixels':np.mean(ed) if ed else None,'es_mean_length_pixels':np.mean(es) if es else None,
            'valid_contours':sum(valid),'total_frames':len(valid),'ed_peaks':len(ed),'es_peaks':len(es),
            'preprocessing':'Upstream native-size center_crop_or_pad to608; grayscale first RGB channel; seven-frame temporal context; exact upstream heatmap/path/length and Savitzky-Golay7,3/peak method. CPU device adaptation only.',
            'limitations':['Single-view contour-length shortening, not speckle tracking or measured three-view GLS.',
                           'No physical calibration fabricated; percent calculation uses pixel lengths. Review pixel geometry, full border and annular endpoints.',
                           'Upstream ED/ES peak selection is heuristic, not ECG-gated beat pairing. Missing/low-confidence contours can invalidate the result.']}

def upstream_strain_functions():
    import scipy,scipy.signal,scipy.fftpack,scipy.stats
    import pandas as pd
    import matplotlib.pyplot as plt
    import math,typing
    source=REPOS['echonet_strain']/'Segmentation Analysis/Vid_to_Strain.py'
    text=source.read_text().replace('np.int0(box)','box.astype(np.intp)')
    tree=ast.parse(text)
    # Use actual upstream numerical functions, excluding broken installer/training/large-batch class imports.
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    ns=dict(np=np,cv2=cv2,scipy=scipy,pd=pd,plt=plt,math=math,typing=typing,os=os,
            norm=np.linalg.norm,signal=scipy.signal,fftpack=scipy.fftpack)
    exec(compile(tree,str(source)+'[function adapter]','exec'),ns)
    return ns

def load_echonet():
    cp=trusted_load(WEIGHTS['echonet_strain'])
    sd={k.removeprefix('module.'):v for k,v in cp['state_dict'].items()}
    model=torchvision.models.segmentation.deeplabv3_resnet50(weights=None,weights_backbone=None,aux_loss=False)
    model.classifier[-1]=torch.nn.Conv2d(model.classifier[-1].in_channels,1,kernel_size=model.classifier[-1].kernel_size)
    model.load_state_dict(sd,strict=True);model.eval();del cp,sd;gc.collect()
    begin=time.monotonic()
    with torch.inference_mode():y=model(torch.zeros(1,3,112,112))['out']
    if not torch.isfinite(y).all():raise ValueError('Nonfinite synthetic mask')
    return model,{'passed':True,'seconds':time.monotonic()-begin,'shape':list(y.shape),
                  'parameters':sum(p.numel() for p in model.parameters()),'architecture':'DeepLabV3 ResNet50 one-channel'}

def square112(frame):
    h,w=frame.shape[:2];side=max(h,w)
    a=np.zeros((side,side,3),np.uint8);y=(side-h)//2;x=(side-w)//2;a[y:y+h,x:x+w]=frame
    return cv2.resize(a,(112,112),interpolation=cv2.INTER_AREA)

def run_echonet(model,frames,meta,out):
    ns=upstream_strain_functions()
    # Already112x112 sources pass unchanged. For other sizes, explicit aspect-preserving full-frame adapter.
    images=np.stack([square112(f) for f in frames])
    rgb=images[...,::-1].copy()
    mean=np.array([31.834011,31.95879,32.082172],np.float32)
    std=np.array([48.866325,49.137333,49.361984],np.float32)
    logits=[]
    with torch.inference_mode():
        for i,frame in enumerate(rgb):
            x=torch.from_numpy(((frame.astype(np.float32)-mean)/std).transpose(2,0,1).copy()).unsqueeze(0)
            y=model(x)['out'][0,0]
            if not torch.isfinite(y).all():raise ValueError(f'Nonfinite frame {i}')
            logits.append(y.numpy().astype(np.float16))
            if i%25==0:print('EchoNet segmentation frame',i,flush=True)
    logits=np.stack(logits)
    smoothed=ns['smooth_segmentation'](logits,0.9)
    masks=(smoothed>0).astype(np.uint8)
    np.savez_compressed(out/'segmentation.npz',logits=logits,masks=masks)
    overlay=images.copy();overlay[:,:,:,1]=np.maximum(overlay[:,:,:,1],masks*255)
    write_avi(out/'segmentation.avi',overlay,meta['fps'])
    save_montage(out/'segmentation_montage.png',overlay)
    cv2.imwrite(str(out/'input_preview.png'),images[0]);cv2.imwrite(str(out/'segmentation_preview.png'),overlay[0])
    # Work directly on the decoded processed frames, avoiding lossy intermediate AVI re-read.
    video=rgb.transpose(3,0,1,2)
    left,right=ns['get_points'](video,masks)
    dilated=ns['get_dilation'](masks,dilations=1)
    ns['savevideo']=lambda filename,array,fps=30:write_avi(filename,array.transpose(1,2,3,0)[...,::-1],meta['fps'])
    table,ratios=ns['strain_lengths'](video,dilated,left,right,'contours.avi',str(out),str(out),str(out),window_size=3,downsample=2,point_radius=1)
    values=np.asarray(ratios,float);values=values[np.isfinite(values)]
    # Upstream notebook reports arithmetic mean of returned shortening ratios.
    ratio=float(np.mean(values)) if len(values) else None
    signed=100*(ratio-1) if ratio is not None else None
    return {'status':'ok' if ratio is not None else 'no_valid_strain',
            'mean_es_ed_length_ratio':ratio,'beat_length_ratios':values.tolist(),
            'signed_longitudinal_contour_shortening_percent':signed,
            'upstream_positive_shortening_percent':100*(1-ratio) if ratio is not None else None,
            'total_frames':len(frames),'valid_masks':int((masks.sum((1,2))>0).sum()),
            'preprocessing':'Full-frame square black padding + isotropic112 resize for non112 input (explicit adaptation); RGB, released fixed normalization, sequential FP32 frame batch1, upstream float16 logits/IIR alpha0.9, contour/dilation1/downsample2/convolve-window3/peak algorithm. No flip, no temporal resampling.',
            'length_csv':'contours.csv','limitations':['A4C single-view segmentation contour shortening, not speckle tracking or segmental strain.',
                'Upstream positive 100*(1-ratio) is also exported; negative signed convention is 100*(ratio-1).',
                'Peak rules use fixed distance32/width5/prominence12 in frame/pixel units; playback frame rate and image quality affect transfer.',
                'Full-frame resize is not verified training sector preprocessing; frames are not matched to ECG-gated ED/ES.',
                'Lossy intermediate segmentation-video round trip was omitted; numerical contour algorithms are upstream.']}

def plot_lengths(out,raw,smooth):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,3));plt.plot(raw,label='raw');plt.plot(smooth,label='Savgol7,3');plt.xlabel('Frame');plt.ylabel('Endocardial length (pixels)');plt.legend();plt.tight_layout();plt.savefig(out/'length_curve.png');plt.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=list(WEIGHTS),required=True)
    ap.add_argument('--input',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--view',choices=['A4C','A2C','A3C'],default='A4C',help='User-verified view; not automatic classification')
    ap.add_argument('--smoke-only',action='store_true');ap.add_argument('--max-frames',type=int);ap.add_argument('--threads',type=int,default=3)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    report={'model':args.model,'status':'error','errors':[],'device':'cpu','dtype':'FP32','view':args.view,'view_source':'explicit caller label, verify visually'}
    begin=time.monotonic()
    try:
        if args.model=='echonet_strain' and args.view!='A4C':raise ValueError('EchoNet-Strain route requires A4C')
        if not args.smoke_only and args.input is None:raise ValueError('--input required outside smoke-only')
        torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
        report['checkpoint']={'path':str(WEIGHTS[args.model]),'bytes':WEIGHTS[args.model].stat().st_size,'sha256':digest(WEIGHTS[args.model])}
        report['source_commit']=subprocess.check_output(['git','-c',f'safe.directory={REPOS[args.model]}','-C',str(REPOS[args.model]),'rev-parse','HEAD'],text=True).strip()
        loaded,smoke=(load_unity if args.model=='unity_gls' else load_echonet)();report['synthetic_forward']=smoke
        print('Synthetic forward passed',args.model,smoke,flush=True)
        if args.smoke_only:report['status']='ok'
        else:
            p=args.input.resolve();frames,meta=read_input(p,args.max_frames)
            report['input']={'path':str(p),'sha256':digest(p),**meta}
            started=time.monotonic();report.update((run_unity if args.model=='unity_gls' else run_echonet)(loaded,frames,meta,out))
            report['inference_seconds']=time.monotonic()-started
    except Exception as e:
        report['errors'].append({'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});traceback.print_exc()
    report['total_seconds']=time.monotonic()-begin;report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    save_json(out/'result.json',report)
    print('FINAL',json.dumps(clean({k:v for k,v in report.items() if k not in {'errors','input','checkpoint','limitations','synthetic_forward'}})),flush=True)
    return 1 if report['status']=='error' else 0
if __name__=='__main__':sys.exit(main())
