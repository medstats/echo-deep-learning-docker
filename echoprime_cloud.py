"""Reusable GPU EchoPrime inference. Downloads occur on the cloud host only."""
from pathlib import Path
import concurrent.futures, gc, hashlib, importlib.util, json, math, os, platform
import resource, shutil, subprocess, sys, time, traceback, urllib.request, zipfile

REPO_COMMIT = '03874a5f203695f38968068c21584656d475b6b1'
RELEASE = 'https://github.com/echonet/EchoPrime/releases/download/v1.0.0/'
ASSETS = {'model_data.zip': 1338009590, 'candidate_embeddings_p1.pt':1260213315,
          'candidate_embeddings_p2.pt':1260213315}
DEFAULTS = {'input_root':'/kaggle/input','output_dir':'/kaggle/working/echoprime_results',
            'cache_dir':'/kaggle/temp/echoprime','spatial_mode':'adaptive',
            'generate_reports':True,'batch_size':1,'retrieval_k':50,'require_gpu':True}

def log(*parts): print(time.strftime('%H:%M:%S'),*parts,flush=True)
def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def clean_json(value):
    import numpy as np
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

def setup(config):
    cache=Path(config['cache_dir']);cache.mkdir(parents=True,exist_ok=True)
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    # Keep the host's matching CUDA torch/torchvision installations.
    needed=[]
    for module, package in [('pydicom','pydicom==2.4.4'),('transformers','transformers==4.57.0')]:
        try:
            from importlib.metadata import version, PackageNotFoundError
            installed=version(module)
            if module=='transformers' and installed!='4.57.0':needed.append(package)
        except (ImportError, PackageNotFoundError):needed.append(package)
    if needed:subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',*needed],check=True)
    repo=cache/'EchoPrime'
    if not (repo/'.git').exists():
        subprocess.run(['git','clone','https://github.com/echonet/EchoPrime.git',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'checkout',REPO_COMMIT],check=True)
    def download(item):
        name,size=item;path=cache/name
        if path.exists() and path.stat().st_size==size:return path
        partial=path.with_suffix(path.suffix+'.part')
        for attempt in range(4):
            try:
                req=urllib.request.Request(RELEASE+name,headers={'User-Agent':'EchoPrime-Cloud-Inference'})
                with urllib.request.urlopen(req,timeout=180) as response,partial.open('wb') as f:
                    shutil.copyfileobj(response,f,length=8*1024*1024)
                if partial.stat().st_size!=size:raise ValueError(f'{name}: wrong byte count')
                partial.replace(path);log('Downloaded on cloud:',name);return path
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    log('Downloading official assets directly on this host (~3.86 GB total).')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(download,ASSETS.items()))
    model_dir=repo/'model_data'
    if not (model_dir/'weights/echo_prime_encoder.pt').exists():
        log('Extracting official model_data.zip')
        with zipfile.ZipFile(cache/'model_data.zip') as z:
            for member in z.infolist():
                if not (repo/member.filename).resolve().is_relative_to(repo.resolve()):raise ValueError('Unsafe archive path')
            z.extractall(repo)
    candidate=model_dir/'candidates_data';candidate.mkdir(parents=True,exist_ok=True)
    for name in ASSETS:
        if name.endswith('.pt'):
            dest=candidate/name
            if not dest.exists():
                try:os.link(cache/name,dest)
                except OSError:shutil.copy2(cache/name,dest)
    required=['weights/echo_prime_encoder.pt','weights/view_classifier.pt','candidates_data/candidate_studies.csv',
              'candidates_data/candidate_reports.pkl','candidates_data/candidate_labels.pkl']
    for name in required:
        if not (model_dir/name).is_file():raise FileNotFoundError(model_dir/name)
    weights={name:{'url':RELEASE+name,'bytes':(cache/name).stat().st_size,'sha256':sha256(cache/name)} for name in ASSETS}
    write_json(out/'download_manifest.json',{'repository_commit':REPO_COMMIT,'assets':weights,
        'download_location':str(cache),'note':'All model assets downloaded directly on cloud host. No weights uploaded from phone.'})
    return repo

def discover_input(config):
    root=Path(config['input_root']);target=Path(config['cache_dir'])/'inputs'
    # A private Kaggle dataset can contain patients.zip. A Vast host can use plain folders.
    zips=sorted(root.rglob('patients.zip')) if root.exists() else []
    if zips:
        target.mkdir(parents=True,exist_ok=True)
        for zp in zips:
            with zipfile.ZipFile(zp) as z:
                for m in z.infolist():
                    if not (target/m.filename).resolve().is_relative_to(target.resolve()):raise ValueError('Unsafe input archive path')
                z.extractall(target)
        root=target
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower()=='.mp4')
    if not files:raise ValueError(f'No MP4 loops found under {root}. Attach private patients.zip or patient folders.')
    groups={}
    for p in files:groups.setdefault(p.parent.name,[]).append(p)
    return groups

class LazyDecodedReports:
    """Identical official phrase decoding, only when a retrieved report is accessed."""
    def __init__(self,encoded,decode):self.encoded=encoded;self.decode=decode;self.cache={}
    def __getitem__(self,index):
        index=int(index)
        if index not in self.cache:
            raw=self.encoded.iloc[index] if hasattr(self.encoded,'iloc') else self.encoded[index]
            if len(self.cache)>=512:self.cache.clear()
            self.cache[index]=self.decode(raw)
        return self.cache[index]
    def __len__(self):return len(self.encoded)

def load_model(repo,config):
    import torch
    os.chdir(repo);sys.path.insert(0,str(repo))
    import utils
    source=(repo/'echo_prime/model.py').read_text()
    # Modern torch defaults weights_only=True. These are trusted official release assets.
    # Keep candidate tensors on CPU and mmap them to avoid unnecessary copies.
    def cloud_load(path,*args,**kwargs):
        kwargs.setdefault('map_location','cpu');kwargs.setdefault('weights_only',False)
        if 'candidate_embeddings' in str(path):kwargs.setdefault('mmap',True)
        try:return torch.load(path,*args,**kwargs)
        except RuntimeError as e:
            if 'mmap' not in str(e):raise
            kwargs.pop('mmap',None);return torch.load(path,*args,**kwargs)
    old='[utils.phrase_decode(vec_phr) for vec_phr in tqdm(candidate_reports)]'
    if source.count(old)!=1:raise RuntimeError('Pinned upstream report loading expression changed')
    patched=source.replace('torch.load(','_cloud_load(').replace(old,'_lazy_reports(candidate_reports)')
    namespace={'__name__':'echoprime_cloud_model','_cloud_load':cloud_load,
               '_lazy_reports':lambda x:LazyDecodedReports(x,utils.phrase_decode)}
    exec(compile(patched,str(repo/'echo_prime/model.py'),'exec'),namespace)
    log('Loading full EchoPrime encoders, view classifier, retrieval bank and labels.')
    ep=namespace['EchoPrime']()
    # Official embed_videos uses bins of50. Lower only batching; preserve FP32 computations.
    def embed(videos):
        chunks=[]
        for i in range(0,len(videos),config['batch_size']):
            chunks.append(ep.echo_encoder(videos[i:i+config['batch_size']].to(ep.device)))
        return torch.cat(chunks)
    ep.embed_videos=embed
    ep.echo_encoder.eval();ep.view_classifier.eval()
    log('Full model loaded; retrieval tensor',tuple(ep.candidate_embeddings.shape))
    return ep,utils

def prepare_video(path,ep,utils,config,preview):
    import cv2,numpy as np,torch
    from PIL import Image
    cap=cv2.VideoCapture(str(path));meta={'file':path.name,'sha256':sha256(path),
        'width':int(cap.get(3)),'height':int(cap.get(4)),'fps':cap.get(5),'total_frames':int(cap.get(7))}
    frames=[];first=None;padding=None
    try:
        # Match upstream process_mp4s: first32frames, take every2nd ->16model frames.
        for i in range(ep.frames_to_take):
            ok,frame=cap.read()
            if not ok:break
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            mode=config['spatial_mode']
            if mode=='adaptive':mode='preserve_full_frame' if frame.shape[0]>frame.shape[1] else 'official'
            if mode=='preserve_full_frame':
                h,w=frame.shape[:2];side=int(math.ceil(max(h,w)/0.8))+2
                top=(side-h)//2;bottom=side-h-top;left=(side-w)//2;right=side-w-left
                frame=cv2.copyMakeBorder(frame,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
                padding={'top':top,'bottom':bottom,'left':left,'right':right}
            processed=utils.crop_and_scale(frame)
            if first is None:first=processed.copy()
            frames.append(processed)
    finally:cap.release()
    if not frames:raise ValueError(f'Cannot decode {path}')
    x=torch.as_tensor(np.stack(frames),dtype=torch.float32).permute(3,0,1,2)
    x.sub_(ep.mean).div_(ep.std)
    if x.shape[1]<ep.frames_to_take:
        x=torch.cat((x,torch.zeros(3,ep.frames_to_take-x.shape[1],ep.video_size,ep.video_size)),dim=1)
    x=x[:,:ep.frames_to_take:ep.frame_stride]
    assert x.shape==(3,16,224,224) and torch.isfinite(x).all()
    Image.fromarray(first).save(preview)
    meta.update({'sampled_source_frames':list(range(0,len(frames),ep.frame_stride)),
        'padding':padding,'spatial_mode':config['spatial_mode'],'applied_spatial_mode':mode,'preprocessing_preview':preview.name})
    return x,meta

def run_all(config=None):
    config={**DEFAULTS,**(config or {})}
    out=Path(config['output_dir']).resolve();out.mkdir(parents=True,exist_ok=True)
    write_json(out/'run_config.json',config)
    started=time.monotonic();summary=[]
    try:
        import torch,numpy as np,cv2
        if config['require_gpu'] and not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; enable GPU accelerator.')
        torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(0);np.random.seed(0)
        # Disable TF32 so inference uses standard FP32 arithmetic.
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        environment={'python':platform.python_version(),'torch':torch.__version__,'gpu':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'meminfo':Path('/proc/meminfo').read_text().splitlines()[:3],'platform':platform.platform()}
        write_json(out/'environment.json',environment);log(environment)
        repo=setup(config);groups=discover_input(config)
        ep,utils=load_model(repo,config)
        with torch.inference_mode():
            fake=torch.zeros(1,3,16,224,224,device=ep.device)
            feature=ep.echo_encoder(fake);view=ep.view_classifier(fake[:,:,0]);assert torch.isfinite(feature).all() and torch.isfinite(view).all()
        del fake,feature,view;torch.cuda.empty_cache()
        for patient,paths in groups.items():
            case_start=time.monotonic();case_out=out/patient;case_out.mkdir(exist_ok=True)
            report={'patient':patient,'status':'error','model':'EchoPrime','repository_commit':REPO_COMMIT,
                'synthetic_test_passed':True,'precision':'float32','batch_size':config['batch_size'],'errors':[],
                'limitations':['Research inference; no clinical ground truth supplied.','Generated report is retrieved by similarity, not verified clinical findings.',
                'Limited views do not support all generated report sections. Missing modalities are not measured.',
                'Annotations, low resolution, screen-recorded playback and padding adaptations can reduce accuracy.'],
                'implementation_notes':['Official model weights, view classifier, MIL section weights, predict_metrics and generate_report.',
                'CPU/mmap candidate loading, lazy official phrase decoding and video batch1 reduce memory; no weight changes.',
                'OpenCV RGB decode; upstream crop_and_scale and mean/std; first32source frames,stride2.'],
                'input_videos':[]}
            try:
                torch.cuda.reset_peak_memory_stats();videos=[]
                for i,path in enumerate(paths):
                    x,meta=prepare_video(path,ep,utils,config,case_out/f'input_{i:02d}.png');videos.append(x);report['input_videos'].append(meta)
                videos=torch.stack(videos)
                with torch.inference_mode():
                    encoded=ep.encode_study(videos)
                    views=ep.get_views(videos,return_view_list=True)
                    torch.cuda.synchronize();report['encoding_seconds']=time.monotonic()-case_start
                    for meta,view in zip(report['input_videos'],views):meta['predicted_view']=view
                    encoded=encoded.cpu();torch.save(encoded,case_out/'study_embedding.pt')
                    t=time.monotonic();pred=ep.predict_metrics(encoded,k=config['retrieval_k'])
                    report['predictions']=clean_json(pred);report['retrieval_seconds']=time.monotonic()-t
                    if config['generate_reports']:
                        t=time.monotonic();text=ep.generate_report(encoded)
                        (case_out/'retrieved_report.txt').write_text('UNVERIFIED ECHOPRIME RETRIEVAL OUTPUT — REQUIRES CLINICAL REVIEW\n\n'+text.replace('[SEP]','\n\n'))
                        report['report_generation_seconds']=time.monotonic()-t
                report['status']='ok';report['peak_gpu_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
                log(patient,'OK EF',report['predictions'].get('ejection_fraction'),'views',views)
                del videos,encoded;gc.collect();torch.cuda.empty_cache()
            except Exception as exc:
                report['errors'].append({'error':str(exc),'traceback':traceback.format_exc()});log(patient,'ERROR',str(exc))
            report['total_seconds']=time.monotonic()-case_start;report['peak_process_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            write_json(case_out/'echoprime.json',report)
            summary.append({'patient':patient,'status':report['status'],'predictions':report.get('predictions',{}),'total_seconds':report['total_seconds'],'errors':report['errors']})
        write_json(out/'summary.json',{'cases':summary,'total_seconds':time.monotonic()-started,'all_cases_passed':all(x['status']=='ok' for x in summary)})
        if not summary or not all(x['status']=='ok' for x in summary):raise RuntimeError('One or more cases failed; inspect JSONs.')
        log('ALL CASES PASSED',len(summary),'total seconds',time.monotonic()-started)
        return summary
    except Exception as exc:
        write_json(out/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'total_seconds':time.monotonic()-started})
        raise

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--input-root',default=DEFAULTS['input_root']);ap.add_argument('--output-dir',default=DEFAULTS['output_dir']);ap.add_argument('--cache-dir',default=DEFAULTS['cache_dir']);ap.add_argument('--no-reports',action='store_true');ap.add_argument('--spatial-mode',choices=['official','preserve_full_frame','adaptive'],default='adaptive');args=ap.parse_args()
    run_all({'input_root':args.input_root,'output_dir':args.output_dir,'cache_dir':args.cache_dir,'generate_reports':not args.no_reports,'spatial_mode':args.spatial_mode})
