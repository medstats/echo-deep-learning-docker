"""Run one or both CPU strain adapters with process/system RAM monitoring."""
import argparse,json,subprocess,time
from pathlib import Path
from run_inference import mem_available_mib,rss_mib
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path);ap.add_argument('--view',choices=['A4C','A2C','A3C'],help='Required verified view for real inference')
    ap.add_argument('--models',nargs='+',choices=['unity_gls','echonet_strain'],default=['unity_gls','echonet_strain'])
    ap.add_argument('--output',type=Path,default=ROOT/'strain_results/manual')
    ap.add_argument('--smoke-only',action='store_true');ap.add_argument('--max-frames',type=int);ap.add_argument('--threads',type=int,default=3)
    ap.add_argument('--max-rss-mib',type=float,default=3584);ap.add_argument('--min-available-mib',type=float,default=384)
    args=ap.parse_args()
    if not args.smoke_only and not args.input:ap.error('--input is required unless --smoke-only')
    if not args.smoke_only and args.view is None:ap.error('--view is required for real inference; verify the apical view')
    if args.max_frames is not None and args.max_frames<7:ap.error('--max-frames must be at least 7')
    if args.threads<1:ap.error('--threads must be positive')
    args.view=args.view or 'A4C'
    failures=0
    for model in args.models:
        out=args.output.resolve()/model;out.mkdir(parents=True,exist_ok=True)
        result=out/'result.json'
        # Prevent a failed worker from leaving an older result looking current.
        if result.exists():result.replace(out/'previous_result.json')
        cmd=['sh',str(ROOT/'runtime/strain-python'),str(ROOT/'strain/worker.py'),'--model',model,'--view',args.view,'--output',str(out),'--threads',str(args.threads)]
        if args.input:cmd+=['--input',str(args.input.resolve())]
        if args.smoke_only:cmd+=['--smoke-only']
        if args.max_frames:cmd+=['--max-frames',str(args.max_frames)]
        start=time.monotonic();peak=0;pressure=0;reason=None
        print('START',model,'available MiB',round(mem_available_mib()),flush=True)
        with (out/'execution.log').open('w') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while p.poll() is None:
                rss=rss_mib(p.pid);avail=mem_available_mib();peak=max(peak,rss)
                pressure=pressure+1 if avail<args.min_available_mib else 0
                if rss>args.max_rss_mib or pressure>=3:
                    reason=f'Memory guard: RSS {rss:.0f} MiB, system available {avail:.0f} MiB';p.terminate()
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    break
                time.sleep(.5)
            rc=p.wait()
        try:r=json.loads(result.read_text())
        except (OSError,ValueError):r={'model':model,'status':'error','errors':[{'message':reason or f'Worker exited {rc} without JSON'}]}
        r['monitor']={'worker_exit_code':rc,'peak_observed_rss_mib':peak,'wall_seconds':time.monotonic()-start,'memory_stop':reason}
        if rc or reason:r['status']='error'
        result.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
        failures+=r['status']!='ok'
        print('END',model,r['status'],'seconds',round(r['monitor']['wall_seconds'],1),'peak MiB',round(r.get('peak_rss_mib',peak)),str(result),flush=True)
    return bool(failures)
if __name__=='__main__':raise SystemExit(main())
