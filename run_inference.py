"""Sequential launcher with process/system memory monitoring (native Python)."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def mem_available_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])/1024
    return 0

def rss_mib(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])/1024
    except FileNotFoundError:pass
    return 0

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--patient',type=Path,help='One patient folder; omit for all folders under patients/')
    ap.add_argument('--models',nargs='+',choices=['echonet','panecho','echoclip'],default=['echonet','panecho','echoclip'])
    ap.add_argument('--clips',type=int,default=4,help='Evenly spread clips per video for EchoNet/PanEcho')
    ap.add_argument('--threads',type=int,default=3)
    ap.add_argument('--smoke-only',action='store_true')
    ap.add_argument('--max-rss-mib',type=float,default=3584)
    ap.add_argument('--min-available-mib',type=float,default=384)
    args=ap.parse_args()
    patients=[args.patient.resolve()] if args.patient else sorted(p for p in (ROOT/'patients').iterdir() if p.is_dir())
    failures=0
    for patient in patients:
        for model in args.models:
            out=ROOT/('smoke_results' if args.smoke_only else 'results')/patient.name/f'{model}.json'
            out.parent.mkdir(parents=True,exist_ok=True)
            log=ROOT/'logs'/f'{patient.name}_{model}{"_smoke" if args.smoke_only else ""}.log'
            print(f'START {patient.name} {model}; available RAM {mem_available_mib():.0f} MiB',flush=True)
            cmd=['sh',str(ROOT/'runtime/python'),str(ROOT/'infer_worker.py'),'--model',model,'--patient',str(patient),'--output',str(out),'--clips',str(args.clips),'--threads',str(args.threads)]
            if args.smoke_only:cmd.append('--smoke-only')
            begin=time.monotonic();maxrss=0;minavail=mem_available_mib();reason=None;pressure=0
            with log.open('w') as log_file:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log_file,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    rss=rss_mib(proc.pid);avail=mem_available_mib()
                    maxrss=max(maxrss,rss);minavail=min(minavail,avail)
                    pressure=pressure+1 if avail<args.min_available_mib else 0
                    if rss>args.max_rss_mib or pressure>=3:
                        reason=f'Stopped for memory pressure: RSS {rss:.0f} MiB, available {avail:.0f} MiB'
                        proc.terminate()
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
                        break
                    time.sleep(.5)
                rc=proc.wait()
            if out.exists() and reason is None:
                try:report=json.loads(out.read_text())
                except Exception:report={}
            else:report={}
            if reason or not report:
                report={'model':model,'patient':patient.name,'status':'error','errors':[{'error':reason or f'Worker exited {rc} without JSON; see {log}'}]}
            report['monitor']={'peak_observed_rss_mib':maxrss,'minimum_system_available_mib':minavail,
                               'wall_seconds_including_imports':time.monotonic()-begin,'worker_exit_code':rc}
            out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            if rc!=0 or report.get('status')=='error':failures+=1
            print(f'END {patient.name} {model}: {report.get("status")}; {report["monitor"]["wall_seconds_including_imports"]:.1f}s; peak RSS {report.get("peak_rss_mib",maxrss):.0f} MiB; {out}',flush=True)
    return 1 if failures else 0
if __name__=='__main__':sys.exit(main())
