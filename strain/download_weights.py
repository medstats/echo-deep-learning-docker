"""Fetch only the official strain inference checkpoints, or verify installed files."""
import argparse,hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    manifest=json.loads((ROOT/'weights/strain/manifest.json').read_text())
    for name,item in manifest.items():
        p=ROOT/item['path'];p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            if args.verify_only:raise FileNotFoundError(p)
            partial=p.with_suffix(p.suffix+'.part')
            for attempt in range(3):
                try:
                    request=urllib.request.Request(item['url'],headers={'User-Agent':'Echo-Strain-Inference'})
                    # Unity HEAD returned403 while an ordinary GET worked: do not depend on HEAD.
                    with urllib.request.urlopen(request,timeout=120) as r,partial.open('wb') as f:
                        while chunk:=r.read(1024*1024):f.write(chunk)
                    if partial.stat().st_size!=item['bytes'] or sha(partial)!=item['sha256']:
                        raise ValueError(f'{name}: byte count/hash mismatch')
                    partial.replace(p);break
                except Exception:
                    if attempt==2:raise
                    time.sleep(2**attempt)
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError(f'{name}: invalid installed checkpoint')
        print(name,'verified',p.stat().st_size,'bytes')
if __name__=='__main__':main()
