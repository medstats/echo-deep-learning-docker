"""Assemble complete downloads early so models can be tested sequentially."""
import hashlib,json,os
from pathlib import Path
from download_weights import FILES,PARTS,CHUNK,ROOT
manifest_path=ROOT/'weights/manifest.json'
manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
for name,url,path,size in FILES:
    if not path.exists() or path.stat().st_size!=size:
        parts=[PARTS/f'{name}-{s:012d}.part' for s in range(0,size,CHUNK)]
        if not all(p.exists() and p.stat().st_size==min(CHUNK,size-i*CHUNK) for i,p in enumerate(parts)):
            print(name,'not complete');continue
        assembled=path.with_suffix('.ready')
        with assembled.open('wb') as out:
            for p in parts:
                with p.open('rb') as source:
                    while b:=source.read(1024*1024):out.write(b)
        os.replace(assembled,path)
    sha=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(4*1024*1024):sha.update(b)
    manifest[name]={'url':url,'path':str(path.relative_to(ROOT)),'bytes':size,'sha256':sha.hexdigest()}
    print(name,'READY',size,sha.hexdigest())
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
