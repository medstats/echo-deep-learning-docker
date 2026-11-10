"""Download official checkpoint bytes with resumable validated HTTP ranges."""
import concurrent.futures, json, pathlib, time, urllib.request, hashlib, os
ROOT=pathlib.Path(__file__).resolve().parent
FILES=[
 ("echonet", "https://github.com/echonet/dynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt", ROOT/"weights/echonet.pt",250572083),
 ("panecho", "https://github.com/CarDS-Yale/PanEcho/releases/download/v1.0/panecho.pt", ROOT/"weights/panecho.pt",505054604),
 ("echoclip", "https://huggingface.co/mkaichristensen/echo-clip/resolve/main/open_clip_pytorch_model.bin", ROOT/"weights/echoclip/open_clip_pytorch_model.bin",606262375)]
CHUNK=2*1024*1024
PARTS=pathlib.Path("/storage/emulated/0/tmp/echo-weight-parts")
PARTS.mkdir(exist_ok=True)
def fetch(job):
 name,url,start,end=job
 part=PARTS/f"{name}-{start:012d}.part"
 size=end-start+1
 if part.exists() and part.stat().st_size==size: return name,size
 for retry in range(6):
  try:
   req=urllib.request.Request(url,headers={"Range":f"bytes={start}-{end}","User-Agent":"echo-inference/1.0"})
   with urllib.request.urlopen(req,timeout=60) as response:
    expected=f"bytes {start}-{end}/"
    if response.status!=206 or not response.headers.get("Content-Range","").startswith(expected):
     raise RuntimeError("Server did not honor requested byte range")
    with part.open("wb") as out:
     left=size
     while left:
      b=response.read(min(65536,left))
      if not b:raise IOError("Truncated HTTP range")
      out.write(b);left-=len(b)
   return name,size
  except Exception as e:
   if retry==5:raise
   time.sleep(min(2**retry,15))
def main():
 jobs=[]
 for name,url,path,size in FILES:
  if path.exists() and path.stat().st_size==size:continue
  for start in range(0,size,CHUNK):jobs.append((name,url,start,min(start+CHUNK-1,size-1)))
 done=0;t=time.monotonic();tot=sum(j[3]-j[2]+1 for j in jobs)
 with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
  fs=[pool.submit(fetch,j) for j in jobs]
  for f in concurrent.futures.as_completed(fs):
   name,size=f.result();done+=size
   print(f"{name}: {done/1e6:.1f}/{tot/1e6:.1f} MB; {(done/1e6)/max(time.monotonic()-t,0.01):.2f} MB/s",flush=True)
 for name,url,path,size in FILES:
  if path.exists() and path.stat().st_size==size:continue
  partial=path.with_suffix(path.suffix+".assembling")
  with partial.open("wb") as out:
   for start in range(0,size,CHUNK):
    part=PARTS/f"{name}-{start:012d}.part"
    if part.stat().st_size!=min(CHUNK,size-start):raise IOError("Wrong part size")
    with part.open("rb") as source:
     while b:=source.read(1024*1024):out.write(b)
  if partial.stat().st_size!=size:raise IOError("Wrong assembled checkpoint size")
  os.replace(partial,path)
  print("ASSEMBLED",name,size,flush=True)
 manifest={}
 for name,url,path,size in FILES:
  sha=hashlib.sha256()
  with path.open("rb") as f:
   while b:=f.read(4*1024*1024):sha.update(b)
  manifest[name]={"url":url,"path":str(path.relative_to(ROOT)),"bytes":size,"sha256":sha.hexdigest()}
 (ROOT/"weights/manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 print("ALL OFFICIAL CHECKPOINTS DOWNLOADED",flush=True)
if __name__=="__main__":main()
