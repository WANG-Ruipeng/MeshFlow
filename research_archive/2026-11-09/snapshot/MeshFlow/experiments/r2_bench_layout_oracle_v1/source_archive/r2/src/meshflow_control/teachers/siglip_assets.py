"""Pinned public SigLIP asset download; no model import or remote code."""
from __future__ import annotations
import hashlib,json,os,time,urllib.request
from pathlib import Path
MODEL_ID="google/siglip-base-patch16-256"
REVISION="b078df89e446d623010d890864d4207fe6399f61"
FILES=("README.md","config.json","preprocessor_config.json","model.safetensors")
WEIGHT_SHA256="f0cee7c815135c44a515eff72ab3040499744920442bc25567cd04efc93f8f65"
def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024),b""): h.update(part)
    return h.hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)
def ensure_assets(out):
    folder=Path(out)/"teacher/assets";folder.mkdir(parents=True,exist_ok=True)
    url=f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url,timeout=60) as response: info=json.load(response)
    if info["sha"]!=REVISION or info.get("cardData",{}).get("license")!="apache-2.0":
        raise ValueError("Unexpected exact model revision/license")
    save(folder/"upstream_model_info_pinned.json",info)
    upstream={row["rfilename"]:row for row in info["siblings"]};records=[]
    for name in FILES:
        row=upstream[name];path=folder/name;started=time.perf_counter()
        reused=path.exists()
        if not reused:
            partial=path.with_suffix(path.suffix+".partial")
            if partial.exists(): raise RuntimeError("Prior partial asset download requires explicit reviewed recovery")
            request=f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
            try:
                with urllib.request.urlopen(request,timeout=120) as response,partial.open("xb") as target:
                    while True:
                        block=response.read(4*1024*1024)
                        if not block: break
                        target.write(block)
            except Exception as error:
                save(folder/("download_failure_"+name.replace(".","_")+".json"),dict(url=request,error=repr(error)))
                raise
            os.replace(partial,path)
        actual=digest(path)
        if path.stat().st_size!=row["size"]: raise ValueError("Asset byte count differs: "+name)
        if "lfs" in row:
            if actual!=row["lfs"]["sha256"]: raise ValueError("Asset upstream LFS SHA differs: "+name)
        else:
            data=path.read_bytes();gitsha=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
            if gitsha!=row["blobId"]: raise ValueError("Asset upstream git-blob SHA differs: "+name)
        records.append(dict(name=name,path=str(path),sha256=actual,bytes=path.stat().st_size,reused=reused,seconds=time.perf_counter()-started))
    receipt=dict(status="PASS",model_id=MODEL_ID,revision=REVISION,license="apache-2.0",
                 files=records,remote_code=False,model_loads=0,model_forwards=0,
                 source_url=f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}")
    save(folder/"assets_verified.json",receipt)
    return receipt
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    result=ensure_assets(parser.parse_args().out)
    print(json.dumps(result))
