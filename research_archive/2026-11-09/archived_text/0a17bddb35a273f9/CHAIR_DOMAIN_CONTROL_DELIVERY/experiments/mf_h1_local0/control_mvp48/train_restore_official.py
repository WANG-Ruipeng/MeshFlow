"""Restore the exact registered official Chair asset; no model/GPU operation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[3]/'native_t1/runs/control_mvp48_v1'
SHA='bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d'
SIZE=2084287208
BASE='https://huggingface.co/datasets/qsun2001/meshflow'
POINTER=BASE+'/raw/main/v1/120m-ot-v-chair/checkpoints/last.pt'
DOWNLOAD=BASE+'/resolve/main/v1/120m-ot-v-chair/checkpoints/last.pt'
DEST=ROOT/'checkpoints/official_chair_last.pt'

def write(value):
    path=ROOT/'registration/train_official_asset.json'
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def main():
    tick=time.perf_counter()
    record=dict(status='VERIFYING_SOURCE',source=DOWNLOAD,pointer_source=POINTER,
        expected_bytes=SIZE,expected_sha256=SHA,path=str(DEST),downloaded_bytes=0,
        model_forwards=0,optimizer_updates=0,CUDA_initialized=False)
    DEST.parent.mkdir(parents=True,exist_ok=True)
    if DEST.exists():
        if DEST.stat().st_size!=SIZE or file_hash(DEST)!=SHA:raise ValueError('Existing destination identity mismatch; no overwrite')
        record.update(status='VERIFIED',existing=True,actual_sha256=SHA,bytes=SIZE)
        write(record);print(json.dumps(record));return
    partial=DEST.with_suffix('.pt.partial')
    if partial.exists():raise FileExistsError('Preserving previous partial download; no automatic restart')
    if shutil.disk_usage(DEST.parent).free < SIZE+12*1024**3:raise RuntimeError('Insufficient disk for asset plus training checkpoints')
    try:
        with urllib.request.urlopen(POINTER,timeout=20) as response: pointer=response.read(4096).decode()
        if ('oid sha256:'+SHA) not in pointer.splitlines() or ('size '+str(SIZE)) not in pointer.splitlines():
            raise ValueError('Official LFS pointer differs from pinned Chair identity')
        record.update(status='DOWNLOADING',pointer=pointer);write(record)
        digest=hashlib.sha256();total=0;last=time.perf_counter()
        with urllib.request.urlopen(DOWNLOAD,timeout=60) as response,partial.open('xb') as output:
            while True:
                block=response.read(8*1024*1024)
                if not block:break
                output.write(block);digest.update(block);total+=len(block)
                if total>SIZE:raise ValueError('Official payload larger than pinned size')
                if time.perf_counter()-last>=10:
                    record.update(downloaded_bytes=total,seconds=time.perf_counter()-tick)
                    write(record);print(json.dumps(dict(status='DOWNLOADING',bytes=total,total=SIZE)),flush=True)
                    last=time.perf_counter()
            output.flush();os.fsync(output.fileno())
        actual=digest.hexdigest()
        if total!=SIZE or actual!=SHA:raise ValueError('Downloaded official Chair content identity mismatch')
        partial.replace(DEST)
        record.update(status='VERIFIED',downloaded_bytes=total,bytes=total,actual_sha256=actual,seconds=time.perf_counter()-tick)
        write(record);print(json.dumps(record),flush=True)
    except Exception as exc:
        record.update(status='BLOCKED_OFFICIAL_DOWNLOAD',error=type(exc).__name__+': '+str(exc),
            partial_bytes=partial.stat().st_size if partial.exists() else 0,seconds=time.perf_counter()-tick)
        write(record);raise

if __name__=='__main__':main()
