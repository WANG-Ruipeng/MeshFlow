from pathlib import Path
import csv, datetime, hashlib, io, json, os, re, subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_slot_c200_v1'
ARMS=('FULL4_C200','SEAM_SLOT_C200')
L=float(np.sqrt(3)*2/.3762)
def resolve(p):
    s=str(p).replace('\\','/')
    if os.name!='nt' and re.match(r'^[A-Za-z]:/',s):s='/mnt/'+s[0].lower()+s[2:]
    if os.name=='nt' and re.match(r'^/mnt/[a-z]/',s):s=s[5].upper()+':/'+s[7:]
    p=Path(s);return p if p.is_absolute() else ROOT/p
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(resolve(p).read_text(encoding='utf-8-sig'))
def arrays(p):
    with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):
    h=hashlib.sha256()
    with resolve(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,Path):return str(x)
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return x.item()
    return x
def write(p,x,replace=False):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True);x=clean(x)
    if p.exists() and not replace:
        if read(p)!=x:raise ValueError('Frozen output differs '+str(p))
        return p
    temp=p.with_name(p.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,p);return p
def csvwrite(p,rows,replace=False):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True);rows=clean(list(rows));keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=keys);w.writeheader()
    for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    text=b.getvalue()
    if p.exists() and not replace:
        if p.read_text(encoding='utf-8-sig')!=text.replace('\r\n','\n'):raise ValueError('Frozen CSV differs '+str(p))
        return p
    p.write_text(text,encoding='utf-8-sig',newline='');return p
def resources(path=None):
    def run(cmd):
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=20);return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except Exception as e:return dict(error=repr(e))
    r=dict(utc=utc(),gpu=run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),processes=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv']),cpu_logical=os.cpu_count(),loadavg=list(os.getloadavg()) if hasattr(os,'getloadavg') else None,memory=Path('/proc/meminfo').read_text() if Path('/proc/meminfo').exists() else None,cpu_work=run(['ps','-eo','pid,pcpu,pmem,args','--sort=-pcpu']))
    if path:write(path,r)
    return r
def event(kind,**fields):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(clean(dict(utc=utc(),event=kind,**fields)),ensure_ascii=False)+'\n')
