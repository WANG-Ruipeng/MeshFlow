"""Isolated CPU attribution infrastructure. No torch/model/OT imports."""
from pathlib import Path
import os,sys,json,csv,hashlib,time,subprocess,collections,traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
SRC=ROOT/'native_t1/runs/chair_domain_control_v1'
OUT=ROOT/'native_t1/runs/chair_control_attribution_v1'
ARMS=('T1','Geo','Full')
NAMES={'T1':'Native','Geo':'Native+Geo','Full':'Native+Geo+Joint'}
ALPHAS=(.9,1.,1.1)
TOL=dict(atol=1e-8,rtol=1e-6)

def js(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(o):
 if isinstance(o,dict): return {str(k):clean(v) for k,v in o.items()}
 if isinstance(o,(tuple,list)): return [clean(x) for x in o]
 if isinstance(o,np.ndarray): return clean(o.tolist())
 if isinstance(o,np.generic): return clean(o.item())
 if isinstance(o,float) and not np.isfinite(o): return None
 if isinstance(o,Path): return str(o)
 return o
def savej(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(clean(o),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');os.replace(tmp,p)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ah(a):
 a=np.ascontiguousarray(a)
 h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def legacy_ah(a):
 # Original native_inpainting_metrics.array_sha definition is verified separately.
 return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def resolve(p):
 p=Path(p);return p if p.is_absolute() else SRC/p
def arrays(p):
 with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def conv(x):
 if x in ('','NA','None'):return None
 if x=='True':return True
 if x=='False':return False
 try:return float(x) if any(y in x for y in '.eE') else int(x)
 except (ValueError,TypeError):return x
def readcsv(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return [{k:conv(v) for k,v in r.items()} for r in csv.DictReader(f)]
def writecsv(name,rows,fields=None):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
 if not fields:fields=['status','reason']
 with p.open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for r in rows:w.writerow({k:('NA' if clean(v) is None else clean(v)) for k,v in r.items()})
 return str(p.relative_to(OUT))
def groups(rows,keys):
 d=collections.defaultdict(list)
 for r in rows:d[tuple(r.get(k) for k in keys)].append(r)
 return dict(sorted(d.items(),key=lambda t:str(t[0])))
def mean(xs):
 xs=[float(x) for x in xs if x is not None and np.isfinite(x)]
 return float(np.mean(xs)) if xs else None
def aggregate(rows,keys,metrics):
 return [dict(zip(keys,key),n_rows=len(rs),**{m:mean(r.get(m) for r in rs) for m in metrics}) for key,rs in groups(rows,keys).items()]
def parent_summary(rows,keys,metrics):
 parents=aggregate(rows,list(keys)+['uid'],metrics)
 sums=aggregate(parents,keys,metrics)
 for r in sums:r['n_parents']=r.pop('n_rows')
 return parents,sums
def event(kind,**kw):
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'execution_ledger.jsonl').open('a',encoding='utf8') as f:
  f.write(json.dumps(clean(dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),kind=kind,**kw)),ensure_ascii=False)+'\n')
def resources(tag):
 def cmd(args):
  try:return subprocess.run(args,text=True,capture_output=True,timeout=12).stdout.strip()
  except Exception as e:return repr(e)
 v=psutil.virtual_memory()
 procs=[]
 for p in psutil.process_iter(['pid','name','memory_info','cpu_times']):
  if any(s in (p.info['name'] or '').lower() for s in ('python','ot','audit')):
   procs.append(dict(pid=p.pid,name=p.info['name'],rss=p.info['memory_info'].rss if p.info['memory_info'] else None))
 r=dict(tag=tag,cpu_count=psutil.cpu_count(),cpu_percent=psutil.cpu_percent(interval=.2),load=os.getloadavg(),ram_total=v.total,ram_available=v.available,processes=procs,
        gpu=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']))
 event('resource_snapshot',**r);return r
def recs():
 p=js(SRC/'panels.json');return {r['case_id']:r for k in ('A','B','C') for r in p[k]}
def generated():
 result=[]
 for p in sorted((SRC/'evaluation').glob('phase*/*/ledger.json')):
  v=js(p)
  for r in v['rows']:result.append(dict(r,checkpoint_sha256=v['identity']['checkpoint_sha256'],ledger_path=str(p.relative_to(ROOT))))
 return result
def raw_for(row):
 r=recs()[row['case_id']];bank=arrays(r['input_path']);raw=arrays(row['raw_path'])
 return raw['raw'][int(bank['K']):].reshape(-1,3,3),bank,raw
def close(a,b,**kw):return bool(np.isclose(a,b,**(kw or TOL)))
def stage(name,fn):
 path=OUT/'stages'/f'{name}.json'
 if path.exists():
  s=js(path)
  if s['status']=='COMPLETE':
   for f,h in s['outputs'].items():
    if digest(OUT/f)!=h:raise RuntimeError('Completed stage output changed: '+f)
   event('stage_reused',stage=name);return s['result']
  event('stage_resume',stage=name,previous=s['status'])
 start=time.monotonic();event('stage_start',stage=name,code_hashes={p.name:digest(p) for p in CODE.glob('*.py')})
 savej(path,dict(status='RUNNING',name=name))
 try:
  result,outputs=fn()
  s=dict(status='COMPLETE',name=name,seconds=time.monotonic()-start,result=result,outputs={str(x):digest(OUT/x) for x in outputs})
  savej(path,s);event('stage_complete',stage=name,seconds=s['seconds'],result=result);return result
 except BaseException as e:
  savej(path,dict(status='FAIL',name=name,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));event('stage_fail',stage=name,error=repr(e));raise
