"""Register the two authorized CPU checks; inspect existing array schemas."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,subprocess,time,sys
from pathlib import Path
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_post_rescue_diagnostic_v1'
SOURCE=ROOT/'experiments/r2_bench_parts_convergence_v1'
LABELS=('SEAM_S15000','SEAM_S21000')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def capture(args):
    r=subprocess.run(args,capture_output=True,text=True);return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def resources():
    return dict(unix=time.time(),gpu=capture(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
      gpu_processes=capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']),
      cpu_cores=os.cpu_count(),cpu_percent=psutil.cpu_percent(interval=.25),available_RAM_bytes=psutil.virtual_memory().available,
      load_average=os.getloadavg(),processes=capture(['ps','-eo','pid,pcpu,pmem,comm','--sort=-pcpu']))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inventory=[];ids={}
    for label in LABELS:
        paths=sorted((SOURCE/'evaluation'/label).glob('*/post/post.npz'));ids[label]=[p.parent.parent.name for p in paths]
        assert len(paths)==48
        for p in paths:
            b=p.parent.parent
            files=[b/'raw/raw.npz',b/'raw/source_condition.npz',p,p.parent/'post_canonical.npz',p.parent/'pre_geo.npz',p.parent/'proposal.npz',p.parent/'geometry.json',p.parent/'receipt.json']
            inventory.append(dict(label=label,request_id=b.name,files={str(f):sha(f) for f in files}))
    assert ids[LABELS[0]]==ids[LABELS[1]]
    protocol=dict(status='FROZEN',version='bench_post_rescue_diagnostic_v1',registered_unix=time.time(),source=str(SOURCE),labels=LABELS,
      per_label=48,common_requests=48,total_inputs=96,
      neural_forwards=0,training_updates=0,backwards=0,GPU_calls=0,new_noises=0,new_queries=0,
      hinge=dict(fraction_L=.015,require_opposite_winding=True,engine='retained HINGE_OPP015',additional_policies=[]),
      point_contact='Independent diagnostic: exact indexed seam-tip/bulk contact, with legal bulk-edge components. Point contact is not a legal edge or validated attachment.',
      deletion='Read saved RAW/D0/PRE_GEO/GEO mapping and exact deletion records. No radius sweep; count and area separately; no unique source reconstruction.',
      workers=dict(hinge=2,deletion=2,root_point_diagnostic=1,BLAS_threads_per_process=1),
      prior_route='Retained Chair HINGE_OPP015 reduced conflicts and retained old legal edges; Bench had no HINGE. This tests category-specific frozen post geometry, not a new generator mechanism.',
      limits='No large audit, training, generation, new defaults, cleanup, commit or push. Closed-C seam fields remain NA.',inventory=inventory)
    write(OUT/'registration.json',protocol);write(OUT/'resources_before.json',resources())
    b=SOURCE/'evaluation'/LABELS[1]/ids[LABELS[1]][0]
    for f in (b/'raw/raw.npz',b/'post/post.npz'):
        print(str(f));print(json.dumps({k:[str(v.dtype),list(v.shape)] for k,v in arrays(f).items()},indent=2))
    print('torch_imported', 'torch' in sys.modules)
if __name__=='__main__':main()
