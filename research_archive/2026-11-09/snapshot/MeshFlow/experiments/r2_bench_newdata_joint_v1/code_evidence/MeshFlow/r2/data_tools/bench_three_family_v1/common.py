from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import hashlib, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'experiments/r2_bench_three_family_v1'
OLD = ROOT/'experiments/r2_bench_deployment_task_data_v1'
SOURCE_TASKS = OLD/'dataset/source_tasks_pending_adapter'
PREFIXES = ('27eb79','945214','585673')
VERSION = 'R2_BENCH_THREE_FAMILY_TARGET_V1'

def resolve(p):
    s=str(p).replace('\\','/')
    for pre in ('/mnt/e/MeshFlow/','E:/MeshFlow/'):
        if s.startswith(pre): return ROOT/s[len(pre):]
    p=Path(s)
    return p if p.is_absolute() else ROOT/p
def read(p): return json.loads(resolve(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(resolve(p).read_bytes()).hexdigest()
def save(p,x):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    txt=json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if p.exists(): assert p.read_text('utf-8')==txt, 'Refuse overwrite: '+str(p)
    else: p.write_text(txt,'utf-8')
def load(p):
    with np.load(resolve(p),allow_pickle=False) as a: return {k:a[k].copy() for k in a.files}
def npz(p,a):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        b=load(p);assert set(a)==set(b) and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes() for k in a)
    else:
        with p.open('xb') as f: np.savez_compressed(f,**a)
def event(kind,**fields):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'execution.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**fields),ensure_ascii=False)+'\n')

def freeze():
    p=OUT/'PROTOCOL.json'
    if p.exists():
        x=read(p)
        for f,h in x['protected_sources'].items(): assert sha(f)==h,f
        return x
    from r2.pilots.bench_adaptation_strength_v1.common import resources
    tasks=read(SOURCE_TASKS/'manifest.json')['rows']
    inventory=read(OLD/'asset_inventory/reviewed_candidates.json')['rows']
    parents=[];protected={}
    for prefix in PREFIXES:
        row=next(r for r in inventory if r['parent_id'].startswith(prefix))
        ts=[r for r in tasks if r['parent_id']==row['parent_id']]
        assert len(ts)==2 and all(not r['hold_reasons'] for r in ts)
        assert {r['recipe_id'] for r in ts}=={'R1','R2'}
        assert all(r['intended_eval_group'] in ('TRAIN_KNOWN_C','SAME_PARENT_HELDOUT_C') for r in ts)
        parents.append(dict(parent_id=row['parent_id'],source=row,tasks=ts))
        for path in [row['clean_case_path'],row['source_geometry_path']]+[r[k] for r in ts for k in ('input_path','paired_path','quality_path','image_path')]:protected[str(resolve(path))]=sha(path)
    for path in [SOURCE_TASKS/'manifest.json',OLD/'split_audit/new_data_split.json',OLD/'split_audit/staged_eval_contract.json',OLD/'COMPLETE.json']:
        protected[str(path)]=sha(path)
    x=dict(version=VERSION,parents=parents,protected_sources=protected,resources=resources(),
      task='At most3 TRAIN-side target versions and6 reconstructed semantic conditions; no generator/D0/training, no cleanup/default/commit/push.',
      current_N_contract=[128,256],static_N_audit_only=True,production_N_contract_unchanged=True,
      fixed_parent_prefixes=list(PREFIXES),reserved_unseen_prefixes=['617fa8','756108'],continued_hold=['750cec'],
      construction=dict(first='One shared collinear patch-boundary simplification, protect R1/R2 semantic boundaries, corner/branch/contact/hole topology; no angular threshold search.',
        fallback='If needed one separately recorded per-part low-poly reconstruction proposal, retain failed first proposal; freeze concrete component recipe before executing it.',
        geometric_tolerance_L=1e-7,normal_dot_min=1-1e-10,
        fidelity_probe_max_L=.0025,area_ratio=[.98,1.02],components_and_boundary_topology='preserve; no arbitrary weld or hole-fill',
        finite_geometry='Reuse original detector; no illegal/unresolved/zero-area/nonmanifold edge or vertex/winding conflict accepted',
        below128='Same-surface deterministic largest-triangle centroid subdivision to128 or129 with lineage; no duplicate or padding face',
        above256='Keep exact result and mark BLOCKED_N; no high-N head/generator trial in this task'),
      conditions='Reuse semantic extents, rebuild new target face identities via protected region lineage; R1 train, R2 same-parent holdout. New M frozen before partition. C/Y exact disjoint complement.',
      frozen_head='Separate freeze receipt before calls; once per eligible open new C on CPU, no augmentation/calibration/threshold search; closedC uses existing empty layout.',
      packing='C-only input and production mask frozen before reading Y; one-to-one real faces only. Failures retained, no source-owner mask corrections.',
      review='Actually viewed AI_PROXY original/new complete meshes plus C/Y, multiple matched views. Not human validation or generated-quality evidence.',
      workers=dict(geometry_max=3,render_max=2,threads_per_worker=1,GPU=0,head_CPU=1),
      initial_failure_events=['Windows profile FNM init warning; subsequent shells use login=False','Windows CIM access denied; existing WSL psutil used','First sandbox WSL access denied before computation; authorized existing WSL environment access then succeeded'],
      stopping='Finish all3 targets and up to6 task outcomes, N audit, packing, visual review, report and recovery; no training auto-launch.')
    save(p,x);event('PROTOCOL_FROZEN',protocol_sha256=sha(p),parents=len(parents),resources=x['resources'])
    return x

if __name__=='__main__':
    x=freeze();print(json.dumps(dict(parents=[p['parent_id'] for p in x['parents']],resources=x['resources']),ensure_ascii=False))
