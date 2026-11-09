"""Read-only CPU readiness inventory. No model construction or new data draws."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import csv, json, hashlib, math, subprocess, time, re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ART = ROOT.parent / 'MeshFlow-Control-artifacts'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def resolve(p,base=ROOT):
    s=str(p).replace('\\','/')
    if re.match(r'^[A-Za-z]:/',s):s='/mnt/'+s[0].lower()+s[2:]
    p=Path(s);return p if p.is_absolute() else base/p
def command(args):
    try:return subprocess.run(args,text=True,capture_output=True,timeout=20).stdout.strip()
    except Exception as exc:return repr(exc)

tick=time.perf_counter()
resources=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),cpu_count=os.cpu_count(),
    loadavg=os.getloadavg(),memory=Path('/proc/meminfo').read_text(),
    gpu=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
    processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
    dispatch=dict(cpu_workers=1,cpu_threads=1,gpu_jobs=0,reason='Metadata, scalar recomputation, and CPU mmap checkpoint inspection only'))
write(OUT/'resources.json',resources)
source=ROOT/'experiments/r2_seam_geometry_guard_v1/normal/source_per_edge.csv'
with open(source,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
for r in rows:
    inward,height=float(r['inward_conormal_model']),float(r['height_model'])
    cosine=-inward/math.hypot(inward,height)
    r.update(recomputed_cosine=cosine,recomputed_angle_deg=math.degrees(math.acos(np.clip(cosine,-1,1))),
        cosine_recompute_abs_error=abs(cosine-float(r['normal_cosine'])),
        reject_strict_135_deg=cosine < -math.sqrt(.5),reject_cos_minus_0p7=cosine < -.7,
        reject_strict_150_deg=cosine < -math.sqrt(3)/2)
table(OUT/'source_angle_gate.csv',rows)
subset=[r for r in rows if r['in_C200_four_conditions']=='True']
summary=[]
for label,group in [('ALL_SOURCE_389',rows),('C200_SOURCE_62',subset)]:
    summary.append(dict(group=label,n=len(group),reject_135_deg=sum(r['reject_strict_135_deg'] for r in group),
        reject_cos_minus_0p7=sum(r['reject_cos_minus_0p7'] for r in group),reject_150_deg=sum(r['reject_strict_150_deg'] for r in group),
        h_positive=sum(float(r['height_model'])>1e-12 for r in group),r_positive=sum(float(r['inward_conormal_model'])>1e-12 for r in group),
        max_cosine_error=max(r['cosine_recompute_abs_error'] for r in group)))
table(OUT/'angle_summary.csv',summary)
table(OUT/'counterexamples.csv',[r for r in rows if r['reject_strict_135_deg']])

oldpath=ART/'data/train_manifest.json'
newpath=ROOT/'experiments/r2_user_data_scaling_12h_v1/data/train_manifest.json'
old,new=read(oldpath),read(newpath)
oldparents={r['uid']:r for r in old['parents']}
newparents=new['parents'];oldt=old['tasks'];newt=list(new['tasks'].values())
trainold=[r for r in oldt if r.get('split')=='train' and r.get('status')=='READY']
trainuids={r['uid'] for r in trainold if math.isclose(float(r['ratio']),.4)}
newuids=set(new['pools']['USER_LARGE_67'])
inventory=[]
for label,tasks,base in [('C20_ALL_LOCAL',[r for r in trainold if math.isclose(float(r['ratio']),.2)],oldpath.parent),
    ('R2_C20_26',[r for r in trainold if math.isclose(float(r['ratio']),.2) and r['uid'] in trainuids],oldpath.parent),
    ('C40',[r for r in trainold if math.isclose(float(r['ratio']),.4)],oldpath.parent),
    ('USER_LARGE_67',[r for r in newt if r.get('parent_id',r.get('uid')) in newuids],newpath.parent)]:
    missing=[];bad=[]
    for r in tasks:
        p=resolve(r.get('condition_npz',r.get('npz',r.get('condition_source'))),base)
        if not p.exists():missing.append(str(p));continue
        expected=r.get('condition_sha256',r.get('sha256'))
        if expected and sha(p)!=expected:bad.append(str(p))
    inventory.append(dict(pool=label,parents=len({r.get('parent_id',r.get('uid')) for r in tasks}),conditions=len(tasks),
        N_min=min(r['N'] for r in tasks),N_max=max(r['N'] for r in tasks),K_min=min(r['K'] for r in tasks),K_max=max(r['K'] for r in tasks),
        missing=len(missing),hash_mismatch=len(bad),missing_json=json.dumps(missing),hash_mismatch_json=json.dumps(bad)))
table(OUT/'data_inventory.csv',inventory)
shared=[]
for uid in sorted(trainuids & newuids):
    a,b=oldparents[uid],newparents[uid]
    pa=resolve(a['npz'],oldpath.parent);pb=resolve(b['npz'],newpath.parent)
    na,nb=a['N'],b['N'];ka=[];kb=[];same=None;equal_fields=[]
    if pa.exists() and pb.exists():
        with np.load(pa,allow_pickle=False) as za,np.load(pb,allow_pickle=False) as zb:
            ka=list(za.files);kb=list(zb.files)
            equal_fields=[k for k in set(ka)&set(kb) if np.array_equal(za[k],zb[k])]
            coords=[k for k in ('full_target','pre_ot_full','model_vertices','source_face_vertex_ids','triangles','output','full','target','vertices') if k in za and k in zb]
            same=bool(coords and all(np.array_equal(za[k],zb[k]) for k in coords))
    shared.append(dict(uid=uid,old_N=na,USER_N=nb,same_N=na==nb,old_path=str(pa),USER_path=str(pb),
        old_exists=pa.exists(),USER_exists=pb.exists(),same_common_geometry_fields=same,
        old_npz_keys=json.dumps(ka),USER_npz_keys=json.dumps(kb),equal_fields=json.dumps(sorted(equal_fields))))
table(OUT/'shared_parent_inventory.csv',shared)

weights_path=ROOT/'experiments/r2_seam_slot_c200_v1/WEIGHTS.json'
weight_rows=[r for r in read(weights_path)['weights'] if r['arm']=='SEAM_SLOT_C200']
checked=[]
for r in weight_rows:
    p=resolve(r['path']);item=dict(r,exists=p.exists())
    if p.exists():item.update(actual_bytes=p.stat().st_size,actual_sha256=sha(p));item['file_verified']=item['actual_sha256']==r['sha256']
    checked.append(item)
ck=next(r for r in checked if r['kind']=='checkpoint')
if ck.get('file_verified'):
    import torch
    payload=torch.load(resolve(ck['path']),map_location='cpu',weights_only=False,mmap=True)
    checks={}
    for key in ('generator_state_sha256','optimizer_state_sha256','head_state_sha256','head_optimizer_state_sha256','rng_sha256'):
        checks[key]=payload.get(key)==ck.get(key)
    def steps(name):
        result=[]
        for s in payload.get(name,{}).get('state',{}).values():
            if 'step' in s:result.append(float(s['step']))
        return dict(states=len(payload.get(name,{}).get('state',{})),steps=sorted(set(result)))
    ck.update(checkpoint_keys=sorted(payload),local_step=payload.get('local_step'),global_step=payload.get('global_step'),
        optimizer=steps('optimizer'),head_optimizer=steps('head_optimizer'),embedded_hash_checks=checks,
        optimizer_parameter_names_present=bool(payload.get('optimizer_parameter_names')),
        rng_present='rng' in payload,cpu_mmap_only=True,model_constructions=0,model_forwards=0)
    del payload
write(OUT/'checkpoint_audit.json',dict(status='CPU_INSPECTED',weights=checked))

sources=[source,oldpath,newpath,weights_path,Path(__file__),ROOT/'r2/docs/research_route_memory.md',
    ROOT/'r2/docs/research_route_registry.json',ROOT/'native_t1/docs/control_metrics.md',
    ROOT/'r2/pilots/seam_slot_c200_v1/train.py',ROOT/'r2/pilots/seam_slot_c200_v1/trainer_support.py',
    ROOT/'r2/prototypes/seam_slot_c200_v1/slot_data.py',ROOT/'experiments/mf_h1_local0/guided_native_geometry_corrector.py']
cards=[
 dict(id='A',name='USER versus historical C20/C40 condition mixture',status='NEEDS_DEFINITION',
      available='Existing condition files checked; no new random conditions materialized.',
      single_variable='A preregistered probability of USER conditions versus fixed C20/C40 pool, on a shared parent/geometry distribution.',
      missing=['Choose p_USER and finite update budget','Choose parent/version-matched subset or admit coupled data-coverage change',
          'Define conditional schedule and exposure policy','If using SEAM: source owner/multiple-edge/third-C-reference eligibility must be audited'],
      gate='Same source geometry, parent exposure, N and fixed panel; test USER use/connection and FF/I costs without selecting best seed.'),
 dict(id='B',name='Extend SEAM training',status='READY_WITH_SPEC',
      available='Full step200 generator/head checkpoints and both optimizer states inspected on CPU; existing inference exports retained.',
      single_variable='More updates on a frozen task/layout policy; freeze occupancy policy across compared endpoints.',
      missing=['Specify total additional updates and endpoints','Define head continuation/freeze without disabling seam_forward',
          'Independent continuation registration/adapter: old entry hardcodes 200 updates and original stream'],
      gate='No inference-only resume; matching generator/head optimizers and RNG, fixed paired panel, no checkpoint selection. Longer exposure is not demonstrated benefit.'),
 dict(id='C',name='Finite-surface seam-third-point protection',status='NEEDS_DEFINITION',
      available='389 source interface local frames and 62 four-C source interfaces; transported seam-third masks and preserved anchors.',
      single_variable='One explicitly defined differentiable illegal-overlap proxy using predicted terminal geometry, no simultaneous occupancy or GEO change.',
      missing=['Define finite-surface illegal gate and tau/margin','Choose penalty strength/time scope and finite training budget',
          'Replace detached diagnostic with gradient-bearing Yhat only at authorized train time','Validate value and gradient on legal and illegal geometric examples'],
      gate='135-degree gate rejects 1/389 legitimate source interfaces; 0/62 is local evidence only. A coordinate frame alone is not a constraint.'),
]
readiness=dict(schema='r2_quality_followup_readiness_v1',status='CPU_PREPARATION_COMPLETE',
    authority='Readiness only; no new training, sampling, optimizer update or default change authorized here.',
    model_calls=0,training_updates=0,new_random_data=0,cards=cards,angle_summary=summary,
    dataset_inventory=inventory,shared_parent_count=len(shared),same_N_shared=sum(r['same_N'] for r in shared),
    source_hashes={str(p):sha(p) for p in sources},seconds=time.perf_counter()-tick)
write(OUT/'readiness.json',readiness)
print(json.dumps(dict(status=readiness['status'],angle_summary=summary,inventory=inventory,
    shared_parent_count=len(shared),checkpoint_steps=ck.get('optimizer'),head_steps=ck.get('head_optimizer'),seconds=readiness['seconds']),ensure_ascii=False))
