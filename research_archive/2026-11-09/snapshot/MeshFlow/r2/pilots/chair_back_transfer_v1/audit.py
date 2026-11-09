"""Read-only CPU identity/call/array acceptance of the frozen back-panel run.

No model imports/construction, rendering or geometric scoring. Historical and
production artifacts are never modified. This module writes only its own
timestamped audit result after the frozen RAW and POST outputs are complete.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
from meshflow_control.seam.slot_data import inference_layout
from meshflow_control.source_winding import validate_source

EXPECTED=dict(generator_file_sha256='46be9a29d6d9e150205cc46cf8c952ae1cd32b2aa58bff24cb879eb451932f43',
              generator_state_sha256='6f27fc5828fe4de295d102dea258616d6a7059a1a2dddc87bb747b42d03ca5a8',
              head_file_sha256='6e8f9ab606403c9c2b96d7f6bf6a2c67dacd11d4c45179369414e420a75982df',
              head_state_sha256='f7e350448f3b2c459c0e18e052c36f60dfc5f41c97e214f7e7749aaa32dd0cb7',
              D0_file_sha256='6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36')
DEFAULT=ROOT/'experiments/r2_chair_back_transfer_v1'

def resolve(p):
    p=str(p)
    if os.name=='nt':p=p.replace('/mnt/e/MeshFlow','E:/MeshFlow')
    else:p=p.replace('E:\\MeshFlow','/mnt/e/MeshFlow').replace('E:/MeshFlow','/mnt/e/MeshFlow')
    q=Path(p);return q if q.is_absolute() else ROOT/q

def read(p):return json.loads(resolve(p).read_text(encoding='utf-8-sig'))
def arrays(p):
    with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with resolve(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def events(p):
    return [json.loads(line) for line in resolve(p).read_text(encoding='utf-8').splitlines() if line.strip()]
def receipt(p):
    value=read(p)
    require(value.get('status') in ('COMPLETE','PASS'),'Receipt is not complete: '+str(p))
    for name,digest in value.get('files',{}).items():
        require(sha(name)==digest,'Receipt artifact changed: '+name)
    return value
def qpath(q,*keys):
    for k in keys:
        if k in q:return resolve(q[k])
    raise KeyError('Missing registered path among '+repr(keys))
def assert_no_training(rows):
    for r in rows:
        text=str(r.get('event','')).upper()
        require(not any(x in text for x in ('BACKWARD','OPTIMIZER','TRAIN_FORWARD')),'Unauthorized training call in audit ledger')

def validate_layout(folder,q):
    a=arrays(folder/'layout.npz');r=receipt(folder/'receipt.json')
    inp=arrays(qpath(q,'inference_npz','input_path','inference_path'))
    require(set(inp)=={'C_vertices','C_faces_original','C_faces_model','total_N'},'C-only input must contain exactly four registered fields')
    n,k=int(inp['total_N']),len(inp['C_faces_model'])
    require(n==q['N'] and k==q['K'],'Registered N/K differ')
    require(128<=n<=256 and 2<=k<n,'Chair N/K domain differs')
    for key in ('C_vertices','C_faces_original','C_faces_model'):
        require(same(inp[key],a[key]),'Layout C changed: '+key)
    logits=a['occupancy_logits'];occupied=a['occupied']
    require(logits.shape==occupied.shape and np.isfinite(logits).all(),'Nonfinite/malformed occupancy logits')
    require(np.array_equal(occupied,logits>=0),'Occupancy decision is not logit>=0')
    expected=inference_layout(inp['C_vertices'],inp['C_faces_original'],inp['C_faces_model'],n,occupied)
    for key,value in expected.items():require(same(a[key],value),'Layout differs from C-only policy: '+key)
    require(not np.any(a.get('alias_corner_ids',np.asarray([-1]))>=0),'Unregistered alias/shared-tip layout')
    require(r['head']['sha256']==EXPECTED['head_file_sha256'] and r['head']['head_state_sha256']==EXPECTED['head_state_sha256'],'Layout producer is not frozen LONG head')
    require(r['C_bundle_sha256']==sha(qpath(q,'inference_npz')) and r['source_labels_read'] is False,'Head input identity/source policy differs')
    calls=events(folder/'calls.jsonl');assert_no_training(calls)
    started=[r for r in calls if r.get('event')=='HEAD_FORWARD_STARTED']
    returned=[r for r in calls if r.get('event')=='HEAD_FORWARD_RETURNED']
    require(len(started)==len(returned)==1,'Need exactly one recorded LONG head call per C')
    require(len(calls)==2,'Unexpected extra head ledger events')
    return a,dict(condition_id=q['condition_id'],edges=len(occupied),active=int(occupied.sum()),zero_active=int(occupied.sum())==0,
                  head_forwards=1,receipt_sha256=sha(folder/'receipt.json'))

def validate_request(base,q,layout):
    rawdir=base/q['request_id']/'raw';postdir=base/q['request_id']/'post'
    rr=receipt(rawdir/'receipt.json');pr=receipt(postdir/'receipt.json');proposal_receipt=receipt(postdir/'proposal_receipt.json')
    raw=arrays(rawdir/'raw.npz');source=arrays(rawdir/'source_condition.npz')
    for key in ('inference_npz','metadata_npz','source_npz','noise_npz'):
        require(sha(q[key])==q[key+'_sha256'],'Frozen input hash differs: '+key)
    original_source=arrays(q['source_npz'])
    require(set(source)==set(original_source) and all(same(source[k],original_source[k]) for k in source),'Output source metadata differs from frozen original C')
    require(rr['checkpoint']['sha256']==EXPECTED['generator_file_sha256'] and rr['checkpoint']['generator_state_sha256']==EXPECTED['generator_state_sha256'],'RAW producer is not frozen S2000')
    require(rr['generator_forwards']==50 and rr['training_updates']==0 and rr['source_labels_read'] is False,'RAW producer budget/source policy differs')
    traj_npz=arrays(rawdir/'trajectory.npz');tr=next((traj_npz[k] for k in ('state','trajectory') if k in traj_npz),None)
    require(tr is not None,'No saved trajectory state')
    zfile=arrays(qpath(q,'noise_npz'));require(set(zfile)=={'z','seed'},'Noise file must contain only z/seed')
    z=zfile['z'];n,k=int(layout['N']),int(layout['K'])
    require(int(zfile['seed'])==int(q.get('seed',q.get('noise_seed'))),'Frozen noise seed identity differs')
    require(z.dtype==np.float32 and z.shape==(n-k,3,3) and np.isfinite(z).all(),'Malformed saved Gaussian')
    require(tr.dtype==np.float32 and tr.shape==(51,n,3,3) and np.isfinite(tr).all(),'Euler50 trajectory malformed')
    require(same(tr[-1],raw['raw']),'RAW differs from last Euler state')
    fixed=layout['anchor_ids']>=0
    initial=np.zeros((n,3,3),np.float32);initial[k:]=z;initial[fixed]=layout['C_vertices'][layout['anchor_ids'][fixed]]
    require(same(tr[0],initial),'Initial state not saved Gaussian plus anchors')
    for step in range(51):require(same(tr[step][fixed],initial[fixed]),'Anchor moved at Euler state '+str(step))
    validate_source(source,layout['C_vertices'][layout['C_faces_model']])
    require(same(raw['C'],source['C_original']),'RAW original C corner order/coordinates differ')
    require(same(raw['vertices'][raw['faces']],raw['output']),'RAW indexed decode geometry differs')
    require(np.array_equal(raw['source_face_ids'],np.arange(n)),'RAW face-slot identity differs')
    require(np.array_equal(raw['raw_anchor_ids'],layout['anchor_ids']),'RAW anchor layout differs')
    require(int(raw['known_faces'])==k and int(raw['input_faces'])==n,'RAW count differs')
    gcalls=events(rawdir/'calls.jsonl');assert_no_training(gcalls)
    starts=[r for r in gcalls if r.get('event')=='FORWARD_STARTED'];ends=[r for r in gcalls if r.get('event')=='FORWARD_RETURNED']
    require(len(starts)==len(ends)==50 and len(gcalls)==100,'Euler call ledger does not contain exactly50 completed forwards')
    for ls in (starts,ends):require([r['euler_step'] for r in ls]==list(range(50)),'Euler call sequence differs')
    proposal=arrays(postdir/'proposal.npz')['official_unconstrained_proposal'];pre=arrays(postdir/'pre_geo.npz')['output'];post=arrays(postdir/'post.npz')
    require(proposal.dtype==np.float32 and proposal.shape==(n,3,3) and np.isfinite(proposal).all(),'D0 proposal malformed')
    restored=proposal.copy();restored[fixed]=layout['C_vertices'][layout['anchor_ids'][fixed]]
    require(same(restored,pre),'Pre-GEO is not single proposal plus exact C/seam restoration')
    require(same(post['C'],source['C_original']),'POST original C corner order/coordinates differ')
    require(same(post['vertices'][post['faces']],post['output']),'POST indexed coordinates differ')
    require(np.array_equal(post['source_face_ids'][:k],np.arange(k)),'POST changed C source faces')
    fids=post['source_face_ids'];corners=post['source_corner_ids']
    require(len(np.unique(fids))==len(fids) and np.all((fids>=0)&(fids<n)),'POST duplicated/invalid source face identity')
    require(np.array_equal(np.sort(corners,axis=1),np.tile(np.arange(3),(len(corners),1))),'POST corner transport invalid')
    transported=raw['raw_anchor_ids'][fids[:,None],corners];mapping={}
    for cv,pv in zip(raw['C_faces_original'].flat,post['faces'][:k].flat):
        cv,pv=int(cv),int(pv);require(cv not in mapping or mapping[cv]==pv,'POST split a C vertex');mapping[cv]=pv
    require(len(mapping)==len(layout['C_vertices']) and len(set(mapping.values()))==len(mapping),'POST C identity is not bijective')
    for ci,pi in mapping.items():
        require(same(post['vertices'][pi],layout['C_vertices'][ci]),'POST C vertex moved')
        require(np.all(post['faces'][transported==ci]==pi),'Retained seam anchor failed exact C index')
    pcalls=events(postdir/'calls.jsonl');assert_no_training(pcalls)
    ps=[r for r in pcalls if r.get('event')=='D0_FORWARD_STARTED'];pe=[r for r in pcalls if r.get('event')=='D0_FORWARD_RETURNED']
    require(len(ps)==len(pe)==1 and len(pcalls)==2,'Expected exactly one D0 call/request')
    require(pr['D0_forwards']==1 and proposal_receipt['D0_forwards']==1,'D0 receipt count differs')
    require(pr.get('official_D0_sha256',proposal_receipt.get('official_checkpoint_sha256'))==EXPECTED['D0_file_sha256'],'Wrong Chair D0 identity')
    require(pr.get('GEO_calls')==1,'Expected exactly one original GEO')
    return dict(request_id=q['request_id'],condition_id=q['condition_id'],N=n,K=k,generator_forwards=50,D0_forwards=1,
                active_seam=int(layout['occupied'].sum()),retained_seam=int(post['retained_seam_mask'].sum()),
                retained_free_faces=len(post['faces'])-k,anchor_states_verified=51,
                raw_last_forward_unix=float(ends[-1]['unix']),D0_started_unix=float(ps[0]['unix']),
                raw_sha256=sha(rawdir/'raw.npz'),post_sha256=sha(postdir/'post.npz'),status='PASS')

def run(root=DEFAULT,label='S2000'):
    root=resolve(root);manifest=read(root/'eval_manifest.json');reg=read(root/'registration.json')
    qs=manifest['requests'];require(len(qs)==16 and len({q['request_id'] for q in qs})==16,'Need exactly16 registered requests')
    unique={q['condition_id']:q for q in qs};require(len(unique)==8,'Need exactly8 independent C')
    for cid in unique:require(sum(q['condition_id']==cid for q in qs)==2,'Each C needs exactly2 frozen noises')
    # Scientific registration must explicitly identify the three frozen assets.
    serialized=json.dumps(reg,sort_keys=True)
    for key in ('generator_file_sha256','head_file_sha256','D0_file_sha256'):
        require(EXPECTED[key] in serialized,'Registration lacks exact protected identity: '+key)
    base=root/'evaluation'/label;layouts={};layout_rows=[]
    for cid,q in sorted(unique.items()):
        layouts[cid],r=validate_layout(base/'layouts'/cid,q);layout_rows.append(r)
    rows=[validate_request(base,q,layouts[q['condition_id']]) for q in sorted(qs,key=lambda q:q['request_id'])]
    require(max(r['raw_last_forward_unix'] for r in rows)<=min(r['D0_started_unix'] for r in rows),'All-RAW barrier violated')
    # Both protected parameter export identities remain present; files are not reloaded as models.
    generator_sidecar=read(ROOT/'experiments/r2_seam_quality_wave1_wave2_v1/inference/PLAIN_S2000_generator.json')
    head_sidecar=read(ROOT/'experiments/r2_seam_quality_four_v1/inference/LONG_occupancy.json')
    require(generator_sidecar['sha256']==EXPECTED['generator_file_sha256'] and generator_sidecar['generator_state_sha256']==EXPECTED['generator_state_sha256'],'S2000 source sidecar differs')
    require(head_sidecar['sha256']==EXPECTED['head_file_sha256'] and head_sidecar['head_state_sha256']==EXPECTED['head_state_sha256'],'LONG head source sidecar differs')
    return dict(status='PASS',schema='chair_back_transfer_independent_CPU_acceptance_v1',label=label,
                registered_requests=16,conditions=8,generator_forwards=800,head_forwards=8,D0_forwards=16,
                generator_training_updates=0,backward_calls=0,all_RAW_before_D0=True,layout_rows=layout_rows,rows=rows,
                expected_weights=EXPECTED,registration_sha256=sha(root/'registration.json'),manifest_sha256=sha(root/'eval_manifest.json'),
                validator_sha256=sha(__file__),validator_model_constructions=0,validator_neural_forwards=0,validator_GPU_calls=0,
                rescore_calls=0,rerender_calls=0,topology_quality_certified=False,
                training_absence_scope='Inference-only registered runner and inspected head/rollout/D0 ledgers; not a whole-machine process attestation')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(DEFAULT));p.add_argument('--label',default='S2000');a=p.parse_args()
    root=resolve(a.root);start=time.perf_counter();stamp=time.strftime('%Y%m%dT%H%M%S')+'_'+str(time.time_ns()%1000000000)
    folder=root/'audit'/stamp;folder.mkdir(parents=True,exist_ok=False)
    try:
        value=run(root,a.label);value['wall_seconds']=time.perf_counter()-start
        with (folder/'receipt.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        print(json.dumps({k:v for k,v in value.items() if k not in ('rows','layout_rows')},indent=2))
    except BaseException as e:
        with (folder/'failure.json').open('x',encoding='utf-8') as f:json.dump(dict(status='FAIL',error=repr(e),traceback=traceback.format_exc(),no_production_mutation=True),f,indent=2)
        raise
if __name__=='__main__':main()
