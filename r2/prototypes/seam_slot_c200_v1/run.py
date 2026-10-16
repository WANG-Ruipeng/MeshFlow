"""Actual CPU prototype packaging/OT checks. Never starts generator training.

Stages: prepare, validate, all, status. Completed work verifies hashes; no
silent retries/overwrites. The separate DESIGN.md specifies the future C200.
"""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,sys,time,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'experiments/r2_seam_slot_c200_prototype_v1'
PRIOR=ROOT/'experiments/r2_idea_1b_3_4a_diagnostic_v1/idea3'
sys.path[:0]=[str(ROOT/'r2/src'),str(HERE)]
import slot_data as S
import corner_contract as CC
from check_corner_contract import run_checks

PREFIXES=('73b369','3f7417','85b169','8c9b20')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise RuntimeError('Refusing overwrite: '+str(p))
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(p,rows):
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def verify(reg):
    for p,h in reg['protected_hashes'].items():
        if sha(p)!=h:raise RuntimeError('Protected source changed: '+p)
    for p,h in reg['entry_hashes'].items():
        if sha(p)!=h:raise RuntimeError('Frozen prototype code changed: '+p)

def prepare():
    path=OUT/'registration.json'
    if path.exists():reg=read(path);verify(reg);return reg
    if not (OUT/'resources_initial.json').exists():raise RuntimeError('Resource record required')
    old=read(PRIOR/'registration.json')
    records=[]
    for prefix in PREFIXES:
        found=[r for r in old['records'] if r['group']=='TRAIN47' and r['parent_id'].startswith(prefix)]
        if len(found)!=1:raise RuntimeError('Ambiguous TRAIN condition')
        records.append(found[0])
    protected={}
    for p in [ROOT/'AGENTS.md',ROOT/'WORKSPACE.md',ROOT/'r2/docs/research_route_memory.md',ROOT/'r2/docs/research_route_registry.json',ROOT/'native_t1/docs/control_metrics.md',
              PRIOR/'registration.json',PRIOR/'per_source_interface.csv',PRIOR/'per_condition.csv',
              ROOT/'r2/src/meshflow_control/training/coupling.py',ROOT/'r2/src/meshflow_control/models/native.py',
              ROOT/'r2/src/meshflow_control/models/geometry.py',ROOT/'r2/src/meshflow_control/training/losses.py']:
        protected[str(p)]=sha(p)
    for r in records:
        for pk,hk in [('condition_path','condition_sha256'),('parent_npz','parent_sha256'),('quality_path','quality_sha256'),('mask_path','mask_sha256'),('proof_path','proof_sha256')]:
            assert sha(r[pk])==r[hk]
            protected[r[pk]]=r[hk]
    code={str(p):sha(p) for p in HERE.glob('*.py')}
    reg=dict(schema='seam_slot_C_only_prototype_cpu_v1',status='FROZEN_BEFORE_REAL_DATA_OT',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),records=records,protected_hashes=protected,entry_hashes=code,
        scope='CPU prototype and concrete C200 design; not execution of 200-step generator training',
        user_choice='C-only learned occupied/Null decisions; no source-occupancy rollout oracle',
        label_scope='Single legal source owner per C edge. Selected four cases have no multi-edge owner or third-C reference. Fail explicitly outside this subset.',
        CPU_cases=dict(parents=4,alphas=[1.,1.05],random_replicates=2,samples_per_arm=16,arms=['FULL4_C200','SEAM_SLOT_C200'],new_bulk_OT_calls=32),
        scientific_limits='Synthetic CPU paths/identity tests, not neural generator results. No generation, D0, GPU, weights loaded or optimizer updates.',
        occupancy_contract='C-only features to independent 32-hidden MLP. Training target source-supported edge label. Null candidate remains a normal bulk face. Inference threshold logit>=0; no Y/source-label input.',
        future_training='Two matched 200-update arms from FULL_S6000 matched generator/AdamW; separately initialized occupancy optimizer; not executed by this entrypoint',
        stopping='Stop after CPU reports; no automatic 200/6000 training, generation, cleanup, commit or push')
    write(path,reg);return reg

def make_case(record,alpha):
    p,c=arr(record['parent_npz']),arr(record['condition_path'])
    known=c['source_face_ids'].astype(np.int64);free=c['free_source_ids'].astype(np.int64)
    n=len(p['source_faces']);k=len(known)
    v=p['model_vertices'].copy();v[:,0]*=np.float32(alpha)
    full=p['full_target'].reshape(n,3,3).copy();full[:,:,0]*=np.float32(alpha)
    pre=p['pre_ot_full'].copy();pre[:,:,0]*=np.float32(alpha)
    source_ids=np.unique(p['source_faces'][known])
    lookup={int(vi):i for i,vi in enumerate(source_ids)}
    c_original=np.asarray([[lookup[int(vi)] for vi in f] for f in p['source_faces'][known]],np.int64)
    c_model=np.asarray([[lookup[int(vi)] for vi in f] for f in p['source_face_vertex_ids'][known]],np.int64)
    cv=v[source_ids];edges,features=S.edge_features(cv,c_original,n)
    with (PRIOR/'per_source_interface.csv').open(encoding='utf-8-sig',newline='') as h:
        legal=[r for r in csv.DictReader(h) if r['condition_id']==record['condition_id']]
    byedge={tuple(sorted((int(r['vertex_u']),int(r['vertex_v'])))):r for r in legal}
    labels=np.zeros(len(edges),bool);owner=np.full(len(edges),-1,np.int64);triple=np.full((len(edges),3),-1,np.int64)
    for ei,(_,u,w) in enumerate(edges):
        su,sw=int(source_ids[u]),int(source_ids[w]);r=byedge.get(tuple(sorted((su,sw))))
        if r is None:continue
        if int(r['Y_legal_C_edges'])!=1 or r['third_vertex_already_in_C']=='True':
            raise RuntimeError('Minimal prototype requires explicit multi-owner/third-C design extension')
        labels[ei]=True;owner[ei]=int(r['Y_source_face']);triple[ei]=[sw,su,int(r['third_vertex_id'])]
    if len(set(owner[labels].tolist()))!=int(labels.sum()):raise RuntimeError('Duplicated owner')
    assert np.array_equal(v[p['source_face_vertex_ids']],full)
    return dict(full_target=full,pre_ot_full=pre,source_face_vertex_ids=p['source_face_vertex_ids'],
        known_ids=known,free_ids=free,C_vertices=cv,C_faces_original=c_original,C_faces_model=c_model,
        C_source_vertex_ids=source_ids,occupied_labels=labels,edge_owner_face=owner,edge_source_triple=triple,
        edge_features=features,N=n,K=k)

def validate(reg):
    if (OUT/'completion.json').exists():
        done=read(OUT/'completion.json');verify(reg)
        for p,h in done['output_hashes'].items():assert sha(p)==h,p
        print(json.dumps(dict(status='COMPLETE_VERIFIED_NO_RERUN',samples=done['sample_rows'])));return
    if (OUT/'attempt.json').exists():raise RuntimeError('Previous incomplete attempt: inspect failure; no silent retry')
    write(OUT/'attempt.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),code=reg['entry_hashes']))
    start=time.perf_counter();rows=[];cases=[];layout_checks=[];ot_calls=0
    contracts=run_checks();write(OUT/'corner_contract_checks.json',contracts)
    try:
        for ci,record in enumerate(reg['records']):
            for ai,alpha in enumerate(reg['CPU_cases']['alphas']):
                case=make_case(record,alpha);n,k=case['N'],case['K'];labels=case['occupied_labels'];E=len(labels)
                prefix=record['parent_id'][:8]+('_clean' if ai==0 else '_width105')
                # A deployable inference bundle contains C only. Labels live in a different training file.
                inference={key:case[key] for key in ['C_vertices','C_faces_original','C_faces_model']}
                inference['total_N']=np.asarray(n,np.int64)
                (OUT/'inference_inputs').mkdir(exist_ok=True)
                np.savez_compressed(OUT/'inference_inputs'/f'{prefix}.npz',**inference)
                (OUT/'training_labels').mkdir(exist_ok=True)
                np.savez_compressed(OUT/'training_labels'/f'{prefix}.npz',**case)
                for label,o in [('all_NULL',np.zeros(E,bool)),('all_OCCUPIED',np.ones(E,bool)),('alternating',np.arange(E)%2==0)]:
                    layout=S.inference_layout(**inference,occupied=o)
                    assert layout['N']==n and len(layout['seam_slots'])+len(layout['bulk_slots'])==n-k
                    assert layout['free_coordinate_count']==9*(n-k)-6*int(o.sum())
                    layout_checks.append(dict(condition=prefix,pattern=label,N=n,seam=int(o.sum()),bulk=len(layout['bulk_slots']),free_coordinates=layout['free_coordinate_count'],source_Y_access=False,status='PASS'))
                cases.append(dict(condition=prefix,split='TRAIN',N=n,K=k,edges=E,occupied=int(labels.sum()),Null=int((~labels).sum()),alpha=alpha))
                for repeat in range(2):
                    seed=2026101600+100*ci+10*ai+repeat
                    z=np.random.RandomState(seed).randn(n-k,3,3);eps=np.random.RandomState(seed+10000).randn(n-k,3,3)
                    t=np.float32(1/(1+np.exp(-np.random.RandomState(seed+20000).randn())))
                    samples=[]
                    for arm in ('FULL4_C200','SEAM_SLOT_C200'):
                        sample=S.supervised_sample(case,z,eps,t,seed+30000,arm);ot_calls+=1
                        expected=case['full_target'][sample['source_face_ids'][:,None],sample['source_corner_permutations']]
                        assert np.array_equal(expected,sample['x1'])
                        kw=dict(known_face_mask=sample['known_mask'],valid_face_mask=sample['valid_mask'],anchor_vertex_ids=sample['anchor_ids'],c_vertices=sample['C_vertices'])
                        yhat=CC.straight_endpoint_estimate(sample['xt'],sample['u'],float(t),**kw)
                        error=float(np.max(np.abs(yhat-sample['x1'])))
                        if error>1e-6:raise ValueError('Straight endpoint numerical check failed')
                        loss=CC.masked_velocity_loss(sample['u'],sample['u'],sample['known_mask'],sample['valid_mask'],sample['anchor_ids'])
                        assert loss['loss']==0 and loss['free_scalar_count']==int(sample['free_coordinate_count'])
                        fake=np.full_like(sample['u'],123.)
                        nxt=CC.euler_step(sample['xt'],fake,.02,**kw)
                        fixed=sample['anchor_ids']>=0
                        assert np.array_equal(nxt[fixed],sample['xt'][fixed])
                        decoded=CC.decode_shared_indices(sample['x1'],**kw)
                        assert len(decoded['faces'])==n
                        assert np.array_equal(decoded['corner_to_vertex'][fixed],sample['anchor_ids'][fixed])
                        # This decoder deliberately does not claim to weld arbitrary free corners.
                        path=OUT/'samples'/arm/f'{prefix}_r{repeat}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(path,**sample)
                        rows.append(dict(condition=prefix,repeat=repeat,arm=arm,N=n,K=k,t=float(t),seed=seed,
                            seam_faces=int(sample['occupied_for_FM'].sum()),free_coordinates=int(sample['free_coordinate_count']),
                            endpoint_max_abs=error,source_face_corner_bijection=True,anchors_exact=True,face_budget_exact=True,
                            new_OT_calls=1,source_occupancy_used='TRAINING_LABEL_ONLY',status='PASS'))
                        samples.append(sample)
                    assert np.array_equal(samples[0]['face_permutation'],samples[1]['face_permutation'])
        assert len(rows)==32 and ot_calls==32
        csvwrite(OUT/'per_sample.csv',rows);csvwrite(OUT/'conditions.csv',cases);csvwrite(OUT/'inference_layout_checks.csv',layout_checks)
        verify(reg)
        write(OUT/'completion.json',dict(status='COMPLETE_CPU_DATA_CONTRACT',sample_rows=len(rows),new_OT_calls=ot_calls,
            C_only_layout_checks=len(layout_checks),generator_forwards=0,D0_forwards=0,GPU_workers=0,optimizer_updates=0,
            corner_checks=contracts,seconds=time.perf_counter()-start,protected_unchanged=True,
            output_hashes={str(p):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('completion.json',)},
            limitations='No trained occupancy predictions, no 200-step FM run, no rollout or D0/GEO_C evaluation. The CPU decoder guarantees only explicit anchor ID reuse.'))
        print(json.dumps(dict(status='COMPLETE_CPU_DATA_CONTRACT',sample_rows=len(rows),OT=ot_calls)))
    except Exception:
        write(OUT/'failed.json',dict(error=traceback.format_exc(),OT_calls_returned=ot_calls,rows_completed=len(rows)))
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','validate','all','status'],default='status');args=ap.parse_args()
    if args.stage=='status':
        print(json.dumps(dict(registration=(OUT/'registration.json').exists(),complete=(OUT/'completion.json').exists(),failed=(OUT/'failed.json').exists())));
    else:
        r=prepare()
        if args.stage in ('validate','all'):validate(r)
