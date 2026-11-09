"""Independent read-only CPU audit; no model construction/forward or rescoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'r2/src')]
import numpy as np
from r2.pilots.bench_parts_convergence_v1 import data as D
from meshflow_control.seam import slot_data as CH
from r2.pilots.seam_abc_v1.slot_data_extension import supervised_sample as owner_sample

OUT=Path(__file__).parent; EXP=OUT.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def saved(name,value):
    p=OUT/name
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def stats(a):
    a=np.asarray(a);return dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()),mean=float(a.mean()))
start=time.perf_counter();manifest=read(EXP/'data/manifest.json');cases={};case_rows=[]
for row in manifest['rows']:
    c=D.load_case(row);cases[row['condition_id']]=c
    n,k=len(c['full_target']),len(c['known_ids']);known=c['known_ids'];free=c['free_ids']
    assert np.array_equal(np.sort(np.r_[known,free]),np.arange(n))
    assert np.array_equal(c['pre_ot_full']*np.float32(2),c['full_target'])
    assert np.array_equal(c['C_vertices'][c['C_faces_model']],c['full_target'][known])
    assert np.array_equal(c['C_source_vertex_ids'][c['C_faces_model']],c['source_face_vertex_ids'][known])
    assert np.array_equal(c['C'],c['full_target'][known]) and np.array_equal(c['Y'],c['full_target'][free])
    edges,feat=D.edge_features(c['C_vertices'],c['C_faces_original'],n)
    assert np.array_equal(edges,c['edge_records'])
    labels=c['occupied_labels'];active=np.flatnonzero(labels);owners=c['edge_owner_face'][active]
    assert len(set(owners.tolist()))==len(owners) and set(owners.tolist())<=set(free.tolist())
    thirdC=0
    for e in active:
        fi=int(c['edge_owner_face'][e]); triple=c['edge_source_triple'][e]
        assert set(triple.tolist())==set(c['source_face_vertex_ids'][fi].tolist())
        assert np.array_equal(triple[:2],c['C_source_vertex_ids'][edges[e,1:][::-1]])
        thirdC+=int(triple[2] in set(c['C_source_vertex_ids'].tolist()))
    alias=np.asarray(c.get('source_alias_labels',np.zeros(len(labels),bool)),bool)
    supported=np.asarray(c.get('source_supported_labels',labels|alias),bool)
    case_rows.append(dict(condition_id=row['condition_id'],parent_id=row['parent_id'],split=row['group'],part=row['part_type'],N=n,K=k,B=len(edges),owner=int(labels.sum()),alias=int(alias.sum()),null=int((~supported).sum()),third_in_C=thirdC,
                          plain_free_scalar=9*(n-k),seam_free_scalar=9*(n-k)-6*int(labels.sum()),keys=sorted(c) if not case_rows else None))
saved('case_identity_audit.json',dict(status='PASS',conditions=len(case_rows),rows=case_rows))

# Sample diverse actual cases and existing registered random streams, never model predictions.
schedule=read(EXP/'data/input_schedule.json')['steps']
selection=[]
for criterion in [lambda r:r['B']==0,lambda r:r['owner']>0,lambda r:r['alias']>0,lambda r:r['third_in_C']>0,lambda r:r['owner']==0 and r['B']>0]:
    for r in case_rows:
        if criterion(r) and r['condition_id'] not in selection:selection.append(r['condition_id']);break
for key in ['N','K','B','owner']:
    for r in [min(case_rows,key=lambda r:r[key]),max(case_rows,key=lambda r:r[key])]:
        if r['condition_id'] not in selection:selection.append(r['condition_id'])
checks=[];ot_calls=0
for ci,cid in enumerate(selection):
    c=cases[cid]
    for aug in [0,1]:
        r=dict(schedule[ci]['samples'][aug]);r['alpha']=[1.,.93][aug]
        for arm in ['PLAIN','SEAM']:
            s=D.sample_for_training(c,r,arm);ot_calls+=1;n=int(s['N']);k=int(s['K']);fixed=s['anchor_ids']>=0;free=~fixed
            ca={key:v.copy() for key,v in c.items()}
            for key in ['full_target','pre_ot_full','C_vertices']:ca[key][...,0]*=np.float32(r['alpha'])
            reconstructed=ca['full_target'][s['source_face_ids'][:,None],s['source_corner_permutations']]
            assert np.array_equal(s['x1'],reconstructed)
            assert np.array_equal(np.sort(s['source_face_ids']),np.arange(n))
            assert not s['u'][fixed].any()
            assert np.array_equal(s['xt'][fixed],ca['C_vertices'][s['anchor_ids'][fixed]])
            assert np.array_equal(s['free_corner_mask'],free)
            assert int(s['free_coordinate_count'])==int(free.sum())*3
            assert int(s['free_coordinate_count'])==9*(n-k)-(6*int(c['occupied_labels'].sum()) if arm=='SEAM' else 0)
            assert np.allclose(s['xt']+(1-float(s['t']))*s['u'],s['x1'],atol=1e-6,rtol=1e-6)
            z=np.random.RandomState(r['noise_seed']).standard_normal((n-k,3,3)).astype(np.float32)
            ep=np.random.RandomState(r['epsilon_seed']).standard_normal(z.shape).astype(np.float32)
            initial=np.zeros((n,3,3),np.float32);initial[k:]=np.float32(np.sqrt(.75))*z+np.float32(.5)*ep
            assert np.array_equal(s['x0'][free],initial[s['face_permutation']][free])
            if len(c['occupied_labels']):
                original=owner_sample(ca,z,ep,float(s['t']),r['permutation_seed']) if arm=='SEAM' else CH.supervised_sample(ca,z,ep,float(s['t']),r['permutation_seed'],'FULL4_C200');ot_calls+=1
                assert set(s)==set(original)
                assert all(np.array_equal(s[key],original[key]) for key in s)
            checks.append(dict(condition_id=cid,arm=arm,alpha=r['alpha'],N=n,K=k,free_scalar=int(free.sum())*3,status='PASS',same_chair_packer=len(c['occupied_labels'])>0))
saved('packing_audit.json',dict(status='PASS',checks=checks,actual_CPU_OT_calls=ot_calls,model_forwards=0))

# Read saved inference layouts only. Labels are used here for diagnostic comparison, never inference.
evalm=read(EXP/'evaluation_inputs/manifest.json');unique={q['condition_id']:q for q in evalm['requests']};layout_rows=[]
for cid,q in sorted(unique.items()):
    c=cases[cid];truth=c['occupied_labels'].astype(bool);alias=np.asarray(c.get('source_alias_labels',np.zeros(len(truth),bool)),bool)
    supported=np.asarray(c.get('source_supported_labels',truth|alias),bool);null=~supported
    a=D.arrays(EXP/'evaluation/SEAM_S21000/layouts'/cid/'layout.npz');pred=a['occupied'].astype(bool)
    assert np.array_equal(a['edge_records'],c['edge_records']) and pred.shape==truth.shape
    expected=D.inference_layout(c['C_vertices'],c['C_faces_original'],c['C_faces_model'],int(c['N']),pred)
    for key in ['anchor_ids','known_mask','free_corner_mask','seam_slots','bulk_slots']:assert np.array_equal(a[key],expected[key])
    row=dict(condition_id=cid,split=q['group'],part=q['part_type'],B=len(truth),source_owner=int(truth.sum()),source_alias=int(alias.sum()),source_null=int(null.sum()),predicted=int(pred.sum()),TP=int((pred&truth).sum()),FN=int((~pred&truth).sum()),FP=int((pred&~truth).sum()),TN=int((~pred&~truth).sum()),alias_activated=int((pred&alias).sum()),null_activated=int((pred&null).sum()),layout_exact=bool(np.array_equal(pred,truth)),
             edge_roles=[dict(edge_index=i,source_role='OWNER' if truth[i] else ('ALIAS' if alias[i] else 'NULL_OR_UNSUPPORTED'),predicted_active=bool(pred[i]),logit=float(a['occupancy_logits'][i]),source_owner_face=int(c['edge_owner_face'][i])) for i in range(len(truth))])
    layout_rows.append(row)
aggregate={}
for group in ['TRAIN','DEV']:
    rr=[r for r in layout_rows if r['split']==group]
    sums={key:sum(r[key] for r in rr) for key in ['B','source_owner','source_alias','source_null','predicted','TP','FN','FP','TN','alias_activated','null_activated']}
    aggregate[group]=dict(conditions=len(rr),open_conditions=sum(r['B']>0 for r in rr),exact_all_conditions=sum(r['layout_exact'] for r in rr),exact_open_conditions=sum(r['layout_exact'] and r['B']>0 for r in rr),**sums)
saved('final_layout_source_diagnostic.json',dict(status='PASS',aggregate=aggregate,rows=layout_rows,no_head_forward=True,not_task_ground_truth=True))

# Check production exposures against the actual registered entries, and logged batch reduction.
log_checks={};paired={};condition_counts={};rows_by_c={r['condition_id']:r for r in case_rows}
for arm,lo,hi in [('COMMON',1,12000),('PLAIN',12001,21000),('SEAM',12001,21000)]:
    nrow=0;bad=[];perbatch=collections.defaultdict(int);cc=collections.Counter();pc=collections.Counter();keys=['condition_id','parent_id','part_type','alpha','noise_seed','epsilon_seed','time_seed','permutation_seed']
    for line in (EXP/'training'/arm/'exposure.jsonl').open(encoding='utf-8'):
        r=json.loads(line);entry=schedule[r['step']-1]['samples'][r['micro']]
        if any(r[key]!=entry[key] for key in keys):bad.append([r['step'],r['micro'],'SCHEDULE'])
        expected=rows_by_c[r['condition_id']]['seam_free_scalar' if arm=='SEAM' else 'plain_free_scalar']
        if r['all_free_scalar_count']!=expected:bad.append([r['step'],r['micro'],'DENOMINATOR'])
        perbatch[r['step']]+=expected;cc[r['condition_id']]+=1;pc[r['parent_id']]+=1;nrow+=1
    assert not bad and nrow==(hi-lo+1)*8 and sorted(perbatch)==list(range(lo,hi+1))
    errors=[];logged=[]
    for line in (EXP/'training'/arm/'training_log.jsonl').open(encoding='utf-8'):
        r=json.loads(line);logged.append(r)
        assert r['free_coordinate_denominator']==perbatch[r['step']]==r['all_free_scalar_count']
        assert np.isclose(r['FM'],r['all_free_velocity_sse']/r['free_coordinate_denominator'],rtol=3e-6,atol=1e-7)
        assert np.isclose(r['seam_tip_velocity_sse']+r['bulk_velocity_sse'],r['all_free_velocity_sse'],rtol=1e-12,atol=1e-8)
    assert len(logged)==hi-lo+1
    log_checks[arm]=dict(status='PASS',steps=len(logged),exposures=nrow,per_condition=stats(list(cc.values())),per_parent=stats(list(pc.values())),first_step=logged[0]['step'],last_step=logged[-1]['step'],batch_loss_reduction_correct=True)
    condition_counts[arm]=dict(cc)
saved('stream_and_denominator_audit.json',dict(status='PASS',arms=log_checks,condition_exposure=condition_counts,paired_stream_exact=True))
saved('training_audit_receipt.json',dict(status='PASS',wall_seconds=time.perf_counter()-start,conditions_read=len(cases),packing_checks=len(checks),actual_CPU_OT_calls=ot_calls,model_constructions=0,neural_forwards=0,GPU_calls=0,rescoring_calls=0,historical_files_modified=0,layout_summary=aggregate,log_summary=log_checks))
print(json.dumps(dict(status='PASS',OT_calls=ot_calls,seconds=time.perf_counter()-start,layout=aggregate,log=log_checks),indent=2))
