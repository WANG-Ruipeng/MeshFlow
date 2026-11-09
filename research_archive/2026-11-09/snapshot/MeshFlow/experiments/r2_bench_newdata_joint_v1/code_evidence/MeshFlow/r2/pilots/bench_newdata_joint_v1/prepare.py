"""Activate one reviewed new parent; reuse91 checked cases, check one new OT sample."""
from __future__ import annotations
import collections, json
import numpy as np
from .common import *
from . import data as D

def main():
    if (OUT/'cpu_gate.json').exists():
        r=read(OUT/'cpu_gate.json')
        for p,h in r['protected'].items():
            assert sha(p)==h,p
        return dict(status='PASS',mode='HASH_REUSE')
    write_new(OUT/'resources_prepare.json',resources())
    old=read(REFERENCE_ROOT/'data/manifest.json')
    old_gate=read(REFERENCE_ROOT/'cpu_gate.json')
    assert old_gate['status']=='PASS' and old_gate['neural_forwards']==0
    protected={str(REFERENCE_ROOT/'data/manifest.json'):sha(REFERENCE_ROOT/'data/manifest.json'),
               str(REFERENCE_ROOT/'cpu_gate.json'):sha(REFERENCE_ROOT/'cpu_gate.json')}
    for rec in old_gate['validation_receipts']:
        assert sha(rec['path'])==rec['sha256']
        protected[rec['path']]=rec['sha256']
    review_path=ROOT/'experiments/r2_bench_three_family_v1/tasks/reviewed_manifest.json'
    review=read(review_path)
    fresh=[dict(r) for r in review['rows'] if r['parent_id'].startswith('27eb79')]
    assert len(fresh)==2
    new=next(r for r in fresh if r['recipe_id']=='R1')
    held=next(r for r in fresh if r['recipe_id']=='R2')
    assert new['actual_semantic_review']=='PASS_AI_PROXY' and new['C_Y_exact_complement']
    new.update(part_type='SEAT_BACK_SHELL',official_split='train',
        review_status='ACTIVE_REVIEWED_SOURCE_CONDITION',
        experiment_training_membership=True,activation_basis='User authorized1500 joint adaptation; prior fixed-head gate remains historical')
    held.update(part_type='ARM_LEG_FRAME',experiment_training_membership=False,
                split='SAME_PARENT_HELDOUT')
    train=[*old['train'],new];dev=old['dev'];tp={r['parent_id'] for r in train}
    assert (len(train),len(tp),len(dev))==(92,27,39)
    assert new['parent_id'] not in {r['parent_id'] for r in old['train']}
    assert not tp & {r['parent_id'] for r in dev}
    protected[str(review_path)]=sha(review_path)
    counts={};role=[];files={}
    for row in train+dev+[held]:
        for k in ('case','input','paired','quality','selection'):
            p=row.get(k+'_path');h=row.get(k+'_sha256')
            if p and h:
                assert sha(p)==h,p
                protected[p]=h
                files[p]=resolve(p).stat().st_size
    for row in train:
        case=D.load_case(row)
        owner=np.asarray(case['occupied_labels'],bool)
        counts[row['condition_id']]=len(owner)
        role.append(dict(condition_id=row['condition_id'],parent_id=row['parent_id'],
                         edges=len(owner),owner=int(owner.sum()),inactive=int((~owner).sum())))
    totals={k:sum(r[k] for r in role) for k in ('edges','owner','inactive')}
    assert totals==dict(edges=1546,owner=1428,inactive=118),totals
    schedule=D.make_schedule(train)
    schedule['head_calls_per_update']=[sum(counts[s['condition_id']]>0 for s in e['samples']) for e in schedule['steps']]
    schedule['head_edges_per_update']=[sum(counts[s['condition_id']] for s in e['samples']) for e in schedule['steps']]
    assert min(schedule['head_edges_per_update'])>0
    cp=collections.Counter(s['parent_id'] for e in schedule['steps'] for s in e['samples'])
    assert sum(cp.values())==12000 and max(cp.values())-min(cp.values())<=2
    # Actual production matching on one formal-stream new-C example; no neural call.
    item=next(s for e in schedule['steps'] for s in e['samples'] if s['condition_id']==new['condition_id'])
    sample=D.sample_for_training(D.load_case(new),item)
    assert int(sample['N'])==255 and int(sample['K'])==85
    assert not np.asarray(sample['occupied_labels'],bool).any()
    assert int(sample['free_coordinate_count'])==1530
    assert np.isfinite(sample['x_t']).all() if 'x_t' in sample else all(np.isfinite(v).all() for v in sample.values() if isinstance(v,np.ndarray) and np.issubdtype(v.dtype,np.floating))
    m=dict(schema='BENCH_NEWDATA_JOINT_DATA_V1',rows=train+dev,train=train,dev=dev,
           same_parent_heldout=[held],train_parents=sorted(tp),protected_hashes=protected,
           actual_data_ready=True,source_manifest=str(REFERENCE_ROOT/'data/manifest.json'))
    inv=dict(train_parents=27,train_conditions=92,new_independent_parents=1,new_conditions=1,
        heldout_conditions=1,unchanged_old_conditions=91,N_range=[min(r['N'] for r in train),max(r['N'] for r in train)],
        K_range=[min(r['K'] for r in train),max(r['K'] for r in train)],
        condition_types=dict(collections.Counter(r['part_type'] for r in train)),
        role_counts=totals,head_class_weights=[totals['edges']/(2*totals['inactive']),totals['edges']/(2*totals['owner'])],
        head_forward_budget=sum(schedule['head_calls_per_update']),head_optimizer_updates=1500,
        planned_parent_exposure=dict(cp),unique_referenced_bytes=sum(files.values()),
        scope='Pool references plus DEV and heldout; no source assets copied; parent—not edge—is independent coverage unit')
    for name,obj in [('manifest',m),('schedule',schedule),('inventory',inv),('source_roles',role)]:
        p=OUT/'data'/f'{name}.json';write_new(p,obj);protected[str(p)]=sha(p)
    gate=dict(status='PASS',neural_forwards=0,new_production_sample_checks=1,
        matching_scope='One new all-Null formal-stream sample; inherited91CPU receipts reused',
        old_validation_receipts=old_gate['validation_receipts'],protected=protected,
        source_layout_teacher_forcing=True,inference_requires_C_only_predicted_layout=True,
        head_forward_budget=inv['head_forward_budget'],new_sample_free_scalars=1530)
    write_new(OUT/'cpu_gate.json',gate)
    return inv

if __name__=='__main__': print(json.dumps(main(),ensure_ascii=False))
