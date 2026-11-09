"""Freeze 81 retained plus 10 accepted conditions, reusing exact CPU receipts."""
from __future__ import annotations
import collections
import json
from .common import ROOT, OUT, BASELINE_ROOT, read, resolve, sha, write_new
from . import data as D

DATA = ROOT / 'experiments/r2_bench_user_relation_rebuild_v1'
OLD_DATA = ROOT / 'experiments/r2_bench_user_data_v2_expansion_v1'


def main():
    if (OUT/'cpu_gate.json').exists():
        gate=read(OUT/'cpu_gate.json')
        for p,h in gate['protected'].items():
            assert sha(p)==h,p
        for record in gate['validation_receipts']:
            assert sha(record['path'])==record['sha256'],record['path']
            for p,h in read(record['path'])['code_hashes'].items():
                assert sha(p)==h,p
        return dict(status='PASS',mode='REUSED_HASH_CHECK_ONLY')
    ready=read(DATA/'DATASET_READY.json')
    manifest_path=resolve(ready['proposed_merged_manifest'])
    assert sha(manifest_path)==ready['proposed_merged_sha256']
    src=read(manifest_path)
    old_ready=read(OLD_DATA/'DATASET_READY.json')
    assert sha(old_ready['manifest_path'])==old_ready['manifest_sha256']
    old=read(old_ready['manifest_path'])
    assert sha(ready['manifest_path'])==ready['manifest_sha256']
    extension=read(ready['manifest_path'])
    train,dev=src['train'],src['dev']
    tp={r['parent_id'] for r in train};dp={r['parent_id'] for r in dev}
    assert (len(train),len(tp),len(dev),len(dp))==(91,26,39,8)
    assert not tp&dp and dev==old['dev'] and dev==extension['dev']
    assert all(r['official_split']=='train' for r in train)
    old_by={r['condition_id']:r for r in old['train']}
    new_by={r['condition_id']:r for r in extension['train']}
    combined={r['condition_id']:r for r in train}
    assert not set(old_by)&set(new_by)
    assert combined==dict(old_by,**new_by)
    protected=dict(src['protected_source_hashes'])
    for p in (DATA/'DATASET_READY.json',manifest_path,resolve(ready['manifest_path']),
              OLD_DATA/'DATASET_READY.json',resolve(old_ready['manifest_path'])):
        protected[str(p)]=sha(p)
    records=[old_ready['validation_new'],old_ready['validation_reused'],ready['production_CPU_receipt']]
    passed={};receipt_details=[]
    for record in records:
        path=record.get('receipt',record.get('validation_receipt'))
        expected=record.get('receipt_sha256',record.get('validation_receipt_sha256'))
        assert sha(path)==expected,path
        receipt=read(path)
        assert receipt['status']=='PASS' and receipt['blocked']==0,path
        protected[path]=expected
        for condition in receipt['condition_results']:
            assert condition['status']=='PASS',condition['condition_id']
            cid=condition['condition_id']
            assert cid not in passed,cid
            passed[cid]=condition
            for p,h in condition['input_hashes'].items():
                assert sha(p)==h,p
                protected[p]=h
        for p,h in receipt['code_hashes'].items():
            assert sha(p)==h,p
            protected[p]=h
        receipt_details.append(dict(path=path,sha256=expected,passed=len(receipt['condition_results'])))
    assert set(passed)==set(combined)
    unique_files={}
    for row in train+dev:
        assert sha(row['case_path'])==row['case_sha256']
        protected[row['case_path']]=row['case_sha256']
    for row in train:
        assert row['review_status'] in ('ACCEPT','ACCEPTED','PASS','ACTIVE_REVIEWED_SOURCE_CONDITION')
        for key in ('case_path','input_path','paired_path','supervision_file'):
            if row.get(key):
                p=resolve(row[key]);unique_files[str(p)]=p.stat().st_size
    m=dict(schema='BENCH_RELATION_TRAINING_DATA_V1',rows=train+dev,train=train,dev=dev,
        train_parents=sorted(tp),dev_parents=sorted(dp),
        source_ready=str(DATA/'DATASET_READY.json'),source_ready_sha256=sha(DATA/'DATASET_READY.json'),
        source_manifest=str(manifest_path),source_manifest_sha256=sha(manifest_path),
        protected_hashes=protected,actual_data_ready=True,full_pool_scale_goal_not_reached=True)
    schedule=D.make_schedule(train)
    cp=collections.Counter(r['parent_id'] for e in schedule['steps'] for r in e['samples'])
    cc=collections.Counter(r['condition_id'] for e in schedule['steps'] for r in e['samples'])
    ct=collections.Counter(r['part_type'] for e in schedule['steps'] for r in e['samples'])
    assert sum(cp.values())==12000 and set(cp)==tp and set(cc)==set(combined)
    assert max(cp.values())-min(cp.values())<=2
    for entry in schedule['steps']:
        assert len(entry['samples'])==8 and len({r['parent_id'] for r in entry['samples']})==4
        for a,b in zip(entry['samples'][:4],entry['samples'][4:]):
            assert a['condition_id']==b['condition_id'] and a['scale_xyz']==[1.,1.,1.]
    panel=read(BASELINE_ROOT/'eval_manifest.json')
    prior=read(ROOT/'experiments/r2_bench_user_v2_training_v1/eval_manifest.json')
    assert panel==prior and len(panel['requests'])==48
    assert not tp&{q['parent_id'] for q in panel['requests'] if q['group']=='DEV'}
    membership=[dict(request_id=q['request_id'],parent_id=q['parent_id'],historical_group=q['group'],
        in_current_training_parent_pool=q['parent_id'] in tp,
        exact_current_training_condition=q['condition_id'] in cc) for q in panel['requests']]
    inventory=dict(train_parents=26,train_conditions=91,retained_conditions=81,new_conditions=10,
        new_independent_parents=1,dev_parents=8,dev_conditions=39,
        N_range=[min(r['N'] for r in train),max(r['N'] for r in train)],
        K_range=[min(r['K'] for r in train),max(r['K'] for r in train)],
        condition_types=dict(collections.Counter(r['part_type'] for r in train)),
        unique_training_npz_files=len(unique_files),unique_training_npz_bytes=sum(unique_files.values()),
        byte_scope='Unique case/input/paired/supervision NPZ references; excludes shared source targets, renders and models',
        planned_instances=12000,parent_exposure_histogram=dict(collections.Counter(cp.values())),
        actual_pool_not_independent_condition_count=True)
    for p,h in protected.items():
        assert sha(p)==h,p
    for rel,obj in [('data/manifest.json',m),('data/schedule.json',schedule),
        ('data/exposure_plan.json',dict(parent=dict(cp),condition=dict(cc),part=dict(ct),
            minimum_parent_exposure=min(cp.values()),maximum_parent_exposure=max(cp.values()))),
        ('data/evaluation_membership.json',membership),('data/inventory.json',inventory)]:
        write_new(OUT/rel,obj)
    for rel in ('manifest.json','schedule.json','evaluation_membership.json','inventory.json'):
        p=OUT/'data'/rel;protected[str(p)]=sha(p)
    gate=dict(status='PASS',neural_forwards=0,new_OT_calls=0,reused_passed_OT_conditions=91,
        validation_receipts=receipt_details,protected=protected,
        data_counts=dict(train=91,train_parents=26,dev=39,dev_parents=8),
        actual_training_panel_requests=sum(r['in_current_training_parent_pool'] for r in membership),
        planned_parent_exposure=[min(cp.values()),max(cp.values())],
        training_adapter='Unchanged production SEAM/FM/OT + unchanged joint XYZ scaling; 81+10 exact CPU receipts reused')
    write_new(OUT/'cpu_gate.json',gate)
    return {k:v for k,v in gate.items() if k!='protected'}


if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False))
