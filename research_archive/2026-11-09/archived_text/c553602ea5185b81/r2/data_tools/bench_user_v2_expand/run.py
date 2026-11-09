"""CPU-only continuation of Bench v2 production; sealed first batch is read-only."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
from r2.data_tools.bench_user_v2 import build as b

ROOT=b.ROOT
OLD=ROOT/'experiments/r2_bench_user_data_v2'
OUT=ROOT/'experiments/r2_bench_user_data_v2_expansion_v1'
VERSION='BENCH_USER_TARGET_V2_EXPANSION_20261021'

def configure():
    b.OUT=OUT;b.VERSION=VERSION

def event(stage,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(unix=time.time(),stage=stage,**kw),ensure_ascii=False)+'\n')

def freeze():
    p=OUT/'PROTOCOL.json'
    if p.exists():
        a=b.read(p)
        for path,h in a['protected_source_hashes'].items():assert b.sha(path)==h,path
        return a
    old=b.read(OLD/'PROTOCOL.json');done=set(old['selected_parent_ids'])
    rows=[r for r in old['parent_queue'] if r['parent_id'] not in done]
    assert len(rows)==47 and len(done)==32
    protected=dict(old['protected_source_hashes'])
    for name in ('manifest.json','candidate_manifest.json','DATASET_READY.json','visual_review.json','PROTOCOL.json'):
        p0=OLD/name;protected[str(p0)]=b.sha(p0)
    a=dict(schema=VERSION,stage='DATA_ONLY',target_qualified_parent_range=[47,67],
        first_stage='Process all 47 remaining existing TRAIN parents under unchanged first-batch geometry rules; reuse sealed first32 receipts.',
        selected_parent_ids=[r['parent_id'] for r in rows],parent_queue=rows,
        geometry_rules=old['rules'],protected_source_hashes=protected,
        workers=4,numeric_threads=1,resources=b.resources(),
        model_calls=0,training_updates=0,DEV_policy='original39/8 unchanged',
        admission='Full target exact geometry PASS, meaningful C actually visually reviewed, production CPU packing PASS; never relax to reach target count.')
    b.save(p,a);event('FROZEN',parents=len(rows),resources=a['resources'])
    return a

def build():
    configure();a=freeze();start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,initializer=configure) as pool:
        new=list(pool.map(b.target_one,a['parent_queue']))
    old=b.read(OLD/'candidate_manifest.json')
    parents=old['parents']+new
    b.save(OUT/'parents_manifest.json',dict(schema=VERSION,parents=parents))
    passed=[r for r in new if not r['reasons']]
    with ProcessPoolExecutor(max_workers=4,initializer=configure) as pool:
        rr=list(pool.map(b.condition_parent,passed))
    rows=old['conditions']+[r for rs in rr for r in rs]
    m=dict(schema=VERSION,parents=parents,conditions=rows,train=[],dev=old['dev'],
        dev_source_manifest_path=old['dev_source_manifest_path'],dev_source_manifest_sha256=old['dev_source_manifest_sha256'],
        protected_source_hashes=a['protected_source_hashes'],status='CANDIDATES_REQUIRE_VISUAL_REVIEW',training_registered=False)
    b.save(OUT/'candidate_manifest.json',m)
    summary=dict(parents=len(parents),new_target_proposals=len(new),new_geometry_pass=len(passed),
        geometry_pass_total=sum(not r['reasons'] for r in parents),new_condition_proposals=sum(map(len,rr)),
        blocked_reasons=dict(Counter(reason for r in parents for reason in r['reasons'])),wall_seconds=time.monotonic()-start)
    b.save(OUT/'batch02_summary.json',summary);b.csvout(OUT/'parents.csv',parents)
    event('BATCH02_COMPLETE',**summary);print(json.dumps(summary))

def render():
    from r2.data_tools.bench_user_v2 import render as r
    r.GALLERY=OUT/'gallery'
    # Render only newly produced geometry-qualified parents and their proposals.
    m=b.read(OUT/'candidate_manifest.json');oldids={p['parent_id'] for p in b.read(OLD/'manifest.json')['parents']}
    m['parents']=[p for p in m['parents'] if p['parent_id'] not in oldids and not p['reasons']]
    ids={p['parent_id'] for p in m['parents']};m['conditions']=[x for x in m['conditions'] if x['parent_id'] in ids]
    p=OUT/'render_manifest.json';b.save(p,m)
    r.run(p,mode='all',threads=2,output_name='review_v1')

def render_supply():
    from r2.data_tools.bench_user_v2 import render as r
    r.GALLERY=OUT/'gallery'
    m=b.read(OUT/'expanded_candidate_manifest.json');oldids={p['parent_id'] for p in b.read(OUT/'candidate_manifest.json')['parents']}
    m['parents']=[p for p in m['parents'] if p['parent_id'] not in oldids and not p['reasons']]
    ids={p['parent_id'] for p in m['parents']};m['conditions']=[x for x in m['conditions'] if x['parent_id'] in ids]
    p=OUT/'render_supply_manifest.json';b.save(p,m)
    r.run(p,mode='all',threads=2,output_name='supply_review')

def render_standard():
    from r2.data_tools.bench_user_v2 import render as r
    r.GALLERY=OUT/'gallery'
    m=b.read(OUT/'standard_candidate_manifest.json')
    oldids={p['parent_id'] for p in b.read(OUT/'expanded_candidate_manifest.json')['parents']}
    m['parents']=[p for p in m['parents'] if p['parent_id'] not in oldids and not p['reasons']]
    ids={p['parent_id'] for p in m['parents']};m['conditions']=[x for x in m['conditions'] if x['parent_id'] in ids]
    p=OUT/'render_standard_manifest.json';b.save(p,m)
    r.run(p,mode='all',threads=2,output_name='standard_review')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['build','render','render_supply','render_standard']);a=p.parse_args()
    configure()
    if a.stage=='build':build()
    elif a.stage=='render':render()
    elif a.stage=='render_supply':render_supply()
    else:render_standard()

if __name__=='__main__':main()
