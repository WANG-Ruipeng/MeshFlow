"""Read-only curriculum preflight over existing frozen metadata; no scientific imports."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import ast, csv, hashlib, json, time

RUN=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1')
REPO=Path('/mnt/e/MeshFlow-Control')
OUT=RUN/'audit/curriculum_preflight'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def dump_exclusive(path,value):
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,sort_keys=True)
def canon(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    tick=time.perf_counter()
    frozen=load(RUN/'registrations/source_hashes.json')
    before={p:sha(REPO/p) for p in frozen}
    assert before==frozen
    plan_path=RUN/'streams/recipe_plan.json'; plan=load(plan_path);plan_sha=sha(plan_path)
    records=plan['mixed_records']; size=len(records)
    assert size==48000 and plan['total_updates']==6000 and plan['batch_size']==8 and plan['microbatch_size']==1
    assert len(plan['eligible_parents'])==26 and plan['eligible20_task_count']==208 and plan['eligible40_task_count']==203
    assert [r['sample_index'] for r in records]==list(range(size))
    data_path=Path('/mnt/e/MeshFlow-Control-artifacts/data/train_manifest.json')
    assert sha(data_path)==plan['data_manifest_sha256']
    tasks={t['task_id']:t for t in load(data_path)['tasks']}
    for r in records:
        task=tasks[r['task_id']]
        assert all(r[k]==task[k] for k in ('uid','N','K','ratio'))
        assert r['origin_sample_index']==r['sample_index']
    pair_path=RUN/'streams/paired_inputs/R2_MIX_H_ALL.json'
    r2=load(pair_path)
    assert r2['plan_sha256']==plan_sha and r2['data_manifest_sha256']==plan['data_manifest_sha256']
    pairs=r2['records']
    assert len(pairs)==size and [p['sample_index'] for p in pairs]==list(range(size))
    for r,p in zip(records,pairs):
        assert p['origin_sample_index']==r['sample_index'] and p['task_id']==r['task_id']
        assert p['lambda_value']==.25 and p['free_coordinate_count']==9*(r['N']-r['K'])
    # Execute only exact side-effect-free metadata validators extracted from the frozen trainer AST.
    # No imports, tensor constructors, stream.sample, model, OT or RNG functions execute.
    trainer_path=REPO/'src/meshflow_control/training/recipe_factorial.py'
    tree=ast.parse(trainer_path.read_text())
    names={'resolved_schedule','lambda_at_step','validate_samples'}
    picked=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in names)
            or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'RECIPES','TOTAL_UPDATES'} for t in n.targets))]
    isolated=ast.Module(body=picked,type_ignores=[])
    scope={}
    exec(compile(isolated,str(trainer_path),'exec'),scope)
    source_line_evidence={n.name:{'start':n.lineno,'end':n.end_lineno} for n in picked if isinstance(n,ast.FunctionDef)}
    stream_tree=ast.parse((REPO/'src/meshflow_control/data/recipe_factorial.py').read_text())
    source_line_evidence['data']={n.name:{'start':n.lineno,'end':n.end_lineno} for n in stream_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {'make_plan','RecipeFactorialStream','precompute','lambda_for_step','set_coupling'}}
    all_results={};batch_csv=[];segment_csv=[]
    expected_origins=list(range(size));baseline_full_counts=Counter(canon(r) for r in records)
    baseline_pair_counts=Counter(canon(p) for p in pairs)
    identity_fields=('uid','task_id','N','K','ratio','alpha','gaussian_seed','epsilon_seed','permutation_seed','t_seed','sample_index','origin_sample_index')
    for recipe in ('R2_MIX_H_ALL','C20_40_MIX','C40_20_MIX'):
        order=expected_origins if recipe=='R2_MIX_H_ALL' else plan['curriculum_orders'][recipe]
        path=pair_path if recipe=='R2_MIX_H_ALL' else RUN/'streams/paired_inputs'/(recipe+'__H_ALL.json')
        saved=load(path)
        assert saved['plan_sha256']==plan_sha and saved['data_manifest_sha256']==plan['data_manifest_sha256']
        course_pairs=saved['records']
        assert len(order)==size and sorted(order)==expected_origins and len(course_pairs)==size
        assert course_pairs==[pairs[i] for i in order] # All fields, including actual x1/target-containing shared hash.
        assert Counter(canon(p) for p in course_pairs)==baseline_pair_counts
        reordered=[records[i] for i in order]
        assert Counter(canon(r) for r in reordered)==baseline_full_counts
        assert order[32000:]==expected_origins[32000:]
        assert course_pairs[32000:]==pairs[32000:]
        assert reordered[32000:]==records[32000:]
        for pos,r in enumerate(reordered):
            assert all(r[k]==records[order[pos]][k] for k in identity_fields)
        denoms=[];boundary_validation=[]
        for start in range(0,size,8):
            step=start//8+1;batch=reordered[start:start+8]
            mock=[dict(object_id=r['uid'],alpha=r['alpha'],step=step,lambda_value=.25,y=r['N'],K=r['K'],ratio=r['ratio'],free_coordinate_count=9*(r['N']-r['K'])) for r in batch]
            assert scope['validate_samples'](mock,recipe,step,'all')==.25
            denominator=sum(s['free_coordinate_count'] for s in mock);denoms.append(denominator)
            if step in (1,2000,2001,4000,4001,6000):boundary_validation.append({'step':step,'ratio_counts':dict(Counter(str(s['ratio']) for s in mock)),'free_coordinates':denominator})
            batch_csv.append({'recipe':recipe,'optimizer_step':step,'origin_sample_indices':';'.join(str(r['sample_index']) for r in batch),'ratio20_samples':sum(r['ratio']==.2 for r in batch),'ratio40_samples':sum(r['ratio']==.4 for r in batch),'clean_samples':sum(r['alpha']==1 for r in batch),'distinct_parents':len({r['uid'] for r in batch}),'free_coordinates':denominator,'lambda_value':.25})
        assert sum(denoms)==plan['effective_free_coordinates_MIX']==sum(p['free_coordinate_count'] for p in course_pairs)
        for a,b in ((0,16000),(16000,32000),(32000,48000)):
            sub=reordered[a:b]
            segment_csv.append({'recipe':recipe,'first_step':a//8+1,'last_step':b//8,'samples':len(sub),'ratio20_samples':sum(r['ratio']==.2 for r in sub),'ratio40_samples':sum(r['ratio']==.4 for r in sub),'clean_samples':sum(r['alpha']==1 for r in sub),'augmented_samples':sum(r['alpha']!=1 for r in sub),'distinct_parents':len({r['uid'] for r in sub}),'free_coordinates':sum(9*(r['N']-r['K']) for r in sub)})
        all_results[recipe]={'status':'PASS','paired_path':str(path),'paired_sha256':sha(path),'samples':size,'updates_metadata_validated':6000,'exact_registered_pair_row_multiset_equal_R2':True,'exact_all_raw_record_multiset_equal_R2':True,'random_identity_fields':list(identity_fields),'all48k_pair_rows_exact_R2_at_origin':True,'unchanged_last2000_updates_last16000_samples':True,'H_ALL_lambda_all_updates':.25,'total_free_coordinates':sum(denoms),'batch_free_coordinates_min':min(denoms),'batch_free_coordinates_max':max(denoms),'boundary_metadata_checks':boundary_validation,'target_hash_evidence':'Exact shared_input_sha256 includes x1, pre-OT target, OT face/corner maps and source IDs; label_bytehash additionally verifies U. Existing actual-precompute receipts are compared, not freshly reassembled.'}
    # Tiny negative metadata checks prove validator really gates course boundaries and H schedule.
    for recipe in ('C20_40_MIX','C40_20_MIX'):
        assert scope['resolved_schedule'](recipe,'all')=='all'
        try:scope['resolved_schedule'](recipe)
        except ValueError:pass
        else:raise AssertionError('Course implicitly chose HYBRID schedule')
        batch=[records[i] for i in plan['curriculum_orders'][recipe][:8]]
        mock=[dict(object_id=r['uid'],alpha=r['alpha'],step=2001,lambda_value=.25,y=r['N'],K=r['K'],ratio=r['ratio'],free_coordinate_count=9*(r['N']-r['K'])) for r in batch]
        try:scope['validate_samples'](mock,recipe,2001,'all')
        except ValueError:pass
        else:raise AssertionError('Wrong ratio at course switch was accepted')
    after={p:sha(REPO/p) for p in frozen};assert after==frozen==before
    critical_paths=[plan_path,pair_path,RUN/'streams/paired_inputs/C20_40_MIX__H_ALL.json',RUN/'streams/paired_inputs/C40_20_MIX__H_ALL.json',RUN/'streams/precompute_receipt.json',data_path,Path(__file__)]
    source_hashes={str(p):sha(p) for p in critical_paths}
    for relative in ('src/meshflow_control/data/recipe_factorial.py','src/meshflow_control/data/stream.py','src/meshflow_control/training/recipe_factorial.py','tools/train_recipe_factorial.py'):
        source_hashes[str(REPO/relative)]=sha(REPO/relative)
    receipt={'schema':'recipe_curriculum_preflight_v1','status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Read-only existing 48k MIX multiset / two H_ALL curriculum views; no dispatch authorization','plan_path':str(plan_path),'plan_sha256':plan_sha,'namespace':plan['namespace'],'recipes':all_results,'frozen_sources_verified':len(frozen),'frozen_sources_before_after_equal':True,'trainer_support':'Exact frozen metadata validators pass all 6000 batches of R2 and both courses; CLI accepts both recipe labels and explicit all. Real run remains gated by a new explicit authorization and owned output directory. No model/update execution performed.','source_line_evidence':source_line_evidence,'loss_limitation':'Same sample multiset and total free scalars do not imply identical optimization weighting: regrouping pure-ratio batches changes per-update free-scalar denominator and sequence of gradients. Final 2000 batches and their denominators are identical to R2. No claim of pure scalar chronology isolation.','condition_pool_limitation':'Unchanged inherited task pools retain the previously documented MVP_LEGACY20 vs DOMAIN_NEW20/40 selector-policy heterogeneity; no pure ratio causal claim.','new_OT':0,'sample_array_reassembly':0,'new_noise':0,'model_calls':0,'optimizer_updates':0,'GPU_dispatches':0,'authorizations_created':0,'scientific_source_edits':0,'wall_seconds':time.perf_counter()-tick}
    for name,rows in (('per_batch_metadata.csv',batch_csv),('segment_exposure.csv',segment_csv)):
        with (OUT/name).open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump_exclusive(OUT/'preflight.json',receipt)
    dump_exclusive(OUT/'source_hashes.json',source_hashes)
    dump_exclusive(OUT/'frozen_source_verification.json',{'status':'PASS','count':len(frozen),'before':before,'after':after})
    text="""# Existing curriculum preflight

Status: PASS, CPU metadata only. No authorization, model, GPU, new OT, new noise or sample-array reconstruction was performed.

Both C20_40_MIX and C40_20_MIX under explicit H_ALL contain exactly the existing R2 48,000-record multiset. Every existing paired row, including shared input, input/label hashes, OT key, free-coordinate count and origin index, is exactly equal to the corresponding R2 origin. Each curriculum ends with R2's identical ordered last 16,000 records and paired rows (updates 4001–6000).

The first 4,000 R2 mixed batches are split in adjacent pairs into pure-ratio batches. C20_40_MIX uses 2,000 pure20 then 2,000 pure40 updates; C40_20_MIX reverses these blocks. Every resulting batch has eight distinct parents and four clean/four augmented samples. The frozen trainer's exact metadata validators were extracted without imports and executed for every batch. This is not a model or training test.

The complete sample-level free supervision total is equal. However, the original loss normalizes by each effective batch's total free scalar count. Regrouping pure-ratio batches changes that denominator and hence sample weighting within the optimizer trajectory. This experiment compares the registered curriculum recipe, not an abstract ordering intervention guaranteed to preserve every scalar's effective optimization weight. The identical final MIX suffix does preserve its batch denominators.

All 97 frozen source hashes still match. No old plan, paired table, source or metric result was changed. The original heterogeneous candidate-selection pools remain unchanged and must still be disclosed. The existing CLI needs an explicit all schedule and a separately authorized output/recipe registration before any execution.

Evidence: preflight.json, segment_exposure.csv, per_batch_metadata.csv, source_hashes.json and frozen_source_verification.json. Target equality here is evidence from the existing actual-precompute hashes (shared hash contains x1 and OT maps), not a new numerical reconstruction of cached arrays.
"""
    with (OUT/'PREFLIGHT.md').open('x',encoding='utf-8') as f:f.write(text)
    print(json.dumps({'status':'PASS','frozen_sources':len(frozen),'samples_per_recipe':size,'trainer_metadata_batches_checked':len(batch_csv),'total_free_coordinates':plan['effective_free_coordinates_MIX'],'output':str(OUT),'wall_seconds':receipt['wall_seconds'],'new_OT':0,'model_calls':0,'GPU_dispatches':0,'authorizations_created':0}))
if __name__=='__main__':main()
