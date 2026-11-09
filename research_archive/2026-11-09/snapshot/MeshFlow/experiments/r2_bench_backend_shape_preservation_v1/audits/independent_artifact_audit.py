"""Independent receipt, accounting and arithmetic audit. No ML/scorer imports."""
import collections,csv,hashlib,json,math,statistics,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def table(name):return list(csv.DictReader((OUT/'scores'/name).open(encoding='utf-8-sig',newline='')))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def number(v):
    if v is None or isinstance(v,bool) or v in ('','True','False'):return None
    try:x=float(v)
    except (TypeError,ValueError):return None
    return x if math.isfinite(x) else None
def same(a,b):
    a,b=number(a),number(b)
    return a is None and b is None or a is not None and b is not None and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)

tick=time.perf_counter();errors=[];checks=[];expected={};receipts=[]
def check(ok,name,detail=None):
    checks.append(dict(name=name,passed=bool(ok),detail=detail))
    if not ok:errors.append(name)
def addreceipt(path):
    r=read(path);receipts.append(str(path))
    for key in ('files','source_hashes','reference_hashes','protected'):
        for p,h in r.get(key,{}).items():
            if p in expected and expected[p]!=h:errors.append('conflicting expected digest: '+p)
            expected[p]=h
    return r

train=addreceipt(OUT/'training/complete.json')
scores=addreceipt(OUT/'scores/COMPLETE.json')
addreceipt(OUT/'evaluation/CPU_registration.json')
addreceipt(OUT/'gallery/COMPLETE.json')
arm_counts={}
for arm in ('CANDIDATE_RAW','OFFICIAL_SOURCE','CANDIDATE_SOURCE'):
    armroot=OUT/'evaluation'/arm;r=addreceipt(armroot/'complete.json')
    rr=[addreceipt(p) for p in sorted(armroot.glob('*/receipt.json'))]
    arm_counts[arm]=dict(outputs=len(rr),D0_forwards=sum(x['D0_forwards'] for x in rr),generator_forwards=sum(x['generator_forwards'] for x in rr),new_noise_draws=sum(x['new_noise_draws'] for x in rr))
    check(r['status']=='COMPLETE' and len(rr)==30 and arm_counts[arm]['D0_forwards']==30,'eval_'+arm,arm_counts[arm])

ledger=[json.loads(x) for x in (OUT/'training/ledger.jsonl').read_text().splitlines()]
losses=[json.loads(x) for x in (OUT/'training/losses.jsonl').read_text().splitlines()]
event_counts=collections.Counter(r['event'] for r in ledger)
for event,n in [('FORWARD_STARTED',2000),('FORWARD_RETURNED',2000),('BACKWARD_STARTED',2000),('BACKWARD_RETURNED',2000),('OPTIMIZER_STEP_STARTED',500),('OPTIMIZER_STEP_RETURNED',500)]:
    rs=[r for r in ledger if r['event']==event]
    identities=[(r['update'],r.get('micro',None)) for r in rs]
    check(len(rs)==n and len(set(identities))==n,'unique_'+event,dict(actual=len(rs),expected=n))
check([r['update'] for r in losses]==list(range(1,501)),'loss_rows_exact_1_to_500')
samples=[s for r in losses for s in r['samples']]
check(len(samples)==2000 and [s['row_index'] for s in samples]==list(range(2000)),'sampling_stream_exact_0_to_1999')
check(all(len(r['samples'])==4 and r['cumulative_forwards']==r['update']*4 and r['cumulative_backwards']==r['update']*4 for r in losses),'loss_counter_consistency')
amend=read(OUT/'engineering_amendment.json');restored=read(OUT/'training/resume_boundary_loaded.json')
check((OUT/'training/failure.json').exists() and restored['restored_step']==100 and restored['next_update']==101 and restored['replayed_updates']==0,'recorded_engineering_failure_zero_replay')
check(all(restored[k] for k in ('python_rng_exact','numpy_rng_exact','cpu_rng_exact','cuda_rng_exact','all_optimizer_steps_100')),'recorded_exact_boundary_restore')
prefix=b''.join((OUT/'training/losses.jsonl').read_bytes().splitlines(keepends=True)[:100])
check(hashlib.sha256(prefix).hexdigest()==train['boundary_recovery']['original_losses_sha256'],'first100_loss_records_unchanged')

rows=table('per_output.csv');new=table('new_per_output.csv');refs=table('original_reference_scores.csv')
fields=scores['metric_fields'];by={(r['label'],r['stage'],r['request_id']):r for r in rows}
check(len(rows)==270 and len(by)==270 and len(new)==60 and len(refs)==210 and len(fields)==177,'table_coverage',dict(total=len(rows),new=len(new),original=len(refs),fields=len(fields)))
check(all(r['C_preserved_exact']=='True' and r['seam_applicability']=='NA_NOT_SEAM_REPRESENTATION' for r in new),'candidate_C_preservation_and_native_applicability')
reference_cells=0
for r in refs:
    source_path=str(Path(r['source_score_path']).parent/'row.json') if r.get('source_score_path') else str(OUT.parent/'r2_bench_geo_only_acceptance_v1'/'scores'/r['request_id']/'row.json')
    source=read(source_path)
    for field in fields:
        if field not in source:
            errors.append('missing original field '+source_path+' '+field);continue
        reference_cells+=1
        if not same(r.get(field),source.get(field)):errors.append('changed original metric '+r['request_id']+' '+field)

def groups_mean(data,keys):
    groups=collections.defaultdict(list)
    for row in data:groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for key,group in groups.items():
        row=dict(zip(keys,key),rows=len(group))
        for field in fields:
            vals=[number(x.get(field)) for x in group];vals=[x for x in vals if x is not None]
            row[field]=statistics.mean(vals) if vals else None;row[field+'__valid_n']=len(vals)
        result.append(row)
    return result

parent_keys=['label','stage','eval_group','parent_id'];condition_keys=parent_keys+['condition_id'];macro_keys=parent_keys[:-1]
cond=groups_mean(rows,condition_keys);parents=groups_mean(cond,parent_keys);macro=groups_mean(parents,macro_keys)
aggregate_cells=0
for filename,recomputed,keys in [('per_condition.csv',cond,condition_keys),('per_parent.csv',parents,parent_keys),('parent_macro_summary.csv',macro,macro_keys)]:
    saved={tuple(r[k] for k in keys):r for r in table(filename)}
    check(len(saved)==len(recomputed),'aggregate_rows_'+filename,len(saved))
    for r in recomputed:
        out=saved[tuple(r[k] for k in keys)]
        for field in ['rows']+[q for f in fields for q in (f,f+'__valid_n')]:
            aggregate_cells+=1
            if not same(r[field],out[field]):errors.append('aggregate mismatch '+filename+' '+str(tuple(r[k] for k in keys))+' '+field)

pairs=table('paired_per_output.csv');pair_cells=0
check(len(pairs)==300,'pair_rows',len(pairs))
for p in pairs:
    a=by[(p['reference_label'],p['reference_stage'],p['request_id'])];b=by[(p['candidate_label'],p['candidate_stage'],p['request_id'])]
    for field in fields:
        av,bv=number(a.get(field)),number(b.get(field));dv=bv-av if av is not None and bv is not None else None
        for suffix,value in [('reference',av),('candidate',bv),('delta',dv)]:
            pair_cells+=1
            if not same(value,p[field+'__'+suffix]):errors.append('pair mismatch '+p['request_id']+' '+field+'__'+suffix)

for i,(p,h) in enumerate(sorted(expected.items())):
    if not Path(p).is_file():errors.append('missing hash asset '+p);continue
    actual=digest(p)
    if actual!=h:errors.append('hash mismatch '+p)
    if i%250==0:print('HASH_AUDIT',i,len(expected),flush=True)

selected=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction','actual_free_faces','deleted_free_faces','C_F_valid_edge_count','reached_free_area_fraction','hinge_diag_isolated_1face_edge_components']
comparisons=[]
mb={tuple(r[k] for k in macro_keys):r for r in macro}
for group in sorted({r['eval_group'] for r in rows}):
    a=mb[('COVERAGE_FINAL','CLEAN_FINAL_POST',group)];b=mb[('SHAPE_D0','CLEAN_FINAL_POST',group)]
    comparisons.append(dict(eval_group=group,parents=b['rows'],metrics={f:dict(official=a[f],candidate=b[f],delta=b[f]-a[f] if a[f] is not None and b[f] is not None else None) for f in selected}))
result=dict(status='PASS' if not errors else 'FAIL',audit_kind='Independent stdlib receipt/hash/accounting/arithmetic; no model, geometry or scorer execution',
    audit_harness_attempts=3,audit_harness_note='Audit-only attempt1 stopped at UTF-8 BOM header parsing; attempt2 stopped at GEO_ONLY rows having no source_score_path. Reader corrected to utf-8-sig and original GEO_ONLY per-request receipt. Both stopped before hash scan, with no model/scorer calls.',
    actual_engineering_failures=1,zero_replayed_updates=0,training_event_counts=dict(event_counts),eval_counts=arm_counts,
    loss_updates=500,training_instances=2000,sample_kinds=dict(collections.Counter(s['kind'] for s in samples)),
    reference_numeric_cells_checked=reference_cells,aggregate_numeric_and_denominator_cells_checked=aggregate_cells,
    pair_cells_checked=pair_cells,hash_files_checked=len(expected),hash_receipts_checked=len(receipts),
    seconds=time.perf_counter()-tick,checks=checks,errors=errors,selected_parent_equal_comparisons=comparisons,
    interpretation_boundary='Arithmetic/hash validity does not certify visual quality, manifoldness or broad generalization; source denoising is synthetic calibration, not real RAW repair supervision.')
dest=OUT/'audits/independent_artifact_audit_recheck.json'
with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps(dict({k:v for k,v in result.items() if k not in ('checks','selected_parent_equal_comparisons','errors')},error_count=len(errors),first_errors=errors[:5]),ensure_ascii=False),flush=True)
if errors:raise SystemExit(1)
