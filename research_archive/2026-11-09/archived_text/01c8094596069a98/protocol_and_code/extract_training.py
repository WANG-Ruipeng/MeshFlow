"""Read-only, stdlib projection of completed training evidence into MINI tables.

No torch/numpy imports; no arrays/weights opened; no scoring or model calls.
JSON decimals are parsed as Decimal and emitted without rounding.
"""
from pathlib import Path
from collections import Counter, defaultdict
from decimal import Decimal
import csv
import datetime
import hashlib
import io
import json

OLD = Path('/mnt/e/MeshFlow/experiments/r2_user_data_scaling_32_67_v1')
DEST = Path(__file__).resolve().parents[1]
ARMS = ('USER_SMALL', 'USER_EXPANDED_67')
NA = 'NA'
MISSING = []
SOURCES = {}
CACHE = {}
EXISTING = {}


def missing(item, reason, count=None):
    MISSING.append(dict(item=item, status='NOT_RECORDED' if reason.startswith('Not recorded') else 'MISSING',
                        reason=reason, affected_records=count))


def source(path, existing=None, existing_reference=None):
    path = Path(path)
    if not path.is_relative_to(OLD) or any(q.is_symlink() for q in (path, *path.parents)):
        raise ValueError('Read-only original experiment paths only')
    name = str(path)
    if name not in SOURCES:
        if not path.exists():
            missing(name, 'Source file absent'); return None, None
        blob = path.read_bytes()
        actual = hashlib.sha256(blob).hexdigest()
        recorded = existing or EXISTING.get(name)
        if recorded and recorded != actual:
            raise ValueError('Existing source hash differs: '+name)
        sid = 'T'+str(len(SOURCES)+1).zfill(3)
        SOURCES[name] = dict(source_id=sid, path=name, existing_sha256=recorded or NA,
            existing_hash_reference=existing_reference or ('training_analysis/final/summary.json:input_receipt_hashes' if name in EXISTING else NA),
            existing_hash_missing_reason=NA if recorded else 'No prior content hash found in consulted small receipts',
            current_reread_sha256=actual, bytes=len(blob), current_reread=True)
        CACHE[name] = blob
    return SOURCES[name]['source_id'], CACHE[name]


def obj(path, existing=None, existing_reference=None):
    sid, blob = source(path, existing, existing_reference)
    return sid, json.loads(blob, parse_float=Decimal) if blob is not None else None


def records(path):
    sid, blob = source(path)
    if blob is None:
        return
    for line_number, line in enumerate(blob.decode('utf-8').splitlines(), 1):
        if line.strip():
            yield sid, line_number, json.loads(line, parse_float=Decimal)


def scalar(value):
    if value is None: return 'NULL'
    if value is True: return 'true'
    if value is False: return 'false'
    if isinstance(value, (str, int, Decimal)): return str(value)
    raise TypeError(type(value).__name__)


def csv_output(name, rows, fieldnames):
    target = DEST/name
    if target.exists(): raise FileExistsError(target)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k:scalar(row.get(k, NA)) for k in fieldnames})
    data = buffer.getvalue().encode('utf-8')
    with target.open('xb') as f:f.write(data)
    return dict(file=name, sha256=hashlib.sha256(data).hexdigest(), rows=len(rows), bytes=len(data), fields=fieldnames)


def main():
    if DEST.name != 'R2_32_67_ANALYSIS_MINI_V1' or any(q.is_symlink() for q in (DEST,*DEST.parents)):
        raise ValueError('Dedicated MINI destination required')
    (DEST/'_work').mkdir(parents=True,exist_ok=True)
    _, summary = obj(OLD/'training_analysis/final/summary.json')
    EXISTING.update(summary['input_receipt_hashes'])
    manifest_sid, manifest = obj(OLD/'data/train_manifest.json')
    _, plan = obj(OLD/'diagnostics/fixtures.json')
    # The fixture plan already binds this manifest; no array needs to be read.
    manifest_sha = SOURCES[str(OLD/'data/train_manifest.json')]['current_reread_sha256']
    if plan['manifest_sha256'] != manifest_sha: raise ValueError('Fixture manifest binding differs')
    SOURCES[str(OLD/'data/train_manifest.json')].update(existing_sha256=plan['manifest_sha256'],
        existing_hash_reference='diagnostics/fixtures.json:manifest_sha256',existing_hash_missing_reason=NA)
    small, expanded = map(set, (manifest['pools'][a] for a in ARMS))
    if not small < expanded or (len(small),len(expanded))!=(32,67): raise ValueError('Nested pools differ')
    tasks=manifest['tasks']; steps=[]; step_sources=[]; exposure_sources=[]; seen_steps={}
    buckets={}; exposure_sample_ids={a:set() for a in ARMS}; actual_parent=Counter(); actual_condition=Counter()
    for arm in ARMS:
        allowed=set(manifest['pools'][arm])
        for task in tasks.values():
            if task['parent_id'] in allowed:
                pk=(arm,task['parent_id'],task['condition_id'])
                buckets[pk]=dict(arm=arm,parent_uid=pk[1],condition_id=pk[2],condition_type=task['condition_type'],
                    parent_role='COMMON_SMALL32' if pk[1] in small else 'EXP_ONLY35',N=task['N'],K=task['K'],
                    actual_exposures=0,clean_exposures=0,width_exposures=0,cumulative_free_scalars=0,
                    clean_free_scalars=0,width_free_scalars=0,t_min=None,t_max=None,alpha_min=None,alpha_max=None,
                    t_sum=Decimal(0),alpha_sum=Decimal(0),source_ids=set())
        for attempt in sorted((OLD/'training'/arm/'attempts').iterdir()):
            final_sid, final = obj(attempt/'final_status.json')
            if final is None: continue
            if final['status'] not in ('SEGMENT_COMPLETE','COMPLETE'): raise ValueError('Open/noncomplete attempt')
            update_den={}; event_den=Counter(); per_step=defaultdict(list)
            for sid, line, row in records(attempt/'training_log.jsonl'):
                step=row['step']; identity=(arm,step)
                if identity in seen_steps: raise ValueError('Duplicate update')
                seen_steps[identity]=True;update_den[step]=row['free_coordinate_denominator']
                projected=dict(arm=arm,attempt=attempt.name,source_id=sid,source_record_key='line:'+str(line))
                projected.update({k:v for k,v in row.items() if not isinstance(v,(dict,list))})
                losses=row.get('sample_losses')
                if losses is None:
                    missing(f'{arm}:{step}:sample_losses','Not recorded per update',1)
                elif len(losses)!=8:
                    raise ValueError('Unexpected recorded microbatch contribution count')
                for i in range(8):
                    projected[f'sample_{i}_FM_contribution']=losses[i]['FM_contribution'] if losses is not None else NA
                den=row['free_coordinate_denominator']
                if den%9:raise ValueError('Free-coordinate denominator indivisible by 9')
                projected.update(valid_free_faces_DERIVED=den//9,FM_numerator_recorded=NA,clip_applied_recorded=NA,
                                 lr_recorded=NA,finite_recorded=NA)
                steps.append(projected)
                if sid not in step_sources:step_sources.append(sid)
            for sid,line,row in records(attempt/'exposure.jsonl'):
                sample=row['sample_index']
                if sample in exposure_sample_ids[arm]:raise ValueError('Duplicate actual exposure index')
                exposure_sample_ids[arm].add(sample)
                if row['step'] not in update_den:raise ValueError('Exposure lacks completed update')
                pk=(arm,row['uid'],row['task_id'])
                if pk not in buckets:raise ValueError('Exposure outside registered arm conditions')
                b=buckets[pk]
                if (row['N'],row['K'])!=(b['N'],b['K']) or row['free_coordinate_count']!=9*(row['N']-row['K']):
                    raise ValueError('Exposure geometry/denominator identity differs')
                clean=row['alpha']==1
                b['actual_exposures']+=1;b['clean_exposures' if clean else 'width_exposures']+=1
                b['cumulative_free_scalars']+=row['free_coordinate_count']
                b['clean_free_scalars' if clean else 'width_free_scalars']+=row['free_coordinate_count']
                b['t_sum']+=row['t'];b['alpha_sum']+=row['alpha'];b['source_ids'].add(sid)
                for prefix in ('t','alpha'):
                    value=row[prefix]
                    b[prefix+'_min']=value if b[prefix+'_min'] is None else min(b[prefix+'_min'],value)
                    b[prefix+'_max']=value if b[prefix+'_max'] is None else max(b[prefix+'_max'],value)
                actual_parent[(arm,row['uid'])]+=1;actual_condition[pk]+=1
                event_den[row['step']]+=row['free_coordinate_count'];per_step[row['step']].append(row)
                if sid not in exposure_sources:exposure_sources.append(sid)
            for step,den in update_den.items():
                events=per_step[step]
                if len(events)!=8 or len({e['uid'] for e in events})!=8 or sum(e['alpha']==1 for e in events)!=4 or event_den[step]!=den:
                    raise ValueError('Recorded effective batch exposure/denominator differs')
    exposures=[]
    for pk,b in sorted(buckets.items()):
        b=dict(b);count=b['actual_exposures'];b['source_ids']='|'.join(sorted(b['source_ids']))
        b['t_mean_DERIVED']=b.pop('t_sum')/count if count else NA
        b['alpha_mean_DERIVED']=b.pop('alpha_sum')/count if count else NA
        b['derivation']='DERIVED_FROM_ACTUAL_COMPLETED_EXPOSURE_EVENTS'
        for k,v in list(b.items()):
            if v is None:b[k]=NA
        exposures.append(b)
    fixture_rows=[];fixture_sources=[];plan_sha=SOURCES[str(OLD/'diagnostics/fixtures.json')]['current_reread_sha256']
    for rec in plan['records']:
        i=rec['sample_index'];sid,meta=obj(OLD/'diagnostics'/f'fixture_{i:02d}.json')
        if meta is None:continue
        if meta['record']!=rec or meta['plan_sha256']!=plan_sha:raise ValueError('Fixture receipt differs from frozen plan')
        m=meta['metadata'];uid=rec['uid'];condition=rec['task_id']
        group={'SMALL_SEEN_EXPANDED_SEEN':'COMMON8','SMALL_UNSEEN_EXPANDED_SEEN':'EXP_ONLY8'}[rec['group']]
        row=dict(fixture=i,group=group,original_group=rec['group'],parent_uid=uid,condition_id=condition,
                 condition_type=rec['condition_type'],region_type=rec['region_type'],N=rec['N'],K=rec['K'],
                 t=rec['t'],alpha=rec['alpha'],clean=rec['clean'],K_over_N=rec['ratio'],
                 free_scalar_count=m['free_coordinate_count'],hybrid_lambda=m['lambda_value'],
                 t_seed=rec['t_seed'],Gaussian_seed=rec['gaussian_seed'],epsilon_seed=rec['epsilon_seed'],
                 permutation_seed=rec['permutation_seed'],input_bytehash=m['input_bytehash'],
                 label_bytehash=m['label_bytehash'],coupled_input_sha256=m['coupled_input_sha256'],
                 shared_input_sha256=m['shared_input_sha256'],ot_cache_key=m['ot_cache_key'],
                 fixture_npz_path=meta['path'],fixture_npz_existing_sha256=meta['sha256'],
                 fixture_npz_hash_verification='EXISTING_RECEIPT_ONLY_NOT_REREAD',source_id=sid,
                 source_record_key='record+metadata',fixture_plan_sha256=plan_sha)
        for arm,short in zip(ARMS,['small','expanded']):
            row[short+'_parent_exposures_at6000_DERIVED']=actual_parent[(arm,uid)]
            row[short+'_condition_exposures_at6000_DERIVED']=actual_condition[(arm,uid,condition)]
            row[short+'_parent_seen_at6000_DERIVED']=actual_parent[(arm,uid)]>0
            row[short+'_condition_seen_at6000_DERIVED']=actual_condition[(arm,uid,condition)]>0
        fixture_rows.append(row);fixture_sources.append(sid)
    diagnostics=[];diagnostic_sources=[]
    for arm in ARMS:
        for step in (0,1000,3000,6000):
            sid,report=obj(OLD/'diagnostics'/arm/f'step_{step:04d}.json')
            if report is None:continue
            identity=report['identity']
            if identity['arm']!=arm or identity['step']!=step or identity['fixtures_sha256']!=plan_sha:
                raise ValueError('Diagnostic identity differs')
            for row in report['rows']:
                rec=plan['records'][row['fixture']]
                if (row['uid'],row['condition_id'],row['N'],row['K'])!=(rec['uid'],rec['task_id'],rec['N'],rec['K']):
                    raise ValueError('Diagnostic fixture identity differs')
                diagnostics.append(dict(arm=arm,step=step,fixture=row['fixture'],parent_uid=row['uid'],
                    condition_id=row['condition_id'],original_group=row['group'],
                    group='COMMON8' if row['group']=='SMALL_SEEN_EXPANDED_SEEN' else 'EXP_ONLY8',
                    N=row['N'],K=row['K'],free_scalar_count=row['free_scalar_count'],FM=row['FM'],
                    diagnostic_status=report['status'],generator_state_existing_sha256=identity['generator_state_sha256'],
                    fixture_plan_sha256=identity['fixtures_sha256'],source_id=sid,source_record_key='rows.fixture:'+str(row['fixture'])))
            diagnostic_sources.append(sid)
    for name,actual,expected in [('updates',len(steps),12000),('exposure groups',len(exposures),285),
                                  ('fixtures',len(fixture_rows),16),('fixed diagnostics',len(diagnostics),128)]:
        if actual!=expected:missing(name,f'Expected {expected}; actual preserved {actual}',actual)
    for field in ('FM numerator','clip applied flag','learning rate','finite flag'):
        missing('training_steps:'+field,'Not recorded as a standalone per-update scalar; not inferred from code, norm, or rounded FM',len(steps))
    if any(sorted(s for a,s in seen_steps if a==arm)!=list(range(1,6001)) for arm in ARMS):
        missing('step coverage','Nonconsecutive actual update coverage; no fabricated rows')
    steps.sort(key=lambda r:(ARMS.index(r['arm']),r['step']))
    outputs=[]
    for filename,rows in [('exposure_by_condition.csv',exposures),('training_steps.csv',steps),
                          ('fixed_fixtures.csv',fixture_rows),('fixed_FM_diagnostics.csv',diagnostics)]:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        out=csv_output(filename,rows,fields)
        out.update(source_ids={'exposure_by_condition.csv':exposure_sources,'training_steps.csv':step_sources,
                               'fixed_fixtures.csv':fixture_sources,'fixed_FM_diagnostics.csv':diagnostic_sources}[filename],
                   precision='Decimal-preserving source JSON projection; no report rounding',
                   derived=('ALL_COUNTS_AND_EXPOSURE_STATISTICS' if filename=='exposure_by_condition.csv' else
                            'ONLY_FIELDS_ENDING_DERIVED' if filename in ('training_steps.csv','fixed_fixtures.csv') else False),
                   field_projection={'exposure_by_condition.csv':'Group actual completed exposure events by arm, original uid and task_id; sum counts/free scalars, preserve extrema, derive decimal means',
                     'training_steps.csv':'All original top-level scalar fields; sample_losses[i].FM_contribution to 8 columns; 9*free denominator division explicitly DERIVED. Nested repeated metadata omitted, hashes/reference retained',
                     'fixed_fixtures.csv':'Frozen record+metadata scalar identity, hashes and augmentation; recorded NPZ hash only; observed training membership/count derived from actual exposures',
                     'fixed_FM_diagnostics.csv':'All 128 original fixture FM scalars and identity; group alias COMMON8/EXP_ONLY8; report-level PASS inherited and explicitly named'}[filename])
        outputs.append(out)
    missing_path=DEST/'_work/training_missing.json'
    provenance_path=DEST/'_work/training_provenance.json'
    if missing_path.exists() or provenance_path.exists():raise FileExistsError('Existing extraction receipt')
    missing_path.write_text(json.dumps(dict(status='COMPLETE_WITH_DECLARED_MISSING_FIELDS',items=MISSING),indent=2)+'\n')
    result=dict(schema='mini_training_extraction_v1',status='COMPLETE_WITH_DECLARED_MISSING_FIELDS',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=list(SOURCES.values()),outputs=outputs,
        model_calls=0,scoring_calls=0,OT_calls=0,weight_or_array_files_read=0,old_experiment_writes=0,
        script_path='protocol_and_code/extract_training.py',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        key_aliases={'parent_uid':'original exposure.uid / fixture.uid','condition_id':'original task_id; no string changes'},
        source_reference_policy='source_id joins sources here; external absolute paths are provenance only, tables need no source access',
        omitted_nested_step_metadata='sample_losses repeated batch denominator/alignment_eligible/raw_align/sample_index/alpha are not added as repeated columns; original source path/hash retained. Eight original FM_contribution values are preserved.',
        derived_interpretation='Exposure frequency/free-scalar sums are descriptive, not gradient contribution or causal evidence.',
        missing_file='_work/training_missing.json',checks={'unique_steps':len(seen_steps),'unique_exposure_sample_ids':{a:len(exposure_sample_ids[a]) for a in ARMS},
          'all_batches_8_distinct_parents_4clean_4width':True,'all_recorded_free_denominators_match_actual_exposure':True,
          'nested_parent_pools':[32,67],'fixture_plan_hash_matched':True,'original_decimal_values_not_rounded':True})
    provenance_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':result['status'],'outputs':[{k:o[k] for k in ('file','rows','bytes','sha256')} for o in outputs],
                      'missing_fields':len(MISSING),'checks':result['checks']},indent=2))


if __name__=='__main__':
    main()
