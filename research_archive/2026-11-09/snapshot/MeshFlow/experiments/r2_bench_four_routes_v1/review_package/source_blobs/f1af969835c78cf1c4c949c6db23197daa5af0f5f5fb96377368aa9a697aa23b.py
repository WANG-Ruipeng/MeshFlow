"""Read execution receipts only; no model imports, evaluation or scientific mutation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'experiments/r2_bench_four_routes_v1'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def lines(p):
    if not p.exists():return []
    result=[]
    with p.open(encoding='utf-8') as f:
        for s in f:
            try:result.append(json.loads(s))
            except json.JSONDecodeError:
                # A live writer can have an incomplete final JSON line.
                result.append({'event':'INCOMPLETE_LIVE_LINE'})
    return result

def training(route,arm):
    folder=OUT/route/'training'/arm
    status=read(folder/'status.json') if (folder/'status.json').exists() else {}
    records=lines(folder/'calls.jsonl')
    events=Counter((r.get('event','UNKNOWN'),r.get('kind','UNKNOWN')) for r in records)
    repeated=[];seen=set()
    for r in records:
        if r.get('event')!='CALL_BEGIN':continue
        key=(r.get('attempt_id'),r.get('call_id'))
        if key in seen:repeated.append(key)
        seen.add(key)
    failures=[r for r in records if any(x in r.get('event','').upper() for x in ('FAIL','ERROR'))]
    optimizer_steps=[r.get('context',{}).get('local_step') for r in records if r.get('event')=='CALL_BEGIN' and r.get('kind')=='optimizer_update']
    duplicate_optimizer_steps=[step for step,count in Counter(optimizer_steps).items() if count>1]
    result={'status':status.get('status','NOT_STARTED'),'local_step_status':status.get('local_step'),
        'global_step_status':status.get('global_step'),'events':{a+'::'+b:n for (a,b),n in events.items()},
        'successful':{kind:events['CALL_DONE',kind] for kind in ('train_forward','train_backward','optimizer_update')},
        'attempted':{kind:events['CALL_BEGIN',kind] for kind in ('train_forward','train_backward','optimizer_update')},
        'failure_events':failures,'duplicate_call_ids':repeated,'duplicate_optimizer_local_steps':duplicate_optimizer_steps,
        'optimizer_step_range':[min(optimizer_steps),max(optimizer_steps)] if optimizer_steps else [],
        'attempt_ids':sorted({r.get('attempt_id','') for r in records}),
        'calls_path':str(folder/'calls.jsonl'),'status_path':str(folder/'status.json'),
        'live_snapshot_not_atomic':status.get('status') not in ('COMPLETE','COMPLETED')}
    for name in ('checkpoint_latest.json','completion.json'):
        p=folder/name
        if p.exists():result[name]={'path':str(p),'sha256':sha(p),'record':read(p)}
    closeout=OUT/route/'closeout/training.json'
    if closeout.exists():result['independent_training_closeout']={'path':str(closeout),'sha256':sha(closeout),'record':read(closeout)}
    return result

def inference(route,label):
    base=OUT/route/'evaluation'/label
    result={'raw':[],'post':[],'layouts':[],'failures':[]}
    for phase in ('raw','post'):
        for attempt in sorted(base.glob('*/'+phase+'/attempt.json')):
            folder=attempt.parent;receipt=folder/'receipt.json';calls=lines(folder/'calls.jsonl')
            counts=Counter(r.get('event','UNKNOWN') for r in calls)
            r=read(receipt) if receipt.exists() else {}
            row={'request_id':folder.parent.name,'status':r.get('status','INCOMPLETE'),
                'receipt_path':str(receipt),'receipt_sha256':sha(receipt) if receipt.exists() else None,
                'events':dict(counts),'raw_sha256':r.get('raw_sha256'),'post_sha256':r.get('post_sha256'),
                'noise_sha256':r.get('noise_sha256'),'checkpoint_sha256':r.get('checkpoint_sha256'),
                'official_D0_sha256':r.get('official_D0_sha256'),
                'generator_forwards':r.get('new_generator_forwards',r.get('generator_forwards',0)),
                'D0_forwards':r.get('new_D0_forwards',r.get('D0_forwards',0))}
            for filename,key in (('raw.npz','raw_sha256'),('post.npz','post_sha256')):
                p=folder/filename
                if p.exists() and row.get(key):row[filename+'_hash_matches']=sha(p)==row[key]
            row['failure_events']=[x for x in calls if any(y in x.get('event','').upper() for y in ('FAIL','ERROR'))]
            if row['failure_events']:result['failures'].extend(row['failure_events'])
            result[phase].append(row)
    for p in sorted((base/'layouts').glob('*/receipt.json')):
        r=read(p);result['layouts'].append({'condition_id':r['condition_id'],'status':r['status'],
            'occupancy_forward_calls':r.get('occupancy_forward_calls',0),'closed_C_skipped':r.get('closed_C_skipped'),
            'input_sha256':r.get('input_sha256'),'receipt_path':str(p),'receipt_sha256':sha(p)})
    result['counts']={'completed_RAW':sum(r['status']=='COMPLETE' for r in result['raw']),
        'completed_D0_POST':sum(r['status']=='COMPLETE' for r in result['post']),
        'new_generator_forwards_receipts':sum(r['generator_forwards'] for r in result['raw']),
        'new_D0_forwards_receipts':sum(r['D0_forwards'] for r in result['post']),
        'generator_FORWARD_STARTED':sum(r['events'].get('FORWARD_STARTED',0) for r in result['raw']),
        'generator_FORWARD_COMPLETE':sum(r['events'].get('FORWARD_COMPLETE',0) for r in result['raw']),
        'D0_FORWARD_STARTED':sum(r['events'].get('D0_FORWARD_STARTED',0) for r in result['post']),
        'D0_FORWARD_RETURNED':sum(r['events'].get('D0_FORWARD_RETURNED',0) for r in result['post']),
        'occupancy_head_forwards':sum(r['occupancy_forward_calls'] for r in result['layouts'])}
    return result

def hybrid_final_checks(h,reg):
    """Validate exact unique requests and completed immutable receipts, without scoring."""
    expected={t['request_id'] for t in reg['tasks']}
    paths=list((h/'outputs').glob('*/complete.json'))
    actual={p.parent.name for p in paths}
    rh=sha(h/'registration.json');checks={}
    checks['hybrid_exact_request_ids']=len(expected)==46 and len(paths)==46 and actual==expected
    rows=[read(p) for p in paths]
    checks['hybrid_per_request_contract']=all(
        r.get('status')=='COMPLETE' and r.get('request_id')==p.parent.name and r.get('registration_sha256')==rh
        and all(r.get(k)==v for k,v in {'model_calls':0,'D0_calls':0,'GEO_mapping_calls':1,'GEO_selected_calls':1,'cascade_scores':2}.items())
        for p,r in zip(paths,rows))
    checks['hybrid_request_artifact_hashes']=all(sha(Path(p))==digest for r in rows for p,digest in r.get('files',{}).items())
    cpu=[read(h/'cpu/HYBRID'/rid/'complete.json') for rid in sorted(actual)]
    checks['hybrid_nested_CPU_hashes']=all(sha(Path(p))==digest for r in cpu for p,digest in r.get('files',{}).items())
    gp=h/'completions/geometry_scores.json';g=read(gp) if gp.exists() else {}
    checks['hybrid_global_completion']=all(g.get(k)==v for k,v in
        {'status':'COMPLETE','requests':46,'new_scores':92,'GEO_calls':92,'model_calls':0,'D0_calls':0}.items())
    checks['hybrid_global_completion_hashes']=set(g.get('files',{}))=={str(p) for p in paths} and all(
        sha(Path(p))==digest for p,digest in g.get('files',{}).items())
    folder=h/'dispatch/preserved_worker_recovery_v1';rp=folder/'registration.json'
    if rp.exists():
        plan=read(rp);cp=folder/'complete.json';r=read(cp) if cp.exists() else {}
        events=lines(folder/'events.jsonl');dispatches=[e.get('request_id') for e in events if e.get('event')=='WORKER_DISPATCH']
        exits=[e.get('request_id') for e in events if e.get('event')=='WORKER_EXIT_COMPLETE']
        checks['hybrid_recovery_exact_five_once']=len(dispatches)==len(exits)==5 and set(dispatches)==set(exits)==set(plan['task_ids']) and len(set(dispatches))==5
        checks['hybrid_recovery_contract']=all(r.get(k)==v for k,v in {'status':'COMPLETE_NO_SCIENTIFIC_REPLAY',
            'new_requests':5,'new_model_calls':0,'new_D0_calls':0,'GEO_mapping_calls':5,'GEO_selected_calls':5,'new_score_rows':10}.items())
        worker_receipts=[read(folder/'workers'/rid/'complete.json') for rid in plan['task_ids']]
        checks['hybrid_five_first_execution_unchanged_on_original_pool_read']=all(
            w.get('status')=='COMPLETE_FIRST_EXECUTION' and w.get('registration_sha256')==rh and w.get('scientific_replay') is False
            and w.get('model_calls')==0 and w.get('request_id')==rid
            and r.get('tasks',{}).get(rid)==w.get('verified')
            and w.get('verified',{}).get('sha256')==sha(h/'outputs'/rid/'complete.json')
            and w.get('verified',{}).get('CPU_sha256')==sha(h/'cpu/HYBRID'/rid/'complete.json')
            for rid,w in zip(plan['task_ids'],worker_receipts))
        checks['hybrid_recovery_resumed']=set(r.get('resumes',{}))=={'slow','renderer'} and all(
            s.get('stopped_cpu_seconds_delta')==0 and s.get('observed',{}).get('event')=='RESUME_CONFIRMED'
            and s.get('observed',{}).get('process',{}).get('state')!='T' for s in r.get('resumes',{}).values())
        checks['hybrid_recovery_files']=bool(r.get('files')) and all(sha(Path(p))==digest for p,digest in r.get('files',{}).items())
    return checks

def run(final=False):
    now=datetime.now(timezone.utc)
    result={'utc':now.isoformat(),'scope':'Read-only execution accounting. No model, scoring or tensor loads.',
        'status':'FINAL' if final else 'SNAPSHOT_RUNNING_ALLOWED','large_weights_rehashed':False,
        'input_audit_policy':'Record frozen registration identities and existing gate/closeout receipts; no repeated whole-dataset or large-weight audit.',
        'weight_identity_policy':'Reuse immutable training/export receipts and registered hash identity; do not repeatedly read multi-GB weights.',
        'arms':{}}
    for route,arm,label in (('exposure','M_PLUS3000','E'),('seam','SEAM1500','S')):
        # Exposure label is read from its frozen plan, never inferred from arm name.
        if route=='exposure':
            ep=read(OUT/route/'data/evaluation_plan.json')
            label=ep.get('new_labels',[label])[0] if ep.get('new_labels') else label
        result['arms'][route]={'training':training(route,arm),'inference':inference(route,label)}
    result['arms']['bestof']={'training_updates':0,'inference':inference('bestof','B4')}
    h=OUT/'hybrid';reg=read(h/'registration.json')
    result['arms']['hybrid']={'training_updates':0,'generator_forwards':0,'D0_forwards':0,
        'completed_requests':len(list((h/'outputs').glob('*/complete.json'))),
        'registered_requests':len(reg['tasks']),'registered_budget':reg['budget'],
        'registration_sha256':sha(h/'registration.json'),'protected_input_hash_count':len(reg['protected'])}
    for route in result['arms']:
        folder=OUT/route
        receipts={}
        candidates=[folder/'registration.json',folder/'training_registration.json',folder/'data/eval_manifest.json',
            folder/'data/evaluation_plan.json',folder/'results/complete.json',folder/'results/completion.json',
            folder/'completions/geometry_scores.json',folder/'selection/complete.json',folder/'gallery/completion.json',
            folder/'visual_review/summary.json',folder/'audits/dispatch_accounting_note.json']
        for p in candidates:
            if p.exists():receipts[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
        result['arms'][route]['receipt_identity']=receipts
        cpu_completions=[]
        for p in sorted((folder/'evaluation/completions').glob('*.json')):
            doc=read(p)
            cpu_completions.append({'path':str(p),'sha256':sha(p),
                **{k:doc[k] for k in ('status','mode','cells','new_requests','workers','seconds','model_calls','new_queries') if k in doc}})
        result['arms'][route]['CPU_scoring_completions']=cpu_completions
        model_label={'exposure':'EXPOSURE','seam':'S','bestof':'B4','hybrid':'HYBRID'}[route]
        actual_scores={}
        for stage in ('RAW','BASE_POST','CLEAN_FINAL_POST'):
            parent=(folder/'evaluation/scores'/model_label) if stage=='RAW' else (folder/'cpu'/model_label)
            score_receipts=sorted(parent.glob('*/'+stage+'/complete.json')) if stage=='RAW' else sorted(parent.glob('*/'+stage+'/scores/complete.json'))
            actual_scores[stage]=len(score_receipts)
        result['arms'][route]['new_score_completion_counts']=actual_scores
        for name in ('training_registration.json','registration.json'):
            p=folder/name
            if p.exists():
                doc=read(p)
                result['arms'][route][name+'_input_identity']={k:doc[k] for k in (
                    'source_global_step','final_global_step','source_checkpoint','source_checkpoint_sha256',
                    'source_generator_sha256','source_optimizer_sha256','source_rng_sha256','manifest','manifest_sha256',
                    'eval_manifest','eval_manifest_sha256','generator_sha256') if k in doc}
        failure_files=[]
        for path in folder.rglob('*.json'):
            if any(x in path.name.lower() for x in ('failure','failed','restart','replay','block','engineering_fix')):
                failure_files.append({'path':str(path),'sha256':sha(path)})
        result['arms'][route]['named_incident_records']=failure_files
    caption=OUT/'exposure/audits/caption_adapter_cpu_check.json'
    result['engineering_exits_separate_from_scientific_calls']=[{'kind':'EXPOSURE_CAPTION_PREFLIGHT_ASSERT',
        'path':str(caption),'sha256':sha(caption),'model_calls':0,'scientific_replay':False}] if caption.exists() else []
    cli=OUT/'seam/closeout/cli_argument_correction.json'
    if cli.exists():result['engineering_exits_separate_from_scientific_calls'].append({
        'kind':'SEAM_UNSUPPORTED_CLI_ARGUMENT_PREFLIGHT','path':str(cli),'sha256':sha(cli),
        'model_calls':0,'scientific_replay':False})
    result['other_engineering_or_reporting_notes']=[]
    for kind,name in (('HYBRID_CONSERVATIVE_RESOURCE_SCHEMA_FALLBACK','dispatch_accounting_note.json'),
                      ('HYBRID_INFORMAL_PHASE_TIME_WITHDRAWN','phase_timing_reporting_note.json'),
                      ('HYBRID_OS_THREAD_PREFLIGHT_CORRECTED_ZERO_CALL','recovery_thread_gate_correction.json'),
                      ('HYBRID_DISPATCH_ORDER_COMPLETION_INFERENCE_CORRECTED','progress_identity_reporting_correction.json')):
        p=h/'audits'/name
        if p.exists():result['other_engineering_or_reporting_notes'].append({'kind':kind,'path':str(p),'sha256':sha(p),'scientific_replay':False})
    recovery=h/'dispatch/preserved_worker_recovery_v1/complete.json'
    if recovery.exists():
        r=read(recovery)
        result['arms']['hybrid']['scheduling_recovery']={'path':str(recovery),'sha256':sha(recovery),
            **{k:r[k] for k in ('status','resumes','slow_phase_at_stop','active_geometry_workers_max','new_requests','GEO_mapping_calls','GEO_selected_calls','new_score_rows','timing_note')}}
        result['arms']['hybrid']['scheduling_recovery']['counting_note']='These5 previously untouched requests belong to original46, not5 additional scientific cases. Original pool later verifies/returns the same completions without GEO/scorer calls.'
    result['no_replay_conclusion']='Only assert zero replay after training closeouts and per-call identities are complete; a live snapshot is not a final attestation.'
    if final:
        checks={}
        for route,updates in (('exposure',3000),('seam',1500)):
            arm=result['arms'][route];tr=arm['training'];inf=arm['inference'];counts=inf['counts']
            checks[route+'_training']=tr['successful']=={'train_forward':8*updates,'train_backward':8*updates,'optimizer_update':updates}
            checks[route+'_fixed_terminal_state']=tr['status']=='COMPLETE' and tr['local_step_status']==updates and tr['global_step_status']==22500+updates
            checks[route+'_calls_balanced']=tr['attempted']==tr['successful'] and not tr['failure_events'] and not tr['duplicate_call_ids'] and not tr['duplicate_optimizer_local_steps'] and tr['optimizer_step_range']==[1,updates]
            checks[route+'_sampling']=all(counts[key]==value for key,value in {
                'completed_RAW':46,'completed_D0_POST':46,'new_generator_forwards_receipts':2300,
                'new_D0_forwards_receipts':46,'generator_FORWARD_STARTED':2300,'generator_FORWARD_COMPLETE':2300,
                'D0_FORWARD_STARTED':46,'D0_FORWARD_RETURNED':46}.items())
        b=result['arms']['bestof']['inference']['counts'];bc=read(OUT/'bestof/results/completion.json')
        checks['bestof_sampling']=b['completed_RAW']==46 and b['new_generator_forwards_receipts']==2300 and b['generator_FORWARD_STARTED']==b['generator_FORWARD_COMPLETE']==2300 and b['completed_D0_POST']==bc['new_D0']==b['new_D0_forwards_receipts']==b['D0_FORWARD_STARTED']==b['D0_FORWARD_RETURNED']==13
        layouts=result['arms']['seam']['inference']['layouts']
        checks['seam_C_only_head_budget']=len(layouts)==23 and all(r['status']=='COMPLETE' for r in layouts) and sum(r['occupancy_forward_calls'] for r in layouts)==21 and sum(bool(r['closed_C_skipped']) for r in layouts)==2
        checks.update(hybrid_final_checks(h,reg))
        checks['hybrid_46_complete']=result['arms']['hybrid']['completed_requests']==46 and (h/'completions/geometry_scores.json').exists()
        checks['scheduling_recovery_resumed_and_accounted']=not recovery.with_name('registration.json').exists() or (recovery.exists() and read(recovery)['status']=='COMPLETE_NO_SCIENTIFIC_REPLAY')
        expected_scores={'exposure':{'RAW':46,'BASE_POST':46,'CLEAN_FINAL_POST':46},
            'seam':{'RAW':46,'BASE_POST':46,'CLEAN_FINAL_POST':46},
            'bestof':{'RAW':46,'BASE_POST':bc['new_D0'],'CLEAN_FINAL_POST':bc['new_D0']},
            'hybrid':{'RAW':0,'BASE_POST':46,'CLEAN_FINAL_POST':46}}
        checks['all_physical_new_score_completions']=all(result['arms'][route]['new_score_completion_counts']==expected for route,expected in expected_scores.items())
        checks['all_small_output_hashes_match']=all(value for arm in result['arms'].values() for phase in ('raw','post') for row in arm.get('inference',{}).get(phase,[]) for key,value in row.items() if key.endswith('_hash_matches'))
        checks['no_inference_failure_events']=all(not arm.get('inference',{}).get('failures') for arm in result['arms'].values())
        result['final_checks']=checks
        if not all(checks.values()):raise RuntimeError('Final execution accounting not yet satisfied: '+repr(checks))
    path=OUT/'audits'/('four_routes_execution_ledger_final.json' if final else 'four_routes_execution_ledger_'+now.strftime('%Y%m%dT%H%M%SZ')+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print({'path':str(path),'status':result['status'],'inference_counts':{k:v.get('inference',{}).get('counts') for k,v in result['arms'].items()}})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');args=p.parse_args();run(args.final)
