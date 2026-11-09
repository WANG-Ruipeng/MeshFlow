"""One-thread, model-free audit of the completed frozen training ledger.

Only outputs in engineering are written. Models, checkpoint tensors, RAW arrays,
and geometry evaluators are never loaded. Checkpoint files are streamed for SHA.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import csv
from collections import Counter,defaultdict
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from r2.pilots.window_policy_rl_v1.bootstrap import bind,OUT
D,E=bind()

SOURCES={};VIOLATIONS=[];CHECKS=Counter();UNCHANGED=[]

def hash_file(path):
    path=D.resolve(path);h=hashlib.sha256()
    before=path.stat()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    after=path.stat()
    check((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'INPUT_STABLE_DURING_READ',str(path))
    SOURCES[str(path)]={'sha256':h.hexdigest(),'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}
    return h.hexdigest()

def read(path):
    path=D.resolve(path);data=path.read_bytes();st=path.stat()
    SOURCES[str(path)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'mtime_ns':st.st_mtime_ns}
    return json.loads(data.decode('utf-8-sig'))

def events(path):
    path=D.resolve(path);data=path.read_bytes();st=path.stat()
    SOURCES[str(path)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'mtime_ns':st.st_mtime_ns}
    check(data.endswith(b'\n'),'EVENT_FILE_CLOSED_LINE',str(path))
    return [json.loads(line) for line in data.splitlines() if line.strip()]

def check(ok,name,detail=None):
    CHECKS[name]+=1
    if not ok:VIOLATIONS.append({'check':name,'detail':detail})

def close(a,b):return math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8)
def stats(xs):
    xs=list(xs)
    return {'n':len(xs),'min':min(xs) if xs else None,'max':max(xs) if xs else None,'mean':statistics.fmean(xs) if xs else None}

def run():
    check('torch' not in sys.modules,'NO_TORCH_IMPORT')
    cfg=read(OUT/'training_config.json');lock=read(OUT/'training_source_lock.json')
    tr=read(OUT/'inputs/training_manifest.json');manifest=read(OUT/'inputs/manifest.json')
    registration=read(OUT/'registration.json');final=read(OUT/'training/final.json')
    check(hash_file(OUT/'training_config.json')==lock['training_config_sha256'],'TRAINING_CONFIG_FROZEN')
    check(hash_file(OUT/'inputs/training_manifest.json')==cfg['training_manifest_sha256'],'TRAINING_MANIFEST_FROZEN')
    check(hash_file(OUT/'inputs/manifest.json')==registration['manifest_sha256'],'EVALUATION_MANIFEST_FROZEN')
    check(hash_file(cfg['gate_path'])==cfg['gate_sha256'],'DECODE_GATE_FROZEN')
    check(cfg['source_files']==lock['source_files'],'SOURCE_LOCK_EQUALS_CONFIG')
    for p,h in cfg['source_files'].items():
        actual=hash_file(p);check(actual==h,'FROZEN_EXECUTION_SOURCE',p);UNCHANGED.append({'path':p,'expected':h,'actual':actual})
    check(final['status']=='COMPLETE' and final['batches']==64 and final['rollouts']==512,'FINAL_CONTROLLER_COMPLETE')
    check(len(final['records'])==64,'FINAL_RECORD_COVERAGE')
    qs={q['request_id']:q for q in tr['training_requests']}
    check(len(qs)==512 and len(tr['groups'])==64,'MANIFEST64_512')
    condition_counts=Counter(q['condition_id'] for q in qs.values())
    check(len(condition_counts)==8 and set(condition_counts.values())=={64},'EACH_C64_ROLLOUTS')
    check(len({q['Gaussian_sha256'] for q in qs.values()})==512,'INITIAL_NOISE512_UNIQUE')
    check(len({q['transition_Gaussian_sha256'] for q in qs.values()})==512,'TRANSITION_NOISE512_UNIQUE')
    global_counts=Counter();batches=[];checkpoints=[];prior=None;updates=0;old_errors=[];row_kls=[]
    scorer_sources={};d0_checkpoint_hashes=set();binary_reward_total=0
    for group in tr['groups']:
        b=group['batch_index'];stage=OUT/'training'/f'batch_{b:03d}'
        expected_ids=group['requests'];check(len(expected_ids)==8,'EIGHT_REQUESTS_PER_BATCH',b)
        check(len({qs[r]['condition_id'] for r in expected_ids})==1,'B1_ONE_CONDITION',b)
        marker={p:read(stage/name/'completion.json') for p,name in (
            ('sample','sample_ledger'),('d0','d0_ledger'),('cpu','cpu_ledger'),('update','update'))}
        check(all(v['status']=='COMPLETE' for v in marker.values()),'FOUR_PHASES_COMPLETE',b)
        for p in ('sample_ledger/failure.json','d0_ledger/failure.json','d0_ledger/phase/failure.json','cpu_ledger/failure.json','update/failure.json'):
            check(not (stage/p).exists(),'NO_FAILURE_MARKER',str(stage/p))
        streams={'sample':events(stage/'sample_ledger/calls.jsonl'),'d0':events(stage/'d0_ledger/events.jsonl'),
                 'update':events(stage/'update/events.jsonl')}
        c=Counter()
        for phase,rows in streams.items():
            for r in rows:c[(phase,r['event'],r.get('kind',''))]+=1
        expected={('sample','ROLLOUT_ATTEMPT',''):8,('sample','ROLLOUT_COMPLETE',''):8,
            ('sample','FORWARD_START','sampling_forward'):400,('sample','FORWARD_RETURN','sampling_forward'):400,
            ('d0','D0_FORWARD_ATTEMPT',''):8,('d0','D0_FORWARD_RETURN',''):8,
            ('update','FORWARD_START','training'):160,('update','FORWARD_RETURN','training'):160,
            ('update','FORWARD_START','postupdate_kl'):160,('update','FORWARD_RETURN','postupdate_kl'):160,
            ('update','FORWARD_START','zero_lora_validation'):2 if b==0 else 0,
            ('update','FORWARD_RETURN','zero_lora_validation'):2 if b==0 else 0,
            ('update','BACKWARD_START',''):160,('update','BACKWARD_RETURN',''):160}
        for key,value in expected.items():check(c[key]==value,'EXACT_EVENT_COUNT',{'batch':b,'key':key,'expected':value,'actual':c[key]})
        global_counts.update(c)
        rewards=read(stage/'rewards.json');m=read(stage/'update/training_metrics.json');cp=read(stage/'update/checkpoint.json')
        reward=[r['reward'] for r in rewards['records']]
        check([r['request_id'] for r in rewards['records']]==expected_ids,'REWARD_ALL_EIGHT_ORDER',b)
        check(all(r in (0,1) for r in reward) and m['rewards']==reward,'BINARY_REWARDS_METRICS_BINDING',b)
        active=0<sum(reward)<8;updates+=int(active);binary_reward_total+=sum(reward)
        for key,value in ((('update','OPTIMIZER_STEP_START',''),int(active)),(('update','OPTIMIZER_STEP_RETURN',''),int(active)),(('update','OPTIMIZER_STEP_SKIPPED',''),int(not active))):
            check(c[key]==value,'ACTIVE_OR_SKIP_EXACT_STEP',{'batch':b,'key':key})
        check(m['actual_optimizer_updates']==int(active) and m['optimizer_steps']==updates and cp['optimizer_steps']==updates,'OPTIMIZER_CLOCK',b)
        check(cp['opportunities_completed']==b+1 and cp['batch_index']==b,'OPPORTUNITY_CLOCK',b)
        check(cp['current_adapters_sha256']==m['current_hash_after'],'CHECKPOINT_ADAPTER_RECEIPT',b)
        if prior:check(m['current_hash_before']==prior['current_adapters_sha256'],'ON_POLICY_CURRENT_CHAIN',b)
        check(hash_file(cp['path'])==cp['sha256'],'CHECKPOINT_FILE_SHA256',b)
        record=final['records'][b]
        check(record['batch_index']==b and record['checkpoint_sha256']==cp['sha256'] and record['reward_sum']==sum(reward),'FINAL_RECORD_BINDS_BATCH',b)
        adv=[r-(sum(reward)-r)/7 for r in reward]
        check(len(m['advantages'])==8 and all(close(a,x) for a,x in zip(adv,m['advantages'])),'RLOO_ADVANTAGE',b)
        check(m['gradient_norm_after_clip']<=1.000005,'GRADIENT_CLIP_BOUND',b)
        if not active:
            check(m['gradient_norm_before_clip']==0 and m['gradient_norm_after_clip']==0 and m['adapter_delta_l2']==0
                  and m['current_hash_before']==m['current_hash_after'] and m['postupdate_trajectory_KL_mean']==0,'INACTIVE_NO_PARAMETER_OR_KL_CHANGE',b)
        else:check(m['adapter_delta_l2']>0 and m['current_hash_before']!=m['current_hash_after'],'ACTIVE_ADAPTER_CHANGE',b)
        rows=m['rows'];check(len(rows)==160,'160_WINDOW_ROWS',b)
        check({(r['member'],r['euler_step']) for r in rows}=={(i,k) for i in range(8) for k in range(10,30)},'FULL_WINDOW_MEMBER_COVERAGE',b)
        traj=[0.]*8
        for row in rows:
            i=row['member'];check(row['request_id']==expected_ids[i] and close(row['advantage'],adv[i]),'WINDOW_ROW_ID_ADV',b)
            check(close(row['loss_contribution'],-adv[i]*row['logp']/8),'FULL_SUM_OBJECTIVE_WEIGHT',b)
            check(row['old_logp_abs_error']==0,'OLD_LOGP_EXACT_MATCH',b)
            check(row['postupdate_conditional_kl']>=0,'NONNEGATIVE_CONDITIONAL_KL',b)
            old_errors.append(row['old_logp_abs_error']);row_kls.append(row['postupdate_conditional_kl']);traj[i]+=row['postupdate_conditional_kl']
        check(close(math.fsum(r['loss_contribution'] for r in rows),m['loss']),'LOSS_ROWS_SUM',b)
        check(all(close(x,y) for x,y in zip(traj,m['postupdate_conditional_KL_sum_by_trajectory'])) and close(statistics.fmean(traj),m['postupdate_trajectory_KL_mean']),'FULL_WINDOW_KL_MEAN',b)
        for rid,rr in zip(expected_ids,rewards['records']):
            q=qs[rid];raw=read(stage/'raw'/rid/'receipt.json');prop=read(stage/'proposal'/rid/'receipt.json')
            check(raw['status']=='COMPLETE' and raw['request_id']==rid and raw['model_mode']=='current' and raw['sigma']==.005,'RAW_POLICY_IDENTITY',rid)
            check(all(raw[k]==q[k] for k in ('input_sha256','C_sha256','Gaussian_sha256','transition_Gaussian_sha256','N','K','condition_id')),'RAW_FROZEN_INPUT_BINDING',rid)
            sampling=raw['sampling']
            check(sampling['known_all51_bitwise'] and sampling['known_state_checks']==51 and sampling['runtime_RNG_unchanged'] and sampling['new_random_draws']==0 and sampling['nfe']==50,'RECORDED_C51_AND_RNG_CONTRACT',rid)
            model=raw['model_identity'];check(model['base_file_sha256']==manifest['source_binding']['sha256'] and model['current_adapters_sha256']==m['current_hash_before'],'ROLLOUT_TRAIN_CURRENT_BINDING',rid)
            if prior:check(model['adapter_checkpoint_sha256']==prior['sha256'],'ROLLOUT_PRIOR_CHECKPOINT_BINDING',rid)
            else:check(model['fresh_zero_lora'] is True and model['init_seed']==cfg['adapter_init_seed'],'FIRST_ROLLOUT_ZERO_LORA_BINDING',rid)
            check(prop['status']=='PASS' and prop['model_forwards']==1 and prop['raw_sha256']==raw['raw_sha256'],'OFFICIAL_D0_ONE_FORWARD_RAW_BINDING',rid)
            d0_checkpoint_hashes.add(prop['checkpoint_sha256'])
            score=read(rr['post_score_path']);check(SOURCES[str(D.resolve(rr['post_score_path']))]['sha256']==rr['post_score_sha256'],'REWARD_SCORE_HASH',rid)
            check(score['status']=='PASS' and score['original_definition_unchanged'] is True and int(score['topology']['C_F_valid_edge_count']>0)==rr['reward'],'REWARD_ORIGINAL_SCORE_VALUE',rid)
            check(score['identity']['input_hashes'][raw['raw_path']]==raw['raw_sha256'],'SCORE_RAW_BINDING',rid)
            scorer_sources.update(score['identity']['definition_hashes'])
        batches.append(dict(batch_index=b,condition_id=group['condition_id'],cycle=qs[expected_ids[0]]['cycle'],reward_sum=sum(reward),active=active,
            optimizer_steps=updates,adapter_delta_l2=m['adapter_delta_l2'],grad_before=m['gradient_norm_before_clip'],grad_after=m['gradient_norm_after_clip'],
            conditional_KL_mean=m['postupdate_trajectory_KL_mean'],old_logp_max_abs_error=m['preupdate_old_logp_max_abs_error']))
        checkpoints.append(cp);prior=cp
    check(final['checkpoint_sha256']==checkpoints[-1]['sha256'] and final['checkpoint_path']==checkpoints[-1]['path'],'FINAL_LAST_CHECKPOINT_ONLY')
    for path,expected in scorer_sources.items():check(hash_file(path)==expected,'ORIGINAL_SCORER_SOURCE_HASH',path)
    check(len(d0_checkpoint_hashes)==1,'ONE_D0_CHECKPOINT_IDENTITY')
    sessions=[]
    for p in sorted((OUT/'training/resident_d0_sessions').glob('*/closed.json')):
        closed=read(p);check(closed['status']=='CLOSED_EOF' and closed['failure'] is None and closed['owned_model_released'],'D0_CLEAN_EOF',str(p))
        check(closed['forward_attempts']==512 and closed['forward_returns']==512 and closed['model_loads_attempted']==1 and closed['model_loads_completed']==1,'D0_SINGLE_LOAD_512_CALLS',str(p));sessions.append(closed)
    check(len(sessions)==1,'ONE_RESIDENT_D0_SESSION')
    pointer=read(OUT/'analysis/training_latest.json');summary=read(pointer['summary_path'])
    check(summary['status']=='COMPLETE' and not summary['violations'] and not summary['input_read_notes'],'TRAINING_SUMMARY_COMPLETE')
    conversions={'sampling_F_returns':('sample','FORWARD_RETURN','sampling_forward'),'D0_F_returns':('d0','D0_FORWARD_RETURN',''),
        'training_F_returns':('update','FORWARD_RETURN','training'),'postupdate_kl_F_returns':('update','FORWARD_RETURN','postupdate_kl'),
        'zero_lora_validation_F_returns':('update','FORWARD_RETURN','zero_lora_validation'),'backward_returns':('update','BACKWARD_RETURN',''),
        'optimizer_step_returns':('update','OPTIMIZER_STEP_RETURN',''),'optimizer_skip_events':('update','OPTIMIZER_STEP_SKIPPED','')}
    for key,value in conversions.items():check(summary['actual_physical_counts'][key]==global_counts[value],'INDEPENDENT_COUNTS_VS_SUMMARY',key)
    conditions=[]
    for cid in condition_counts:
        own=[r for r in batches if r['condition_id']==cid]
        check(len(own)==8 and sorted(r['cycle'] for r in own)==list(range(8)),'EACH_C_EIGHT_CYCLES',cid)
        conditions.append(dict(condition_id=cid,opportunities=8,rollouts=64,reward_1=sum(r['reward_sum'] for r in own),
            active=sum(r['active'] for r in own),skipped=sum(not r['active'] for r in own),reward_sums_by_cycle=[r['reward_sum'] for r in sorted(own,key=lambda r:r['cycle'])]))
    counts={key:global_counts[value] for key,value in conversions.items()}
    counts.update(rollouts=global_counts[('sample','ROLLOUT_COMPLETE','')],opportunities=64,
        total_model_F_returns=sum(global_counts[k] for k in global_counts if (k[0] in ('sample','update') and k[1]=='FORWARD_RETURN') or (k[0]=='d0' and k[1]=='D0_FORWARD_RETURN')))
    check(counts['total_model_F_returns']==46594 and updates==54 and counts['optimizer_skip_events']==10,'FINAL_EXACT_BUDGET')
    model_free='torch' not in sys.modules;check(model_free,'NO_MODEL_RUNTIME_AT_END')
    hash_file(__file__)
    result=dict(version='window_policy_rl_independent_training_execution_audit_v1',status='PASS' if not VIOLATIONS else 'FAIL',
        checks=dict(CHECKS),violations=VIOLATIONS,actual_counts=counts,condition_exposure=conditions,batches=batches,
        binary_reward_1_total=binary_reward_total,binary_reward_denominator=512,old_logp_errors=stats(old_errors),
        conditional_KL_rows=stats(row_kls),metrics={name:stats(r[name] for r in batches) for name in ('adapter_delta_l2','grad_before','grad_after','conditional_KL_mean')},
        active_metrics={name:stats(r[name] for r in batches if r['active']) for name in ('adapter_delta_l2','grad_before','grad_after','conditional_KL_mean')},
        frozen_source_files=UNCHANGED,original_scorer_source_files=scorer_sources,training_summary=pointer,
        final_checkpoint={'path':final['checkpoint_path'],'sha256':final['checkpoint_sha256'],'all64_file_hashes_verified':True},
        resource_ranges=summary['resource_ranges'],resource_observations=summary['resource_observations'],
        resource_scope=summary['resource_scope'],process_memory_NA_observations=summary['process_memory_NA_observations'],
        wall_seconds_phase_sums=summary['per_phase_wall_seconds_sum'],wall_seconds_batch_overall_sum=summary['overall_wall_seconds_sum'],
        per_phase_wall_seconds=summary['per_phase_wall_seconds'],training_metric_ranges=summary['training_metric_ranges'],
        D0_sessions=sessions,read_input_sources=SOURCES,model_calls=0,checkpoint_tensor_loads=0,geometry_rescores=0,
        limitations=['This is an execution/accounting audit, not evidence of final-panel improvement.',
            'Online rewards use changing policies and fresh noise; condition exposure is balanced but these are not a fixed test curve.',
            'C51 bit-exact preservation and elementwise exact mean equality are runtime receipts/assertions under frozen source; no forward or RAW-array replay is performed here. torch.equal means exact numerical equality, not raw byte identity; no separate mean max-error field was stored.',
            'All64 adapter checkpoint file hashes are verified by streaming bytes; no tensor deserialization or adapter-distance recomputation.',
            'Official large FULL/D0 file bytes are not rehashed here; their frozen identities are checked in receipts. Original scoring source files are rehashed.',
            'Global GPU samples can include reference-evaluation overlap and resident D0. CUDA allocator peaks do not equal whole-process VRAM.',
            'No final evaluation reviews, anonymous mapping, geometry re-score or models are read/executed.'])
    D.write(OUT/'engineering/TRAINING_EXECUTION_AUDIT.json',result)
    print(json.dumps({'status':result['status'],'violations':VIOLATIONS,'counts':counts,'conditions':conditions,
        'metrics':result['metrics'],'active_metrics':result['active_metrics'],'resources':result['resource_ranges'],
        'wall_seconds_phase_sums':result['wall_seconds_phase_sums'],'wall_seconds_batch_overall_sum':result['wall_seconds_batch_overall_sum'],
        'audit_sha256':D.sha(OUT/'engineering/TRAINING_EXECUTION_AUDIT.json')},sort_keys=True))

if __name__=='__main__':run()
