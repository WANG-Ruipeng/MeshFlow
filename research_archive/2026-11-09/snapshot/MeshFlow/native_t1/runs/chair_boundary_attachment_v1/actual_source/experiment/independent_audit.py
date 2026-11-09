"""Independent read-only CPU accounting and table arithmetic, no model imports.

The final audit reads only this experiment's named evidence. It never draws
noise, runs OT, recomputes geometry, loads a model, or calls the aggregator.
Checkpoint tensors are CPU memory-mapped only to inspect metadata/state layout.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import csv
import datetime
import hashlib
import json
import math
import statistics
import traceback

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'native_t1/runs/chair_boundary_attachment_v1'
ARMS=('A_FM','B_BOUNDARY','C_SEAM','D_BOUNDARY_SEAM')
LABELS=('START_HYBRID',)+tuple(a+'_END' for a in ARMS)
BOUNDARY={'B_BOUNDARY','D_BOUNDARY_SEAM'}
SEAM={'C_SEAM','D_BOUNDARY_SEAM'}
START_SHA='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
METRICS=('SeamCov','SeamCov_upper','SeamCov_0.005','SeamCov_0.01','SeamCov_0.02',
    'ConflictCov_C','ConflictCov_C_upper','ConflictCov_C_GT','ConflictCov_C_minus_GT',
    'ConflictCov_C_cross','ConflictCov_C_dup','ConflictCov_C_cross_and_dup',
    'ExposedRetain','ExposedRetain_upper','ExposedRetain_global_lower','ExposedRetain_global_upper','reference_exposed_fraction','ExposedRetain_uncertain_fraction','reference_visibility_uncertain_fraction',
    'D_sym2048','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','fixed_query_MSE','fixed_query_p95_L',
    'coverage_fraction','FF_illegal_affected_free_fraction','CF_proper_affected_free_fraction',
    'free_minimum_angle_lt5_fraction','free_near_degenerate_fraction','free_exact_zero_area_fraction',
    'free_area_over_GT_free_area','free_out_of_GT_cube_corner_fraction','generation_seconds',
    'peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','common_RMS_L0','common_p95_L0','common_recall_0.01')
CONTRASTS=(('D-A','D_BOUNDARY_SEAM_END','A_FM_END'),('B-A','B_BOUNDARY_END','A_FM_END'),
    ('C-A','C_SEAM_END','A_FM_END'),('D-B','D_BOUNDARY_SEAM_END','B_BOUNDARY_END'),
    ('D-C','D_BOUNDARY_SEAM_END','C_SEAM_END'))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def mean(v):return statistics.fmean(v) if v else None
def numeric(v):
    if v is None or v in ('','NA','N/A','None','null'):return None
    if v in ('True','true'):return True
    if v in ('False','false'):return False
    try:return float(v)
    except (ValueError,TypeError):return v
def csv_rows(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return [{k:numeric(v) for k,v in r.items()} for r in csv.DictReader(f)]
def json_lines(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        for i,line in enumerate(f,1):
            if line.strip():yield i,json.loads(line)
def close(a,b,rtol=2e-7,atol=1e-10):
    return a is None and b is None or finite(a) and finite(b) and math.isclose(float(a),float(b),rel_tol=rtol,abs_tol=atol)
def write_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


class Audit:
    def __init__(self):self.counts=Counter();self.failures=[]
    def check(self,name,ok,**details):
        self.counts[name]+=1
        if not bool(ok):self.failures.append(dict(check=name,**details))


def _readiness(out):
    states={}
    for arm in ARMS:
        p=out/'training'/f'{arm}_status.json'
        states[arm]=read(p) if p.exists() else {'status':'PENDING'}
    p=out/'aggregation/summary.json';agg=read(p) if p.exists() else {'status':'PENDING'}
    ready=all(s.get('status')=='DONE' and s.get('attachment_step')==1000 for s in states.values()) and agg.get('status')=='COMPLETE'
    return ready,{a:dict(status=s.get('status'),attachment_step=s.get('attachment_step',0)) for a,s in states.items()},agg.get('status')


def audit_ledger(out,audit):
    begins={};ended={};meta=Counter();attempts=Counter();returns=Counter();failed=Counter();seconds=Counter()
    train=Counter();samples=Counter();files=[]
    for p in sorted((out/'ledger').glob('*.jsonl')):
        files.append(dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size))
        for line,r in json_lines(p):
            ev=r.get('event')
            if isinstance(ev,str) and ev.startswith('CALL'):
                audit.check('known_CALL_record_format',ev in ('CALL_BEGIN','CALL_RETURN','CALL_FAILED') and all(k in r for k in ('call_id','kind','context')),file=p.name,line=line)
                if ev not in ('CALL_BEGIN','CALL_RETURN','CALL_FAILED') or not all(k in r for k in ('call_id','kind','context')):continue
                cid=r['call_id'];kind=r['kind']
                if ev=='CALL_BEGIN':
                    audit.check('call_id_begins_once',cid not in begins,call_id=cid)
                    begins[cid]=r;attempts[kind]+=1
                    if kind in ('training_forward','training_backward','optimizer_update'):
                        c=r['context'];train[(kind,c.get('arm'),c.get('attachment_step'),c.get('microbatch'))]+=1
                    if kind=='sampling_forward':samples[r['context']['request_id']]+=1
                else:
                    audit.check('call_id_returns_once',cid not in ended,call_id=cid)
                    ended[cid]=r;(returns if ev=='CALL_RETURN' else failed)[kind]+=1;seconds[kind]+=r.get('wall_seconds',0.)
            else:meta[(p.name,str(ev) if ev is not None else 'NON_CALL_METADATA_WITHOUT_EVENT')]+=1
    audit.check('every_attempt_has_one_terminal',set(begins)==set(ended),missing_returns=sorted(set(begins)-set(ended)),orphan_returns=sorted(set(ended)-set(begins)))
    for cid,r in begins.items():
        e=ended.get(cid,{})
        audit.check('terminal_matches_actual_call',r['kind']==e.get('kind') and r['context']==e.get('context'),call_id=cid)
    expected=dict(training_forward=32000,training_backward=32000,optimizer_update=4000,precheck_forward=44,precheck_backward=16,sampling=306,sampling_forward=15300)
    audit.check('exact_authorized_actual_calls',dict(attempts)==expected and dict(returns)==expected and not failed,attempts=dict(attempts),returns=dict(returns),failed=dict(failed))
    for arm in ARMS:
        for step in range(1,1001):
            for kind in ('training_forward','training_backward'):
                audit.check('8_actual_microcalls_per_update',all(train[(kind,arm,step,i)]==1 for i in range(8)),arm=arm,step=step,kind=kind)
            audit.check('one_actual_optimizer_call_per_update',train[('optimizer_update',arm,step,None)]==1,arm=arm,step=step)
    audit.check('each_sampling_request_exactly_50_NFE',len(samples)==306 and set(samples.values())=={50},requests=len(samples),histogram=dict(Counter(samples.values())))
    accounting=read(out/'accounting.json')
    for name,actual in (('attempts',attempts),('returns',returns),('failed',failed)):
        audit.check('accounting_matches_ledger_'+name,{k:v for k,v in accounting.get(name,{}).items() if v}==dict(actual))
    audit.check('accounting_pending_empty',not accounting.get('pending'))
    for kind,v in seconds.items():audit.check('accounting_call_seconds',close(v,accounting['wall_seconds_by_kind'].get(kind),rtol=1e-10,atol=1e-7),kind=kind)
    return dict(attempts=dict(attempts),returns=dict(returns),failed=dict(failed),metadata_rows=[dict(file=k[0],event=k[1],rows=v) for k,v in sorted(meta.items())],files=files,sampling_NFE_per_request=dict(samples)),begins


def audit_training(out,audit,call_begins):
    plan=read(out/'train_plan.json');records=plan['records'];protocol=read(out/'protocol.json');lam=protocol['lambda_seam']
    audit.check('fixed_plan_identity',plan['seed']==20261007 and len(records)==8000 and len(plan['object_counts'])==32 and set(plan['object_counts'].values())=={250})
    common_batches={};common_samples={};common_targets={};report={}
    call_identity={(r['kind'],r['context'].get('arm'),r['context'].get('attachment_step'),r['context'].get('microbatch')):r['context'] for r in call_begins.values() if r['kind'] in ('training_forward','training_backward')}
    for arm in ARMS:
        logs=csv_rows(out/'training'/f'{arm}_training_log.csv');expo=csv_rows(out/'training'/f'{arm}_sample_exposure.csv')
        done=[];began=[];checkpoint_events=[]
        for _,r in json_lines(out/'training'/f'{arm}_events.jsonl'):
            if r.get('event')=='UPDATE_DONE':done.append(r)
            elif r.get('event')=='UPDATE_BEGIN':began.append(r)
            elif r.get('event')=='CHECKPOINT_DONE':checkpoint_events.append(r)
        audit.check('exact_1000_completed_updates',len(logs)==len(done)==len(began)==1000 and [int(r['attachment_step']) for r in logs]==list(range(1,1001)) and [r['attachment_step'] for r in done]==list(range(1,1001)) and [r['attachment_step'] for r in began]==list(range(1,1001)),arm=arm)
        audit.check('exact_8000_exposure_rows',len(expo)==8000 and [int(r['sample_index']) for r in expo]==list(range(8000)),arm=arm)
        parents=Counter(r['uid'] for r in expo);audit.check('32parents_250_each',parents==Counter(plan['object_counts']),arm=arm,counts=dict(parents))
        no_gate=no_seam=boundary_active=0;max_weight_error=0.;target_counts=Counter()
        for index,r in enumerate(expo):
            expected=records[index];step=index//8+1;micro=index%8
            audit.check('exposure_matches_frozen_plan',r['arm']==arm and int(r['attachment_step'])==step and all(r[k]==expected[k] for k in ('uid','task_id','alpha','N','K')),arm=arm,sample_index=index)
            audit.check('actual_free_scalar_count',int(r['free_coordinate_count'])==9*(int(r['N'])-int(r['K'])),arm=arm,sample_index=index)
            identity=tuple(r[k] for k in ('uid','task_id','alpha','N','K','t','free_coordinate_count','shared_input_sha256','coupled_input_sha256','ot_cache_key'))
            if arm==ARMS[0]:common_samples[index]=identity
            else:audit.check('crossarm_actual_sample_hashes',identity==common_samples.get(index),arm=arm,sample_index=index)
            for kind in ('training_forward','training_backward'):
                c=call_identity.get((kind,arm,step,micro),{})
                audit.check('actual_call_uses_registered_sample_hash',c.get('sample_index')==index and c.get('shared_input_sha256')==r['shared_input_sha256'],arm=arm,sample_index=index,kind=kind)
        for i,(row,event) in enumerate(zip(logs,done)):
            batch=expo[8*i:8*i+8];details=event['sample_diagnostics'];step=i+1
            den=sum(int(x['free_coordinate_count']) for x in batch)
            audit.check('eight_distinct_parents_four_alpha1',len(batch)==8 and len({x['uid'] for x in batch})==8 and sum(x['alpha']==1. for x in batch)==4,arm=arm,step=step)
            audit.check('effective_batch_FM_denominator',den==int(row['free_coordinate_denominator'])==event['free_coordinate_denominator'],arm=arm,step=step)
            h={key:hashlib.sha256(''.join(str(x[key.replace('batch_','input_')]) for x in batch).encode()).hexdigest() for key in ('shared_batch_sha256','coupled_batch_sha256')}
            audit.check('batch_hash_reconstructed_from_actual_samples',all(row[k]==event[k]==h[k] for k in h),arm=arm,step=step)
            if arm==ARMS[0]:common_batches[step]=tuple(h.values())
            else:audit.check('crossarm_batch_hashes',tuple(h.values())==common_batches.get(step),arm=arm,step=step)
            audit.check('eight_loss_components_logged',len(details)==8,arm=arm,step=step)
            summed_fm=0.;summed_E=0.;summed_gateE=0.;summed_weight=0.;gates=0
            for micro,d in enumerate(details):
                index=i*8+micro;gate=float(batch[micro]['t']>=.5);E=d['unweighted_E'];seam=d['seam'];eligible=seam['counts']['eligible_interface_edges'];candidate=seam['counts']['candidate_interface_edges']
                audit.check('gate_and_target_identity',d['sample_index']==index and d['gate']==gate and seam['gate']==bool(gate) and eligible<=candidate and eligible>=0,arm=arm,sample_index=index)
                target_counts[eligible]+=1
                if arm==ARMS[0]:common_targets[index]=seam['target_sha256']
                else:audit.check('crossarm_seam_target_hash',seam['target_sha256']==common_targets.get(index),arm=arm,sample_index=index)
                expected_weight=lam*gate*E/8 if arm in SEAM else 0.
                error=abs(d['weighted_contribution']-expected_weight);max_weight_error=max(max_weight_error,error)
                audit.check('exact_registered_seam_coefficient',close(d['weighted_contribution'],expected_weight,rtol=3e-7,atol=1e-10),arm=arm,sample_index=index,expected=expected_weight,actual=d['weighted_contribution'])
                if gate==0:
                    no_gate+=1;audit.check('zero_gate_no_auxiliary',d['weighted_contribution']==0.,arm=arm,sample_index=index)
                if eligible==0:
                    no_seam+=1;audit.check('no_seam_zero_not_skipped',seam['status']=='NO_VALID_SEAM' and E==0 and d['weighted_contribution']==0.,arm=arm,sample_index=index)
                if arm in BOUNDARY:
                    b=d.get('boundary');b=b[0] if isinstance(b,list) and len(b)==1 else b
                    audit.check('boundary_diagnostics_present',isinstance(b,dict) and all(k in b for k in ('open_edges','ambiguous_edges','marked_known_corner_fraction','nearly_constant')),arm=arm,sample_index=index)
                summed_fm+=d['FM_contribution'];summed_E+=E/8;summed_gateE+=gate*E/8;summed_weight+=d['weighted_contribution'];gates+=gate/8
            values=dict(FM=summed_fm,seam_unweighted_mean8=summed_E,seam_gated_mean8=summed_gateE,seam_weighted=summed_weight,t_gate_fraction=gates,loss=summed_fm+summed_weight)
            for key,v in values.items():audit.check('logged_loss_arithmetic',close(v,row[key],rtol=3e-7,atol=2e-8) and close(row[key],event[key],rtol=1e-12),arm=arm,step=step,key=key)
            if arm in BOUNDARY:
                keys=('boundary_parameter_norm','boundary_grad_norm_before_clip','boundary_update_norm')
                audit.check('boundary_parameter_gradient_update_finite',all(finite(row[k]) and row[k]>=0 for k in keys),arm=arm,step=step)
                boundary_active+=int(row['boundary_update_norm']>0 and row['boundary_grad_norm_before_clip']>0)
        if arm in BOUNDARY:audit.check('boundary_actual_updates_observed',boundary_active>0 and logs[-1]['boundary_parameter_norm']>0,arm=arm)
        audit.check('all_fixed_200step_recovery_points',{200,400,600,800,1000}.issubset({r['attachment_step'] for r in checkpoint_events}),arm=arm)
        report[arm]=dict(updates=len(done),exposure_rows=len(expo),parents=dict(parents),zero_gate_samples=no_gate,no_valid_seam_samples=no_seam,
            no_valid_seam_observation='not observed; do not infer an empirical zero-edge test' if not no_seam else 'observed and checked',
            positive_boundary_gradient_and_update_steps=boundary_active,max_auxiliary_weight_rounding_error=max_weight_error,
            eligible_seam_count_histogram=dict(sorted(target_counts.items())))
    return report


def audit_checkpoints(out,audit):
    import torch
    rows=[]
    for arm in ARMS:
        pointer=read(out/'training'/f'{arm}_pointer.json');p=Path(pointer['path']);side=read(p.with_suffix('.json'))
        audit.check('endpoint_sidecar_pointer_identity',pointer['checkpoint_sha256']==side['checkpoint_sha256'] and pointer['metadata']==side['metadata'] and p.is_file(),arm=arm)
        # mmap parses state metadata and touches only tiny step tensors. It never
        # restores an optimizer/RNG or transfers tensors to CUDA.
        payload=torch.load(p,map_location='cpu',weights_only=True,mmap=True);m=payload['metadata'];ss=payload['stream_state']
        audit.check('strict_final_checkpoint_metadata',payload['schema']=='chair_boundary_attachment_checkpoint_v1' and payload['update_in_progress'] is False and m['arm']==arm and m['attachment_step']==1000 and m['conditional_total_step']==6000 and m['source_checkpoint_sha256']==START_SHA and m['lambda_hybrid']==.25 and ss['batch_index']==1000 and ss['sample_index']==8000 and ss['arm']==arm,arm=arm)
        groups=payload['optimizer']['param_groups'];states=payload['optimizer']['state'];oldids=groups[0]['params']
        audit.check('strict_optimizer_group_counts',len(groups)==(2 if arm in BOUNDARY else 1) and all((g['lr'],tuple(g['betas']),g['weight_decay'])==(1e-5,(.9,.95),0.) for g in groups),arm=arm)
        audit.check('old_AdamW6000_new1000',all(float(states[i]['step'])==6000 for i in oldids) and (arm not in BOUNDARY or len(groups[1]['params'])==1 and float(states[groups[1]['params'][0]]['step'])==1000),arm=arm)
        audit.check('FP32_complete_AdamW_moments',all(set(('step','exp_avg','exp_avg_sq')).issubset(s) and s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32 and s['exp_avg'].shape==s['exp_avg_sq'].shape for s in states.values()),arm=arm)
        audit.check('boundary_parameter_schema',('e_boundary' in payload['model'])==(arm in BOUNDARY) and (arm not in BOUNDARY or tuple(payload['model']['e_boundary'].shape)==(768,)),arm=arm)
        audit.check('all_parameters_CPU_FP32',all(t.device.type=='cpu' and t.dtype==torch.float32 for t in payload['model'].values()),arm=arm)
        rows.append(dict(arm=arm,path=str(p),bytes=p.stat().st_size,recorded_checkpoint_sha256=side['checkpoint_sha256'],
            actual_parameter_tensors=len(payload['model']),optimizer_states=len(states),old_parameter_step=6000,new_boundary_step=1000 if arm in BOUNDARY else None,
            checkpoint_file_hash_recomputed=False,scope='Metadata/state-layout/actual scalar step audit only; full file content hashes are recorded by checkpoint writer'))
        del payload
    audit.check('audit_never_initializes_CUDA',not torch.cuda.is_initialized())
    return rows


def audit_outputs(out,audit,ledger):
    requests=read(out/'requests.json');scores=csv_rows(out/'per_output_metrics.csv');byid={r['request_id']:r for r in scores};counts=Counter();shared={};file_hashes={}
    def actual_hash(p):
        if p not in file_hashes:file_hashes[p]=digest(p)
        return file_hashes[p]
    audit.check('400_fixed_unique_requests',len(requests)==len({r['request_id'] for r in requests})==400 and Counter(r['label'] for r in requests)==Counter({l:80 for l in LABELS}))
    audit.check('400_unique_score_rows',len(scores)==len(byid)==400 and set(byid)=={r['request_id'] for r in requests})
    for r in requests:
        result=read(out/'outputs'/r['request_id']/'result.json');counts[result['status']]+=1;s=byid[r['request_id']]
        audit.check('score_request_identity',all(s[k]==r[k] for k in ('label','uid','panel','candidate','condition_id','base_request_id')),request_id=r['request_id'])
        if r['status']=='NOT_APPLICABLE':
            audit.check('original_NA_preserved',result['status']==s['status']=='NOT_APPLICABLE' and result.get('model_calls',0)==0,request_id=r['request_id']);continue
        audit.check('actual_output_identity',result['status'] in ('SAMPLED','CACHED') and result['input_sha256']==r['input_sha256'] and result['condition_sha256']==r['condition_sha256'] and s['status']=='PASS' and s['raw_sha256']==result['raw_sha256'],request_id=r['request_id'])
        audit.check('actual_registered_raw_and_input_file_hashes',actual_hash(result['raw_path'])==result['raw_sha256'] and actual_hash(r['input_npz'])==r['input_sha256'] and actual_hash(r['condition_npz'])==r['condition_sha256'],request_id=r['request_id'])
        identity=tuple(result[k] for k in ('input_sha256','condition_sha256','gaussian_sha256','C_array_sha256','known_mask_sha256','N','K'))
        bid=r['base_request_id']
        if bid not in shared:shared[bid]=identity
        else:audit.check('all_five_labels_same_actual_generation_input',shared[bid]==identity,base_request_id=bid,label=r['label'])
        m=result['metadata'];audit.check('Euler50_and_C_preservation',m['nfe']==50 and m['accepted_states']==51 and m['known_all51_bitwise'] and m['known_state_checks']==51 and m['solver']=='original_Euler50' and m['forward_dtype']=='bfloat16' and m['integration_dtype']=='float32',request_id=r['request_id'])
        expected_calls=0 if r['label']=='START_HYBRID' else 50
        audit.check('output_model_calls_match_ledger',result['model_calls']==expected_calls and ledger['sampling_NFE_per_request'].get(r['request_id'],0)==expected_calls,request_id=r['request_id'])
        audit.check('no_new_output_contract_failure',s.get('C_preserved') is True and s.get('N_preserved') is True and s.get('finite') is True,request_id=r['request_id'])
    audit.check('304_new76cache20NA',counts==Counter(SAMPLED=304,CACHED=76,NOT_APPLICABLE=20),actual=dict(counts))
    replay=read(out/'replays/summary.json')
    audit.check('two_initial_exact_replays',replay['status']=='PASS' and replay['replays']==2 and replay['NFE']==100 and len(replay['rows'])==2 and all(r['identity_replay']['bitwise_equal'] and r['identity_replay']['max_abs']==0 for r in replay['rows']))
    return requests,scores,dict(counts)


def audit_arithmetic(out,audit,requests,scores,*,write_tables=True):
    byid={r['request_id']:r for r in scores};groups=defaultdict(list)
    for r in requests:groups[(r['label'],r['panel'],r['uid'])].append(r)
    parents={};parent_csv=[]
    original_parents={(r['label'],r['panel'],r['uid']):r for r in read(out/'aggregation/parent_means.json')}
    for key,reqs in groups.items():
        applicable=[r for r in reqs if r['status']!='NOT_APPLICABLE'];record={}
        for metric in METRICS:
            vals=[byid[r['request_id']].get(metric) for r in applicable if byid[r['request_id']]['status']=='PASS'];vals=[float(v) for v in vals if finite(v)]
            v=mean(vals);complete=bool(applicable) and len(vals)==len(applicable);record[metric]=(v,len(vals),complete)
            logged=original_parents[key]
            audit.check('independent_parent_mean',close(v,logged[metric],rtol=1e-12) and len(vals)==logged[metric+'_n'] and complete==logged[metric+'_complete'],label=key[0],panel=key[1],uid=key[2],metric=metric)
            parent_csv.append(dict(label=key[0],panel=key[1],uid=key[2],metric=metric,mean=v,n=len(vals),applicable=len(applicable),registered=len(reqs),complete=complete))
        parents[key]=record
    panels=read(out/'aggregation/panel_means.json')
    for row in panels:
        group=[(k,v) for k,v in parents.items() if k[:2]==(row['label'],row['panel'])]
        eligible=[(k,v) for k,v in group if any(r['status']!='NOT_APPLICABLE' for r in groups[k])]
        vals=[v[row['metric']][0] for k,v in eligible if v[row['metric']][2]]
        audit.check('independent_equal_parent_panel_mean',close(mean(vals),row['mean'],rtol=1e-12) and row['n']==len(vals) and row['expected']==len(eligible) and row['registered_parents']==len(group),label=row['label'],panel=row['panel'],metric=row['metric'])
        if row['panel']=='P1':audit.check('P1_denominator_never_shrinks',row['expected']==row['registered_parents']==8,label=row['label'],metric=row['metric'])
    contrast_rows=read(out/'aggregation/paired_parent_differences.json')
    for r in contrast_rows:
        a=parents[(r['treatment'],r['panel'],r['uid'])][r['metric']];b=parents[(r['reference'],r['panel'],r['uid'])][r['metric']]
        expected=a[0]-b[0] if a[2] and b[2] else None
        audit.check('independent_five_parent_contrasts',close(expected,r['difference'],rtol=1e-11),contrast=r['contrast'],panel=r['panel'],uid=r['uid'],metric=r['metric'])
    bybase=defaultdict(dict);base_requests={}
    for r in requests:bybase[r['base_request_id']][r['label']]=byid[r['request_id']];base_requests[r['base_request_id']]=r
    noise=[];interactions={};parent_interactions=defaultdict(list)
    for bid,arms in bybase.items():
        r=base_requests[bid];base=dict(base_request_id=bid,panel=r['panel'],uid=r['uid'],condition_id=r['condition_id'],candidate=r['candidate'])
        for metric in METRICS:
            for name,treatment,reference in CONTRASTS:
                a=arms[treatment].get(metric);b=arms[reference].get(metric)
                valid=finite(a) and finite(b) and arms[treatment]['status']==arms[reference]['status']=='PASS'
                noise.append(dict(base,metric=metric,contrast=name,treatment=treatment,reference=reference,treatment_value=a,reference_value=b,difference=float(a-b) if valid else None,status='PASS' if valid else 'NA'))
            labels=('A_FM_END','B_BOUNDARY_END','C_SEAM_END','D_BOUNDARY_SEAM_END');values=[arms[l].get(metric) for l in labels]
            valid=all(finite(v) for v in values) and all(arms[l]['status']=='PASS' for l in labels)
            value=values[3]-values[1]-values[2]+values[0] if valid else None
            interactions[(bid,metric)]=value;parent_interactions[(r['panel'],r['uid'],metric)].append(value)
            noise.append(dict(base,metric=metric,contrast='D-B-C+A',A=values[0],B=values[1],C=values[2],D=values[3],difference=value,status='PASS' if valid else 'NA'))
    for row in read(out/'aggregation/interaction_per_output.json'):
        audit.check('independent_per_noise_interaction',close(interactions[(row['base_request_id'],row['metric'])],row['interaction'],rtol=1e-11),base_request_id=row['base_request_id'],metric=row['metric'])
    for row in read(out/'aggregation/interaction_per_parent.json'):
        values=parent_interactions[(row['panel'],row['uid'],row['metric'])];vals=[v for v in values if finite(v)]
        audit.check('independent_parent_interaction',close(mean(vals),row['mean'],rtol=1e-11) and row['n']==len(vals) and row['expected']==len(values),panel=row['panel'],uid=row['uid'],metric=row['metric'])
    if write_tables:
        write_csv(out/'paired_noise_differences.csv',noise)
        write_csv(out/'independent_parent_means.csv',parent_csv)
    return dict(metrics=len(METRICS),parent_groups=len(parents),parent_metric_rows=len(parent_csv),paired_noise_and_interaction_rows=len(noise),
        independence='Direct per_output_metrics.csv arithmetic; never imports or calls aggregation.py; all registered candidates enter parent denominators; panels stay separate')


def run(out=OUT):
    out=Path(out);ready,states,aggregation_status=_readiness(out)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    if not ready:
        result=dict(status='PENDING',utc=now,training=states,aggregation_status=aggregation_status,
            reason='Final audit waits for all four fixed endpoints and complete registered scoring; active files are not treated as final evidence',model_calls=0)
        write_json(out/'independent_audit.json',result);return result
    audit=Audit();sections={};errors=[]
    try:
        sections['ledger'],begins=audit_ledger(out,audit)
        sections['training']=audit_training(out,audit,begins)
        sections['checkpoints']=audit_checkpoints(out,audit)
        requests,scores,sections['generation']=audit_outputs(out,audit,sections['ledger'])
        sections['aggregation']=audit_arithmetic(out,audit,requests,scores)
    except BaseException as exc:errors.append(dict(error=repr(exc),traceback=traceback.format_exc()))
    result=dict(status='PASS' if not audit.failures and not errors else 'BLOCKED',utc=now,sections=sections,
        check_counts=dict(audit.counts),total_checks=sum(audit.counts.values()),failed_checks=audit.failures,errors=errors,
        source_sha256=digest(__file__),real_model_forwards=0,backward_calls=0,optimizer_updates=0,noise_draws=0,OT_calls=0,geometry_score_calls=0,
        limitations='Audits recorded actual tensor hashes and output identities; does not regenerate tensors or recompute geometry. Final checkpoints are mmap metadata/state-layout inspected, not fully rehashed. This is an arithmetic/accounting audit, not a claim of scientific success.')
    write_json(out/'independent_audit.json',result);return result


def smoke(out=OUT):
    """Inspect only the already committed first five A rows, never final verdict."""
    out=Path(out);rows=csv_rows(out/'training/A_FM_training_log.csv')[:5];exposure=csv_rows(out/'training/A_FM_sample_exposure.csv')[:40]
    checks=[]
    for i,r in enumerate(rows):
        batch=exposure[i*8:(i+1)*8]
        checks.append(dict(step=int(r['attachment_step']),PASS=len(batch)==8 and len({x['uid'] for x in batch})==8 and sum(x['alpha']==1 for x in batch)==4 and sum(int(x['free_coordinate_count']) for x in batch)==int(r['free_coordinate_denominator']) and r['FM']==r['loss'] and r['seam_weighted']==0))
    result=dict(status='PASS' if len(checks)==5 and all(r['PASS'] for r in checks) else 'BLOCKED',scope='first five committed A_FM CSV rows only; final audit remains PENDING',checks=checks,model_calls=0)
    write_json(out/'independent_audit_smoke.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));p.add_argument('--smoke',action='store_true');a=p.parse_args()
    result=smoke(a.out) if a.smoke else run(a.out)
    print(json.dumps({k:result[k] for k in ('status','total_checks','errors','reason') if k in result},indent=2))
