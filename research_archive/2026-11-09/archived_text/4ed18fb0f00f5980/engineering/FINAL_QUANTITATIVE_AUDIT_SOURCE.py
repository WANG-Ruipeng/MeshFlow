"""Independent arithmetic audit of existing final scores; no review access.

Uses stored numeric metric definitions directly, not the main aggregation code.
No models, geometry rescoring, image review, or private mappings are loaded.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import csv
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
from r2.pilots.window_policy_rl_v1.bootstrap import bind,OUT
D,E=bind()
EV=OUT/'evaluation';DEST=EV/'analysis'
LABELS=('FULL_DET','FULL_STOCH','RL_STOCH')
CONTRASTS=(('RL_STOCH','FULL_STOCH','PRIMARY'),('FULL_STOCH','FULL_DET','STOCHASTIC_DECODING_REFERENCE'),('RL_STOCH','FULL_DET','DETERMINISTIC_REFERENCE'))
METRIC_KEYS={'CF_illegal_fraction':'CF_illegal_affected_free_fraction','FF_illegal_fraction':'FF_illegal_affected_free_fraction',
    'I_lower':'ConflictCov_C','I_upper':'ConflictCov_C_upper','free_near_degenerate_fraction':'free_near_degenerate_fraction',
    'free_small_angle_fraction':'free_minimum_angle_lt5_fraction','free_surface_area':'free_total_surface_area',
    'actual_free_faces':'actual_free_faces','deleted_free_faces':'deleted_free_faces',
    'CF_unresolved_pairs':'CF_unresolved_or_degenerate_pair_count','FF_unresolved_pairs':'FF_unresolved_or_degenerate_pair_count'}
NUMERIC=tuple(METRIC_KEYS)+('original_free_faces','free_retained_fraction','legal_CF_edges','legal_CF_length','reached_free_area_fraction')
PP={'CF_illegal_fraction','FF_illegal_fraction','I_lower','I_upper','free_retained_fraction','free_near_degenerate_fraction','free_small_angle_fraction','reached_free_area_fraction'}
SOURCES={};HASHES={};CHECKS=Counter();ERRORS=[]

def check(ok,key,detail=None):
    CHECKS[key]+=1
    if not ok:ERRORS.append(dict(check=key,detail=detail))

def blob(path):
    p=D.resolve(path);data=p.read_bytes();HASHES[str(p)]=hashlib.sha256(data).hexdigest()
    SOURCES[str(p)]={'sha256':HASHES[str(p)],'bytes':len(data)}
    return data

def read(path):return json.loads(blob(path).decode('utf-8-sig'))
def table(path):return list(csv.DictReader(io.StringIO(blob(path).decode('utf-8-sig'))))

def sha(path):
    p=D.resolve(path);key=str(p)
    if key not in HASHES:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
        HASHES[key]=h.hexdigest();SOURCES[key]={'sha256':HASHES[key],'bytes':p.stat().st_size}
    return HASHES[key]

def num(x):
    if x is None or x=='':return None
    v=float(x)
    return v if math.isfinite(v) else None

def equal(a,b):
    if a is None or b is None or a=='' or b=='':return num(a) is None and num(b) is None
    return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10)

def numeric(values):
    supplied=list(values);xs=sorted(float(v) for v in supplied if v is not None and math.isfinite(float(v)))
    if not xs:return dict(n=0,missing=len(supplied),mean=None,minimum=None,maximum=None,p90=None,median=None,sum=None)
    position=(len(xs)-1)*.9;lower=math.floor(position);upper=math.ceil(position)
    p90=xs[lower]+(xs[upper]-xs[lower])*(position-lower)
    return dict(n=len(xs),missing=len(supplied)-len(xs),mean=statistics.fmean(xs),minimum=xs[0],maximum=xs[-1],
        p90=p90,median=statistics.median(xs),sum=sum(xs))

def aggregate(rows,**identity):
    r=dict(identity,outputs=len(rows),legal_CF_outputs=sum(x['has_legal_CF'] for x in rows),
        I_status_counts=dict(Counter(x['I_status'] for x in rows)),
        topology_status_counts=dict(Counter(x['topology_status'] for x in rows)),
        source_C_connectivity_status_counts=dict(Counter(x['source_C_connectivity_status'] for x in rows)))
    r['legal_CF_output_rate']=r['legal_CF_outputs']/len(rows)
    for key in NUMERIC:r.update({key+'_'+k:v for k,v in numeric(x[key] for x in rows).items()})
    return r

def compare_numeric(ours,other,fields,context):
    for field in fields:check(field in other and equal(ours[field],other.get(field)),'MAIN_NUMERIC_EQUAL',dict(context,field=field,ours=ours[field],main=other.get(field)))

def phase_accounting(manifest):
    result={}
    for phase,expected in (('calibration',256),('evaluation',132)):
        counts=Counter();workers=[]
        for path in sorted((OUT/'dispatch'/phase).glob('sample_*/raw_index.json')):
            index=read(path);completion=read(path.parent/'completion.json')
            event_rows=[json.loads(line) for line in blob(path.parent/'calls.jsonl').splitlines() if line.strip()]
            c=Counter((r['event'],r.get('kind','')) for r in event_rows)
            n=index['requested_trajectories']
            check(index['status']=='COMPLETE' and completion['status']=='COMPLETE' and index['actual_rollout_attempts']==n
                and index['actual_sampling_forwards']==50*n and index['hidden_validation_forwards']==0,'PHASE_SAMPLER_COMPLETE',str(path))
            for event,k,nexpected in (('ROLLOUT_ATTEMPT','',n),('ROLLOUT_COMPLETE','',n),('FORWARD_START','sampling_forward',50*n),('FORWARD_RETURN','sampling_forward',50*n)):
                check(c[event,k]==nexpected,'PHASE_SAMPLER_EVENT_COUNT',str(path))
            check(not (path.parent/'failure.json').exists(),'NO_CAL_FINAL_SAMPLE_FAILURE',str(path))
            counts.update(c);workers.append(str(path))
        d0=read(OUT/'dispatch'/phase/'d0/completion.json')
        check(d0['status']=='COMPLETE' and d0['new_D0_forwards']==expected and d0['model_calls_this_invocation']==expected,'PHASE_D0_COMPLETE',phase)
        check(counts['ROLLOUT_COMPLETE','']==expected,'PHASE_TOTAL_ROLLOUTS',phase)
        labels=('FULL_DET','S005','S010','S020') if phase=='calibration' else LABELS
        requests=manifest['calibration_requests'] if phase=='calibration' else manifest['evaluation_requests']
        receipt_raw=receipt_d0=0
        for label in labels:
            for q in requests:
                rid=q['request_id'];raw=read(OUT/phase/label/'raw'/rid/'receipt.json');prop=read(OUT/phase/label/'proposal'/rid/'receipt.json')
                check(raw['status']=='COMPLETE' and raw['new_sampling_F']==50 and raw['new_rollouts']==1,'PHASE_RAW_RECEIPT_COUNTS',(phase,label,rid))
                check(prop['status']=='PASS' and prop['model_forwards']==1 and prop['raw_sha256']==raw['raw_sha256'],'PHASE_D0_RECEIPT_COUNTS',(phase,label,rid))
                receipt_raw+=raw['new_sampling_F'];receipt_d0+=prop['model_forwards']
        result[phase]=dict(rollouts=expected,worker_count=len(workers),sampling_F_attempts=counts['FORWARD_START','sampling_forward'],
            sampling_F_returns=counts['FORWARD_RETURN','sampling_forward'],D0_F_returns=d0['new_D0_forwards'],
            receipt_sampling_F=receipt_raw,receipt_D0_F=receipt_d0,hidden_validation_forwards=0,sample_worker_indices=workers)
    training=read(OUT/'engineering/TRAINING_EXECUTION_AUDIT.json')
    check(training['status']=='PASS','TRAINING_AUDIT_PASS')
    result['training']=training['actual_counts']
    result['registered_three_phase_model_F_total']=training['actual_counts']['total_model_F_returns']+sum(result[p]['sampling_F_returns']+result[p]['D0_F_returns'] for p in ('calibration','evaluation'))
    result['scope']='Registered calibration, training and final ledger/receipts only; includes two first-update zero-LoRA validation forwards. Calibration/final indices explicitly report zero hidden-validation forwards. No unlogged out-of-scope invocation is claimed to be ruled out.'
    check(result['registered_three_phase_model_F_total']==66382,'REGISTERED_TOTAL66382')
    return result


def run():
    check('torch' not in sys.modules,'NO_MODEL_RUNTIME')
    manifest=read(OUT/'inputs/manifest.json');registration=read(OUT/'registration.json');gate=read(OUT/'analysis/decode_gate.json')
    final=read(OUT/'training/final.json');quant=read(DEST/'quantitative.json');qs=manifest['evaluation_requests']
    check(quant['status']=='COMPLETE' and quant['outputs']==132 and quant['stage_scores']==264,'MAIN_QUANTITATIVE_COMPLETE')
    check(len(qs)==44 and len({q['request_id'] for q in qs})==44 and Counter(q['split'] for q in qs)==Counter(TRAIN=32,DEV=12),'44_FROZEN_REQUESTS')
    check(sha(OUT/'inputs/manifest.json')==registration['manifest_sha256'],'MANIFEST_FROZEN')
    check(final['status']=='COMPLETE' and final['batches']==64 and final['rollouts']==512,'FIXED_LAST_ENDPOINT')
    check(sha(final['checkpoint_path'])==final['checkpoint_sha256'],'FINAL_ADAPTER_FILE_HASH')
    rows=[];bindings=[];definitions={};receipt_count=0
    for label in LABELS:
        for q in qs:
            rid=q['request_id'];base=EV/label;expected_sigma=0. if label=='FULL_DET' else gate['selected_sigma'];mode='current' if label=='RL_STOCH' else 'reference'
            raw=read(base/'raw'/rid/'receipt.json');post=read(base/'post'/rid/'receipt.json');proposal=read(base/'proposal'/rid/'receipt.json');receipt_count+=1
            check(raw['status']=='COMPLETE' and raw['label']==label and raw['request_id']==rid and raw['model_mode']==mode and raw['sigma']==expected_sigma,'RAW_LABEL_POLICY_IDENTITY',(label,rid))
            check(all(raw[k]==q[k] for k in ('input_sha256','C_sha256','Gaussian_sha256','transition_Gaussian_sha256','condition_id','N','K')),'RAW_FROZEN_INPUT_IDENTITY',(label,rid))
            check(raw['registration_sha256']==sha(OUT/'registration.json') and raw['model_identity']['base_file_sha256']==manifest['source_binding']['sha256'],'RAW_SOURCE_REGISTRATION_BINDING',(label,rid))
            if label=='RL_STOCH':check(raw['model_identity']['adapter_checkpoint_sha256']==final['checkpoint_sha256'],'RL_FIXED_LAST_ADAPTER',(label,rid))
            else:check(raw['model_identity'].get('adapter_modules')==0 and not raw['model_identity'].get('adapter_checkpoint'),'REFERENCE_PHYSICAL_NO_ADAPTER',(label,rid))
            check(raw['sampling']['known_all51_bitwise'] and raw['sampling']['nfe']==50 and raw['sampling']['new_random_draws']==0,'RAW_RUNTIME_RECEIPT',(label,rid))
            for key in ('raw','ply','result','window_buffer'):check(sha(raw[key+'_path'])==raw[key+'_sha256'],'RAW_ARTIFACT_HASH',(label,rid,key))
            check(sha(q['input_path'])==q['input_sha256'] and sha(q['transition_noise_path'])==q['transition_noise_sha256'],'FROZEN_NOISE_FILES',(label,rid))
            check(post['status']=='COMPLETE' and post['label']==label and post['request_id']==rid and post['input_sha256']==q['input_sha256'],'POST_INPUT_IDENTITY',(label,rid))
            check(proposal['status']=='PASS' and proposal['model_forwards']==1 and proposal['raw_sha256']==raw['raw_sha256'],'D0_ONCE_RAW_BINDING',(label,rid))
            bindings.append(dict(label=label,request_id=rid,condition_id=q['condition_id'],split=q['split'],sigma=expected_sigma,
                raw_sha256=raw['raw_sha256'],input_sha256=q['input_sha256'],Gaussian_sha256=q['Gaussian_sha256'],
                transition_Gaussian_sha256=q['transition_Gaussian_sha256'],adapter_sha256=raw['model_identity'].get('adapter_checkpoint_sha256')))
            for stage in ('RAW','POST'):
                sp=base/'scores'/rid/stage/'score.json';score=read(sp);identity=score['identity'];metrics=score['metrics'];top=score['topology']
                check(all(identity[k]==v for k,v in (('request_id',rid),('condition_id',q['condition_id']),('label',label),('stage',stage),('N',q['N']),('K',q['K']))),'SCORE_IDENTITY',(label,rid,stage))
                check(score['status']=='PASS' and score['C_preserved_exact'] is True and score['original_definition_unchanged'] is True,'ORIGINAL_SCORE_PASS',(label,rid,stage))
                for path,expected in identity['input_hashes'].items():check(sha(path)==expected,'SCORE_INPUT_FILE_HASH',(label,rid,stage,path))
                definitions.update(identity['definition_hashes'])
                r=dict(label=label,request_id=rid,condition_id=q['condition_id'],parent_id=q['parent_id'],split=q['split'],stage=stage,
                    score_path=str(sp),score_sha256=sha(sp),I_status=metrics.get('ConflictCov_C_status','UNKNOWN'),
                    topology_status=top['status'],source_C_connectivity_status=score.get('source_C_connectivity_status','UNKNOWN'))
                r.update({key:num(metrics.get(source)) for key,source in METRIC_KEYS.items()})
                r.update(original_free_faces=q['N']-q['K'],free_retained_fraction=r['actual_free_faces']/(q['N']-q['K']),
                    legal_CF_edges=top['C_F_valid_edge_count'],legal_CF_length=top['C_F_valid_edge_total_length'],
                    has_legal_CF=int(top['C_F_valid_edge_count']>0),reached_free_area_fraction=num(top.get('reached_free_area_fraction')))
                check(r['actual_free_faces']==r['original_free_faces']-r['deleted_free_faces'],'ACTUAL_FREE_DENOMINATOR',(label,rid,stage))
                rows.append(r)
    for path,expected in definitions.items():check(sha(path)==expected,'ORIGINAL_SCORER_SOURCE_HASH',path)
    model_groups=[];condition_groups=[]
    for label in LABELS:
        for stage in ('RAW','POST'):
            own=[r for r in rows if r['label']==label and r['stage']==stage]
            for split in ('ALL','TRAIN','DEV'):
                sub=own if split=='ALL' else [r for r in own if r['split']==split]
                model_groups.append(aggregate(sub,label=label,stage=stage,split=split))
            for cid in sorted({r['condition_id'] for r in own}):
                sub=[r for r in own if r['condition_id']==cid]
                check(len(sub)==4,'FOUR_NOISES_PER_C',(label,stage,cid))
                condition_groups.append(aggregate(sub,label=label,stage=stage,condition_id=cid,split=sub[0]['split'],parent_id=sub[0]['parent_id']))
    paired=[];paired_groups=[];index={(r['label'],r['stage'],r['request_id']):r for r in rows}
    for candidate,baseline,role in CONTRASTS:
        for stage in ('RAW','POST'):
            for q in qs:
                rid=q['request_id'];a=index[candidate,stage,rid];b=index[baseline,stage,rid]
                r=dict(candidate_label=candidate,baseline_label=baseline,comparison_role=role,stage=stage,request_id=rid,condition_id=q['condition_id'],parent_id=q['parent_id'],split=q['split'],
                    legal_CF_gained=int(a['has_legal_CF']==1 and b['has_legal_CF']==0),legal_CF_lost=int(a['has_legal_CF']==0 and b['has_legal_CF']==1),
                    baseline_has_legal_CF=b['has_legal_CF'],candidate_has_legal_CF=a['has_legal_CF'])
                for key in NUMERIC:
                    av,bv=a[key],b[key];r['candidate_'+key]=av;r['baseline_'+key]=bv
                    r[key+'_delta']=None if av is None or bv is None else av-bv
                    if key in PP:r[key+'_delta_pp']=None if r[key+'_delta'] is None else 100*r[key+'_delta']
                paired.append(r)
            own=[r for r in paired if r['candidate_label']==candidate and r['baseline_label']==baseline and r['stage']==stage]
            groups=[('SPLIT',split,own if split=='ALL' else [r for r in own if r['split']==split]) for split in ('ALL','TRAIN','DEV')]
            groups += [('CONDITION',cid,[r for r in own if r['condition_id']==cid]) for cid in sorted({r['condition_id'] for r in own})]
            for scope,name,sub in groups:
                r=dict(candidate_label=candidate,baseline_label=baseline,comparison_role=role,stage=stage,scope=scope,group=name,
                    paired_requests=len(sub),legal_CF_gained=sum(x['legal_CF_gained'] for x in sub),legal_CF_lost=sum(x['legal_CF_lost'] for x in sub))
                for key in NUMERIC:
                    r.update({key+'_delta_'+stat:v for stat,v in numeric(x[key+'_delta'] for x in sub).items()})
                    if key in PP:r[key+'_mean_delta_pp']=None if r[key+'_delta_mean'] is None else 100*r[key+'_delta_mean']
                paired_groups.append(r)
    mainrows=table(DEST/'per_output_stage.csv');mainindex={(r['label'],r['stage'],r['request_id']):r for r in mainrows}
    check(len(mainindex)==len(mainrows)==264,'MAIN264_ROW_COVERAGE')
    for r in rows:
        ctx={k:r[k] for k in ('label','stage','request_id')};main=mainindex[r['label'],r['stage'],r['request_id']]
        compare_numeric(r,main,NUMERIC+('has_legal_CF',),ctx)
        for key in ('condition_id','split','score_sha256','I_status','topology_status','source_C_connectivity_status'):check(r[key]==main[key],'MAIN_ID_STATUS_EQUAL',dict(ctx,field=key))
    for ours,mainlist,keys in ((model_groups,quant['model_summaries'],('label','stage','split')),
            (condition_groups,table(DEST/'per_condition_stage.csv'),('label','stage','condition_id')),
            (paired,table(DEST/'paired_request_stage.csv'),('candidate_label','baseline_label','stage','request_id')),
            (paired_groups,quant['paired_summaries'],('candidate_label','baseline_label','stage','scope','group'))):
        lookup={tuple(r[k] for k in keys):r for r in mainlist}
        check(len(lookup)==len(ours),'MAIN_GROUP_COVERAGE',keys)
        for r in ours:
            identity=tuple(r[k] for k in keys);main=lookup[identity]
            fields=[k for k,v in r.items() if isinstance(v,(int,float)) or v is None]
            compare_numeric(r,main,fields,{'identity':identity})
    tails=[]
    for candidate,baseline,role in CONTRASTS:
        for split in ('TRAIN','DEV'):
            sub=[r for r in paired if r['candidate_label']==candidate and r['baseline_label']==baseline and r['stage']=='POST' and r['split']==split]
            for key in ('CF_illegal_fraction','FF_illegal_fraction','I_lower','I_upper','free_small_angle_fraction','legal_CF_length'):
                valid=[r for r in sub if r[key+'_delta'] is not None]
                for rank,r in enumerate(sorted(valid,key=lambda x:(-x[key+'_delta'],x['request_id'])),1):
                    tails.append(dict(candidate_label=candidate,baseline_label=baseline,comparison_role=role,split=split,metric=key,
                        rank_largest_delta_first=rank,request_id=r['request_id'],condition_id=r['condition_id'],
                        baseline=r['baseline_'+key],candidate=r['candidate_'+key],delta=r[key+'_delta'],
                        delta_pp=r.get(key+'_delta_pp'),direction='Positive is longer, not automatically better' if key=='legal_CF_length' else 'Positive is higher affected fraction'))
    for path in (Path(__file__),OUT.parents[1]/'r2/pilots/window_policy_rl_v1/final_analysis.py',OUT.parents[1]/'r2/pilots/terminal_reward_nft_v1/final_analysis.py'):
        sha(path)
    budget=phase_accounting(manifest)
    check('torch' not in sys.modules,'NO_MODEL_RUNTIME_AT_END')
    result=dict(version='window_policy_rl_final_quantitative_independent_audit_v1',status='PASS' if not ERRORS else 'FAIL',checks=dict(CHECKS),violations=ERRORS,
        request_count=44,output_count=receipt_count,stage_score_count=len(rows),phase_forward_accounting=budget,model_summaries=model_groups,condition_summaries=condition_groups,
        paired_summaries=paired_groups,per_output_stage=rows,paired_request_stage=paired,all_POST_geometry_tails=tails,receipt_bindings=bindings,
        original_scorer_source_hashes=definitions,input_sources=SOURCES,model_calls=0,checkpoint_tensor_loads=0,geometry_rescores=0,
        actual_review_accessed=False,private_mapping_accessed=False,
        limitations=['Numerical geometry only; no adoption, body-completeness, or joint-success inference.',
            'CF/FF and small-angle fractions retain actual stage free-face denominators; per-output macro means, no zero-fill.',
            'I point/lower and upper/status remain distinct; unknown/missing values are never treated as zero.',
            'Length is the original scorer coordinate length, not a percentage or a new normalized metric.',
            'TRAIN has eight C and DEV has three C, each four frozen noises; descriptive small panel, not universal generalization.',
            'Receipt C51/forward checks are runtime evidence; this audit hashes existing outputs but does not rerun sampling or geometry.'])
    D.write(OUT/'engineering/FINAL_QUANTITATIVE_AUDIT.json',result)
    compact_models=[{k:r[k] for k in ('label','stage','split','outputs','legal_CF_outputs','legal_CF_edges_sum','legal_CF_length_sum','CF_illegal_fraction_mean','FF_illegal_fraction_mean','I_lower_mean','I_upper_mean','free_small_angle_fraction_mean','free_retained_fraction_mean')} for r in model_groups if r['stage']=='POST']
    compact_pairs=[{k:r[k] for k in ('candidate_label','baseline_label','group','paired_requests','legal_CF_gained','legal_CF_lost','legal_CF_length_delta_sum','CF_illegal_fraction_mean_delta_pp','FF_illegal_fraction_mean_delta_pp','I_lower_mean_delta_pp','I_upper_mean_delta_pp','free_small_angle_fraction_mean_delta_pp','free_retained_fraction_mean_delta_pp')} for r in paired_groups if r['stage']=='POST' and r['scope']=='SPLIT']
    print(json.dumps(dict(status=result['status'],violations=ERRORS,phase_forward_accounting=budget,audit_sha256=D.sha(OUT/'engineering/FINAL_QUANTITATIVE_AUDIT.json'),models=compact_models,pairs=compact_pairs,
        primary_tails=[r for r in tails if r['comparison_role']=='PRIMARY' and r['rank_largest_delta_first']==1]),sort_keys=True))

if __name__=='__main__':run()
