"""Read-only SEAM training/evaluation closure; no model or scorer execution."""
from __future__ import annotations
import argparse,collections,json,math,statistics,time
from pathlib import Path
from . import common as C

ARM='SEAM1500'
DEST=C.OUT/'closeout'


def require(value,message):
    if not value:raise ValueError(message)


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def verify(book):
    for path,digest in book.items():require(C.sha(path)==digest,'Changed file: '+str(path))


def calls(rows,updates=1500):
    expected=[]
    for step in range(1,updates+1):
        expected.extend((step,mi,k) for mi in range(8) for k in ('train_forward','train_backward'))
        expected.append((step,None,'optimizer_update'))
    require(len(rows)==2*len(expected),'Scientific ledger length differs')
    counts=collections.Counter();seconds=collections.Counter()
    for i,(step,mi,kind) in enumerate(expected):
        start,end=rows[2*i:2*i+2]
        require(start['event']=='CALL_BEGIN' and end['event']=='CALL_DONE','Failed or incomplete scientific call')
        require(start['call_id']==end['call_id']==i+1,'Scientific call identity differs')
        require(start['arm']==end['arm']==ARM and start['kind']==end['kind']==kind,'Scientific arm/kind differs')
        require(start['context']==end['context'] and start['attempt_id']==end['attempt_id'],'Call context differs')
        context=start['context']
        require(context['stage_step']==context['local_step']==context['seam_total_step']==step,'SEAM cursor differs')
        require(context['global_step']==22500+step and context['curriculum_step']==4500+step
                and context['native_total_step']==10500 and context.get('microbatch')==mi,'Cumulative cursor differs')
        require(math.isfinite(end['seconds']) and end['seconds']>=0,'Invalid timing')
        counts[kind]+=1;seconds[kind]+=end['seconds']
    return dict(successful=dict(counts),attempted=dict(counts),failed=0,pending=0,
                head_calls=0,probe_calls=0,replayed_updates=0,scientific_call_seconds=dict(seconds))


def training():
    done=DEST/'training.json'
    if done.exists():
        result=C.read(done);verify(result['files']);return result
    folder=C.OUT/'training'/ARM;state=C.read(folder/'status.json')
    require(state['status']=='COMPLETE','Training must be COMPLETE before closure')
    require((state['local_step'],state['global_step'],state['seam_total_step'],state['native_total_step'])
            ==(1500,24000,1500,10500),'Wrong fixed endpoint')
    reg=C.read(C.OUT/'training_registration.json')
    for name in ('training_code_hashes','gate_hashes'):verify(reg[name])
    for name in ('manifest','eval_manifest'):
        require(C.sha(reg[name])==reg[name+'_sha256'],'Frozen manifest changed')
    require(C.sha(reg['source_checkpoint'])==reg['source_checkpoint_sha256'],'Source checkpoint changed')
    schedule=C.read(reg['schedules'][ARM]['path'])
    require(C.sha(reg['schedules'][ARM]['path'])==reg['schedules'][ARM]['sha256'],'Schedule changed')
    init=C.read(folder/'branch_initialization.json')
    for field in ('source_checkpoint_sha256','source_generator_sha256','source_optimizer_sha256','source_rng_sha256'):
        require(init[field]==reg[field],'Initial live/AdamW/RNG identity differs')
    require(init['inherited_optimizer'] and init['inherited_rng'] and not init['fresh_AdamW'],'Invalid continuation policy')
    ledger=calls(jsonl(folder/'calls.jsonl'))
    for kind in ('attempted','successful'):require(state['calls'][kind]==ledger[kind],'Status/ledger budget mismatch')
    require(not state['calls']['failed'] and not state['calls']['pending_call_ids'],'Failed/pending calls')
    require(not list((folder/'attempts').glob('*/failure.json')),'Scientific failure requires separate review')
    logs=jsonl(folder/'training_log.jsonl');exposure=jsonl(folder/'exposure.jsonl')
    require(len(logs)==1500 and len(exposure)==12000,'Missing update/exposure logs')
    groups=collections.Counter();parents=collections.Counter();loss_bins=[]
    for entry,log in zip(schedule['steps'],logs):
        step=entry['stage_step'];actual=exposure[(step-1)*8:step*8]
        require(log['stage_step']==step and log['objective']==log['FM'],'Unexpected objective')
        for expected,row in zip(entry['samples'],actual):
            require(all(row[k]==v for k,v in expected.items()),'Frozen instance mismatch')
            require(row['head_forward_called'] is False and row['mechanism']['policy']=='SOURCE_OWNER_SEAM_OT','Wrong training mechanism')
            groups[row['replay_group']]+=1;parents[row['parent_id']]+=1
        den=sum(r['all_free_scalar_count'] for r in actual)
        sse=sum(r['all_free_velocity_sse'] for r in actual)
        require(den==log['free_coordinate_denominator'],'Free-coordinate denominator differs')
        require(math.isclose(sse/den,log['FM'],rel_tol=4e-6,abs_tol=1e-7),'Actual MSE accounting differs')
    require(dict(groups)=={'OLD':9000,'FOCUS':1500,'HIGH':1500},'Group exposure budget differs')
    for first in range(0,1500,50):
        block=logs[first:first+50]
        row=dict(first_step=first+1,last_step=first+50,mean_effective_batch_FM=statistics.mean(x['FM'] for x in block))
        for name in ('all_free','bulk','seam_tip'):
            denom=sum(x[name+'_scalar_count'] for x in block)
            row[name+'_coordinate_weighted_velocity_MSE']=sum(x[name+'_velocity_sse'] for x in block)/denom if denom else None
            row[name+'_coordinate_weighted_endpoint_MSE']=sum(x[name+'_endpoint_sse'] for x in block)/denom if denom else None
        loss_bins.append(row)
    artifacts={}
    for key in ('checkpoint','inference_export'):
        side=state[key];path=C.resolve(side['path'])
        require(C.read(path.with_suffix('.json'))==side and C.sha(path)==side['sha256'],'Final artifact differs')
        require(side['stage']=='SEAM' and side['stage_step']==1500 and side['global_step']==24000,'Artifact endpoint differs')
        artifacts[key]=side
    require(artifacts['checkpoint']['generator_state_sha256']==artifacts['inference_export']['generator_state_sha256'],'Export/recovery mismatch')
    from r2.pilots.bench_group_weight_match_v1.execution_closeout import parse_samples
    attempts=[];samples=[]
    for path in sorted((folder/'attempts').iterdir()):
        status=C.read(path/'final_status.json');resources=C.read(path/'resources.json')
        require(status['status'] in ('COMPLETE','INTERRUPTED_SAFE'),'Unresolved attempt')
        attempts.append(dict(attempt=path.name,status=status['status'],active_wall_seconds=status['elapsed_seconds']))
        samples.extend(parse_samples(resources['samples']))
    require(bool(samples),'Device resource observations missing')
    resource=dict(attempts=attempts,active_wall_seconds=sum(x['active_wall_seconds'] for x in attempts),
        sum_update_seconds=sum(x['update_seconds'] for x in logs),sum_input_seconds=sum(x['input_seconds'] for x in logs),
        aggregate_device_peak_used_mib=max(x['used_mib'] for x in samples),aggregate_device_min_free_mib=min(x['free_mib'] for x in samples),
        aggregate_device_mean_util_percent=statistics.mean(x['util_percent'] for x in samples),
        torch_peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in logs),torch_peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in logs),
        scope='Device observations include concurrent M training and user processes. Not isolated throughput or per-process VRAM.')
    DEST.mkdir(exist_ok=True)
    C.write_new(DEST/'loss_bins.json',dict(rows=loss_bins,note='Training fitting curve; not a quality ranking against Native with different masks.'))
    C.write_new(DEST/'exposure.json',dict(groups=dict(groups),parents=dict(parents),instances=12000,source_schedule_sha256=reg['schedules'][ARM]['sha256']))
    C.write_new(DEST/'resources.json',resource)
    result=dict(status='PASS',local_updates=1500,global_step=24000,ledger=ledger,artifacts=artifacts,
        source_unchanged=True,initial_live_AdamW_RNG_preserved=True,model_calls=0,tensor_loads=0,
        files={str(p):C.sha(p) for p in DEST.iterdir() if p.is_file()},code_sha256=C.sha(__file__))
    C.write_new(done,result);return result


def cpu_check():
    rows=[];call_id=0
    for step in (1,2):
        for mi,kind in [(mi,k) for mi in range(8) for k in ('train_forward','train_backward')]+[(None,'optimizer_update')]:
            call_id+=1
            context=dict(stage_step=step,local_step=step,seam_total_step=step,global_step=22500+step,
                curriculum_step=4500+step,native_total_step=10500)
            if mi is not None:context['microbatch']=mi
            row=dict(arm=ARM,kind=kind,call_id=call_id,attempt_id='CPU_FIXTURE',context=context)
            rows.extend((dict(row,event='CALL_BEGIN'),dict(row,event='CALL_DONE',seconds=.01)))
    require(calls(rows,2)['successful']==dict(train_forward=16,train_backward=16,optimizer_update=2),'CPU fixture failed')
    try:calls(rows[:-1],2)
    except ValueError:pass
    else:raise AssertionError('Incomplete scientific ledger accepted')
    return dict(status='PASS',checks=2,model_calls=0,OT_calls=0,tensor_loads=0)


def evaluation(require_visual=True):
    from . import evaluation as E
    sample=C.read(C.OUT/'evaluation/S/raw_complete_full.json')
    require(sample['status']=='COMPLETE' and sample['total_new_generator_forwards']==2300
            and sample['total_new_head_forwards']==21 and sample['outputs']==46,'RAW/head budget differs')
    post=C.read(C.OUT/'evaluation/S/post_complete_full.json')
    require(post['new_D0_forwards']==post['new_GEO_calls']==46,'D0/GEO budget differs')
    E.verify_raw('S');verify(sample['files']);verify(post['files'])
    score=C.read(C.OUT/'results/completion.json');verify(score['files']);verify(score['source_rows'])
    require((score['rows'],score['new_score_rows'],score['reused_score_rows'])==(276,138,138),'Scoring budget differs')
    require(len(score['metric_fields'])==177,'Original full metric dictionary changed')
    verify(C.read(C.OUT/'audits/BW_cache_audit.json')['files'])
    visual=None
    if require_visual:
        visual=C.read(C.OUT/'visual_review/summary_v1/completion.json');verify(visual['files']);verify(visual['sources'])
        require(visual['status']=='COMPLETE' and visual['actual_observations']==276,'Incomplete actual review')
    result=dict(status='PASS',new_generator_forwards=2300,new_head_forwards=21,new_D0_forwards=46,
        new_RAW=46,new_GEO_calls=46,new_score_rows=138,reused_score_rows=138,all_fields=177,
        actual_visual_observations=276 if visual else None,model_calls=0,scoring_calls=0,
        inputs={str(p):C.sha(p) for p in (C.OUT/'evaluation/S/raw_complete_full.json',
            C.OUT/'evaluation/S/post_complete_full.json',C.OUT/'results/completion.json')},code_sha256=C.sha(__file__))
    C.write_new(DEST/('evaluation.json' if require_visual else 'evaluation_before_visual.json'),result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('cpu-check','training','evaluation','evaluation-before-visual'))
    args=parser.parse_args();result=cpu_check() if args.stage=='cpu-check' else training() if args.stage=='training' else evaluation(args.stage=='evaluation')
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','artifacts','inputs')}))
