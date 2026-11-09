"""Independent CPU accounting of actual M continuation, without model loads."""
from __future__ import annotations
import argparse,collections,csv,json,math,statistics
from pathlib import Path
from . import common as C
from r2.pilots.bench_group_weight_match_v1.execution_closeout import parse_samples

ARM='M_PLUS3000'
DEST=C.OUT/'closeout'

def require(ok,message):
    if not ok:raise ValueError(message)

def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def verify(book):
    for p,h in book.items():require(C.sha(p)==h,'Changed frozen file: '+str(p))

def calls(rows,updates=3000):
    wanted=[]
    for step in range(1,updates+1):
        wanted.extend((step,mi,k) for mi in range(8) for k in ('train_forward','train_backward'))
        wanted.append((step,None,'optimizer_update'))
    require(len(rows)==2*len(wanted),'Ledger event count differs')
    counts=collections.Counter();seconds=collections.Counter()
    for i,(step,mi,kind) in enumerate(wanted):
        a,b=rows[2*i:2*i+2]
        require(a['event']=='CALL_BEGIN' and b['event']=='CALL_DONE','Failed or pending call')
        require(a['call_id']==b['call_id']==i+1,'Duplicated or missing call identity')
        require(a['arm']==b['arm']==ARM and a['kind']==b['kind']==kind,'Wrong arm or operation')
        require(a['context']==b['context'] and a['attempt_id']==b['attempt_id'],'Context changed')
        c=a['context']
        require(c['stage_step']==c['local_step']==step and c.get('microbatch')==mi,'Wrong update/microbatch')
        require(c['global_step']==22500+step and c['native_total_step']==10500+step
            and c['curriculum_step']==4500+step and c['seam_total_step']==0,'Wrong cumulative cursor')
        require(math.isfinite(b['seconds']) and b['seconds']>=0,'Invalid timing')
        counts[kind]+=1;seconds[kind]+=b['seconds']
    return dict(successful=dict(counts),attempted=dict(counts),failed=0,pending=0,
        replayed_updates=0,head_calls=0,probe_calls=0,scientific_call_seconds=dict(seconds))

def training():
    done=DEST/'training.json'
    if done.exists():
        value=C.read(done);verify(value['files']);return value
    folder=C.OUT/'training'/ARM;state=C.read(folder/'status.json');reg=C.read(C.OUT/'training_registration.json')
    require(state['status']=='COMPLETE','Fixed training endpoint not complete')
    require((state['local_step'],state['global_step'],state['native_total_step'],state['seam_total_step'])==(3000,25500,13500,0),'Wrong endpoint')
    require(state['registration_sha256']==C.sha(C.OUT/'training_registration.json'),'Registration changed')
    for key in ('training_code_hashes','gate_hashes','protected'):verify(reg[key])
    for key in ('manifest','eval_manifest','source_checkpoint','coefficient_schedule'):
        require(C.sha(reg[key])==reg[key+'_sha256'],'Changed '+key)
    spec=reg['schedules'][ARM];require(C.sha(spec['path'])==spec['sha256'],'Stream changed')
    init=C.read(folder/'branch_initialization.json')
    for key in ('source_checkpoint_sha256','source_generator_sha256','source_optimizer_sha256','source_rng_sha256'):
        require(init[key]==reg[key],'Initialization identity changed')
    require(init['inherited_optimizer'] and init['inherited_rng'] and not init['fresh_AdamW'] and not init['head_used'],'Continuation policy differs')
    ledger=calls(lines(folder/'calls.jsonl'))
    for kind in ('attempted','successful'):require(state['calls'][kind]==ledger[kind],'Status and scientific ledger differ')
    require(not state['calls']['failed'] and not state['calls']['pending_call_ids'],'Unresolved calls')
    logs=lines(folder/'training_log.jsonl');exposure=lines(folder/'exposure.jsonl')
    schedule=C.read(spec['path'])['steps'];coeff=C.read(reg['coefficient_schedule'])['steps']
    require(len(logs)==3000 and len(exposure)==24000,'Update/exposure count differs')
    groups=collections.Counter();parents=collections.Counter();high=collections.Counter();seedtuples=[]
    for step,(entry,co,log) in enumerate(zip(schedule,coeff,logs),1):
        batch=exposure[(step-1)*8:step*8]
        require(log['local_step']==step and log['group_allocation']==co,'Coefficient schedule differs')
        denom=sum(x['all_free_scalar_count'] for x in batch);sse=sum(x['all_free_velocity_sse'] for x in batch)
        require(denom==co['D_W']==log['free_coordinate_denominator'],'Effective denominator differs')
        objective=0.
        for mi,(a,b) in enumerate(zip(entry['samples'],batch)):
            require(all(b[k]==v for k,v in a.items()),'Actual sample differs from registered stream')
            require(not b['head_forward_called'] and b['mechanism']['policy']=='NATIVE_ALL_FREE_OT','Wrong visible task')
            require(b['loss_gain']==co['micro_gains'][mi],'Gain differs')
            objective+=b['all_free_velocity_sse']/denom*b['loss_gain']
            groups[b['replay_group']]+=1;parents[b['parent_id']]+=1
            if b['replay_group']=='HIGH':high[b['parent_id']]+=1
            seedtuples.append(tuple(b[k] for k in ('noise_seed','epsilon_seed','time_seed','permutation_seed')))
        require(math.isclose(sse/denom,log['FM'],rel_tol=4e-6,abs_tol=1e-7),'Unweighted FM accounting differs')
        require(math.isclose(objective,log['objective'],rel_tol=4e-6,abs_tol=1e-7),'Weighted objective accounting differs')
    require(dict(groups)==dict(OLD=18000,FOCUS=3000,HIGH=3000),'Group exposure differs')
    require(len(high)==6 and set(high.values())=={500},'Six HIGH parents need exactly500 added exposures')
    require(len(seedtuples)==len(set(seedtuples))==24000,'Duplicated stochastic instance')
    old=C.M_ROOT/'training/W_MATCHED/exposure.jsonl';old_rows=lines(old)
    old_high=collections.Counter(r['parent_id'] for r in old_rows if r['replay_group']=='HIGH')
    require(all(old_high[p]==250 for p in high),'Historical HIGH exposure differs from250')
    old_seeds={tuple(r[k] for k in ('noise_seed','epsilon_seed','time_seed','permutation_seed')) for r in old_rows}
    require(not old_seeds.intersection(seedtuples),'Old stochastic instance reused')
    artifacts={}
    for key in ('checkpoint','inference_export'):
        side=state[key];p=C.resolve(side['path'])
        require(C.read(p.with_suffix('.json'))==side and C.sha(p)==side['sha256'],'Final artifact identity differs')
        require(side['stage_step']==3000 and side['global_step']==25500,'Artifact endpoint differs')
        artifacts[key]=side
    require(artifacts['checkpoint']['generator_state_sha256']==artifacts['inference_export']['generator_state_sha256'],'Export/recovery differ')
    attempts=[];samples=[]
    require(not list((folder/'attempts').glob('*/failure.json')),'Unresolved failure')
    for p in sorted((folder/'attempts').iterdir()):
        st=C.read(p/'final_status.json');rs=C.read(p/'resources.json')
        require(st['status'] in ('COMPLETE','INTERRUPTED_SAFE'),'Unresolved attempt')
        attempts.append(dict(attempt=p.name,status=st['status'],active_wall_seconds=st['elapsed_seconds']))
        samples.extend(parse_samples(rs['samples']))
    require(samples,'Missing GPU observer')
    bins=[]
    for first in range(0,3000,50):
        block=logs[first:first+50];den=sum(x['all_free_scalar_count'] for x in block)
        bins.append(dict(first=first+1,last=first+50,mean_batch_objective=statistics.mean(x['objective'] for x in block),
            mean_batch_FM=statistics.mean(x['FM'] for x in block),coordinate_weighted_velocity_MSE=sum(x['all_free_velocity_sse'] for x in block)/den,
            coordinate_weighted_endpoint_MSE=sum(x['all_free_endpoint_sse'] for x in block)/den))
    DEST.mkdir(exist_ok=True)
    C.write_new(DEST/'loss_bins.json',dict(rows=bins,meaning='Fitting curves; never an endpoint quality criterion'))
    C.write_new(DEST/'exposure.json',dict(groups=dict(groups),parents=dict(parents),HIGH_added=dict(high),HIGH_cumulative={p:high[p]+old_high[p] for p in high},fresh_seed_tuples=24000,old_seed_overlap=0))
    C.write_new(DEST/'resources.json',dict(attempts=attempts,active_wall_seconds=sum(x['active_wall_seconds'] for x in attempts),
        sum_update_seconds=sum(x['update_seconds'] for x in logs),sum_input_seconds=sum(x['input_seconds'] for x in logs),
        device_peak_used_mib=max(x['used_mib'] for x in samples),device_min_free_mib=min(x['free_mib'] for x in samples),
        mean_GPU_util_percent=statistics.mean(x['util_percent'] for x in samples),
        allocator_peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in logs),allocator_peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in logs),
        scope='Concurrent SEAM, sampling and user processes included. Not isolated throughput or per-process VRAM.'))
    result=dict(status='PASS',updates=3000,global_step=25500,ledger=ledger,artifacts=artifacts,
        source_unchanged=True,matching_AdamW_RNG=True,additional_compute_not_pure_coverage_effect=True,
        model_calls=0,tensor_loads=0,files={str(p):C.sha(p) for p in DEST.iterdir() if p.is_file()},code_sha256=C.sha(__file__))
    C.write_new(done,result);return result

def cpu_check():
    rows=[];cid=0
    for step in (1,2):
        for mi,k in [(m,k) for m in range(8) for k in ('train_forward','train_backward')]+[(None,'optimizer_update')]:
            cid+=1;c=dict(stage_step=step,local_step=step,global_step=22500+step,native_total_step=10500+step,curriculum_step=4500+step,seam_total_step=0)
            if mi is not None:c['microbatch']=mi
            base=dict(arm=ARM,kind=k,call_id=cid,attempt_id='CPU_FIXTURE',context=c)
            rows.extend((dict(base,event='CALL_BEGIN'),dict(base,event='CALL_DONE',seconds=.1)))
    require(calls(rows,2)['successful']['train_forward']==16,'Fixture count differs')
    for invalid in (rows[:-1],rows+rows[-2:]):
        try:calls(invalid,2)
        except ValueError:pass
        else:raise AssertionError('Invalid ledger accepted')
    return dict(status='PASS',checks=3,model_calls=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['training','cpu-check']);a=p.parse_args()
    r=training() if a.stage=='training' else cpu_check()
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','artifacts')}))
