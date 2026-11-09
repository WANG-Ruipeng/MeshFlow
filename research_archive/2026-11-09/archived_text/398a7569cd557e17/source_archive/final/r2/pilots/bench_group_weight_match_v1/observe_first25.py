"""Read-only formal-training observation; no model imports or dispatch."""
import collections,json,math,subprocess,time
from pathlib import Path
from .common import OUT,read,sha,write_new


def main():
    folder=OUT/'training/W_MATCHED';dest=OUT/'dispatch/first25_readonly_observation.json'
    if dest.exists():return read(dest)
    samples=[];deadline=time.monotonic()+90
    while True:
        status=read(folder/'status.json')
        proc=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        fields=[x.strip() for x in proc.stdout.strip().split(',')]
        samples.append(dict(unix=time.time(),step=status['stage_step'],state=status['status'],
            name=fields[0],total_mib=float(fields[1]),used_mib=float(fields[2]),free_mib=float(fields[3]),util_percent=float(fields[4])))
        if status['stage_step']>=25 or status['status'] not in ('LOADING','TRAINING'):break
        if time.monotonic()>deadline:raise RuntimeError('Observation budget ended before25; preserve no-science scope')
        time.sleep(2)
    def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    logs=rows(folder/'training_log.jsonl')[:25];exposures=rows(folder/'exposure.jsonl')[:200]
    ledger=[x for x in rows(folder/'calls.jsonl') if x['context']['stage_step']<=25]
    assert len(logs)==25 and len(exposures)==200
    counts=collections.Counter(x['kind'] for x in ledger if x['event']=='CALL_DONE')
    assert dict(counts)==dict(train_forward=200,train_backward=200,optimizer_update=25)
    assert all(x['event'] in ('CALL_BEGIN','CALL_DONE') for x in ledger)
    for index,log in enumerate(logs):
        batch=exposures[index*8:(index+1)*8];a=log['group_allocation']
        assert log['global_step']==21001+index
        objective=(log['group_velocity_sse']['OLD']+(a['rho'] if a['branch']=='HIGH' else 1.)*
            (log['group_velocity_sse']['HIGH'] if a['branch']=='HIGH' else log['group_velocity_sse']['FOCUS']))/a['D_B']
        assert math.isclose(objective,log['objective'],rel_tol=5e-6,abs_tol=2e-7)
        assert math.isclose(sum(x['loss_contribution'] for x in batch),log['objective'],rel_tol=1e-10,abs_tol=1e-8)
    init=read(folder/'branch_initialization.json');reg=read(OUT/'training_registration.json')
    assert all(init[k]==reg[k] for k in ('source_checkpoint_sha256','source_generator_sha256','source_optimizer_sha256','source_rng_sha256'))
    assert init['inherited_optimizer'] and init['inherited_rng'] and not init['fresh_AdamW']
    result=dict(status='PASS',observed_prefix_updates=25,formal_prefix_calls=dict(counts),
        source_restoration_record_matches=True,first_formal_backward=read(folder/'first_formal_backward.json'),
        last_formal_log=logs[-1],formal_prefix_update_seconds=sum(x['update_seconds'] for x in logs),
        formal_prefix_input_seconds=sum(x['input_seconds'] for x in logs),
        allocator_peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in logs),
        allocator_peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in logs),
        live_gpu_samples=samples,device_peak_used_mib=max(x['used_mib'] for x in samples),
        device_min_free_mib=min(x['free_mib'] for x in samples),
        observation_scope='Read-only live samples only; not full interval process peak. Full trainer observer available at completion; WDDM process attribution unavailable.',
        resources_step25=read(folder/'resources_step25.json'),new_scientific_calls=0,
        code_sha256=sha(__file__),frozen_registration_sha256=sha(OUT/'training_registration.json'))
    write_new(dest,result);return result


if __name__=='__main__':
    r=main();print(json.dumps({k:r[k] for k in ('status','formal_prefix_calls','device_peak_used_mib','device_min_free_mib','formal_prefix_update_seconds')}))
