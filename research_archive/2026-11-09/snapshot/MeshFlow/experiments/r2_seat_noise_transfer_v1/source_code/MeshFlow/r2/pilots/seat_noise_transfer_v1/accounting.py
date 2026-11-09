"""CPU-only end-of-run receipts, immutable-source checks and resource accounting."""
import csv,json,math
from pathlib import Path
from . import common as C
EV=C.OUT/'evaluation'
def run():
    r=C.read(EV/'registration.json');m=C.read(C.OUT/'inputs/manifest.json');labels=r['labels'];raw=C.read(EV/'raw_index.json')['rows'];post=C.read(EV/'post_index.json')['rows']
    if len(raw)!=144 or len(post)!=144:raise ValueError('Incomplete cells')
    ledger_rows=[];returned=reserved=0
    for x in raw:
        path=C.resolve(x['raw_path']).parent/'calls.jsonl';calls=[json.loads(line) for line in path.read_text().splitlines()]
        res=sum(z.get('event')=='RESERVE' and z.get('kind')=='sampling_forward' for z in calls);ret=sum(z.get('event')=='RETURN' and z.get('kind')=='sampling_forward' for z in calls)
        if (res,ret)!=(50,50):raise ValueError('Euler call mismatch')
        returned+=ret;reserved+=res
        ledger_rows.append(dict(label=x['label'],request_id=x['request_id'],panel=x['panel'],stage='RAW',reserved=res,returned=ret,input_sha256=x['input_sha256'],checkpoint_sha256=x['checkpoint_sha256'],receipt_sha256=C.sha(path.parent/'receipt.json')))
    d0_seconds=0;d0_alloc=[];d0_reserved=[]
    for x in post:
        a=C.read(x['proposal_receipt'])
        if a['model_forwards']!=1 or a['checkpoint_sha256']!=r['d0_checkpoint']['sha256']:raise ValueError('D0 contract')
        if C.sha(x['proposal_path'])!=a['output_sha256']:raise ValueError('Proposal hash')
        d0_seconds+=a['seconds'];d0_alloc.append(a['peak_allocated_bytes']);d0_reserved.append(a['peak_reserved_bytes'])
        ledger_rows.append(dict(label=x['label'],request_id=x['request_id'],panel=x['panel'],stage='D0',reserved=1,returned=1,input_sha256=x['raw_sha256'],checkpoint_sha256=a['checkpoint_sha256'],receipt_sha256=C.sha(x['proposal_receipt'])))
    calls=[json.loads(line) for line in (C.OUT/'execution/worker_0_of_1_events.jsonl').read_text().splitlines()]
    d0_res=sum(x['event']=='D0_FORWARD_RESERVED' for x in calls);d0_ret=sum(x['event']=='D0_FORWARD_RETURNED' for x in calls)
    if (d0_res,d0_ret)!=(144,144) or any(x['event']=='POST_STOPPED' for x in calls):raise ValueError('D0 event ledger incomplete or unexpected stop')
    C.csvwrite(C.OUT/'call_accounting.csv',ledger_rows)
    protected={**m['source_hashes'],**r['code_hashes'],**{x['path']:x['sha256'] for x in r['models'].values()},r['d0_checkpoint']['path']:r['d0_checkpoint']['sha256']}
    actual={p:C.sha(p) for p in protected}
    if actual!=protected:raise ValueError('Protected source changed')
    C.write(C.OUT/'protected_assets_final.json',dict(status='PASS',count=len(protected),hashes=actual))
    gen={arm:C.read(EV/'execution'/arm/'generator_0_1_complete.json') for arm in labels}
    d0={arm:C.read(EV/'execution'/arm/'D0_0_1_complete.json') for arm in labels}
    resources=[]
    for p in sorted((C.OUT/'resources').glob('*.json')):
        z=C.read(p);s=z['gpu']['stdout'].strip().splitlines()[0].split(',')
        resources.append(dict(path=str(p),utc=z['utc'],total_MiB=float(s[1]),used_MiB=float(s[2]),free_MiB=float(s[3]),utilization_percent=float(s[4])))
    C.csvwrite(C.OUT/'resource_observations.csv',resources)
    result=dict(status='COMPLETE',generator_rollouts=144,generator_reserved=reserved,generator_returned=returned,D0_reserved=d0_res,D0_returned=d0_ret,training_updates=0,backwards=0,unresolved_model_calls=0,
        generator_phases=gen,D0_phases=d0,raw_generation_loop_seconds=sum(x['sampling_seconds'] for x in gen.values()),generator_load_sample_audit_seconds=sum(x['seconds'] for x in gen.values()),D0_geo_export_audit_seconds=sum(x['seconds'] for x in d0.values()),D0_forward_seconds=d0_seconds,
        max_D0_allocated_bytes=max(d0_alloc),max_D0_reserved_bytes=max(d0_reserved),min_observed_whole_device_free_MiB=min(x['free_MiB'] for x in resources),max_observed_whole_device_used_MiB=max(x['used_MiB'] for x in resources),
        GPU_workers=1,reason='User ~6GiB incremental budget; retained conservative generator/D0 process estimates and loading/context headroom while graphics app active. CPU scoring parallel only.',
        memory_scope='Allocated/reserved are PyTorch only; whole-device includes user apps; WDDM per-PID peak unavailable. Stage boundary samples are observations, not continuous process peak.',protected_files=len(protected))
    C.write(C.OUT/'call_accounting.json',result);C.resources(C.OUT/'resources_final.json');print(json.dumps(result),flush=True)
if __name__=='__main__':run()