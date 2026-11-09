"""Final resource accounting from saved logs only; never dispatches work."""
import json,pathlib,statistics,hashlib,re
OUT=pathlib.Path('/mnt/e/MeshFlow/experiments/r2_seam_pool47_couple_v1')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    dest=OUT/'training_resources_final.json'
    if dest.exists():raise RuntimeError('Final resource accounting already exists')
    result={'status':'COMPLETE_FROM_SAVED_LOGS','extra_forward':0,'extra_backward':0,'extra_resource_sampling':0,
        'resource_policy':'Initially ~6GiB combined training allocation; user explicitly removed cap near PLAIN2844. One incremental COUPLE worker then launched with measured margin; no scientific changes.',
        'attribution_limit':'Per-process GPU used_memory unavailable in WDDM. Device usage includes background and other arm during overlap; allocator counters are not process VRAM.',
        'timing_limit':'Short concurrent tail changes throughput; arm wall/update times are actual accounting, not an isolated compute comparison.',
        'arms':{},'sources':{}}
    for arm in ('PLAIN3000','COUPLE3000'):
        folder=OUT/'training'/arm;s=read(folder/'status.json')
        if s['status']!='COMPLETE' or s['local_step']!=3000:raise RuntimeError('Both final endpoints required')
        logs=rows(folder/'training_log.jsonl');rp=folder/'attempts'/s['attempt']/'resources.json';r=read(rp)
        checks=rows(folder/'resource_checks.jsonl')
        available=[int(m.group(1))*1024 for x in checks if (m:=re.search(r'^MemAvailable:\s+(\d+) kB',x.get('memory',''),re.M))]
        parsed=[]
        for sample in r['samples']:
            try:
                v=sample['gpu']['stdout'].splitlines()[0].split(',');parsed.append({'used_mib':float(v[2]),'free_mib':float(v[3]),'util_percent':float(v[4])})
            except (ValueError,KeyError,IndexError):pass
        result['arms'][arm]={'wall_seconds':s['wall_seconds'],'updates':len(logs),
          'update_seconds_mean':statistics.mean(x['update_seconds'] for x in logs),
          'update_seconds_sum':sum(x['update_seconds'] for x in logs),
          'input_seconds_sum':sum(x['input_seconds'] for x in logs),
          'forward_seconds_sum':sum(x['forward_seconds'] for x in logs),
          'backward_seconds_sum':sum(x['backward_seconds'] for x in logs),
          'allocator_peak_allocated_bytes':max(x['cuda_peak_allocated_bytes'] for x in logs),
          'allocator_peak_reserved_bytes':max(x['cuda_peak_reserved_bytes'] for x in logs),
          'allocator_scope':'Maximum of actual per-update reset peak counters spanning forward/backward/optimizer; full process sampled separately',
          'GPUObserver':r['summary'],'gpu_util_mean':statistics.mean(x['util_percent'] for x in parsed),
          'gpu_util_peak':max(x['util_percent'] for x in parsed),'device_free_min_mib':min(x['free_mib'] for x in parsed),
          'CPU_RAM_checks':len(checks),'logical_CPU_cores':checks[0].get('cpu_logical'),
          'RAM_available_min_bytes':min(available) if available else None,
          'CPU_loadavg1_peak':max(x['loadavg'][0] for x in checks)}
        for p in (folder/'status.json',folder/'training_log.jsonl',folder/'resource_checks.jsonl',rp):result['sources'][str(p)]=sha(p)
    for name in ('first_worker_resource_summary.json','concurrent_resource_summary.json','serial_training_dispatch.jsonl','incremental_training_dispatch.jsonl','concurrent_resource_observer_failure.json'):
        p=OUT/name;result['sources'][str(p)]=sha(p)
    result['concurrency']=read(OUT/'concurrent_resource_summary.json')
    dest.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
