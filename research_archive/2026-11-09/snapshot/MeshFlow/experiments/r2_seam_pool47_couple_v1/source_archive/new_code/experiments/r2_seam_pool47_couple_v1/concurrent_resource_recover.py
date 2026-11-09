"""Summarize already recorded concurrency, without observing or rerunning work."""
import json,pathlib,statistics,hashlib
OUT=pathlib.Path('/mnt/e/MeshFlow/experiments/r2_seam_pool47_couple_v1')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def mean(rr,k):return statistics.mean(r[k] for r in rr) if rr else None
def main():
    dest=OUT/'concurrent_resource_summary.json'
    if dest.exists():raise RuntimeError('Summary already exists')
    p=rows(OUT/'training/PLAIN3000/training_log.jsonl');c=rows(OUT/'training/COUPLE3000/training_log.jsonl')
    ps=read(OUT/'training/PLAIN3000/status.json')
    if ps['status']!='COMPLETE' or len(c)<100:raise RuntimeError('Actual completed PLAIN and first100 COUPLE required')
    events=rows(OUT/'incremental_training_dispatch.jsonl');start=next(r['unix'] for r in events if r['event']=='COUPLE_PROCESS_STARTED')
    full=read(OUT/'training/PLAIN3000/attempts'/ps['attempt']/'resources.json')
    gpu=[r for r in full['samples'] if start<=r['time_unix']<=p[-1]['time_unix']]
    parsed=[]
    for r in gpu:
        try:
            v=r['gpu']['stdout'].splitlines()[0].split(',');parsed.append(dict(used=float(v[2]),free=float(v[3]),util=float(v[4])))
        except (KeyError,ValueError,IndexError):pass
    lo=c[9]['time_unix'];hi=p[-1]['time_unix'];po=[r for r in p if lo<=r['time_unix']<=hi];co=[r for r in c if lo<=r['time_unix']<=hi]
    prev=[r for r in p if r['time_unix']<start][-100:]
    baseline_rate=(len(prev)-1)/(prev[-1]['time_unix']-prev[0]['time_unix'])
    rate=(len(po)+len(co))/(hi-lo)
    cpu=rows(OUT/'concurrent_resource_audit.jsonl')
    result=dict(status='COMPLETE_FROM_EXISTING_LOGS',observer_error_preserved='concurrent_resource_observer_failure.json',
      extra_model_calls=0,extra_resource_sampling=0,
      scope='GPU: trainer PLAIN full-stage observer restricted to actual concurrent interval; CPU/RAM: partial independent observer until atomic-file-read race',
      gpu_sample_count=len(parsed),device_used_peak_mib=max(r['used'] for r in parsed),device_free_min_mib=min(r['free'] for r in parsed),
      gpu_util_mean=mean(parsed,'util'),gpu_util_max=max(r['util'] for r in parsed),CPU_RAM_partial_sample_count=len(cpu),
      RAM_available_min_bytes=min(r['ram_available_bytes'] for r in cpu),CPU_percent_mean=mean(cpu,'cpu_percent'),
      process_memory_limitation='WDDM per-process GPU memory N/A; device totals include background, allocator peaks are not full-process estimates',
      PLAIN_total_wall_seconds=ps['wall_seconds'],PLAIN_fullstage_resources=ps['resources'],
      post_warmup_concurrent_interval=dict(from_unix=lo,through_unix=hi,PLAIN_updates=len(po),COUPLE_updates=len(co),
          aggregate_updates_per_second=rate,PLAIN_recent_isolated_updates_per_second=baseline_rate,
          aggregate_rate_ratio=rate/baseline_rate,PLAIN_update_seconds_mean=mean(po,'update_seconds'),COUPLE_update_seconds_mean=mean(co,'update_seconds')),
      throughput_limit='Short tail overlap; arms have different loss cost and shared checkpoint IO. Actual throughput observation, not a controlled benchmark',
      COUPLE_first100=dict(update_seconds_mean=mean(c[:100],'update_seconds'),allocated_peak_bytes=max(r['cuda_peak_allocated_bytes'] for r in c[:100]),reserved_peak_bytes=max(r['cuda_peak_reserved_bytes'] for r in c[:100])),
      source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [OUT/'concurrent_resource_audit.jsonl',OUT/'training/PLAIN3000/attempts'/ps['attempt']/'resources.json']})
    dest.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
