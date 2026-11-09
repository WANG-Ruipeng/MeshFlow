"""CPU-only observer of already authorized work; no model imports or calls."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import time,json,psutil,pathlib,statistics,hashlib
from r2.pilots.seam_abc_v1 import generate as G
OUT=pathlib.Path('/mnt/e/MeshFlow/experiments/r2_seam_pool47_couple_v1')
DEST=OUT/'concurrent_resource_audit.jsonl'
def read(p):return json.loads(p.read_text())
def logs(arm):
    p=OUT/'training'/arm/'training_log.jsonl'
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def main():
    if DEST.exists():raise RuntimeError('Observer already run')
    started=time.time();samples=[]
    for _ in range(180):
        r=G.resources();r['arms']={}
        for arm,pid in [('PLAIN3000',322),('COUPLE3000',10428)]:
            p=OUT/'training'/arm/'status.json';s=read(p) if p.exists() else {}
            r['arms'][arm]={k:s.get(k) for k in ('status','local_step','wall_seconds','peak_allocated_bytes','peak_reserved_bytes')}
            try:r['arms'][arm]['cpu_rss_bytes']=psutil.Process(pid).memory_info().rss
            except psutil.NoSuchProcess:r['arms'][arm]['process_exited']=True
        with DEST.open('a') as f:f.write(json.dumps(r)+'\n')
        samples.append(r)
        states=[x.get('status') for x in r['arms'].values()]
        if any(s in ('FAILED','OOM','INTERRUPTED_SAFE') for s in states):break
        if r['arms']['PLAIN3000'].get('status')=='COMPLETE' and (r['arms']['COUPLE3000'].get('local_step') or 0)>=100:break
        time.sleep(10)
    result={'status':'OBSERVED','wall_seconds':time.time()-started,'sample_count':len(samples),
      'scope':'Actual concurrent tail and first COUPLE updates only; full stage counters retained by each trainer',
      'process_memory_limitation':'WDDM per-process used memory unavailable; device totals include background',
      'no_extra_model_calls':True,'arms':{}}
    device=[];util=[];free=[]
    for r in samples:
        try:
            v=r['gpu']['stdout'].splitlines()[0].split(',');device.append(float(v[2]));free.append(float(v[3]));util.append(float(v[4]))
        except (KeyError,ValueError,IndexError):pass
    result.update(device_used_peak_mib=max(device),device_free_min_mib=min(free),gpu_util_mean=statistics.mean(util),gpu_util_max=max(util),RAM_available_min_bytes=min(r['ram_available_bytes'] for r in samples))
    for arm in ('PLAIN3000','COUPLE3000'):
        rr=logs(arm);seen=[r for r in rr if r['time_unix']>=started]
        first=rr[:100];result['arms'][arm]={'last_step':rr[-1]['local_step'] if rr else None,
            'observed_update_count':len(seen),'observed_update_seconds_mean':statistics.mean(r['update_seconds'] for r in seen) if seen else None,
            'all_recorded_allocated_peak_bytes':max((r['cuda_peak_allocated_bytes'] for r in rr),default=None),
            'all_recorded_reserved_peak_bytes':max((r['cuda_peak_reserved_bytes'] for r in rr),default=None),
            'first100_update_seconds_mean':statistics.mean(r['update_seconds'] for r in first) if first else None}
    p,c=logs('PLAIN3000'),logs('COUPLE3000')
    if len(c)>=10 and p:
        lo=c[9]['time_unix'];hi=p[-1]['time_unix'];n=sum(lo<=r['time_unix']<=hi for r in p+c)
        result['post_warmup_overlap']={'from_unix':lo,'through_unix':hi,'updates_both_arms':n,
          'aggregate_updates_per_second':n/(hi-lo) if hi>lo else None,
          'warning':'Short overlap; arms have different loss cost and shared checkpoint IO, not a controlled throughput benchmark'}
    result['audit_sha256']=hashlib.sha256(DEST.read_bytes()).hexdigest()
    (OUT/'concurrent_resource_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
