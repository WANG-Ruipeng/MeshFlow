"""CPU-only accounting from actual saved RAW/D0 stage records."""
import json,pathlib,statistics,hashlib
OUT=pathlib.Path('/mnt/e/MeshFlow/experiments/r2_seam_pool47_couple_v1')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def main():
    dest=OUT/'sampling_resources_final.json'
    if dest.exists():raise RuntimeError('Already summarized')
    reg=read(OUT/'generation_registration.json');data={};intervals={'RAW':[],'D0':[]}
    for arm in reg['arms']:
        folder=OUT/'evaluation'/arm;raw=[];post=[];times={'RAW':[],'D0':[]}
        for q in reg['requests']:
            d=folder/q['request_id'];raw.append(read(d/'raw/audit.json'));post.append(read(d/'post/proposal_receipt.json'))
            times['RAW'].extend(r['unix'] for r in rows(d/'raw/calls.jsonl'))
            times['D0'].extend(r['unix'] for r in rows(d/'post/calls.jsonl'))
        snaps=[]
        for p in folder.rglob('resources*.json'):
            r=read(p)
            try:
                v=r['gpu']['stdout'].splitlines()[0].split(',');snaps.append(dict(used=float(v[2]),free=float(v[3]),util=float(v[4])))
            except (KeyError,ValueError,IndexError):pass
        result={'RAW_outputs':len(raw),'D0_outputs':len(post),
          'RAW_phase_seconds':read(folder/'raw_complete.json')['wall_seconds'],
          'POST_phase_seconds':read(folder/'post_complete.json')['wall_seconds'],
          'RAW_rollout_seconds_mean':statistics.mean(r['seconds'] for r in raw),
          'D0_call_seconds_mean':statistics.mean(r['seconds'] for r in post),
          'RAW_allocator_allocated_peak_bytes':max(r['peak_allocated_bytes'] for r in raw),
          'RAW_allocator_reserved_peak_bytes':max(r['peak_reserved_bytes'] for r in raw),
          'D0_allocator_allocated_peak_bytes':max(r['peak_allocated_bytes'] for r in post),
          'D0_allocator_reserved_peak_bytes':max(r['peak_reserved_bytes'] for r in post),
          'resource_snapshot_count':len(snaps),'device_used_max_snapshot_mib':max(r['used'] for r in snaps),
          'device_free_min_snapshot_mib':min(r['free'] for r in snaps),'gpu_util_max_snapshot_percent':max(r['util'] for r in snaps)}
        for stage in ('RAW','D0'):
            pair=(min(times[stage]),max(times[stage]));intervals[stage].append(pair);result[stage+'_call_interval_unix']=pair
        data[arm]=result
    overlap={stage:max(0,min(p[1] for p in pairs)-max(p[0] for p in pairs)) for stage,pairs in intervals.items()}
    result=dict(status='COMPLETE',arms=data,call_interval_overlap_seconds=overlap,
      observed_peak_workers=dict(RAW=2,D0=1),extra_model_calls=0,
      scope='Allocator peaks reset per real rollout/D0; snapshots are not a continuous device-memory peak, and WDDM process usage is unavailable.',
      wall_scope='RAW phase timer starts after model load; POST includes D0 load. Protected input/hash checks precede both timers. Dispatch timestamps preserve end-to-end stage scheduling.',
      concurrency='Two incremental RAW workers after real measurement. D0 first arm completed before the measurement/dispatch roundtrip could overlap the second; no resource shortage claimed.',
      timing_limit='Actual concurrent wall times are not isolated compute benchmarks.',
      generation_stage_audit_sha256=hashlib.sha256((OUT/'generation_stage_audit.json').read_bytes()).hexdigest())
    dest.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
