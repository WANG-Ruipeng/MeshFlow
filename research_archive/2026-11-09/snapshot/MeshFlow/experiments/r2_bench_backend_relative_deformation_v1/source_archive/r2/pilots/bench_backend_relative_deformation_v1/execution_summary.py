"""Accounting-only phase/resource summary, no scientific calls."""
from __future__ import annotations
from .common import *

def run():
    path=OUT/'execution_summary.json'
    if path.exists(): return read(path)
    training=read(OUT/'training/complete.json')
    updates=[json.loads(s) for s in (OUT/'training/losses.jsonl').read_text().splitlines()]
    phases=[]
    for folder in [OUT/'training',OUT/'evaluation/CANDIDATE_RAW',OUT/'evaluation/CANDIDATE_SOURCE']:
        observations=[]
        for p in sorted(folder.glob('resources_*.json')):
            r=read(p)
            if not r.get('gpu',{}).get('stdout'):continue
            v=[x.strip() for x in r['gpu']['stdout'].strip().split(',')]
            if len(v)!=5:continue
            observations.append(dict(source=str(p),gpu=v[0],total_mib=float(v[1]),
                used_mib=float(v[2]),free_mib=float(v[3]),utilization_percent=float(v[4])))
        result=dict(phase=folder.name,actual_gpu_workers=1,
            whole_gpu_max_sampled_used_mib=max(r['used_mib'] for r in observations),
            whole_gpu_min_sampled_free_mib=min(r['free_mib'] for r in observations),observations=observations)
        if folder.name!='training':
            r=read(folder/'complete.json'); receipts=[read(p) for p in folder.glob('*/receipt.json')]
            result.update(total_seconds=r['seconds'],forwards=r['new_forwards'],
                per_output_seconds_sum=sum(a['seconds'] for a in receipts),
                max_allocated_bytes=max(a['peak_allocated_bytes'] for a in receipts),
                max_reserved_bytes=max(a['peak_reserved_bytes'] for a in receipts))
        phases.append(result)
    out=dict(status='COMPLETE',training=training['counts'],
        training_total_seconds=training['total_seconds'],training_segment_seconds=training['training_seconds'],
        training_update_loop_seconds=sum(r['seconds'] for r in updates),
        training_optimizer_updates_per_loop_second=500/sum(r['seconds'] for r in updates),
        training_peak_allocated_bytes=training['peak_allocated_bytes'],training_peak_reserved_bytes=training['peak_reserved_bytes'],
        phases=phases,cpu_score_workers=3,cpu_threads_per_worker=1,
        cpu_render_workers=1,cpu_source_analysis_workers=1,
        scheduling='Only one authorized training arm. Candidate RAW evaluation completed before second evaluation worker dispatch; SOURCE then overlapped independent CPU scoring and rendering. No concurrent GPU timing comparison is claimed.',
        timing_scope='Runtime total includes in-phase load/check/save work but not all root orchestration or final hashes. Update-loop sum excludes checkpoint serialization. Whole GPU sampled values include other user activity and are not a complete per-process high-water mark.',
        no_oom=True,no_training_failure=True,no_training_replay=True,
        additional_generator_forwards=0,new_random_draws=0,additional_evaluation_D0_forwards=60)
    write(path,out);return out

if __name__=='__main__':
    result=run();print({k:v for k,v in result.items() if k!='phases'})
