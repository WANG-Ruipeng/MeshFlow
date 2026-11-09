"""Record observed CPU scheduling and score counts from completed receipts."""
import json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EV=ROOT/'evaluation'
arms=[]
for label in ('ORIGINAL_LOW_N','DEDUP_LOW_N'):
    p=EV/'score_completion'/f'{label}.json'
    s=json.loads(p.read_text());assert s['status']=='COMPLETE' and s['stages']==48 and s['workers']==2
    end=p.stat().st_mtime;start=end-s['seconds']
    arms.append(dict(label=label,workers=2,quality_scans=48,seconds=s['seconds'],
        approximate_start_utc=datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),
        receipt_written_utc=datetime.datetime.fromtimestamp(end,datetime.timezone.utc).isoformat(),
        start_epoch=start,end_epoch=end))
overlap=max(0,min(a['end_epoch'] for a in arms)-max(a['start_epoch'] for a in arms))
result=dict(status='COMPLETE',new_model_calls_in_cpu_phase=0,new_quality_scores=96,cached_original_N_quality_scores=96,
    planned_workers_per_arm=2,actual_peak_score_workers=4 if overlap>0 else 2,
    approximate_score_overlap_seconds=overlap,arms=arms,
    timing_note='Score seconds from completion receipts. Absolute starts inferred from receipt write mtime minus measured stage duration, thus approximate. Two arms overlapped for a bounded tail interval; figures rendered concurrently with scoring. These concurrent wall times are not isolated compute benchmarks.',
    resource_snapshots=[str(p) for p in sorted((EV/'execution').glob('cpu_dispatch_*json'))],
    memory_note='Initial registered RAW score RSS ~101 MiB is a first-cell observation, not a guarantee of every later geometry cell. Each process capped to one BLAS/OMP thread; two workers per arm under 24 logical CPUs and >61 GiB available RAM.')
(EV/'execution/actual_cpu_dispatch_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
