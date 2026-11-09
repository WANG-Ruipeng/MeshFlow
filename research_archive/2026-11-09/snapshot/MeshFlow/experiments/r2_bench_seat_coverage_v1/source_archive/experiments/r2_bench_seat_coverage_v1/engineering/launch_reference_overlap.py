"""Explicit incremental dispatch of the six already-registered reference RAWs."""
from pathlib import Path
import math,time
from r2.pilots.bench_seat_coverage_v1.common import OUT,read,sha,resources,write_new
from r2.pilots.bench_seat_coverage_v1.run import launch,status

path=OUT/'dispatch/reference_overlap_override.json'
if path.exists():raise RuntimeError('Already considered; no automatic redispatch')
decision=read(OUT/'controller/training_overlap_decision.json')
observations=decision['comparable_sampler_observations']
peak=max(x['summary']['total_device_used_peak_mib'] for x in observations)
r=resources();cells=r['gpu']['stdout'].strip().split(',')
total,used,free,util=[float(c.strip().split()[0]) for c in cells[1:5]]
headroom=max(2048.,.1*total)
slots=math.floor(max(0.,free-headroom)/(1.2*peak))
train=read(OUT/'training/NATIVE/status.json')
allowed=(train['status']=='TRAINING' and train['stage_step']>=25 and slots>=1
    and util<80 and r['RAM_available']>=8*2**30 and status()['raw-native']['status']=='NOT_STARTED')
record=dict(status='RELEASE' if allowed else 'WAIT_NO_DISPATCH',time=time.time(),resources=r,
    previous_decision_sha256=sha(OUT/'controller/training_overlap_decision.json'),
    conservative_sampling_memory_upper_bound_mib=peak,attributable_process_peak_mib=None,
    upper_bound_scope='Maximum total-device peak of completed comparable BF16 Native Euler50 sampling; includes background VRAM, intentionally conservative. Not a process attribution.',
    comparable=observations,retained_headroom_mib=headroom,incremental_slots=slots,
    current_training_step=train['stage_step'],maximum_added_workers=1,actual_outputs_authorized=6,
    unchanged_Euler50_and_input_manifest=True,no_additional_samples_or_probes=True,
    reason='Incremental root override using a measured conservative whole-device upper bound; original WDDM process-peak-NA decision remains intact.')
write_new(path,record)
print(record['status'],free,util,slots,flush=True)
if allowed:print(launch('raw-native'),flush=True)
