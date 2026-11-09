"""Measured, root-controlled overlap of the already-authorized eight B requests."""
import json,math
from pathlib import Path
from r2.pilots.bench_high_joint_coverage_v1.common import OUT,resources,sha,write_new,read,NARROW_ROOT
from r2.pilots.bench_high_joint_coverage_v1.workflow import launch

def main():
    training=read(OUT/'training/WIDE_HIGH/status.json')
    assert training['status']=='TRAINING' and training['stage_step']>=5
    logs=[json.loads(x) for x in (OUT/'training/WIDE_HIGH/training_log.jsonl').read_text().splitlines() if x.strip()]
    maxNrow=next(x for x in logs if x['stage_step']==4)
    historic=next((NARROW_ROOT/'evaluation/B').glob('sampling_attempt_*/resource_observation.json'))
    obs=read(historic)['summary'];peak=obs['total_device_used_peak_mib']
    r=resources();v=[x.strip() for x in r['gpu']['stdout'].strip().split(',')]
    total=float(v[1].split()[0]);free=float(v[3].split()[0]);util=float(v[4].split()[0])
    head=max(2048.,total*.1);slots=math.floor(max(0,free-head)/(1.2*peak))
    assert slots>=1 and util<85 and r['RAM_available']>16*1024**3,'Do not add worker without measured capacity'
    record=dict(authorized=True,job='sample-b',new_requests=8,forward_budget=400,resources=r,
        existing_training_step=training['stage_step'],maxN776_formal_step4=maxNrow,
        comparator_observation_path=str(historic),comparator_observation_sha256=sha(historic),
        observed_sampling_device_upper_bound_mib=peak,process_peak_mib=None,
        estimate_note='WDDM process usage unavailable. Prior same Native BF16/FP32 Euler50 N<=724 total-device peak used conservatively as worker bound; 1.2 margin, new maxN776. Not a measured new-worker peak.',
        additional_slots=slots,workers_after_dispatch=2,retained_headroom_mib=head,
        policy='One independent frozen-B sampling worker only, separately owned model/RNG/output. No D0 before all54 RAW.',
        no_automatic_retry=True,registration_sha256=sha(OUT/'training_registration.json'))
    write_new(OUT/'dispatch/sample_b_go.json',record)
    print(json.dumps(launch('sample-b'),indent=2))

if __name__=='__main__':main()
