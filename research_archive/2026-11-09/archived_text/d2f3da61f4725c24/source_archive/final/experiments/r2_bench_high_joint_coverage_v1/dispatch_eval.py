"""Explicit one-job release after resource and dependency checks."""
import argparse,json,math,time
from r2.pilots.bench_high_joint_coverage_v1.common import OUT,NARROW_ROOT,read,resources,sha,write_new
from r2.pilots.bench_high_joint_coverage_v1 import evaluation as E
from r2.pilots.bench_high_joint_coverage_v1.workflow import launch

def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',choices=('sample-w','d0-b','d0-w'));a=parser.parse_args()
    if a.job=='sample-w':
        train=read(OUT/'training/WIDE_HIGH/status.json')
        assert train['status']=='COMPLETE' and train['stage_step']==1500
        source=next((NARROW_ROOT/'evaluation/B').glob('sampling_attempt_*/resource_observation.json'))
        estimated=read(source)['summary']['total_device_used_peak_mib']
        evidence='Prior same Native Euler50 N<=724 isolated total-device peak used as conservative worker bound, plus20% margin. Current B maxN776 already completed without pressure alongside training; its mixed device peak is not charged as a sampling process peak.'
    else:
        E.all_raw_barrier()
        current=list((OUT/'evaluation/B').glob('D0_resource_observer_*/resource_observation.json'))
        source=current[0] if current else next((NARROW_ROOT/'evaluation/B').glob('D0_resource_observer_*/resource_observation.json'))
        estimated=read(source)['summary']['total_device_used_peak_mib']
        evidence=('Current official Bench D0 B8 including N776 completed; total-device peak used as conservative worker bound with20% margin.' if current else 'Prior same official Bench D0 N<=724 total-device peak used as conservative worker bound,20% margin; current maxN776 measured in authorized D0 outputs.')
    r=resources();v=[x.strip() for x in r['gpu']['stdout'].strip().split(',')]
    total=float(v[1].split()[0]);free=float(v[3].split()[0]);head=max(2048.,.1*total)
    slots=math.floor(max(0,free-head)/(1.2*estimated))
    assert slots>=1 and r['RAM_available']>=16*1024**3,'Insufficient measured headroom; do not dispatch'
    dest=OUT/'dispatch'/(a.job.replace('-','_')+'_go_'+str(time.time_ns())+'.json')
    write_new(dest,dict(authorized=True,job=a.job,resources=r,estimated_worker_bound_mib=estimated,
        actual_process_peak_mib=None,observation=str(source),observation_sha256=sha(source),
        estimation_scope=evidence,additional_slots=slots,dispatched_workers=1,retained_headroom_mib=head,
        registration_sha256=sha(OUT/'training_registration.json'),no_extra_calls=True,no_automatic_retry=True))
    print(json.dumps(launch(a.job),indent=2))

if __name__=='__main__':main()
