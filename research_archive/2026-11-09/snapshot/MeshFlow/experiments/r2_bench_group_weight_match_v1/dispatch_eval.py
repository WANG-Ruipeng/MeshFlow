"""Explicit one-job measured release; no automatic next stage or retry."""
import argparse,json,math,time
from r2.pilots.bench_group_weight_match_v1.common import OUT,WIDE_ROOT,read,resources,sha,write_new
from r2.pilots.bench_group_weight_match_v1 import evaluation as E
from r2.pilots.bench_group_weight_match_v1.workflow import launch

def main():
    p=argparse.ArgumentParser();p.add_argument('job',choices=('sample-m','d0-m'));a=p.parse_args()
    t=read(OUT/'training/W_MATCHED/status.json')
    assert t['status']=='COMPLETE' and t['stage_step']==1500
    if a.job=='sample-m':
        observation=next((WIDE_ROOT/'evaluation/W').glob('sampling_attempt_*/resource_observation.json'))
    else:
        E.all_raw_barrier()
        observation=next((WIDE_ROOT/'evaluation/W').glob('D0_resource_observer_*/resource_observation.json'))
    peak=read(observation)['summary']['total_device_used_peak_mib']
    r=resources();v=[x.strip() for x in r['gpu']['stdout'].strip().split(',')]
    total=float(v[1].split()[0]);free=float(v[3].split()[0]);head=max(2048.,total*.1)
    slots=math.floor(max(0,free-head)/(1.2*peak))
    assert slots>=1 and r['RAM_available']>=16*1024**3,'Resource wait, do not dispatch'
    f=OUT/'dispatch'/(a.job.replace('-','_')+'_go_'+str(time.time_ns())+'.json')
    write_new(f,dict(authorized=True,job=a.job,resources=r,estimated_worker_bound_mib=peak,
        process_VRAM_peak=None,observation=str(observation),observation_sha256=sha(observation),
        estimation_scope='Exact original W46 same N/precision/model/stage; whole-device used peak is a conservative worker bound, not exclusive process measurement',
        additional_slots=slots,dispatched_workers=1,retained_headroom_mib=head,
        registration_sha256=sha(OUT/'training_registration.json'),
        scientific_scope='Only M46 already registered; all RAW before D0. No calibration or extra forwards.',
        no_automatic_retry=True))
    print(json.dumps(launch(a.job),indent=2))

if __name__=='__main__':main()
