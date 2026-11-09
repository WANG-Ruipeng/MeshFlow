"""Release the one authorized worker after live resource and frozen-input checks."""
import argparse,json,time
from .common import OUT, ARMS, resources, read, sha, write_new

def release():
    reg=read(OUT/'training_registration.json')
    if reg['arms']!=list(ARMS) or reg['updates']!=1500:
        raise ValueError('Only the registered one-arm1500 budget is authorized')
    snapshot=resources()
    text=snapshot['gpu']['stdout'].strip().splitlines()
    if len(text)!=1 or snapshot['gpu']['returncode']:
        raise RuntimeError('Cannot determine current GPU resources')
    values=[v.strip() for v in text[0].split(',')]
    def number(v):return float(v.split()[0])
    total,used,free,util=map(number,values[1:5])
    headroom=max(2048.,.1*total)
    estimate=4500.
    budget=1.2*estimate+headroom
    snapshot.update(measured_total_mib=total,measured_used_mib=used,measured_free_mib=free,
                    GPU_utilization=util,required_free_mib=budget,
                    process_peak_planning_mib=estimate,process_peak_is_measured=False,
                    allocation_reference='Previous USER_V2_S1500 same recipe: allocator peak reserved about3158MiB; device peak6666MiB includes background, process memory unavailable under WDDM. Conservative process planning4500MiB; remeasure first authorized real update')
    stamp=str(time.time_ns())
    write_new(OUT/'dispatch'/('resources_'+stamp+'.json'),snapshot)
    if free<budget or snapshot['RAM_available']<8*(1<<30):
        raise RuntimeError('RESOURCE_WAIT: preserve headroom, do not start')
    dest=OUT/'training_dispatch_go.json'
    payload=dict(authorized=True,arms=list(ARMS),training_registration_sha256=sha(OUT/'training_registration.json'),
                 actual_gpu_workers=1,reason='Only one authorized training arm; CPU work may overlap with bounded threads',
                 measured_resources=snapshot,retained_headroom_mib=headroom,
                 measured_first_update_required=True,unbudgeted_probe_jobs=0,
                 on_failure='Preserve attempt; no automatic replay or scientific setting changes')
    if dest.exists():
        old=read(dest)
        if old['training_registration_sha256']!=payload['training_registration_sha256']:
            raise ValueError('Dispatch registration changed')
        return dict(status='RELEASE_ALREADY_RECORDED',free_mib=free)
    write_new(dest,payload)
    return dict(status='RELEASED_ONE_WORKER',free_mib=free,headroom_mib=headroom)

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(release(),indent=2))
