"""Resource-gated, incremental dispatch for registered tasks only."""
import argparse, json, time
from .common import OUT, resources, read, sha, write_new

def snapshot(tag='inspection'):
    value = resources()
    path = OUT / 'dispatch' / (tag + '_' + str(time.time_ns()) + '.json')
    write_new(path, value)
    return path, value

def release(stage):
    reg = read(OUT / 'training_registration.json')
    assert stage in reg['stage_updates']
    path, value = snapshot(stage)
    lines = value['gpu']['stdout'].strip().splitlines()
    if value['gpu']['returncode'] or len(lines) != 1:
        raise RuntimeError('RESOURCE_WAIT: GPU measurement unavailable')
    cells = [s.strip() for s in lines[0].split(',')]
    total, used, free, util = [float(s.split()[0]) for s in cells[1:5]]
    headroom = max(2048., total * .1)
    # Conservative planning estimate, NOT a measured process peak under WDDM.
    planning_peak = 4500.
    if free < 1.2 * planning_peak + headroom or value['RAM_available'] < 8 * 2**30:
        raise RuntimeError('RESOURCE_WAIT: insufficient measured resource margin')
    result = dict(authorized=True, stage=stage, actual_training_workers=1,
        training_registration_sha256=sha(OUT/'training_registration.json'),
        resource_snapshot=str(path), free_mib=free, used_mib=used,
        compute_utilization_percent=util, retained_headroom_mib=headroom,
        planning_process_peak_mib=planning_peak, process_peak_measured=False,
        reference='Comparable original FM BF16 batch8 micro1 worker reserved about3.2GiB; WDDM process peak unavailable. Observe first authorized real updates.',
        reason='One sequential curriculum; stage dependency requires Native then SEAM. Independent registered baseline sampling may overlap only after representative training peak and resource remeasurement.',
        no_unbudgeted_probe=True, automatic_replay=False)
    dest = OUT/'dispatch'/(stage+'_go.json')
    if dest.exists():
        old=read(dest)
        assert old['training_registration_sha256']==result['training_registration_sha256']
        return old
    write_new(dest,result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('NATIVE','SEAM'))
    a=p.parse_args()
    print(json.dumps(release(a.stage) if a.stage else snapshot()[1],indent=2))
