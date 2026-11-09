"""Measured dispatch; no additional models or probe updates."""
import time
from .common import OUT, read, sha, write_new, resources

def release():
    r = resources()
    path = OUT / 'dispatch' / ('NATIVE_' + str(time.time_ns()) + '.json')
    write_new(path, r)
    lines = r['gpu']['stdout'].strip().splitlines()
    if r['gpu']['returncode'] or len(lines) != 1:
        raise RuntimeError('RESOURCE_WAIT_GPU_MEASUREMENT')
    cells = lines[0].split(',')
    total, used, free, util = [float(c.strip().split()[0]) for c in cells[1:5]]
    margin = max(2048., .1 * total)
    estimate = 4500.
    if free < margin + 1.2 * estimate or r['RAM_available'] < 8 * 2**30:
        raise RuntimeError('RESOURCE_WAIT_MEMORY')
    out = dict(authorized=True, stage='NATIVE', actual_training_workers=1,
        training_registration_sha256=sha(OUT/'training_registration.json'),
        resource_snapshot=str(path), free_mib=free, used_mib=used,
        compute_utilization_percent=util, retained_headroom_mib=margin,
        planning_process_peak_mib=estimate, process_peak_measured=False,
        reference='Same Native BF16 batch8/micro1 predecessor: allocator reserved~3.2GiB; WDDM process peak NA. Observe actual formal updates.',
        reason='One authorized training arm. Other GPU work follows measured compute and memory availability.',
        no_unbudgeted_probe=True, automatic_replay=False)
    dest=OUT/'dispatch/NATIVE_go.json'
    if dest.exists():
        old=read(dest)
        if old['training_registration_sha256'] != out['training_registration_sha256']:
            raise ValueError('Dispatch registration changed')
        return old
    write_new(dest,out)
    return out

if __name__=='__main__': print(release())
