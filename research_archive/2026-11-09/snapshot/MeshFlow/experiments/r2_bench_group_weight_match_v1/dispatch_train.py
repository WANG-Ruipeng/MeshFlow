"""Root-owned measured release of the one authorized training job."""
import json
from r2.pilots.bench_group_weight_match_v1.common import OUT,resources,read,sha,write_new
from r2.pilots.bench_group_weight_match_v1.workflow import launch


def main():
    r=resources()
    values=[x.strip() for x in r['gpu']['stdout'].strip().split(',')]
    total=float(values[1].split()[0]);free=float(values[3].split()[0]);headroom=max(2048.,total*.1)
    reg=read(OUT/'training_registration.json')
    assert reg['status']=='FROZEN' and reg['arms']==['W_MATCHED'] and reg['updates']==1500
    assert free>=9000 and r['RAM_available']>=16*1024**3,'Resource wait, no dispatch'
    go=dict(authorized=True,arm='W_MATCHED',training_registration_sha256=sha(OUT/'training_registration.json'),
        resources=r,retained_headroom_mib=headroom,minimum_free_mib=9000,workers_after_dispatch=1,
        comparable='Completed WIDE_HIGH same model/N776/batch8/micro1/AdamW/precision; scoped allocator4.03GB reserved4.37GB; process VRAM NA, whole-device peak8863MiB included overlap',
        policy='Only one authorized training arm; no extra arm/probe. Sampling depends on final checkpoint.',
        numeric_threads_per_worker=2,CPU_overlap='Bounded independent cached-evidence preparation',
        user_budget='One W_MATCHED1500 endpoint; B/W fully read-only',no_automatic_retry=True,
        process_peak_limit='WDDM process memory may be unavailable; record whole-device observer and per-update allocator separately')
    write_new(OUT/'dispatch/W_MATCHED_go.json',go)
    print(json.dumps(launch('train-matched'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
