"""Root-controlled single authorized training dispatch; no automatic retry."""
import json
from r2.pilots.bench_high_joint_coverage_v1.common import OUT,resources,sha,write_new,read
from r2.pilots.bench_high_joint_coverage_v1.workflow import launch

def main():
    r=resources()
    values=[x.strip() for x in r['gpu']['stdout'].strip().split(',')]
    total=float(values[1].split()[0]);free=float(values[3].split()[0]);headroom=max(2048.,total*.1)
    reg=read(OUT/'training_registration.json')
    assert reg['status']=='FROZEN' and reg['arms']==['WIDE_HIGH'] and reg['updates']==1500
    assert free>=9000 and r['RAM_available']>=16*1024**3,'Insufficient measured resources; no dispatch'
    go=dict(authorized=True,arm='WIDE_HIGH',training_registration_sha256=sha(OUT/'training_registration.json'),
        resources=r,retained_headroom_mib=headroom,minimum_free_mib=9000,workers_after_dispatch=1,
        policy='No comparable maxN776 training peak: one authorized formal arm only; first N776 at step4. No probe.',
        numeric_threads_per_worker=1,CPU_overlap='Independent four-RAW original geometry audit; resource record includes actual workers',
        user_budget='One WIDE_HIGH1500 endpoint; original narrowB read-only',no_automatic_retry=True,
        process_peak_limit='WDDM process accounting may be NA; keep aggregate GPU observer and per-update allocator scope')
    write_new(OUT/'dispatch/WIDE_HIGH_go.json',go)
    print(json.dumps(launch('train-wide'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
