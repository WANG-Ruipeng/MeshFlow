"""Freeze finite forward budget and resource/input identities before model calls."""
from .common import *
from datetime import datetime, timezone

def run():
    dest=OUT/'registration.json'
    if dest.exists(): return read(dest)
    OUT.mkdir(parents=True,exist_ok=True)
    write_new(OUT/'resources_initial.json',resources())
    inputs=[ROOT/'AGENTS.md',ROOT/'WORKSPACE.md',ROOT/'native_t1/docs/control_metrics.md',
            ROOT/'r2/docs/research_route_memory.md',ROOT/'r2/docs/research_route_registry.json',
            ROOT/'experiments/r2_bench_seat_coverage_v1/eval_manifest.json',
            ROOT/'experiments/r2_bench_full_seat_curriculum_v1/eval_manifest.json',OUT/'PROTOCOL.md']
    dependencies=[ROOT/'r2/pilots/bench_parts_convergence_v1/data.py',
      ROOT/'r2/pilots/seam_abc_v1/slot_data_extension.py',
      ROOT/'r2/src/meshflow_control/seam/slot_data.py',
      ROOT/'r2/src/meshflow_control/training/coupling.py',
      ROOT/'r2/src/meshflow_control/seam/model_adapter.py',
      ROOT/'r2/src/meshflow_control/models/native.py',
      ROOT/'r2/pilots/bench_control_transfer_v1/sample.py']
    models={}
    for label,p in MODELS.items():
        side=read(p.with_suffix('.json')); h=sha(p)
        assert side['sha256']==h, label
        models[label]=dict(path=str(p),sha256=h,bytes=p.stat().st_size,sidecar_sha256=sha(p.with_suffix('.json')),
                           probe_forwards=20 if label=='SEAM_S3000' else 60)
    value=dict(status='FROZEN',experiment='R2_BENCH_FLOW_INTEGRITY_DIAGNOSTIC_V1',
       registered_utc=datetime.now(timezone.utc).isoformat(),models=models,
       budget=dict(updates=0,backwards=0,head_forwards=0,CPU_OT=16,probe_forwards=200,
          cached_trajectory_max=108,conditional_Euler100_rollouts=4,conditional_Euler100_forwards=400,conditional_D0=4),
       initial_GPU_workers=1,additional_workers='Only after representative measured peak; 1.2x incremental peak plus max(2GiB,10% total) headroom',
       CPU_threads_per_worker=1,protected={str(p):sha(p) for p in inputs+dependencies},
       risks=['Existing source-owner vs C-only deployment layout difference is explicit, not proof of OT bug',
              'No official-unconditional-vs-conditional equivalence claim','Controlled probes contain target information, not rollout success',
              'Euler100 small sensitivity test is not a converged numerical reference'])
    write_new(dest,value); print({k:v for k,v in value.items() if k not in ('models','protected')})
    return value
if __name__=='__main__':run()
