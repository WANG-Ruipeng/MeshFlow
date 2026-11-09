"""Explicit recovery entry. Never silently replays the two lost calls."""
from .common import *
import argparse,subprocess

def main():
    # Finalized experiments are read-only. Old recovery flags must never replay them.
    if (OUT/'COMPLETE.json').exists():
        from .closeout import verify_complete
        verify_complete();return
    if all((OUT/'execution'/label/'complete.json').exists() for label in LABELS):
        print('All registered model calls are complete. No GPU replay. Use closeout --verify-only after final review.');return
    p=argparse.ArgumentParser();p.add_argument('--gpu-recovered',action='store_true');p.add_argument('--allow-two-forward-replay',action='store_true');a=p.parse_args()
    if not(a.gpu_recovered and a.allow_two_forward_replay):
        print('BLOCKED: no model calls. GPU health must be restored and the logged2-call replay explicitly allowed. Completed12 outputs stay cached.');return
    # Bounded health query. No synthetic CUDA probe, driver reset or training.
    status=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    if status.returncode:raise RuntimeError(status.stderr)
    verify(manifest()['files'])
    verify(read(OUT/'layouts/complete.json')['files'])
    events=[json.loads(x) for x in (OUT/'execution/COMMON/calls.jsonl').read_text().splitlines()]
    broken=read(OUT/'evaluation_inputs/manifest.json')['requests'][12]['request_id']
    calls=[x for x in events if x['request_id']==broken]
    assert len(calls)==4 and [x['stage'] for x in calls]==['STARTED','RETURNED','STARTED','RETURNED']
    folder=OUT/'evaluation/COMMON'/broken/'raw';archive=folder.with_name('raw_failed_2call_attempt')
    for path in (folder,archive):assert path.resolve().is_relative_to(OUT.resolve())
    if folder.exists() and not archive.exists():folder.rename(archive)
    save(OUT/'recovery_authorization.json',dict(gpu_query=status.stdout,replayed_calls=2,reason='lost unsaved partial trajectory after CUDA illegal memory access',fixed_inputs=True,training_updates=0,
        expected_total_generator_calls=2382,core_max=2480,source_layout_identical_skips=100))
    marker=OUT/'gpu_dispatch_blocked.json'
    if marker.exists():marker.rename(OUT/'gpu_dispatch_blocked_history.json')
    save(OUT/'recovery_dispatch.json',dict(resources=resources(),gpu_workers=1,cpu_threads_per_GPU_worker=1,
        rationale='Observed prior device/NVML failure has unknown cause. Serial recovery isolates model loading and execution; CPU analysis may overlap. No inference batch/precision/solver change.',
        authorization='User 2026-10-22: continue; previous message disclosed lost2 forwards and remaining fixed experiment',
        source_hashes={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}))
    # Conservative serial recovery; no simultaneous loading after observed device failure.
    for label in LABELS:subprocess.run([sys.executable,'-B','-m','r2.pilots.bench_trajectory_diagnostic_v1.gpu',label],check=True)
    for label in LABELS:
        subprocess.run([sys.executable,'-B','-m','r2.pilots.bench_parts_convergence_v1.postprocess','--root',str(OUT),'--label',label,'--barrier-labels',*LABELS],check=True)
    for mode in ('raw','post'):
        subprocess.run([sys.executable,'-B','-m','r2.pilots.bench_trajectory_diagnostic_v1.score',mode,'--workers','3'],check=True)
    print('Core inference and original scores completed. Remaining: functional activation analysis, full-stage gallery/review and final report. No automatic rollback or training.')
if __name__=='__main__':main()
