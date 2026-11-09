"""One authorized frozen 32-request sampling wave under the maintained GPU lock."""
from native_t1 import runtime
import fcntl,time,traceback
import torch
from experiments.mf_h1_local0.chair_boundary_attachment_v1.resources import Monitor,before_dispatch,resource_snapshot
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.worker import sample_manifest
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.checkpoint import build_model
from experiments.mf_h1_local0.chair_domain_control_v1.checkpoint import tensor_state_hash
from .common import *
from .ledger import Budget
from .blend import SCHEMA
def run():
    runtime.configure_stable_runtime();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    protocol=read_json(OUT/'protocol.json');assert protocol['alpha']==.5
    for panel,expected in protocol['manifest_sha256'].items():assert digest(OUT/'manifests'/f'{panel}.json')==expected
    for path,expected in protocol['protected_files'].items():assert digest(path)==expected, path
    # Inference never loads original optimizer/RNG or an old training schema.
    side=read_json(WEIGHT.with_suffix('.json'));assert digest(WEIGHT)==side['checkpoint_sha256']
    p=torch.load(WEIGHT,map_location='cpu',weights_only=True,mmap=True)
    assert p['schema']==SCHEMA and p['metadata']['alpha']==.5 and 'optimizer' not in p
    assert tensor_state_hash(p['model'])==side['metadata']['model_state_sha256']
    stage='HALF_CAPPED500_OLD_FRESH'
    occupancy=ledger_rows(OUT/'ledger/gpu_occupancy.jsonl')
    assert sum(r['event']=='GPU_ACQUIRE' for r in occupancy)==sum(r['event']=='GPU_RELEASE' for r in occupancy)
    before_dispatch(OUT,stage)
    with open(GPU_LOCK,'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tick=time.perf_counter();append_event(OUT/'ledger/gpu_occupancy.jsonl',dict(event='GPU_ACQUIRE',stage=stage))
        monitor=Monitor(OUT,stage).start();budget=Budget();status='FAILED'
        try:
            model=build_model(p,'cuda',False)
            assert all(not x.requires_grad for x in model.parameters())
            for panel in ('OLD','FRESH'):
                sample_manifest(OUT,LABEL,OUT/'manifests'/f'{panel}.json',model,budget,side['checkpoint_sha256'])
            status='PASS'
        except BaseException as exc:
            atomic_json(OUT/'failures/sample.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
        finally:
            budget.snapshot();summary=monitor.close()
            summary['dispatch_reason']=protocol['GPU_plan'];atomic_json(OUT/'resources'/f'{stage}_summary.json',summary)
            seconds=time.perf_counter()-tick
            append_event(OUT/'ledger/gpu_occupancy.jsonl',dict(event='GPU_RELEASE',stage=stage,seconds=seconds,status=status))
            resource_snapshot(OUT,'after_'+stage);atomic_json(OUT/'status/sampling.json',dict(status=status,seconds=seconds))
            fcntl.flock(lock,fcntl.LOCK_UN)
    return dict(status=status,accounting=budget.snapshot())
if __name__=='__main__':
    import json
    print(json.dumps(run()))
