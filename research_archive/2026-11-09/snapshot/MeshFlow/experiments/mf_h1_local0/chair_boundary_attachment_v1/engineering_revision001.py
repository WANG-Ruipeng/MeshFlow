"""Recorded zero-call startup repair and independent pre-dispatch review fixes."""
from pathlib import Path
import difflib
import shutil
from .common import *
def run():
    folder=OUT/'engineering/revision001';folder.mkdir(parents=True,exist_ok=True)
    for p in [OUT/'protocol/precheck_sources.json',OUT/'protocol/protected_sources_before.json',OUT/'controller_status.json',OUT/'logs/precheck_attempt1.log',OUT/'accounting.json']:
        if p.exists():shutil.copyfile(p,folder/p.name)
    edits={}
    p=CODE/'worker.py';before=p.read_text();s=before
    s=s.replace("path=GPU_LOCK,pid=os.getpid()","path=GPU_LOCK,worker_pid=os.getpid()")
    s=s.replace("stage=name,pid=os.getpid()))","stage=name,worker_pid=os.getpid()))")
    s=s.replace("if p['arm_status'][arm]!='PASS':return dict(status='BLOCKED',arm=arm,reason=p['arm_status'][arm])", "if p['arm_status'][arm]!='PASS':\n                    blocked=dict(status='BLOCKED',arm=arm,reason=p['arm_status'][arm])\n                    atomic_json(status_path,blocked);return blocked")
    s=s.replace("resource_snapshot(out,'loaded_'+name)", "if path.resolve()!=START.resolve():assert meta['protocol_sha256']==json_hash(p)\n                resource_snapshot(out,'loaded_'+name)")
    edits[p]=(before,s)
    p=CODE/'protocol.py';before=p.read_text();s=before
    s=s.replace("'experiments/mf_h1_local0/chair_domain_control_v1/prepare.py')", "'experiments/mf_h1_local0/chair_domain_control_v1/prepare.py',\n    'experiments/mf_h1_local0/evaluate.py','experiments/mf_h1_local0/native_inpainting_metrics.py',\n    'experiments/mf_h1_local0/native_patch_seams.py','experiments/mf_h1_local0/guided_native_geometry_audit.py',\n    'experiments/mf_h1_local0/guided_native_geometry_geometry.py')")
    s=s.replace("def freeze_precheck(out=OUT):\n    out=Path(out)", "def freeze_precheck(out=OUT):\n    out=Path(out)\n    if (out/'protocol/precheck_sources.json').exists():return verify(out)")
    edits[p]=(before,s)
    diffs=[]
    for p,(before,after) in edits.items():
        (folder/(p.name+'.before')).write_text(before)
        p.write_text(after);compile(after,str(p),'exec')
        (folder/(p.name+'.after')).write_text(after)
        diffs+=difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=p.name+'.before',tofile=p.name+'.after')
    (folder/'code.diff').write_text(''.join(diffs))
    # This explicit repair only updates the current execution manifest; original
    # freeze files and source snapshot remain intact in revision001/precheck.
    import importlib
    from . import protocol
    importlib.reload(protocol)
    atomic_json(OUT/'protocol/precheck_sources.json',{str((CODE/name).relative_to(ROOT)):digest(CODE/name) for name in protocol.SCIENCE})
    atomic_json(OUT/'protocol/protected_sources_before.json',{p:digest(ROOT/p) for p in protocol.DEPENDENCIES})
    atomic_json(folder/'revision.json',dict(reason='Inherited append_event injects pid; duplicate explicit pid failed before CUDA/model calls. Also complete independent reviewer recovery/source-guard fixes.',
        model_forwards=0,network_backwards=0,optimizer_updates=0,scientific_math_changed=False,old_results_unchanged=True,
        updated_files=[str(p.relative_to(ROOT)) for p in edits],status=protocol.verify(OUT)['status']))
    print('Recorded revision001 verified; actual model calls still zero')
if __name__=='__main__':run()
