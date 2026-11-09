"""Freeze conservative metric numerical fixes before any formal training."""
import difflib
import shutil
from .common import *
def run():
    from .ledger import Budget
    account=Budget(OUT,'revision002').snapshot()
    assert account['attempts'].get('optimizer_update',0)==0 and account['attempts'].get('training_forward',0)==0
    assert read_json(OUT/'metric_tests_v1r1/metric_test_results.json')['status']=='PASS'
    folder=OUT/'engineering/revision002_integration';folder.mkdir(parents=True,exist_ok=True)
    for p in [OUT/'protocol.json',OUT/'protocol/protocol_identity.json',OUT/'protocol/precheck_sources.json',OUT/'accounting.json']:
        shutil.copyfile(p,folder/p.name)
    edits={}
    for name in ('protocol.py','run_all.py'):
        p=CODE/name;before=p.read_text();s=before.replace('metric_tests_final/metric_test_results.json','metric_tests_v1r1/metric_test_results.json')
        if name=='protocol.py':s=s.replace("'protocol.py','control_metrics.py','metric_tests.py')","'protocol.py','control_metrics.py','metric_tests.py','run_all.py')")
        edits[p]=(before,s)
    p=CODE/'sampling.py';before=p.read_text();s=before.replace("assert pointer['metadata']['attachment_step']==1000", "assert pointer['metadata']['attachment_step']==1000\n    assert digest(path)==pointer['checkpoint_sha256']")
    edits[p]=(before,s)
    diffs=[]
    for p,(before,after) in edits.items():
        (folder/(p.name+'.before')).write_text(before);p.write_text(after);compile(after,str(p),'exec')
        (folder/(p.name+'.after')).write_text(after)
        diffs+=difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=p.name+'.before',tofile=p.name+'.after')
    (folder/'code.diff').write_text(''.join(diffs))
    import importlib
    from . import protocol
    importlib.reload(protocol)
    atomic_json(OUT/'protocol/precheck_sources.json',{str((CODE/name).relative_to(ROOT)):digest(CODE/name) for name in protocol.SCIENCE})
    p=read_json(OUT/'protocol.json')
    p.update(metric_tests_sha256=digest(OUT/'metric_tests_v1r1/metric_test_results.json'),metric_source_sha256=digest(CODE/'control_metrics.py'),precheck_sources_sha256=digest(OUT/'protocol/precheck_sources.json'))
    atomic_json(OUT/'protocol.json',p)
    atomic_json(OUT/'protocol/protocol_identity.json',dict(file_sha256=digest(OUT/'protocol.json'),semantic_sha256=json_hash(p)))
    assert protocol.freeze_training(OUT)==p
    atomic_json(folder/'revision.json',dict(status='PASS',reason='Conservative numerical metric fix and source identity integration before formal updates; strict endpoint pointer identity check',
        scientific_thresholds_changed=False,model_or_loss_math_changed=False,training_calls_before_revision=0,
        retained_real_precheck_forwards=44,retained_real_precheck_backwards=16,retained_zero_replays=2,
        numerical_revision=str(OUT/'engineering/metric_revision002'),protocol_sha256=json_hash(p)))
    print('Revision002 frozen; 44F/16B and two exact replays retained without repetition')
if __name__=='__main__':run()
