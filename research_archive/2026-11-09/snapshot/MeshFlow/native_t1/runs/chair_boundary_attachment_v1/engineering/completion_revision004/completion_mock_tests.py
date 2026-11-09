"""Mock the outer completion state machine; never call real run_all/finish."""
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from experiments.mf_h1_local0.chair_boundary_attachment_v1 import execute,run_all

folder=Path(__file__).resolve().parent
real_out=execute.OUT.resolve()
checks=[]
all_writes=[]
def require(name,ok):
    assert ok,name
    checks.append(dict(name=name,status='PASS'))
with tempfile.TemporaryDirectory(prefix='chair-completion-mock-') as td:
    test_base=Path(td).resolve()
    require('mock_root_is_outside_real_experiment',not test_base.is_relative_to(real_out))
    def guarded_write(path,value):
        p=Path(path).resolve()
        assert p.is_relative_to(test_base),str(p)
        assert not p.is_relative_to(real_out)
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value,indent=2))
        all_writes.append(str(p))
    def load(path):return json.loads(Path(path).read_text())
    def prior_stage_done(out,through='all'):
        guarded_write(Path(out)/'controller_status.json',dict(status='DONE',source='mock run_all'))
        return dict(status='COMPLETE')
    def fail_final(out):
        require('finish_observes_FINAL_AUDIT',load(Path(out)/'controller_status.json')['status']=='FINAL_AUDIT')
        raise RuntimeError('mock final audit failure')
    def success_final(out):
        require('successful_finish_observes_FINAL_AUDIT',load(Path(out)/'controller_status.json')['status']=='FINAL_AUDIT')
        result=dict(status='DONE',mock=True)
        guarded_write(Path(out)/'final_completion.json',result)
        guarded_write(Path(out)/'controller_status.json',dict(status='DONE',result=result))
        return result
    with patch.object(execute,'atomic_json',side_effect=guarded_write),patch.object(execute,'freeze_json',side_effect=guarded_write):
        target=test_base/'finish_failure'
        guarded_write(target/'final_completion.json',dict(status='DONE',stale=True))
        with patch.object(run_all,'run',side_effect=prior_stage_done) as upstream,patch.object(execute,'finish',side_effect=fail_final) as final:
            try:execute.run(target,through='all')
            except RuntimeError as exc:require('final_failure_propagates','mock final audit failure' in str(exc))
            else:raise AssertionError('final failure was swallowed')
            require('no_real_run_all_or_finish_called',upstream.call_count==final.call_count==1)
        for name in ('controller_status.json','final_completion.json'):
            result=load(target/name)
            require('finish_failure_blocks_'+name,result['status']=='BLOCKED' and result['final_audits_not_complete'] and 'mock final audit failure' in result['traceback'])
            require('stale_done_removed_'+name,'stale' not in result)
        target=test_base/'success'
        with patch.object(run_all,'run',side_effect=prior_stage_done),patch.object(execute,'finish',side_effect=success_final):
            result=execute.run(target,through='all')
        require('successful_return_done',result['status']=='DONE')
        require('successful_controller_done',load(target/'controller_status.json')['status']=='DONE')
        require('successful_final_completion_done',load(target/'final_completion.json')['status']=='DONE')
        target=test_base/'upstream_failure'
        with patch.object(run_all,'run',side_effect=RuntimeError('mock upstream failure')),patch.object(execute,'finish') as final:
            try:execute.run(target,through='all')
            except RuntimeError:pass
            else:raise AssertionError('upstream failure was swallowed')
            require('failed_upstream_does_not_call_finish',final.call_count==0)
        require('upstream_failure_blocks_both',all(load(target/p)['status']=='BLOCKED' for p in ('controller_status.json','final_completion.json')))
        target=test_base/'keyboard_interrupt'
        with patch.object(run_all,'run',side_effect=prior_stage_done),patch.object(execute,'finish',side_effect=KeyboardInterrupt):
            try:execute.run(target,through='all')
            except KeyboardInterrupt:pass
            else:raise AssertionError('interrupt was swallowed')
        require('BaseException_blocks_both',all(load(target/p)['status']=='BLOCKED' for p in ('controller_status.json','final_completion.json')))
        target=test_base/'limited_stage'
        with patch.object(run_all,'run',side_effect=prior_stage_done),patch.object(execute,'finish') as final:
            execute.run(target,through='train')
            require('limited_stage_skips_finish',final.call_count==0)
        require('limited_stage_does_not_claim_final_completion',not (target/'final_completion.json').exists())
    require('all_state_writes_confined_to_temporary_root',bool(all_writes) and all(Path(p).is_relative_to(test_base) for p in all_writes))
result=dict(status='PASS',check_count=len(checks),checks=checks,execute_source_sha256=hashlib.sha256(Path(execute.__file__).read_bytes()).hexdigest(),
    state_writes=len(all_writes),state_write_scope='OS TemporaryDirectory only; removed after test',
    real_run_all_calls=0,real_finish_calls=0,model_calls=0,geometry_score_calls=0,training_or_sampling_calls=0,
    real_OUT_mutations='Only this requested test script/receipt/log under engineering/completion_revision004; real controller/final_completion untouched',
    success_test_scope='Mock finish implements its documented DONE persistence; test confirms outer FINAL_AUDIT ordering and no overwrite')
(folder/'completion_mock_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
