"""One explicit engineering recovery: no neural call occurred in failed v1 entry."""
import ast,json,time
from .common import ROOT,OUT,read,sha,write_new

def main():
    receipt=OUT/'engineering/pre_forward_import_recovery.json'
    if receipt.exists():return read(receipt)
    label='RELATION_V1_S1500'
    failure=OUT/'workflow/raw-baseline/output.log'
    trace=failure.read_text()
    assert 'AssertionError: Invalid device id' in trace
    assert 'with torch.no_grad(), torch.autocast' in trace
    src=OUT/'evaluation'/label/'layouts/27eb79ddf26fd7ac7898580cb9d16864__R1__THREE_FAMILY_V1'
    events=[json.loads(x) for x in (src/'calls.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events]==['HEAD_FORWARD_STARTED']
    assert set(p.name for p in src.iterdir())=={'attempt.json','calls.jsonl'}
    # Existing baseline RAW products are only eight authenticated historical copies.
    rr=list((OUT/'evaluation'/label).glob('*/raw/receipt.json'))
    assert len(rr)==8 and all(read(p)['execution']=='REUSED' for p in rr)
    assert {p.parent for p in (OUT/'evaluation'/label).glob('*/raw/calls.jsonl')} <= {p.parent for p in rr}
    code=ROOT/'r2/pilots/bench_newdata_joint_v1'
    ast.parse((code/'sample_v2.py').read_text())
    assert 'from r2.data_tools.bench_three_family_v1.production_contract import' not in (code/'sample_v2.py').read_text()
    frozen=read(OUT/'training_registration.json')['training_code_hashes']
    assert all(sha(p)==h for p,h in frozen.items())
    dest=OUT/'engineering/failed_v1_layout_before_forward'
    root=OUT.resolve()
    assert src.resolve().is_relative_to(root) and dest.resolve().is_relative_to(root) and not dest.exists()
    hashes={str(p):sha(p) for p in src.iterdir()}
    dest.parent.mkdir(parents=True,exist_ok=True)
    src.rename(dest)
    result=dict(status='RECOVERY_PREPARED_NO_NEURAL_REPLAY',time=time.time(),
        failure_log_sha256=sha(failure),preserved_failed_layout=str(dest),original_files=hashes,
        v1_sha256=sha(code/'sample.py'),v2_sha256=sha(code/'sample_v2.py'),
        attempted_context_entries=1,actual_head_forward_calls=0,generator_forwards=0,D0_forwards=0,
        extra_generator_loads=1,weights_noise_data_threshold_solver_unchanged=True,
        training_code_and_registration_unchanged=True,
        cause='CPU-only provenance helper import sets CUDA_VISIBLE_DEVICES empty after GPU load',
        correction='Use independently verified native head digest and avoid CPU audit import',
        next='Explicit raw-baseline-v2 stage; retain original failed stage and its START ledger')
    write_new(receipt,result);return result

if __name__=='__main__':print(json.dumps(main()))
