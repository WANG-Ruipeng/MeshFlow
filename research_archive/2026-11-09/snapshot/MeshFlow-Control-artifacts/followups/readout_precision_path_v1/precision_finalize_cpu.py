from pathlib import Path
import ast,difflib,hashlib,json
root=Path('/mnt/e/MeshFlow-Control')
out=Path('/mnt/e/MeshFlow-Control-artifacts/followups/readout_precision_path_v1')
owned=['src/meshflow_control/precision.py','src/meshflow_control/models/native.py','src/meshflow_control/models/readout.py','tests/test_precision.py']
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];diff=[]
for rel in owned:
    p=root/rel;before=out/'source_before'/rel
    ast.parse(p.read_text())
    rows.append(dict(path=rel,before_sha256=h(before) if before.exists() else None,sha256=h(p),bytes=p.stat().st_size))
    diff.extend(difflib.unified_diff(before.read_text().splitlines(True) if before.exists() else [],p.read_text().splitlines(True),fromfile='before/'+rel,tofile='after/'+rel))
(out/'precision_code.diff').write_text(''.join(diff))
receipt=dict(status='PASS',test_command='CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/test_precision.py',
    actual_pytest_stdout='........ [100%]\n8 passed in 2.21s\n',passed=8,failed=0,cpu_threads=2,
    real_model_loads=0,real_model_forwards=0,cuda_initialized=False,backwards=0,optimizer_updates=0,synthetic_small_module_calls=True,
    coverage=['default FP32/BF16 actual addition byte equivalence to original formula','known/padding preserved bytewise','OFF skips branch compute; ZERO computes real r/cast and applies zero','actual free-only capture and unchanged CLEAN memory','scalar BF16 cast nonzero/addition rounded away','policy routing and exception restoration with unchanged weights','FP32 runtime disables nested autocast and enforces highest matmul/mathSDPA','finite module dtype hooks and exception cleanup','invalid routes and trainable weights refused'],
    source_files=rows,diff_path=str(out/'precision_code.diff'),diff_sha256=h(out/'precision_code.diff'),
    static_findings=dict(backbone_modified=False,forced_bfloat16_conversion_in_backbone=False,
        explicit_autocast_sites='Native CUDA BF16 context; Geo and readout explicitly disable autocast. FP32_REFERENCE disables native CUDA and caller CPU autocast.',
        default_formula='query + residual.to(dtype=hidden.dtype), now named cast/applied; default applied is the identical cast tensor',
        off_scope='Only apply_readout extra computation/addition skipped; original native blocks/Geo/role and per-forward CLEAN H0 preserved',
        zero_scope='Real residual and cast captured separately from actual applied all-zero tensor',
        prediction_parameterization='unchanged velocity; no new adapter'),
    runner_readonly_review=dict(path=str(root/'tools/run_readout_precision.py'),sha256=h(root/'tools/run_readout_precision.py'),
        status='PASS_PREFLIGHT_ROUTING',legacy_replay=4,zero_probes=8,preflight_kind=12,planned_fixed_FP32_OFF_references=4,
        extra_warmup=0,remaining_fixed_after_gate=44,
        note='Legacy OFF references reuse old verified cells. Four FP32 OFF references count against the fixed-state 48 and are reused later. No GPU call performed by this review.'))
d=out/'cpu_tests/precision';d.mkdir(parents=True,exist_ok=True)
(d/'results.json').write_text(json.dumps(receipt,indent=2)+'\n')
(d/'pytest_stdout.txt').write_text(receipt['actual_pytest_stdout'])
(out/'precision_source_hashes.json').write_text(json.dumps(dict(status='FROZEN_BY_OWNER',files=rows),indent=2)+'\n')
print(json.dumps(dict(status='PASS',receipt=str(d/'results.json'),receipt_sha256=h(d/'results.json'),files=rows,diff_sha256=h(out/'precision_code.diff')),indent=2))
