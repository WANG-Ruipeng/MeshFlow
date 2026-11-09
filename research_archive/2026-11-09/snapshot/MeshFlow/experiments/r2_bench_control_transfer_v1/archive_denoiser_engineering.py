"""Archive already completed CPU-only loader/config and scale review evidence.

No model loading, model forward, geometry scoring, or frozen-source changes.
"""
from pathlib import Path
import datetime
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = RUN / 'audits/denoiser_engineering_00'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = data.encode('utf-8') if isinstance(data, str) else data
    if path.exists():
        assert path.read_bytes() == data, str(path)
    else:
        path.write_bytes(data)


def run():
    post = ROOT / 'r2/pilots/bench_control_transfer_v1/postprocess.py'
    after = post.read_bytes()
    new = (
        "    # Official Bench uses fixed_std=.05; Chair's independent contract is .02.\n"
        "    # This is the checkpoint's identity-encoder setting, not a new noise draw.\n"
        "    expected_kwargs = dict(D.INIT_KWARGS, fixed_std=0.05)\n"
        "    if kwargs != expected_kwargs or '02828884' not in str(data['data_path']):\n"
    ).encode()
    old = b"    if kwargs != D.INIT_KWARGS or '02828884' not in str(data['data_path']):\n"
    assert after.count(new) == 1
    before = after.replace(new, old)
    cpu_audit = json.loads((RUN / 'official_denoiser_cpu_audit.json').read_text('utf-8'))
    assert sha(after) == cpu_audit['identity']['loader_sha256']
    assert cpu_audit['status'] == 'PASS'
    assert cpu_audit['identity']['init_kwargs']['fixed_std'] == .05
    save(OUT / 'postprocess.before_reconstructed.py.txt', before)
    save(OUT / 'postprocess.after_cpu_verified.py.txt', after)
    diff = ''.join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
        fromfile='postprocess.py.before_reconstructed', tofile='postprocess.py.after_cpu_verified'))
    save(OUT / 'code.diff', diff)
    official = 'experiments/r2_postprocess_avmr_pmr_v1/source_snapshot/official_denoiser_db86bc3/'
    sources = [
        'datasets/mesh_dataset.py', 'train.py', 'configs/snet/base-120m-ot-v-bench.yaml',
        official + 'datasets/mesh_dataset.py', official + 'train_vae.py', official + 'post_mesh.py',
        'r2/src/meshflow_control/denoise_model.py', 'r2/pilots/bench_control_transfer_v1/data_prepare.py',
        'r2/pilots/bench_control_transfer_v1/postprocess.py',
    ]
    source_hashes = {}
    for relative in sources:
        value = (ROOT / relative).read_bytes()
        source_hashes[relative] = sha(value)
        save(OUT / 'source_snapshot' / relative, value)
    receipts = {}
    for name in ('official_denoiser_cpu_audit.json', 'official_denoiser_cpu_preflight_failure_00.json',
                 'official_denoiser_cpu_resources.json'):
        data = (RUN / name).read_bytes()
        receipts[name] = sha(data)
        save(OUT / name, data)
    report = '''# Bench D0 loader and scale evidence

The official Bench D0 strict CPU load passed with 296 FP32 EMA state keys and
455,592,963 parameters. No D0 forward or CUDA initialization occurred in the
preflight. The actual checkpoint is synset `02828884`, SHA256
`67765c728c503b61c21624c05997ea026c9de486d1529052fafa134be3336b14`.

The first pre-forward configuration guard expected the Chair identity-encoder
`fixed_std=0.02`; the actual official Bench checkpoint declares `fixed_std=0.05`.
Only the independent Bench loader expectation was corrected. The official
checkpoint, architecture, scientific sampling definition, frozen scorer,
training code, and sample code were not changed. The retained official proposal
API calls `pe_nerf` and residual `decode`, without drawing an encoder posterior.

The before snapshot is explicitly reconstructed from the recorded single patch;
its hash is calculated now, not claimed to have been captured before the failed
preflight. The after hash matches the loader hash saved by the successful strict
CPU load. The failure and success receipts remain unchanged and are copied here.

## Normalization contract

Let `v` be the source coordinates after the official shared mesh normalization to
the interval [-0.5, 0.5]. The shared official generator `train.py` explicitly
constructs the dataset with `do_dataset_normalize=True`; that dataset sets
`std=0.3762` and emits `2*v/0.3762`. The Bench config uses this same dataset and
synset 02828884. The new Bench data adapter performs the same transformation.

The locally archived official D0 `train_vae.py` explicitly sets `vae=True` for
training and validation. Its dataset divides by its separate dataset standard
deviation only when `not self.vae`, so that branch is skipped for D0 even when
`do_dataset_normalize=True`. Its emitted D0 coordinates are `0.95*v/0.5 = 1.9*v`.
Consequently the generator-to-D0 multiplier is `(1.9)/(2/0.3762) = 0.35739`,
exactly `0.95*0.3762` in the retained wrapper. The inverse returns the proposal
to generator coordinates. This is supported by the shared training code and
actual Bench configuration, not an assumption that a Chair-only scale applies.
The archived official inference `post_mesh.py` consumes coordinates at the
[-0.95, 0.95] scale without a category-specific replacement.

## One-time CLI/dependency review

- Run in the repository root with `r2/src` on PYTHONPATH; the existing WSL venv
  already imported the required scoring and D0 dependencies successfully.
- Postprocessing refuses to start before both arms have all 48 registered RAWs,
  matching hashes and 50 generator forwards. It uses one D0 call per output,
  then exact C/seam-anchor restoration, original GEO_C and source winding.
- Root dispatch must enforce GPU resource headroom before each D0 worker;
  postprocess records resources but does not independently choose concurrency.
- Unresolved D0/geometry attempts stop instead of silently replaying. A saved
  proposal can finish geometry without another forward when no geometry attempt
  has started. Complete outputs are hash-verified; scoring verifies all POST
  artifact files again.
- Old assets, scoring definitions and prior results are read-only. New files
  belong to this independent Bench experiment. No GT reconstruction score is
  introduced; ordinary C/F metrics remain applicable to PLAIN, while structured
  seam-only metrics are NA for PLAIN, not zero.
- Conditional aggregation risk: the existing aggregator discovers numerical
  fields by whether any row has a finite value. A legitimately all-NA field in
  this category could trigger its 124-column guard. No scorer change was made.
  If this occurs, reuse the already recorded 124 names in
  evaluation/input_audit/receipt.json and preserve the NA values in aggregation;
  no additional geometry scans are required.

This archive adds no forwards, samples, optimizer updates, noise draws, queries,
geometry scans, cleanup or source-code changes. It does not claim the future
GPU postprocessing has already completed.
'''
    save(OUT / 'REVIEW.md', report)
    result = dict(status='PASS_WITH_CONDITIONAL_AGGREGATION_NOTE',
        operation='Archive completed pre-forward engineering verification only',
        loader_fix=dict(before_sha256=sha(before), after_sha256=sha(after),
            before_snapshot_origin='Reconstructed from exact recorded single patch; not a contemporaneous before hash',
            after_snapshot_origin='Measured current hash matches successful CPU audit loader hash',
            old_expectation_fixed_std=.02, official_Bench_fixed_std=.05,
            changes_scientific_definition=False),
        normalization=dict(generator_std=.3762, generator_multiplier=2., D0_multiplier=1.9,
            scale=.35739, expression='.95 * .3762', category='Bench', synset='02828884',
            verified_shared_dataset=True, D0_vae_skips_dataset_std=True),
        source_hashes=source_hashes, prior_receipt_hashes=receipts,
        model_loads=0, model_forwards=0, GPU_calls=0, geometry_scans=0, queries=0,
        source_code_changes=0, frozen_scorer_unchanged=True,
        artifact_hashes={str(p.relative_to(OUT)):sha(p.read_bytes()) for p in sorted(OUT.rglob('*'))
                         if p.is_file() and p.name!='receipt.json'})
    save(OUT / 'receipt.json', json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifact_hashes','source_hashes','prior_receipt_hashes')}))


if __name__ == '__main__':
    run()
