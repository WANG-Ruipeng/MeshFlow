"""Pure-stdlib, read-only source capture for the contextual-alignment study.
Run after sources freeze; never imports experiment/model/scoring modules.
Only writes this study's source_code tree and code.diff. No archive is made.
"""
from __future__ import annotations
import argparse
import csv
import difflib
import hashlib
import io
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

REPO = Path('/mnt/e/MeshFlow-Control')
OLD_REPO = Path('/mnt/e/MeshFlow')
OUT = Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1')
NEW_PATTERNS = (
    'configs/coal_*.yaml', 'src/meshflow_control/data/contextual_stream.py',
    'src/meshflow_control/teachers/*.py', 'src/meshflow_control/training/context_*.py',
    'tools/check_contextual_stream.py', 'tools/contextual_alignment_*.py',
    'tools/prepare_contextual_alignment.py', 'tools/run_contextual_alignment.py',
    'tools/sample_contextual_alignment.py', 'tests/test_context_alignment.py',
    'tests/test_contextual_alignment_evaluation.py', 'tests/test_siglip_targets.py',
    'docs/contextual_alignment_evaluation.md', 'docs/contextual_alignment_teacher.md',
)
DOCS = ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'AGENTS.md', 'README.md',
        'docs/data_contract.md', 'docs/data_training_source_map.json',
        'docs/evaluation_source_map.json', 'docs/integration.md', 'docs/method.md',
        'docs/metrics.md', 'docs/model_source_map.json', 'docs/reproduction.md',
        'docs/source_map.json', 'docs/source_map.md')

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha(path):
    return sha_bytes(Path(path).read_bytes())

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def relpath(path):
    value = str(path).replace('\\', '/')
    if 'MeshFlow-Control/' in value:
        value = value.split('MeshFlow-Control/', 1)[1]
    return value

def immutable_write(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Refuse to replace a different existing evidence file: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def file_record(source, destination, role, **extra):
    data = source.read_bytes()
    immutable_write(destination, data)
    assert source.read_bytes() == destination.read_bytes()
    return dict(source_path=str(source), snapshot_path=str(destination.relative_to(OUT)),
                sha256=sha_bytes(data), bytes=len(data), role=role, **extra)

def added_diff(relative, data):
    # Preserve source line endings/BOM; this is a full-file addition, not a diff
    # against an absent Git HEAD. Verify exact reconstruction below.
    text = data.decode('utf-8')
    lines = text.splitlines(keepends=True)
    patch = list(difflib.unified_diff([], lines, fromfile='/dev/null', tofile='b/' + relative))
    if not patch:
        raise RuntimeError(f'Empty source addition not supported silently: {relative}')
    body = []
    for line in patch:
        body.append(line)
        if not line.endswith('\n'):
            body.append('\n\\ No newline at end of file\n')
    reconstruction = ''.join(line[1:] for line in patch[3:])
    if reconstruction.encode('utf-8') != data:
        raise RuntimeError(f'Diff reconstruction failure: {relative}')
    return f'diff --git a/{relative} b/{relative}\nnew file mode 100644\n' + ''.join(body)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.resolve() != OUT.resolve():
        raise ValueError('This evidence builder is bound to the registered external run')
    target = OUT / 'source_code'
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    frozen_path = OUT / 'audit/code_frozen_before_training.json'
    baseline_path = OUT / 'audit/source_before.json'
    frozen = {relpath(x['Path']): x['Sha256'].lower() for x in read_json(frozen_path)}
    baseline = {relpath(x['Path']): x['Sha256'].lower() for x in read_json(baseline_path)}
    checks = []
    for kind, mapping in [('before_training', frozen), ('original_src_baseline', baseline)]:
        for relative, expected in sorted(mapping.items()):
            source = REPO / relative
            actual = sha(source) if source.is_file() else None
            checks.append(dict(scope=kind, path=relative, expected_sha256=expected,
                               actual_sha256=actual, pass_hash=actual == expected))
    verification = dict(schema='contextual_alignment_source_verification_v1',
                        created_utc=stamp, status='PASS' if all(x['pass_hash'] for x in checks) else 'BLOCKED',
                        frozen_count=len(frozen), original_src_count=len(baseline), checks=checks,
                        imports_experiment_modules=False, model_calls=0, scores_recomputed=0)
    json_write(target / 'frozen_source_verification.json', verification)
    if verification['status'] != 'PASS':
        raise RuntimeError('Frozen source mismatch; evidence capture stopped')
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--verify', 'HEAD'], capture_output=True, text=True)
    if head.returncode == 0:
        raise RuntimeError('Unexpected Git HEAD; reconsider the explicit no-HEAD diff contract')
    new = sorted({str(p.relative_to(REPO)).replace('\\', '/')
                  for pattern in NEW_PATTERNS for p in REPO.glob(pattern) if p.is_file()})
    if len(new) != 27:
        raise RuntimeError(f'Expected 27 scoped new source/config/test/doc files, found {len(new)}')
    if any(x in baseline for x in new):
        raise RuntimeError('An explicitly new file appears in the pre-experiment src baseline')
    expected_configs = {'configs/coal_fm.yaml', 'configs/coal_free.yaml', 'configs/coal_known.yaml'}
    if {x for x in new if x.startswith('configs/')} != expected_configs:
        raise RuntimeError('Three experiment configs differ from fixed inventory')
    unfrozen_new = [x for x in new if x not in frozen]
    if set(unfrozen_new) != {'docs/contextual_alignment_evaluation.md', 'docs/contextual_alignment_teacher.md'}:
        raise RuntimeError(f'Unexpected non-frozen new executable source: {unfrozen_new}')
    records = []
    all_repository = sorted(set(frozen) | set(new) | set(DOCS))
    for relative in all_repository:
        role = ('new_experiment_file' if relative in new else
                'unchanged_pre_experiment_src' if relative in baseline else
                'preexisting_frozen_support_file' if relative in frozen else 'documentation_or_notice')
        records.append(file_record(REPO / relative, target / 'repository' / relative, role,
                                   repository_relative_path=relative,
                                   in_training_freeze=relative in frozen,
                                   in_original_src_baseline=relative in baseline,
                                   no_head_scope_note=('47-file original_src baseline does not establish historical status of old tools/configs/tests')))
    # Preserve notices directly next to the reviewable scoring snapshots too.
    for relative in ('LICENSE', 'THIRD_PARTY_NOTICES.md'):
        records.append(file_record(REPO / relative, target / relative, 'license_or_notice'))
    scorer_map = read_json(OUT / 'evaluation/source_map.json')
    scorer_checks = []
    for name, expected in sorted(scorer_map['scientific_scorers'].items()):
        source = REPO / 'src/meshflow_control/evaluation' / name
        snapshot = target / 'original_scoring' / name
        actual = sha(source)
        if actual != expected or sha(snapshot) != expected:
            raise RuntimeError(f'Original scorer identity mismatch: {name}')
        scorer_checks.append(dict(name=name, source_path=str(source), snapshot_path=str(snapshot.relative_to(OUT)),
                                  expected_sha256=expected, actual_sha256=actual, pass_hash=True))
        records.append(file_record(source, snapshot, 'actual_unchanged_scoring_module',
                                   repository_relative_path=str(source.relative_to(REPO))))
    old_map = read_json(REPO / 'docs/evaluation_source_map.json')
    origin_checks = []
    for item in old_map['files']:
        source = OLD_REPO / item['source']
        actual = sha(source) if source.exists() else None
        expected = item['source_sha256']
        destination = REPO / item['destination']
        check = dict(source_path=str(source), expected_sha256=expected, actual_sha256=actual,
                     migrated_path=item['destination'], migrated_expected_sha256=item['destination_sha256'],
                     migrated_actual_sha256=sha(destination),
                     source_hash_pass=actual == expected,
                     migrated_hash_pass=sha(destination) == item['destination_sha256'])
        origin_checks.append(check)
        if not check['source_hash_pass'] or not check['migrated_hash_pass']:
            raise RuntimeError(f'Historical scoring origin identity mismatch: {source}')
        records.append(file_record(source, target / 'historical_scoring' / item['source'],
                                   'historical_scoring_origin_read_only', migrated_destination=item['destination']))
    for name, expected in scorer_map['adapters'].items():
        actual = sha(REPO / 'tools' / name)
        if actual != expected:
            raise RuntimeError(f'Frozen experiment scoring adapter mismatch: {name}')
    for relative in ('audit/code_frozen_before_training.json', 'audit/source_before.json',
                     'audit/execution_binding.json', 'audit/git_status_initial.txt', 'evaluation/source_map.json'):
        source = OUT / relative
        records.append(file_record(source, target / 'identity' / relative, 'source_identity_receipt'))
    # These small audit helpers are execution evidence, not part of the training mathematics.
    for name in ('audit_contextual_training.py', 'freeze_execution_binding.py', 'live_status.py'):
        source = OUT / 'audit' / name
        if source.exists():
            records.append(file_record(source, target / 'audit_tools' / name, 'run_audit_helper'))
    diff = ''.join(added_diff(p, (REPO / p).read_bytes()) for p in new).encode('utf-8')
    immutable_write(OUT / 'code.diff', diff)
    json_write(target / 'scoring_identity_verification.json', dict(
        schema='contextual_alignment_scoring_identity_v1', status='PASS', created_utc=stamp,
        actual_scoring_count=len(scorer_checks), historical_origin_count=len(origin_checks),
        actual_scoring=scorer_checks, historical_origins=origin_checks,
        metric_definitions_changed=False, imports_experiment_modules=False,
        note='Historical originals and migrated standalone runtime scorers are distinct files; provenance hashes are verified separately.'))
    # Recheck the entire frozen scope after capture to detect concurrent source mutation.
    changed_after = [p for p,h in frozen.items() if sha(REPO / p) != h]
    if changed_after:
        raise RuntimeError(f'Frozen source changed during capture: {changed_after}')
    for rec in records:
        if sha(OUT / rec['snapshot_path']) != rec['sha256']:
            raise RuntimeError(f'Snapshot readback mismatch: {rec}')
    inventory = dict(schema='contextual_alignment_source_inventory_v1', status='PASS', created_utc=stamp,
                     repository=str(REPO), git_head_exists=False,
                     new_file_scope='Explicit task-owned additions; not inferred by subtracting incomplete baseline from freeze.',
                     new_files=new, new_file_count=len(new), new_source_config_test_count=len(new)-2,
                     documentation_not_in_training_freeze=unfrozen_new,
                     training_freeze_count=len(frozen), original_src_baseline_count=len(baseline),
                     repository_snapshot_count=len(all_repository), records=records,
                     exclusions=['weights', 'raw meshes', 'training outputs', 'large model assets', 'ZIP'],
                     code_diff=dict(path='code.diff', sha256=sha_bytes(diff), bytes=len(diff),
                                    files=len(new), base='/dev/null', reconstructed_exact_bytes=True),
                     baseline_limitation='The pre-experiment baseline records 47 src files only. Old tools/configs/tests are included as preexisting supporting files, not claimed new in code.diff.',
                     protected_source_read_only=True, model_calls=0, scoring_calls=0)
    json_write(target / 'source_inventory.json', inventory)
    fieldnames=['source_path','snapshot_path','sha256','bytes','role','repository_relative_path',
                'in_training_freeze','in_original_src_baseline']
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fieldnames, extrasaction='ignore', lineterminator='\n')
    writer.writeheader(); writer.writerows(records)
    immutable_write(target / 'source_inventory.csv', text.getvalue().encode('utf-8'))
    notice = '''# Source review evidence

This directory is a byte-preserving source snapshot, not a runnable model bundle.
No model, teacher weight, raw mesh, or training output is copied here. No ZIP is produced.

- `repository/`: all 94 files in the before-training freeze, plus the two new evaluation/teacher documents and necessary existing provenance documents and notices. Exact scope is in `source_inventory.json`.
- `original_scoring/`: the 19 **actual migrated standalone runtime scoring modules** used by this experiment. The historical directory name does not mean these are byte-identical to their earlier pre-migration files.
- `historical_scoring/`: 10 original small scoring sources from the read-only sibling workspace. Each old source hash and its separate migrated hash is verified in `scoring_identity_verification.json`.
- `repository/tools/contextual_alignment_{evaluation,analysis,gallery}.py`: the frozen experiment adapters; original metric definitions are unchanged.
- `repository/src/meshflow_control/training/context_alignment.py`: the new training-only attention-message capture/head. `repository/src/meshflow_control/models/native.py` and `models/backbone/{attention,equidit,utils}.py` preserve the exact model/capture dependencies as full source files.
- `repository/configs/coal_{fm,free,known}.yaml`: the three actual experiment recipes.
- `repository/tests/test_{context_alignment,contextual_alignment_evaluation,siglip_targets}.py`: task-specific CPU/contract tests. Capturing tests does not claim that any new test execution happened during source delivery.
- `identity/`: original before/at-training source identities and execution binding, copied without edits.
- `audit_tools/`: small run auditing helpers, distinguished from training/scoring code.
- `../code.diff`: 27 explicit new files as unified full-file additions against `/dev/null`, with exact byte reconstruction checked. This is **not** `git diff HEAD`: this independent repository has no HEAD. The initial source hash list covers only 47 src files; it cannot establish historical membership of every old tools/configs/tests file. Those supporting files are not falsely listed as new additions.

`source_inventory.{json,csv}`, `frozen_source_verification.json`, and `scoring_identity_verification.json` give the hashes, scope, classifications and snapshot checks. The source freeze is rechecked after copying. This is a source-only capture during the fixed run, not a final experiment completion audit.

## Notices

`LICENSE` and `THIRD_PARTY_NOTICES.md` are preserved byte-for-byte. The NVIDIA notice in the captured `models/backbone/utils.py` remains intact. Its independent distribution-license basis has not been verified; the repository MIT notice does not override that retained file notice. Historical source headers are also preserved. Model/teacher weights and datasets have separate terms and are not included.
'''
    immutable_write(target / 'README.md', notice.encode('utf-8'))
    receipt = dict(status='PASS', created_utc=stamp, frozen_sources_verified=len(frozen),
                   original_src_unchanged=len(baseline), new_diff_files=len(new),
                   repository_snapshot_files=len(all_repository), physical_snapshot_records=len(records),
                   actual_scoring_files=len(scorer_checks), historical_scoring_sources=len(origin_checks),
                   code_diff_sha256=sha_bytes(diff), code_diff_bytes=len(diff),
                   source_inventory_sha256=sha(target / 'source_inventory.json'),
                   builder_sha256=sha(Path(__file__)), total_snapshot_bytes=sum(x['bytes'] for x in records),
                   source_readback_pass=True, before_after_freeze_pass=True,
                   git_head_exists=False, model_calls=0, scores_recomputed=0, archive_created=False)
    json_write(target / 'capture_receipt.json', receipt)
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
