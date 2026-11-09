"""Package existing evidence only; never imports or executes scientific code."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import zipfile

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT / 'native_t1/runs/chair_domain_control_v1'
SOURCE = ROOT / 'native_t1/runs/control_mvp48_v1'
HISTORY = ROOT / 'native_t1/runs/control_mvp48_budget_single_v1'
WORK = Path(__file__).resolve().parent.parent
FINAL = WORK.parent / (WORK.name + '.zip')
PARTIAL = FINAL.with_suffix('.zip.partial')
PREFIX = 'CHAIR_DOMAIN_CONTROL_DELIVERY/'
WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx'}
files = {}
excluded = []
checks = Counter()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name, value):
    (WORK / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def add(path, category, expected=None, archive_path=None, check_group=None):
    path = path.resolve()
    assert path.is_relative_to(ROOT.resolve()), str(path)
    assert path.is_file(), str(path)
    assert path.suffix.lower() not in WEIGHTS, str(path)
    rel = archive_path or path.relative_to(ROOT).as_posix()
    if rel not in files:
        files[rel] = {'source': path, 'categories': set(), 'expected': set(), 'groups': []}
    item = files[rel]
    assert item['source'] == path, rel
    item['categories'].add(category)
    if expected:
        assert len(expected) == 64, (rel, expected)
        item['expected'].add(expected)
    if check_group:
        assert expected, rel
        item['groups'].append(check_group)


def tree(base, category):
    for path in sorted(base.rglob('*')):
        if not path.is_file():
            continue
        reason = None
        if path.suffix.lower() in WEIGHTS:
            reason = 'Weight binary excluded by original delivery protocol'
        elif '__pycache__' in path.parts or path.suffix in {'.pyc', '.pyo'}:
            reason = 'Interpreter cache'
        elif path.suffix in {'.lock', '.tmp', '.partial'}:
            reason = 'Runtime lock or temporary file'
        if reason:
            excluded.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                             'reason': reason})
        else:
            add(path, category)


def mapping(data, base, category, group):
    for name, value in data.items():
        expected = value.get('sha256') if isinstance(value, dict) else value
        add(base / name, category, expected, check_group=group)


def checkpoint_records(node, evidence, target):
    if isinstance(node, dict):
        path, sha = node.get('path'), node.get('checkpoint_sha256')
        if isinstance(path, str) and sha and Path(path).suffix in WEIGHTS:
            target.setdefault(path, []).append({'identity_source': evidence, 'checkpoint_sha256': sha,
                                                'metadata': node.get('metadata', {}),
                                                'historical_hash_verified': node.get('hash_verified'),
                                                'origin': node.get('origin')})
        for value in node.values():
            checkpoint_records(value, evidence, target)
    elif isinstance(node, list):
        for value in node:
            checkpoint_records(value, evidence, target)


def build():
    started = time.time()
    assert not FINAL.exists(), 'Final package already exists; verify it rather than overwrite'
    assert not PARTIAL.exists(), 'Partial archive exists; inspect before a retry'
    state = read(OUT / 'run_state.json')
    assert state['active_stage'] is None
    assert len(state['stages']) == 23
    assert all(s['status'] == 'PASS' for s in state['stages'].values())
    accounting = read(OUT / 'execution_accounting.json')
    assert accounting['complete_generations'] == 582
    assert accounting['actual_repetitions'] == 0
    assert accounting['all_model_forward_attempts'] == 101415
    tree(OUT, 'current_experiment')
    protocol = read(OUT / 'protocol.json')
    mapping(read(OUT / 'registration/code_hashes.json'), ROOT, 'frozen_scientific_code', 'frozen_code')
    mapping(protocol['protected_adapter_files'], ROOT, 'protected_adapter', 'adapter_identity')
    mapping(protocol['source_scorer_identity'], ROOT, 'protected_scorer', 'scorer_identity')
    mapping(read(OUT / 'registration/source_identity.json')['source_hashes'], ROOT,
            'protected_model_dependency', 'source_code_identity')
    mapping(protocol['frozen_files'], OUT, 'frozen_registration', 'protocol_frozen_files')
    mapping(read(OUT / 'panels.json')['frozen_files'], OUT, 'frozen_input', 'panel_frozen_files')
    reporting = read(OUT / 'reporting_status.json')
    for artifact in reporting['artifacts']:
        add(OUT / artifact['path'], 'reported_artifact', artifact['sha256'], check_group='reporting_artifacts')
    ready = read(SOURCE / 'DATA_READY.json')
    mapping(ready['files'], SOURCE, 'source_data', 'data_ready_files')
    mapping(ready['code'], ROOT, 'source_data_code', 'data_code_identity')
    mapping(read(SOURCE / 'TRAIN_READY.json')['source_hashes'], ROOT,
            'source_training_dependency', 'train_ready_source_hashes')
    data = read(SOURCE / 'data/manifest.json')
    assert len(data['objects']) == 48
    for obj in data['objects']:
        add(ROOT / obj['source_path'], 'original_parent_npz', obj['source_sha256'], check_group='original_parents')
    for info in data['official_split_files'].values():
        add(ROOT / info['path'], 'official_split', info['sha256'], check_group='official_splits')
    mapping(data['source_code_and_registration'], ROOT, 'source_data_registration', 'source_data_registration')
    for name in ['DATA_READY.json', 'TRAIN_READY.json', 'DATA_BLOCKED.json']:
        add(SOURCE / name, 'source_identity')
    add(SOURCE / 'DATA_READY.json', 'source_identity', protocol['source_data_ready_sha256'], check_group='source_ready_identity')
    add(SOURCE / 'TRAIN_READY.json', 'source_identity', protocol['source_train_ready_sha256'], check_group='source_ready_identity')
    tree(SOURCE / 'registration', 'source_registration')
    tree(SOURCE / 'training', 'source_training_evidence')
    for path in sorted((SOURCE / 'checkpoints').glob('*.json')):
        add(path, 'source_checkpoint_identity')
    for path in sorted((ROOT / 'experiments/mf_h1_local0/control_mvp48').glob('*.py')):
        add(path, 'source_mvp48_code')
    for name in ['MVP48_REPORT.md', 'gpu_run.json', 'scoring_run.json', 'eval_input_inventory.csv',
                 'teacher.csv', 'teacher_validation.csv', 'per_parent.csv']:
        add(SOURCE / 'evaluation' / name, 'source_evaluation_background')
    with (OUT / 'C_historical_background.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 192
    for row in rows:
        for field in ['raw', 'score']:
            add(Path(row[field + '_path']), 'historical_holdout_' + field, row[field + '_sha256'],
                check_group='historical_' + field)
    for name in ['ALL_IN_ONE_REPORT.md', 'protocol.json', 'checkpoint_manifest.json',
                 'execution_accounting.json', 'report_accounting.json', 'protected_assets_after.json',
                 'holdout_regression.csv', 'run_state.json', 'restart.txt']:
        add(HISTORY / name, 'historical_background_identity')
    for arm in ['T1', 'Geo', 'Full']:
        add(HISTORY / 'evaluation/holdout' / (arm + '_5000') / 'ledger.json', 'historical_holdout_ledger')
        add(HISTORY / 'logs' / ('holdout_' + arm + '_5000.log'), 'historical_holdout_ledger')
    for path in sorted((HISTORY / 'checkpoints').glob('*.json')):
        add(path, 'historical_checkpoint_identity')
    for name in ['AGENTS.md', 'WORKSPACE.md', 'README.md', 'requirements.txt', 'environment.yml', 'LICENSE']:
        if (ROOT / name).is_file():
            add(ROOT / name, 'repository_context')
    for name in ['START_HERE.md', 'RESULT_SUMMARY.md', 'EVIDENCE_INDEX.md', 'REVIEW_PROMPT.md',
                 'verify_package.py', 'packaging/build_package.py']:
        add(WORK / name, 'delivery_documentation_or_tool', archive_path=name)

    # Keep omitted identities separate from independently verified payload bytes.
    identities = {}
    for rel, item in list(files.items()):
        if item['source'].suffix == '.json' and ('checkpoint' in rel or '/checkpoints/' in rel):
            checkpoint_records(read(item['source']), rel, identities)
    for entry in excluded:
        if Path(entry['path']).suffix in WEIGHTS:
            identities.setdefault(str(ROOT / entry['path']), [])
    omissions = []
    for path, records in sorted(identities.items()):
        f = Path(path)
        assert f.is_relative_to(ROOT), path
        hashes = sorted({r['checkpoint_sha256'] for r in records})
        assert len(hashes) <= 1, 'Conflicting registered weight identities: ' + path
        omissions.append({'source_path': path, 'repository_path': f.relative_to(ROOT).as_posix(),
                          'exists_at_packaging': f.is_file(), 'bytes': f.stat().st_size if f.is_file() else None,
                          'registered_sha256': hashes[0] if hashes else None,
                          'weight_bytes_rehashed_during_packaging': False, 'identity_evidence': records,
                          'reason': 'Original protocol excludes duplicate large weights from delivery'})
    write('OMITTED_WEIGHTS.json', {'scope': 'Checkpoint identities present in included manifests/sidecars; not a repository-wide weight inventory',
                                 'warning': 'Recorded identities only. Weight binaries are not included or independently rehashed by this packaging job.',
                                 'weights': omissions})
    write('EXCLUSIONS.json', {'current_experiment_excluded_files': excluded,
                            'historical_scope': 'Only the 192 raw and 192 score files cited by C_historical_background.csv, plus named background/identity dependencies. Other old experiment outputs are out of scope.',
                            'third_party_software': 'Not included', 'model_weights': 'Not included; see OMITTED_WEIGHTS.json',
                            'official_dataset_scope': 'Only the 48 registered original parent files and referenced split metadata; not the full dataset',
                            'source_evaluation_redundant_aggregates': ['per_output.csv', 'results.json'],
                            'not_an_independent_runtime_distribution': True})
    for name in ['OMITTED_WEIGHTS.json', 'EXCLUSIONS.json']:
        add(WORK / name, 'delivery_manifest', archive_path=name)
    print(json.dumps({'phase': 'hashing_sources', 'files': len(files),
                      'uncompressed_bytes': sum(x['source'].stat().st_size for x in files.values())}), flush=True)
    assert shutil.disk_usage(WORK).free > sum(x['source'].stat().st_size for x in files.values()) * 1.1
    records = []
    snapshots = {}
    for index, (rel, item) in enumerate(sorted(files.items()), 1):
        p = item['source']
        before = p.stat()
        sha = sha_file(p)
        after = p.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), str(p)
        assert not item['expected'] or item['expected'] == {sha}, 'Frozen SHA256 mismatch: ' + rel
        for group in item['groups']:
            checks[group] += 1
        snapshots[rel] = (after.st_size, after.st_mtime_ns)
        records.append({'path': rel, 'source_repository_path': p.relative_to(ROOT).as_posix(),
                        'bytes': after.st_size, 'sha256': sha, 'categories': sorted(item['categories']),
                        'registered_sha256_verified': bool(item['expected'])})
        if index % 750 == 0:
            print(json.dumps({'phase': 'hashing_sources', 'completed': index}), flush=True)
    assert checks['frozen_code'] == 19
    assert checks['reporting_artifacts'] == 169
    assert checks['data_ready_files'] == 81
    assert checks['original_parents'] == 48
    assert checks['historical_raw'] == checks['historical_score'] == 192
    write('PACKAGE_CHECKS.json', {'status': 'PASS', 'checked_utc': datetime.now(timezone.utc).isoformat(),
                                'registered_hash_checks': dict(sorted(checks.items())),
                                'completed_stages': 23, 'completed_generations': 582, 'repeat_generations': 0,
                                'all_model_forward_attempts_in_existing_ledger': 101415,
                                'packaging_optimizer_updates': 0, 'packaging_model_forwards': 0,
                                'packaging_model_generations': 0, 'packaging_scoring_calls': 0,
                                'source_files_modified': False, 'weight_binaries_included': False})
    path = WORK / 'PACKAGE_CHECKS.json'
    records.append({'path': path.name, 'source_repository_path': path.relative_to(ROOT).as_posix(),
                    'bytes': path.stat().st_size, 'sha256': sha_file(path), 'categories': ['delivery_manifest'],
                    'registered_sha256_verified': False})
    files[path.name] = {'source': path}
    snapshots[path.name] = (path.stat().st_size, path.stat().st_mtime_ns)
    records.sort(key=lambda x: x['path'])
    manifest = {'schema': 'chair_evidence_delivery_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'archive_root': PREFIX.rstrip('/'), 'payload_file_count': len(records),
                'payload_uncompressed_bytes': sum(x['bytes'] for x in records),
                'source_root_mapping': {'/mnt/e/MeshFlow/': '<extracted package root>/',
                                        'E:\\MeshFlow\\': '<extracted package root>/'},
                'self_hash_note': 'This manifest excludes itself; the external ZIP SHA256 covers it.',
                'files': records}
    write('PACKAGE_MANIFEST.json', manifest)
    print(json.dumps({'phase': 'compressing', 'files': len(records)}), flush=True)
    with zipfile.ZipFile(PARTIAL, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for index, record in enumerate(records, 1):
            path = files[record['path']]['source']
            st = path.stat()
            assert (st.st_size, st.st_mtime_ns) == snapshots[record['path']], 'Source changed: ' + str(path)
            z.write(path, PREFIX + record['path'])
            if index % 500 == 0:
                print(json.dumps({'phase': 'compressing', 'completed': index, 'of': len(records)}), flush=True)
        z.write(WORK / 'PACKAGE_MANIFEST.json', PREFIX + 'PACKAGE_MANIFEST.json')
    print(json.dumps({'phase': 'verifying_entire_archive', 'compressed_bytes': PARTIAL.stat().st_size}), flush=True)
    spec = importlib.util.spec_from_file_location('delivery_verify', WORK / 'verify_package.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verified = verifier.verify(PARTIAL)
    for rel, item in files.items():
        st = item['source'].stat()
        assert (st.st_size, st.st_mtime_ns) == snapshots[rel], 'Source changed during packaging: ' + rel
    PARTIAL.rename(FINAL)
    zip_sha = sha_file(FINAL)
    FINAL.with_suffix('.zip.sha256').write_text(zip_sha + '  ' + FINAL.name + '\n', encoding='ascii')
    verified.update({'zip_path': str(FINAL), 'zip_bytes': FINAL.stat().st_size, 'zip_sha256': zip_sha,
                     'archive_member_count_including_manifest': len(records) + 1,
                     'registered_hash_checks': dict(sorted(checks.items())),
                     'source_sizes_and_mtimes_unchanged': True,
                     'wall_seconds': time.time() - started, 'status': 'PASS'})
    write('package_verification.json', verified)
    print(json.dumps(verified, indent=2), flush=True)


if __name__ == '__main__':
    build()
