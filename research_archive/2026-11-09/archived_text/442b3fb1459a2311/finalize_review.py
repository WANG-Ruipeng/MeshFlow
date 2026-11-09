"""Package the completed saved-data analysis; never execute models or scorers."""
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = Path('/mnt/e/MeshFlow')
OLD = WORK / 'native_t1/runs/chair_control_feedback_auto_v1'
assert ROOT == WORK / 'native_t1/runs/chair_capped500_offline_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))


def write(relative, data):
    (ROOT / relative).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


expected = {}
for mapping in (
    read('geometry/receipt.json')['source_hashes'],
    read('distribution/source_identity.json')['source_files'],
    read('dynamics/source_hashes.json')['read_only_sources'],
    {
        str(OLD / 'CONTROL_FEEDBACK_AUTONOMOUS_REPORT.md'): '0cd6e8a91b7ed92d43f95d9ce09dd7f62465b390e834483a4da053083cc9c031',
        str(OLD / 'delivery/CONTROL_FEEDBACK_AUTONOMOUS_REVIEW.zip'): 'b22d4f09027ff1bb603d1920348b4f8d46db6e105dad5214dd7328e25f79bb1c',
    },
):
    for path, value in mapping.items():
        assert path not in expected or expected[path] == value, path
        expected[path] = value
for path, value in expected.items():
    assert digest(path) == value, ('source changed', path)

visual_src = OLD / 'visuals/DEV_START_explore_FM_CONTROL_500_explore_REPLAY_FM_CONTROL_500_explore_TRAJECTORY_REPLAY_500_explore_TRAJECTORY_REPLAY_CAPPED_500_d8832e1ee9af'
reviewed = {
    'c1b8c2f0c4f3cc43aff7c08b060f5ed6.png': 'Both noises, all four views and wireframes; free-back fragmentation and seat artifacts.',
    '866e7248434968051cc7750d0afca796.png': 'Both noises, all four views and wireframes; round-back opening benefit and remaining thin fragments.',
    '341bcae1e7f366a233d93f57729688db.png': 'Both noises, all four views and wireframes; noise1 lower-body gaps and fragmentation.',
    '534b0e6e2c53a8e1a7cdfd3874ed3723_feedback.png': 'Both noises, complete saved S/I/E overlay only; known-seat duplicate coverage, especially noise1.',
}
(ROOT / 'visuals').mkdir(exist_ok=True)
visual_receipt = []
for name, scope in reviewed.items():
    source = visual_src / name
    target = ROOT / 'visuals' / name
    shutil.copy2(source, target)
    assert digest(source) == digest(target)
    visual_receipt.append({'source': str(source), 'copy': str(target.relative_to(ROOT)), 'sha256': digest(source), 'bytes': source.stat().st_size, 'view_scope': scope})
write('visuals/review_scope.json', {
    'review_type': 'AI_VISUAL_REVIEW', 'human_review': 'NOT_RUN',
    'mesh_outputs_actually_reviewed_this_followup': 6,
    'additional_overlay_outputs_actually_reviewed_this_followup': 2,
    'whole_DEV_re_reviewed': False, 'new_mesh_render_calls': 0,
    'display_method': 'Original full PNGs displayed with lossless adjacent vertical in-memory crops for large mesh pages; saved copies are unchanged source bytes.',
    'columns': ['C', 'START', 'FM500', 'teacher500', 'TR500', 'CAPPED500', 'GT'],
    'files': visual_receipt,
})

code_sources = {
    'evaluate.py': 'experiments/mf_h1_local0/evaluate.py',
    'control_metrics.py': 'experiments/mf_h1_local0/chair_boundary_attachment_v1/control_metrics.py',
    'scoring.py': 'experiments/mf_h1_local0/chair_control_usability_quick_v1/scoring.py',
    'evaluation_wrapper.py': 'experiments/mf_h1_local0/chair_control_feedback_auto_v1/evaluation.py',
    'loss_balance.py': 'experiments/mf_h1_local0/chair_control_feedback_auto_v1/loss_balance.py',
    'training.py': 'experiments/mf_h1_local0/chair_control_feedback_auto_v1/training.py',
}
(ROOT / 'source_scoring').mkdir(exist_ok=True)
copied_code = []
for name, original in code_sources.items():
    source = WORK / original
    target = ROOT / 'source_scoring' / name
    shutil.copy2(source, target)
    assert digest(source) == digest(target)
    copied_code.append({'original': str(source), 'copy': str(target.relative_to(ROOT)), 'sha256': digest(source), 'bytes': source.stat().st_size})
write('source_scoring/SOURCE_HASHES.json', copied_code)
(ROOT / 'code.diff').write_bytes(b'')

expected_rows = {
    'geometry/query_per_point.csv': 40960,
    'distribution/CAPPED500_parent_digest.csv': 8,
    'distribution/CAPPED500_output_digest.csv': 16,
    'dynamics/per_step.csv': 4000,
    'dynamics/actual_hooks.csv': 123,
}
inventory = {}
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or 'delivery' in path.relative_to(ROOT).parts or '__pycache__' in path.parts:
        continue
    relative = path.relative_to(ROOT).as_posix()
    if relative in ('PROVENANCE.json', 'VERIFICATION.json'):
        continue
    entry = {'sha256': digest(path), 'bytes': path.stat().st_size}
    if path.suffix == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = 0
            for row in reader:
                assert len(row) == len(header), (relative, rows)
                rows += 1
        entry.update(rows=rows, columns=len(header))
        if relative in expected_rows:
            assert rows == expected_rows[relative], (relative, rows)
    elif path.suffix in ('.md', '.json', '.py'):
        content = path.read_text(encoding='utf-8-sig')
        assert '\ufffd' not in content, relative
        if path.suffix == '.json':
            json.loads(content)
    inventory[relative] = entry
assert all(x in inventory for x in expected_rows)
assert read('geometry/receipt.json')['checks'] == 768
assert read('dynamics/completion.json')['checks'] == 80094

now = datetime.now(timezone.utc).isoformat()
write('PROVENANCE.json', {
    'task': 'CAPPED500 saved-result followup', 'completed_utc': now,
    'old_run': str(OLD), 'source_hashes_rechecked_unchanged': expected,
    'copied_actual_source': copied_code,
    'outputs': inventory,
    'archive_omission': 'distribution/distribution_results.json is retained locally but omitted from ZIP; it duplicates the complete CSV tables and summary JSON that are included.',
    'hash_scope': 'This index excludes itself, VERIFICATION.json, and delivery to avoid circular hashes; delivery receipt hashes the final ZIP.',
    'large_weights': 'No checkpoint tensor load or rehash; availability/stat and prior hashes are in dynamics/checkpoint_availability.csv.',
    'code_diff': 'Empty: maintained scientific code unchanged. New analysis scripts exist only in this separate output tree.',
})
write('VERIFICATION.json', {
    'status': 'PASS', 'completed_utc': now,
    'source_hashes_rechecked_count': len(expected), 'source_mismatches': 0,
    'csv_count': sum('rows' in entry for entry in inventory.values()),
    'csv_total_data_rows': sum(entry.get('rows', 0) for entry in inventory.values()),
    'required_row_counts': expected_rows,
    'geometry_reconstruction_checks': 768, 'dynamics_checks': 80094,
    'ledger': {'training_updates': 0, 'model_forward_calls': 0, 'generated_meshes': 0, 'resampled_noise': 0, 'resampled_queries': 0, 'original_scorer_runs': 0, 'gpu_dispatches': 0, 'CONFIRM_evaluations': 0, 'new_mesh_renders': 0},
    'new_CPU_geometry': 'Existing GT512/2048 query to original C finite-triangle distances for descriptive spatial partitions only; no model or model scorer.',
    'NA': ['900-step DEV outputs absent', 'actual optimizer parameter-step norms not logged', 'independent training replication not run', 'new-noise stability not tested', 'human visual review not run'],
    'START_default_changed': False, 'old_gate_changed': False, 'git_commit_or_push': False,
})

delivery = ROOT / 'delivery'
delivery.mkdir(exist_ok=True)
archive = delivery / 'CAPPED500_DEEP_DIVE_REVIEW.zip'
members = sorted(p for p in ROOT.rglob('*') if p.is_file() and 'delivery' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts and p.relative_to(ROOT).as_posix() != 'distribution/distribution_results.json')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for path in members:
        z.write(path, path.relative_to(ROOT).as_posix())
assert archive.stat().st_size < 10_000_000, archive.stat().st_size
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(members)
    for path in members:
        assert hashlib.sha256(z.read(path.relative_to(ROOT).as_posix())).hexdigest() == digest(path)
receipt = {'status': 'PASS', 'archive': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest(archive), 'members': len(members), 'all_member_hashes_verified': True, 'contains_raw_or_weights': False, 'complete_CSV_rows_included': True, 'omitted_redundant_local_file': 'distribution/distribution_results.json'}
write('delivery/PACKAGE_RECEIPT.json', receipt)
print(json.dumps(receipt))
