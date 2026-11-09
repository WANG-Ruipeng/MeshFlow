"""Verify existing delivery records and seal completion; stdlib, no model calls.

This does not rescore, regenerate, or reload checkpoints. Large-weight hashes
are inherited explicitly from the completed CPU content-audit receipts.
"""
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
LABELS = ['NATIVE_S21000', 'SEAM_S21000']
TAG = '__'.join(LABELS) + '_full'


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_exclusive(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    out = ROOT / 'completion.json'
    if out.exists():
        raise FileExistsError('Completion already sealed; do not overwrite it.')
    registration = read('registration.json')
    registration_sha = sha(ROOT / 'registration.json')
    assert registration_sha == 'b477f2dd77f0e1b216a80a86982086d045394d646faf3ce1950400856c1c7677'
    frozen = {'data/manifest.json': 'manifest_sha256',
              'data/input_schedule.json': 'input_schedule_sha256',
              'fixed_loss_probe/manifest.json': 'probe_manifest_sha256',
              'evaluation_inputs/manifest.json': 'evaluation_manifest_sha256'}
    frozen_hashes = {}
    for relative, key in frozen.items():
        frozen_hashes[relative] = sha(ROOT / relative)
        assert frozen_hashes[relative] == registration[key], relative
    decision = read('dispatch/decisions/step21000.json')
    assert decision['status'] == 'BUDGET_EXHAUSTED_NOT_CONVERGED'
    assert decision['current_global_step'] == 21000 and decision['stop_training']
    assert not read('dispatch/status.json')['owned_active']
    ready = read('report_ready.json')
    assert ready['GPU_tasks_finished'] and ready['endpoint'] == 21000
    requests = {r['request_id'] for r in read('evaluation_inputs/manifest.json')['requests']}
    assert len(requests) == 48
    table_dir = ROOT / 'evaluation' / 'tables' / TAG
    table_receipt = read(f'evaluation/tables/{TAG}/completion.json')
    assert table_receipt['status'] == 'COMPLETE'
    assert len(table_receipt['numeric_fields']) == 124
    with (table_dir / 'per_output.csv').open(encoding='utf-8-sig', newline='') as stream:
        scores = list(csv.DictReader(stream))
    expected = {(q, label, stage) for q in requests for label in LABELS for stage in ('RAW', 'POST')}
    keys = [(r['request_id'], r['label'], r['stage']) for r in scores]
    assert len(keys) == 192 and set(keys) == expected and len(set(keys)) == 192
    assert all(r['C_preserved_exact'].lower() == 'true' for r in scores)
    review_dir = ROOT / 'visual_review' / TAG
    reviews = [json.loads(s) for s in (review_dir / 'review.jsonl').read_text(encoding='utf-8').splitlines()]
    review_keys = [(r['request_id'], r['label'], r['stage']) for r in reviews]
    assert len(review_keys) == 192 and set(review_keys) == expected
    assert len(set(review_keys)) == 192
    assert all(r['actual_image_viewed'] and r['reviewer'] == 'AI_PROXY' and not r['metric_values_used_for_labels'] for r in reviews)
    receipt = read(f'visual_review/{TAG}/receipt.json')
    assert receipt['all_actual_image_hashes_match'] and receipt['comparison_rows'] == 96
    # Original review merge already checked all image hashes; verify no changed
    # reviewed image now, once per unique request.
    image_hashes = {}
    for row in reviews:
        q = row['request_id']
        if q not in image_hashes:
            image_hashes[q] = sha(ROOT / 'gallery' / TAG / (q + '.png'))
        assert image_hashes[q] == row['image_sha256']
    checkpoint_receipts = {}
    for arm in ('PLAIN', 'SEAM'):
        audit = read(f'dispatch/final_checkpoint_{arm}.json')
        assert audit['status'] == 'PASS' and audit['step'] == 21000
        assert audit['full_schema_model_optimizer_head_RNG_identity']
        assert audit['registration_sha256'] == registration_sha
        assert (ROOT / 'training' / arm / 'step21000.pt').stat().st_size > 1_000_000_000
        assert (ROOT / 'training' / arm / 'generator_step21000.pt').stat().st_size > 500_000_000
        checkpoint_receipts[arm] = audit
    accounting_rel = 'audits/execution_accounting_v1/snapshots/20261019T105654_981140Z/execution_counts.json'
    accounting = read(accounting_rel)
    assert accounting['status'] == 'FINAL' and accounting['consistency_issues'] == []
    totals = accounting['totals']
    assert totals['unique_generator_optimizer_updates'] == 30000
    assert totals['RAW_complete'] == totals['POST_complete'] == 312
    docs = [ROOT / 'REPORT.md', ROOT / 'DELIVERY_INDEX.md', review_dir / 'REVIEW.md']
    links = []
    for doc in docs:
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if target.startswith(('https:', 'http:', '#')):
                continue
            target = target.strip('<>').split('#', 1)[0]
            resolved = (doc.parent / target).resolve()
            assert resolved.exists(), (str(doc), target)
            links.append({'document': str(doc.relative_to(ROOT)), 'target': target})
    source_paths = sorted((WORKSPACE / 'r2/pilots/bench_parts_convergence_v1').glob('*.py'))
    source_paths += sorted((ROOT / 'scoring_source_snapshot').rglob('*.py'))
    source_paths += [Path(__file__), ROOT / 'report_support/build_numeric_summary.py',
                     ROOT / 'report_support/build_runtime_summary.py',
                     ROOT / 'audits/execution_accounting_v1/build.py',
                     review_dir / 'merge_completed_reviews.py']
    sources = {str(p.relative_to(WORKSPACE)): sha(p) for p in source_paths}
    seal = ROOT / 'audits/final_delivery_20261019'
    seal.mkdir(exist_ok=False)
    write_exclusive(seal / 'source_hashes.json', {'scope': 'final small source files only; prior engineering diffs remain in recovery audits', 'sha256': sources})
    checks = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'model_calls': 0, 'rescoring': False, 'checkpoint_tensor_reload': False,
              'registration_sha256': registration_sha, 'frozen_input_hashes': frozen_hashes,
              'final_rows': len(scores), 'numeric_fields': 124, 'actual_image_review_rows': len(reviews),
              'unique_review_images': len(image_hashes), 'linked_targets_verified': len(links),
              'split_score_rows': dict(Counter(r['split'] for r in scores)),
              'links': links, 'document_hashes': {str(p.relative_to(ROOT)): sha(p) for p in docs},
              'known_recovery_backward_unknown_count': totals['recovery_backward_unknown'],
              'final_weights_verified_by_existing_complete_CPU_audit': checkpoint_receipts}
    write_exclusive(seal / 'verification.json', checks)
    completion = {'schema': 'bench_parts_final_delivery_v1', 'status': 'COMPLETE',
                  'finished_utc': datetime.now(timezone.utc).isoformat(),
                  'training_stop_status': decision['status'], 'endpoint_global_step': 21000,
                  'unique_generator_updates': 30000, 'recovery_extra_completed_updates': 325,
                  'converged_both_arms': False, 'GPU_tasks_finished': True,
                  'evaluation_complete': True, 'actual_visual_review_complete': True,
                  'final_requests_per_model': 48, 'final_RAW': 96, 'final_POST': 96,
                  'all_process_RAW': 312, 'all_process_POST': 312,
                  'actual_final_review_stage_rows': 192, 'numeric_fields': 124,
                  'report': 'REPORT.md', 'delivery_index': 'DELIVERY_INDEX.md',
                  'report_sha256': sha(ROOT / 'REPORT.md'),
                  'verification_receipt': 'audits/final_delivery_20261019/verification.json',
                  'source_hashes': 'audits/final_delivery_20261019/source_hashes.json',
                  'accounting': accounting_rel, 'source_registration_sha256': registration_sha,
                  'earlier_phase_receipts_preserved': True, 'automatic_future_training': False,
                  'default_changed': False, 'commit_or_push': False,
                  'scientific_conclusion': 'C-side local interfaces improve; no DEV bulk reach; POST CF/FF/I costs remain. Not an overall quality promotion.'}
    write_exclusive(out, completion)
    print(json.dumps({k: completion[k] for k in ('status', 'training_stop_status', 'endpoint_global_step', 'final_RAW', 'final_POST', 'actual_final_review_stage_rows')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
