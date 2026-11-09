"""Seal a complete fixed pilot after reports and independent audit; no model calls."""
import csv
import json
import sys
from . import eval_common as C


def main():
    root = C.OUT
    target = root / 'completion.json'
    if target.exists():
        old = C.read(target)
        for path, digest in old['sealed_files'].items():
            assert C.sha(path) == digest, path
        print(json.dumps(dict(status='COMPLETE_REVERIFIED', new_model_calls=0)))
        return
    audit = C.read(root / 'audit/final_execution.json')
    review = C.read(root / 'evaluation/visual_review_receipt.json')
    score = C.read(root / 'evaluation/score_completion.json')
    evidence = C.read(root / 'evidence/receipt.json')
    assert audit['status'] == review['status'] == 'PASS'
    assert score['status'] == evidence['status'] == 'COMPLETE'
    assert score['cells'] == review['row_count'] == 224
    raw_calls = d0_calls = 0
    for arm in C.ARMS:
        raw = C.read(root / 'evaluation' / arm / 'raw_complete.json')
        post = C.read(root / 'evaluation' / arm / 'post_complete.json')
        assert raw['status'] == post['status'] == 'COMPLETE'
        raw_calls += raw['new_rollouts']
        d0_calls += post['new_D0_forwards']
        if arm != 'LONG':
            status = C.read(root / 'training' / arm / 'status.json')
            assert status['status'] == 'COMPLETE' and status['local_step'] == 800
            assert not status['calls']['failed'] and not status['calls']['pending_call_ids']
    assert raw_calls == d0_calls == 96
    with (root / 'evaluation/per_output.csv').open(encoding='utf-8-sig', newline='') as f:
        assert len(list(csv.DictReader(f))) == 256
    report = root / 'REPORT.md'
    text = report.read_text(encoding='utf-8')
    assert len(text) > 5000 and '\ufffd' not in text
    names = ('REPORT.md', 'PROTOCOL.md', 'registration.json', 'execution_freeze.json',
             'audit/final_execution.json', 'evaluation/score_completion.json',
             'evaluation/summary_completion.json', 'evaluation/visual_review_receipt.json',
             'evaluation/per_output.csv', 'evaluation/paired_per_parent.csv',
             'evaluation/paired_per_noise.csv', 'evaluation/visual_review.csv',
             'analysis/tables_receipt.json', 'evidence/receipt.json',
             'training_analysis/final_resource_summary.json')
    assert 'torch' not in sys.modules
    C.write(target, dict(status='COMPLETE', utc=C.utc(), new_model_calls=0,
        fixed_training_updates_per_arm=800, new_raw_rollouts=96, new_D0_forwards=96,
        logical_requests_per_model=28, new_scoring_cells=224, total_score_rows_with_history=256,
        actually_viewed_pages=56, actual_AI_PROXY_rows=224,
        inference_shared_tip_nonnull_requests=0, automatic_promotion=False,
        committed=False, pushed=False, cleanup_performed=False,
        sealed_files={str(root / name): C.sha(root / name) for name in names},
        recovery='resume defaults to read-only status; sealed run refuses further dispatch.'))
    C.event('fixed_pilot_completed', completion=str(target), new_model_calls=0)
    print(json.dumps(dict(status='COMPLETE', raw=96, D0=96, reviews=224)))


if __name__ == '__main__':
    main()
