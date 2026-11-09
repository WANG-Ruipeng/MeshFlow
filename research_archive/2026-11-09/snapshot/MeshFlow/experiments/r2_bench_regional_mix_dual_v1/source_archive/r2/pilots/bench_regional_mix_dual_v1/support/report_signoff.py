"""Bind the completed actual review and independent numerical check to REPORT."""
from r2.pilots.bench_regional_mix_dual_v1.common import OUT, read, sha, write, verify, utc


def run():
    report = OUT / 'REPORT.md'
    reviewed_sha = 'e3f035d664003998d007b20cde41a5d1c100ed86807066a9d43fffb387a13722'
    assert sha(report) == reviewed_sha
    inputs = {}
    for name, status in [
        ('audits/execution/complete.json', 'PASS'), ('analysis/COMPLETE.json', 'COMPLETE'),
        ('visual_review/COMPLETE.json', 'COMPLETE'), ('dispatch/complete.json', 'COMPLETE'),
        ('render_queue/complete.json', 'COMPLETE'), ('gallery/index.json', 'COMPLETE'),
    ]:
        path = OUT / name
        record = read(path)
        assert record['status'] == status, (name, record['status'])
        if 'files' in record:
            verify(record['files'])
        inputs[str(path)] = sha(path)
    v = read(OUT / 'visual_review/summary.json')
    assert (v['actual_pages'], v['unique_outputs']) == (24, 42)
    assert v['counts']['ASSIST11']['three_axis_joint'] == 4
    assert v['counts']['ASSIST11']['support_PRESENT'] == 6
    s = read(OUT / 'analysis/summary.json')['stages']['CLEAN_FINAL_POST']['ALL']
    assert (s['parent_count'], s['request_count'], s['I_common_PASS_requests']) == (3, 6, 6)
    target = OUT / 'audits/report_review.json'
    if target.exists():
        old = read(target)
        assert old['report_sha256'] == reviewed_sha and old['status'] == 'PASS'
        return old
    receipt = dict(
        status='PASS', utc=utc(), report_sha256=reviewed_sha, reviewer='AI_PROXY_root',
        independent_reviewer='AI_PROXY_backend_data',
        independent_review_evidence='Actual collaboration message/final review on 2026-10-28: PASS for this exact REPORT SHA; no source or score rerun.',
        checks={
            'all_main_metrics_and_three_parent_tables_match': True,
            'area_denominators_distinguished': True,
            'legal_edge_losses_and_new_illegal_pairs_disclosed': True,
            'all_24_pages_actually_reviewed_42_unique_outputs': True,
            'visual_nonblind_AI_PROXY_and_uncertain_disclosed': True,
            'assisted_and_automatic_results_separate': True,
            'all_assisted_combinations_preserved_no_best_selection': True,
            'six_development_requests_not_generalization': True,
            'all_177_values_preserved_and_NA_retained': True,
            'zero_model_training_D0_or_new_generation': True,
            '23_physical_jobs_46_new_scores_no_failure_or_replay': True,
            'report_15_existing_delivery_links_checked': True,
            'continuation_receipt_limitation_documented': True,
            'no_default_change_cleanup_commit_push': True,
        },
        checked_stage_receipts=inputs,
        scientific_calls=0, new_visual_labels=0,
    )
    write(target, receipt)
    return receipt


if __name__ == '__main__':
    import json
    print(json.dumps(run(), ensure_ascii=False))
