"""Finite-task completion gate. Reads evidence only; forbids implicit restart."""
from __future__ import annotations
import csv, datetime, hashlib, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'experiments/r2_seam_hinge_quality_v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    done = OUT/'completion.json'
    if done.exists():
        old = read(done)
        for p, h in old['evidence_hashes'].items():
            assert sha(Path(p)) == h, p
        return dict(status='ALREADY_COMPLETE_VERIFIED', no_dispatch=True)
    required = {
        'audit/execution.json': 'PASS',
        'evaluation_audit/completion.json': 'PASS',
        'training_summary/completion.json': 'COMPLETE',
        'hinge/completion.json': 'COMPLETE',
        'hinge/baseline_audit.json': 'PASS_READ_ONLY_AUDIT',
        'hinge/endpoint_audit.json': 'PASS_READ_ONLY_AUDIT',
        'evaluation/main_score_completion.json': 'COMPLETE',
        'binding/main/completion.json': 'COMPLETE',
        'source_snapshot.json': 'COMPLETE',
        'review/completion.json': 'COMPLETE_LINKS_VERIFIED',
        'visual/hinge_baseline_completion.json': 'ACTUAL_REVIEW_COMPLETE',
        'visual/smooth_actual_review_completion.json': 'ACTUAL_REVIEW_COMPLETE',
    }
    evidence = {}
    for name, status in required.items():
        p = OUT/name
        assert read(p)['status'] == status, name
        evidence[str(p)] = sha(p)
    for name in ('REPORT.md', 'METHODS.md', 'RUN.md', 'WEIGHTS.json', 'decision.json',
                 'code.diff', 'report_receipt.json', 'results/final/completion.json',
                 'visual/CLEAN_EDGE_COH_four_stage_review_completion.json',
                 'visual/CLEAN_EDGE_COH_comparison_completion.json',
                 'visual/CLEAN_EDGE_COH_HINGE_review_completion.json',
                 'visual/smooth_hinge_actual_review_completion.json',
                 'visual/BOUND015_four_tail_cases_actual_review.json',
                 'sections/fixed_cases/actual_four_stage_review.json',
                 'sections/fixed_cases/actual_hinge_review.json'):
        p = OUT/name
        assert p.is_file(), name
        evidence[str(p)] = sha(p)
    qids = {r['request_id'] for r in read(OUT/'protocol.json')['evaluation']['requests']}
    coverage = {}
    for name in ('CLEAN_EDGE_COH_four_stage_actual_review.csv',
                 'CLEAN_EDGE_COH_vs_PLAIN_S2000_POST_actual_review.csv',
                 'smooth_actual_review.csv', 'hinge_baseline_pages.csv',
                 'CLEAN_EDGE_COH_HINGE_actual_review.csv', 'smooth_hinge_actual_review.csv'):
        p = OUT/'visual'/name
        with p.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        seen = {r['request_id'] for r in rows}
        assert seen == qids, (name, seen ^ qids)
        coverage[name] = dict(requests=len(seen), observation_rows=len(rows))
        evidence[str(p)] = sha(p)
    # The final source snapshot also covers this completion gate, its report
    # writer and all actual review recording helpers. Verify without reloading models.
    snapshot = read(OUT/'source_snapshot.json')
    for r in snapshot['files']:
        assert sha(Path(r['source_path'])) == r['sha256']
        assert sha(Path(r['snapshot_path'])) == r['sha256']
    for p, h in read(OUT/'report_receipt.json')['outputs'].items():
        assert sha(Path(p)) == h
    started = json.loads((OUT/'ledger.jsonl').read_text().splitlines()[0])['unix']
    now = time.time()
    result = dict(status='COMPLETE_STOPPED', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_wall_seconds_since_protocol=now-started,
        actual_budget=dict(arms=2, optimizer_updates=2000, train_forward=16000, train_backward=16000,
            new_RAW=56, generator_inference_forwards=2800, D0_forwards=56,
            original_definition_scores=112, bound_cells=56, hinge_cells=168,
            aggregate_rows=504, original_metric_fields=124, additional_topology_fields=11),
        actual_visual_coverage=coverage,
        visual_scope='Two arms full28 four-stage+POST; each of three models full28 HINGE. BOUND actual visual follow-up limited to four declared cost cases. AI_PROXY_NONBLIND.',
        evidence_hashes=evidence, model_calls_in_closeout=0, new_scoring_in_closeout=0,
        default_changed=False, cleanup=False, commit=False, push=False,
        automatic_further_dispatch=False)
    with (OUT/'ledger.jsonl').open('a', encoding='utf-8') as f:
        for stage, name in [('TRAINING','training_summary/completion.json'),
                            ('SCORING','evaluation/main_score_completion.json'),
                            ('BOUND','binding/main/completion.json'),
                            ('HINGE','hinge/completion.json'),
                            ('AUDIT','audit/execution.json'), ('CLOSEOUT','report_receipt.json')]:
            f.write(json.dumps(dict(event=stage+'_COMPLETE_RECORDED_AT_CLOSEOUT', unix=now,
                                    receipt=name, receipt_sha256=sha(OUT/name)))+'\n')
    done.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return dict(status=result['status'], elapsed_wall_seconds_since_protocol=now-started,
                actual_budget=result['actual_budget'], no_further_dispatch=True)


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
