"""End-only verification of saved results. Never dispatches unfinished stages.

Run after training, sampling, scoring, gallery and actual review are complete.
Missing data produces a BLOCKED verification receipt; it never fabricates labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import sys
from audit_io import *

AXES = {
    'overall_shape': {'REASONABLE', 'INCOMPLETE', 'UNCERTAIN'},
    'C_use': {'PLAUSIBLE', 'REPLACED_OR_IGNORED', 'UNCERTAIN'},
    'support_retention': {'PRESENT', 'MISSING_OR_DAMAGED', 'UNCERTAIN'},
    'visible_fragment_or_gap': {'YES', 'NO_OBVIOUS', 'UNCERTAIN'},
    'layering_or_fold': {'YES', 'NO_OBVIOUS', 'UNCERTAIN'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_training(audit):
    registration = audit.read(OUT / 'registration.json')
    audit.verify(registration['data'])
    audit.verify({registration['source']: registration['source_sha256']})
    schedule = audit.read(OUT / 'data/schedule.json')['steps']
    coefficients = audit.read(OUT / 'data/coefficients.json')
    require(len(schedule) == len(coefficients) == 1000, 'Schedule budget')
    require([s['stage_step'] for s in schedule] == list(range(1, 1001)), 'Schedule order')
    streams = {}
    result = {}
    keys = ('condition_id', 'parent_id', 'noise_seed', 'epsilon_seed', 'time_seed',
            'permutation_seed', 'alpha', 'layout_drop', 'replay_group')
    for arm in ('PLAIN', 'LAYOUT'):
        path = OUT / 'training' / arm
        reg = audit.read(OUT / 'registrations' / (arm + '.json'))
        audit.verify(reg['protected'])
        require(reg['source_sha256'] == registration['source_sha256'], 'Same initial source')
        status = audit.read(path / 'status.json')
        require(status['status'] == 'COMPLETE' and status['step'] == 1000, arm + ' incomplete')
        for artifact in ('checkpoint', 'inference_export'):
            r = status[artifact]
            audit.verify({r['path']: r['sha256']})
            side = audit.read(resolve(r['path']).with_suffix('.json'))
            require(side['local_step'] == 1000 and side['global_step'] == 26500, 'Endpoint step')
            require(side['sha256'] == r['sha256'], 'Endpoint sidecar differs')
        events = audit.jsonl(path / 'calls.jsonl')
        starts = [r for r in events if r['event'] == 'CALL_BEGIN']
        done = [r for r in events if r['event'] == 'CALL_DONE']
        require(len(starts) == len(done) == 17000, 'Training call total')
        require({r['call_id'] for r in starts} == {r['call_id'] for r in done} and
                len({r['call_id'] for r in starts}) == 17000, 'Unpaired/duplicate training calls')
        require(not any('FAIL' in r['event'] for r in events), 'Failed training call')
        counts = Counter(r['kind'] for r in starts)
        require(counts == dict(train_forward=8000, train_backward=8000, optimizer_update=1000), 'Training budget')
        for kind in counts:
            actual = [(r['context']['step'], r['context'].get('microbatch')) for r in starts if r['kind'] == kind]
            expected = [(s, m) for s in range(1, 1001) for m in (range(8) if kind != 'optimizer_update' else (None,))]
            require(actual == expected, 'Training call cursor ' + arm + '/' + kind)
        exposures = audit.jsonl(path / 'exposure.jsonl')
        require(len(exposures) == 8000, 'Exposure budget')
        for index, row in enumerate(exposures):
            si, mi = divmod(index, 8)
            src = schedule[si]['samples'][mi]
            require(row['step'] == si + 1 and row['microbatch'] == mi, 'Exposure order')
            require(all(row[k] == src[k] for k in keys), 'Exposure stream differs from registration')
            require(abs(row['gain'] - coefficients[si]['micro_gains'][mi]) < 1e-12, 'FM gain changed')
        streams[arm] = [{k: r[k] for k in (*keys, 'step', 'microbatch', 'N', 'K', 't', 'gain')} for r in exposures]
        log = audit.jsonl(path / 'training_log.jsonl')
        require([r['step'] for r in log] == list(range(1, 1001)), 'Training curve coverage')
        require(all(numeric(r['objective']) and numeric(r['gradient_norm']) for r in log), 'Finite training curve')
        require(not list(path.glob('attempt_*/failure.json')), 'Training failure evidence present')
        result[arm] = dict(step=1000, global_step=26500, calls=dict(counts), exposures=8000,
                           checkpoint=status['checkpoint'], inference_export=status['inference_export'])
    require(streams['PLAIN'] == streams['LAYOUT'], 'Actual paired training streams differ')
    return dict(status='PASS', paired_stream_equal=True, arms=result,
                checkpoint_contents='Whole-file hashes and producer validation; no torch checkpoint reload')


def verify_generation(audit, requests):
    results = {}
    raw_books = {}
    last_raw_calls, first_d0_calls = [], []
    for label in NEW:
        folder = OUT / 'evaluation' / label
        raw_done = audit.read(folder / 'raw_complete_full.json')
        post_done = audit.read(folder / 'post_complete_full.json')
        require(raw_done['status'] == post_done['status'] == 'COMPLETE', 'Generation completion')
        require(raw_done['outputs'] == post_done['outputs'] == len(requests), 'Generation output budget')
        require(raw_done['total_new_generator_forwards'] == 600, 'Generator budget')
        require(post_done['new_D0_forwards'] == post_done['new_GEO_calls'] == 12, 'D0/GEO budget')
        audit.verify(raw_done['files']); audit.verify(post_done['files'])
        require(not list(folder.glob('*attempt_*/failure.json')), 'Generation failure evidence present')
        for kind in ('RAW', 'D0'):
            inf = audit.read(OUT / 'evaluation/inference_registrations' / (label + '__' + kind + '.json'))
            audit.verify(inf['code'])
            audit.verify(inf.get('protected', {}))
        for q in requests:
            rp = output_dir(label, q['request_id'], 'raw')
            pp = output_dir(label, q['request_id'], 'post')
            rr, pr = audit.read(rp / 'receipt.json'), audit.read(pp / 'receipt.json')
            require(rr['status'] == pr['status'] == 'COMPLETE', 'Per-request generation completion')
            require(rr['generator_forwards'] == 50 and rr['D0_forwards'] == 0, 'RAW accounting')
            require(pr['D0_forwards'] == 1 and pr['GEO_calls'] == 1, 'POST accounting')
            require(rr['noise_sha256'] == q['noise_sha256'], 'Saved fixed noise changed')
            audit.verify(rr['files']); audit.verify(pr['files'])
            raw_books[str(rp / 'receipt.json')] = audit.hash(rp / 'receipt.json')
            calls = audit.jsonl(rp / 'calls.jsonl')
            require([r['step'] for r in calls if r['event'] == 'FORWARD_STARTED'] == list(range(50)), 'Euler started budget')
            require([r['step'] for r in calls if r['event'] == 'FORWARD_COMPLETE'] == list(range(50)), 'Euler completed budget')
            last_raw_calls.append(max(r['unix'] for r in calls if r['event'] == 'FORWARD_COMPLETE'))
            dcalls = audit.jsonl(pp / 'calls.jsonl')
            require(sum(r['event'] == 'D0_FORWARD_STARTED' for r in dcalls) == 1 and
                    sum(r['event'] == 'D0_FORWARD_RETURNED' for r in dcalls) == 1, 'Single D0 calls')
            first_d0_calls.extend(datetime.fromisoformat(r['utc'].replace('Z', '+00:00')).timestamp()
                                  for r in dcalls if r['event'] == 'D0_FORWARD_STARTED')
        results[label] = dict(RAW=12, D0=12, generator_forwards=600)
    for label in NEW:
        registered = audit.read(OUT / 'evaluation/inference_registrations' / (label + '__D0.json'))['RAW_barrier']
        normalized = {str(resolve(k)): v for k, v in registered.items()}
        require(normalized == raw_books, 'D0 did not seal every new RAW before its dispatch')
    require(max(last_raw_calls) <= min(first_d0_calls) + 1, 'D0 started before every RAW completed')
    return dict(status='PASS', modes=results, total_RAW=36, total_D0=36, generator_forwards=1800,
                all_RAW_barrier=True, latest_RAW_forward_unix=max(last_raw_calls), first_D0_forward_unix=min(first_d0_calls),
                time_tolerance_seconds=1, time_tolerance_reason='Legacy UTC D0 stamps may have second precision; all36 hash barrier also verified',
                scope='LAYOUT1000 privileged source P; LAYOUT_NULL same checkpoint with masked P')


def verify_scores(audit, requests):
    panel = audit.read(OUT / 'data/eval_manifest.json')
    fields = panel['metric_fields']
    require(len(fields) == len(set(fields)) == 177, '177 original fields')
    reg = audit.read(OUT / 'evaluation/score_registration.json'); audit.verify(reg['source_hashes'])
    complete = audit.read(OUT / 'results/completion.json')
    audit.verify(complete['files']); audit.verify(complete['source_rows'])
    rows = audit.read(OUT / 'results/metric_rows.json')
    expected = {(label, q['request_id'], stage) for label in LABELS for q in requests for stage in STAGES}
    require(len(rows) == 144 and {metric_identity(r) for r in rows} == expected, 'Full 144 unique score cells')
    for row in rows:
        require(all(k in row for k in fields), 'Missing numerical metric field')
        folder = score_dir(*metric_identity(row))
        done = audit.read(folder / 'complete.json'); audit.verify(done['files'])
        require(done['status'] == 'PASS' and done['model_calls'] == 0, 'Score producer status')
        src = audit.read(folder / 'row.json')
        require(all(row[k] == src[k] for k in fields), 'Aggregated original metric changed')
    cached = audit.read(OUT / 'audits/reference_cache_audit.json')
    audit.verify(cached['files']); audit.verify(cached['protected'])
    return dict(status='PASS', rows=144, metric_fields=177, new_rows=108, reused_rows=36,
                no_rescoring=True, NA_values_preserved=True)


def verify_gallery(audit, requests):
    done = audit.read(OUT / 'gallery/complete.json')
    require(done['status'] == 'COMPLETE' and done['pages'] == 36 and done['ratings_generated'] == 0, 'Gallery contract')
    audit.verify(done['files']); audit.verify(done['protected'])
    for q in requests:
        for stage in STAGES:
            r = audit.read(OUT / 'gallery' / (stage + '__' + q['request_id']) / 'comparison.json')
            require([c['label'] for c in r['identity']['cells']] == list(LABELS), 'Gallery four modes')
            require(r['identity']['bounds'] == q['gallery_bounds'] and
                    r['identity']['depth_bounds'] == q['gallery_depth_bounds'], 'Original views and sections')
            audit.verify({r['image_path']: r['image_sha256']})
    return dict(status='PASS', pages=36, mode_stage_cells=144, ratings_generated=0)


def verify_review(audit, requests, paths):
    require(bool(paths), 'Missing actual review files (no labels synthesized)')
    rows = []
    for path in paths:
        payload = audit.read(path)
        current = payload if isinstance(payload, list) else payload.get('rows', payload.get('records', []))
        for row in current:
            # Reviewer metadata may be stored once at the actual review-file level.
            rows.append({**({'reviewer_kind': payload.get('reviewer_kind')} if isinstance(payload, dict) else {}), **row})
    expected = {(label, q['request_id'], stage) for label in LABELS for q in requests for stage in STAGES}
    require(len(rows) == 144 and {metric_identity(r) for r in rows} == expected, 'Actual review must cover 144 unique cells')
    for r in rows:
        require(r.get('actually_viewed') is True and r.get('views_and_sections_checked') is True, 'Unviewed review cell')
        require(r.get('reviewer_kind') == 'AI_PROXY', 'Reviewer kind must disclose AI_PROXY')
        for axis, allowed in AXES.items():
            require(r.get(axis) in allowed, 'Unfilled or invalid visual axis ' + axis)
        require(bool(r.get('evidence') or r.get('visible_evidence') or r.get('notes') or r.get('note')), 'Missing visual evidence text')
        require(bool(r.get('view_event') or r.get('view_events') or r.get('actual_view_event') or r.get('view_event_id')), 'Missing actual view event')
        require(bool(r.get('viewed_at_utc')), 'Missing actual-view UTC')
        page = OUT / 'gallery' / (r['stage'] + '__' + r['request_id'])
        actual = r.get('image_sha256') or r.get('source_image_sha256')
        require(actual == audit.hash(page / 'comparison.png'), 'Review refers to a different image')
        require(r.get('image_receipt_sha256') == audit.hash(page / 'comparison.json'), 'Missing/different image producer receipt')
    return dict(status='PASS', rows=144, reviewer_kind='AI_PROXY', independent_human_certification=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', action='append', default=[])
    parser.add_argument('--output', default=str(OUT / 'results/final_verification.json'))
    args = parser.parse_args()
    audit = FileAudit(); results = {}
    requests = read(OUT / 'data/eval_manifest.json')['requests']
    reviews = [resolve(p) for p in args.review] or sorted((OUT / 'visual_review/completed').glob('*.json'))
    for name, fn in (
        ('training', lambda: verify_training(audit)),
        ('generation', lambda: verify_generation(audit, requests)),
        ('scores', lambda: verify_scores(audit, requests)),
        ('gallery', lambda: verify_gallery(audit, requests)),
        ('actual_review', lambda: verify_review(audit, requests, reviews)),
    ):
        try:
            results[name] = fn()
        except FileNotFoundError as exc:
            results[name] = dict(status='BLOCKED_MISSING', reason=str(exc))
        except Exception as exc:
            results[name] = dict(status='FAIL', reason=repr(exc))
    if 'torch' in sys.modules:
        raise RuntimeError('Final verifier imported torch')
    status = 'PASS' if all(r['status'] == 'PASS' for r in results.values()) else 'INCOMPLETE_OR_FAILED'
    result = dict(schema='bench_layout_oracle_final_verification_v1', status=status, checks=results,
                  model_calls=0, scoring_calls=0, updates=0, images_rendered=0, ratings_generated=0,
                  files=audit.files, source_sha256=sha(__file__))
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=False))
    return 0 if status == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
