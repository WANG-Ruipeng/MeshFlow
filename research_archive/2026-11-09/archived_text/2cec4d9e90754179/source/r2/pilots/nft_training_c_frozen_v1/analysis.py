"""Complete-only analysis of 8 trained C x 4 fresh noises x 3 frozen models.

Reuses the original numerical extractor and raw terminal reward, never its
within-C training normalization. No model or geometry scoring is performed.
Unknown R stays None; separate identification intervals are not imputations.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_nft_training_c_frozen_v1'
EV = OUT / 'evaluation'
DEST = EV / 'analysis'
VERSION = 'nft_training_C_frozen_analysis_v1'
LABELS = ('BASE', 'NFT', 'SELF_FM')
CONTRASTS = (('NFT', 'BASE'), ('SELF_FM', 'BASE'), ('NFT', 'SELF_FM'))
REVIEW_FIELDS = ('RAW_adoption', 'POST_adoption', 'POST_body_complete')
REVIEW_VALUES = {'YES', 'NO', 'UNKNOWN', 'UNCERTAIN', 'NA'}
EVENTS = ('RAW_adoption_known', 'POST_adoption_known', 'POST_body_complete_known',
          'joint_RAW_adoption_POST_legal_CF', 'joint_both_adoptions_body_and_CF')
GEOMETRY = ('CF_illegal_fraction', 'FF_illegal_fraction', 'I_lower', 'I_upper',
            'free_retained_fraction', 'legal_CF_edges', 'legal_CF_length',
            'free_near_degenerate_fraction', 'free_small_angle_fraction',
            'reached_free_area_fraction', 'free_surface_area')


def _modules():
    if 'torch' in sys.modules:
        raise RuntimeError('Analysis requires a separate torch-free CPU process')
    from .bootstrap import bind
    bind()
    from r2.pilots.terminal_reward_nft_v1 import data as D, evaluation as E, final_analysis as F
    if D.OUT.resolve() != OUT.resolve() or E.OUT.resolve() != OUT.resolve():
        raise RuntimeError('Bootstrap did not bind the independent experiment output')
    return D, E, F


def _stats(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return dict(count=len(values), finite=len(finite), unavailable=len(values) - len(finite),
                mean=statistics.mean(finite) if finite else None,
                median=statistics.median(finite) if finite else None,
                minimum=min(finite) if finite else None, maximum=max(finite) if finite else None)


def _truth(value):
    return True if value == 'YES' else False if value == 'NO' else None


def _and(values):
    return False if any(v is False for v in values) else None if any(v is None for v in values) else True


def _event(rows, key):
    yes = sum(r[key] is True for r in rows)
    no = sum(r[key] is False for r in rows)
    unknown = len(rows) - yes - no
    return dict(total=len(rows), YES=yes, NO=no, UNKNOWN=unknown,
                positive_fraction_lower=yes / len(rows), positive_fraction_upper=(yes + unknown) / len(rows))


def _interval(value):
    return (0.0, 1.0) if value is None else (float(value), float(value))


def _reward_interval(reward, review):
    """Bounds on the frozen formula; never substitutes a missing observed R."""
    if reward['status'] == 'SCORABLE' and reward.get('reward') is not None:
        value = float(reward['reward'])
        return value, value, 'OBSERVED_RAW_R'
    if any(review[k] == 'NO' for k in REVIEW_FIELDS):
        return 0.0, 0.0, 'KNOWN_NO_FORCES_GATE_ZERO_OBSERVED_R_STILL_MISSING'
    if reward.get('task') is not None and reward.get('quality') is not None:
        cap = min(1.5, max(0.0, float(reward['task']) * float(reward['quality'])))
        return 0.0, cap, 'FIXED_GEOMETRY_TASK_TIMES_QUALITY_UPPER_BOUND'
    return 0.0, 1.5, 'GLOBAL_FROZEN_FORMULA_RANGE_GEOMETRY_UNSCORABLE'


def _csv(D, path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in D.clean(row).items()})
    content = stream.getvalue().encode('utf-8-sig')
    path = D.checked(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Refuse changed completed analysis CSV: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(content)


def readiness(reviews_path=None):
    D, _, _ = _modules()
    reviews_path = D.resolve(reviews_path or EV / 'reviews_actual.json')
    manifest_path = OUT / 'inputs/manifest.json'
    required = [manifest_path, OUT / 'registration.json', reviews_path]
    if manifest_path.exists():
        m = D.read(manifest_path)
        required += [EV / label / 'scores' / q['request_id'] / stage / 'score.json'
                     for label in LABELS for q in m['evaluation_requests'] for stage in ('RAW', 'POST')]
    missing = [str(p) for p in required if not p.is_file()]
    return dict(status='READY_FOR_VALIDATION' if not missing else 'INCOMPLETE', missing=missing,
                missing_count=len(missing), reviews_path=str(reviews_path), complete_summary_written=False)


def _mean_by_condition(rows, key):
    cids = sorted({r['condition_id'] for r in rows})
    return _stats([_stats([r[key] for r in rows if r['condition_id'] == cid])['mean'] for cid in cids])


def _summarize_group(rows):
    result = dict(outputs=len(rows), conditions=len({r['condition_id'] for r in rows}),
                  parents=len({r['parent_id'] for r in rows}),
                  events={k: _event(rows, k) for k in EVENTS},
                  POST_interface_outputs=sum(r['POST_has_legal_CF'] for r in rows),
                  reward_status_counts=dict(Counter(r['reward_status'] for r in rows)),
                  reward_observed=_stats([r['reward'] for r in rows]),
                  reward_observed_equal_condition=_mean_by_condition(rows, 'reward'),
                  reward_all_registered_mean_lower=_mean_by_condition(rows, 'reward_lower')['mean'],
                  reward_all_registered_mean_upper=_mean_by_condition(rows, 'reward_upper')['mean'], metrics={})
    for stage in ('RAW', 'POST'):
        result[stage + '_I_status_counts'] = dict(Counter(r[stage + '_I_status'] for r in rows))
        result[stage + '_topology_status_counts'] = dict(Counter(r[stage + '_topology_status'] for r in rows))
        for metric in GEOMETRY:
            key = stage + '_' + metric
            result['metrics'][key] = dict(output_distribution=_stats([r[key] for r in rows]),
                                          equal_condition_distribution=_mean_by_condition(rows, key))
    return result


def _paired(outputs):
    lookup = {(r['label'], r['request_id']): r for r in outputs}
    requests = sorted({r['request_id'] for r in outputs})
    pairs = []
    for candidate, reference in CONTRASTS:
        for rid in requests:
            a, b = lookup[candidate, rid], lookup[reference, rid]
            known = a['reward'] is not None and b['reward'] is not None
            row = dict(candidate_model=candidate, reference_model=reference, request_id=rid,
                       condition_id=a['condition_id'], parent_id=a['parent_id'], candidate=a['candidate'],
                       split=a['split'], noise_partition=a['evaluation_noise_partition'],
                       reward_candidate=a['reward'], reward_reference=b['reward'], reward_both_known=known,
                       reward_delta=a['reward'] - b['reward'] if known else None,
                       reward_delta_lower=a['reward_lower'] - b['reward_upper'],
                       reward_delta_upper=a['reward_upper'] - b['reward_lower'])
            for event in EVENTS:
                x, y = a[event], b[event]
                xl, xu = _interval(x); yl, yu = _interval(y)
                row.update({event + '_candidate': x, event + '_reference': y,
                            event + '_delta': int(x) - int(y) if x is not None and y is not None else None,
                            event + '_delta_lower': xl - yu, event + '_delta_upper': xu - yl})
            row['POST_has_legal_CF_delta'] = a['POST_has_legal_CF'] - b['POST_has_legal_CF']
            for stage in ('RAW', 'POST'):
                for metric in GEOMETRY:
                    key = stage + '_' + metric
                    row[key + '_delta'] = a[key] - b[key] if a[key] is not None and b[key] is not None else None
            pairs.append(row)
    return pairs


def _paired_group(rows):
    common = [r for r in rows if r['reward_both_known']]
    delta_keys = [k for k in rows[0] if k.endswith('_delta')]
    result = dict(paired_outputs=len(rows), conditions=len({r['condition_id'] for r in rows}),
                  reward_common_known_pairs=len(common), reward_unknown_pairs=len(rows) - len(common),
                  reward_common_known_condition_count=len({r['condition_id'] for r in common}),
                  reward_common_known_candidate=_stats([r['reward_candidate'] for r in common]),
                  reward_common_known_reference=_stats([r['reward_reference'] for r in common]),
                  reward_common_known_candidate_equal_condition=_mean_by_condition(common, 'reward_candidate'),
                  reward_common_known_reference_equal_condition=_mean_by_condition(common, 'reward_reference'),
                  reward_common_known_delta=_stats([r['reward_delta'] for r in common]),
                  reward_common_known_delta_equal_condition=_mean_by_condition(rows, 'reward_delta'),
                  reward_all_registered_delta_lower=_mean_by_condition(rows, 'reward_delta_lower')['mean'],
                  reward_all_registered_delta_upper=_mean_by_condition(rows, 'reward_delta_upper')['mean'],
                  differences={}, events={})
    for key in delta_keys:
        means = [_stats([r[key] for r in rows if r['condition_id'] == cid])['mean']
                 for cid in sorted({r['condition_id'] for r in rows})]
        result['differences'][key] = dict(output_distribution=_stats([r[key] for r in rows]),
                                          equal_condition_distribution=_stats(means),
                                          conditions_positive=sum(v is not None and v > 0 for v in means),
                                          conditions_zero=sum(v == 0 for v in means),
                                          conditions_negative=sum(v is not None and v < 0 for v in means),
                                          conditions_unavailable=sum(v is None for v in means))
    for event in EVENTS:
        values = [r[event + '_delta'] for r in rows]
        result['events'][event] = dict(gained=sum(v == 1 for v in values), lost=sum(v == -1 for v in values),
                                      unchanged_known=sum(v == 0 for v in values), unknown_pairs=sum(v is None for v in values),
                                      all_registered_delta_lower=_mean_by_condition(rows, event + '_delta_lower')['mean'],
                                      all_registered_delta_upper=_mean_by_condition(rows, event + '_delta_upper')['mean'])
    return result


def analyze(reviews_path=None):
    D, E, F = _modules()
    ready = readiness(reviews_path)
    if ready['status'] != 'READY_FOR_VALIDATION':
        raise RuntimeError('Incomplete 96-output evaluation; no complete report written: ' + str(ready['missing_count']))
    protected = {}

    def protect(path, expected=None):
        path = D.resolve(path)
        value = D.sha(path)
        if expected is not None and value != expected:
            raise ValueError('Input/source hash mismatch: ' + str(path))
        if str(path) in protected and protected[str(path)] != value:
            raise ValueError('Input changed during analysis: ' + str(path))
        protected[str(path)] = value
        return value

    manifest_path = OUT / 'inputs/manifest.json'
    manifest_hash = protect(manifest_path)
    m = D.read(manifest_path)
    registration_path = OUT / 'registration.json'
    registration_hash = protect(registration_path)
    registration = D.read(registration_path)
    if registration.get('manifest_sha256') != manifest_hash or registration.get('status') != 'FROZEN':
        raise ValueError('Registration does not certify this frozen manifest')
    if m.get('status') != 'FROZEN':
        raise ValueError('Input manifest must be frozen')
    qs = m['evaluation_requests']
    qmap = {q['request_id']: q for q in qs}
    if len(qs) != 32 or len(qmap) != 32 or sorted(Counter(q['condition_id'] for q in qs).values()) != [4] * 8:
        raise ValueError('Expected exactly eight C with four distinct registered requests each')
    if len({q['parent_id'] for q in qs}) != 8 or any(q['split'] != 'TRAIN' for q in qs):
        raise ValueError('Expected eight original TRAIN parents')
    source_manifest = ROOT / 'experiments/r2_terminal_reward_nft_v1/inputs/manifest.json'
    protect(source_manifest)
    source = D.read(source_manifest)
    original = {c['condition_id']: c for c in source['train_conditions']}
    if set(original) != {q['condition_id'] for q in qs}:
        raise ValueError('Evaluation conditions differ from original eight reward conditions')
    old_seeds = {q['noise_seed'] for q in source['training_requests'] + source['evaluation_requests']}
    old_noise = {q['Gaussian_sha256'] for q in source['training_requests'] + source['evaluation_requests']}
    if len({q['noise_seed'] for q in qs}) != 32 or any(q['noise_seed'] in old_seeds or q['Gaussian_sha256'] in old_noise for q in qs):
        raise ValueError('New evaluation noise collides with registered pilot noise')
    for q in qs:
        c = original[q['condition_id']]
        if any(q[k] != c[k] for k in ('parent_id', 'N', 'K', 'condition_source_sha256')):
            raise ValueError('Original reward C identity differs')
        protect(q['input_path'], q['input_sha256'])
        D.load_request(q)
    review_path = D.resolve(ready['reviews_path'])
    protect(review_path)
    payload = D.read(review_path)
    review_rows = payload['rows'] if isinstance(payload, dict) else payload
    expected = {(label, rid) for label in LABELS for rid in qmap}
    reviews = {}
    for r in review_rows:
        key = r.get('label'), r.get('request_id')
        if key in reviews or key not in expected:
            raise ValueError('Duplicate or unexpected actual review key')
        if r.get('review_status') != 'REVIEWED' or r.get('reviewer_kind') not in ('AI_PROXY', 'HUMAN') or not str(r.get('reviewer', '')).strip():
            raise ValueError('Actual review identity/status missing')
        if r.get('blinded') is not True and str(r.get('blinded')).lower() != 'true':
            raise ValueError('Model-blinded review required')
        if any(r.get(k) not in REVIEW_VALUES for k in REVIEW_FIELDS):
            raise ValueError('Unknown review vocabulary')
        image_hash = r.get('image_sha256') or r.get('reviewed_image_sha256') or r.get('evidence_sha256')
        if not isinstance(image_hash, str) or not re.fullmatch('[0-9a-fA-F]{64}', image_hash):
            raise ValueError('Actual image hash missing')
        if r.get('image_path'):
            protect(r['image_path'], image_hash)
        reviews[key] = r
    if len(review_rows) != 96 or set(reviews) != expected:
        raise ValueError('All 96 actual reviews required; explicit UNKNOWN is permitted')
    protect(__file__)
    protect(Path(__file__).with_name('bootstrap.py'))
    protect(D.__file__)
    protect(E.__file__)
    protect(F.__file__)
    definitions = None
    outputs, stages, reward_rows = [], [], []
    for label in LABELS:
        for q in qs:
            rid = q['request_id']
            review = reviews[label, rid]
            row = {k: q[k] for k in ('request_id', 'parent_id', 'condition_id', 'candidate', 'split', 'N', 'K')}
            row.update(label=label, evaluation_noise_partition=q.get('evaluation_noise_partition', 'FRESH'),
                       noise_seed=q['noise_seed'], input_sha256=q['input_sha256'], reviewer_kind=review['reviewer_kind'],
                       reviewer=review['reviewer'], blinded=True,
                       image_sha256=review.get('image_sha256') or review.get('reviewed_image_sha256') or review.get('evidence_sha256'))
            row.update({k: review[k] for k in REVIEW_FIELDS})
            row.update({k + '_known': _truth(review[k]) for k in REVIEW_FIELDS})
            post_score = None
            for stage in ('RAW', 'POST'):
                path = EV / label / 'scores' / rid / stage / 'score.json'
                score_hash = protect(path)
                score = D.read(path)
                if score['identity'].get('registration_sha256') != registration_hash:
                    raise ValueError('Score registration is not this experiment registration')
                if score.get('original_definition_unchanged') is not True or score.get('model_calls') != 0:
                    raise ValueError('Original model-free numerical scoring contract changed')
                if definitions is None:
                    definitions = score['identity']['definition_hashes']
                    for p, h in definitions.items():
                        protect(p, h)
                elif definitions != score['identity']['definition_hashes']:
                    raise ValueError('Scoring definitions differ between outputs')
                for p, h in score['identity']['input_hashes'].items():
                    protect(p, h)
                values = F.extract_score(score, q, label, stage)
                if score['free_face_denominator'] != values['actual_free_faces']:
                    raise ValueError('Actual stage free denominator differs')
                if values['actual_free_faces'] + values['deleted_free_faces'] != q['N'] - q['K']:
                    raise ValueError('Free-face retention accounting differs')
                row.update({stage + '_' + k: v for k, v in values.items()})
                stages.append(dict(label=label, stage=stage, request_id=rid, condition_id=q['condition_id'],
                                   parent_id=q['parent_id'], candidate=q['candidate'], split=q['split'],
                                   score_path=str(path), score_sha256=score_hash, **values,
                                   original_metrics=score['metrics'],
                                   original_topology_scalars={k: v for k, v in score['topology'].items() if not isinstance(v, (list, dict))}))
                if stage == 'POST':
                    post_score = score
            try:
                reward = E.geometry_review_reward(post_score, review)
            except ValueError as exc:
                reward = dict(version='terminal_NFT_locked_reward_v1', status='GEOMETRY_UNSCORABLE',
                              reward=None, reasons=[str(exc)], review_labels={k: review[k] for k in REVIEW_FIELDS})
            low, high, bound_status = _reward_interval(reward, review)
            row.update(reward=reward['reward'], reward_status=reward['status'], reward_reasons=reward['reasons'],
                       reward_lower=low, reward_upper=high, reward_bound_status=bound_status,
                       reward_gate=reward.get('review_gate'))
            for block in ('components', 'quality_factors', 'task_terms'):
                row.update({'reward_' + block + '__' + k: v for k, v in reward.get(block, {}).items()})
            for key in ('quality', 'task', 'C_boundary_perimeter', 'free_denominator', 'original_free_denominator'):
                row['reward_' + key] = reward.get(key)
            row['joint_RAW_adoption_POST_legal_CF'] = _and([row['RAW_adoption_known'], bool(row['POST_has_legal_CF'])])
            row['joint_both_adoptions_body_and_CF'] = _and([row['RAW_adoption_known'], row['POST_adoption_known'],
                                                         row['POST_body_complete_known'], bool(row['POST_has_legal_CF'])])
            outputs.append(row)
            reward_rows.append(dict(label=label, request_id=rid, condition_id=q['condition_id'], **reward,
                                    reward_lower=low, reward_upper=high, bound_status=bound_status))
    if len(outputs) != 96 or len(stages) != 192:
        raise ValueError('Complete output/stage coverage differs')
    groups, conditions = [], []
    for label in LABELS:
        selected = [r for r in outputs if r['label'] == label]
        groups.append(dict(label=label, **_summarize_group(selected)))
        for cid in sorted(original):
            local = [r for r in selected if r['condition_id'] == cid]
            conditions.append(dict(label=label, condition_id=cid, parent_id=local[0]['parent_id'], **_summarize_group(local)))
    pairs = _paired(outputs)
    paired_summary, paired_conditions = [], []
    for candidate, reference in CONTRASTS:
        selected = [r for r in pairs if r['candidate_model'] == candidate and r['reference_model'] == reference]
        paired_summary.append(dict(candidate_model=candidate, reference_model=reference, **_paired_group(selected)))
        for cid in sorted(original):
            local = [r for r in selected if r['condition_id'] == cid]
            paired_conditions.append(dict(candidate_model=candidate, reference_model=reference, condition_id=cid,
                                          parent_id=local[0]['parent_id'], **_paired_group(local)))
    # Every ranked request is retained; no tail cutoff or new accept/reject threshold.
    tails = []
    for candidate, reference in CONTRASTS:
        selected = [r for r in pairs if r['candidate_model'] == candidate and r['reference_model'] == reference]
        for stage in ('RAW', 'POST'):
            for metric in ('CF_illegal_fraction', 'FF_illegal_fraction', 'I_lower', 'I_upper'):
                key = stage + '_' + metric + '_delta'
                ordered = sorted(selected, key=lambda r: (r[key] is None, -r[key] if r[key] is not None else 0, r['request_id']))
                for rank, r in enumerate(ordered, 1):
                    tails.append(dict(candidate_model=candidate, reference_model=reference, stage=stage, metric=metric,
                                      descending_increase_rank=rank, request_id=r['request_id'], condition_id=r['condition_id'],
                                      parent_id=r['parent_id'], delta=r[key]))
    if 'torch' in sys.modules:
        raise RuntimeError('Analysis imported torch')
    for p, h in protected.items():
        if D.sha(p) != h:
            raise ValueError('Protected input changed during analysis: ' + p)
    tables = {'per_output': outputs, 'per_output_stage': stages, 'per_condition': conditions,
              'paired_per_request': pairs, 'paired_per_condition': paired_conditions, 'geometry_tails': tails}
    for name, rows in tables.items():
        _csv(D, DEST / (name + '.csv'), rows)
        D.write(DEST / (name + '.json'), rows)
    D.write(DEST / 'reward_details.json', reward_rows)
    D.write(DEST / 'paired_summary.json', paired_summary)
    table_hashes = {str(p): D.sha(p) for p in DEST.iterdir() if p.is_file() and p.name != 'summary.json'}
    result = dict(version=VERSION, status='COMPLETE', outputs=96, distinct_requests=32, parent_conditions=8,
                  RAW_scores=96, POST_scores=96, actual_reviews=96, models=list(LABELS), groups=groups,
                  definitions=definitions, protected_hashes=protected, tables=table_hashes,
                  analysis_source_sha256=D.sha(__file__), model_calls=0, metrics_recomputed=False,
                  new_geometry_samples=0, torch_imported=False, reward_group_normalization_performed=False,
                  aggregation='Four registered fresh noises per C; first within C, then equal eight C/parents. Known-only means explicitly retain missing-condition counts.',
                  reward_definition='Unchanged terminal_NFT_locked_reward_v1 raw R, range [0,1.5]; no r_NFT standardization.',
                  reward_interval_definition='Observed R stays None for abstention/unscorable. Bounds are separate: known R [R,R]; any explicit NO [0,0]; otherwise [0,task*quality] when geometry available, [0,1.5] if unscorable. I lower in quality gives an optimistic upper when I is partial.',
                  primary_joint_definition='RAW adoption YES and POST at least one valid CF edge',
                  strict_joint_definition='RAW/POST adoption YES and POST body complete YES and POST valid CF edge',
                  limitations=['All eight C and parents were used in reward training; this measures fresh-noise performance on trained conditions, not held-out-condition generalization.',
                               'UNKNOWN/UNCERTAIN/NA reviews are never observed reward zero; common-known paired estimates and all-request bounds are separate.',
                               'Known-only means and common-known subsets can have model-dependent missingness; inspect all per-C coverage and intervals.',
                               'The reward uses its frozen geometric coefficients and an actual visual gate; it is not a calibrated probability or a certificate of complete attachment.',
                               'CF/FF use actual stage free-face denominators, I uses fixed C queries; both stages and retention remain separate.',
                               'No best-of-four selection, reward recalibration, model selection, new threshold, training, or scientific conclusion is performed.'])
    D.write(DEST / 'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--reviews')
    args = p.parse_args()
    result = readiness(args.reviews) if args.check else analyze(args.reviews)
    print(json.dumps({k: result[k] for k in ('status', 'outputs', 'distinct_requests', 'missing_count', 'complete_summary_written') if k in result}))


if __name__ == '__main__':
    main()
