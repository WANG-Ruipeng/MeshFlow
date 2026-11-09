"""Validate and summarize recorded actual visual reviews, without rerating.

Only the two review JSONs, frozen panel/jobs, image receipts and PNG bytes are
read. No score/metric files, meshes, detector, renderer or model are consulted.
Repeated baseline columns retain one observation per request/model/stage.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'experiments/r2_bench_regional_mix_dual_v1'
DEST = OUT / 'visual_review'
VARIANTS = ('ASSIST10', 'ASSIST01', 'ASSIST11', 'AUTO_COVER')
RUNTIME_MODELS = ('RAW', 'M_D0', 'RAW_ALL') + VARIANTS
LABELS = ('RAW', 'D0_ALL', 'RAW_ALL') + VARIANTS
AXES = {
    'overall_shape': ('REASONABLE', 'INCOMPLETE', 'UNCERTAIN'),
    'C_use': ('PLAUSIBLE', 'REPLACED_OR_IGNORED', 'UNCERTAIN'),
    'support_retention': ('PRESENT', 'MISSING_OR_DAMAGED', 'UNCERTAIN'),
    'visible_fragment_or_gap': ('YES', 'NO_OBVIOUS', 'UNCERTAIN'),
    'layering_or_fold': ('YES', 'NO_OBVIOUS', 'UNCERTAIN'),
}
JOINT_AXES = ('overall_shape', 'C_use', 'support_retention')
ASSIGNMENTS = (('root_review.json', 'AI_PROXY_root', range(3)),
               ('backend_data_review.json', 'AI_PROXY_backend_data', range(3, 6)))
VERSION = 'regional_mix_actual_visual_collect_v1'


def resolve(value):
    text = str(value).replace('\\', '/')
    for prefix in ('E:/MeshFlow/', '/mnt/e/MeshFlow/'):
        if text.startswith(prefix):
            return ROOT / text[len(prefix):]
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def csvwrite(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple))
                             else value for key, value in row.items()})


def utc_value(value):
    if value.endswith(' UTC'):
        result = datetime.strptime(value, '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    else:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.utcoffset() != timezone.utc.utcoffset(result):
        raise ValueError('Actual view time must be explicitly UTC')
    return result.isoformat()


def counts(rows):
    return dict(
        unique_outputs=len(rows), three_axis_joint=sum(r['three_axis_joint'] for r in rows),
        three_axis_uncertain_inputs=sum(any(r[a] == 'UNCERTAIN' for a in JOINT_AXES) for r in rows),
        support_PRESENT=sum(r['support_retention'] == 'PRESENT' for r in rows),
        support_MISSING_OR_DAMAGED=sum(r['support_retention'] == 'MISSING_OR_DAMAGED' for r in rows),
        support_UNCERTAIN=sum(r['support_retention'] == 'UNCERTAIN' for r in rows),
        axes={axis: {state: sum(r[axis] == state for r in rows) for state in states}
              for axis, states in AXES.items()})


def transition(reference, candidate):
    before, after = reference['three_axis_joint'], candidate['three_axis_joint']
    fate = 'RETAINED' if before and after else 'LOST' if before else 'GAINED' if after else 'NEITHER'
    uncertain = any(r[a] == 'UNCERTAIN' for r in (reference, candidate) for a in JOINT_AXES)
    row = dict(request_id=candidate['request_id'], parent_id=candidate['parent_id'],
               noise_index=candidate['noise_index'], report_group=candidate['report_group'],
               reference=reference['label'], label=candidate['label'],
               reference_runtime_model=reference['runtime_model'], runtime_model=candidate['runtime_model'],
               reference_stage=reference['stage'], stage=candidate['stage'],
               reference_three_axis_joint=before, candidate_three_axis_joint=after,
               three_axis_boolean_fate=fate,
               three_axis_fate='UNCERTAIN_COMPARISON' if uncertain else fate,
               three_axis_transition_has_uncertainty=uncertain,
               reference_notes=reference['notes'], candidate_notes=candidate['notes'],
               reference_image_evidence=reference['image_evidence'], candidate_image_evidence=candidate['image_evidence'],
               reference_reviewer=reference['reviewer'], candidate_reviewer=candidate['reviewer'])
    for axis in AXES:
        row['reference_' + axis] = reference[axis]
        row['candidate_' + axis] = candidate[axis]
        row['transition_' + axis] = reference[axis] + ' -> ' + candidate[axis]
    return row


def run():
    done = DEST / 'COMPLETE.json'
    if done.exists():
        receipt = read(done)
        for path, digest in {**receipt['inputs'], **receipt['files']}.items():
            if sha(path) != digest:
                raise ValueError('Completed visual collection input/output changed: ' + path)
        return receipt
    output_names = ('per_output.csv', 'per_request_transitions.csv', 'summary.json')
    if any((DEST / name).exists() for name in output_names):
        raise ValueError('Partial visual collection retained; no overwrite')
    if any(not (DEST / name).is_file() for name, _, _ in ASSIGNMENTS):
        raise FileNotFoundError('Wait for both actual review JSONs; no polling or partial summary')
    panel_path = OUT / 'panel/manifest.json'
    jobs_path = OUT / 'jobs.json'
    protocol_path = DEST / 'PROTOCOL.md'
    panel = read(panel_path)['requests']
    reg = read(jobs_path)
    registration_sha = sha(jobs_path)
    protocol_sha = sha(protocol_path)
    if len(panel) != 6 or len({p['request_id'] for p in panel}) != 6:
        raise ValueError('Expected the frozen six-request panel')
    if reg['panel_sha256'] != sha(panel_path):
        raise ValueError('Panel changed after job registration')
    jobs = {j['job_id']: j for j in reg['jobs']}
    inputs = {str(p): sha(p) for p in (panel_path, jobs_path, protocol_path, Path(__file__))}
    review_hashes = {}
    pages = {}
    rows = []
    output_identity = {}
    reviewers = []
    page_evidence = {}
    for filename, owner, indices in ASSIGNMENTS:
        path = DEST / filename
        review = read(path)
        digest = sha(path)
        inputs[str(path)] = digest
        review_hashes[str(path)] = digest
        if (review['status'] != 'ACTUAL_VIEW_COMPLETE' or review['reviewer'] != owner
                or review['review_kind'] != 'AI_PROXY_NONBLIND'
                or review['protocol_sha256'] != protocol_sha
                or review['actual_viewed_page_count'] != 12
                or review.get('unique_output_count', review.get('unique_observation_count')) != 21):
            raise ValueError('Actual review status, identity or declared counts differ: ' + filename)
        if review.get('candidate_metrics_read_before_review', review.get('numerical_scores_read_before_rating')) is not False:
            raise ValueError('Missing explicit no-candidate-metrics-before-review declaration')
        if review.get('labels_generated_from_metrics', False) is not False:
            raise ValueError('Visual labels cannot be derived from metric values')
        if review.get('registration_sha256', registration_sha) != registration_sha:
            raise ValueError('Review registration differs')
        expected_pages = {f'J{i:02d}_{variant}' for i in indices for variant in VARIANTS}
        actual_pages = review['pages']
        if len(actual_pages) != 12 or {p['job_id'] for p in actual_pages} != expected_pages:
            raise ValueError('Actual assigned pages differ: ' + filename)
        for page in actual_pages:
            jid = page['job_id']
            if jid in pages:
                raise ValueError('Duplicate actual page across reviewers')
            q = jobs[jid]['request']
            image_path = resolve(page['image_path'])
            expected_image = OUT / 'runs' / jid / 'gallery/CLEAN_FINAL_POST.png'
            image_receipt_path = expected_image.with_suffix('.json')
            if image_path != expected_image:
                raise ValueError('Unexpected actual image path')
            image_hash = sha(image_path)
            receipt_hash = sha(image_receipt_path)
            if image_hash != page['image_sha256'] or receipt_hash != page.get('image_receipt_sha256', page.get('receipt_sha256')):
                raise ValueError('Reviewed PNG or receipt hash differs: ' + jid)
            image_receipt = read(image_receipt_path)
            if (image_receipt['job_id'] != jid or image_receipt['request_id'] != q['request_id']
                    or image_receipt['registration_sha256'] != registration_sha
                    or image_receipt['stage'] != 'CLEAN_FINAL_POST'
                    or image_receipt['image_sha256'] != image_hash):
                raise ValueError('Render producer identity differs: ' + jid)
            views = page.get('views', page.get('views_checked'))
            alternatives = ({'oblique_-60', 'oblique_120', 'fixed_front_section', 'fixed_side_section'},
                            {'azimuth_-60', 'azimuth_120', 'original_front_section', 'original_side_section'})
            if not views or set(views) not in alternatives or len(views) != 4:
                raise ValueError('Both views and both fixed sections must be recorded')
            if page.get('actually_viewed', True) is not True or not page.get('view_event') or not page.get('view_time_precision'):
                raise ValueError('Missing actual view event/assertion')
            viewed_at = utc_value(page['viewed_by_utc'])
            cells = image_receipt['cells']
            expected_models = ('RAW', 'M_D0', 'RAW_ALL', jobs[jid]['label'])
            if tuple(c['model'] for c in cells) != expected_models:
                raise ValueError('Four-column visual definition changed')
            for cell in cells:
                expected_stage = 'RAW' if cell['model'] == 'RAW' else 'CLEAN_FINAL_POST'
                if cell['stage'] != expected_stage:
                    raise ValueError('Incorrect visual stage')
                mesh = str(resolve(cell['path']))
                mesh_hashes = {str(resolve(p)): h for p, h in image_receipt['source_meshes'].items()}
                identity = dict(path=mesh, sha256=mesh_hashes[mesh])
                key = q['request_id'], cell['model'], cell['stage']
                if key in output_identity and output_identity[key] != identity:
                    raise ValueError('Repeated baseline columns are different outputs')
                output_identity[key] = identity
            inputs[str(image_path)] = image_hash
            inputs[str(image_receipt_path)] = receipt_hash
            pages[jid] = page
            page_evidence[jid] = dict(reviewer=owner, request_id=q['request_id'], image_path=str(image_path),
                                      image_sha256=image_hash, image_receipt_path=str(image_receipt_path),
                                      image_receipt_sha256=receipt_hash, view_event=page['view_event'],
                                      original_view_utc=page['viewed_by_utc'], normalized_view_utc=viewed_at,
                                      view_time_precision=page['view_time_precision'], recorded_views=views)
        observations = review['observations']
        expected_keys = {(panel[i]['request_id'], model) for i in indices for model in RUNTIME_MODELS}
        if len(observations) != 21 or {(o['request_id'], o['model']) for o in observations} != expected_keys:
            raise ValueError('Each reviewer must record exactly21 unique outputs')
        for observation_index, observation in enumerate(observations):
            i = observation['panel_index']
            if i not in indices or observation['request_id'] != panel[i]['request_id']:
                raise ValueError('Review panel index differs')
            model = observation['model']
            stage = 'RAW' if model == 'RAW' else 'CLEAN_FINAL_POST'
            if observation['stage'] != stage or not str(observation.get('notes', '')).strip():
                raise ValueError('Observation stage or concrete notes missing')
            if observation.get('reviewer', owner) != owner or observation.get('review_kind', 'AI_PROXY_NONBLIND') != 'AI_PROXY_NONBLIND':
                raise ValueError('Observation reviewer identity differs')
            if observation.get('actually_viewed', True) is not True:
                raise ValueError('Unviewed observation cannot be aggregated')
            for axis, allowed in AXES.items():
                if observation[axis] not in allowed:
                    raise ValueError('Unknown original visual label: ' + axis)
            joint = all(observation[axis] == AXES[axis][0] for axis in JOINT_AXES)
            if type(observation['three_axis_joint']) is not bool or observation['three_axis_joint'] != joint:
                raise ValueError('Original joint flag conflicts with recorded axes; no automatic relabeling')
            expected_evidence = ({f'J{i:02d}_{v}' for v in VARIANTS} if model in RUNTIME_MODELS[:3]
                                 else {f'J{i:02d}_{model}'})
            if set(observation['image_evidence']) != expected_evidence or len(observation['image_evidence']) != len(expected_evidence):
                raise ValueError('Observation image evidence differs')
            q = panel[i]['request']
            identity = output_identity[observation['request_id'], model, stage]
            row = dict(label='D0_ALL' if model == 'M_D0' else model, runtime_model=model,
                       request_id=observation['request_id'], parent_id=q['parent_id'],
                       condition_id=q['condition_id'], report_group=q['report_group'],
                       noise_index=q['noise_index'], panel_index=i, N=q['N'], K=q['K'], stage=stage,
                       **{axis: observation[axis] for axis in AXES},
                       three_axis_joint=observation['three_axis_joint'], notes=observation['notes'],
                       reviewer=owner, review_kind='AI_PROXY_NONBLIND',
                       image_evidence=observation['image_evidence'],
                       source_mesh_path=identity['path'], source_mesh_sha256=identity['sha256'],
                       source_review_path=str(path), source_review_sha256=digest,
                       source_observation_index=observation_index,
                       source_observation_json=observation)
            rows.append(row)
        reviewers.append(dict(reviewer=owner, actual_pages=12, displayed_cells=48,
                              unique_observations=21, input_path=str(path), input_sha256=digest,
                              limitations=review.get('limitations')))
    if len(pages) != 24 or len(rows) != 42 or len(output_identity) != 42:
        raise ValueError('Whole-panel actual-page/unique-output coverage differs')
    expected = {(p['request_id'], label) for p in panel for label in LABELS}
    if {(r['request_id'], r['label']) for r in rows} != expected:
        raise ValueError('Expected seven models by six requests')
    rows.sort(key=lambda r: (r['panel_index'], LABELS.index(r['label'])))
    lookup = {(r['request_id'], r['label']): r for r in rows}
    transitions = []
    for item in panel:
        rid = item['request_id']
        for reference in ('RAW', 'D0_ALL', 'RAW_ALL'):
            for label in LABELS[1:]:
                if label != reference:
                    transitions.append(transition(lookup[rid, reference], lookup[rid, label]))
    if len(transitions) != 96:
        raise ValueError('Expected16 explicit comparisons per request')
    parents = sorted({r['parent_id'] for r in rows})
    if len(parents) != 3 or any(len({r['request_id'] for r in rows if r['parent_id'] == parent}) != 2
                                for parent in parents):
        raise ValueError('Expected three parents with two requests each')
    per_parent = {parent: {label: counts([r for r in rows if r['parent_id'] == parent and r['label'] == label])
                           for label in LABELS} for parent in parents}
    comparisons = {}
    for reference, label in dict.fromkeys((r['reference'], r['label']) for r in transitions):
        group = [r for r in transitions if r['reference'] == reference and r['label'] == label]
        comparisons[reference + ' -> ' + label] = dict(
            requests=len(group), joint_fates=dict(Counter(r['three_axis_fate'] for r in group)),
            gained_request_ids=[r['request_id'] for r in group if r['three_axis_fate'] == 'GAINED'],
            lost_request_ids=[r['request_id'] for r in group if r['three_axis_fate'] == 'LOST'],
            retained_request_ids=[r['request_id'] for r in group if r['three_axis_fate'] == 'RETAINED'],
            uncertain_request_ids=[r['request_id'] for r in group if r['three_axis_fate'] == 'UNCERTAIN_COMPARISON'],
            axis_transitions={axis: dict(Counter(r['transition_' + axis] for r in group)) for axis in AXES},
            per_parent={parent: dict(Counter(r['three_axis_fate'] for r in group if r['parent_id'] == parent))
                        for parent in parents})
    summary = dict(status='COMPLETE', version=VERSION, registration_sha256=registration_sha,
                   protocol_sha256=protocol_sha, actual_pages=24, displayed_cells=96, unique_outputs=42,
                   duplicate_baseline_display_cells_not_recounted=54,
                   models=list(LABELS), unique_requests=6, unique_parents=3,
                   label_display_alias={'M_D0': 'D0_ALL'}, runtime_model_preserved=True,
                   review_kind='AI_PROXY_NONBLIND', input_review_hashes=review_hashes, reviewers=reviewers,
                   counts={label: counts([r for r in rows if r['label'] == label]) for label in LABELS},
                   per_parent=per_parent,
                   per_group={group: {label: counts([r for r in rows if r['report_group'] == group and r['label'] == label])
                                      for label in LABELS} for group in sorted({r['report_group'] for r in rows})},
                   per_request={p['request_id']: {label: {key: lookup[p['request_id'], label][key]
                                  for key in (*AXES, 'stage', 'three_axis_joint', 'notes', 'reviewer', 'image_evidence')}
                                  for label in LABELS} for p in panel},
                   comparisons=comparisons, actual_page_evidence=page_evidence,
                   definition='Original five-axis labels and notes are copied, not rejudged. Strict three-axis flag is checked against original axes. Each repeated baseline is counted once per request. RAW is the pre-backend reference; all other models are CLEAN_FINAL_POST.',
                   limitations='Two nonblind AI proxy reviewers on repeatedly viewed development requests, not independent human certification. Disagreements/UNCERTAIN are retained. Joint positives do not certify no fragmentation, folding, conflict, watertightness, or exact source reconstruction.',
                   metrics_read=0, models_loaded=0, new_visual_labels=0, rendering_calls=0)
    # Validate all immutable review, page and protocol bytes again before output.
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError('Visual evidence changed during collection: ' + path)
    csvwrite(DEST / 'per_output.csv', rows)
    csvwrite(DEST / 'per_request_transitions.csv', transitions)
    write(DEST / 'summary.json', summary)
    receipt = dict(status='COMPLETE', version=VERSION, utc=datetime.now(timezone.utc).isoformat(),
                   registration_sha256=registration_sha, actual_pages=24, displayed_cells=96,
                   unique_outputs=42, unique_models=7, unique_requests=6, transition_rows=len(transitions),
                   inputs=inputs, input_review_hashes=review_hashes, original_labels_and_notes_unchanged=True,
                   metrics_read=0, models_loaded=0, detector_calls=0, scorer_calls=0, rendering_calls=0,
                   new_visual_labels=0,
                   files={str(DEST / name): sha(DEST / name) for name in output_names})
    write(done, receipt)
    return receipt


if __name__ == '__main__':
    result = run()
    print(json.dumps({key: value for key, value in result.items() if key not in ('inputs', 'files')},
                     ensure_ascii=False, indent=2))
