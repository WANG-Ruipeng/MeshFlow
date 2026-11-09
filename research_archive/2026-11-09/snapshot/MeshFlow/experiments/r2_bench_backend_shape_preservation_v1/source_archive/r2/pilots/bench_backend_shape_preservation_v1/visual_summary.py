"""Validate and aggregate only completed, actual five-axis AI_PROXY reviews.

This script does not assign ratings, inspect scores to infer labels, or copy old
labels. It refuses missing, duplicate, unviewed, or image-mismatched cells.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path

from .common import OUT, read, sha, verify, write

AXES = {
    'overall_shape': ['REASONABLE', 'INCOMPLETE', 'UNCERTAIN'],
    'C_use': ['PLAUSIBLE', 'REPLACED_OR_IGNORED', 'UNCERTAIN'],
    'support_retention': ['PRESENT', 'MISSING_OR_DAMAGED', 'UNCERTAIN'],
    'visible_fragment_or_gap': ['YES', 'NO_OBVIOUS', 'UNCERTAIN'],
    'layering_or_fold': ['YES', 'NO_OBVIOUS', 'UNCERTAIN'],
}
STRICT = dict(overall_shape='REASONABLE', C_use='PLAUSIBLE', support_retention='PRESENT')
COLUMNS = ['COVERAGE_RAW', 'COVERAGE_FINAL_POST', 'COVERAGE_GEO_ONLY', 'SHAPE_D0_BASE', 'SHAPE_D0_FINAL']


def csvwrite(path, rows, empty_fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or list(empty_fields or [])
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def joint(row):
    return all(row[key] == value for key, value in STRICT.items())


def run():
    dest = OUT / 'visual_review'
    done = dest / 'COMPLETE.json'
    if done.exists():
        result = read(done)
        verify(result['files'])
        verify(result['review_input_hashes'])
        return result
    if dest.exists():
        raise RuntimeError('Partial visual summary retained; inspect before explicit recovery')
    manifest_path = OUT / 'gallery/review_manifest.json'
    manifest = read(manifest_path)
    templates = {row['cell_id']: row for row in manifest['cells_to_review']}
    if manifest['axes'] != AXES or len(templates) != 150 or manifest['pages'] != 30:
        raise ValueError('Unexpected frozen five-axis150-cell review contract')
    shards = sorted((OUT / 'gallery/review_shards').glob('*.json'))
    if len(shards) != 3:
        raise ValueError('Wait for the three complete actual-review shards')
    rows, observed = [], set()
    input_hashes = {str(manifest_path): sha(manifest_path)}
    for path in shards:
        data = read(path)
        input_hashes[str(path)] = sha(path)
        for source_row in data['rows']:
            row = dict(source_row)
            cell_id = row['cell_id']
            if cell_id in observed or cell_id not in templates:
                raise ValueError('Duplicate or unknown reviewed cell: ' + cell_id)
            template = templates[cell_id]
            for key in ('request_id', 'parent_id', 'condition_id', 'noise_index', 'eval_group',
                        'column', 'label', 'stage', 'image_sha256'):
                if row.get(key) != template[key]:
                    raise ValueError('Review identity mismatch: ' + cell_id + ' / ' + key)
            if row.get('actually_viewed') is not True or row.get('reviewer_kind') != 'AI_PROXY' or row.get('blind') is not False:
                raise ValueError('Incomplete or mislabeled actual review: ' + cell_id)
            if not row.get('reviewer') or not row.get('actual_view_source') or not row.get('viewed_at_utc') or not row.get('note'):
                raise ValueError('Actual-view provenance or observation note is missing: ' + cell_id)
            image = Path(template['image_path'])
            if sha(image) != template['image_sha256']:
                raise ValueError('Reviewed image changed: ' + cell_id)
            input_hashes[str(image)] = template['image_sha256']
            for axis, options in AXES.items():
                if row.get(axis) not in options:
                    raise ValueError('Missing/unknown actual rating: ' + cell_id + ' / ' + axis)
            row['strict_visual_joint'] = joint(row)
            row['column_name'] = COLUMNS[int(row['column'])]
            row['review_shard'] = str(path)
            rows.append(row)
            observed.add(cell_id)
    if observed != set(templates):
        raise ValueError('Actual review is incomplete; missing ' + str(len(set(templates) - observed)))
    rows.sort(key=lambda row: (row['request_id'], int(row['column'])))
    groups = sorted({row['eval_group'] for row in rows})
    group_rows = []
    for column in range(5):
        for group in groups + ['ALL']:
            selected = [r for r in rows if int(r['column']) == column and (group == 'ALL' or r['eval_group'] == group)]
            if not selected:
                continue
            parents = defaultdict(list)
            for row in selected:
                parents[row['parent_id']].append(row)
            summary = dict(column=column, column_name=COLUMNS[column], eval_group=group,
                requests=len(selected), parents=len(parents), strict_visual_joint_count=sum(joint(r) for r in selected),
                strict_visual_joint_fraction=sum(joint(r) for r in selected) / len(selected),
                strict_visual_joint_parent_macro_fraction=sum(sum(joint(r) for r in pr) / len(pr) for pr in parents.values()) / len(parents))
            for axis, options in AXES.items():
                counts = Counter(r[axis] for r in selected)
                for option in options:
                    summary[axis + '__' + option] = counts[option]
            group_rows.append(summary)
    indexed = {(row['request_id'], int(row['column'])): row for row in rows}
    requests = sorted({row['request_id'] for row in rows})
    transitions = []
    for rid in requests:
        for ref_col, candidate_col in ((1, 3), (1, 4), (2, 3), (2, 4)):
            before, after = indexed[(rid, ref_col)], indexed[(rid, candidate_col)]
            old, new = joint(before), joint(after)
            transition = 'PRESERVED_POSITIVE' if old and new else 'LOST_POSITIVE' if old else 'GAINED_POSITIVE' if new else 'REMAINED_NONPOSITIVE'
            result = dict(request_id=rid, parent_id=after['parent_id'], condition_id=after['condition_id'],
                noise_index=after['noise_index'], eval_group=after['eval_group'], reference=COLUMNS[ref_col], candidate=COLUMNS[candidate_col],
                reference_strict_visual_joint=old, candidate_strict_visual_joint=new, strict_joint_transition=transition)
            for axis in AXES:
                result[axis + '__reference'] = before[axis]
                result[axis + '__candidate'] = after[axis]
            transitions.append(result)
    transition_summary = []
    for reference, candidate in sorted({(r['reference'], r['candidate']) for r in transitions}):
        for group in groups + ['ALL']:
            selected = [r for r in transitions if r['reference'] == reference and r['candidate'] == candidate and (group == 'ALL' or r['eval_group'] == group)]
            counts = Counter(r['strict_joint_transition'] for r in selected)
            transition_summary.append(dict(reference=reference, candidate=candidate, eval_group=group,
                requests=len(selected), parents=len({r['parent_id'] for r in selected}),
                **{name: counts[name] for name in ('PRESERVED_POSITIVE', 'LOST_POSITIVE', 'GAINED_POSITIVE', 'REMAINED_NONPOSITIVE')}))
    losses = [r for r in transitions if r['reference'] == COLUMNS[1] and r['candidate'] == COLUMNS[4] and r['strict_joint_transition'] == 'LOST_POSITIVE']
    final_pairs = [r for r in transitions if r['reference'] == COLUMNS[1] and r['candidate'] == COLUMNS[4]]
    group_gate = {}
    for group in groups:
        selected = [r for r in final_pairs if r['eval_group'] == group]
        b = sum(r['reference_strict_visual_joint'] for r in selected)
        a = sum(r['candidate_strict_visual_joint'] for r in selected)
        group_gate[group] = dict(reference_count=b, candidate_count=a, nondecrease=a >= b)
    nonnew_gained = sorted({r['parent_id'] for r in final_pairs if r['eval_group'] != 'NEW_TRAIN' and r['strict_joint_transition'] == 'GAINED_POSITIVE'})
    dest.mkdir(parents=True)
    csvwrite(dest / 'per_cell.csv', rows)
    csvwrite(dest / 'group_summary.csv', group_rows)
    csvwrite(dest / 'paired_transitions.csv', transitions)
    csvwrite(dest / 'strict_joint_transition_by_group.csv', transition_summary)
    csvwrite(dest / 'original_positive_losses.csv', losses, list(transitions[0]))
    result = dict(status='COMPLETE', reviewer_kind='AI_PROXY', blind=False, pages=30, cells=150,
        all_cells_actually_viewed=True, no_inferred_or_copied_labels=True,
        strict_visual_joint_definition=STRICT,
        limitation='Visual conjunction of overall plausibility, C use and support presence; not legal-interface/mesh-quality certification and not independent human review.',
        original_final_to_candidate_final=group_gate,
        all_groups_visual_nondecrease=all(v['nondecrease'] for v in group_gate.values()),
        original_positive_losses=len(losses), non_new_training_parents_with_strict_gains=nonnew_gained,
        review_input_hashes=input_hashes, summary_source_sha256=sha(__file__),
        files={str(path): sha(path) for path in dest.iterdir() if path.is_file()})
    write(done, result)
    return result


if __name__ == '__main__':
    result = run()
    print('ACTUAL_VISUAL_SUMMARY_COMPLETE', result['pages'], result['cells'], result['original_positive_losses'])
