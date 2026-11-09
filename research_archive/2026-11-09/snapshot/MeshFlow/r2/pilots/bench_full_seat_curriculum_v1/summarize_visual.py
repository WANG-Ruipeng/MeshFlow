"""Summarize completed actual image observations; never assign visual labels."""
from __future__ import annotations
import collections
import itertools
import json
from . import visual_review as V


def run():
    validation = V.validate()
    if validation['status'] != 'COMPLETE_ACTUAL_AI_PROXY_REVIEW':
        raise ValueError('Actual 216-cell review is not complete')
    import csv
    assignment = V.read(V.REVIEW / 'assignments.json')
    rows, inputs = [], {}
    for task in assignment['tasks'].values():
        path = V.resolve(task['records_csv'])
        inputs[str(path.relative_to(V.ROOT))] = V.sha(path)
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows.extend(csv.DictReader(f))
    rows.sort(key=lambda r: (r['parent_id'], r['request_id'], r['stage'], r['label']))
    groups = ('CURRENT_TRAIN', 'DEV')
    counts, parent_rows = [], []
    for label, stage, group in itertools.product(V.LABELS, V.STAGES, groups):
        block = [r for r in rows if (r['label'], r['stage'], r['eval_group']) == (label, stage, group)]
        parents = sorted({r['parent_id'] for r in block})
        for axis, values in V.AXES.items():
            for value in values:
                fractions = []
                for parent in parents:
                    pr = [r for r in block if r['parent_id'] == parent]
                    count = sum(r[axis] == value for r in pr)
                    fractions.append(count / len(pr))
                    parent_rows.append(dict(label=label, stage=stage, eval_group=group,
                        parent_id=parent, axis=axis, category=value, count=count,
                        requests=len(pr), fraction=count / len(pr)))
                counts.append(dict(label=label, stage=stage, eval_group=group,
                    axis=axis, category=value, count=sum(r[axis] == value for r in block),
                    requests=len(block), parents=len(parents),
                    parent_equal_fraction=sum(fractions) / len(fractions)))
    indexed = {(r['label'], r['stage'], r['request_id']): r for r in rows}
    paired = []
    for reference, candidate in itertools.combinations(V.LABELS, 2):
        for stage in V.STAGES:
            for base in [r for r in rows if r['label'] == reference and r['stage'] == stage]:
                cand = indexed[candidate, stage, base['request_id']]
                for axis in V.AXES:
                    paired.append(dict(comparison=candidate+'__minus__'+reference,
                        stage=stage, eval_group=base['eval_group'], parent_id=base['parent_id'],
                        request_id=base['request_id'], axis=axis,
                        reference=base[axis], candidate=cand[axis],
                        unchanged=base[axis] == cand[axis]))
    stage_pairs = []
    for label in V.LABELS:
        for before, after in zip(V.STAGES, V.STAGES[1:]):
            for base in [r for r in rows if r['label'] == label and r['stage'] == before]:
                cand = indexed[label, after, base['request_id']]
                for axis in V.AXES:
                    stage_pairs.append(dict(label=label, before_stage=before, after_stage=after,
                        eval_group=base['eval_group'], parent_id=base['parent_id'],
                        request_id=base['request_id'], axis=axis,
                        before=base[axis], after=cand[axis], unchanged=base[axis] == cand[axis]))
    transition_counts = collections.Counter((r['comparison'], r['stage'], r['eval_group'],
        r['axis'], r['reference'], r['candidate']) for r in paired)
    transitions = [dict(zip(('comparison','stage','eval_group','axis','reference','candidate'), key),
        request_count=count) for key, count in sorted(transition_counts.items())]
    output = V.REVIEW / 'summary_v1'
    artifacts = {}
    for name, table in (('actual_observations.csv',rows), ('axis_counts.csv',counts),
        ('per_parent_axis_counts.csv',parent_rows), ('paired_model_transitions.csv',paired),
        ('model_transition_counts.csv',transitions), ('paired_stage_transitions.csv',stage_pairs)):
        path = output / name
        V.immutable(path, V.csv_text(table, list(table[0])))
        artifacts[name] = V.sha(path)
    result = dict(status='COMPLETE', validation=validation, input_sha256=inputs,
        code_sha256=V.sha(__file__), artifacts=artifacts, generated_ratings=0,
        model_calls=0, scoring_calls=0, rendered_images=0,
        interpretation='Nonblind AI_PROXY; independent axes, all uncertainty retained; no new joint-success definition. Equal-parent rates use two fixed noises per C. DEV is reused development evidence.')
    V.immutable(output/'completion.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False))
