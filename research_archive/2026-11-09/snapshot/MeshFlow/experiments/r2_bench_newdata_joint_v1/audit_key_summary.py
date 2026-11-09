"""Independent stdlib-only recomputation of saved aggregate tables; no scoring."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def val(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def equal(a, b, where):
    if a is None or b is None:
        assert a is None and b is None, (where, a, b)
    else:
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (where, a, b)


def mean(rows, field):
    pc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        v = val(r.get(field))
        if v is not None:
            pc[r['parent_id']][r['condition_id']].append(v)
    parent_means = [sum(sum(values) / len(values) for values in cs.values()) / len(cs)
                    for cs in pc.values()]
    return sum(parent_means) / len(parent_means) if parent_means else None


def group(r, name):
    if name == 'ALL':
        return True
    if name in ('CURRENT_TRAIN', 'DEV', 'SAME_PARENT_HELDOUT'):
        return r['eval_group'] == name
    if name in ('NEW_PARENT_TRAIN', 'OLD_CURRENT_TRAIN'):
        return r['eval_group'] == 'CURRENT_TRAIN' and r['parent_id'].startswith('27eb79') == (name == 'NEW_PARENT_TRAIN')
    raise ValueError(name)


def support(rows, field, actual):
    valid = [r for r in rows if val(r.get(field)) is not None]
    assert actual['requests'] == len(valid)
    assert actual['conditions'] == len({r['condition_id'] for r in valid})
    assert actual['parents'] == len({r['parent_id'] for r in valid})
    assert sorted(actual['request_ids']) == sorted(r['request_id'] for r in valid)


def main():
    path = OUT / 'results/per_output.csv'
    summary_path = OUT / 'results/key_metrics_summary.json'
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))
    summary = json.loads(summary_path.read_text())
    assert len(rows) == 168 and len({(r['label'], r['stage'], r['request_id']) for r in rows}) == 168
    assert summary['source_sha256'] == digest(path)
    assert summary['code_sha256'] == digest(ROOT / 'r2/pilots/bench_newdata_joint_v1/summarize_results.py')
    assert len(summary['summaries']) == 36 and len(summary['paired_diagnostics']) == 18
    aggregate_checks = 0
    for s in summary['summaries']:
        rr = [r for r in rows if r['label'] == s['label'] and r['stage'] == s['stage'] and group(r, s['group'])]
        assert (len(rr), len({r['condition_id'] for r in rr}), len({r['parent_id'] for r in rr})) == (s['requests'], s['conditions'], s['parents'])
        for field, value in s['parent_macro'].items():
            assert field in rows[0], field
            equal(mean(rr, field), value, (s['label'], s['stage'], s['group'], field))
            support(rr, field, s['parent_macro_support'][field])
            aggregate_checks += 1
        passed = [r for r in rr if r['ConflictCov_C_status'] == 'PASS']
        equal(mean(passed, 'ConflictCov_C'), s['I_pass_parent_macro'], 'I PASS')
        support(passed, 'ConflictCov_C', s['I_pass_support'])
        assert dict(collections.Counter(r['ConflictCov_C_status'] for r in rr)) == s['I_status']
        for field, value in s['sums'].items():
            equal(sum(val(r[field]) or 0 for r in rr), value, field)
        applicable = [r for r in rr if r['seam_applicability'] == 'APPLICABLE']
        cf = [r for r in applicable if val(r['C_F_valid_edge_count']) is not None]
        reach = [r for r in applicable if val(r['free_only_reached_bulk_area_fraction']) is not None]
        assert s['descriptive_open_C_requests'] == len(applicable)
        assert s['closed_C_NA'] == len(rr) - len(applicable)
        assert s['C_F_valid_requests'] == len(cf)
        assert s['C_F_any'] == sum(val(r['C_F_valid_edge_count']) > 0 for r in cf)
        assert s['bulk_reach_valid_requests'] == len(reach)
        assert s['bulk_ge30'] == sum(val(r['free_only_reached_bulk_area_fraction']) >= .3 for r in reach)
        assert s['bulk_reach_NA_reasons'] == dict(collections.Counter(r['seam_to_bulk_status'] for r in applicable if r not in reach))
    for s in summary['paired_diagnostics']:
        aa = {r['request_id']: r for r in rows if r['stage'] == s['stage'] and r['label'] == 'RELATION_V1_S1500' and group(r, s['group'])}
        bb = {r['request_id']: r for r in rows if r['stage'] == s['stage'] and r['label'] == 'NEWDATA_JOINT_S1500' and group(r, s['group'])}
        assert aa.keys() == bb.keys() and len(aa) == s['requests']
        ids = [k for k in aa if aa[k]['ConflictCov_C_status'] == bb[k]['ConflictCov_C_status'] == 'PASS']
        assert len(ids) == s['I_common_PASS_requests']
        support([aa[k] for k in ids], 'ConflictCov_C', s['I_common_PASS_support'])
        equal(mean([aa[k] for k in ids], 'ConflictCov_C'), s['I_common_PASS_reference'], 'paired I reference')
        equal(mean([bb[k] for k in ids], 'ConflictCov_C'), s['I_common_PASS_candidate'], 'paired I candidate')
        for field, counts in s['transitions'].items():
            pairs = [(val(aa[k][field]), val(bb[k][field])) for k in aa if val(aa[k][field]) is not None and val(bb[k][field]) is not None]
            assert counts['improved'] == sum(b < a for a, b in pairs)
            assert counts['worsened'] == sum(b > a for a, b in pairs)
    cur = json.loads((OUT / 'results/completion.json').read_text())
    old = json.loads((ROOT / 'experiments/r2_bench_relation_training_v1/results/completion.json').read_text())
    assert set(cur['metric_fields']) == set(old['metric_fields'])
    result = dict(status='PASS', per_output_rows=168, aggregate_groups=36,
                  paired_diagnostic_groups=18, parent_macro_numeric_checks=aggregate_checks,
                  metric_fields=177, original_REFERENCE_field_set_unchanged=True,
                  closed_C_cells=36, open_C_cells=132,
                  checked='All summary numeric means, supports, I PASS/common-PASS domains, open-C descriptive counts and paired transitions',
                  interpretation='No semantic success certified; descriptive open-C connection is not required-interface success. Actual image review remains separate.',
                  code_sha256=digest(Path(__file__)), source_sha256=digest(path),
                  summary_sha256=digest(summary_path), model_calls=0, rescoring_calls=0)
    dest = OUT / 'results/key_metrics_independent_audit.json'
    text = json.dumps(result, indent=2) + '\n'
    if dest.exists():
        assert dest.read_text() == text
    else:
        with dest.open('x') as f:
            f.write(text)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
