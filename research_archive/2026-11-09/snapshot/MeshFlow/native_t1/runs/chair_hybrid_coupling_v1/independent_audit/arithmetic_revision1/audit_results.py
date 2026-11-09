"""Independent arithmetic audit of saved final results; zero model calls."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
import numpy as np

OUT = Path(__file__).resolve().parents[1]
P = OUT / 'paired_results'
def read(name):
    return json.loads((P / (name + '.json')).read_text(encoding='utf-8'))
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-13), (a, b)

rows = read('per_output')
assert len(rows) == 432
assert sum(r['status'] == 'NOT_APPLICABLE' for r in rows) == 18
assert sum(r['status'] == 'PASS' for r in rows) == 414
for r in rows:
    if r['status'] != 'PASS':
        continue
    assert r['C_preserved'] and r['N_preserved'] and r['finite']
    assert sha(r['fixed_vector_path']) == r['fixed_vector_sha256']
    with np.load(r['fixed_vector_path'], allow_pickle=False) as f:
        d = f['distances_L']
        assert len(d) == r['Q'] == 512
        close(float(np.mean(d*d)), r['fixed_query_MSE'])
        close(float(np.mean(d <= .01)), r['coverage_fraction'])
        close(float(np.quantile(d, .95)), r['fixed_query_p95_L'])

primary = [r for r in rows if r['table'] == 'A' and r['split'] == 'validation' and r['ratio'] == .2]
lookup = {(r['checkpoint_label'], r['uid'], r['noise_index']): r for r in primary}
assert len(primary) == len(lookup) == 48
uids = sorted({r['uid'] for r in primary})
assert len(uids) == 8
derived = []
for before, after in [('START', 'OT_BASE'), ('OT_BASE', 'OT_HYBRID'), ('START', 'OT_HYBRID')]:
    parent_deltas = []
    before_values = []
    after_values = []
    for uid in uids:
        b = mean(lookup[before, uid, n]['fixed_query_MSE'] for n in (0, 1))
        a = mean(lookup[after, uid, n]['fixed_query_MSE'] for n in (0, 1))
        for n in (0, 1):
            x, y = lookup[before, uid, n], lookup[after, uid, n]
            for field in ('input_sha256', 'query_sha256', 'noise_sha256', 'mask_sha256', 'L'):
                assert x[field] == y[field]
        before_values.append(b); after_values.append(a); parent_deltas.append(a-b)
    name = after + '-' + before
    saved = next(r for r in read('primary_summary') if r['comparison'] == name)
    for field, value in [('delta_mean', mean(parent_deltas)), ('paired_delta_median', median(parent_deltas)),
                         ('before_mean', mean(before_values)), ('after_mean', mean(after_values))]:
        close(saved[field], value)
    assert saved['improved_count'] == sum(d < 0 for d in parent_deltas)
    assert saved['valid_pairs'] == 8
    derived.append(dict(comparison=name, before=mean(before_values), after=mean(after_values),
                        mean_delta=mean(parent_deltas), paired_delta_median=median(parent_deltas),
                        improved=sum(d < 0 for d in parent_deltas), parents=8))

key_fields = ('checkpoint_label', 'candidate', 'uid', 'noise_index')
matrices = defaultdict(dict)
for r in read('cross_matrix'):
    key = tuple(r[k] for k in key_fields)
    cell = (r['alpha'], r['target_beta'])
    assert cell not in matrices[key]
    matrices[key][cell] = r['E_RMS_L0']
assert len(matrices) == 144 and all(len(m) == 9 for m in matrices.values())
alphas = (.9, 1., 1.1)
for r in read('target_column_audit'):
    m = matrices[tuple(r[k] for k in key_fields)]
    beta = r['target_beta']
    value = mean(m[a, beta] for a in alphas if a != beta) - m[beta, beta]
    close(value, r['A_column'])
    assert r['column_strict_win'] == all(m[beta, beta] < m[a, beta] for a in alphas if a != beta)
    if r['candidate'] == 'NO_RESPONSE':
        close(value, 0.)
        assert not r['column_strict_win']

for r in read('width_per_curve'):
    if r['response_status'] != 'PASS':
        continue
    close(r['r_base'], r['W_10'] / r['W_GT_10'])
    close(r['b_self'], (r['W_11'] - r['W_09']) / (.2*r['W_10']))
    close(r['b_GT'], (r['W_11'] - r['W_09']) / (.2*r['W_GT_10']))
    close(r['b_self_abs_error'], abs(r['b_self'] - 1.))
    close(r['b_GT_abs_error'], abs(r['b_GT'] - 1.))

parts = defaultdict(list)
for r in read('coverage_partition'):
    parts[r['comparison'], r['case_id']].append(r)
for group in parts.values():
    assert {r['partition'] for r in group} == {'both_in', 'enter', 'leave', 'both_out'}
    close(sum(r['fraction'] for r in group), 1.)
    close(sum(r['fixed_query_MSE_contribution'] for r in group), group[0]['total_MSE_delta'])

result = dict(status='PASS', actual_model_calls=0, fixed_vector_rows_recomputed=414,
              primary=derived, full_matrices=144, matrix_cells=1296, target_columns=432,
              width_curves=144, paired_coverage_outputs=len(parts),
              inputs={str(f.relative_to(OUT)): sha(f) for f in sorted(P.glob('*.json'))},
              audit_script_sha256=sha(__file__),
              limits='Independent saved-distance and arithmetic audit; no new surface queries or inference.')
(OUT / 'independent_audit/results_arithmetic_audit.json').write_text(
    json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k != 'inputs'}))
