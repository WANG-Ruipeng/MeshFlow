"""Read-only cross-check of saved CPU evidence; writes only a root verification receipt.

No neural imports, scoring, optimization, generation, source edits or retries.
Run inside WSL using the retained Python environment. Re-running this verifier
does not re-run any diagnostic. It checks current bytes and existing tables.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
errors = []

def j(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def number(r, key):
    return float(r[key])

def check(ok, label):
    if not ok:
        errors.append(label)

def same(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)

def to_path(p):
    if len(p) > 2 and p[1] == ':':
        return Path('/mnt/' + p[0].lower() + p[2:].replace('\\', '/'))
    return Path(p)

def hashcheck(mapping, name):
    for path, digest in mapping.items():
        p = to_path(path)
        check(p.is_file(), f'{name}: missing {path}')
        if p.is_file():
            check(hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'{name}: changed {path}')
    return len(mapping)

a = rows(ROOT / 'idea1b/per_output.csv')
summaries = rows(ROOT / 'idea1b/summary.csv')
check(len(a) == 48, '1B row budget')
check(len({r['request_id'] for r in a}) == 24, '1B request budget')
check(len({(r['request_id'], r['label']) for r in a}) == 48, '1B unique score cells')
sum_fields = ['screening_joint', 'valid_cf_edges', 'valid_cf_length',
              'CF_lost_legal_edges', 'CF_new_legal_edges', 'FF_lost_legal_edges',
              'FF_new_illegal_pairs', 'FF_illegal_pairs_removed', 'deleted_source_faces']
for s in summaries:
    group = [r for r in a if r['label'] == s['label'] and (s['split'] == 'ALL' or r['split'] == s['split'])]
    check(len(group) == int(s['outputs']), f"1B group size {s['label']}/{s['split']}")
    check(sum(number(r, 'valid_cf_edges') > 0 for r in group) == int(s['interfaces']), '1B interface summary')
    for key in sum_fields:
        check(same(sum(number(r, key) for r in group), s[key]), f'1B sum {key}')
    for key in s:
        if key.startswith('delta_'):
            check(same(sum(number(r, key) for r in group) / len(group), s[key]), f'1B mean {key}')
for r in a:
    check(number(r, 'optimizer_actual_nfev') <= 300, '1B objective budget')
    check(number(r, 'affine_POST_max_pctL') <= 3.500001, '1B POST affine cap')
    check(number(r, 'affine_pre_geo_max_pctL') <= 3.500001, '1B pre affine cap')
    check(number(r, 'B2_bound_max_from_transformed_pre_pctL') <= 2.000001, '1B binding cap')
    if r['label'] == 'AFFINE_B2':
        check(number(r, 'incremental_B2_CF_new_illegal_pairs') == 0, 'B2 incremental CF guard')
affine = [r for r in a if r['label'] == 'AFFINE']
b2 = [r for r in a if r['label'] == 'AFFINE_B2']
pairs = rows(ROOT / 'idea1b/paired_screening_vs_old_B2.csv')
new_ids = [r['short_id'] for r in pairs if r['new_vs_original_B2'].lower() == 'true']
check(new_ids == ['2d08a64e_n1'], '1B identity of sole B2 increment')
result_1b = {
    'rows': len(a), 'baseline_joint': sum(int(r['baseline_joint']) for r in affine),
    'affine_joint': sum(int(r['screening_joint']) for r in affine),
    'affine_B2_joint': sum(int(r['screening_joint']) for r in b2),
    'sole_new_vs_B2': new_ids,
    'CF_retained': sum(int(r['CF_retained_legal_edges']) for r in b2),
    'CF_new': sum(int(r['CF_new_legal_edges']) for r in b2),
    'FF_new_illegal': sum(int(r['FF_new_illegal_pairs']) for r in b2),
    'FF_removed_illegal': sum(int(r['FF_illegal_pairs_removed']) for r in b2)
}
result_1b.update({
    'new_CF_illegal_pairs': sum(int(r['CF_new_illegal_pairs']) for r in b2),
    'removed_CF_illegal_pairs': sum(int(r['CF_illegal_pairs_removed']) for r in b2),
    'requests_with_new_CF_illegal_pairs': sum(int(r['CF_new_illegal_pairs']) > 0 for r in b2),
    'old_B2_shared_length': sum(float(r['old_B2_valid_cf_length']) for r in affine),
    'affine_B2_shared_length': sum(float(r['valid_cf_length']) for r in b2)
})

c = rows(ROOT / 'idea3/per_condition.csv')
s3 = j(ROOT / 'idea3/summary.json')
check(len(c) == 53 and len({r['condition_id'] for r in c}) == 50, '3 condition identities')
keys3 = ['C_open_edges', 'legal_CY_edges', 'legal_CY_opposite_edges', 'source_null_slots',
         'distinct_interface_Y_faces', 'single_edge_Y_faces', 'multi_edge_Y_faces',
         'third_vertex_is_C_ID_edges', 'free_faces']
for group, subset in [('TRAIN47', [r for r in c if r['group'] == 'TRAIN47']),
                      ('PANEL6', [r for r in c if r['group'] == 'PANEL6']),
                      ('PANEL_DEV', [r for r in c if r['group'] == 'PANEL6' and r['split'] == 'DEV'])]:
    check(len(subset) == s3[group]['conditions'], f'3 {group} count')
    for key in keys3:
        check(same(sum(number(r, key) for r in subset), s3[group][key]), f'3 {group} {key}')
    check(sum(number(r, 'FF_illegal_pairs') > 0 for r in subset) == s3[group]['conditions_with_inherited_FF_illegal'], f'3 {group} FF risks')
controls = rows(ROOT / 'idea3/controls.csv')
check(len(controls) == 9, '3 controls count')
for r in controls:
    check(r['control_status'] == 'PASS', f"3 control {r['control']}")
    check(r['expected_valid_CF_edges'] == r['C_F_valid_edge_count'], '3 expected interface')
    check(r['expected_opposite_CF_edges'] == r['C_F_valid_opposite_winding_edge_count'], '3 expected orientation')
for r in c:
    check(r['C_source_vertex_ID_roundtrip'] == 'True', '3 C source identities')
    check(r['owner_plus_bulk_exact_source_face_budget'] == 'True', '3 face budget')

d = rows(ROOT / 'idea4a/path_diagnostics.csv')
dc = rows(ROOT / 'idea4a/data_contract_checks.csv')
rec4 = j(ROOT / 'idea4a/receipt.json')
check(len(d) == 20 and len(dc) == 4, '4A sample/time budget')
check({float(r['t']) for r in d} == {.1, .3, .5, .7, .9}, '4A fixed times')
for key, receiptkey in [('finite_difference_max_abs', 'finite_difference_max_abs'),
                        ('perfect_velocity_corrected_endpoint_max_abs', 'corrected_endpoint_max_abs'),
                        ('perfect_velocity_naive_endpoint_max_pct_L', 'perfect_velocity_naive_endpoint_max_pct_L'),
                        ('delta_state_max_pct_L', 'max_state_displacement_pct_L')]:
    check(same(max(number(r, key) for r in d), rec4[receiptkey]), f'4A {key}')
for r in dc:
    for key in ['endpoint0_exact', 'endpoint1_exact', 'source_corner_transport_exact',
                'interface_transport_exact', 'known_identity_exact',
                'affine_commutes_with_face_corner_permutation', 'synthetic_padding_zero_preserved']:
        check(r[key] == 'True', f'4A {key}')
    check(r['mask_changed'] == 'False' and r['OT_cache'] == 'HIT', '4A mask/cache')
check(rec4['actual_OT_calls'] == 0 and rec4['model_forwards'] == 0 and rec4['optimizer_updates'] == 0, '4A no models/OT')

protected = {r['path']: r['sha256'] for r in j(ROOT / 'root_protected_before.json')}
protected_counts = {'root': hashcheck(protected, 'root protected')}
protected_counts['idea1b'] = hashcheck(j(ROOT / 'idea1b/registration.json')['protected_files'], '1B protected')
protected_counts['idea3'] = hashcheck(j(ROOT / 'idea3/source_hashes.json')['original_sources'], '3 protected')
protected_counts['idea4a'] = hashcheck(j(ROOT / 'idea4a/source_hashes.json'), '4A protected')
output_counts = {
    'idea1b': hashcheck(j(ROOT / 'idea1b/complete.json')['files'], '1B output'),
    'idea3': hashcheck(j(ROOT / 'idea3/completion.json')['output_hashes'], '3 output')
}

out = {
    'status': 'PASS' if not errors else 'FAIL',
    'utc': datetime.now(timezone.utc).isoformat(), 'errors': errors,
    'protected_hash_count_by_scope': protected_counts,
    'recorded_output_hash_counts': output_counts,
    'idea1b_recomputed': result_1b, 'idea3_train_recomputed': s3['TRAIN47'],
    'idea4a_rows': len(d), 'actual_model_calls': 0, 'new_scoring_calls': 0,
    'limits': 'Independent arithmetic/identity/hash checks of saved records. Does not independently reclassify intersections or rerun scientific experiments. Does not certify model learning.'
}
(ROOT / 'independent_verification.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': out['status'], 'errors': errors, 'protected_hash_count_by_scope': protected_counts}, ensure_ascii=False))
raise SystemExit(bool(errors))
