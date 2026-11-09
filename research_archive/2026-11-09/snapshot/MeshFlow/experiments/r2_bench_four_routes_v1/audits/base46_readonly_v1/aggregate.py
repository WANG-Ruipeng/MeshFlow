"""Independent BASE-only read aggregation. No model/mesh/scorer imports or writes."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
HYBRID = ROOT / 'experiments/r2_bench_four_routes_v1/hybrid'
FIELDS = {
    'CF': 'CF_illegal_affected_free_fraction',
    'FF': 'FF_illegal_affected_free_fraction',
    'legal_CF_edge_count': 'C_F_valid_edge_count',
    'legal_CF_edge_length': 'C_F_valid_edge_total_length',
    'has_legal_CF_edge': 'C_F_has_valid_edge',
    'legal_CF_reached_free_area_fraction': 'reached_free_area_fraction',
    'actual_free_faces': 'actual_free_faces',
    'free_total_surface_area': 'free_total_surface_area',
}
sources = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    sources[str(path)] = sha(path)
    return value


def write_json(name, value):
    with (DEST / name).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(name, rows):
    with (DEST / name).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checked_score(path, q, expected_label, protected=None):
    path = Path(path)
    row = read(path)
    complete = read(path.with_name('complete.json'))
    score = read(path.with_name('score.json'))
    if complete['status'] != 'PASS' or score['status'] != 'PASS' or complete['row'] != row:
        raise ValueError('Completed score/row differs: ' + str(path))
    if row['label'] != expected_label or score['label'] != expected_label:
        raise ValueError('Label differs')
    for field in ('request_id', 'condition_id', 'parent_id', 'noise_index', 'N', 'K'):
        if row[field] != q[field] or score[field] != q[field]:
            raise ValueError('Registered identity differs: ' + field)
    for selected in (path, path.with_name('score.json')):
        if complete['files'].get(str(selected)) != sha(selected):
            raise ValueError('Score receipt hash differs')
        if protected is not None and protected.get(str(selected)) != sha(selected):
            raise ValueError('Original M frozen hash differs')
    if row['scoring_definition_version'] != complete['version'] or complete['version'] != score['version']:
        raise ValueError('Scoring definition mismatch')
    if row['C_preserved_exact'] is not True or score['C_preserved_exact'] is not True:
        raise ValueError('C preservation differs')
    return row


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def averages(rows, metrics):
    return {key: mean(r[key] for r in rows if numeric(r[key])) if any(numeric(r[key]) for r in rows) else None
            for key in metrics}


def run():
    if (DEST / 'receipt.json').exists():
        raise ValueError('Independent receipt already exists; refusing repeat/overwrite')
    registration = HYBRID / 'registration.json'
    reg = read(registration)
    regsha = sha(registration)
    expected = {t['request_id'] for t in reg['tasks']}
    actual = {p.parents[2].name for p in (HYBRID / 'cpu/HYBRID').glob('*/BASE_POST/scores/row.json')}
    if len(expected) != 46 or actual != expected:
        raise ValueError('BASE row identity set is not exactly registered46')
    rows = []
    definitions = set()
    for task in reg['tasks']:
        q = task['request']
        rid = q['request_id']
        started = read(HYBRID / 'outputs' / rid / 'started.json')
        if started['registration_sha256'] != regsha or started['model_calls'] != 0:
            raise ValueError('Candidate execution registration identity differs')
        reference = checked_score(task['paths']['BASE_POST_row'], q, 'M', reg['protected'])
        candidate = checked_score(HYBRID / 'cpu/HYBRID' / rid / 'BASE_POST/scores/row.json', q, 'HYBRID')
        definitions.update((reference['scoring_definition_version'], candidate['scoring_definition_version']))
        selection = read(HYBRID / 'outputs' / rid / 'selection_summary.json')
        if selection['N'] != q['N'] or selection['K'] != q['K'] or selection['C_bytes_preserved'] is not True:
            raise ValueError('Selection identity differs')
        if selection['model_calls'] != 0 or selection['new_noise'] != 0 or selection['expansion_rounds'] != 1:
            raise ValueError('Selection contract differs')
        common_I = reference['I_status'] == candidate['I_status'] == 'PASS'
        row = {key:q[key] for key in ('request_id','condition_id','parent_id','noise_index','report_group','N','K','closed_C')}
        row.update(stage='BASE_POST', comparison='HYBRID_minus_M',
                   M_I_status=reference['I_status'], HYBRID_I_status=candidate['I_status'],
                   I_common_PASS=common_I, M_original_ConflictCov_C=reference['ConflictCov_C'],
                   HYBRID_original_ConflictCov_C=candidate['ConflictCov_C'],
                   M_seam_to_bulk_status=reference['seam_to_bulk_status'],
                   HYBRID_seam_to_bulk_status=candidate['seam_to_bulk_status'],
                   M_original_row_stage=reference['stage'], HYBRID_original_row_stage=candidate['stage'])
        for label, value in (('M', reference), ('HYBRID', candidate)):
            for short, original in FIELDS.items():
                row[label + '_' + short] = value[original]
            row[label + '_I_common_PASS'] = value['ConflictCov_C'] if common_I else None
            row[label + '_total_faces'] = value['actual_free_faces'] + q['K']
            row[label + '_free_face_retention_fraction'] = value['actual_free_faces'] / (q['N'] - q['K'])
        for short in (*FIELDS, 'I_common_PASS', 'total_faces', 'free_face_retention_fraction'):
            a, b = row['M_' + short], row['HYBRID_' + short]
            row['delta_' + short] = b - a if numeric(a) and numeric(b) else None
        row.update(selected_RAW_faces=selection['selected_RAW_faces'], original_free_faces=q['N'] - q['K'],
                   selected_RAW_free_face_fraction=selection['selected_RAW_faces'] / (q['N'] - q['K']),
                   selected_RAW_source_area_fraction=selection['selected_raw_area']/selection['free_raw_area'] if selection['free_raw_area'] else None,
                   all_free_selected_RAW=int(selection['selected_RAW_faces'] == q['N'] - q['K']))
        rows.append(row)
    if len(definitions) != 1:
        raise ValueError('Mixed original scorer definitions')
    parents = defaultdict(list)
    for row in rows:
        parents[row['parent_id']].append(row)
    if len(parents) != 23 or any(len(rs) != 2 or {r['noise_index'] for r in rs} != {0,1}
                                or len({r['condition_id'] for r in rs}) != 1 for rs in parents.values()):
        raise ValueError('Expected23 parents/conditions, two fixed noises each')
    if sum(r['I_common_PASS'] for r in rows) != 38 or len({r['parent_id'] for r in rows if r['I_common_PASS']}) != 19:
        raise ValueError('I common-PASS support differs from19 parents/38 requests')
    metric_names = [k for k in rows[0] if k.startswith(('M_', 'HYBRID_', 'delta_')) and
                    k not in ('M_I_status','HYBRID_I_status','M_seam_to_bulk_status','HYBRID_seam_to_bulk_status',
                              'M_original_row_stage','HYBRID_original_row_stage','M_original_ConflictCov_C','HYBRID_original_ConflictCov_C')]
    metric_names += ['selected_RAW_free_face_fraction','selected_RAW_source_area_fraction','all_free_selected_RAW']
    parent_rows = []
    for pid, rs in parents.items():
        parent_rows.append(dict(parent_id=pid, report_group=rs[0]['report_group'], stage='BASE_POST',
                                requests=len(rs), I_common_PASS_requests=sum(r['I_common_PASS'] for r in rs),
                                closed_C=rs[0]['closed_C'], **averages(rs, metric_names)))
    groups = ['ALL'] + list(dict.fromkeys(r['report_group'] for r in parent_rows))
    group_rows = []
    for group in groups:
        ps = [p for p in parent_rows if group == 'ALL' or p['report_group'] == group]
        rs = [r for r in rows if group == 'ALL' or r['report_group'] == group]
        for metric in metric_names:
            vals = [p[metric] for p in ps if numeric(p[metric])]
            group_rows.append(dict(report_group=group, stage='BASE_POST', metric=metric,
                                   parent_macro_mean=mean(vals) if vals else None,
                                   valid_parent_count=len(vals), total_parent_count=len(ps),
                                   valid_request_count=sum(numeric(r[metric]) for r in rs), total_request_count=len(rs)))
    totals = {}
    for group in groups:
        rs = [r for r in rows if group == 'ALL' or r['report_group'] == group]
        totals[group] = dict(requests=len(rs), parents=len({r['parent_id'] for r in rs}),
                            I_common_PASS_requests=sum(r['I_common_PASS'] for r in rs),
                            I_common_PASS_parents=len({r['parent_id'] for r in rs if r['I_common_PASS']}),
                            M_legal_edge_requests=sum(r['M_has_legal_CF_edge'] for r in rs),
                            HYBRID_legal_edge_requests=sum(r['HYBRID_has_legal_CF_edge'] for r in rs),
                            M_legal_edge_length_total=sum(r['M_legal_CF_edge_length'] for r in rs),
                            HYBRID_legal_edge_length_total=sum(r['HYBRID_legal_CF_edge_length'] for r in rs),
                            all_free_selected_RAW_requests=sum(r['all_free_selected_RAW'] for r in rs),
                            selected_RAW_faces_total=sum(r['selected_RAW_faces'] for r in rs),
                            original_free_faces_total=sum(r['original_free_faces'] for r in rs),
                            pooled_RAW_face_fraction=sum(r['selected_RAW_faces'] for r in rs)/sum(r['original_free_faces'] for r in rs))
    summary = dict(status='COMPLETE_BASE_STAGE_ONLY_FINAL_NOT_ASSESSED', utc=datetime.now(timezone.utc).isoformat(),
                   stage='BASE_POST', registered_requests=46, parents=23, original_metric_version=next(iter(definitions)),
                   aggregation='Two fixed noises averaged within condition; one condition per parent here; equal parent means by report_group and ALL.',
                   I_policy='Only the same requests where both original I_status values are PASS;19 parents/38 requests. Other values remain NA, never zero-filled.',
                   CF_FF_policy='Original illegal affected free-face fraction on each BASE output actual free-face denominator; all23 parents/46 requests.',
                   interface_policy='Original legal C/F edges and reached free area are descriptive; closed C has no required open interface and zero is not failure.',
                   face_policy='Counts/ratios from existing score rows; more retained faces or area alone do not prove completeness or quality.',
                   selection_policy='RAW fallback fractions describe pre-GEO whole-face selection among original N-K free slots; pooled fraction is separate from equal-parent mean.',
                   stage_identity_policy='BASE directory and completed score receipt fix the stage; original generic row.stage strings are preserved in CSV.',
                   source_status_counts={label:dict(Counter(r[label+'_I_status'] for r in rows)) for label in ('M','HYBRID')},
                   totals=totals, all_parent_macro={r['metric']:r['parent_macro_mean'] for r in group_rows if r['report_group']=='ALL'},
                   per_group_parent_macro={g:{r['metric']:r['parent_macro_mean'] for r in group_rows if r['report_group']==g} for g in groups},
                   model_calls=0, scoring_calls=0, geometry_calls=0, final_chain_read=False, visual_labels_read=False)
    for path, digest in sources.items():
        if sha(path) != digest:
            raise ValueError('BASE evidence changed during read aggregation')
    write_csv('per_request.csv', rows)
    write_csv('per_parent.csv', parent_rows)
    write_csv('group_parent_macro.csv', group_rows)
    write_json('summary.json', summary)
    write_json('source_identities.json', dict(registration_sha256=regsha, files=sources))
    outputs = {str(p):sha(p) for p in DEST.iterdir() if p.is_file()}
    write_json('receipt.json', dict(status='PASS_BASE46_READ_ONLY', stage='BASE_POST', registered_requests=46,
                                  M_rows=46, HYBRID_rows=46, I_common_PASS_parents=19, I_common_PASS_requests=38,
                                  registration_sha256=regsha, output_files=outputs, script_sha256=sha(__file__),
                                  model_calls=0, scoring_calls=0, geometry_calls=0, final_chain_read=False,
                                  source_files_unchanged=True, final_result_claim=False))
    print(json.dumps(dict(status=summary['status'], destination=str(DEST), requests=46, parents=23,
                          model_calls=0, scoring_calls=0, final_chain_read=False)))


if __name__ == '__main__':
    run()
