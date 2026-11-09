"""Package completed CPU evidence; no geometry, model, scorer, or inferred ratings.

The root's ratings below are a transcription of four actually viewed pages.
Other reviewers' original records remain unchanged.
"""
from __future__ import annotations
import csv
import hashlib
import html
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_local_shape_mix_v1'
DEST = OUT / 'closure'
STAGE = 'CLEAN_FINAL_POST'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if p.exists():
        if read(p) != v:
            raise RuntimeError('Existing closure evidence differs: ' + str(p))
        return
    p.write_text(text, encoding='utf-8')

def csvread(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def csvwrite(p, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k,v in r.items()})

def root_review():
    path = OUT / 'visual_review/reviewer_a8.json'
    if path.exists():
        return read(path)
    # Manual observations, explicitly transcribed after actual tools.view_image.
    ratings = {
        'RAW': ('REASONABLE', 'PLAUSIBLE', 'PRESENT', 'UNCERTAIN', 'NO_OBVIOUS'),
        'D0_ALL': ('INCOMPLETE', 'PLAUSIBLE', 'MISSING_OR_DAMAGED', 'YES', 'NO_OBVIOUS'),
        'ASSIST11': ('REASONABLE', 'PLAUSIBLE', 'PRESENT', 'YES', 'NO_OBVIOUS'),
        'LOCKED_RAW': ('REASONABLE', 'PLAUSIBLE', 'PRESENT', 'YES', 'NO_OBVIOUS'),
        'LOCAL_SHAPE': ('REASONABLE', 'PLAUSIBLE', 'PRESENT', 'YES', 'NO_OBVIOUS'),
    }
    notes = {
        'RAW': 'Two oblique views show four substantial legs under the slatted blue C seat. Front section retains a long underside and side sections have slat widths; fine gaps are ambiguous, not proof of a physical crack. No clear replacement seat.',
        'D0_ALL': 'The gray supports are largely reduced to needle-like lines or missing major faces in both oblique views. Blue C remains the seat. Sections show lost side/underside portions; some disconnected strokes alone do not certify cracks.',
        'ASSIST11': 'Four supports retain visible width; blue C remains the seat. Slat side/underside deficits persist compared with RAW. Restoring supports does not certify closed or intersection-free geometry.',
        'LOCKED_RAW': 'Four supports again have width. Macro appearance matches ASSIST11; side/underside deficits remain at the fixed sections. The section is a visual observation rather than a topology test.',
        'LOCAL_SHAPE': 'No clear macro improvement or new damage relative to LOCKED_RAW in either oblique view or the inherited front/side cuts. Legs remain substantial; the same slat side/underside deficits persist.',
    }
    displays, unique, events = [], {}, []
    now = datetime.now(timezone.utc).isoformat()
    for idx in (0, 1):
        for stage in ('BASE_POST', STAGE):
            page = f'R{idx:02d}_{stage}'
            r = read(OUT / 'gallery' / (page + '.json'))
            img = OUT / 'gallery' / (page + '.png')
            assert sha(img) == r['image_sha256']
            event = dict(event_id='root_actual_view_' + page, page_id=page,
                         tool='functions.exec -> tools.view_image',
                         evidence='Root viewed all four pages together in the active turn, after an earlier individual review; 2915x1485 images displayed at 2048x1043.',
                         transcription_utc=now, image_sha256=sha(img))
            events.append(event)
            for cell in r['cells']:
                key = r['request_id'] + '|' + cell['model'] + '|' + cell['stage']
                row = dict(unique_id=key, request_id=r['request_id'], parent_id=r['parent_id'],
                           noise_index=r['noise_index'], model=cell['model'], stage=cell['stage'],
                           display_stage=stage, page_id=page, image_path=str(img), image_sha256=sha(img),
                           mesh_path=cell['path'], mesh_sha256=r['source_meshes'][cell['path']],
                           reviewer='root', review_type='AI_PROXY_NONBLIND', actually_viewed=True,
                           views_and_sections_checked=True, view_event_id=event['event_id'],
                           transcription_utc=now, note=notes[cell['model']])
                row.update(dict(zip(('overall','C_use','support','visible_fragment_or_gap','layering_or_fold'), ratings[cell['model']])))
                displays.append(row)
                if key in unique:
                    unique[key].setdefault('additional_view_events', []).append(event)
                else:
                    unique[key] = dict(row)
    result = dict(status='COMPLETE_ACTUAL_VIEW', schema='local_shape_five_axis_review_v1',
                  reviewer='root', review_type='AI_PROXY_NONBLIND', metric_scores_read=True,
                  score_exposure='Root read quantitative results before visual review; not blind or independent human assessment.',
                  old_labels_copied=False, unique_source_style_not_required=True,
                  sections_do_not_certify_topology=True, display_count=20, unique_count=18,
                  actual_view_events=events, rows=list(unique.values()), display_records=displays)
    write(path, result)
    return result

def aggregate_review():
    root_review()
    rows, displays, books = [], [], {}
    for name in ('reviewer_a8.json', 'reviewer_27.json', 'reviewer_459.json'):
        p = OUT / 'visual_review' / name
        r = read(p); assert r['status'] in ('COMPLETE_ACTUAL_VIEW','ACTUAL_VIEW_COMPLETE')
        books[str(p)] = sha(p)
        assert len(r['rows']) == 18 and len(r['display_records']) == 20
        for row in r['rows']:
            assert row['actually_viewed'] and row['views_and_sections_checked']
            for k in ('overall','C_use','support','visible_fragment_or_gap','layering_or_fold'):
                assert row[k]
        rows.extend(r['rows']); displays.extend(r['display_records'])
    assert len(rows) == len({r['unique_id'] for r in rows}) == 54
    counts = []
    for stage in ('RAW','BASE_POST',STAGE):
        for label in ('RAW','D0_ALL','ASSIST11','LOCKED_RAW','LOCAL_SHAPE'):
            selected = [r for r in rows if r['stage'] == stage and r['model'] == label]
            if not selected: continue
            assert len(selected) == 6
            x = dict(stage=stage,label=label,requests=len(selected))
            for axis,val in [('overall','REASONABLE'),('C_use','PLAUSIBLE'),('support','PRESENT'),('visible_fragment_or_gap','YES'),('layering_or_fold','YES')]:
                x[axis+'_'+val] = sum(r[axis] == val for r in selected)
                x[axis+'_UNCERTAIN'] = sum(r[axis] == 'UNCERTAIN' for r in selected)
            x['three_axis_intersection'] = sum(r['overall']=='REASONABLE' and r['C_use']=='PLAUSIBLE' and r['support']=='PRESENT' for r in selected)
            counts.append(x)
    csvwrite(DEST / 'visual_unique54.csv', rows)
    csvwrite(DEST / 'visual_display60.csv', displays)
    csvwrite(DEST / 'visual_counts.csv', counts)
    write(DEST / 'visual_review_complete.json', dict(status='COMPLETE_ACTUAL_VIEW',
          source_records=books,unique_outputs=54,display_cells=60,actual_pages=12,
          reviewers=['root','n_range_audit','backend_runtime'],review_type='AI_PROXY_NONBLIND',
          limitations='Three AI reviewer identities, not independent humans; score/context exposure disclosed individually. Fixed sections do not prove cracks. Same-asset historical rating drift is not a model improvement.', counts=counts))
    return counts

def key_accounting():
    finals = [r for r in csvread(OUT/'analysis/per_output_full177.csv') if r['stage']==STAGE]
    assert len(finals)==30
    for r in finals:
        assert r['ConflictCov_C_status']=='PASS'
    macros = csvread(OUT/'analysis/parent_macro.csv')
    fields = ['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
              'actual_free_faces','deleted_free_faces','free_minimum_angle_lt5_fraction','free_q_gt10_fraction',
              'C_F_valid_edge_count','C_F_valid_edge_total_length','nonmanifold_edges',
              'hinge_diag_isolated_1face_edge_components','hinge_diag_legal_opposed_edge_C_reached_free_area_fraction']
    summary = {}
    for label in ('D0_ALL','ASSIST11','RAW_ALL','LOCKED_RAW','LOCAL_SHAPE'):
        summary[label] = {r['field']:float(r['parent_equal_mean']) for r in macros if r['group']=='ALL' and r['stage']==STAGE and r['label']==label and r['field'] in fields}
        assert len(summary[label])==len(fields)
    solvers=[]
    for i in range(6):
        p=OUT/'runs'/f'J{i:02d}_LOCAL_SHAPE'/'geometry.json'
        g=read(p); s=g['solver']
        solvers.append(dict(request_id=g['request_id'],solver=s))
    comparisons=csvread(OUT/'analysis/connectivity_pairwise.csv')
    connectivity=[]
    for r in comparisons:
        if r['reference']=='LOCKED_RAW' and r['label']=='LOCAL_SHAPE':
            connectivity.append(r)
    write(DEST/'key_accounting.json',dict(final_parent_equal=summary,solvers=solvers,
          local_vs_locked_connectivity=connectivity, all_final_I_status='PASS 30/30',
          scientific_dispatch=read(OUT/'dispatch/complete.json'),render_dispatch=read(OUT/'gallery/dispatch_complete.json')))
    return summary

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST/'visual_review_complete.json').exists():
        raise RuntimeError('Closure already transcribed; verify rather than overwrite')
    counts=aggregate_review(); metrics=key_accounting()
    print(json.dumps(dict(visual_counts=counts,final_macro=metrics),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
