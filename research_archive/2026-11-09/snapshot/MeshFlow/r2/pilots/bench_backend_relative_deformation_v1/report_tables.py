"""Readable subsets of frozen scores; no new metric definition or evaluation."""
from __future__ import annotations
from .common import *

def rows(path):
    with resolve(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def run():
    dest = OUT / 'report_tables'
    done = dest / 'COMPLETE.json'
    if done.exists():
        r = read(done); verify(r['files']); return r
    fields = ['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
        'actual_free_faces','free_area','free_minimum_angle_lt5_fraction',
        'hinge_diag_isolated_1face_edge_components','nonmanifold_edges','C_F_has_valid_edge',
        'C_F_valid_edge_total_length','opposite_winding_reached_free_area_fraction',
        'CF_illegal_pair_count','FF_illegal_pair_count']
    data = rows(OUT/'scores/parent_macro_summary.csv')
    keep = [('COVERAGE_FINAL','CLEAN_FINAL_POST'),('COVERAGE_FINAL','GEO_ONLY'),
            ('SHAPE_D0','CLEAN_FINAL_POST'),('RELDEF_D0','BASE_POST'),('RELDEF_D0','CLEAN_FINAL_POST')]
    selected = [r for r in data if (r['label'],r['stage']) in keep]
    csvwrite(dest/'key_metrics.csv', [{k:r[k] for k in ['label','stage','eval_group','rows']+fields} for r in selected])
    current = rows(OUT/'visual_review/per_cell.csv')
    old = {(r['request_id'],int(r['column'])):r for r in rows(PREV/'visual_review/per_cell.csv')}
    changes=[]
    axes=['overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold']
    for r in current:
        column=int(r['column'])
        if column not in (0,1,2,3): continue
        prior=old[(r['request_id'],4 if column==3 else column)]
        item={k:r[k] for k in ('request_id','parent_id','eval_group','column','column_name')}
        for axis in axes:
            item[axis+'_previous']=prior[axis]; item[axis+'_current']=r[axis]
        item['previous_strict_visual_joint']=prior['strict_visual_joint']
        item['current_strict_visual_joint']=r['strict_visual_joint']
        item['any_axis_changed']=any(prior[a]!=r[a] for a in axes)
        item['strict_joint_changed']=prior['strict_visual_joint']!=r['strict_visual_joint']
        changes.append(item)
    csvwrite(dest/'historical_visual_label_comparison.csv',changes)
    result=dict(status='COMPLETE',same_panel_reassessment=True,
        historical_same_mesh_strict_joint_changes=sum(r['strict_joint_changed'] for r in changes),
        historical_same_mesh_any_axis_changes=sum(r['any_axis_changed'] for r in changes),
        files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
    write(done,result);return result

if __name__=='__main__':print(run())
