"""Read-only result/review aggregation; no model, geometry, or synthetic ratings."""
from pathlib import Path
import csv, hashlib, json, math, statistics

ROOT = Path(__file__).resolve().parents[1]
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def csvout(p, rows):
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
def mean(xs):
    xs = [x for x in xs if finite(x)]
    return statistics.mean(xs) if xs else None

def main():
    dest = ROOT/'results'
    rows = read(dest/'metric_rows.json')
    assert len(rows)==144
    panel = read(ROOT/'data/eval_manifest.json')['requests']
    group = {q['request_id']: ('TRAIN' if q['report_group'] in ('FOCUS','NEW_WIDE_TRAIN') else q['report_group']) for q in panel}
    labels = ['EXPOSURE','PLAIN1000','LAYOUT1000','LAYOUT_NULL']
    stages = ['RAW','BASE_POST','CLEAN_FINAL_POST']
    fields = ['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction',
        'ConflictCov_C','free_minimum_angle_lt5_fraction','actual_free_faces','deleted_free_faces',
        'free_total_surface_area','nonmanifold_edges','hinge_diag_isolated_1face_edge_components',
        'C_F_valid_edge_count','C_F_valid_edge_total_length','opposite_winding_reached_free_area_fraction',
        'A_VMR__L_tau_0.005','PMR__L_tau_0.005']
    primary = []
    for label in labels:
        for stage in stages:
            for g in ['ALL','TRAIN','DEV','HOLDOUT']:
                rr = [r for r in rows if r['label']==label and r['stage']==stage and (g=='ALL' or group[r['request_id']]==g)]
                parents = sorted({r['parent_id'] for r in rr})
                out = dict(label=label,stage=stage,group=g,parents=len(parents),outputs=len(rr))
                for field in fields:
                    values = [mean([r[field] for r in rr if r['parent_id']==p]) for p in parents]
                    out[field] = mean(values); out[field+'__valid_parents'] = sum(finite(x) for x in values)
                primary.append(out)
    csvout(dest/'report_primary_metrics.csv', primary)
    paths=sorted((ROOT/'visual_review/completed').glob('review_part_*.json'))
    assert len(paths)==3, 'Wait for all three actual review files'
    review=[]
    for p in paths:
        data=read(p)
        for r in data['rows']:
            review.append(dict(r,reviewer=data['reviewer'],reviewer_kind=data['reviewer_kind']))
    assert len(review)==144 and all(r['actually_viewed'] and r['views_and_sections_checked'] for r in review)
    assert len({(r['label'],r['stage'],r['request_id']) for r in review})==144
    def joint(r):
        return r['overall_shape']=='REASONABLE' and r['C_use']=='PLAUSIBLE' and r['support_retention']=='PRESENT'
    visual=[]
    for label in labels:
        for stage in stages:
            for g in ['ALL','TRAIN','DEV','HOLDOUT']:
                rr=[r for r in review if r['label']==label and r['stage']==stage and (g=='ALL' or group[r['request_id']]==g)]
                out=dict(label=label,stage=stage,group=g,n=len(rr),joint=sum(joint(r) for r in rr))
                for axis, value in [('overall_shape','REASONABLE'),('C_use','PLAUSIBLE'),('support_retention','PRESENT'),('visible_fragment_or_gap','YES'),('layering_or_fold','YES')]:
                    out[axis+'__'+value]=sum(r[axis]==value for r in rr)
                    out[axis+'__UNCERTAIN']=sum(r[axis]=='UNCERTAIN' for r in rr)
                visual.append(out)
    csvout(dest/'report_visual_summary.csv', visual)
    by={(r['label'],r['stage'],r['request_id']):r for r in review}
    transitions=[]
    for ref,cand in [('EXPOSURE','PLAIN1000'),('EXPOSURE','LAYOUT1000'),('PLAIN1000','LAYOUT1000'),('LAYOUT_NULL','LAYOUT1000')]:
        for stage in stages:
            for q in panel:
                a,b=by[ref,stage,q['request_id']],by[cand,stage,q['request_id']]
                out=dict(reference=ref,candidate=cand,stage=stage,request_id=q['request_id'],parent_id=q['parent_id'],group=group[q['request_id']],reference_joint=joint(a),candidate_joint=joint(b))
                for axis in ['overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold']:
                    out[axis+'__reference']=a[axis];out[axis+'__candidate']=b[axis]
                transitions.append(out)
    csvout(dest/'report_visual_transitions.csv', transitions)
    # Preserve full notes and uncertainties, rather than only the joint count.
    write(dest/'actual_visual_review_rows.json', review)
    result=dict(status='COMPLETE',metric_rows=len(rows),review_rows=len(review),primary=primary,visual=visual,
        reviewer_kind='AI_PROXY',blind=False,independent_human_certification=False,
        definition='Two fixed noises averaged within parent then parent equal; visual counts are outputs. I is ConflictCov_C, not whole-mesh Ri.',
        calls=dict(model=0,scoring=0,new_ratings=0),inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    write(dest/'report_numbers.json',result)
    print(json.dumps({'status':'COMPLETE','visual_all':[r for r in visual if r['group']=='ALL']},ensure_ascii=False))

if __name__=='__main__':
    main()
