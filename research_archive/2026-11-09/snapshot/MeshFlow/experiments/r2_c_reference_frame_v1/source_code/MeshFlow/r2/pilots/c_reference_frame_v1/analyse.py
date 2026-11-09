"""Offline summaries and unblinding, after all actual visual reviews are saved."""
import collections,csv,json,math
from pathlib import Path
import numpy as np
from . import common as C
EV=C.OUT/'evaluation';G=EV/'gallery'
from .evaluate import LABELS, COMPARISONS
METRICS=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction','A_VMR__raw_tau_1e-6','PMR__raw_tau_1e-6','C_F_has_valid_edge','C_F_valid_edge_count','C_F_valid_opposite_winding_edge_count','opposite_winding_reached_free_area_fraction','actual_free_faces')
FIELDS=('C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer','whole_body_complete','support_missing','fragments_or_folded_layers','visible_join')
def rows(p):
    with C.resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def val(s):
    if s in ('True','False'):return float(s=='True')
    if s in ('',None):return None
    return float(s)
def mean(v):
    a=[x for x in v if x is not None and math.isfinite(x)]
    return sum(a)/len(a) if a else None

def run():
    r=C.read(EV/'registration.json');byid={q['request_id']:q for q in r['requests']}
    original=rows(G/'AI_PROXY_review_template.csv');expected={(x['request_id'],x['anonymous_label'],x['stage']) for x in original}
    files=[G/f'review_{name}.csv' for name in ('3f','73b','familiar_full','dev')]
    review=[x for p in files for x in rows(p)]
    if len(review)!=384 or {(x['request_id'],x['anonymous_label'],x['stage']) for x in review}!=expected:raise ValueError('Actual reviews incomplete/duplicate')
    for x in review:
        if x['review_status']!='REVIEWED' or any(x[k] not in ('YES','NO','UNCERTAIN') for k in FIELDS):raise ValueError('Unreviewed categorical field')
        if x['image_sha256']!=x['reviewed_image_sha256'] or C.sha(x['image_path'])!=x['image_sha256']:raise ValueError('Review image mismatch')
    # The key is read only after all 384 actual rows pass.
    saved_review_hashes={str(p):C.sha(p) for p in files}
    key=C.read(EV/'blinding/model_key.json')['assignments']
    for x in review:x['label']=key[x['condition_id']][x['anonymous_label']]
    C.csvwrite(G/'AI_PROXY_review_unblinded.csv',review)
    styles=[x for name in ('3f','73b','familiar_full','dev') for x in rows(G/f'style_{name}.csv')]
    if len(styles)!=32 or len({(x['condition_id'],x['anonymous_label']) for x in styles})!=32:raise ValueError('Style reviews incomplete')
    for x in styles:
        if x['review_status']!='REVIEWED' or x['reviewed_image_sha256']!=C.sha(x['image_path']):raise ValueError('Style actual review missing')
        x['label']=key[x['condition_id']][x['anonymous_label']]
    C.csvwrite(G/'AI_PROXY_noise_style_unblinded.csv',styles)
    C.write(G/'review_completion.json',dict(status='COMPLETE_ACTUAL_AI_PROXY',rows=384,style_rows=32,request_pages=48,contact_sheets=8,review_files=saved_review_hashes,unblind_key_sha256=C.sha(EV/'blinding/model_key.json'),notes='Names hidden until individual reviews saved; reviewers had prior familiarity with historical meshes. Not human evaluation and not guaranteed blind identification.'))
    numeric=rows(EV/'per_output.csv')
    if len(numeric)!=384:raise ValueError('Numerical output incomplete')
    reviewmap={(x['label'],x['request_id'],x['stage']):x for x in review}
    groups=collections.defaultdict(list)
    for x in numeric:groups[x['panel'],x['parent_id'],x['condition_id'],x['label'],x['stage']].append(x)
    per=[]
    for (panel,parent,cid,label,stage),rr in sorted(groups.items()):
        out=dict(panel=panel,parent_id=parent,condition_id=cid,label=label,stage=stage,n=len(rr))
        for k in METRICS:out[k]=mean([val(x[k]) for x in rr])
        out['valid_edge_outputs']=sum(val(x['C_F_has_valid_edge'])==1 for x in rr)
        out['UNKNOWN_topology_outputs']=sum(x.get('topology_status')!='PASS' for x in rr)
        for field in FIELDS:
            counts=collections.Counter(reviewmap[label,x['request_id'],stage][field] for x in rr)
            for v in ('YES','NO','UNCERTAIN'):out[field+'__'+v]=counts[v]
        per.append(out)
    C.csvwrite(C.OUT/'results/per_condition.csv',per)
    panels=[]
    for panel in ('TRAIN_FAMILIAR','DEV_HELDOUT'):
        for label in LABELS:
            for stage in ('RAW','POST'):
                rr=[x for x in per if (x['panel'],x['label'],x['stage'])==(panel,label,stage)]
                out=dict(panel=panel,label=label,stage=stage,parents=len(rr),n=sum(x['n'] for x in rr))
                for k in METRICS:out[k]=mean([x[k] for x in rr])
                for k in rr[0]:
                    if '__' in k and k not in METRICS or k in ('valid_edge_outputs','UNKNOWN_topology_outputs'):out[k]=sum(x[k] for x in rr)
                panels.append(out)
    C.csvwrite(C.OUT/'results/per_panel.csv',panels)
    lookup={(x['label'],x['request_id'],x['stage']):x for x in numeric};paired=[]
    for before,after in COMPARISONS:
        for q in r['requests']:
            for stage in ('RAW','POST'):
                a=lookup[before,q['request_id'],stage];b=lookup[after,q['request_id'],stage]
                av=reviewmap[before,q['request_id'],stage]['C_adoption'];bv=reviewmap[after,q['request_id'],stage]['C_adoption']
                status='UNCERTAIN' if 'UNCERTAIN' in (av,bv) else ('GAIN' if (av,bv)==('NO','YES') else ('LOSS' if (av,bv)==('YES','NO') else 'SAME'))
                ac,bc=val(a['C_F_has_valid_edge']),val(b['C_F_has_valid_edge'])
                row=dict(panel=q['panel'],parent_id=q['parent_id'],condition_id=q['condition_id'],request_id=q['request_id'],candidate=q['candidate'],stage=stage,comparison=after+'-minus-'+before,C_adoption_before=av,C_adoption_after=bv,C_adoption_pair=status,local_edge_pair='GAIN' if bc>ac else ('LOSS' if bc<ac else 'SAME'))
                for metric in METRICS:
                    va,vb=val(a[metric]),val(b[metric]);row[metric+'__delta']=None if va is None or vb is None else vb-va
                paired.append(row)
    C.csvwrite(C.OUT/'results/paired_decisions.csv',paired)
    psummary=[]
    for comparison in sorted({x['comparison'] for x in paired}):
        for panel in ('TRAIN_FAMILIAR','DEV_HELDOUT'):
            for stage in ('RAW','POST'):
                rr=[x for x in paired if (x['comparison'],x['panel'],x['stage'])==(comparison,panel,stage)]
                out=dict(comparison=comparison,panel=panel,stage=stage,n=len(rr))
                for metric in ('C_adoption_pair','local_edge_pair'):
                    counter=collections.Counter(x[metric] for x in rr)
                    for v in ('GAIN','LOSS','SAME','UNCERTAIN'):out[metric+'__'+v]=counter[v]
                psummary.append(out)
    C.csvwrite(C.OUT/'results/paired_panel_decisions.csv',psummary)
    old=C.ROOT/'experiments/r2_seat_quick_learnability_v1/evaluation/per_output.csv'
    C.csvwrite(C.OUT/'results/OLD_NOISE_REGRESSION_REFERENCE_ONLY.csv',rows(old))
    C.write(C.OUT/'results/OLD_REFERENCE.json',dict(path=str(old),sha256=C.sha(old),new_model_calls=0,not_pooled_with_new_noise=True))
    summary=dict(status='COMPLETE',per_panel=panels,per_condition=per,paired_decisions=psummary,actual_review_rows=384,style_reviews=styles,old_regression='separate frozen reference')
    # Preserve the old AI_PROXY judgments and expose re-review disagreement.
    prior=rows(C.ROOT/'experiments/r2_seat_noise_transfer_v1/evaluation/gallery/AI_PROXY_review_unblinded.csv')
    priormap={(x['label'],x['request_id'],x['stage']):x for x in prior}
    drift=[]
    for x in review:
        key0=(x['label'],x['request_id'],x['stage'])
        if key0 not in priormap:continue
        oldrow=priormap[key0]
        for field in FIELDS:
            drift.append(dict(label=x['label'],request_id=x['request_id'],panel=x['panel'],parent_id=x['parent_id'],stage=x['stage'],field=field,previous_value=oldrow[field],current_value=x[field],changed=oldrow[field]!=x[field]))
    C.csvwrite(C.OUT/'results/baseline_review_agreement.csv',drift)
    sensitivity=[]
    for before in LABELS[:2]:
        for panel in ('TRAIN_FAMILIAR','DEV_HELDOUT'):
            for stage in ('RAW','POST'):
                cc=collections.Counter()
                for q in r['requests']:
                    if q['panel']!=panel:continue
                    av=priormap[before,q['request_id'],stage]['C_adoption'];bv=reviewmap['CREF_C200',q['request_id'],stage]['C_adoption']
                    status='UNCERTAIN' if 'UNCERTAIN' in (av,bv) else ('GAIN' if (av,bv)==('NO','YES') else ('LOSS' if (av,bv)==('YES','NO') else 'SAME'))
                    cc[status]+=1
                sensitivity.append(dict(comparison='CREF_C200-minus-historical-review-'+before,panel=panel,stage=stage,**dict(cc)))
    C.csvwrite(C.OUT/'results/historical_review_sensitivity.csv',sensitivity)
    summary['historical_review_sensitivity']=sensitivity
    C.write(C.OUT/'summary.json',summary)
    print(json.dumps(dict(status='COMPLETE',panels=panels,paired=psummary),ensure_ascii=False),flush=True)
if __name__=='__main__':run()