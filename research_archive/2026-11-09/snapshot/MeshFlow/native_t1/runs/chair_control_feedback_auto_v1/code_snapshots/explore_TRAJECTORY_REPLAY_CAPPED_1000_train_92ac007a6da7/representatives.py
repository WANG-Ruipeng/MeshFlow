"""Deterministic DEV-only representative selection from completed saved scores.

No mesh/metric/model calls. Whole-cohort galleries remain authoritative.
"""
from __future__ import annotations
import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import statistics
from . import assets

VERSION='chair_control_feedback_representatives_v1'
TIE_ATOL=1e-15
FIELDS={'S':('SeamCov','SeamCov','SeamCov_upper'),
        'I':('ConflictCov_C','ConflictCov_C','ConflictCov_C_upper'),
        'E':('ExposedRetain','ExposedRetain_global_lower','ExposedRetain_global_upper'),
        'Dsym':('D_sym2048','D_sym2048','D_sym2048')}


def _finite(value):return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def _scores(out,label):
    if '/' in label or '\\' in label or label in ('.','..'):raise ValueError('Invalid label')
    p=Path(out)/'evaluation'/label/'DEV/per_output.json'
    rows=assets.read_json(p); expected=assets.read_json(Path(out)/'manifests/DEV.json')['rows']
    if len(rows)!=16 or len({r['request_id'] for r in rows})!=16:raise ValueError('Completed full DEV16 required')
    index={r['request_id']:r for r in rows}
    if set(index)!={r['id'] for r in expected}:raise ValueError('DEV request set differs')
    for r in expected:
        actual=index[r['id']]
        if actual.get('split')!='DEV' or actual.get('label')!=label:raise ValueError('Score pool/label differs')
        if actual['uid']!=r['uid'] or actual['candidate']!=r['candidate']:raise ValueError('Parent/noise pairing differs')
        if actual['status'] not in ('PASS','INVALID_OUTPUT','FAILED'):raise ValueError('Score is not a completed result')
    return index,p


def paired_rows(out,candidate_label,matched_label):
    """Read every registered DEV pair, retaining NA/invalid values as None."""
    candidate,cp=_scores(out,candidate_label);matched,mp=_scores(out,matched_label)
    rows=[]
    for rid in sorted(candidate):
        c,m=candidate[rid],matched[rid]
        row=dict(request_id=rid,uid=c['uid'],candidate=c['candidate'],candidate_label=candidate_label,
                 matched_label=matched_label,candidate_status=c['status'],matched_status=m['status'],
                 candidate_raw_sha256=c.get('raw_sha256'),matched_raw_sha256=m.get('raw_sha256'))
        for metric,(point,lower,upper) in FIELDS.items():
            values=[c.get(point),m.get(point),c.get(lower),c.get(upper),m.get(lower),m.get(upper)]
            row.update({metric+'_'+key:value if _finite(value) else None for key,value in zip(
                ('candidate','matched','candidate_lower','candidate_upper','matched_lower','matched_upper'),values)})
            eligible=c['status']=='PASS' and m['status']=='PASS' and all(_finite(v) for v in values)
            row[metric+'_numeric_pair']=eligible
            row[metric+'_delta']=values[0]-values[1] if eligible else None
            row[metric+'_delta_lower']=values[2]-values[5] if eligible else None
            row[metric+'_delta_upper']=values[3]-values[4] if eligible else None
        rows.append(row)
    return rows,dict(candidate=cp,matched=mp),dict(candidate=candidate,matched=matched)


def _choose(rows,field,direction,*,predicate=None,target=None):
    eligible=[r for r in rows if _finite(r.get(field)) and (predicate is None or predicate(r))]
    if not eligible:return dict(status='NA',reason='No completed eligible pair meets this event definition',request_ids=[])
    value=lambda r:abs(r[field]-target) if target is not None else r[field]
    optimum=(min if direction=='min' else max)(value(r) for r in eligible)
    tied=sorted((r for r in eligible if abs(value(r)-optimum)<=TIE_ATOL),key=lambda r:r['request_id'])
    return dict(status='SELECTED',ranking_field=field,direction=direction,target=target,ranking_value=optimum,
                request_ids=[r['request_id'] for r in tied],display_first_request_id=tied[0]['request_id'],
                eligible_pairs=len(eligible),all_ties_preserved=True)


def _review_records(visual_review,candidate_label,matched_label):
    if visual_review is None:return {},None
    if isinstance(visual_review,(str,Path)):
        p=assets.resolve(visual_review);document=assets.read_json(p)
        source=dict(path=str(p),sha256=assets.digest(p))
    else:
        document=visual_review;source=dict(inline=document)
    records=document.get('records',[]) if isinstance(document,dict) else document
    result={}
    for row in records:
        if row.get('candidate_label')!=candidate_label or row.get('matched_label')!=matched_label:continue
        if row.get('review_type')!='AI_VISUAL_REVIEW' or row.get('actually_viewed') is not True:continue
        if row.get('judgment')!='WORSE':continue
        paths=[assets.resolve(p) for p in row.get('image_paths',[])]
        if not paths or not all(p.is_file() for p in paths):continue
        result[row['request_id']]=dict(record=row,images={str(p):assets.digest(p) for p in paths})
    return result,source


def _gallery(out,labels,scores):
    matches=[]
    for p in sorted((Path(out)/'visuals').glob('*/manifest.json')):
        m=assets.read_json(p)
        if m.get('split')!='DEV' or m.get('status')!='COMPLETE' or not set(labels)<=set(m['labels']):continue
        if m.get('feedback_overlay_status')!='COMPLETE':continue
        available=m['identity']['availability'];files=m['source_files'];valid=True
        for label,index in scores.items():
            for rid,row in index.items():
                raw=available.get(rid+'|'+label)
                if not raw or files.get(raw)!=row.get('raw_sha256'):valid=False;break
        if valid:matches.append((p,m))
    if not matches:return None
    # For identical frozen RAW bytes all matching galleries are eligible.
    # Lexicographic path rule is stable and independent of visual appearance.
    p,m=matches[0]
    return dict(path=str(p),sha256=assets.digest(p),index=m['index'],contact_sheet=m['contact_sheet'],
                parent_pages=m['parent_pages'],selection='lexicographically first complete matching-source gallery path')


def select(out,candidate_label,matched_label,visual_review=None):
    """Select fixed representative cases; caller explicitly names the candidate.

    visual_review schema: {records:[{request_id,candidate_label,matched_label,
    review_type:'AI_VISUAL_REVIEW',actually_viewed:true,judgment:'WORSE',
    image_paths:[actual inspected PNG paths],notes:'...'}]}.
    """
    out=assets.resolve(out)
    if candidate_label==matched_label or candidate_label=='START' or 'FM_CONTROL' in candidate_label:
        raise ValueError('A baseline/matched FM control cannot be declared the final candidate')
    rows,paths,index=paired_rows(out,candidate_label,matched_label)
    start,sp=_scores(out,'START')
    values=[r['S_delta'] for r in rows if r['S_numeric_pair']]
    median=statistics.median(values) if values else None
    selections={
        'largest_S_improvement':_choose(rows,'S_delta','max',predicate=lambda r:r['S_delta']>0),
        'closest_to_median_S_change':_choose(rows,'S_delta','min',target=median) if median is not None else dict(status='NA',request_ids=[]),
        'largest_S_deterioration':_choose(rows,'S_delta','min',predicate=lambda r:r['S_delta']<0),
        'largest_I_increase':_choose(rows,'I_delta_upper','max',predicate=lambda r:r['I_delta_upper']>0),
        'largest_E_decrease':_choose(rows,'E_delta_lower','min',predicate=lambda r:r['E_delta_lower']<0),
    }
    reviews,review_source=_review_records(visual_review,candidate_label,matched_label)
    def improved(r):
        return (r['S_numeric_pair'] and r['S_delta_lower']>0) or (
            r['S_numeric_pair'] and r['S_delta_lower']>=-.005 and (
                (r['I_numeric_pair'] and r['I_delta_upper']<0) or (r['E_numeric_pair'] and r['E_delta_lower']>0)))
    counterexamples=[r for r in rows if r['request_id'] in reviews and improved(r)]
    selections['metric_improvement_but_reviewed_visual_deterioration']=dict(
        status='SELECTED' if counterexamples else 'NA',request_ids=sorted(r['request_id'] for r in counterexamples),
        reason=None if counterexamples else 'No applicable actual AI_VISUAL_REVIEW WORSE record with conservative metric improvement; no visual decline inferred from scores',
        display_first_request_id=min((r['request_id'] for r in counterexamples),default=None),
        all_reviewed_counterexamples_preserved=True)
    gallery=_gallery(out,['START',candidate_label,matched_label],
                     {'START':start,candidate_label:index['candidate'],matched_label:index['matched']})
    source_files={str(p):assets.digest(p) for p in list(paths.values())+[sp,out/'manifests/DEV.json',Path(__file__)]}
    rules=dict(tie_atol=TIE_ATOL,ties='retain all ties; lexicographic request_id is display first',
        S='Largest positive nominal delta; most negative nominal delta; nearest to the median over every numeric pair. Delta intervals accompany all selections.',
        I='Largest positive conservative delta upper = candidate upper - matched lower',
        E='Most negative conservative global delta lower = candidate global lower - matched global upper',
        visual='Explicit actual AI_VISUAL_REVIEW WORSE for these exact model labels and request; conservative S increase, or S lower delta>=-.005 and conservative I reduction or E increase',
        multiple_categories='One case may legitimately occur in multiple categories. No replacement to increase apparent diversity.',
        missing='NA is retained, never replaced by0/1; all16 registered paired rows remain delivered',
        selection_is_not='Not a success decision, checkpoint choice, generalization claim, or human usability study')
    identity=dict(version=VERSION,candidate_label=candidate_label,matched_label=matched_label,
                  source_files=source_files,visual_review=review_source,rules=rules,
                  gallery=None if gallery is None else dict(path=gallery['path'],sha256=gallery['sha256']))
    key=__import__('hashlib').sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:12]
    dest=out/'visuals/representatives'/(candidate_label+'__vs__'+matched_label+'__'+key);dest.mkdir(parents=True,exist_ok=True)
    target=dest/'selection_manifest.json'
    if target.exists():return assets.read_json(target)
    with (dest/'paired_all16.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    selected_ids=sorted({rid for s in selections.values() for rid in s.get('request_ids',[])})
    lookup={r['request_id']:r for r in rows}
    def link(p,label):
        relative=os.path.relpath(p,dest).replace(os.sep,'/')
        return '<a href="'+html.escape(relative,quote=True)+'">'+html.escape(label)+'</a>'
    body=['<!doctype html><meta charset="utf-8"><title>DEV representatives</title>',
          '<style>body{font:16px sans-serif;margin:25px;max-width:1450px;color:#223}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}small{color:#667}</style>',
          '<h1>DEV paired representatives</h1>',
          '<p>'+html.escape(candidate_label)+' versus '+html.escape(matched_label)+'. All16 pairs remain in paired_all16.csv. These representatives do not replace the full gallery or establish success.</p>',
          '<p>Human usability: NOT_RUN. Visual deterioration is only reported when backed by actual supplied AI_VISUAL_REVIEW records.</p>']
    if gallery:body.append('<p>'+link(gallery['index'],'Complete all16 gallery')+' | '+link(gallery['contact_sheet'],'Complete contact sheet')+'</p>')
    else:body.append('<p>Complete matching comparison gallery with all three feedback overlays is PENDING. Numerical selection is still retained.</p>')
    body.append('<table><tr><th>Fixed selection</th><th>Request(s)</th><th>Details</th></tr>')
    for name,s in selections.items():
        body.append('<tr><td>'+html.escape(name)+'</td><td>'+('<br>'.join(html.escape(rid) for rid in s.get('request_ids',[])) or 'NA')+'</td><td>'+html.escape(s.get('reason') or s.get('ranking_field','actual visual review'))+'</td></tr>')
    body.append('</table><h2>Selected rows, including tradeoffs</h2><table><tr><th>Request</th><th>Delta S</th><th>Delta I interval</th><th>Delta E interval</th><th>Delta Dsym</th><th>Original full pages</th></tr>')
    for rid in selected_ids:
        r=lookup[rid];links=[]
        if gallery:
            for page in gallery['parent_pages']:
                if Path(page['path']).name in (r['uid']+'.png',r['uid']+'_feedback.png'):
                    links.append(link(page['path'],Path(page['path']).stem))
        body.append('<tr><td>'+html.escape(rid)+'</td><td>'+str(r['S_delta'])+'</td><td>'+str([r['I_delta_lower'],r['I_delta_upper']])+'</td><td>'+str([r['E_delta_lower'],r['E_delta_upper']])+'</td><td>'+str(r['Dsym_delta'])+'</td><td>'+'<br>'.join(links)+'</td></tr>')
    body.append('</table><p>Positive S/E is favorable; negative I/Dsym is favorable. Fractions are not percentages. Interval endpoints include saved uncertainty. No mesh processing, rendering, score recalculation or extra sampling occurred.</p>')
    (dest/'index.html').write_text('\n'.join(body),encoding='utf-8')
    result=dict(status='COMPLETE' if gallery else 'SELECTION_COMPLETE_GALLERY_PENDING',identity=identity,
        candidate_label=candidate_label,matched_label=matched_label,pool='DEV',registered_pairs=16,
        numeric_pairs={m:sum(r[m+'_numeric_pair'] for r in rows) for m in FIELDS},median_S_delta=median,
        selections=selections,all16_rows=rows,reviewed_visual_counterexamples={rid:reviews[rid] for rid in selected_ids if rid in reviews},
        gallery=gallery,index=str(dest/'index.html'),paired_csv=str(dest/'paired_all16.csv'),
        paired_csv_sha256=assets.digest(dest/'paired_all16.csv'),model_calls=0,render_calls=0,score_calls=0,
        confirmed_holdout_read=False,raw_geometry_modified=False,human_usability='NOT_RUN')
    assets.freeze_json(target,result)
    return result


def interface_check(out=assets.OUT):
    """Use completed control500 only to validate schema; do not select it."""
    out=assets.resolve(out);rows,paths,_=paired_rows(out,'explore_FM_CONTROL_500','START')
    assert len(rows)==16 and len({r['uid'] for r in rows})==8
    for r in rows:
        for metric in FIELDS:
            if r[metric+'_numeric_pair']:
                assert r[metric+'_delta_lower'] <= r[metric+'_delta_upper']+TIE_ATOL
    r=dict(status='PASS',scope='Paired metric schema only; FM_CONTROL is not declared a candidate or selected as final',
        pairs=16,parents=8,source_files={str(p):assets.digest(p) for p in paths.values()},
        module_sha256=assets.digest(__file__),formal_selection_calls=0,model_calls=0,render_calls=0,score_calls=0)
    assets.freeze_json(out/'tests/representatives/results.json',r);return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(assets.OUT));p.add_argument('--candidate-label');p.add_argument('--matched-label')
    p.add_argument('--visual-review');p.add_argument('--interface-check',action='store_true');a=p.parse_args()
    if a.interface_check:r=interface_check(a.out)
    else:
        if not a.candidate_label or not a.matched_label:p.error('candidate and matched labels are required')
        r=select(a.out,a.candidate_label,a.matched_label,a.visual_review)
    print(json.dumps({k:r[k] for k in ('status','index','registered_pairs','pairs') if k in r}))
