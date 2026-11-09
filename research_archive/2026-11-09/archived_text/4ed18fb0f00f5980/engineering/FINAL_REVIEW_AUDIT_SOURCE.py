"""Independent audit of sealed actual reviews and three-valued summaries.

No score files, model checkpoints, geometric evaluator, or image labels are
recomputed. Images are hashed solely to bind the recorded reviews to sources.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from r2.pilots.window_policy_rl_v1.bootstrap import bind,OUT
D,E=bind();EV=OUT/'evaluation'
LABELS=('FULL_DET','FULL_STOCH','RL_STOCH')
FIELDS=('RAW_adoption','POST_adoption','POST_body_complete')
EVENTS=tuple(x+'_known' for x in FIELDS)+('joint_RAW_adoption_POST_legal_CF','joint_both_adoptions_body_and_CF')
CONTRASTS=(('RL_STOCH','FULL_STOCH','PRIMARY'),('FULL_STOCH','FULL_DET','STOCHASTIC_DECODING_REFERENCE'),('RL_STOCH','FULL_DET','DETERMINISTIC_REFERENCE'))
SOURCES={};CHECKS=Counter();ERRORS=[]

def check(ok,name,detail=None):
    CHECKS[name]+=1
    if not ok:ERRORS.append(dict(check=name,detail=detail))

def blob(path):
    p=D.resolve(path);data=p.read_bytes();SOURCES[str(p)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)};return data
def read(path):return json.loads(blob(path).decode('utf-8-sig'))
def sha(path):
    p=D.resolve(path)
    if str(p) not in SOURCES:blob(p)
    return SOURCES[str(p)]['sha256']
def table(path):return list(csv.DictReader(io.StringIO(blob(path).decode('utf-8-sig'))))
def event_value(value):return {'YES':True,'NO':False,'UNKNOWN':None}[value]
def conjunction(values):
    if False in values:return False
    return None if None in values else True
def stats(values):
    values=list(values);n=len(values);yes=sum(v is True for v in values);no=sum(v is False for v in values);unknown=n-yes-no
    return dict(YES=yes,NO=no,UNKNOWN=unknown,denominator=n,known_denominator=yes+no,known_rate=yes/(yes+no) if yes+no else None,
        full_panel_lower=yes/n,full_panel_upper=(yes+unknown)/n)
def pair_stats(values):
    known=[int(a)-int(b) for a,b in values if a is not None and b is not None]
    low=sum((0 if a is None else int(a))-(1 if b is None else int(b)) for a,b in values)
    high=sum((1 if a is None else int(a))-(0 if b is None else int(b)) for a,b in values)
    return dict(common_known_pairs=len(known),unknown_pairs=len(values)-len(known),gained=known.count(1),lost=known.count(-1),
        common_known_mean_delta=sum(known)/len(known) if known else None,full_panel_mean_delta_lower=low/len(values),full_panel_mean_delta_upper=high/len(values))
def equal(a,b):
    if a is None or b is None:return a is None and b is None
    return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)

def run():
    check('torch' not in sys.modules,'MODEL_FREE')
    merged_path=EV/'blind_gallery/reviews_actual.json';merged=read(merged_path)
    check(sha(merged_path)=='3bdca7408d95d87b015c4ebe0a136c03601a8c3d4657a08466c4c3613ae1f45c','SEALED_MERGED_HASH')
    check(merged['status']=='COMPLETE' and merged['blinded'] is True and merged['reviewer_kind']=='AI_PROXY' and len(merged['rows'])==132,'MERGED_COMPLETE132')
    gallery=read(EV/'blind_gallery/manifest.json');mapping=read(EV/'private/label_mapping.json');manifest=read(OUT/'inputs/manifest.json')
    qmap={q['request_id']:q for q in manifest['evaluation_requests']};images={r['request_id']:r for r in gallery['images']};maps={r['request_id']:r['aliases'] for r in mapping['rows']}
    check(len(qmap)==len(images)==len(maps)==44 and set(qmap)==set(images)==set(maps),'REGISTERED44_GALLERY_MAP')
    check(mapping['status']=='FROZEN' and mapping['manifest_sha256']==sha(OUT/'inputs/manifest.json'),'MAPPING_FROZEN_BINDING')
    check(gallery['status']=='COMPLETE' and gallery['identity']['model_blinded'] is True,'MODEL_BLIND_GALLERY')
    for rid,im in images.items():
        check(sha(im['image_path'])==im['image_sha256'],'44_ORIGINAL_IMAGE_HASH',rid)
        check(set(maps[rid])==set('ABC') and set(maps[rid].values())==set(LABELS),'ALIAS_BIJECTION',rid)
    originals={};assigned_all=set();reviewers=[]
    for i,source in enumerate(merged['sources'],1):
        part=read(source['path']);check(sha(source['path'])==source['sha256'],'ORIGINAL_REVIEW_SOURCE_HASH',source['path'])
        assignment=read(OUT/'review'/f'final_assignment_reviewer_{i}.json');assigned=set(assignment['request_ids'])
        expected={(rid,alias) for rid in assigned for alias in ('A','B','C')};seen=set()
        check(not (assigned_all & assigned),'ASSIGNMENTS_DISJOINT',i);assigned_all|=assigned
        check(part['status']=='COMPLETE' and part['blinded'] is True and part['reviewer_kind']=='AI_PROXY','ORIGINAL_REVIEW_COMPLETE',i)
        check(len(part['rows'])==len(expected)==source['rows'],'ORIGINAL_ASSIGNED_OUTPUT_COUNT',i)
        hashes={x['path']:x['sha256'] for x in part.get('source_inputs',[])};hashes.update(part.get('source_hashes',{}))
        for path,expected_hash in hashes.items():check(sha(path)==expected_hash,'REVIEWER_SOURCE_INPUT_HASH',path)
        for row in part['rows']:
            key=row['request_id'],row['label'];check(key not in seen and key not in originals,'ORIGINAL_NO_DUPLICATE',key)
            seen.add(key);originals[key]=row
            check(row['reviewer'] in ({part['reviewer'],part['reviewer']+' ('+part['reviewer_assignment']+')'} if part.get('reviewer_assignment') else {part['reviewer']}) and row['reviewer_kind']=='AI_PROXY' and row['blinded'] is True and row['review_status']=='REVIEWED','ACTUAL_REVIEW_IDENTITY',key)
            check(all(row[k] in ('YES','NO','UNKNOWN') for k in FIELDS),'THREE_VALUE_LABELS',key)
            q=qmap[row['request_id']];im=images[row['request_id']]
            check(row['image_sha256']==im['image_sha256'] and D.resolve(row['image_path'])==D.resolve(im['image_path']),'REVIEW_IMAGE_BINDING',key)
            check(row['condition_id']==q['condition_id'] and row['split']==q['split'] and row['parent_id']==q['parent_id'],'REVIEW_INPUT_METADATA',key)
        check(seen==expected,'ASSIGNMENT_ALL_ALIASES',i)
        actual_images=part.get('images',part.get('images_actually_viewed',[]))
        check({x['request_id'] for x in actual_images}==assigned,'DECLARED_ACTUAL_IMAGE_COVERAGE',i)
        for im in actual_images:check(im['image_sha256']==images[im['request_id']]['image_sha256'],'DECLARED_IMAGE_HASH',im['request_id'])
        reviewers.append(dict(reviewer=part['reviewer'],assigned_requests=len(assigned),review_rows=len(part['rows']),source=source,
            evidence='Original review declarations and image hashes; no new image judgment performed'))
    check(assigned_all==set(qmap),'ASSIGNMENTS_COVER44')
    merged_keys=[]
    for row in merged['rows']:
        key=row['request_id'],row['label'];merged_keys.append(key)
        check(row==originals.get(key),'MERGED_ROW_EXACT_ORIGINAL',key)
    check(len(set(merged_keys))==132 and set(merged_keys)==set(originals),'MERGED_EXACT132_UNION')
    numerical=read(OUT/'engineering/FINAL_QUANTITATIVE_AUDIT.json')
    check(numerical['status']=='PASS' and sha(OUT/'engineering/FINAL_QUANTITATIVE_AUDIT.json')=='b2178c60fa73af497463a8a28aa00d2ce163d3b5caf5639db2a892fa88ca4da0','PRIOR_NUMERICAL_AUDIT_REUSED')
    legal={(r['label'],r['request_id']):bool(r['has_legal_CF']) for r in numerical['per_output_stage'] if r['stage']=='POST'}
    outputs=[]
    for row in merged['rows']:
        rid=row['request_id'];label=maps[rid][row['label']];q=qmap[rid]
        r=dict(label=label,request_id=rid,condition_id=q['condition_id'],split=q['split'],reviewer=row['reviewer'],POST_legal_CF=legal[label,rid])
        r.update({f+'_known':event_value(row[f]) for f in FIELDS})
        r['joint_RAW_adoption_POST_legal_CF']=conjunction([r['RAW_adoption_known'],r['POST_legal_CF']])
        r['joint_both_adoptions_body_and_CF']=conjunction([r[f+'_known'] for f in FIELDS]+[r['POST_legal_CF']])
        outputs.append(r)
    summaries=[];pairs=[];index={(r['label'],r['request_id']):r for r in outputs}
    for label in LABELS:
        own=[r for r in outputs if r['label']==label]
        groups=[('SPLIT',s,own if s=='ALL' else [r for r in own if r['split']==s]) for s in ('ALL','TRAIN','DEV')]
        groups += [('CONDITION',c,[r for r in own if r['condition_id']==c]) for c in sorted({r['condition_id'] for r in own})]
        for scope,name,rows in groups:summaries.append(dict(label=label,scope=scope,group=name,outputs=len(rows),**{event:stats(r[event] for r in rows) for event in EVENTS}))
    for candidate,baseline,role in CONTRASTS:
        groups=[('SPLIT',s,list(qmap) if s=='ALL' else [rid for rid,q in qmap.items() if q['split']==s]) for s in ('ALL','TRAIN','DEV')]
        groups += [('CONDITION',c,[rid for rid,q in qmap.items() if q['condition_id']==c]) for c in sorted({q['condition_id'] for q in qmap.values()})]
        for scope,name,ids in groups:pairs.append(dict(candidate_label=candidate,baseline_label=baseline,comparison_role=role,scope=scope,group=name,paired_requests=len(ids),
            **{event:pair_stats([(index[candidate,rid][event],index[baseline,rid][event]) for rid in ids]) for event in EVENTS}))
    summary=read(EV/'analysis/summary.json')
    check(summary['status']=='COMPLETE' and summary['review_rows']==132 and summary['actual_reviews_sha256']==sha(merged_path),'MAIN_FINAL_SUMMARY_REVIEW_BINDING')
    for ours,main,identity in ((summaries,summary['review_summaries'],('label','scope','group')),
            (pairs,summary['paired_review_summaries'],('candidate_label','baseline_label','scope','group'))):
        lookup={tuple(r[k] for k in identity):r for r in main};check(len(lookup)==len(ours),'SUMMARY_GROUP_COVERAGE',identity)
        for r in ours:
            key=tuple(r[k] for k in identity);other=lookup[key]
            for event in EVENTS:
                for field,value in r[event].items():check(equal(value,other[event].get(field)),'SUMMARY_EVENT_AND_BOUNDS_EQUAL',dict(identity=key,event=event,field=field,ours=value,main=other[event].get(field)))
    mainrows=table(EV/'analysis/per_output_review_metrics.csv');rowmap={(r['label'],r['request_id']):r for r in mainrows}
    check(len(rowmap)==len(mainrows)==132,'MAIN_REVIEW_OUTPUT132')
    for r in outputs:
        main=rowmap[r['label'],r['request_id']]
        for event in EVENTS:
            converted={'True':True,'False':False,'':None}[main[event]]
            check(r[event] is converted,'MAIN_OUTPUT_EVENT_EQUAL',(r['label'],r['request_id'],event))
    sha(__file__);check('torch' not in sys.modules,'MODEL_FREE_AT_END')
    result=dict(version='window_policy_rl_final_review_independent_audit_v1',status='PASS' if not ERRORS else 'FAIL',checks=dict(CHECKS),violations=ERRORS,
        audit_adjustment={'prior_path':str(OUT/'engineering/FINAL_REVIEW_AUDIT_INITIAL_REVIEWER_STRING_MISMATCH.json'),'prior_sha256':sha(OUT/'engineering/FINAL_REVIEW_AUDIT_INITIAL_REVIEWER_STRING_MISMATCH.json'),'reason':'Initial audit required literal reviewer string equality. Reviewer2 original rows append the exact declared assignment in parentheses. Corrected identity check allows only the base identity or its exact declared assignment suffix; no original data was changed.'},original_reviewers=reviewers,original_images_hashed=44,merged_rows=132,review_summaries=summaries,paired_review_summaries=pairs,
        per_output_events=outputs,source_files=SOURCES,model_calls=0,score_files_read=0,geometry_rescores=0,labels_changed=0,
        limitations=['Arithmetic, coverage, source identity and merge-preservation audit only; no independent re-rating of semantic labels.',
            'One assigned AI_PROXY reviewer per output, 24/20 pages; no double-review agreement or human certification claim.',
            'UNKNOWN is retained. A known FALSE conjunct makes a joint event FALSE even if adoption/body is unknown.',
            'Paired full-panel bounds allow unknown labels separately; they are logical bounds, not confidence intervals.'])
    D.write(OUT/'engineering/FINAL_REVIEW_AUDIT.json',result)
    print(json.dumps(dict(status=result['status'],violations=ERRORS,checks=sum(CHECKS.values()),audit_sha256=D.sha(OUT/'engineering/FINAL_REVIEW_AUDIT.json'),
        splits=[r for r in summaries if r['scope']=='SPLIT'],primary_pairs=[r for r in pairs if r['scope']=='SPLIT' and r['comparison_role']=='PRIMARY']),sort_keys=True))

if __name__=='__main__':run()
