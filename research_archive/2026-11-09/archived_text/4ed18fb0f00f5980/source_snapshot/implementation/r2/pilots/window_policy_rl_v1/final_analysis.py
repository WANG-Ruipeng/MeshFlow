"""Frozen 44-request final numerical comparison, with optional actual blind review.

Primary contrast is RL_STOCH versus FULL_STOCH on identical initial/transition
noise. FULL_DET remains a separate deterministic quality/use reference. No model
or geometric metric is executed, and missing review never becomes a quality claim.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sys
import uuid
from .bootstrap import bind,ROOT,OUT
D,E=bind()
sys.path.insert(0,str(ROOT/'r2/src'))
from . import analyze as A
from r2.pilots.terminal_reward_nft_v1 import final_analysis as F

VERSION='window_policy_rl_final_analysis_v1'
EV=OUT/'evaluation'
DEST=EV/'analysis'
LABELS=('FULL_DET','FULL_STOCH','RL_STOCH')
CONTRASTS=(('RL_STOCH','FULL_STOCH','PRIMARY'),
           ('FULL_STOCH','FULL_DET','STOCHASTIC_DECODING_REFERENCE'),
           ('RL_STOCH','FULL_DET','DETERMINISTIC_REFERENCE'))
SPLITS=('ALL','TRAIN','DEV')
STAGES=('RAW','POST')


def readiness(value):
    value=dict(version=VERSION,model_calls=0,geometry_rescores=0,**value)
    snap=DEST/'readiness'/(A.digest(value)[:20]+'.json')
    D.write(snap,value)
    pointer=D.checked(DEST/'readiness.json');pointer.parent.mkdir(parents=True,exist_ok=True)
    temp=pointer.with_name(pointer.name+'.tmp-'+uuid.uuid4().hex)
    payload=dict(value,snapshot_path=str(snap),snapshot_sha256=D.sha(snap))
    with temp.open('x',encoding='utf8') as f:
        json.dump(payload,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    os.replace(temp,pointer)
    return payload


def contract():
    manifest=D.read(OUT/'inputs/manifest.json');registration=D.read(OUT/'registration.json')
    if D.sha(OUT/'inputs/manifest.json')!=registration['manifest_sha256']:
        raise ValueError('Final manifest/registration chain differs')
    qs=manifest['evaluation_requests']
    if (len(qs)!=44 or len({q['request_id'] for q in qs})!=44
            or Counter(q['split'] for q in qs)!=Counter(TRAIN=32,DEV=12)
            or sorted(Counter(q['condition_id'] for q in qs).values())!=[4]*11):
        raise ValueError('Expected 8TRAIN + 3DEV C, four frozen noises per C')
    if registration['final_evaluation']['models']!=list(LABELS) or registration['final_evaluation']['RL_DET'] is not False:
        raise ValueError('Final model panel differs; RL_DET is not authorized')
    return manifest,registration


def required(manifest):
    paths=[OUT/'training/final.json']
    for label in LABELS:
        for q in manifest['evaluation_requests']:
            base=EV/label;rid=q['request_id']
            paths.extend(base/k/rid/'receipt.json' for k in ('raw','post','proposal'))
            paths.extend(base/'scores'/rid/s/'score.json' for s in STAGES)
    return paths


def collect(manifest,gate,protected):
    from .sample_worker import verify_window_receipt
    final=D.read(OUT/'training/final.json')
    if final['status']!='COMPLETE' or final['batches']!=64 or final['rollouts']!=512:
        raise ValueError('Final endpoint must follow all64 registered opportunities, not best checkpoint')
    rows=[];originals=[];seen={}
    def protect(path,expected=None):
        path=str(D.resolve(path))
        if path not in seen:seen[path]=D.sha(path)
        if expected is not None and seen[path]!=expected:raise ValueError('Changed final input: '+path)
        protected[path]=seen[path]
    protect(OUT/'training/final.json')
    for label in LABELS:
        sigma=0. if label=='FULL_DET' else float(gate['selected_sigma'])
        mode='current' if label=='RL_STOCH' else 'reference'
        for q in manifest['evaluation_requests']:
            rid=q['request_id'];base=EV/label
            rawp,postp,propp=(base/k/rid/'receipt.json' for k in ('raw','post','proposal'))
            raw,post,proposal=map(D.read,(rawp,postp,propp))
            for p in (rawp,postp,propp):protect(p)
            if (raw['status']!='COMPLETE' or raw['label']!=label or raw['request_id']!=rid
                    or raw['sigma']!=sigma or raw['model_mode']!=mode
                    or raw['registration_sha256']!=D.sha(OUT/'registration.json')
                    or raw['model_identity']['base_file_sha256']!=manifest['source_binding']['sha256']):
                raise ValueError('Final RAW policy/stream identity differs')
            if label=='RL_STOCH' and raw['model_identity'].get('adapter_checkpoint_sha256')!=final['checkpoint_sha256']:
                raise ValueError('RL final RAW does not use predeclared last checkpoint')
            verify_window_receipt(q,raw)
            if (post['status']!='COMPLETE' or post['label']!=label or post['request_id']!=rid
                    or post['input_sha256']!=q['input_sha256']):
                raise ValueError('Final POST identity differs')
            if proposal['status']!='PASS' or proposal['model_forwards']!=1:
                raise ValueError('Final POST must use exactly one original D0')
            for p,h in post['files'].items():protect(p,h)
            for key in ('raw','ply','result','window_buffer'):protect(raw[key+'_path'],raw[key+'_sha256'])
            for stage in STAGES:
                sp=base/'scores'/rid/stage/'score.json';score=D.read(sp);protect(sp)
                if (score.get('original_definition_unchanged') is not True
                        or score['identity']['registration_sha256']!=D.sha(OUT/'registration.json')):
                    raise ValueError('Final original scorer definition/registration differs')
                for source in ('definition_hashes','input_hashes'):
                    for p,h in score['identity'][source].items():protect(p,h)
                for p,h in score['files'].items():protect(p,h)
                metrics=F.extract_score(score,q,label,stage)
                if metrics['actual_free_faces']!=q['N']-q['K']-metrics['deleted_free_faces']:
                    raise ValueError('Final actual free denominator differs')
                row={k:q[k] for k in ('request_id','condition_id','parent_id','candidate','split','N','K','noise_seed')}
                row.update(label=label,stage=stage,sigma=sigma,row_status='PASS',**metrics,
                    input_sha256=q['input_sha256'],Gaussian_sha256=q['Gaussian_sha256'],
                    transition_Gaussian_sha256=q['transition_Gaussian_sha256'],
                    score_path=str(sp),score_sha256=D.sha(sp),raw_receipt_path=str(rawp),post_receipt_path=str(postp))
                rows.append(row)
                originals.append(dict(label=label,request_id=rid,stage=stage,score_path=str(sp),score_sha256=D.sha(sp),
                    metrics=score['metrics'],topology_summary={k:v for k,v in score['topology'].items()
                        if v is None or isinstance(v,(str,int,float,bool))}))
    return rows,originals


def summary_group(rows,**identity):
    result=dict(identity,outputs=len(rows),legal_CF_outputs=sum(r['has_legal_CF'] for r in rows),
        I_status_counts=dict(Counter(r['I_status'] for r in rows)),
        topology_status_counts=dict(Counter(r['topology_status'] for r in rows)),
        source_C_connectivity_status_counts=dict(Counter(r['source_C_connectivity_status'] for r in rows)))
    result['legal_CF_output_rate']=result['legal_CF_outputs']/len(rows)
    for key in A.NUMERIC:
        for name,value in A.numeric_summary(rows,key).items():result[key+'_'+name]=value
    return result


def numerical_tables(rows):
    models=[];conditions=[];pairs=[];paired_summaries=[]
    for label in LABELS:
        for stage in STAGES:
            subset=[r for r in rows if r['label']==label and r['stage']==stage]
            for split in SPLITS:
                group=subset if split=='ALL' else [r for r in subset if r['split']==split]
                models.append(summary_group(group,label=label,stage=stage,split=split))
            for cid in sorted({r['condition_id'] for r in subset}):
                group=[r for r in subset if r['condition_id']==cid]
                conditions.append(summary_group(group,label=label,stage=stage,split=group[0]['split'],condition_id=cid,parent_id=group[0]['parent_id']))
    indexed={(r['label'],r['request_id'],r['stage']):r for r in rows}
    for candidate,baseline,role in CONTRASTS:
        for (label,rid,stage),b in sorted(indexed.items()):
            if label!=baseline:continue
            a=indexed[(candidate,rid,stage)]
            if any(a[k]!=b[k] for k in ('input_sha256','Gaussian_sha256','transition_Gaussian_sha256','condition_id','split')):
                raise ValueError('Final comparison does not share frozen request/noise arrays')
            row=dict(candidate_label=candidate,baseline_label=baseline,comparison_role=role,request_id=rid,stage=stage,
                condition_id=b['condition_id'],parent_id=b['parent_id'],split=b['split'],candidate=b['candidate'],
                baseline_has_legal_CF=b['has_legal_CF'],candidate_has_legal_CF=a['has_legal_CF'],
                legal_CF_gained=int(a['has_legal_CF']==1 and b['has_legal_CF']==0),
                legal_CF_lost=int(a['has_legal_CF']==0 and b['has_legal_CF']==1))
            for key in A.NUMERIC:
                av,bv=a[key],b[key];delta=None if av is None or bv is None else av-bv
                row['baseline_'+key],row['candidate_'+key],row[key+'_delta']=bv,av,delta
                if key in A.PP_FIELDS:row[key+'_delta_pp']=None if delta is None else 100*delta
            pairs.append(row)
        for stage in STAGES:
            group=[r for r in pairs if r['candidate_label']==candidate and r['baseline_label']==baseline and r['stage']==stage]
            scopes=[('SPLIT',s,group if s=='ALL' else [r for r in group if r['split']==s]) for s in SPLITS]
            scopes += [('CONDITION',cid,[r for r in group if r['condition_id']==cid]) for cid in sorted({r['condition_id'] for r in group})]
            for scope,name,sub in scopes:
                output=dict(candidate_label=candidate,baseline_label=baseline,comparison_role=role,stage=stage,
                    scope=scope,group=name,paired_requests=len(sub),legal_CF_gained=sum(r['legal_CF_gained'] for r in sub),
                    legal_CF_lost=sum(r['legal_CF_lost'] for r in sub))
                for key in A.NUMERIC:
                    for stat,value in A.numeric_summary(sub,key+'_delta').items():output[key+'_delta_'+stat]=value
                    if key in A.PP_FIELDS:
                        v=output[key+'_delta_mean'];output[key+'_mean_delta_pp']=None if v is None else 100*v
                paired_summaries.append(output)
    return models,conditions,pairs,paired_summaries


def actual_reviews(manifest,path,protected):
    mp=EV/'private/label_mapping.json';gp=EV/'blind_gallery/manifest.json'
    mapping,gallery=D.read(mp),D.read(gp)
    if mapping['status']!='FROZEN' or mapping['manifest_sha256']!=D.sha(OUT/'inputs/manifest.json'):
        raise ValueError('Final anonymous mapping differs')
    if gallery['status']!='COMPLETE' or gallery['identity'].get('model_blinded') is not True:
        raise ValueError('Final gallery incomplete/nonblind')
    qids={q['request_id'] for q in manifest['evaluation_requests']}
    maps={r['request_id']:r['aliases'] for r in mapping['rows']};images={r['request_id']:r for r in gallery['images']}
    if len(mapping['rows'])!=44 or len(gallery['images'])!=44 or set(maps)!=qids or set(images)!=qids:
        raise ValueError('Final44 mapping/gallery coverage differs')
    if any(set(m)!=set('ABC') or set(m.values())!=set(LABELS) for m in maps.values()):
        raise ValueError('Final aliases are not per-request bijections')
    for p in (mp,gp,D.resolve(path)):protected[str(p)]=D.sha(p)
    rows=A.review_rows(path)
    if len(rows)!=132:raise ValueError('All132 actual final blind reviews required')
    result={};verified=set()
    for row in rows:
        rid,alias=row['request_id'],row['label']
        if rid not in maps or alias not in maps[rid]:raise ValueError('Unexpected final anonymous review')
        label=maps[rid][alias];key=(label,rid)
        if key in result or row.get('review_status')!='REVIEWED':raise ValueError('Duplicate/unreviewed final output')
        if row.get('reviewer_kind')!='AI_PROXY' or not row.get('reviewer') or not F._is_true(row.get('blinded')):
            raise ValueError('Actual model-blind AI_PROXY identity required')
        if any(row.get(k) not in ('YES','NO','UNKNOWN') for k in A.REVIEW_FIELDS):raise ValueError('Unknown review schema')
        im=images[rid]
        if row.get('image_sha256')!=im['image_sha256']:raise ValueError('Final actual image hash differs')
        if rid not in verified:
            if D.sha(im['image_path'])!=im['image_sha256']:raise ValueError('Final image changed')
            protected[im['image_path']]=im['image_sha256'];verified.add(rid)
        result[key]=dict(row,anonymous_label=alias,label=label)
    if set(result)!={(label,rid) for label in LABELS for rid in qids}:raise ValueError('Final review panel incomplete')
    return result


def event_summary(rows,key):
    yes=sum(r[key] is True for r in rows);no=sum(r[key] is False for r in rows);unknown=len(rows)-yes-no
    return dict(YES=yes,NO=no,UNKNOWN=unknown,denominator=len(rows),known_denominator=yes+no,
        known_rate=yes/(yes+no) if yes+no else None,full_panel_lower=yes/len(rows),full_panel_upper=(yes+unknown)/len(rows))


def review_tables(rows,reviews,manifest):
    indexed={(r['label'],r['request_id'],r['stage']):r for r in rows}
    qmap={q['request_id']:q for q in manifest['evaluation_requests']};outputs=[]
    events=('RAW_adoption_known','POST_adoption_known','POST_body_complete_known',
            'joint_RAW_adoption_POST_legal_CF','joint_both_adoptions_body_and_CF')
    for (label,rid),review in sorted(reviews.items()):
        q=qmap[rid];row=dict(review,condition_id=q['condition_id'],parent_id=q['parent_id'],split=q['split'],candidate=q['candidate'])
        for stage in STAGES:
            score=indexed[(label,rid,stage)]
            row.update({stage+'_'+k:score[k] for k in A.NUMERIC+('has_legal_CF','I_status')})
        for field in A.REVIEW_FIELDS:row[field+'_known']=F._truth(row[field])
        row['reward']=row['POST_has_legal_CF']
        row['joint_RAW_adoption_POST_legal_CF']=F._and([row['RAW_adoption_known'],bool(row['reward'])])
        row['joint_both_adoptions_body_and_CF']=F._and([row[field+'_known'] for field in A.REVIEW_FIELDS]+[bool(row['reward'])])
        outputs.append(row)
    summaries=[]
    for label in LABELS:
        own=[r for r in outputs if r['label']==label]
        groups=[('SPLIT',s,own if s=='ALL' else [r for r in own if r['split']==s]) for s in SPLITS]
        groups += [('CONDITION',cid,[r for r in own if r['condition_id']==cid]) for cid in sorted({r['condition_id'] for r in own})]
        for scope,name,group in groups:
            summaries.append(dict(label=label,scope=scope,group=name,outputs=len(group),
                **{event:event_summary(group,event) for event in events}))
    paired=[];index={(r['label'],r['request_id']):r for r in outputs}
    for candidate,baseline,role in CONTRASTS:
        for rid,q in sorted(qmap.items()):
            a,b=index[(candidate,rid)],index[(baseline,rid)]
            row=dict(candidate_label=candidate,baseline_label=baseline,comparison_role=role,request_id=rid,
                condition_id=q['condition_id'],parent_id=q['parent_id'],split=q['split'])
            for event in events:
                av,bv=a[event],b[event]
                row['candidate_'+event],row['baseline_'+event]=av,bv
                row[event+'_paired_delta']=None if av is None or bv is None else int(av)-int(bv)
                row[event+'_full_panel_delta_lower']=(0 if av is None else int(av))-(1 if bv is None else int(bv))
                row[event+'_full_panel_delta_upper']=(1 if av is None else int(av))-(0 if bv is None else int(bv))
            paired.append(row)
    paired_summaries=[]
    for candidate,baseline,role in CONTRASTS:
        own=[r for r in paired if r['candidate_label']==candidate and r['baseline_label']==baseline]
        groups=[('SPLIT',s,own if s=='ALL' else [r for r in own if r['split']==s]) for s in SPLITS]
        groups += [('CONDITION',cid,[r for r in own if r['condition_id']==cid]) for cid in sorted({r['condition_id'] for r in own})]
        for scope,name,group in groups:
            row=dict(candidate_label=candidate,baseline_label=baseline,comparison_role=role,scope=scope,group=name,paired_requests=len(group))
            for event in events:
                values=[r[event+'_paired_delta'] for r in group if r[event+'_paired_delta'] is not None]
                row[event]=dict(common_known_pairs=len(values),unknown_pairs=len(group)-len(values),
                    gained=sum(x==1 for x in values),lost=sum(x==-1 for x in values),
                    common_known_mean_delta=sum(values)/len(values) if values else None,
                    full_panel_mean_delta_lower=sum(r[event+'_full_panel_delta_lower'] for r in group)/len(group),
                    full_panel_mean_delta_upper=sum(r[event+'_full_panel_delta_upper'] for r in group)/len(group))
            paired_summaries.append(row)
    return outputs,summaries,paired,paired_summaries


def run(reviews_path=None):
    A.model_free();manifest,registration=contract();gatepath=OUT/'analysis/decode_gate.json'
    if not gatepath.exists():return readiness(dict(status='NOT_RUN_PENDING_CALIBRATION_GATE',complete_summary_written=False))
    gate=D.read(gatepath)
    if gate['status']!='PASS':return readiness(dict(status='NOT_RUN_CALIBRATION_GATE_'+gate['status'],complete_summary_written=False,
        reason='This prepared final-analysis code does not authorize final rollouts after a failed/NA calibration'))
    missing=[str(p) for p in required(manifest) if not p.exists()]
    if missing:return readiness(dict(status='PENDING_FINAL_SCORES',missing=missing,missing_count=len(missing),complete_summary_written=False))
    protected={str(p):D.sha(p) for p in (OUT/'inputs/manifest.json',OUT/'registration.json',gatepath,Path(__file__),Path(A.__file__),Path(F.__file__))}
    rows,originals=collect(manifest,gate,protected)
    models,conditions,pairs,paired_summaries=numerical_tables(rows)
    for name,data in (('per_output_stage.csv',rows),('per_model_split_stage.csv',models),('per_condition_stage.csv',conditions),
                      ('paired_request_stage.csv',pairs),('paired_condition_and_split_stage.csv',paired_summaries)):
        A.csv_write(DEST/name,data)
    tails=[]
    for candidate,baseline,role in CONTRASTS:
        for stage in STAGES:
            sub=[r for r in pairs if r['candidate_label']==candidate and r['baseline_label']==baseline and r['stage']==stage]
            for metric in ('CF_illegal_fraction','FF_illegal_fraction','I_lower','I_upper'):
                for rank,row in enumerate(sorted((r for r in sub if r[metric+'_delta'] is not None),key=lambda r:(-r[metric+'_delta'],r['request_id'])),1):
                    tails.append(dict(candidate_label=candidate,baseline_label=baseline,comparison_role=role,stage=stage,
                        metric=metric,rank_largest_degradation_first=rank,request_id=row['request_id'],condition_id=row['condition_id'],
                        split=row['split'],baseline=row['baseline_'+metric],candidate=row['candidate_'+metric],delta_pp=row[metric+'_delta_pp']))
    A.csv_write(DEST/'all_geometry_tails.csv',tails)
    D.write(DEST/'original_metrics.json',dict(status='COMPLETE',rows=originals))
    quantitative=dict(status='COMPLETE',version=VERSION,outputs=132,stage_scores=264,requests=44,TRAIN_requests=32,DEV_requests=12,
        conditions=11,train_conditions=8,dev_conditions=3,model_summaries=models,paired_summaries=paired_summaries,
        primary_contrast='RL_STOCH minus FULL_STOCH, paired identical frozen initial and transition noises',
        deterministic_reference='FULL_DET separately describes added stochasticity and absolute quality/use reference',
        normalization='Per-output actual stage free faces for CF/FF; per-output point/lower I and upper/status retained; per-output macro means with explicit finite n/missing counts, no zero fill, four noises per C',
        limitations=['No unique parent GT reconstruction objective','Three DEV C and four noises each are a small descriptive panel',
            'Shared noise pairing controls inputs; no proof of unseen-parent generalization','Metric dispersion is not geometric diversity',
            'No visual usability or joint-success claim before actual reviews'],input_sources=protected,model_calls=0,geometry_rescores=0)
    D.write(DEST/'quantitative.json',quantitative)
    path=D.resolve(reviews_path or EV/'blind_gallery/reviews_actual.json')
    review_paths=[path,EV/'private/label_mapping.json',EV/'blind_gallery/manifest.json']
    missing=[str(p) for p in review_paths if not p.exists()]
    if missing:return readiness(dict(status='PENDING_ACTUAL_BLIND_REVIEW',missing=missing,quantitative_status='COMPLETE',complete_summary_written=False))
    try:reviews=actual_reviews(manifest,path,protected)
    except ValueError as error:return readiness(dict(status='PENDING_VALID_ACTUAL_BLIND_REVIEW',reason=str(error),complete_summary_written=False))
    outputs,summaries,review_pairs,review_pair_summaries=review_tables(rows,reviews,manifest)
    for name,data in (('per_output_review_metrics.csv',outputs),('review_condition_and_split.csv',summaries),
                      ('paired_request_reviews.csv',review_pairs),('paired_condition_and_split_reviews.csv',review_pair_summaries)):
        A.csv_write(DEST/name,data)
    result=dict(quantitative,actual_reviews_path=str(path),actual_reviews_sha256=D.sha(path),review_rows=132,
        review_summaries=summaries,paired_review_summaries=review_pair_summaries,input_sources=protected,
        primary_joint='RAW adoption YES and POST legal CF edge count >0',
        stricter_joint='RAW and POST adoption YES, POST body complete YES, and POST legal CF edge count >0',
        semantics='Actual independent model-blind AI_PROXY, not human certification; UNKNOWN retained with common-known and full-panel bounds',
        decision='DESCRIPTIVE_ONLY_NO_AUTOMATIC_DEFAULT_CHANGE_OR_FURTHER_TRAINING')
    D.write(DEST/'summary.json',result);A.model_free()
    return readiness(dict(status='COMPLETE',summary_path=str(DEST/'summary.json'),summary_sha256=D.sha(DEST/'summary.json'),complete_summary_written=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews',help='Actual anonymous final reviews JSON/CSV, all132 rows')
    args=parser.parse_args();result=run(args.reviews)
    print(json.dumps({k:v for k,v in result.items() if k!='missing'},ensure_ascii=False,sort_keys=True))
