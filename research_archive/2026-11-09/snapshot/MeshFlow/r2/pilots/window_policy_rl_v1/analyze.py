"""Read-only calibration aggregation and the preregistered decode gate.

No geometry is rescored, no model is loaded, and no semantic label is inferred.
The final decode_gate.json is immutable and requires all 256 actual blind reviews.
Incomplete readiness is preserved separately; it can never authorize training.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import uuid
import numpy as np
from .bootstrap import bind, ROOT, OUT
D, E = bind()
sys.path.insert(0, str(ROOT / 'r2/src'))
from r2.pilots.terminal_reward_nft_v1 import final_analysis as F

VERSION = 'window_policy_rl_calibration_analysis_v1'
DEST = OUT / 'analysis'
LABELS = ('FULL_DET', 'S005', 'S010', 'S020')
STAGES = ('RAW', 'POST')
REVIEW_FIELDS = ('RAW_adoption', 'POST_adoption', 'POST_body_complete')
NUMERIC = F.CONTINUOUS + ('actual_free_faces', 'original_free_faces', 'deleted_free_faces',
                         'CF_unresolved_pairs', 'FF_unresolved_pairs')
PP_FIELDS = ('CF_illegal_fraction', 'FF_illegal_fraction', 'I_lower', 'I_upper',
             'free_retained_fraction', 'free_near_degenerate_fraction',
             'free_small_angle_fraction', 'reached_free_area_fraction')


def model_free():
    if 'torch' in sys.modules:
        raise RuntimeError('Calibration analysis must remain torch-free')


def digest(value):
    return hashlib.sha256(json.dumps(D.clean(value), sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def csv_write(path, rows):
    F._csv(path, rows)


def publish_readiness(value):
    """Only the convenience readiness pointer changes; every version is preserved."""
    value = dict(version=VERSION, **value)
    snap = DEST / 'readiness' / (digest(value)[:20] + '.json')
    D.write(snap, value)
    pointer = D.checked(DEST / 'calibration_readiness.json')
    payload = dict(value, snapshot_path=str(snap), snapshot_sha256=D.sha(snap))
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temp = pointer.with_name(pointer.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('x', encoding='utf8') as f:
        json.dump(D.clean(payload), f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    os.replace(temp, pointer)
    return payload


def load_contract():
    paths = [OUT/'inputs/manifest.json', OUT/'registration.json', OUT/'gate_contract.json']
    manifest, registration, gate = map(D.read, paths)
    if registration['manifest_sha256'] != D.sha(paths[0]) or gate['registration_sha256'] != D.sha(paths[1]):
        raise ValueError('Frozen manifest/registration/gate chain differs')
    if gate['status'] != 'FROZEN_BEFORE_MODEL_CALLS':
        raise ValueError('Missing frozen gate contract')
    q = manifest['calibration_requests']
    if (len(q) != 64 or len({x['request_id'] for x in q}) != 64
            or sorted(Counter(x['condition_id'] for x in q).values()) != [8]*8):
        raise ValueError('Calibration must contain eight C, eight distinct requests each')
    sigmas = registration['calibration']['labels']
    if sigmas != {'FULL_DET': 0., 'S005': .005, 'S010': .01, 'S020': .02}:
        raise ValueError('Registered calibration labels differ')
    determinate = re.search(r'At least (\d+)/(\d+) determinate', gate['body_unknown_policy'])
    if not determinate or tuple(map(int, determinate.groups())) != (56,64):
        raise ValueError('Unrecognized frozen body missing-value contract')
    if gate['deterministic_zero_positive_policy'] != 'NA_STOP_NO_TRAINING; retention cannot be evaluated':
        raise ValueError('Unrecognized deterministic zero-positive contract')
    return manifest, registration, gate, {str(p): D.sha(p) for p in paths}


def required_paths(manifest):
    paths = []
    for label in LABELS:
        for q in manifest['calibration_requests']:
            cell = OUT/'calibration'/label
            rid = q['request_id']
            paths += [cell/'raw'/rid/'receipt.json', cell/'post'/rid/'receipt.json',
                      cell/'proposal'/rid/'receipt.json']
            paths += [cell/'scores'/rid/stage/'score.json' for stage in STAGES]
    return paths


def read_outputs(manifest, registration, protected):
    from .sample_worker import verify_window_receipt
    from .postprocess import _verify
    rows, originals, violations = [], [], []
    registration_sha = D.sha(OUT/'registration.json')
    verified = {}
    def protect(path, expected=None):
        path = str(D.resolve(path))
        actual = verified.get(path)
        if actual is None:
            actual = D.sha(path); verified[path] = actual
        if expected is not None and actual != expected:
            raise ValueError('Input artifact hash differs: ' + path)
        protected[path] = actual
    for label in LABELS:
        for q in manifest['calibration_requests']:
            rid = q['request_id']; cell = OUT/'calibration'/label
            rawp, postp, prop = (cell/k/rid/'receipt.json' for k in ('raw','post','proposal'))
            raw, post, proposal = map(D.read, (rawp, postp, prop))
            for p in (rawp, postp, prop): protect(p)
            try:
                if (raw['label'] != label or raw['request_id'] != rid or raw['model_mode'] != 'reference'
                        or raw['sigma'] != registration['calibration']['labels'][label]
                        or raw['registration_sha256'] != registration_sha):
                    raise ValueError('RAW calibration identity differs')
                if raw['model_identity']['base_file_sha256'] != manifest['source_binding']['sha256']:
                    raise ValueError('RAW source is not frozen FULL')
                verify_window_receipt(q, raw)
                if (post['status'] != 'COMPLETE' or post['request_id'] != rid or post['label'] != label
                        or post['input_sha256'] != q['input_sha256']):
                    raise ValueError('POST calibration identity differs')
                if proposal['status'] != 'PASS' or proposal['model_forwards'] != 1:
                    raise ValueError('Expected one completed official D0 call')
                for p,h in post['files'].items(): protect(p,h)
                for key in ('raw','ply','result','window_buffer'):
                    protect(raw[key+'_path'],raw[key+'_sha256'])
            except (ValueError, KeyError) as exc:
                violations.append(dict(label=label,request_id=rid,kind='ARTIFACT_CONTRACT',error=str(exc)))
            for stage in STAGES:
                sp = cell/'scores'/rid/stage/'score.json'; score = D.read(sp); protect(sp)
                row = {k:q[k] for k in ('request_id','condition_id','parent_id','candidate','N','K','split','noise_seed') if k in q}
                row.update(label=label,sigma=registration['calibration']['labels'][label],stage=stage,
                    input_sha256=q['input_sha256'],initial_Gaussian_sha256=q['Gaussian_sha256'],
                    transition_Gaussian_sha256=q['transition_Gaussian_sha256'],
                    score_path=str(sp),score_sha256=D.sha(sp),raw_receipt_path=str(rawp),
                    post_receipt_path=str(postp),score_status=score.get('status'),
                    C_preserved_exact=score.get('C_preserved_exact'))
                try:
                    if score.get('original_definition_unchanged') is not True:
                        raise ValueError('Original scoring definition is not certified unchanged')
                    if score['identity']['registration_sha256'] != registration_sha:
                        raise ValueError('Original score registration differs')
                    for p,h in score['identity']['definition_hashes'].items(): protect(p,h)
                    for p,h in score['identity']['input_hashes'].items(): protect(p,h)
                    for p,h in score['files'].items(): protect(p,h)
                    row.update(F.extract_score(score,q,label,stage))
                    if row['actual_free_faces'] != q['N']-q['K']-row['deleted_free_faces']:
                        raise ValueError('Actual free/deletion denominator differs')
                    row['row_status'] = 'PASS'
                except (ValueError, KeyError) as exc:
                    row.update(row_status='FAIL',error=str(exc))
                    violations.append(dict(label=label,request_id=rid,stage=stage,kind='SCORE_CONTRACT',error=str(exc)))
                rows.append(row)
                originals.append(dict(label=label,request_id=rid,stage=stage,score_path=str(sp),
                    score_sha256=row['score_sha256'],metrics=score.get('metrics',{}),
                    topology_summary={k:v for k,v in score.get('topology',{}).items()
                        if v is None or isinstance(v,(str,int,float,bool))}))
    model_free()
    return rows, originals, violations


def numeric_summary(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    return dict(n=len(values),missing=len(rows)-len(values),mean=float(np.mean(values)) if values else None,
        minimum=min(values) if values else None,maximum=max(values) if values else None,
        p90=float(np.quantile(values,.9)) if values else None,median=float(np.median(values)) if values else None,
        sum=float(sum(values)) if values else None)


def aggregate(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[(row['label'],row['stage'],row['condition_id'])].append(row)
    conditions=[];models=[]
    def result(group,label,stage,cid=None):
        output=dict(label=label,stage=stage,outputs=len(group),failed_outputs=sum(r['row_status']!='PASS' for r in group),
            condition_id=cid,legal_CF_outputs=sum(int(r.get('has_legal_CF',0)) for r in group),
            I_status_counts=dict(Counter(r.get('I_status','MISSING') for r in group)),
            topology_status_counts=dict(Counter(r.get('topology_status','MISSING') for r in group)),
            source_C_connectivity_status_counts=dict(Counter(r.get('source_C_connectivity_status','MISSING') for r in group)))
        for key in NUMERIC:
            stats=numeric_summary(group,key)
            for name,value in stats.items():output[key+'_'+name]=value
        output['legal_CF_known_positive_count']=output['legal_CF_outputs']
        output['legal_CF_missing']=sum(r.get('has_legal_CF') is None for r in group)
        if output['legal_CF_missing']:
            output['legal_CF_outputs']=None
        output['legal_CF_output_rate']=None if output['legal_CF_missing'] else output['legal_CF_outputs']/len(group)
        output['mixed_reward_condition']=None if output['legal_CF_missing'] else int(0<output['legal_CF_outputs']<len(group))
        return output
    for (label,stage,cid),group in sorted(grouped.items()):conditions.append(result(group,label,stage,cid))
    for label in LABELS:
        for stage in STAGES:
            group=[r for r in rows if r['label']==label and r['stage']==stage]
            models.append(result(group,label,stage))
    return conditions,models


def pair_rows(rows):
    bykey={(r['label'],r['request_id'],r['stage']):r for r in rows}
    pairs=[]
    for label in LABELS[1:]:
        for (base,rid,stage),b in sorted(bykey.items()):
            if base!='FULL_DET':continue
            a=bykey[(label,rid,stage)]
            out=dict(candidate_label=label,baseline_label='FULL_DET',request_id=rid,
                condition_id=b['condition_id'],parent_id=b['parent_id'],candidate=b['candidate'],stage=stage,
                baseline_has_legal_CF=b.get('has_legal_CF'),candidate_has_legal_CF=a.get('has_legal_CF'))
            if a.get('has_legal_CF') is not None and b.get('has_legal_CF') is not None:
                out['legal_CF_gained']=int(a['has_legal_CF']==1 and b['has_legal_CF']==0)
                out['legal_CF_lost']=int(a['has_legal_CF']==0 and b['has_legal_CF']==1)
                out['DET_positive_retained']=int(a['has_legal_CF']==1 and b['has_legal_CF']==1)
            for key in NUMERIC:
                av,bv=a.get(key),b.get(key)
                out['baseline_'+key],out['candidate_'+key]=bv,av
                out[key+'_delta']=None if av is None or bv is None else av-bv
                if key in PP_FIELDS:
                    out[key+'_delta_pp']=None if av is None or bv is None else 100*(av-bv)
            pairs.append(out)
    return pairs


def quantitative_gates(rows,conditions,models,pairs,registration,violations):
    gate=registration['gate'];qids={r['request_id'] for r in rows}
    model={(r['label'],r['stage']):r for r in models}
    base=model[('FULL_DET','POST')]
    base_positive=base['legal_CF_outputs']
    results=[]
    for label in LABELS[1:]:
        own=model[(label,'POST')]
        paired=[r for r in pairs if r['candidate_label']==label and r['stage']=='POST']
        mixed_values=[r['mixed_reward_condition'] for r in conditions if r['label']==label and r['stage']=='POST']
        mixed=None if None in mixed_values else sum(mixed_values)
        retained=sum(r.get('DET_positive_retained',0) for r in paired)
        rate_ratio=own['legal_CF_outputs']/base_positive if base_positive and own['legal_CF_outputs'] is not None else None
        retention=retained/base_positive if base_positive else None
        checks=dict(all_C_finite_original_score_and_receipts_PASS=not violations,
            mixed_conditions=None if mixed is None else mixed>=gate['mixed_conditions_minimum'],
            rate_vs_DET=None if rate_ratio is None else rate_ratio>=gate['stochastic_connection_rate_minimum_fraction_of_DET'],
            DET_positive_retention=None if retention is None else retention>=gate['paired_DET_positive_retention_minimum'])
        deltas={}
        for key,threshold in (('CF_illegal_fraction','max_mean_CF_delta_percentage_points'),
                              ('FF_illegal_fraction','max_mean_FF_delta_percentage_points'),
                              ('I_lower','max_mean_I_delta_percentage_points')):
            a,b=own[key+'_mean'],base[key+'_mean']
            available=own[key+'_missing']==0 and base[key+'_missing']==0
            delta=100*(a-b) if available else None
            deltas[key+'_mean_delta_pp']=delta
            checks[key]=None if delta is None else delta<=gate[threshold]
        status='FAIL' if violations else 'NA' if base_positive in (None,0) or any(v is None for v in checks.values()) else 'PASS' if all(checks.values()) else 'FAIL'
        results.append(dict(label=label,sigma=registration['calibration']['labels'][label],status=status,
            mixed_conditions=mixed,candidate_legal_CF_outputs=own['legal_CF_outputs'],DET_legal_CF_outputs=base_positive,
            candidate_connection_rate=own['legal_CF_output_rate'],DET_connection_rate=base['legal_CF_output_rate'],
            connection_rate_ratio=rate_ratio,DET_positive_retained_count=retained,DET_positive_retention=retention,
            deltas=deltas,checks=checks,NA_reasons=['DET_ZERO_POSITIVE'] if base_positive==0 else
                [name+'_UNAVAILABLE' for name,value in checks.items() if value is None]))
    return dict(status='COMPLETE',candidates=results,violations=violations,outputs=256,stage_scores=len(rows),
        request_count=len(qids),conditions=8,trajectories_per_C=8,
        reward='int(POST topology.C_F_valid_edge_count > 0)',
        gate_units='Rates are fractions; CF/FF/I mean differences are percentage points (100 times fraction difference)',
        I_gate='Original ConflictCov_C point/lower value only; original upper bounds and status separately retained',
        normalization='Each output CF/FF denominator is its own actual stage free-face count; mean over all64 outputs, equal8C exposure',
        body_review_status='NOT_USED_IN_QUANTITATIVE_GATE; mandatory for final decode_gate',
        selected_sigma=None,training_authorized=False)


def review_rows(path):
    if str(path).lower().endswith('.csv'):
        with D.resolve(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    value=D.read(path)
    if isinstance(value,dict):
        if value.get('status')!='COMPLETE':raise ValueError('Actual review bundle incomplete')
        return value['rows']
    return value


def read_reviews(manifest,path,protected):
    mp=OUT/'calibration/private/label_mapping.json';gp=OUT/'calibration/blind_gallery/manifest.json'
    mapping,gallery=D.read(mp),D.read(gp)
    if mapping['status']!='FROZEN' or mapping['manifest_sha256']!=D.sha(OUT/'inputs/manifest.json'):
        raise ValueError('Anonymous assignment changed')
    if gallery['status']!='COMPLETE' or gallery['identity'].get('model_blinded') is not True:
        raise ValueError('Incomplete/nonblind gallery')
    qids={q['request_id'] for q in manifest['calibration_requests']}
    maps={r['request_id']:r['aliases'] for r in mapping['rows']}
    images={r['request_id']:r for r in gallery['images']}
    if set(maps)!=qids or len(maps)!=64 or len(mapping['rows'])!=64 or set(images)!=qids or len(images)!=64 or len(gallery['images'])!=64:
        raise ValueError('Blind mapping/gallery coverage differs')
    if any(set(v)!=set('ABCD') or set(v.values())!=set(LABELS) for v in maps.values()):
        raise ValueError('Anonymous assignment is not a bijection per request')
    for p in (mp,gp,D.resolve(path)):protected[str(p)]=D.sha(p)
    rows=review_rows(path)
    if len(rows)!=256:raise ValueError('All256 POST actual blind rows required')
    result={};verified=set()
    for row in rows:
        rid,alias=row['request_id'],row['label']
        if rid not in maps or alias not in maps[rid]:raise ValueError('Unexpected anonymous review key')
        label=maps[rid][alias];key=(label,rid)
        if key in result:raise ValueError('Duplicate actual blind review')
        if row.get('review_status')!='REVIEWED':raise ValueError('Actual review row still pending')
        if row.get('reviewer_kind')!='AI_PROXY' or not str(row.get('reviewer','')).strip():
            raise ValueError('Frozen actual AI_PROXY reviewer identity required')
        if not F._is_true(row.get('blinded')):raise ValueError('Actual model-blind status missing')
        if any(row.get(k) not in ('YES','NO','UNKNOWN') for k in REVIEW_FIELDS):
            raise ValueError('Review labels must be YES/NO/UNKNOWN without imputation')
        image=images[rid]
        if row.get('image_sha256')!=image['image_sha256']:raise ValueError('Review image identity differs')
        if rid not in verified:
            if D.sha(image['image_path'])!=image['image_sha256']:raise ValueError('Reviewed image changed')
            protected[image['image_path']]=image['image_sha256'];verified.add(rid)
        result[key]=dict(row,anonymous_label=alias,label=label)
    if set(result)!={(label,rid) for label in LABELS for rid in qids}:
        raise ValueError('Full review coverage differs')
    return result


def decode_gate(quant,reviews,registration,gate_contract):
    bylabel={label:[r for (lab,_),r in reviews.items() if lab==label] for label in LABELS}
    counts={label:dict(Counter(r['POST_body_complete'] for r in rows)) for label,rows in bylabel.items()}
    minimum=int(re.search(r'At least (\d+)/',gate_contract['body_unknown_policy']).group(1))
    det=counts['FULL_DET'];details=[]
    for candidate in quant['candidates']:
        label=candidate['label'];own=counts[label]
        newdamage=[]
        for (lab,rid),base in reviews.items():
            if lab=='FULL_DET' and base['POST_body_complete']=='YES' and reviews[(label,rid)]['POST_body_complete'] in ('NO','UNKNOWN'):
                newdamage.append(rid)
        det_known=det.get('YES',0)+det.get('NO',0);own_known=own.get('YES',0)+own.get('NO',0)
        missing_ok=det_known>=minimum and own_known>=minimum
        drop=det.get('YES',0)-own.get('YES',0)
        checks=dict(body_determinate_at_least_56=missing_ok,
            DET_YES_to_candidate_NO_or_UNKNOWN=len(newdamage)<=registration['gate']['max_new_POST_body_NO_relative_to_DET_YES'],
            body_YES_drop=drop<=registration['gate']['max_drop_POST_body_YES_count'])
        status='NA' if candidate['status']=='NA' or not missing_ok else 'FAIL' if candidate['status']=='FAIL' or not all(checks.values()) else 'PASS'
        details.append(dict(candidate,quantitative_status=candidate['status'],status=status,
            body_checks=checks,DET_body_counts=det,candidate_body_counts=own,
            DET_determinate=det_known,candidate_determinate=own_known,
            new_damage_upper_bound=len(newdamage),new_damage_request_ids=newdamage,body_YES_drop=drop))
    passed=sorted((d for d in details if d['status']=='PASS'),key=lambda d:d['sigma'])
    selected=passed[0] if passed else None
    status='PASS' if selected else 'NA' if any(d['status']=='NA' for d in details) else 'FAIL'
    return dict(version=VERSION,status=status,candidates=details,review_rows=256,
        selected_label=selected['label'] if status=='PASS' else None,
        selected_sigma=selected['sigma'] if status=='PASS' else None,
        selection='Smallest sigma passing every preregistered quantitative and actual blind-body gate',
        training_authorized_by_gate=status=='PASS',
        action='PROCEED_TO_PREAUTHORIZED_FINITE_TRAINING' if status=='PASS' else 'STOP_NO_TRAINING',
        semantic_review='Actual model-blind AI_PROXY; not human certification or automatic quality labels',
        UNKNOWN_policy='Never YES/zero imputed; require >=56 determinate per64, count DET YES->UNKNOWN as potential new damage',
        review_label_counts={label:{field:dict(Counter(r[field] for r in rows)) for field in REVIEW_FIELDS} for label,rows in bylabel.items()})


def run(reviews_path=None):
    model_free();manifest,registration,contract,protected=load_contract()
    protected.update({str(Path(__file__)):D.sha(__file__),str(Path(F.__file__)):D.sha(F.__file__)})
    missing=[str(p) for p in required_paths(manifest) if not p.exists()]
    if missing:
        return publish_readiness(dict(status='PENDING_SCORES',missing=missing,missing_count=len(missing),
            expected_outputs=256,expected_stage_scores=512,training_authorized=False,model_calls=0))
    rows,originals,violations=read_outputs(manifest,registration,protected)
    conditions,models=aggregate(rows);pairs=pair_rows(rows)
    quant=quantitative_gates(rows,conditions,models,pairs,registration,violations)
    quant.update(version=VERSION,registration_sha256=D.sha(OUT/'registration.json'),
        gate_contract_sha256=D.sha(OUT/'gate_contract.json'),source_sha256=D.sha(__file__),
        model_summaries=models,condition_summaries=conditions,input_sources=protected,model_calls=0)
    csv_write(DEST/'calibration_per_output_stage.csv',rows)
    csv_write(DEST/'calibration_per_condition_stage.csv',conditions)
    csv_write(DEST/'calibration_per_model_stage.csv',models)
    csv_write(DEST/'calibration_paired_stage.csv',pairs)
    pairgroups=defaultdict(list)
    for row in pairs:pairgroups[(row['candidate_label'],row['stage'],row['condition_id'])].append(row)
    pairconditions=[]
    for (label,stage,cid),group in sorted(pairgroups.items()):
        row=dict(candidate_label=label,baseline_label='FULL_DET',stage=stage,condition_id=cid,requests=len(group),
            legal_CF_gained=sum(x.get('legal_CF_gained',0) for x in group),
            legal_CF_lost=sum(x.get('legal_CF_lost',0) for x in group))
        for field in NUMERIC:
            stats=numeric_summary(group,field+'_delta')
            for name,value in stats.items():row[field+'_delta_'+name]=value
            if field in PP_FIELDS:row[field+'_mean_delta_pp']=None if stats['mean'] is None else 100*stats['mean']
        pairconditions.append(row)
    csv_write(DEST/'calibration_paired_condition_stage.csv',pairconditions)
    tails=[]
    for label in LABELS[1:]:
        for stage in STAGES:
            sub=[r for r in pairs if r['candidate_label']==label and r['stage']==stage]
            for field in ('CF_illegal_fraction','FF_illegal_fraction','I_lower','I_upper'):
                ordered=sorted((r for r in sub if r[field+'_delta'] is not None),
                    key=lambda r:(-r[field+'_delta'],r['request_id']))
                for rank,r in enumerate(ordered,1):
                    tails.append(dict(candidate_label=label,stage=stage,metric=field,rank_largest_degradation_first=rank,
                        request_id=r['request_id'],condition_id=r['condition_id'],parent_id=r['parent_id'],
                        baseline=r['baseline_'+field],candidate=r['candidate_'+field],delta_pp=r[field+'_delta_pp']))
    csv_write(DEST/'calibration_all_geometry_tails.csv',tails)
    D.write(DEST/'calibration_original_metrics.json',dict(status='COMPLETE',rows=originals,
        definition='All original metrics dictionaries, plus scalar/string topology summaries; full original scores remain at exact hashed paths'))
    D.write(DEST/'calibration_quantitative.json',quant)
    reviewpath=D.resolve(reviews_path or OUT/'calibration/blind_gallery/reviews_actual.json')
    review_required=[reviewpath,OUT/'calibration/blind_gallery/manifest.json',OUT/'calibration/private/label_mapping.json']
    missing_review=[str(p) for p in review_required if not p.exists()]
    if missing_review:
        return publish_readiness(dict(status='PENDING_ACTUAL_BLIND_REVIEW',missing=missing_review,
            quantitative_status='COMPLETE',quantitative_path=str(DEST/'calibration_quantitative.json'),
            quantitative_sha256=D.sha(DEST/'calibration_quantitative.json'),training_authorized=False,model_calls=0))
    try:
        reviews=read_reviews(manifest,reviewpath,protected)
    except ValueError as exc:
        return publish_readiness(dict(status='PENDING_VALID_ACTUAL_BLIND_REVIEW',reason=str(exc),
            quantitative_status='COMPLETE',training_authorized=False,model_calls=0))
    decision=decode_gate(quant,reviews,registration,contract)
    decision.update(quantitative_path=str(DEST/'calibration_quantitative.json'),
        quantitative_sha256=D.sha(DEST/'calibration_quantitative.json'),actual_reviews_path=str(reviewpath),
        actual_reviews_sha256=D.sha(reviewpath),input_sources=protected,model_calls=0)
    csv_write(DEST/'calibration_reviews_decoded.csv',list(reviews.values()))
    merged=[]
    bykey={(r['label'],r['request_id'],r['stage']):r for r in rows}
    for (label,rid),review in sorted(reviews.items()):
        row=dict(review)
        for stage in STAGES:
            score=bykey[(label,rid,stage)]
            row.update({stage+'_'+k:score.get(k) for k in NUMERIC+('has_legal_CF','I_status')})
        row['reward']=row['POST_has_legal_CF']
        row['joint_RAW_adoption_POST_legal_CF']=False if row['reward']==0 or row['RAW_adoption']=='NO' else True if row['reward']==1 and row['RAW_adoption']=='YES' else None
        merged.append(row)
    csv_write(DEST/'calibration_per_output_review_metrics.csv',merged)
    D.write(DEST/'decode_gate.json',decision)
    model_free()
    return publish_readiness(dict(status='COMPLETE',decode_gate_status=decision['status'],
        selected_sigma=decision['selected_sigma'],decode_gate_path=str(DEST/'decode_gate.json'),
        decode_gate_sha256=D.sha(DEST/'decode_gate.json'),training_authorized=decision['status']=='PASS',model_calls=0))


def self_test():
    """Synthetic scalar gate checks only; no geometry/model/real labels read."""
    import copy
    manifest,registration,contract,_=load_contract()
    rows=[]
    for label in LABELS:
        for c in range(8):
            for i in range(8):
                for stage in STAGES:
                    r=dict(label=label,condition_id=f'C{c}',parent_id=f'P{c}',request_id=f'C{c}_n{i}',
                        candidate=i,stage=stage,row_status='PASS',has_legal_CF=int(i<4),I_status='PASS',
                        topology_status='PASS',source_C_connectivity_status='PASS')
                    r.update({key:.01 for key in NUMERIC})
                    r.update(actual_free_faces=128,original_free_faces=128,deleted_free_faces=0,legal_CF_edges=int(i<4),
                        free_retained_fraction=1.,legal_CF_length=float(i<4))
                    rows.append(r)
    conditions,models=aggregate(rows);pairs=pair_rows(rows)
    quant=quantitative_gates(rows,conditions,models,pairs,registration,[])
    reviews={(label,f'C{c}_n{i}'):dict(label=label,request_id=f'C{c}_n{i}',
        RAW_adoption='YES',POST_adoption='YES',POST_body_complete='YES')
        for label in LABELS for c in range(8) for i in range(8)}
    checks={}
    checks['all_equal_mixed_quant_PASS']=all(r['status']=='PASS' and r['mixed_conditions']==8 for r in quant['candidates'])
    checks['smallest_passing_sigma']=decode_gate(quant,reviews,registration,contract)['selected_sigma']==.005
    damaged=copy.deepcopy(reviews)
    for i in range(3):damaged[('S005',f'C0_n{i}')]['POST_body_complete']='NO'
    decision=decode_gate(quant,damaged,registration,contract)
    checks['three_damage_rejects_first_sigma']=decision['candidates'][0]['status']=='FAIL' and decision['selected_sigma']==.01
    damaged[('S005','C0_n2')]['POST_body_complete']='UNKNOWN'
    checks['UNKNOWN_counts_in_damage_upper']=decode_gate(quant,damaged,registration,contract)['candidates'][0]['new_damage_upper_bound']==3
    unknown=copy.deepcopy(reviews)
    for j in range(9):unknown[('S005',f'C{j//8}_n{j%8}')]['POST_body_complete']='UNKNOWN'
    decision=decode_gate(quant,unknown,registration,contract)
    checks['fewer_than56_determinate_NA']=decision['candidates'][0]['status']=='NA' and decision['selected_sigma']==.01
    for j in range(9):unknown[('FULL_DET',f'C{j//8}_n{j%8}')]['POST_body_complete']='UNKNOWN'
    checks['DET_indeterminate_stops_all']=decode_gate(quant,unknown,registration,contract)['status']=='NA'
    zero=copy.deepcopy(rows)
    for row in zero:
        if row['label']=='FULL_DET':row['has_legal_CF']=0
    c,m=aggregate(zero);q=quantitative_gates(zero,c,m,pair_rows(zero),registration,[])
    checks['DET_zero_positive_NA']=all(x['status']=='NA' and x['DET_positive_retention'] is None for x in q['candidates'])
    failure=quantitative_gates(rows,conditions,models,pairs,registration,[{'kind':'C_CONTRACT_FAIL'}])
    checks['hard_artifact_failure_FAIL']=all(x['status']=='FAIL' for x in failure['candidates'])
    missing=copy.deepcopy(rows)
    next(r for r in missing if r['label']=='S005' and r['stage']=='POST')['I_lower']=None
    c,m=aggregate(missing);q=quantitative_gates(missing,c,m,pair_rows(missing),registration,[])
    checks['I_missing_not_zero_or_shrunk_denominator']=q['candidates'][0]['status']=='NA' and q['candidates'][0]['deltas']['I_lower_mean_delta_pp'] is None
    costly=copy.deepcopy(rows)
    for r in costly:
        if r['label']=='S005' and r['stage']=='POST':r['CF_illegal_fraction']=.031
    c,m=aggregate(costly);q=quantitative_gates(costly,c,m,pair_rows(costly),registration,[])
    checks['CF_2_1pp_fails_2pp_gate']=q['candidates'][0]['status']=='FAIL' and abs(q['candidates'][0]['deltas']['CF_illegal_fraction_mean_delta_pp']-2.1)<1e-10
    assert all(checks.values()),checks
    model_free()
    result=dict(status='PASS',version=VERSION,checks=checks,source_sha256=D.sha(__file__),
        actual_model_calls=0,geometry_rescores=0,torch_imported=False,
        scope='Synthetic scalar aggregation/gate tests only; not actual calibration outcomes')
    D.write(OUT/'engineering/calibration_analysis_cpu_checks.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews',help='Complete actual anonymous review JSON or CSV; default calibration/blind_gallery/reviews_actual.json')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    result=self_test() if args.self_test else run(args.reviews)
    print(json.dumps({k:v for k,v in result.items() if k not in ('missing','input_sources')},ensure_ascii=False,sort_keys=True))
