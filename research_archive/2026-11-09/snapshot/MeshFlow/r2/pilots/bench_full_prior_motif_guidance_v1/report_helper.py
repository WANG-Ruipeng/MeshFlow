"""Final protocol screens from sealed metrics and actual visual observations.

This independent closeout module never invokes a model, scorer or renderer and
never changes the frozen prepare/evaluate/analyze implementations. Incomplete
visual evidence returns PENDING without writing a completion receipt. A PASS
screen is research evidence, not an instruction to promote a model or default.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import sys

from . import prepare as P
from . import evaluate as E

VERSION = 'motif_final_cross_parent_protocol_and_R_drift_v1'
DEST = P.OUT/'analysis_numeric/final/report_helper'
AXES = ('body', 'adoption', 'support', 'fragment_evidence', 'folding')
OLD_AXES = dict(body='overall_shape', adoption='C_use', support='support_retention',
                fragment_evidence='visible_fragment_or_gap', folding='layering_or_fold')
NEGATIVE = dict(body={'IMPAIRED','UNREASONABLE'}, adoption={'IMPLAUSIBLE'},
                support={'MISSING','IMPAIRED'})
STAGES = ('RAW', 'CLEAN_FINAL_POST')
SCOPES = ('CONFIRMATION_ALL', 'CONFIRMATION_ADAPTATION_UNSEEN')
METRICS = {'CF_illegal_affected_free_fraction':(.02,.05),
           'FF_illegal_affected_free_fraction':(.02,.10), 'ConflictCov_C':(.02,.10)}
OLD_R = P.ROOT/'experiments/r2_bench_four_routes_v1/exposure/visual_review/summary_v1/actual_observations.csv'


def _number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _parse(value):
    if value=='':return None
    if value=='True':return True
    if value=='False':return False
    try:return json.loads(value)
    except (ValueError,TypeError):return value


def _csvread(path):
    with P.resolve(path).open(encoding='utf-8-sig',newline='') as handle:
        return [{k:_parse(v) for k,v in row.items()} for row in csv.DictReader(handle)]


def _verify(files):
    for path,digest in files.items():
        if P.resolve(path).suffix.lower() in ('.pt','.pth','.ckpt'):
            raise ValueError('Report helper does not hash or load weight files')
        if P.sha(path)!=digest:raise ValueError('Sealed evidence changed: '+str(path))


def _book(paths):
    return {str(P.resolve(p)):P.sha(p) for p in paths}


def _receipt(path, protected, source=None, statuses=('COMPLETE',)):
    saved=P.read(path)
    if saved.get('status') not in statuses:raise ValueError('Incomplete receipt: '+str(path))
    if source is not None and saved.get('source_sha256')!=P.sha(source):
        raise ValueError('Receipt implementation changed: '+str(path))
    _verify(saved.get('files',{}));_verify(saved.get('protected',{}))
    protected.update(_book([path]));protected.update(saved.get('files',{}));protected.update(saved.get('protected',{}))
    return saved


def _sealed(path, receipt):
    path=P.resolve(path)
    digest=next((h for p,h in receipt['files'].items() if P.resolve(p)==path),None)
    if digest is None or P.sha(path)!=digest:raise ValueError('Input is not sealed by its receipt: '+str(path))


def _cohort(q):
    if q['panel_role']=='calibration':return 'CALIBRATION'
    if q['split']=='DEV':return 'CONFIRMATION_DEV'
    if q['split'] in ('NEW_PARENT_HOLDOUT','HIGH_HOLDOUT'):return 'CONFIRMATION_HOLDOUT'
    if q['split'] in ('CURRENT_TRAIN','NEW_TRAIN','HIGH_TRAIN','NEW_WIDE_TRAIN'):
        return 'CONFIRMATION_EXPOSURE_TRAIN'
    raise ValueError('Unexpected split')


def _scope_ids(requests, scope):
    if scope not in SCOPES:raise ValueError(scope)
    ids=[rid for rid,q in requests.items() if q['panel_role']=='confirmation' and
         (scope=='CONFIRMATION_ALL' or _cohort(q) in ('CONFIRMATION_DEV','CONFIRMATION_HOLDOUT'))]
    expected=38 if scope=='CONFIRMATION_ALL' else 14
    if len(ids)!=expected:raise ValueError('Frozen confirmation scope changed')
    return sorted(ids)


def _joint(row):
    return int(row['body']=='REASONABLE' and row['adoption']=='PLAUSIBLE' and row['support']=='PRESENT')


def drift_canonical(axis, value):
    """Visibility vocabulary only; NO means no observed evidence, not a topology guarantee.

    The frozen old R CSV uses YES/NO_OBVIOUS (plus UNCERTAIN for folds).
    New normalized review tables may retain YES/NO or use VISIBLE/NOT_CLEAR.
    This mapping is used only for historical drift, never for calibration reuse.
    """
    if axis in ('fragment_evidence','folding'):
        aliases={'YES':'VISIBLE','VISIBLE':'VISIBLE','NO_OBVIOUS':'NOT_CLEAR',
                 'NO':'NOT_CLEAR','NOT_CLEAR':'NOT_CLEAR','UNCERTAIN':'UNCERTAIN','UNKNOWN':'UNKNOWN'}
        if value not in aliases:raise ValueError('Unknown visibility token for R drift: '+str(value))
        return aliases[value]
    return value


def visual_contrast(index, requests, reference, stage, scope):
    """Pure request sums; a qualifying parent must meet both predicates itself."""
    rows=[];groups=defaultdict(list)
    for rid in _scope_ids(requests,scope):
        candidate=index.get(('G3',rid,stage));base=index.get((reference,rid,stage));q=requests[rid]
        seen=bool(candidate and base and candidate['review_status']==base['review_status']=='ACTUALLY_VIEWED')
        row=dict(candidate='G3',reference=reference,stage=stage,scope=scope,request_id=rid,
                 parent_id=q['parent_id'],noise_index=int(q['noise_index']),cohort=_cohort(q),report_group=q['report_group'],
                 review_status='COMPLETE' if seen else 'UNKNOWN')
        if seen:
            cj,bj=_joint(candidate),_joint(base)
            base_no=any(base[k] in values for k,values in NEGATIVE.items())
            candidate_no=any(candidate[k] in values for k,values in NEGATIVE.items())
            body_loss=bj==1 and candidate['body'] in NEGATIVE['body']
            support_loss=bj==1 and candidate['support'] in NEGATIVE['support']
            row.update(candidate_J=cj,reference_J=bj,J_delta=cj-bj,
                       strict_new=int(cj==1 and base_no),descriptive_uncertain_new=int(cj==1 and bj==0 and not base_no),
                       retained=int(cj==bj==1),lost=int(bj==1 and cj==0),
                       explicit_task_loss=int(bj==1 and candidate_no),uncertain_loss=int(bj==1 and cj==0 and not candidate_no),
                       explicit_body_loss=int(body_loss),explicit_support_loss=int(support_loss),
                       explicit_body_or_support_loss=int(body_loss or support_loss))
            for axis in AXES:
                row['candidate_'+axis]=candidate[axis];row['reference_'+axis]=base[axis]
        else:
            row.update({k:None for k in ('candidate_J','reference_J','J_delta','strict_new','descriptive_uncertain_new',
                         'retained','lost','explicit_task_loss','uncertain_loss','explicit_body_loss',
                         'explicit_support_loss','explicit_body_or_support_loss')})
        rows.append(row);groups[q['parent_id']].append(row)
    parents=[]
    sum_fields=('candidate_J','reference_J','J_delta','strict_new','descriptive_uncertain_new','retained','lost',
                'explicit_task_loss','uncertain_loss','explicit_body_loss','explicit_support_loss','explicit_body_or_support_loss')
    for pid,values in sorted(groups.items()):
        if len(values)!=2 or {v['noise_index'] for v in values}!={0,1}:
            raise ValueError('Parent is missing its registered n0/n1 pair')
        complete=all(v['review_status']=='COMPLETE' for v in values)
        parent=dict(candidate='G3',reference=reference,stage=stage,scope=scope,parent_id=pid,
                    requests=2,review_status='COMPLETE' if complete else 'UNKNOWN')
        for field in sum_fields:parent[field]=sum(v[field] for v in values) if complete else None
        parent['qualifying_strict_gain_parent']=bool(parent['J_delta']>0 and parent['strict_new']>=1) if complete else None
        parents.append(parent)
    complete=all(p['review_status']=='COMPLETE' for p in parents)
    summary=dict(candidate='G3',reference=reference,stage=stage,scope=scope,
                 expected_requests=len(rows),reviewed_request_pairs=sum(r['review_status']=='COMPLETE' for r in rows),
                 expected_parents=len(parents),complete_parents=sum(p['review_status']=='COMPLETE' for p in parents),
                 qualifying_parent_ids=[p['parent_id'] for p in parents if p['qualifying_strict_gain_parent'] is True],
                 minimum_J_net_gain=2,minimum_qualifying_parents=2,
                 aggregation='J counts are sums of individual requests, never means or rounded parent rates')
    for field in sum_fields:summary[field]=sum(r[field] for r in rows) if complete else None
    summary['qualifying_parent_count']=len(summary['qualifying_parent_ids'])
    summary['J_gate_status']=('UNKNOWN' if not complete else
        'PASS' if summary['J_delta']>=2 and summary['qualifying_parent_count']>=2 else 'FAIL')
    return summary,rows,parents


def quality_combination(gates, reference, stage, scope, requests):
    """Use sealed original metric gates; an incomplete registered I domain stays UNKNOWN."""
    eligible=set(_scope_ids(requests,scope));rows=[]
    expected_parents=19 if scope=='CONFIRMATION_ALL' else 7
    for metric,(mean_limit,tail_limit) in METRICS.items():
        matches=[g for g in gates if (g['candidate'],g['reference'],g['stage'],g['scope'],g['metric'])==
                 ('G3',reference,stage,scope,metric)]
        if len(matches)!=1:raise ValueError('Missing or duplicate sealed geometric gate')
        g=matches[0]
        if g['maximum_parent_mean_increase']!=mean_limit or g['maximum_request_increase']!=tail_limit:
            raise ValueError('Registered geometric thresholds changed')
        expected=16 if metric=='ConflictCov_C' and scope=='CONFIRMATION_ALL' else expected_parents
        tails=g.get('severe_tails',[])
        if any(t['request_id'] not in eligible or not _number(t.get('delta')) or t['delta']<=tail_limit for t in tails):
            raise ValueError('Malformed registered severe-tail evidence')
        complete=(g.get('expected_parents')==expected and g.get('valid_parents')==expected and
                  not g.get('missing_reasons') and _number(g.get('parent_mean_delta')))
        mean_breach=complete and g['parent_mean_delta']>mean_limit
        status='UNKNOWN' if not complete else 'FAIL' if mean_breach or tails else 'PASS'
        row=dict(g,source_gate_status=g.get('status'),status=status,registered_expected_parents=expected,
                 domain_complete=complete,observed_severe_tail=bool(tails),complete_mean_breach=bool(mean_breach))
        rows.append(row)
    # A known severe tail still violates the fixed rule even if another domain is incomplete.
    status=('FAIL' if any(r['status']=='FAIL' or r['observed_severe_tail'] for r in rows) else
            'UNKNOWN' if any(r['status']=='UNKNOWN' for r in rows) else 'PASS')
    return dict(candidate='G3',reference=reference,stage=stage,scope=scope,quality_gate_status=status,
                I_domain_status=next(r['status'] for r in rows if r['metric']=='ConflictCov_C'),
                original_I_denominator='Original query weights; no rescaling by interior support count'),rows


def _combine(j_status,quality_status):
    if 'FAIL' in (j_status,quality_status):return 'FAIL'
    if 'UNKNOWN' in (j_status,quality_status):return 'UNKNOWN'
    return 'PASS'


def _calibration_reuse(lock, visual, tasks, protected):
    """Read the actual locked file itself; supplying a replacement review cannot bypass it."""
    review_path=P.resolve(lock['review_path'])
    mapping_path=P.OUT/'calibration/hidden_method_mapping.json'
    for path in (review_path,mapping_path):
        digest=next((h for p,h in lock['protected'].items() if P.resolve(p)==path),None)
        if digest is None or P.sha(path)!=digest:raise ValueError('Calibration evidence not sealed by lambda lock')
    payload=P.read(review_path);records=payload['records'] if isinstance(payload,dict) else payload
    mapping=P.read(mapping_path)['records']
    if len(records)!=48 or len(mapping)!=48:raise ValueError('Original forty-eight calibration observations required')
    by={(m['request_id'],m['display_label']):m['cell_id'] for m in mapping}
    if len(by)!=48:raise ValueError('Duplicate locked display identity')
    expected={t['cell_id'] for t in tasks.values() if t['panel_role']=='calibration' and t['arm']!='G2'}
    if len(expected)!=48:raise ValueError('Expected R8 + G0/G1 16 + G3 three-lambda24')
    found={};rows=[]
    for source in records:
        record=E._normalize_review(source)
        cell=by.get((record['request_id'],record['display_label']))
        if cell not in expected or cell in found or record.get('stage')!='RAW':raise ValueError('Bad locked calibration identity')
        if record.get('actually_viewed') is not True or record.get('reviewer')!='AI_PROXY' or record.get('method_labels_hidden') is not True:
            raise ValueError('Locked calibration lacks actual hidden-label review evidence')
        current=visual.get((cell,'RAW'))
        if current is None or current['review_status']!='ACTUALLY_VIEWED':raise ValueError('Locked calibration observation not carried into final review')
        for axis in AXES:
            if current[axis]!=record[axis]:raise ValueError('Locked calibration label changed: '+cell+' '+axis)
        found[cell]=record
        rows.append(dict(cell_id=cell,request_id=record['request_id'],stage='RAW',all_five_axes_identical=True,J_identical=_joint(current)==_joint(record)))
    if set(found)!=expected:raise ValueError('Locked calibration panel differs')
    protected.update(_book([review_path,mapping_path]))
    return dict(status='PASS',locked_review_path=str(review_path),locked_review_sha256=P.sha(review_path),
                unique_cells=48,main_panel_RAW_cells=32,auxiliary_RAW_cells=16,axis_comparisons=240,
                changed_axes=0,records=rows)


def _R_drift(visual, requests, protected):
    historical_receipt=_receipt(OLD_R.parent/'completion.json',protected)
    _sealed(OLD_R,historical_receipt)
    historical=[r for r in _csvread(OLD_R) if r['label']=='EXPOSURE' and r['stage'] in STAGES]
    if len(historical)!=92:raise ValueError('Expected original R46 at RAW and FINAL')
    by={};rows=[]
    for original in historical:
        key=(original['request_id'],original['stage'])
        if key in by:raise ValueError('Duplicate historical R review')
        by[key]=original
        q=requests[key[0]];new=visual.get(('R__'+key[0],key[1]))
        if new is None or new['review_status']!='ACTUALLY_VIEWED':raise ValueError('R drift requires current actual review')
        if original['parent_id']!=q['parent_id']:raise ValueError('Historical R parent differs')
        if original.get('actually_viewed') is not True or original.get('reviewer_kind')!='AI_PROXY':
            raise ValueError('Historical R lacks its original actual AI_PROXY review identity')
        normalized=E._normalize_review(original)
        for axis in AXES:
            old_canonical=drift_canonical(axis,normalized[axis])
            new_canonical=drift_canonical(axis,new[axis])
            rows.append(dict(request_id=key[0],parent_id=q['parent_id'],noise_index=int(q['noise_index']),stage=key[1],
                             panel_role=q['panel_role'],cohort=_cohort(q),report_group=q['report_group'],axis=axis,
                             old_field=OLD_AXES[axis],old_token=original[OLD_AXES[axis]],old_normalized=normalized[axis],
                             new_token=new[axis],new_normalized=new[axis],
                             old_canonical=old_canonical,new_canonical=new_canonical,changed=old_canonical!=new_canonical,
                             token_only_change=normalized[axis]!=new[axis] and old_canonical==new_canonical,
                             old_J=_joint(normalized),new_J=_joint(new),
                             attribution='REVIEW_LABEL_DRIFT_ONLY_UNCHANGED_EXPOSURE_GEOMETRY_NOT_METHOD_GAIN'))
    if len(by)!=92 or len(rows)!=460:raise ValueError('R label drift panel mismatch')
    summaries=[]
    for keys in (('stage','axis'),('stage','cohort','axis')):
        grouped=defaultdict(list)
        for row in rows:grouped[tuple(row[k] for k in keys)].append(row)
        for values,group in sorted(grouped.items()):
            summaries.append(dict(zip(keys,values),scope='ALL' if len(keys)==2 else 'BY_COHORT',
                                  axis_labels=len(group),changed=sum(r['changed'] for r in group),
                                  unchanged=sum(not r['changed'] for r in group),model_change=False))
    protected.update(_book([OLD_R]))
    return rows,summaries


def _write_csv(path, rows):
    return E._csv(path,rows)


def _write_text(path,value):
    path=P.resolve(path);data=value.encode('utf-8')
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Refusing to rewrite report artifact')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:handle.write(data)


def _cached():
    path=DEST/'complete.json'
    if not path.exists():return None
    result=P.read(path)
    if result.get('status')!='COMPLETE' or result['source_sha256']!=P.sha(__file__):
        raise ValueError('Completed report helper identity changed')
    _verify(result['files']);_verify(result['protected'])
    return result


def build():
    """Build only when all 476 unique RAW/FINAL observations are sealed and complete."""
    if 'torch' in sys.modules:raise RuntimeError('Use a fresh CPU-only process')
    saved=_cached()
    if saved:return saved
    metrics_folder=P.OUT/'analysis_numeric/final/metrics'
    visual_folder=P.OUT/'analysis_numeric/final/visual'
    required=[P.OUT/'lock.json',metrics_folder/'complete.json',metrics_folder/'numeric_summary.json',
              visual_folder/'complete.json',visual_folder/'actual_review_per_output.csv']
    missing=[str(p) for p in required if not p.is_file()]
    if missing:return dict(status='PENDING_INPUTS',missing=missing,completion_written=False)
    protected={};source=Path(__file__).with_name('analyze.py')
    lock=_receipt(P.OUT/'lock.json',protected,E.__file__,('LOCKED',))
    metric_done=_receipt(metrics_folder/'complete.json',protected,source)
    visual_done=_receipt(visual_folder/'complete.json',protected,source)
    _sealed(metrics_folder/'numeric_summary.json',metric_done)
    _sealed(visual_folder/'actual_review_per_output.csv',visual_done)
    requests={q['request_id']:q for q in P.load_requests()}
    cohort_counts=Counter({q['parent_id']:_cohort(q) for q in requests.values()}.values())
    if dict(cohort_counts)!={'CALIBRATION':4,'CONFIRMATION_EXPOSURE_TRAIN':12,'CONFIRMATION_DEV':4,'CONFIRMATION_HOLDOUT':3}:
        raise ValueError('Frozen parent classification changed')
    all_tasks=P.load_tasks(resolved=True)+P.read(P.OUT/'baseline.json')
    tasks={t['cell_id']:t for t in all_tasks}
    selected={t['cell_id'] for t in P.selected_tasks()}|{t['cell_id'] for t in all_tasks if t['arm']=='R'}
    expected={(cell,stage) for cell in selected for stage in STAGES}|{(cell,'RAW') for cell in tasks if cell not in selected}
    if len(expected)!=476 or len(selected)!=230:raise ValueError('Expected 460 main + 16 auxiliary visual cells')
    observations=_csvread(visual_folder/'actual_review_per_output.csv')
    visual={(r['cell_id'],r['stage']):r for r in observations}
    if len(visual)!=len(observations) or set(visual)!=expected:raise ValueError('Unique final visual panel differs')
    pending=[{'cell_id':key[0],'stage':key[1]} for key,r in visual.items() if r.get('review_status')!='ACTUALLY_VIEWED']
    if pending:return dict(status='PENDING_ACTUAL_REVIEW',pending_cells=len(pending),pending=pending,completion_written=False)
    if visual_done.get('pending_cells')!=0:raise ValueError('Visual completion disagrees with actual rows')
    main_count=0
    for (cell,stage),row in visual.items():
        task=tasks[cell];q=task['q']
        if any(row.get(k)!=q[k] for k in ('request_id','parent_id','panel_role')) or row['arm']!=task['arm']:
            raise ValueError('Final visual task metadata differs')
        normalized=E._normalize_review(row)
        if any(normalized[axis]!=row[axis] for axis in AXES) or row['J']!=_joint(normalized):
            raise ValueError('Final visual normalization/J differs')
        if row['main_panel']!=(cell in selected):raise ValueError('Main/auxiliary visual identity differs')
        main_count+=int(row['main_panel'])
    if main_count!=460:raise ValueError('Expected 460 main visual cells')
    calibration=_calibration_reuse(lock,visual,tasks,protected)
    numeric=P.read(metrics_folder/'numeric_summary.json')
    if numeric.get('stage_rows')!=706 or numeric.get('original_fields')!=177:raise ValueError('Frozen metric panel differs')
    index={(r['arm'],r['request_id'],r['stage']):r for r in visual.values() if r['main_panel']}
    verdicts=[];request_rows=[];parent_rows=[];geometry_rows=[];losses=[];g2_rows=[]
    contrasts=(('MECHANISM','G1','RAW'),('DELIVERED_CANDIDATE','R','CLEAN_FINAL_POST'),
               ('G3_VS_G2_DESCRIPTIVE','G2','RAW'),('G3_VS_G2_DESCRIPTIVE','G2','CLEAN_FINAL_POST'))
    for claim,reference,stage in contrasts:
        for scope in SCOPES:
            vg,vr,vp=visual_contrast(index,requests,reference,stage,scope)
            qg,qr=quality_combination(numeric['quality_gates'],reference,stage,scope,requests)
            request_rows.extend(dict(r,claim=claim) for r in vr)
            parent_rows.extend(dict(r,claim=claim) for r in vp)
            geometry_rows.extend(dict(r,claim=claim) for r in qr)
            losses.extend(dict(r,claim=claim,claim_qualifier='Explicit body/support loss from an original J-positive output; disclosed, no extra hard gate')
                          for r in vr if r['explicit_body_or_support_loss']==1)
            result=dict(vg,**{k:v for k,v in qg.items() if k not in vg},claim=claim,
                        body_support_loss_request_ids=[r['request_id'] for r in vr if r['explicit_body_or_support_loss']==1],
                        task_losses_are_disclosed_not_an_additional_hard_gate=True)
            if claim=='G3_VS_G2_DESCRIPTIVE':
                result.pop('J_gate_status')
                result['interpretation']='Same locked intensity comparison; no independently tuned G2 or automatic direct-free-necessity claim'
                g2_rows.append(result)
            else:
                result['protocol_combined_status']=_combine(vg['J_gate_status'],qg['quality_gate_status'])
                result['interpretation']='Pre-registered research screening; no automatic default/model promotion'
                verdicts.append(result)
    drift,drift_summary=_R_drift(visual,requests,protected)
    protected.update(_book([P.OUT/'registration.json',P.REQUESTS,P.TASKS,P.OUT/'baseline.json',
                           E.__file__,source,OLD_R,Path(__file__).with_name('prepare.py')]))
    files=[]
    for name,rows in (('visual_gate_per_request.csv',request_rows),('visual_gate_per_parent.csv',parent_rows),
                      ('geometry_gate_combination.csv',geometry_rows),('task_losses.csv',losses),
                      ('G3_vs_G2_descriptive.csv',g2_rows),('R_label_drift_per_axis.csv',drift),
                      ('R_label_drift_summary.csv',drift_summary)):
        files.append(_write_csv(DEST/name,rows))
    result=dict(version=VERSION,visual_unique_cells=476,main_visual_cells=460,auxiliary_RAW_cells=16,
                confirmation_requests=38,confirmation_parents=19,adaptation_unseen_requests=14,adaptation_unseen_parents=7,
                verdicts=verdicts,G3_vs_G2_descriptive=g2_rows,
                R_drift=dict(outputs=92,axis_labels=460,changed_axes=sum(r['changed'] for r in drift),
                             token_only_changes=sum(r['token_only_change'] for r in drift),
                             canonical_visibility='YES/VISIBLE -> VISIBLE; NO_OBVIOUS/NO/NOT_CLEAR -> NOT_CLEAR; UNCERTAIN and UNKNOWN remain distinct. NO denotes no observed visible evidence, not proof of geometric absence.',
                             model_changed=False,changes_not_counted_as_method_gains=True),
                task_loss_note='Original J-positive body/support losses remain listed; J net gain already includes task losses. No extra veto was invented.',
                pending_reviews=0,automatic_promotion=False,new_model_calls=0,new_scoring_calls=0,new_render_calls=0)
    for name,data in (('protocol_verdicts.json',result),('calibration_label_reuse_audit.json',calibration)):
        path=DEST/name;P.write_immutable(path,data);files.append(path)
    lines=['# Frozen motif guidance: protocol screens','',
           'All 476 unique non-BASE observations were actually reviewed; the original 48 calibration labels are unchanged.',
           'J counts below are sums over requests. The geometric screen preserves the original registered I domain.','',
           '| Claim | Scope | J net | Qualifying parents | J gate | Geometry | Combined | Body/support losses |',
           '|---|---|---:|---:|---|---|---|---:|']
    for r in verdicts:
        lines.append('| '+ ' | '.join(map(str,(r['claim'],r['scope'],r['J_delta'],r['qualifying_parent_count'],
                     r['J_gate_status'],r['quality_gate_status'],r['protocol_combined_status'],r['explicit_body_or_support_loss'])))+' |')
    lines+=['','Each qualifying parent has positive own J count change and at least one strict new request.',
            'UNCERTAIN-to-positive changes are descriptive only. Explicit losses are reported without inventing another hard gate.',
            f"Historical R label drift: {result['R_drift']['changed_axes']}/460 axis labels changed across 92 unchanged outputs. This is rating drift, not a model gain.",
            'G3 versus G2 is a separate same-intensity descriptive comparison. A passing screen does not change the maintained default.','']
    readout=DEST/'READOUT.md';_write_text(readout,'\n'.join(lines));files.append(readout)
    _verify(protected)
    done=dict(status='COMPLETE',version=VERSION,source_sha256=P.sha(__file__),
              visual_unique_cells=476,main_visual_cells=460,calibration_labels_reused=48,pending_reviews=0,
              new_model_calls=0,new_scoring_calls=0,new_render_calls=0,protected=protected,files=_book(files))
    P.write_immutable(DEST/'complete.json',done)
    return done


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('build',))
    parser.parse_args()
    result=build()
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','protected','pending')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
