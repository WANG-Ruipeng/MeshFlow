"""Read closed evaluation tables only after the actual visual-review gate.

No scorer, model, rendering, bootstrap, metric selection or scientific verdict.
The outputs expose the registered arithmetic, all metric directions, missing
values, detector uncertainty, paired noises and both tails of every metric.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
OUT = ROOT / 'experiments/r2_user_data_scaling_32_67_v1'
EVAL = OUT / 'evaluation'
DEST = OUT / 'report_tables'
LABELS = ('USER_SMALL', 'USER_EXPANDED_67', 'HIST_USER_MIX_FROZEN')
COMPARISON = 'USER_EXPANDED_67-USER_SMALL'
PANELS = ('NEW32', 'OLD6')
STAGES = ('RAW', 'POST')
VERSION = 'r2_scaling_32_67_closed_table_report_v1'
TAIL_PARENTS = 5


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20), b''):h.update(block)
    return h.hexdigest()


def csvread(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def number(value):
    if value is None or value=='':return None
    if value in (True,'True','true'):return 1.
    if value in (False,'False','false'):return 0.
    try:v=float(value)
    except (TypeError,ValueError):return None
    return v if math.isfinite(v) else None


def average(values):
    values=[v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def direction(metric):
    """Directions follow source definitions, never the observed model ranking."""
    if metric=='C_preserved_exact':
        return 'HIGHER_HARD_CONTRACT','Exact C preservation; semantic use is a separate observation.'
    if metric.startswith(('A_VMR__','PMR__')):
        return 'HIGHER_MATCHING_DIAGNOSTIC','Free-only matching at the named fixed threshold; neither watertightness nor a true-join success rate.'
    if metric.startswith('boundary_distance_') and metric.endswith('_model'):
        return 'LOWER_PROXIMITY_DIAGNOSTIC','Boundary-to-free-vertex proximity only; no certification of shared edges or useful C adoption.'
    if metric in {'C_F_valid_edge_count','C_F_valid_edge_total_length','C_F_valid_opposite_winding_edge_count','C_F_has_valid_edge','C_F_has_opposite_winding_edge','reached_free_area_fraction','opposite_winding_reached_free_area_fraction'}:
        return 'HIGHER_CONNECTION_EVIDENCE','Legal graph connection/reachability evidence; more does not prove a complete seam, no intersection, or useful whole shape.'
    if metric in {'ConflictCov_C','ConflictCov_C_upper','ConflictCov_C_cross','ConflictCov_C_dup','ConflictCov_C_cross_and_dup','ConflictCov_C_coplanar_illegal_pairs','ConflictCov_C_proper_pairs'}:
        return 'LOWER_CONFLICT_DIAGNOSTIC','Original fixed-C conflict definition; lower bound and upper bound stay separate, with PARTIAL/LOW_SUPPORT retained.'
    if metric in {'ConflictCov_C_uncertain_fraction','ConflictCov_C_unresolved_pairs','ConflictCov_C_segment_extraction_failures','C_F_rejected_unknown_edge_count'} or metric.endswith('_unresolved_or_degenerate_pair_count'):
        return 'LOWER_DETECTOR_UNCERTAINTY','Numerical reduction of unresolved evidence, not an independently established geometry improvement.'
    if metric.startswith(('CF_','FF_')) and any(x in metric for x in ('proper_pair_count','coplanar_pair_count','illegal_pair_count','affected_free_fraction')):
        return 'LOWER_CONFLICT_DIAGNOSTIC','Original illegal/proper affected-free fraction or pair count; counts remain dependent on actual face count.'
    if metric in {'free_angle_p05_degrees'}:
        return 'HIGHER_ANGLE_DIAGNOSTIC','Triangle shape diagnostic; a necessary small angle is not automatically an unusable mesh.'
    if metric in {'free_minimum_angle_lt5_fraction','whole_minimum_angle_lt5_fraction','free_q_gt10_fraction','free_q_mean','free_near_degenerate_count','free_near_degenerate_fraction','free_exact_zero_area_count','free_exact_zero_area_fraction','whole_near_degenerate_count','whole_exact_zero_area_count','C_near_degenerate_count','C_exact_zero_area_count','free_exact_duplicate_excess_fraction','exact_duplicate_faces','nonmanifold_edges','C_F_nonmanifold_edge_count','C_F_rejected_duplicate_edge_count','C_F_rejected_illegal_edge_count','EXACT_COORD_nonmanifold_edges','EXACT_COORD_same_direction_two_face_edges'}:
        return 'LOWER_DEFECT_DIAGNOSTIC','Original shape/topology defect diagnostic; removing useful geometry can reduce it. RAW direction counts retain the canonical source convention.'
    return 'DESCRIPTIVE_NO_UNIVERSAL_ORDER','No monotone quality claim. Area, output count/deletion, components, raw contacts and support denominators are descriptive.'


def favorable(delta,metric):
    if delta is None:return 'NA'
    order=direction(metric)[0]
    if order.startswith('DESCRIPTIVE'):return 'NO_ORDER'
    if delta==0:return 'EQUAL'
    better=delta>0 if order.startswith('HIGHER') else delta<0
    return 'FAVORABLE_NUMERIC' if better else 'UNFAVORABLE_NUMERIC'


def slices(row):
    region=str(row['region_type']).upper();tier=str(row['tier']).upper()
    return [('ALL','ALL'),('region_type',region),('tier',tier),('condition_type',region+'_'+tier)]


def status_evidence(row,metric):
    result=[]
    if metric.startswith('ConflictCov_C'):
        result.append('ConflictCov_C_status='+str(row.get('ConflictCov_C_status','UNKNOWN_MISSING_STATUS')))
    if metric.startswith('boundary_distance_'):
        result.append('boundary_distance_status='+str(row.get('boundary_distance_status','UNKNOWN_MISSING_STATUS')))
    if metric=='C_preserved_exact':
        result.append('source_C_connectivity_status='+str(row.get('source_C_connectivity_status','UNKNOWN')))
    if number(row.get(metric)) is None:result.append('NA_NONNUMERIC_OR_MISSING='+str(row.get(metric,'')))
    return result


def uncertain(flags):
    return any(any(x in flag.upper() for x in ('PARTIAL','UNKNOWN','LOW_SUPPORT','UNCERTAIN','POSITIVE_INFINITY','NA_')) for flag in flags)


def review_gate():
    # Reuse the maintained receipt checker without invoking any controller stage.
    # This is called before opening a numeric result table.
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from r2.pilots.user_data_scaling_32_67_v1.resume_all import reviewed_record
    return reviewed_record()


def check_closed(reg,outputs,requests,conditions,parents,summary):
    if reg['labels']!=list(LABELS):raise ValueError('Unexpected registered methods')
    expected={(q['request_id'],label,stage) for q in reg['requests'] for label in LABELS for stage in STAGES}
    observed=[(q['request_id'],q['label'],q['stage']) for q in outputs]
    if len(observed)!=len(set(observed)) or set(observed)!=expected:
        raise ValueError('per_output is not a closed, unique table of the registered outputs')
    if set(q['panel'] for q in outputs)!=set(PANELS):raise ValueError('NEW32 and OLD6 must both remain registered')
    byid={q['request_id']:q for q in reg['requests']}
    for q in outputs:
        source=byid[q['request_id']]
        for k in ('parent_id','condition_id','panel','candidate','region_type','tier'):
            if str(q[k])!=str(source[k]):raise ValueError('Output metadata drift: '+k)
    metrics=sorted({r['metric'] for r in requests})
    comparisons={label+'-USER_SMALL' for label in LABELS[1:]}
    expected_pairs={(q['request_id'],stage,c,metric) for q in reg['requests'] for stage in STAGES for c in comparisons for metric in metrics}
    keys=[(q['request_id'],q['stage'],q['comparison'],q['metric']) for q in requests]
    if len(keys)!=len(set(keys)) or set(keys)!=expected_pairs:raise ValueError('Paired requests are not closed')
    for rows,fields in ((conditions,('comparison','panel','stage','parent_id','condition_id','metric')),
                        (parents,('comparison','panel','stage','parent_id','metric')),
                        (summary,('comparison','panel','stage','metric'))):
        keys=[tuple(x[k] for k in fields) for x in rows]
        expected_aggregate=set()
        for q in reg['requests']:
            for stage in STAGES:
                for comparison in comparisons:
                    for metric in metrics:
                        meta=dict(q,stage=stage,comparison=comparison,metric=metric)
                        expected_aggregate.add(tuple(meta[k] for k in fields))
        if len(keys)!=len(set(keys)) or set(keys)!=expected_aggregate:raise ValueError('Missing or duplicate aggregate rows')
    return metrics


def absolute_tables(outputs,metrics):
    groups=collections.defaultdict(list)
    for row in outputs:
        for stratum,value in slices(row):
            for metric in metrics:
                groups[(row['panel'],row['stage'],stratum,value,row['label'],row['parent_id'],row['condition_id'],metric)].append(row)
    condition_rows=[]
    for key,rows in sorted(groups.items()):
        panel,stage,stratum,value,label,parent,cid,metric=key
        nums=[number(r.get(metric)) for r in rows];flags=sorted(set(f for r in rows for f in status_evidence(r,metric)))
        complete=len(rows)==2 and {int(r['candidate']) for r in rows}=={0,1} and all(v is not None for v in nums)
        condition_rows.append(dict(panel=panel,stage=stage,stratum=stratum,stratum_value=value,label=label,parent_id=parent,
            condition_id=cid,metric=metric,value=average(nums),registered_noises=len(rows),finite_noises=sum(v is not None for v in nums),
            numeric_status='PASS' if complete else 'PARTIAL_OR_NA',metric_status_evidence=';'.join(flags),detector_uncertain=uncertain(flags)))
    grouped=collections.defaultdict(list)
    for row in condition_rows:
        grouped[tuple(row[k] for k in ('panel','stage','stratum','stratum_value','label','parent_id','metric'))].append(row)
    parent_rows=[]
    for key,rows in sorted(grouped.items()):
        keep=[r for r in rows if r['numeric_status']=='PASS']
        row=dict(zip(('panel','stage','stratum','stratum_value','label','parent_id','metric'),key))
        row.update(value=average([r['value'] for r in keep]),registered_conditions=len(rows),valid_conditions=len(keep),
            numeric_status='PASS' if len(rows)==len(keep) else 'PARTIAL_OR_NA',detector_uncertain=any(r['detector_uncertain'] for r in rows),
            metric_status_evidence=';'.join(sorted({r['metric_status_evidence'] for r in rows})))
        parent_rows.append(row)
    groups=collections.defaultdict(list)
    for row in parent_rows:groups[tuple(row[k] for k in ('panel','stage','stratum','stratum_value','label','metric'))].append(row)
    means=[]
    for key,rows in sorted(groups.items()):
        good=[r for r in rows if r['value'] is not None];vals=[r['value'] for r in good]
        row=dict(zip(('panel','stage','stratum','stratum_value','label','metric'),key));order,note=direction(row['metric'])
        row.update(parent_equal_mean=average(vals),parent_median=statistics.median(vals) if vals else None,
            registered_parents=len(rows),finite_parents=len(good),numerically_complete_parents=sum(r['numeric_status']=='PASS' for r in rows),
            detector_uncertain_parents=sum(r['detector_uncertain'] for r in rows),
            status='PASS' if len(good)==len(rows) and all(r['numeric_status']=='PASS' for r in rows) else 'PARTIAL_OR_NA',
            direction=order,interpretation=note,aggregation='original hierarchy: mean finite noises; parent mean uses numerically complete conditions; equal finite parent means; missing counts retained')
        means.append(row)
    return condition_rows,parent_rows,means


def sign_pattern(values):
    if len(values)!=2 or any(v is None for v in values):return 'NA_OR_MISSING_NOISE'
    a,b=values
    if a==b==0:return 'BOTH_ZERO'
    if a>0 and b>0:return 'BOTH_POSITIVE'
    if a<0 and b<0:return 'BOTH_NEGATIVE'
    if a==0 or b==0:return 'ONE_ZERO'
    return 'OPPOSITE_SIGNS'


def noise_tables(requests,output_lookup):
    groups=collections.defaultdict(list)
    for r in requests:
        if r['comparison']==COMPARISON:
            groups[tuple(r[k] for k in ('panel','stage','parent_id','condition_id','region_type','tier','metric'))].append(r)
    conditions=[];parent_noise=collections.defaultdict(list)
    for key,rr in sorted(groups.items()):
        row=dict(zip(('panel','stage','parent_id','condition_id','region_type','tier','metric'),key));rr=sorted(rr,key=lambda x:int(x['candidate']))
        values=[number(r['delta']) if r['status']=='PASS' else None for r in rr]
        if [int(x['candidate']) for x in rr]!=[0,1]:raise ValueError('Registered two-noise pairing differs')
        flags=[]
        for r in rr:
            for label in ('USER_SMALL','USER_EXPANDED_67'):
                flags.extend(status_evidence(output_lookup[(r['request_id'],label,r['stage'])],row['metric']))
        pattern=sign_pattern(values);order=direction(row['metric'])[0]
        row.update(comparison=COMPARISON,noise0_delta=values[0],noise1_delta=values[1],noise_pattern=pattern,
            direction=order,noise0_direction=favorable(values[0],row['metric']),noise1_direction=favorable(values[1],row['metric']),
            numeric_status='PASS' if all(v is not None for v in values) else 'NA_OR_MISSING',
            detector_uncertain=uncertain(flags),metric_status_evidence=';'.join(sorted(set(flags))))
        conditions.append(row)
        for stratum,value in slices(row):
            parent_noise[(row['panel'],row['stage'],stratum,value,row['parent_id'],row['metric'])].append(row)
    parents=[]
    for key,rr in sorted(parent_noise.items()):
        row=dict(zip(('panel','stage','stratum','stratum_value','parent_id','metric'),key))
        values=[average([r['noise'+str(i)+'_delta'] for r in rr]) if all(r['noise'+str(i)+'_delta'] is not None for r in rr) else None for i in (0,1)]
        row.update(comparison=COMPARISON,noise0_parent_delta=values[0],noise1_parent_delta=values[1],noise_pattern=sign_pattern(values),
            registered_conditions=len(rr),direction=direction(row['metric'])[0],detector_uncertain=any(r['detector_uncertain'] for r in rr),
            note='Within each noise index average registered C of this parent; any missing C keeps this noise NA. Two noises are not two training seeds.')
        parents.append(row)
    return conditions,parents


def paired_tables(requests,original_parents,original_summary,output_lookup):
    # ALL-panel parent values are read directly from the original paired table.
    parents=[dict(r,stratum='ALL',stratum_value='ALL') for r in original_parents if r['comparison']==COMPARISON]
    groups=collections.defaultdict(list)
    for r in requests:
        if r['comparison']!=COMPARISON:continue
        for stratum,value in slices(r)[1:]:
            groups[(r['panel'],r['stage'],stratum,value,r['parent_id'],r['condition_id'],r['metric'])].append(r)
    condition_rows=[]
    for key,rr in sorted(groups.items()):
        row=dict(zip(('panel','stage','stratum','stratum_value','parent_id','condition_id','metric'),key))
        good=[r for r in rr if r['status']=='PASS' and number(r['delta']) is not None]
        row.update(comparison=COMPARISON,status='PASS' if len(good)==len(rr) else 'PARTIAL_OR_NA',n_registered=len(rr),n_valid=len(good))
        for field in ('baseline_value','candidate_value','delta'):row[field]=average([number(r[field]) for r in good])
        condition_rows.append(row)
    groups=collections.defaultdict(list)
    for r in condition_rows:groups[tuple(r[k] for k in ('panel','stage','stratum','stratum_value','parent_id','metric'))].append(r)
    for key,rr in sorted(groups.items()):
        row=dict(zip(('panel','stage','stratum','stratum_value','parent_id','metric'),key));good=[r for r in rr if r['status']=='PASS']
        row.update(comparison=COMPARISON,status='PASS' if len(good)==len(rr) else 'PARTIAL_OR_NA',n_registered=len(rr),n_valid=len(good))
        for field in ('baseline_value','candidate_value','delta'):row[field]=average([r[field] for r in good])
        parents.append(row)
    # Preserve I / boundary uncertainty even when the original numeric pair is finite.
    request_status=collections.defaultdict(list)
    for r in requests:
        if r['comparison']!=COMPARISON:continue
        for stratum,value in slices(r):
            key=(r['panel'],r['stage'],stratum,value,r['parent_id'],r['metric'])
            for label in ('USER_SMALL','USER_EXPANDED_67'):
                request_status[key].extend(status_evidence(output_lookup[(r['request_id'],label,r['stage'])],r['metric']))
    groups=collections.defaultdict(list)
    for row in parents:
        for field in ('baseline_value','candidate_value','delta'):row[field]=number(row[field])
        k=tuple(row[x] for x in ('panel','stage','stratum','stratum_value','parent_id','metric'));flags=request_status[k]
        row.update(direction=direction(row['metric'])[0],numeric_direction=favorable(row['delta'],row['metric']),
            detector_uncertain=uncertain(flags),metric_status_evidence=';'.join(sorted(set(flags))))
        groups[tuple(row[x] for x in ('panel','stage','stratum','stratum_value','metric'))].append(row)
    summary=[];tails=[]
    authoritative={(r['panel'],r['stage'],r['metric']):r for r in original_summary if r['comparison']==COMPARISON}
    for key,rr in sorted(groups.items()):
        row=dict(zip(('panel','stage','stratum','stratum_value','metric'),key));good=[r for r in rr if r['delta'] is not None];d=[r['delta'] for r in good]
        order,note=direction(row['metric'])
        row.update(comparison=COMPARISON,direction=order,interpretation=note,registered_parents=len(rr),finite_parents=len(good),
            numerically_complete_parents=sum(r['status']=='PASS' for r in rr),detector_uncertain_parents=sum(r['detector_uncertain'] for r in rr),
            mean_delta=average(d),median_parent_delta=statistics.median(d) if d else None,
            negative_delta_parents=sum(v<0 for v in d),equal_delta_parents=sum(v==0 for v in d),positive_delta_parents=sum(v>0 for v in d),
            favorable_numeric_parents=sum(favorable(v,row['metric'])=='FAVORABLE_NUMERIC' for v in d),
            unfavorable_numeric_parents=sum(favorable(v,row['metric'])=='UNFAVORABLE_NUMERIC' for v in d),
            minimum_delta=min(d) if d else None,maximum_delta=max(d) if d else None,
            status='PASS' if len(good)==len(rr) and all(r['status']=='PASS' for r in rr) else 'PARTIAL_OR_NA')
        if row['stratum']=='ALL':
            old=authoritative[(row['panel'],row['stage'],row['metric'])]
            for newkey,oldkey in (('mean_delta','mean_delta'),('median_parent_delta','median_delta')):
                a,b=row[newkey],number(old[oldkey])
                if (a is None)!=(b is None) or a is not None and not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Original paired summary disagrees')
        summary.append(row)
        for tail,reverse in (('LOWEST_DELTA',False),('HIGHEST_DELTA',True)):
            ranked=sorted(good,key=lambda r:(-r['delta'] if reverse else r['delta'],r['parent_id']))[:TAIL_PARENTS]
            is_worst=(order.startswith('HIGHER') and not reverse) or (order.startswith('LOWER') and reverse)
            for rank,r in enumerate(ranked,1):
                tails.append(dict(r,tail=tail,rank=rank,worst_numeric_direction=is_worst,
                    note='Both tails saved for every metric; descriptive metrics have no declared worst direction.'))
    return parents,summary,tails


def counts_table(rows,group_fields,value_field):
    counts=collections.Counter(tuple(r.get(k,'UNKNOWN') for k in group_fields)+(str(r.get(value_field,'UNKNOWN')), ) for r in rows)
    return [dict(zip((*group_fields,value_field),key),count=count) for key,count in sorted(counts.items())]


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def fmt(value):
    if value is None:return 'NA'
    return format(value,'.7g') if isinstance(value,(int,float)) else str(value).replace('|','\\|')


def markdown(means,paired,review):
    lookup={(r['panel'],r['stage'],r['label'],r['metric']):r for r in means if r['stratum']=='ALL'}
    text=['# Closed evaluation tables: all registered metrics','',
        'This appendix is generated after actual visual-review receipt verification. It contains no method promotion or causal verdict.',
        'One nested split and one training stream; two noises are paired samples, not independent training seeds. NEW32 and OLD6 remain separate.',
        'Absolute values are in each original metric unit. Fractions remain 0–1; native deltas are not silently converted to percentage points.',
        'Finite parent means retain valid/registered counts. Detector PARTIAL/LOW_SUPPORT/UNKNOWN evidence remains in CSV fields and is not certified away by a numeric mean.',
        'Graph/vertex metrics are diagnostic evidence, not proof that C was usefully adopted or that the body is complete.',
        'Actual visual review status: '+review['status']+'; rows viewed: '+str(review['actually_reviewed'])+'.','']
    for panel in PANELS:
        for stage in STAGES:
            text += ['## '+panel+' / '+stage,'','| Metric | Direction | SMALL | EXPANDED67 | HIST | E−S mean | Parent median Δ | − / 0 / + parents | finite / registered | detector-uncertain parents |',
                     '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for r in sorted((x for x in paired if x['panel']==panel and x['stage']==stage and x['stratum']=='ALL'),key=lambda x:x['metric']):
                vals=[lookup[(panel,stage,label,r['metric'])]['parent_equal_mean'] for label in LABELS]
                fields=[r['metric'],r['direction'],*vals,r['mean_delta'],r['median_parent_delta'],
                    str(r['negative_delta_parents'])+' / '+str(r['equal_delta_parents'])+' / '+str(r['positive_delta_parents']),
                    str(r['finite_parents'])+' / '+str(r['registered_parents']),r['detector_uncertain_parents']]
                text.append('| '+' | '.join(fmt(x) for x in fields)+' |')
            text.append('')
    text += ['Full seat/back, original tier and combined condition-type slices are in CSVs. OLD6 retains its historical tier names; a historical medium tier is not silently renamed HALF.',
        'Both five-parent delta tails are retained for every metric; only an explicitly directional metric identifies a numerical unfavorable tail. No bootstrap, new metric, favorable-three selection or source-GT success ranking is performed.','']
    return '\n'.join(text)


def run():
    review=review_gate()  # Must execute before even opening a numeric result file.
    names=('per_output.csv','quality_per_output.csv','paired_per_request.csv','paired_per_condition.csv','paired_per_parent.csv','paired_summary.csv')
    sources={str(EVAL/name):digest(EVAL/name) for name in names}
    reg=read(EVAL/'registration.json');done=read(EVAL/'aggregate_completion.json');scored=read(EVAL/'score_completion.json')
    if done.get('status')!='COMPLETE' or scored.get('status')!='COMPLETE' or done.get('stage_records')!=840:
        raise ValueError('Report requires the closed original aggregate/score receipts, not a running partial output stream')
    identity=dict(version=VERSION,code_sha256=digest(__file__),tables=sources,
        evaluation_registration_sha256=digest(EVAL/'registration.json'),review_receipt_sha256=digest(EVAL/'gallery/output_review_receipt.json'))
    receipt=DEST/'receipt.json'
    if receipt.exists():
        prior=read(receipt)
        if prior['identity']!=identity:raise ValueError('Existing report identity differs; preserve it and use explicit engineering review')
        for path,h in prior['outputs'].items():
            if digest(path)!=h:raise ValueError('Report output changed')
        return dict(status='VERIFIED_REUSE',receipt=str(receipt))
    if DEST.exists():raise RuntimeError('Incomplete report packaging; no overwrite or silent retry')
    outputs=csvread(EVAL/'per_output.csv');requests=csvread(EVAL/'paired_per_request.csv')
    condition_pairs=csvread(EVAL/'paired_per_condition.csv');parent_pairs=csvread(EVAL/'paired_per_parent.csv');original_summary=csvread(EVAL/'paired_summary.csv')
    metrics=check_closed(reg,outputs,requests,condition_pairs,parent_pairs,original_summary)
    ac,ap,am=absolute_tables(outputs,metrics);lookup={(r['request_id'],r['label'],r['stage']):r for r in outputs}
    nc,np=noise_tables(requests,lookup);pp,ps,tails=paired_tables(requests,parent_pairs,original_summary,lookup)
    status_rows=[]
    for r in outputs:
        for field,value in r.items():
            if field.endswith('_status'):
                status_rows.append(dict(panel=r['panel'],stage=r['stage'],label=r['label'],field=field,value=value or 'MISSING'))
    statuses=counts_table(status_rows,('panel','stage','label','field'),'value')
    noise_counts=counts_table(np,('panel','stage','stratum','stratum_value','metric'),'noise_pattern')
    directions=[dict(metric=m,direction=direction(m)[0],interpretation=direction(m)[1],unit='original; ratios remain 0..1; no automatic percentage conversion') for m in metrics]
    DEST.mkdir(parents=True,exist_ok=False)
    products={'absolute_per_condition.csv':ac,'absolute_per_parent.csv':ap,'absolute_parent_equal.csv':am,
        'expanded_minus_small_per_parent.csv':pp,'expanded_minus_small_summary.csv':ps,
        'expanded_minus_small_both_tails.csv':tails,'two_noise_per_condition.csv':nc,'two_noise_per_parent.csv':np,
        'two_noise_counts.csv':noise_counts,'categorical_status_counts.csv':statuses,'metric_directions.csv':directions}
    for name,rows in products.items():write_csv(DEST/name,rows)
    (DEST/'REPORT_TABLES.md').write_text(markdown(am,ps,review),encoding='utf8')
    result=dict(status='COMPLETE_TABLES_ONLY_NO_SCIENTIFIC_VERDICT',identity=identity,registered_metrics=len(metrics),
        source_stage_records=len(outputs),actual_review_status=review['status'],actual_reviewed=review['actually_reviewed'],
        outputs={str(p):digest(p) for p in DEST.iterdir() if p.is_file()},model_calls=0,new_scores=0,new_bootstrap=0,
        caveats=['All metrics retained; no favorable selection.','Detector uncertainty and missing numeric values are distinct.','Historic USER_MIX is a frozen reference, not a strict data-size causal arm.'])
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8')
    return dict(status=result['status'],receipt=str(receipt),metrics=len(metrics))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    print(json.dumps(run(),ensure_ascii=False,indent=2))
