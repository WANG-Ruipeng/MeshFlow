"""Portable stdlib-only arithmetic on this package. Never import original scorers.

Run: python protocol_and_code/recompute.py --out /path/to/new_derived_directory
Reads only package CSVs, writes only the requested derived directory.
"""
import argparse, csv, json, math, statistics
from collections import defaultdict, Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
S='USER_SMALL';E='USER_EXPANDED_67';H='HIST_USER_MIX_FROZEN'
FIELDS=['C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer','whole_body_complete','support_missing','fragments_or_folded_layers','visible_join']
STATES=['YES','NO','UNCERTAIN']
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def number(v):
    if v=='True':return 1.
    if v=='False':return 0.
    try:x=float(v)
    except (TypeError,ValueError):return None
    return x if math.isfinite(x) else None
def mean(v):
    v=[x for x in v if x is not None]
    return sum(v)/len(v) if v else None
def key(r):return r['request_id'],r.get('label',r.get('model_label')),r['stage']
def write(out,name,rr,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rr for k in r))
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rr)

def run(out,metrics_path=None):
    out.mkdir(parents=True,exist_ok=True)
    mr=rows(metrics_path or ROOT/'metrics_per_output.csv');vr=rows(ROOT/'visual_review_840.csv')
    metrics=[r['metric'] for r in rows(ROOT/'protocol_and_code/registered_metric_fields.csv')]
    ref=rows(ROOT/'protocol_and_code/original_aggregate_reference.csv')
    mi={key(r):r for r in mr};vi={key(r):r for r in vr}
    assert len(mi)==len(mr)==840 and set(mi)==set(vi)
    assert len(set(metrics))==96 and all(all(m in r for m in metrics) for r in mr)
    aggregates=[];maxdiff=0.;mismatch=[]
    for rr in ref:
        selected=[r for r in mr if all(r[k]==rr[k] for k in ('panel','stage','label'))]
        cm=defaultdict(list)
        for r in selected:cm[r['parent_id'],r['condition_id']].append(number(r[rr['metric']]))
        pm=defaultdict(list)
        for (p,c),vs in sorted(cm.items()):
            # Original numerical contract: preserve missing counts; use complete conditions in parent means.
            if all(v is not None for v in vs):pm[p].append(mean(vs))
        pv=[mean(v) for p,v in sorted(pm.items())]
        computed=mean(pv);original=number(rr['parent_equal_mean'])
        diff=computed-original if computed is not None and original is not None else None
        status='MATCH' if diff is not None and abs(diff)<=1e-12 else 'MISSING_OR_DIFFERENT'
        if diff is not None:maxdiff=max(maxdiff,abs(diff))
        result=dict(kind='DERIVED',panel=rr['panel'],stage=rr['stage'],label=rr['label'],metric=rr['metric'],
                    original_parent_equal_mean=rr['parent_equal_mean'],recomputed_parent_equal_mean=computed,
                    delta=diff,status=status,original_parent_median=rr['parent_median'],
                    recomputed_parent_median=statistics.median(pv) if pv else None,
                    registered_outputs=len(selected),valid_parents=len(pv),registered_parents=rr['registered_parents'])
        aggregates.append(result)
        if status!='MATCH':mismatch.append(result)
    write(out,'aggregate_recomputed.csv',aggregates)
    base=sorted([r for r in mr if r['panel']=='NEW32' and r['label']==S and r['stage']=='POST'],key=lambda r:r['request_id'])
    assert len(base)==128
    pairs=[];transition_ids={};tr=[];unmatched=[];half=[]
    for r in base:
        req=r['request_id'];sv=vi.get((req,S,'POST'));ev=vi.get((req,E,'POST'))
        er=mi.get((req,E,'POST'))
        if sv is None or ev is None or er is None:
            unmatched.append(dict(kind='DERIVED',request_id=req,reason='MISSING_METRIC_OR_VISUAL_PAIR'));continue
        assert all(r[k]==er[k] for k in ('parent_id','condition_id','candidate','region_type','tier'))
        cat=('BOTH' if number(r['C_F_has_valid_edge']) else 'GAINED') if number(er['C_F_has_valid_edge']) else ('LOST' if number(r['C_F_has_valid_edge']) else 'NEITHER')
        pair=dict(kind='DERIVED',request_id=req,parent_uid=r['parent_id'],condition_id=r['condition_id'],noise_id=r['candidate'],condition_type=r['region_type']+'_'+r['tier'],interface_transition=cat)
        for lab,rr,v in [('SMALL',r,sv),('EXP',er,ev)]:
            for f in ['C_F_has_valid_edge','C_F_valid_edge_count','C_F_valid_edge_total_length','C_F_valid_opposite_winding_edge_count','opposite_winding_reached_free_area_fraction']:
                pair[lab+'__'+f]=rr[f]
            pair[lab+'__C_adoption']=v['C_adoption']
        pairs.append(pair)
        if pair['condition_type']=='SEAT_HALF':
            hp={k:pair[k] for k in ('kind','request_id','parent_uid','condition_id','noise_id','condition_type')}
            for f in FIELDS:hp['SMALL__'+f]=sv[f];hp['EXP__'+f]=ev[f]
            half.append(hp)
    write(out,'interface_pairs_NEW32_POST.csv',pairs)
    write(out,'SEAT_HALF_all30_visual_pairs.csv',half)
    for group in ['ALL','SEAT_FULL','SEAT_HALF','BACK_FULL','BACK_HALF']:
        selected=[r for r in base if group=='ALL' or r['region_type']+'_'+r['tier']==group]
        for f in FIELDS:
            for a in STATES:
                for b in STATES:
                    ids=[r['request_id'] for r in selected if vi.get((r['request_id'],S,'POST'),{}).get(f)==a and vi.get((r['request_id'],E,'POST'),{}).get(f)==b]
                    k='|'.join([group,f,a,b]);transition_ids[k]=ids
                    tr.append(dict(kind='DERIVED',panel='NEW32',stage='POST',condition_type=group,field=f,SMALL_state=a,EXP_state=b,count=len(ids),registered_pairs=len(selected),request_list_key=k))
            for r in selected:
                a=vi.get((r['request_id'],S,'POST'),{}).get(f,'MISSING');b=vi.get((r['request_id'],E,'POST'),{}).get(f,'MISSING')
                if a not in STATES or b not in STATES:unmatched.append(dict(kind='DERIVED',request_id=r['request_id'],reason=f'{group}|{f}|SMALL={a}|EXP={b}'))
    write(out,'visual_transitions.csv',tr)
    (out/'visual_transition_requests.json').write_text(json.dumps(dict(kind='DERIVED',request_lists=transition_ids),ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    write(out,'unmatched_or_non3state.csv',unmatched,['kind','request_id','reason'])
    components={'I':'ConflictCov_C','cross':'ConflictCov_C_cross','dup':'ConflictCov_C_dup','intersection':'ConflictCov_C_cross_and_dup'}
    shifts=[];residual=[]
    for r in base:
        req=r['request_id'];d=dict(kind='DERIVED',request_id=req,parent_uid=r['parent_id'],condition_id=r['condition_id'],noise_id=r['candidate'],condition_type=r['region_type']+'_'+r['tier'],source_quality=r['source_quality'])
        for comp,field in components.items():
            for stage in ['RAW','POST']:
                ss=mi[req,S,stage][field];ee=mi[req,E,stage][field]
                d[f'SMALL_{stage}_{comp}']=ss;d[f'EXP_{stage}_{comp}']=ee
                d[f'delta_{stage}_{comp}']=number(ee)-number(ss)
            d['shift_'+comp]=d['delta_POST_'+comp]-d['delta_RAW_'+comp]
            d['macro_shift_contribution_'+comp]=d['shift_'+comp]/128
        d['decomposition_residual']=d['shift_I']-(d['shift_cross']+d['shift_dup']-d['shift_intersection'])
        d['delta_I_negative_to_positive']=d['delta_RAW_I']<0<d['delta_POST_I']
        residual.append(abs(d['decomposition_residual']));shifts.append(d)
    write(out,'I_stage_shift_per_request.csv',shifts)
    ps=[];quality=[]
    for parent in sorted({r['parent_uid'] for r in shifts}):
        rr=[r for r in shifts if r['parent_uid']==parent]
        conds=sorted({r['condition_id'] for r in rr})
        tags={c:sorted({r['source_quality'] for r in rr if r['condition_id']==c}) for c in conds}
        labels=sorted({t for ts in tags.values() for t in ts})
        d=dict(kind='DERIVED',parent_uid=parent,requests=len(rr),conditions=len(conds))
        for k in [k for k in rr[0] if k.startswith(('delta_','shift_','macro_shift_')) and k!='delta_I_negative_to_positive']:
            # Explicit two-level mean rather than treating condition count as new parents.
            d[k]=mean([mean([x[k] for x in rr if x['condition_id']==c]) for c in conds])
        for comp in components:d['macro_shift_contribution_'+comp]=d['shift_'+comp]/32
        d['delta_I_negative_to_positive']=d['delta_RAW_I']<0<d['delta_POST_I'];ps.append(d)
        quality.append(dict(kind='DERIVED',parent_uid=parent,condition_ids=json.dumps(conds,separators=(',',':')),condition_quality_labels=json.dumps(tags,separators=(',',':')),
                            distinct_quality_labels=json.dumps(labels,separators=(',',':')),parent_quality=(labels[0] if len(labels)==1 else 'MIXED'),
                            rule='LABEL_ONLY_IF_ALL_SAVED_CONDITION_LABELS_AGREE; NOT_MAX_SEVERITY',conditions=len(conds),
                            delta_RAW_I=d['delta_RAW_I'],delta_POST_I=d['delta_POST_I'],shift_I=d['shift_I']))
    write(out,'I_stage_shift_per_parent.csv',ps)
    write(out,'dev_quality_parent_check.csv',quality)
    interface_counts=dict(Counter(r['interface_transition'] for r in pairs))
    summary=dict(kind='DERIVED',rows=len(mr),registered_metrics=len(metrics),aggregate_comparisons=len(aggregates),aggregate_mismatches=len(mismatch),
                 maximum_absolute_aggregate_difference=maxdiff,interface_counts=interface_counts,
                 interface_transition_requests={c:[r['request_id'] for r in pairs if r['interface_transition']==c] for c in ['BOTH','LOST','GAINED']},
                 parent_64871dc2=[r for r in pairs if r['parent_uid'].startswith('64871dc2')],seat_half_pairs=len(half),
                 unmatched_or_non3state=len(unmatched),I_macro={k:mean([p[k] for p in ps]) for k in ps[0] if k.startswith(('delta_','shift_')) and k!='delta_I_negative_to_positive'},
                 max_I_decomposition_residual=max(residual),I_positive_shift_parents=sum(r['shift_I']>0 for r in ps),
                 I_parent_negative_to_positive=[r['parent_uid'] for r in ps if r['delta_I_negative_to_positive']],
                 I_top_positive_shift=sorted(ps,key=lambda r:r['shift_I'],reverse=True)[:5],
                 DEV_quality_counts=dict(Counter(r['parent_quality'] for r in quality)),
                 unknown_parents=[r for r in quality if r['parent_quality']=='UNKNOWN'],
                 original_tables_modified=False,model_or_geometry_calls=0)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('aggregate_comparisons','aggregate_mismatches','maximum_absolute_aggregate_difference','interface_counts','seat_half_pairs','DEV_quality_counts','unknown_parents')}))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--metrics',type=Path);args=ap.parse_args();run(args.out.resolve(),args.metrics)
