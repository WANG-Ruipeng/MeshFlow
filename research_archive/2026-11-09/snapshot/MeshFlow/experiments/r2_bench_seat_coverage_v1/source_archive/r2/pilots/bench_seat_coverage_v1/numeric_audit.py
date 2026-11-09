"""Independent, read-only table arithmetic and existing GEO receipt audit."""
from __future__ import annotations
import collections
import json
import math
import statistics
from . import report_tables as R

OUT=R.OUT/'audits/numeric_crosscheck_v1'
META=('label','stage','eval_group')

def num(x):
    if x in ('True', 'true'): return 1.0
    if x in ('False', 'false'): return 0.0
    try: y=float(x)
    except (ValueError,TypeError): return None
    return y if math.isfinite(y) else None

def mean(values):
    vals=[v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None

def check_equal(a,b,context):
    if a is None or b is None:
        assert a is b, (context,a,b)
    else: assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(context,a,b)

def grouped(rows,keys):
    out=collections.defaultdict(list)
    for row in rows:out[tuple(row[k] for k in keys)].append(row)
    return out

def audit_aggregate(source,dest,keys,metrics):
    groups=grouped(source,keys)
    assert len(groups)==len(dest)
    cells=0
    for row in dest:
        identity=tuple(row[k] for k in keys); src=groups[identity]
        assert int(row['rows'])==len(src),(identity,'rows')
        for m in metrics:
            vals=[num(r[m]) for r in src]
            check_equal(mean(vals),num(row[m]),(identity,m))
            assert int(row[m+'__valid_n'])==sum(v is not None for v in vals),(identity,m,'valid_n')
            cells+=1
    return dict(rows=len(dest),metric_mean_cells=cells,valid_n_cells=cells,status='PASS')

def run():
    names=('per_output','per_condition','per_parent','parent_macro_summary')
    tables={n:R.csvread(R.RESULTS/(n+'.csv'))[0] for n in names}
    complete=R.read(R.RESULTS/'completion.json');metrics=complete['metric_fields']
    assert len(metrics)==177 and len(tables['per_output'])==252
    aggregate_audits={
        'output_to_condition':audit_aggregate(tables['per_output'],tables['per_condition'],META+('parent_id','condition_id'),metrics),
        'condition_to_parent':audit_aggregate(tables['per_condition'],tables['per_parent'],META+('parent_id',),metrics),
        'parent_to_macro':audit_aggregate(tables['per_parent'],tables['parent_macro_summary'],META,metrics)}
    rows=tables['per_output'];by={(r['label'],r['stage'],r['request_id']):r for r in rows}
    assert len(by)==252
    balance=[]
    for row in rows:
        total=int(row['N'])-int(row['K']);deleted=num(row['deleted_free_faces']);bridge=num(row['new_bridge_faces']);actual=num(row['actual_free_faces'])
        assert None not in (deleted,bridge,actual)
        check_equal(total-deleted+bridge,actual,(row['request_id'],row['label'],row['stage'],'balance'))
        check_equal(actual,num(row['free_face_denominator']),(row['request_id'],'current_free_denominator'))
        balance.append(dict(**{k:row[k] for k in ('label','stage','eval_group','parent_id','request_id')},requested_free=total,original_free_retained=total-deleted,deleted_original_free=deleted,new_bridge_faces=bridge,actual_free=actual,retained_original_free_fraction=(total-deleted)/total,free_total_surface_area=num(row['free_total_surface_area'])))
    ledger_path=R.RESULTS/'existing_geo_deletion_ledger/per_output.csv'
    ledger,_=R.csvread(ledger_path);assert len(ledger)==84
    for row in ledger:
        ref=by[row['label'],'BASE_POST',row['request_id']]
        for left,right in [('input_free_faces',None),('output_free_faces','actual_free_faces'),('deleted_free_faces','deleted_free_faces')]:
            expected=int(ref['N'])-int(ref['K']) if right is None else num(ref[right])
            check_equal(num(row[left]),expected,(row['label'],row['request_id'],left))
        check_equal(num(row['deleted_exact_repeated_vertex'])+num(row['deleted_duplicate_free']),num(row['deleted_free_faces']),(row['request_id'],'deletion_partition'))
    summaries=[]
    for identity,gr in sorted(grouped(rows,META).items()):
        p=next(p for p in tables['parent_macro_summary'] if tuple(p[k] for k in META)==identity)
        b=[x for x in balance if tuple(x[k] for k in META)==identity]
        rec=dict(zip(META,identity));rec.update(requests=len(gr),parents=len({r['parent_id'] for r in gr}),conditions=len({r['condition_id'] for r in gr}),I_status_counts=dict(collections.Counter(r['ConflictCov_C_status'] for r in gr)),seam_status_counts=dict(collections.Counter(r['seam_applicability'] for r in gr)),legal_edge_requests=sum(num(r['C_F_has_valid_edge'])==1 for r in gr),I_queries=sorted({r['ConflictCov_C_interior_samples'] for r in gr}),I_domains=sorted({r['domain'] for r in gr}),I_weight_denominator=sorted({r['C_query_weight_denominator'] for r in gr}),free_denominator_min=min(num(r['free_face_denominator']) for r in gr),free_denominator_max=max(num(r['free_face_denominator']) for r in gr))
        for m in R.CORE:rec[m]=num(p[m]);rec[m+'__valid_parents']=int(p[m+'__valid_n'])
        for m in ('requested_free','original_free_retained','deleted_original_free','new_bridge_faces','actual_free','retained_original_free_fraction'):
            rec[m]=mean([mean([num(r[m]) for r in rs]) for rs in grouped(b,('parent_id',)).values()])
        summaries.append(rec)
    ledger_summaries=[]
    for identity,gr in sorted(grouped(ledger,('label','eval_group')).items()):
        rec=dict(zip(('label','eval_group'),identity));rec['requests']=len(gr)
        for m in ('input_free_faces','output_free_faces','deleted_free_faces','deleted_exact_repeated_vertex','deleted_duplicate_free'):
            rec[m]=mean([mean([num(r[m]) for r in rs]) for rs in grouped(gr,('parent_id',)).values()])
        ledger_summaries.append(rec)
    fields=['label','stage','eval_group','parent_id','request_id','N','K']+list(R.CORE)
    new_detail=[{k:r[k] for k in fields} for r in rows if r['eval_group'] in ('NEW_TRAIN','NEW_PARENT_HOLDOUT')]
    core_parents=[{k:r[k] for k in ('label','stage','eval_group','parent_id')+R.CORE} for r in tables['per_parent']]
    files={'core_parent_macro.csv':summaries,'core_per_parent.csv':core_parents,'original_free_accounting_per_output.csv':balance,'new_cases_all_stages.csv':new_detail,'existing_geo_deletion_macro.csv':ledger_summaries}
    for name,data in files.items():R.csvwrite(OUT/name,data)
    receipt=dict(status='PASS',model_calls=0,scoring_calls=0,ratings_generated=0,original_numeric_fields=177,rows=252,aggregate_audits=aggregate_audits,face_balance_rows=252,existing_geo_receipts_matched=84,units=R.audit_old_units(),I_status_counts=dict(collections.Counter(r['ConflictCov_C_status'] for r in rows)),limitations=['Means remain request-within-condition then condition-within-parent then parent-equal.','Original numeric NA retained; Native seam metrics are not zero.','free_total_surface_area is current generated free area, not preserved source GT area and not a known subset of RAW area.','Original free retention counts requested slots minus recorded deleted slots, not semantic completeness.','Existing GEO repeated-vertex receipt records consolidated index collapse; it does not identify the upstream cause.','No current visual labels used or changed; no promotion decision.'],source_hashes={str(R.RESULTS/(n+'.csv')):R.sha(R.RESULTS/(n+'.csv')) for n in names}|{str(ledger_path):R.sha(ledger_path)},script_sha256=R.sha(__file__),output_hashes={n:R.sha(OUT/n) for n in files})
    R.immutable(OUT/'receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
