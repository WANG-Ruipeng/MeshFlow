"""Read-only analysis of frozen score tables; no new geometry/model calls."""
import collections, csv, json, math, statistics
from .common import OUT, LABEL, REFERENCE_LABEL, read, sha, write_new

METRICS = (
    'CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction',
    'free_minimum_angle_lt5_fraction','actual_free_faces',
    'deleted_free_faces','nonmanifold_edges','C_F_valid_edge_count',
    'C_F_valid_edge_total_length','free_only_reached_bulk_area_fraction',
    'legal_seam_bulk_edge_count','hinge_diag_isolated_1face_edge_components',
    'free_total_surface_area','fixed_parent_CF_illegal_affected_fraction',
    'fixed_parent_FF_illegal_affected_fraction')

def number(value):
    try:
        n=float(value)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError): return None

def macro(rows, field):
    groups=collections.defaultdict(list)
    for r in rows:
        n=number(r.get(field))
        if n is not None: groups[(r['parent_id'],r['condition_id'])].append(n)
    parents=collections.defaultdict(list)
    for (p,c),vals in groups.items(): parents[p].append(statistics.mean(vals))
    return statistics.mean(statistics.mean(x) for x in parents.values()) if parents else None

def groups(row):
    result=['ALL',row['eval_group']]
    if row['eval_group']=='CURRENT_TRAIN':
        result.append('NEW_PARENT_TRAIN' if row['parent_id'].startswith('27eb79') else 'OLD_CURRENT_TRAIN')
    return result

def support(rows, field):
    valid=[r for r in rows if number(r.get(field)) is not None]
    return dict(requests=len(valid),conditions=len({r['condition_id'] for r in valid}),
        parents=len({r['parent_id'] for r in valid}),request_ids=[r['request_id'] for r in valid])

def run():
    source=OUT/'results/per_output.csv'
    completion=read(OUT/'results/completion.json')
    assert completion['rows']==168
    assert any(p.replace('\\','/').endswith('/results/per_output.csv') and h==sha(source) for p,h in completion['files'].items())
    rows=list(csv.DictReader(source.open(encoding='utf-8-sig',newline='')))
    assert len(rows)==168 and len({(r['label'],r['stage'],r['request_id']) for r in rows})==168
    buckets=collections.defaultdict(list)
    for r in rows:
        for g in groups(r): buckets[(r['label'],r['stage'],g)].append(r)
    summaries=[]
    for (label,stage,group),rr in sorted(buckets.items()):
        passed=[r for r in rr if r.get('ConflictCov_C_status')=='PASS']
        # Descriptive interface coverage only; these are not required-interface success rates.
        applicable=[r for r in rr if r.get('seam_applicability')=='APPLICABLE']
        assert all(r.get('seam_applicability') in ('APPLICABLE','NA_CLOSED_C_NO_SEAM_INTERFACE') for r in rr)
        valid_reach=[r for r in applicable if number(r.get('free_only_reached_bulk_area_fraction')) is not None]
        valid_cf=[r for r in applicable if number(r.get('C_F_valid_edge_count')) is not None]
        summaries.append(dict(label=label,stage=stage,group=group,requests=len(rr),
            parents=len({r['parent_id'] for r in rr}),conditions=len({r['condition_id'] for r in rr}),
            parent_macro={m:macro(rr,m) for m in METRICS},
            parent_macro_support={m:support(rr,m) for m in METRICS},
            I_pass_parent_macro=macro(passed,'ConflictCov_C'),I_pass_support=support(passed,'ConflictCov_C'),
            I_status=dict(collections.Counter(r.get('ConflictCov_C_status','MISSING') for r in rr)),
            sums={m:sum(number(r.get(m)) or 0 for r in rr) for m in ('CF_illegal_pair_count','FF_illegal_pair_count','actual_free_faces','deleted_free_faces')},
            descriptive_open_C_requests=len(applicable),closed_C_NA=len(rr)-len(applicable),
            C_F_any=sum(number(r['C_F_valid_edge_count'])>0 for r in valid_cf),C_F_valid_requests=len(valid_cf),
            bulk_ge30=sum(number(r['free_only_reached_bulk_area_fraction'])>=.3 for r in valid_reach),
            bulk_reach_valid_requests=len(valid_reach),
            bulk_reach_NA_reasons=dict(collections.Counter(r.get('seam_to_bulk_status','MISSING') for r in applicable if r not in valid_reach))))
    pairs=[]
    for stage in sorted({r['stage'] for r in rows}):
        aa={r['request_id']:r for r in rows if r['stage']==stage and r['label']==REFERENCE_LABEL}
        bb={r['request_id']:r for r in rows if r['stage']==stage and r['label']==LABEL}
        assert aa.keys()==bb.keys()
        for group in sorted({g for a in aa.values() for g in groups(a)}):
            ids=[i for i,a in aa.items() if group in groups(a)]
            good=[i for i in ids if aa[i].get('ConflictCov_C_status')==bb[i].get('ConflictCov_C_status')=='PASS']
            pairs.append(dict(stage=stage,group=group,requests=len(ids),I_common_PASS_requests=len(good),
                I_common_PASS_support=support([aa[i] for i in good],'ConflictCov_C'),
                I_common_PASS_reference=macro([aa[i] for i in good],'ConflictCov_C'),
                I_common_PASS_candidate=macro([bb[i] for i in good],'ConflictCov_C'),
                transitions={m:dict(improved=sum(number(bb[i].get(m))<number(aa[i].get(m)) for i in ids if number(bb[i].get(m)) is not None and number(aa[i].get(m)) is not None),
                    worsened=sum(number(bb[i].get(m))>number(aa[i].get(m)) for i in ids if number(bb[i].get(m)) is not None and number(aa[i].get(m)) is not None))
                    for m in ('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','deleted_free_faces')}))
    value=dict(status='COMPLETE',source_sha256=sha(source),code_sha256=sha(__file__),
        aggregation='noise mean -> condition mean within parent -> parent equal mean; fixed groups; I PASS separately',
        note='Open boundary and reachability are descriptive, not required-interface or semantic success; new 27 R1 has all-Null source, R2 closed.',
        summaries=summaries,paired_diagnostics=pairs,model_calls=0,scoring_calls=0)
    write_new(OUT/'results/key_metrics_summary.json',value)
    print(json.dumps({'status':value['status'],'summary_groups':len(summaries)},indent=2))

if __name__=='__main__':run()
