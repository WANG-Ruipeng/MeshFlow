"""Read-only source-cache aggregation for S2000 versus independent HINGE endpoints.

Run with an existing Python 3 interpreter. No torch/model/scorer imports.
All writes are confined to this script's directory. Source values stay unchanged.
"""
import csv
import hashlib
import html
import json
import math
import os
import statistics as st
import time
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SRC = ROOT / 'experiments/r2_seam_hinge_quality_v1'
STAGES = ['POST', 'POST_HINGE015', 'POST_HINGE025']
META = {'request_id','condition_id','parent_id','noise_index','N','K','split','label','stage','mesh_path','domain','topology_definition_version'}
TOL = 1e-12

def readcsv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def writecsv(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)

def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def number(v):
    if str(v).lower() in ('true','false'):
        return float(str(v).lower() == 'true')
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (ValueError,TypeError):
        return None

def quantile(a, p):
    a = sorted(a)
    if not a:
        return None
    x = (len(a)-1)*p
    lo, hi = math.floor(x), math.ceil(x)
    return a[lo] + (a[hi]-a[lo])*(x-lo)

LOW = set('''CF_illegal_pair_count CF_illegal_affected_free_fraction CF_proper_pair_count CF_coplanar_pair_count CF_proper_affected_free_fraction CF_unresolved_or_degenerate_pair_count FF_illegal_pair_count FF_illegal_affected_free_fraction FF_proper_pair_count FF_coplanar_pair_count FF_proper_affected_free_fraction FF_unresolved_or_degenerate_pair_count free_minimum_angle_lt5_fraction free_q_gt10_fraction free_q_mean free_near_degenerate_count free_near_degenerate_fraction free_exact_zero_area_count free_exact_zero_area_fraction ConflictCov_C ConflictCov_C_upper ConflictCov_C_cross ConflictCov_C_dup ConflictCov_C_uncertain_fraction ConflictCov_C_unresolved_pairs ConflictCov_C_coplanar_illegal_pairs ConflictCov_C_proper_pairs ConflictCov_C_segment_extraction_failures whole_minimum_angle_lt5_fraction whole_q_gt10_fraction whole_near_degenerate_count whole_exact_zero_area_count free_exact_duplicate_excess_fraction exact_duplicate_faces nonmanifold_edges C_F_nonmanifold_edge_count C_F_rejected_duplicate_edge_count C_F_rejected_illegal_edge_count C_F_rejected_unknown_edge_count boundary_distance_mean_model boundary_distance_median_model boundary_distance_p95_model boundary_distance_max_model hinge_diag_isolated_1face_edge_components hinge_diag_isolated_1face_unfiltered_edge_components'''.split())
HIGH = set('''free_angle_p05_degrees C_preserved_exact C_F_valid_edge_count C_F_has_valid_edge C_F_valid_edge_total_length C_F_valid_opposite_winding_edge_count C_F_has_opposite_winding_edge graph_valid_edges reached_free_faces reached_free_area reached_free_area_fraction opposite_winding_reached_free_faces opposite_winding_reached_free_area opposite_winding_reached_free_area_fraction legal_CF_seed_seam_faces legal_CF_seed_seam_area free_only_reached_seam_faces free_only_reached_bulk_faces free_only_reached_bulk_area free_only_reached_bulk_area_fraction legal_seam_bulk_edge_count legal_seam_bulk_edge_length seed_connected_legal_seam_bulk_edge_count seed_connected_legal_seam_bulk_edge_length bulk_reach_ge_0_30_descriptive hinge_diag_bulk_only_valid_shared_edge_count hinge_diag_legal_edge_C_reached_free_area_fraction hinge_diag_legal_opposed_edge_C_reached_free_area_fraction'''.split())
LOW.update({'EXACT_COORD_same_direction_two_face_edges', 'EXACT_COORD_nonmanifold_edges'})
KEY = ['A_VMR__raw_tau_1e-6','PMR__raw_tau_1e-6','A_VMR__L_tau_0.01','PMR__L_tau_0.01','C_F_valid_edge_count','C_F_valid_edge_total_length','free_only_reached_bulk_area_fraction','bulk_reach_ge_0_30_descriptive','CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','ConflictCov_C_cross','ConflictCov_C_dup','free_minimum_angle_lt5_fraction','free_angle_p05_degrees','free_q_gt10_fraction','free_near_degenerate_fraction','free_exact_zero_area_fraction','free_exact_duplicate_excess_fraction','nonmanifold_edges','hinge_diag_isolated_1face_unfiltered_edge_components','hinge_diag_vertex_attached_only_unfiltered_free_area_fraction','hinge_diag_fully_C_disconnected_free_area_fraction','hinge_diag_bulk_only_valid_shared_edge_count','boundary_edges','actual_free_faces','deleted_free_faces','C_preserved_exact']
KEY.extend(['free_q_mean', 'EXACT_COORD_same_direction_two_face_edges', 'opposite_winding_reached_free_area_fraction'])

def direction(k):
    if k in LOW or k.startswith('illegal_pairs_'):
        return 'lower_descriptive'
    if k in HIGH or k.startswith(('A_VMR','PMR')):
        return 'higher_descriptive'
    return 'context_only'

def classify(d, k):
    if abs(d) <= TOL:
        return 'unchanged'
    direct = direction(k)
    if direct == 'context_only':
        return 'increased' if d > 0 else 'decreased'
    return 'improved' if (d < 0) == (direct == 'lower_descriptive') else 'worsened'

def main():
    started = time.time()
    inputs = [SRC/'results/final/per_output.csv', SRC/'results/final/completion.json', SRC/'hinge/baseline_audit.json', SRC/'hinge/baseline_audit/per_output.csv', SRC/'REPORT.md', SRC/'source_snapshot.json', ROOT/'native_t1/docs/control_metrics.md']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    protocol = {'baseline':'PLAIN_S2000 POST', 'candidates':STAGES[1:], 'scope':'same 28 saved requests, 4 TRAIN parents x 4 and 3 repeatedly observed DEV parents x 4', 'aggregation':'macro per output; each parent has 4 noises; ALL uses 16:12 output weights', 'thresholds':'015 primary; 025 existing stress setting, no new threshold search', 'paired_tolerance':TOL, 'training_updates':0, 'model_calls':0, 'new_samples':0, 'new_geometry_scoring':0, 'source_cache_hashes':hashes, 'missing':'preserve missing/status, never fill zero', 'direction_note':'descriptive diagnostics only, no weighted omnibus score; boundary count, area, denominators context-only', 'resource_plan':'one aggregation process and two independent read-only audits, at most 3 CPU threads; zero CUDA initialization'}
    old_protocol = OUT/'protocol.json'
    if old_protocol.exists():
        assert json.loads(old_protocol.read_text(encoding='utf-8'))['source_cache_hashes'] == hashes, 'Source changed since protocol freeze'
    dump('protocol.json',protocol)
    allrows = readcsv(inputs[0])
    rows = [r for r in allrows if r['label']=='PLAIN_S2000' and r['stage'] in STAGES]
    assert len(rows)==84
    lookup = {(r['stage'],r['request_id']):r for r in rows}
    assert len(lookup)==84
    ids = sorted({r['request_id'] for r in rows})
    assert len(ids)==28
    for rid in ids:
        b = lookup['POST',rid]
        for stage in STAGES[1:]:
            r = lookup[stage,rid]
            for k in ('condition_id','parent_id','noise_index','N','K','split','actual_free_faces','deleted_free_faces','free_face_denominator','C_query_weight_denominator'):
                assert r[k]==b[k], (rid,stage,k,r[k],b[k])
    complete = json.loads(inputs[1].read_text(encoding='utf-8-sig'))
    declared = complete['original_numeric_fields']
    fields = [k for k in rows[0] if k not in META and any(number(r[k]) is not None for r in rows)]
    assert set(declared) <= set(fields)
    assert len(declared)==124
    status_fields = [k for k in rows[0] if k not in META and k not in fields]
    writecsv('per_output.csv', sorted(rows,key=lambda r:(r['split'],r['parent_id'],r['noise_index'],STAGES.index(r['stage']))))
    statusrows=[]
    for stage in STAGES:
        for k in status_fields:
            for value,n in Counter(r[k] for r in rows if r['stage']==stage).items():
                statusrows.append({'stage':stage,'field':k,'value':value,'n':n})
    writecsv('status_coverage.csv',statusrows)
    deltas=[]
    for rid in ids:
        b=lookup['POST',rid]
        for stage in STAGES[1:]:
            r=lookup[stage,rid]
            for k in fields:
                x,y=number(b[k]),number(r[k])
                good = x is not None and y is not None
                d=y-x if good else None
                deltas.append({'request_id':rid,'condition_id':r['condition_id'],'parent_id':r['parent_id'],'noise_index':r['noise_index'],'split':r['split'],'candidate':stage,'metric':k,'baseline':x,'candidate_value':y,'delta':d,'direction':direction(k),'change':classify(d,k) if good else 'NA','status':'PASS' if good else 'NA_SOURCE_VALUE'})
    writecsv('paired_per_noise.csv',deltas)
    writecsv('all_regressions.csv',[r for r in deltas if r['change']=='worsened'])
    summaries=[]
    def summarize(group, selected_ids, extra):
        for k in fields:
            rr={'group':group,**extra,'metric':k,'direction':direction(k),'n_registered':len(selected_ids)}
            for stage in STAGES:
                vals=[number(lookup[stage,rid][k]) for rid in selected_ids]
                valid=[v for v in vals if v is not None]
                rr[stage+'_n']=len(valid)
                for stat,func in [('mean',st.mean),('median',st.median),('min',min),('max',max),('sum',sum),('p95',lambda x:quantile(x,.95))]:
                    rr[stage+'_'+stat]=func(valid) if valid else None
                if stage!='POST':
                    ds=[d for d in deltas if d['request_id'] in selected_ids and d['candidate']==stage and d['metric']==k]
                    dd=[d['delta'] for d in ds if d['status']=='PASS']
                    rr[stage+'_delta_mean']=st.mean(dd) if dd else None
                    rr[stage+'_paired_n']=len(dd)
                    for c in ('improved','unchanged','worsened','increased','decreased','NA'):
                        rr[stage+'_'+c]=sum(d['change']==c for d in ds)
                    adverse=[d for d in ds if d['change']=='worsened']
                    adverse.sort(key=lambda d:abs(d['delta']),reverse=True)
                    rr[stage+'_worst_request']=adverse[0]['request_id'] if adverse else ''
                    rr[stage+'_worst_delta']=adverse[0]['delta'] if adverse else 0
            rr['status']='PASS' if all(rr[s+'_n']==len(selected_ids) for s in STAGES) else 'PARTIAL'
            yield rr
    for group in ('TRAIN','DEV','ALL'):
        sel=[rid for rid in ids if group=='ALL' or lookup['POST',rid]['split']==group]
        summaries.extend(summarize(group,sel,{}))
    parents=[]
    for parent in sorted({r['parent_id'] for r in rows}):
        sel=[rid for rid in ids if lookup['POST',rid]['parent_id']==parent]
        assert len(sel)==4
        parents.extend(summarize(parent,sel,{'split':lookup['POST',sel[0]]['split']}))
    writecsv('summary_all_metrics.csv',summaries)
    writecsv('paired_per_parent.csv',parents)
    key=[r for r in summaries if r['metric'] in KEY]
    writecsv('key_metrics.csv',key)
    audit=json.loads(inputs[2].read_text(encoding='utf-8'))
    writecsv('preservation_ledger.csv',audit['summary'])
    def fmt(v):
        if v is None: return 'NA'
        return format(v,'.8g') if isinstance(v,(float,int)) else str(v)
    hs=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>HINGE 与 S2000 完整指标对照</title><style>body{font:15px system-ui;margin:28px;color:#172431}table{border-collapse:collapse;width:100%;margin:18px 0}th,td{border:1px solid #ccd4df;padding:7px;text-align:right}th{background:#edf2f8;position:sticky;top:0}td:first-child,th:first-child{text-align:left;overflow-wrap:anywhere}tr:nth-child(even){background:#f7f9fb}small{color:#526275}details{margin:25px 0}input{padding:10px;width:400px;max-width:90%}</style><h1>HINGE 与 S2000 完整指标对照</h1><p>同一28份生成结果：TRAIN 16，DEV 12。原POST独立比较HINGE 1.5%L和2.5%L，未与BOUND串联。表内保留原数值单位，fraction为0至1；均值为逐输出宏平均。改善／持平／退化是描述性计数，不合成总分。DEV是反复使用的三个开发父，不是新泛化测试。</p><p><a href="REPORT.md">结论与关键指标</a> · <a href="summary_all_metrics.csv">完整汇总CSV</a> · <a href="paired_per_noise.csv">逐噪声配对</a> · <a href="paired_per_parent.csv">逐父配对</a> · <a href="metric_catalog.csv">字段定义</a></p><input id="q" placeholder="按指标名称筛选，例如 FF、PMR、area"><p>Δ＝候选减baseline。边界边、表面积等未指定单调优劣的字段仅报告变化。</p>']
    for group in ('TRAIN','DEV','ALL'):
        hs.append('<h2>'+group+'</h2><table><thead><tr><th>指标（原单位）</th><th>POST</th><th>HINGE015</th><th>Δ015</th><th>改善/持平/退化</th><th>HINGE025</th><th>Δ025</th><th>改善/持平/退化</th></tr></thead><tbody>')
        for r in summaries:
            if r['group']!=group:continue
            cells=[r['metric'],fmt(r['POST_mean'])]
            for s in STAGES[1:]:
                counts='/'.join(str(r[s+'_'+c]) for c in ('improved','unchanged','worsened')) if r['direction']!='context_only' else 'context only'
                cells.extend([fmt(r[s+'_mean']),fmt(r[s+'_delta_mean']),counts])
            hs.append('<tr data-metric="'+html.escape(r['metric'].lower())+'">'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
        hs.append('</tbody></table>')
    hs.append('<script>document.getElementById("q").addEventListener("input",e=>{let q=e.target.value.toLowerCase();document.querySelectorAll("tr[data-metric]").forEach(r=>r.hidden=!r.dataset.metric.includes(q))})</script></html>')
    (OUT/'index.html').write_text('\n'.join(hs),encoding='utf-8')
    unchanged={str(p.relative_to(ROOT)):sha(p)==hashes[str(p.relative_to(ROOT))] for p in inputs}
    assert all(unchanged.values())
    result={'status':'COMPLETE_CACHE_RECOMPUTATION','original_fields':len(declared),'numeric_fields':len(fields),'derived_topology_fields':[k for k in fields if k not in declared],'source_rows':len(allrows),'selected_rows':len(rows),'paired_values':len(deltas),'summary_rows':len(summaries),'parent_summary_rows':len(parents),'missing_numeric_cells':sum(number(r[k]) is None for r in rows for k in fields),'source_unchanged':unchanged,'elapsed_seconds':time.time()-started,'model_calls':0,'new_geometry_scoring':0,'script_sha256':sha(Path(__file__))}
    dump('completion.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_unchanged','derived_topology_fields')},ensure_ascii=False))

if __name__=='__main__':
    main()
