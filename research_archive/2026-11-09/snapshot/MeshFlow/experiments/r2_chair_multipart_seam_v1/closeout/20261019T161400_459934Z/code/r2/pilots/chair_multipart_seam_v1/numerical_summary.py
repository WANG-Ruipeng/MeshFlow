"""Read completed paired CSVs only; no model, mesh, geometry or scorer imports."""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_chair_multipart_seam_v1'
SRC=OUT/'comparison'
DEST=OUT/'analysis'
STAGES=('RAW','BASE_POST','GUARD_GEO','UNIFIED_FINAL')
GROUPS=(('SEAT_FULL','TRAIN'),('SEAT_FULL','DEV'),('BACK_FULL','FAMILIAR'),('BACK_FULL','DEV'))
NEW='MULTIPART_S1500'
QUALITY=(('CF_illegal_affected_free_fraction','CF affected free %',100),
 ('FF_illegal_affected_free_fraction','FF affected free %',100),
 ('ConflictCov_C','ConflictCov_C %',100),
 ('free_minimum_angle_lt5_fraction','Free angle <5 degrees %',100),
 ('hinge_diag_isolated_1face_edge_components','Isolated single faces: legal graph',1),
 ('hinge_diag_isolated_1face_unfiltered_edge_components','Isolated single faces: unfiltered graph',1),
 ('nonmanifold_edges','Nonmanifold edges',1))
CONNECTION=(('C_F_valid_edge_count','Legal C/F edges',1),
 ('C_F_valid_edge_total_length','Legal C/F edge length',1),
 ('legal_seam_bulk_edge_count','Legal seam/bulk edges',1),
 ('free_only_reached_bulk_area_fraction','Reached bulk area %',100),
 ('actual_free_faces','Retained free faces',1),('deleted_free_faces','Deleted free faces',1),
 ('free_total_surface_area','Free surface area',1))

def resolve(p):
    p=str(p)
    if p.startswith('/mnt/e/MeshFlow/'):
        return ROOT/p[len('/mnt/e/MeshFlow/'):]
    return Path(p)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def num(v):
    if v in ('True',True):return 1.0
    if v in ('False',False):return 0.0
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):return None
def avg(xs):return sum(xs)/len(xs) if xs else None
def parent_mean_paired(rows):
    c=collections.defaultdict(list)
    for r in rows:
        v=num(r['delta'])
        if v is not None:c[r['parent_id'],r['condition_id']].append(v)
    p=collections.defaultdict(list)
    for (pid,cid),values in c.items():p[pid].append(avg(values))
    return avg([avg(v) for v in p.values()]),len(p)
def fmt(v,scale=1):return 'NA' if v is None else f'{v*scale:.4f}'
def save_csv(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def run():
    completion=read(SRC/'completion.json');assert completion['status']=='COMPLETE' and completion['requests']==44
    prediction=read(OUT/'layout_diagnostic/prediction_completion.json');assert prediction['status']=='COMPLETE'
    for rec in (completion,prediction):
        for p,h in rec['files'].items():assert sha(resolve(p))==h,p
    summary=csvread(SRC/'summary_parent_equal.csv');paired=csvread(SRC/'paired_per_output.csv')
    output=csvread(SRC/'per_output.csv')
    assert len(output)==396 and len({r['request_id'] for r in output})==44
    index={(r['model'],r['stage'],r['part_type'],r['split'],r['metric']):r for r in summary}
    lines=['# CHAIR_MULTIPART_SEAM_V1 — independent numerical summary','',
      'Cache-only descriptive comparison. S2000 global15000 versus MULTIPART global15500 changes the mixed condition pool, C-only layout head, and cumulative budget together. It does not isolate any single component.',
      '', 'The four part/split strata remain separate. BACK improvement cannot cancel SEAT regression. Every cell compares the same processing stage and fixed request identities.',
      '', 'Cells are old → new (paired new−old). Fractions are converted to percentages; their deltas are percentage points. Other fields retain model units or counts. Means are equal-parent after condition/noise averaging. NA is preserved, not replaced by zero.','']
    tables=[];extremes=[];thresholds=[]
    for part,split in GROUPS:
        lines += [f'## {part} / {split}','']
        for title,metrics in (('Quality',QUALITY),('Connection and surface retention',CONNECTION)):
            lines += [f'### {title}','','| Metric | '+' | '.join(STAGES)+' |','|---|'+ '|'.join(['---']*4)+'|']
            for metric,label,scale in metrics:
                cells=[]
                for stage in STAGES:
                    a=index.get(('S2000',stage,part,split,metric));b=index.get((NEW,stage,part,split,metric))
                    old=num(a['mean_parent_equal']) if a else None;new=num(b['mean_parent_equal']) if b else None
                    pp=[r for r in paired if (r['part_type'],r['split'],r['stage'],r['metric'])==(part,split,stage,metric)]
                    delta,nparents=parent_mean_paired(pp)
                    cells.append(f'{fmt(old,scale)} → {fmt(new,scale)} ({fmt(delta,scale)})')
                    tables.append(dict(part_type=part,split=split,stage=stage,metric=metric,scale=scale,
                        old_parent_equal=old,new_parent_equal=new,paired_delta_parent_equal=delta,
                        paired_valid_parents=nparents,old_valid_outputs=int(a['n_valid']) if a else 0,
                        new_valid_outputs=int(b['n_valid']) if b else 0))
                    valid=[r for r in pp if num(r['delta']) is not None]
                    if valid:
                        for which,rr in [('minimum_delta',min(valid,key=lambda r:num(r['delta']))),('maximum_delta',max(valid,key=lambda r:num(r['delta'])))]:
                            extremes.append(dict(extreme=which,**rr))
                lines.append('| '+label+' | '+' | '.join(cells)+' |')
            lines += ['']
        lines += ['### Descriptive ≥30% bulk-area request outcomes','','| Stage | Old / valid | New / valid | Newly above | Lost |','|---|---|---|---|---|']
        for stage in STAGES:
            pp=[r for r in paired if (r['part_type'],r['split'],r['stage'],r['metric'])==(part,split,stage,'bulk_reach_ge_0_30_descriptive')]
            valid=[r for r in pp if num(r['old_value']) is not None and num(r['new_value']) is not None]
            old=sum(num(r['old_value'])>=.5 for r in valid);new=sum(num(r['new_value'])>=.5 for r in valid)
            gained=[r['request_id'] for r in valid if num(r['old_value'])<.5<=num(r['new_value'])]
            lost=[r['request_id'] for r in valid if num(r['new_value'])<.5<=num(r['old_value'])]
            lines.append(f"| {stage} | {old}/{len(valid)} | {new}/{len(valid)} | {', '.join(gained) or '—'} | {', '.join(lost) or '—'} |")
            thresholds.append(dict(part_type=part,split=split,stage=stage,valid=len(valid),old=old,new=new,gained=gained,lost=lost))
        lines += ['']
    lines += ['## Saved C-only layout versus source reference','','Source owner is one diagnostic completion layout, not the only valid generated topology. Source labels were never sent to inference. Fixed threshold is logit 0.','','| Part / split | Edges | Active | TP | FN | FP | TN | Alias active | Unsupported active | Owner recall | All-off C |','|---|---|---|---|---|---|---|---|---|---|---|']
    layout=[]
    for part,split in GROUPS:
        rr=[r for r in prediction['conditions'] if (r['part_type'],r['split'])==(part,split)]
        v={k:sum(int(r[k]) for r in rr) for k in ('edges','active','TP','FN','FP','TN','alias_activated','unsupported_activated')}
        recall=v['TP']/(v['TP']+v['FN']) if v['TP']+v['FN'] else None
        off=[r['condition_id'] for r in rr if not r['active']]
        lines.append('| '+part+' / '+split+' | '+' | '.join(str(v[k]) for k in ('edges','active','TP','FN','FP','TN','alias_activated','unsupported_activated'))+' | '+fmt(recall,100)+'% | '+(', '.join(off) or '—')+' |')
        layout.append(dict(part_type=part,split=split,owner_recall=recall,all_off_conditions=off,**v))
    lines += ['', '## Interpretation limits','',
       'This document prints cache-derived numbers only. Root SEAT28 and separate BACK16 actual AI_PROXY image reviews decide visual usefulness; no visual labels are derived from these tables. Legal-edge reachability does not certify a complete useful body, watertightness, manifoldness, or mechanical support. ConflictCov_C is not whole-mesh intersection coverage.',
       '', 'Full metrics, per-parent records, all paired request deltas, prior final baseline and missing-field denominators remain in the frozen comparison outputs. No checkpoint selection, default promotion, new scoring, sampling, training, or threshold tuning occurred.']
    DEST.mkdir(parents=True,exist_ok=True)
    paths=[DEST/'NUMERICAL_SUMMARY.md',DEST/'key_paired_summary.csv',DEST/'key_pair_extremes.csv',DEST/'numerical_summary.json']
    if any(p.exists() for p in paths):raise FileExistsError('Preserve existing summary; explicit version required')
    paths[0].write_text('\n'.join(lines),encoding='utf8');save_csv(paths[1],tables);save_csv(paths[2],extremes)
    result=dict(status='COMPLETE_CACHE_ONLY',source_comparison_sha256=sha(SRC/'completion.json'),
        source_layout_sha256=sha(OUT/'layout_diagnostic/prediction_completion.json'),code_sha256=sha(__file__),
        requests=44,groups=[list(g) for g in GROUPS],key_metrics=14,tables=tables,thresholds=thresholds,layout=layout,
        visual_judgments=None,model_calls=0,scoring_calls=0,geometry_calls=0,
        files={str(p):sha(p) for p in paths[:3]})
    with paths[3].open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('tables','thresholds','layout')},indent=2))
    return result

if __name__=='__main__':run()
