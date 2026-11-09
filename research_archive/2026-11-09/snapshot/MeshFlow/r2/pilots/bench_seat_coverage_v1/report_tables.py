"""Read-only 177-field and actual visual review tables. No scorer/model imports.

Default only reports readiness. --final requires completed collect AND actual
visual summary, verifies their recorded hashes, and writes immutable report_tables
artifacts. It never generates ratings, promotes a model, or changes old evidence.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import io
import itertools
import json
import math
import statistics
from pathlib import Path
from r2.pilots.bench_full_seat_curriculum_v1 import summarize_results as S

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'experiments/r2_bench_seat_coverage_v1'
RESULTS = OUT/'results'
DEST = RESULTS/'report_tables_v1'
OLD = ROOT/'experiments/r2_bench_full_seat_curriculum_v1'
LABELS = ('NATIVE_S6000', 'COVERAGE_FINAL', 'SEAM_S3000')
STAGES = ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')
GROUPS = ('OLD_TRAIN', 'NEW_TRAIN', 'NEW_PARENT_HOLDOUT', 'DEV')
AXES = {
    'overall_shape': ('REASONABLE', 'INCOMPLETE', 'UNCERTAIN'),
    'C_use': ('PLAUSIBLE', 'REPLACED_OR_IGNORED', 'UNCERTAIN'),
    'support_retention': ('PRESENT', 'MISSING_OR_DAMAGED', 'UNCERTAIN'),
    'visible_fragment_or_gap': ('YES', 'NO_OBVIOUS', 'UNCERTAIN'),
    'layering_or_fold': ('YES', 'NO_OBVIOUS', 'UNCERTAIN'),
}
CORE = ('CF_illegal_affected_free_fraction', 'FF_illegal_affected_free_fraction',
        'ConflictCov_C', 'CF_illegal_pair_count', 'FF_illegal_pair_count',
        'free_minimum_angle_lt5_fraction', 'free_q_gt10_fraction',
        'deleted_free_faces', 'actual_free_faces', 'free_total_surface_area',
        'new_bridge_faces', 'bridge_area', 'deleted_original_POST_faces',
        'hinge_diag_isolated_1face_edge_components',
        'hinge_diag_isolated_1face_unfiltered_edge_components',
        'C_F_valid_edge_count', 'C_F_valid_edge_total_length',
        'C_F_has_valid_edge', 'C_F_valid_opposite_winding_edge_count',
        'nonmanifold_edges', 'C_F_nonmanifold_edge_count',
        'hinge_diag_legal_opposed_edge_C_reached_free_area_fraction',
        'hinge_diag_fully_C_disconnected_free_area_fraction',
        'hinge_diag_vertex_attached_only_free_area_fraction',
        'free_only_reached_bulk_area_fraction', 'retained_bulk_area')
COLLECT_TABLES = ('per_output', 'per_condition', 'per_parent', 'parent_macro_summary',
                  'paired_per_output', 'paired_per_condition', 'paired_per_parent')

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return S.digest(path)
def csvread(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rd=csv.DictReader(f); return list(rd), rd.fieldnames
def immutable(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    raw=data if isinstance(data, bytes) else (json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=raw: raise ValueError('Existing report differs; do not overwrite: '+str(path))
    else:
        with path.open('xb') as f: f.write(raw)
def csvwrite(path, rows, fields=None):
    if fields is None: fields=list(dict.fromkeys(k for r in rows for k in r))
    text=io.StringIO(newline=''); w=csv.DictWriter(text,fieldnames=fields,lineterminator='\n');w.writeheader()
    w.writerows({k:S.csv_cell(r.get(k)) for k in fields} for r in rows)
    immutable(path, ('\ufeff'+text.getvalue()).encode())
def group(value):
    value='OLD_TRAIN' if value=='CURRENT_TRAIN' else value
    if value not in GROUPS: raise ValueError('Unexpected evaluation group: '+value)
    return value

def readiness():
    paths=[OUT/'eval_manifest.json',RESULTS/'completion.json',
           OUT/'visual_review/summary_v1/completion.json',
           OUT/'visual_review/summary_v1/actual_observations.csv',
           OLD/'results/PRIMARY_TABLES.md', OLD/'results/analysis_v1/primary_parent_macro.csv']
    paths += [RESULTS/(name+'.csv') for name in COLLECT_TABLES]
    missing=[str(p) for p in paths if not p.is_file()]
    return dict(status='PENDING' if missing else 'READY_FOR_READONLY_AGGREGATION',
                missing=missing, output_written=False, model_calls=0, scoring_calls=0,
                ratings_generated=0, automatic_promotion=False)

def verify_files(mapping, root=None):
    for path,h in mapping.items():
        p=(root/path) if root else S.resolve(path)
        if sha(p)!=h: raise ValueError('Recorded evidence hash differs: '+str(p))

def audit_old_units():
    """Check the published baseline text and every displayed mean vs its CSV."""
    md=OLD/'results/PRIMARY_TABLES.md'; source=OLD/'results/analysis_v1/primary_parent_macro.csv'
    text=md.read_text('utf-8-sig')
    if 'fraction以0–1表示，不是百分数' not in text: raise ValueError('Baseline unit statement changed')
    rows,_=csvread(source)
    by={(r['label'],r['stage'],r['eval_group'],r['metric']):r for r in rows}
    labels=('COMMON_S12000','NATIVE_S6000','SEAM_S3000'); checked=0;current=None
    for line in text.splitlines():
        if line.startswith('## '):
            current=tuple(line[3:].split(' / '))
        if not current or not line.startswith('| '): continue
        cells=[s.strip() for s in line.strip().strip('|').split('|')]
        if len(cells)!=4 or cells[0] not in S.PRIMARY: continue
        for label,cell in zip(labels,cells[1:]):
            raw=by[label,current[1],current[0],cells[0]]
            observed=S.number(cell.split('〔')[0]); expected=S.number(raw['mean'])
            if (observed is None)!=(expected is None): raise ValueError('Baseline NA differs')
            if expected is not None and not math.isclose(expected,observed,rel_tol=1e-7,abs_tol=5e-9):
                raise ValueError('Baseline displayed unit/value differs: '+cells[0])
            checked+=1
    if checked!=len(rows): raise ValueError('Did not authenticate all old primary table cells')
    return dict(status='PASS', published_cells_checked=checked,
        table_sha256=sha(md), source_csv_sha256=sha(source),
        fractions='Original fraction 0..1, never multiplied by100 in machine tables.',
        deltas='candidate minus reference in original units; fraction deltas are not percentage points.',
        area='model coordinate squared; no rescaling', length='model coordinates; no rescaling',
        support='Request and parent support retained; Native SEAM-only NA is not zero.')

def audit_preflight():
    """CPU interface preflight on old evidence only, before new results exist."""
    units=audit_old_units()
    old=read(OLD/'results/completion.json')
    original,fields=csvread(OLD/'results/per_output.csv')
    metrics=old['metric_fields']
    if len(metrics)!=177 or not set(metrics).issubset(fields):raise ValueError('Old177 CSV fields differ')
    if not set(CORE).issubset(metrics):raise ValueError('Core field missing from old177')
    if len(original)!=216:raise ValueError('Old score row count differs')
    native=[r for r in original if r['label']=='NATIVE_S6000']
    for r in native:
        if not r['seam_applicability'].startswith('NA'):raise ValueError('Old Native applicability differs')
        if S.number(r['free_only_reached_bulk_area_fraction']) is not None:raise ValueError('Native SEAM-only NA replaced by number')
    receipt=dict(status='PASS',unit_audit=units,original_numeric_fields=177,
        all_core_fields_compatible=True,old_score_rows=216,native_seam_NA_rows_checked=len(native),
        script_sha256=sha(__file__),source_completion_sha256=sha(OLD/'results/completion.json'),
        source_per_output_sha256=sha(OLD/'results/per_output.csv'),model_calls=0,scoring_calls=0,
        new_result_rows_read=0,new_ratings=0,
        invocation_note='Initial PowerShell/WSL inline python command failed parsing before imports; reran via explicit module CLI flag. No model/scorer or old evidence mutation.')
    immutable(RESULTS/'report_tables_preflight.json',receipt)
    return receipt

def unit(metric):
    if 'fraction' in metric or metric.startswith(('A_VMR__','PMR__')) or metric in S.PASS_ONLY_I or metric in ('ConflictCov_C_upper','C_F_has_valid_edge','C_F_has_opposite_winding_edge','bulk_reach_ge_0_30_descriptive'):
        return 'fraction_0_to_1'
    if 'degrees' in metric: return 'degrees'
    if 'length' in metric or metric.startswith('boundary_distance_'): return 'model_length'
    if 'area' in metric and not any(x in metric for x in ('zero_area','near_degenerate')): return 'model_length_squared'
    if metric.endswith('q_mean'): return 'dimensionless_shape_factor'
    if metric=='C_query_weight_denominator': return 'normalized_weight'
    if any(x in metric for x in ('count','faces','edges','vertices','components','pairs','samples','face_denominator','pair_denominator')): return 'count'
    return 'original_scorer_unit_no_conversion'

def decorate_report_group(row):
    return dict(row, report_group=group(row['eval_group']))

def load_inputs():
    state=readiness()
    if state['status']=='PENDING': return state,None
    complete=read(RESULTS/'completion.json'); visual=read(OUT/'visual_review/summary_v1/completion.json')
    if complete['status']!='COMPLETE' or visual['status']!='COMPLETE': raise ValueError('Incomplete producer')
    if any(complete.get(k)!=0 for k in ('model_calls','scoring_calls')): raise ValueError('Collector ran unexpected computation')
    if any(visual.get(k)!=0 for k in ('generated_ratings','model_calls','scoring_calls')): raise ValueError('Visual summary computation differs')
    verify_files(complete['files']); verify_files(visual['artifacts'], OUT/'visual_review/summary_v1')
    manifest=read(OUT/'eval_manifest.json'); qs={q['request_id']:q for q in manifest['requests']}
    rows,fields=csvread(RESULTS/'per_output.csv'); observations,_=csvread(OUT/'visual_review/summary_v1/actual_observations.csv')
    metrics=list(complete['metric_fields']); baseline=read(OLD/'results/completion.json')['metric_fields']
    if len(metrics)!=177 or set(metrics)!=set(baseline): raise ValueError('Original177 metric dictionary differs')
    if not set(metrics).issubset(fields): raise ValueError('Collected numeric field missing')
    expected={(label,stage,rid) for rid,q in qs.items() for label in (LABELS if q['panel_origin']=='INHERITED24' else LABELS[:2]) for stage in STAGES}
    for table in (rows,observations):
        keys=[(r['label'],r['stage'],r['request_id']) for r in table]
        if len(keys)!=252 or set(keys)!=expected: raise ValueError('Fixed252 unique cells not complete')
        for r in table:
            q=qs[r['request_id']]
            if any(r[k]!=str(q[k]) for k in ('condition_id','parent_id','eval_group')): raise ValueError('Registered identity mismatch')
    return dict(status='VALIDATED',rows=len(rows),metrics=len(metrics)),dict(rows=rows,fields=fields,
        observations=observations,metrics=metrics,manifest=manifest,complete=complete,visual=visual)

def pair_rows(reference,candidate,metrics,meta):
    by={r['request_id']:r for r in candidate}; output=[]
    for a in reference:
        b=by.get(a['request_id'])
        if b is None:continue
        r=dict(meta,request_id=a['request_id'],condition_id=a['condition_id'],parent_id=a['parent_id'],noise_index=a['noise_index'])
        for m in metrics:
            av,bv=S.number(a.get(m)),S.number(b.get(m))
            r[m+'__reference']=a.get(m);r[m+'__candidate']=b.get(m)
            r[m+'__delta']=None if av is None or bv is None else bv-av
        for side,row in (('reference',a),('candidate',b)):
            r[side+'_I_status']=row.get('ConflictCov_C_status');r[side+'_seam_status']=row.get('seam_applicability')
        output.append(r)
    return output

def numeric_tables(data):
    rows=data['rows'];metrics=data['metrics']; macros=[];primary=[];parents=[];pairmacro=[];pairreq=[];pairparents=[];retention=[]
    for label,stage,g in itertools.product(LABELS,STAGES,GROUPS):
        block=[r for r in rows if (r['label'],r['stage'],group(r['eval_group']))==(label,stage,g)]
        meta=dict(label=label,stage=stage,report_group=g,
                  coverage_status='EVALUATED' if block else 'NOT_EVALUATED_ON_NEW_PANEL')
        for m in metrics:
            rr=dict(meta,unit=unit(m),**S.aggregate(block,m,'ORIGINAL_FINITE_ONLY'));macros.append(rr)
            if m in CORE:
                primary.append(dict(meta,unit=unit(m),**S.aggregate(block,m,'I_PASS_ONLY' if m in S.PASS_ONLY_I else 'ORIGINAL_FINITE_ONLY')))
            for parent in sorted({r['parent_id'] for r in block}):
                subset=[r for r in block if r['parent_id']==parent]
                parents.append(dict(meta,parent_id=parent,unit=unit(m),**S.aggregate(subset,m,'ORIGINAL_FINITE_ONLY')))
    for stage,g in itertools.product(STAGES,GROUPS):
        for reference,candidate in itertools.combinations(LABELS,2):
            aa=[r for r in rows if (r['label'],r['stage'],group(r['eval_group']))==(reference,stage,g)]
            bb=[r for r in rows if (r['label'],r['stage'],group(r['eval_group']))==(candidate,stage,g)]
            meta=dict(reference=reference,candidate=candidate,stage=stage,report_group=g)
            if not aa or not bb:continue
            pairreq.extend(pair_rows(aa,bb,metrics,meta))
            for m in metrics:
                pp=S.paired(aa,bb,m,'ORIGINAL_FINITE_ONLY')
                pairmacro.append(dict(meta,unit=unit(m),**pp))
                for parent in sorted({r['parent_id'] for r in aa}):
                    pairparents.append(dict(meta,parent_id=parent,unit=unit(m),**S.paired(
                        [r for r in aa if r['parent_id']==parent],[r for r in bb if r['parent_id']==parent],m,'ORIGINAL_FINITE_ONLY')))
    for r in rows:
        n,k=(S.number(r.get(x)) for x in ('N','K'))
        deleted,actual,bridge=(S.number(r.get(x)) for x in ('deleted_free_faces','actual_free_faces','new_bridge_faces'))
        requested=None if n is None or k is None else n-k
        retained=None if requested is None or deleted is None else requested-deleted
        if all(x is not None for x in (retained,actual,bridge)) and actual!=retained+bridge:raise ValueError('Free face accounting differs')
        retention.append(dict(label=r['label'],stage=r['stage'],report_group=group(r['eval_group']),parent_id=r['parent_id'],condition_id=r['condition_id'],request_id=r['request_id'],
            requested_original_free_faces=requested,retained_original_free_faces=retained,deleted_original_free_faces=deleted,
            new_bridge_faces=bridge,actual_free_faces=actual,free_total_surface_area=r.get('free_total_surface_area'),
            bridge_area=r.get('bridge_area'),free_area=r.get('free_area'),
            area_interpretation='Current scored free surface including retained/added faces as original scorer defines; not source GT area retention.'))
    return dict(all_numeric_parent_macro=macros,core_parent_macro=primary,all_numeric_per_parent=parents,
        paired_per_request=pairreq,all_numeric_paired_parent_macro=pairmacro,all_numeric_paired_per_parent=pairparents,
        per_output_face_and_area_accounting=retention)

def strict_visual(row):
    axes={'C_use':'PLAUSIBLE','overall_shape':'REASONABLE','support_retention':'PRESENT'}
    if all(row[k]==v for k,v in axes.items()):return 'YES'
    if any(row[k] not in (v,'UNCERTAIN') for k,v in axes.items()):return 'NO'
    return 'UNCERTAIN'

def visual_tables(data):
    rows=[]
    for r in data['observations']:
        if any(r[k] not in values for k,values in AXES.items()):raise ValueError('Invalid actual visual label')
        rows.append(dict(decorate_report_group(r),strict_visual_three_axis=strict_visual(r)))
    categories=dict(AXES,strict_visual_three_axis=('YES','NO','UNCERTAIN')); counts=[];parents=[];pairs=[];parentpairs=[]
    for label,stage,g in itertools.product(LABELS,STAGES,GROUPS):
        block=[r for r in rows if (r['label'],r['stage'],r['report_group'])==(label,stage,g)]
        for axis,values in categories.items():
            for value in values:
                pv=S.parent_values([(r,float(r[axis]==value)) for r in block])
                counts.append(dict(label=label,stage=stage,report_group=g,axis=axis,category=value,
                    count=sum(r[axis]==value for r in block),requests=len(block),parents=len(pv),
                    parent_equal_fraction=statistics.fmean(pv.values()) if pv else None,
                    coverage_status='EVALUATED' if block else 'NOT_EVALUATED_ON_NEW_PANEL'))
                for parent,rate in pv.items():
                    pr=[r for r in block if r['parent_id']==parent]
                    parents.append(dict(label=label,stage=stage,report_group=g,parent_id=parent,axis=axis,category=value,
                        count=sum(r[axis]==value for r in pr),requests=len(pr),fraction=rate))
    index={(r['label'],r['stage'],r['request_id']):r for r in rows}
    for reference,candidate in itertools.combinations(LABELS,2):
        for a in (r for r in rows if r['label']==reference):
            b=index.get((candidate,a['stage'],a['request_id']))
            if b is None:continue
            for axis in categories:
                pairs.append(dict(reference_label=reference,candidate_label=candidate,stage=a['stage'],report_group=a['report_group'],
                    parent_id=a['parent_id'],condition_id=a['condition_id'],request_id=a['request_id'],axis=axis,
                    reference=a[axis],candidate=b[axis],unchanged=a[axis]==b[axis],
                    strict_visual_gained=axis=='strict_visual_three_axis' and a[axis]!='YES' and b[axis]=='YES',
                    strict_visual_lost=axis=='strict_visual_three_axis' and a[axis]=='YES' and b[axis]!='YES'))
    for key in sorted({(r['reference_label'],r['candidate_label'],r['stage'],r['report_group'],r['parent_id']) for r in pairs}):
        block=[r for r in pairs if (r['reference_label'],r['candidate_label'],r['stage'],r['report_group'],r['parent_id'])==key and r['axis']=='strict_visual_three_axis']
        parentpairs.append(dict(zip(('reference_label','candidate_label','stage','report_group','parent_id'),key),
            requests=len(block),reference_yes=sum(r['reference']=='YES' for r in block),candidate_yes=sum(r['candidate']=='YES' for r in block),
            gained_request_ids=[r['request_id'] for r in block if r['strict_visual_gained']],lost_request_ids=[r['request_id'] for r in block if r['strict_visual_lost']],
            uncertain_reference=sum(r['reference']=='UNCERTAIN' for r in block),uncertain_candidate=sum(r['candidate']=='UNCERTAIN' for r in block)))
    return dict(actual_observations_with_strict_visual_intersection=rows,visual_axis_and_strict_counts=counts,
        visual_per_parent_counts=parents,visual_paired_per_request=pairs,strict_visual_paired_per_parent=parentpairs)

def table_text(table, metrics, title):
    by={(r['label'],r['stage'],r['report_group'],r['metric']):r for r in table}
    lines=[title,'','原评分单位；fraction 为0–1，不是百分数。单元为父等权均值〔有效请求/有效父〕。',
        'OLD_TRAIN 是原 CURRENT_TRAIN 的报告别名。SEAM 未在新增6请求评测，写 NA；Native 的 SEAM 专属量仍 NA。',
        '保留所有支持分母与排除原因；不自动晋级。','']
    for g,stage in itertools.product(GROUPS,STAGES):
        lines += [f'## {g} / {stage}','','| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |','|---|---|---:|---:|---:|']
        for m in metrics:
            cells=[]
            for label in LABELS:
                r=by[label,stage,g,m];n=r['mean']
                cells.append(('NA' if n is None else f'{n:.8g}')+f"〔{r['valid_requests']}/{r['valid_parents']}〕")
            lines.append('| '+m+' | '+unit(m)+' | '+' | '.join(cells)+' |')
        lines.append('')
    return '\n'.join(lines)+'\n'

def visual_text(counts):
    by={(r['label'],r['stage'],r['report_group'],r['axis'],r['category']):r for r in counts}
    categories=dict(AXES,strict_visual_three_axis=('YES','NO','UNCERTAIN'))
    lines=['# 实际AI_PROXY五轴与同输出严格三轴交集','',
        '每格：类别请求数/全部已评请求〔父等权比例；父数〕。不确定项保留；没有评测的模型/面板写NA。',
        '严格三轴仅是同一输出 PLAUSIBLE + REASONABLE + PRESENT，不代表拓扑合法接续或几何质量通过。','']
    for g,stage in itertools.product(GROUPS,STAGES):
        lines += [f'## {g} / {stage}','','| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |','|---|---|---:|---:|---:|']
        for axis,values in categories.items():
            for value in values:
                cells=[]
                for label in LABELS:
                    r=by[label,stage,g,axis,value]
                    cells.append('NA' if not r['requests'] else f"{r['count']}/{r['requests']}〔{r['parent_equal_fraction']:.8g}; {r['parents']}父〕")
                lines.append('| '+axis+' | '+value+' | '+' | '.join(cells)+' |')
        lines.append('')
    return '\n'.join(lines)+'\n'

def run(final=False):
    ready=readiness()
    if not final or ready['status']=='PENDING':return ready
    if (DEST/'completion.json').exists():
        done=read(DEST/'completion.json');verify_files(done['files']);return done
    validated,data=load_inputs()
    if data is None:return validated
    units=audit_old_units(); tables=numeric_tables(data); tables.update(visual_tables(data))
    files={};per_output=[decorate_report_group(r) for r in data['rows']]
    csvwrite(DEST/'all_177_per_output.csv',per_output, data['fields']+['report_group'])
    files[str(DEST/'all_177_per_output.csv')]=sha(DEST/'all_177_per_output.csv')
    for name,rows in tables.items():
        p=DEST/(name+'.csv');csvwrite(p,rows);files[str(p)]=sha(p)
    for name,rows,metrics,title in (
        ('ALL_177_PARENT_TABLES.md',tables['all_numeric_parent_macro'],data['metrics'],'# 全部177字段：父等权完整对照'),
        ('CORE_PARENT_TABLES.md',tables['core_parent_macro'],CORE,'# 核心几何量：父等权对照（I仅PASS）')):
        p=DEST/name;immutable(p,table_text(rows,metrics,title).encode());files[str(p)]=sha(p)
    p=DEST/'unit_audit.json';immutable(p,units);files[str(p)]=sha(p)
    p=DEST/'VISUAL_TABLES.md';immutable(p,visual_text(tables['visual_axis_and_strict_counts']).encode());files[str(p)]=sha(p)
    notes='''# 汇总范围与解释边界

只读原 collect 的252行评分和已完成的252单元真实AI_PROXY图审；无新模型调用、评分、渲染或自动标签。

全177字段原值、状态和NA在all_177_per_output.csv保留。all_numeric系列只按原有限值汇总；核心表的I按原PASS状态筛选，未改检测器。父等权先平均同条件噪声，再平均条件，最后平均父；逐请求与逐父配对和增减范围均保存。

strict_visual_three_axis仅表示同一输出同时C_use=PLAUSIBLE、overall_shape=REASONABLE、support_retention=PRESENT。它不是新的几何接续成功定义，未包含合法接口或CF/FF/I阈值。UNCERTAIN不算YES且单独计数；总请求分母包括不确定。原五轴及每个类别均保存。

NEW_PARENT_HOLDOUT只有一个父两个噪声；NEW_TRAIN只有两个父。旧SEAM仅有原24请求，不补齐新增结果。旧CURRENT_TRAIN显示为OLD_TRAIN，源记录不改。图审非盲AI_PROXY；历史标签漂移保留在原visual_review/summary_v1中。

free_total_surface_area是当前阶段实际free表面面积，可能包含新增BRIDGE；不是未知真实成品面积的保留率，也不证明支撑完整。原free面、删除面与新增面单独记账。单位沿用旧PRIMARY_TABLES，fraction差值不自动转换成百分点。

本轮候选同时增加训练步数与改变数据/采样分布；结果不是新增父的单因素因果证明。不输出自动promote决定。
'''
    p=DEST/'README.md';immutable(p,notes.encode());files[str(p)]=sha(p)
    result=dict(status='COMPLETE',score_rows=252,numeric_fields=177,actual_visual_cells=252,
        groups=list(GROUPS), stages=list(STAGES),files=files,
        input_completion_sha256=sha(RESULTS/'completion.json'),
        input_visual_completion_sha256=sha(OUT/'visual_review/summary_v1/completion.json'),
        input_eval_manifest_sha256=sha(OUT/'eval_manifest.json'),code_sha256=sha(__file__),
        model_calls=0,scoring_calls=0,ratings_generated=0,automatic_promotion=False)
    immutable(DEST/'completion.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--final',action='store_true');mode.add_argument('--audit-old-units',action='store_true');a=p.parse_args()
    result=audit_preflight() if a.audit_old_units else run(a.final)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
