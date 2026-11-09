"""Read-only-result summaries and bounded review package; no model/scorer calls.

Numerics include B/W/M. Actual V2 observations include only W/M. This module
does not generate ratings, choose checkpoints, alter the protocol, or promote M.
"""
from __future__ import annotations
import argparse
import collections
import csv
import difflib
import hashlib
import io
import json
import math
import statistics
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_group_weight_match_v1'
CODE=ROOT/'r2/pilots/bench_group_weight_match_v1'
PRIOR_CODE=ROOT/'r2/pilots/bench_high_joint_coverage_v1'
GROUPS=('OLD','FOCUS','DEV','HOLDOUT','HIGH_TRAIN','HIGH_HOLDOUT','NEW_WIDE_TRAIN')
LABELS=('B','W','M')
VISUAL_LABELS=('W','M')
STAGES=('RAW','BASE_POST','CLEAN_FINAL_POST')
CONTRASTS=(('W','M'),('B','M'))
DEST=OUT/'results/delivery_v1'
MAX_PACKAGE_BYTES=10_000_000


def resolve(path):
    value=str(path).replace('\\','/')
    for prefix in ('E:/MeshFlow/','/mnt/e/MeshFlow/'):
        if value.startswith(prefix):return ROOT/value[len(prefix):]
    value=Path(value)
    return value if value.is_absolute() else ROOT/value


def read(path):return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with resolve(path).open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


def sha(path):
    h=hashlib.sha256()
    with resolve(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    path=resolve(path)
    raw=(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Refuse overwrite: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as handle:handle.write(raw)


def csvwrite(path,data):
    if not data:raise ValueError('Empty untyped table: '+str(path))
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write(path,buffer.getvalue())


def verify(book):
    for path,digest in book.items():
        if sha(path)!=digest:raise ValueError('Frozen evidence changed: '+str(path))


def finite(value):
    if value is None or value=='':return None
    result=float(value)
    return result if math.isfinite(result) else None


def macro(data,key):
    """Valid noises within C, valid C within parent, then equal valid parents."""
    by=collections.defaultdict(lambda:collections.defaultdict(list))
    for row in data:
        value=finite(row[key])
        if value is not None:by[row['parent_id']][row['condition_id']].append(value)
    parents=[statistics.fmean(statistics.fmean(values) for values in conditions.values()) for conditions in by.values()]
    return statistics.fmean(parents) if parents else None


def prerequisites(visual=False):
    completed=read(OUT/'results/completion.json')
    if completed['status']!='COMPLETE' or (completed['rows'],completed['new_score_rows'],completed['reused_score_rows'])!=(414,138,276):
        raise ValueError('Complete 414-cell numeric universe required')
    if len(completed['metric_fields'])!=177:raise ValueError('Original metric dictionary changed')
    verify(completed['files']);verify(completed['source_rows'])
    analysis=read(OUT/'results/analysis_v1/completion.json');verify(analysis['files'])
    if analysis['status']!='COMPLETE':raise ValueError('Complete numeric analysis required')
    protected={str(path):sha(path) for path in (OUT/'results/completion.json',OUT/'results/analysis_v1/completion.json')}
    if visual:
        protocol=read(OUT/'visual_review/protocol.json')
        if protocol.get('interpretation_version')!=2 or protocol['labels']!=list(VISUAL_LABELS):
            raise ValueError('Only root-frozen paired V2 W/M ratings are eligible')
        review=read(OUT/'visual_review/summary_v1/completion.json')
        if review['status']!='COMPLETE' or review['actual_observations']!=276:raise ValueError('Complete V2 actual review required')
        verify(review['files']);verify(review['sources'])
        protected.update({str(path):sha(path) for path in (OUT/'visual_review/protocol.json',OUT/'visual_review/summary_v1/completion.json')})
    return protected


def numeric():
    protected=prerequisites()
    data=rows(OUT/'results/analysis_v1/primary_parent_macro.csv')
    by={(r['report_group'],r['stage'],r['label'],r['metric']):r for r in data}
    if len(by)!=len(data):raise ValueError('Duplicate primary summary identity')
    selected={
        'CF_illegal_affected_free_fraction':('CF%',100.),
        'FF_illegal_affected_free_fraction':('FF%',100.),
        'ConflictCov_C':('I%（PASS支持域）',100.),
        'free_minimum_angle_lt5_fraction':('小角%',100.),
        'C_F_has_valid_edge':('有合法接口%',100.),
        'hinge_diag_legal_opposed_edge_C_reached_free_area_fraction':('定向可达面积%',100.),
        'actual_free_faces':('实际free面数',1.),
        'free_total_surface_area':('free面积',1.),
        'deleted_free_faces':('GEO删面数',1.),
        'hinge_diag_isolated_1face_edge_components':('合法边图单面分量',1.),
    }
    lines=['# 固定面板数值摘要','',
        'B为完成的窄HIGH，W为完成的宽HIGH，M为本轮组权重匹配候选。主比较M−W，M−B仅定量参考。三者同P0与1500步预算；不据本表自动晋级。',
        '噪声在条件内、条件在父内平均，再父等权；NA不转0。I只展示原PASS支持域，所有原177字段另存。比例转换为百分数，差值单位为百分点。','']
    compact=[]
    for group in GROUPS:
        for stage in STAGES:
            lines+=['## '+group+' / '+stage,'','|指标|B|W|M|M−W|M−B|','|---|---:|---:|---:|---:|---:|']
            for key,(name,scale) in selected.items():
                records=[by[group,stage,label,key] for label in LABELS]
                values=[finite(r['mean']) for r in records]
                values=[None if value is None else scale*value for value in values]
                deltas=[None if values[2] is None or values[index] is None else values[2]-values[index] for index in (1,0)]
                fmt=lambda value:'NA' if value is None else f'{value:.4f}'
                shown=[fmt(value)+f" ({row['valid_parents']}/{row['expected_parents']}父)" for value,row in zip(values,records)]
                lines.append('|'+name+'|'+'|'.join(shown+[fmt(value) for value in deltas])+'|')
                compact.append(dict(report_group=group,stage=stage,metric=key,display_unit=name,
                    B=values[0],W=values[1],M=values[2],M_minus_W=deltas[0],M_minus_B=deltas[1],
                    B_valid_parents=records[0]['valid_parents'],W_valid_parents=records[1]['valid_parents'],
                    M_valid_parents=records[2]['valid_parents'],expected_parents=records[2]['expected_parents']))
            lines.append('')
    area=rows(OUT/'results/analysis_v1/deletion_area_per_output.csv')
    fates=rows(OUT/'results/analysis_v1/deletion_fate_per_slot.csv')
    retention=[];reasons=[]
    area_fields=('raw_free_faces','retained_original_free_faces','deleted_original_free_faces','raw_free_area',
        'pre_geo_free_area','deleted_raw_reference_area','deleted_pre_geo_reference_area','actual_final_free_area','deleted_raw_area_fraction')
    for group in GROUPS:
        for stage in STAGES[1:]:
            for label in LABELS:
                subset=[dict(row) for row in area if (row['report_group'],row['label'],row['stage'])==(group,label,stage)]
                if not subset:raise ValueError('Missing deletion group')
                rec=dict(report_group=group,stage=stage,label=label,parents=len({r['parent_id'] for r in subset}),requests=len(subset))
                for key in area_fields:rec[key]=macro(subset,key)
                for r in subset:
                    denom=float(r['raw_free_area']);count=float(r['raw_free_faces'])
                    r['final_over_raw_area']=float(r['actual_final_free_area'])/denom if denom>0 else None
                    r['retained_original_slot_fraction']=float(r['retained_original_free_faces'])/count if count>0 else None
                rec['final_over_raw_area_parent_macro']=macro(subset,'final_over_raw_area')
                rec['retained_slot_fraction_parent_macro']=macro(subset,'retained_original_slot_fraction')
                denominator=sum(float(r['raw_free_area']) for r in subset)
                rec['pooled_final_over_raw_area']=sum(float(r['actual_final_free_area']) for r in subset)/denominator if denominator>0 else None
                rec['pooled_deleted_raw_reference_fraction']=sum(float(r['deleted_raw_reference_area']) for r in subset)/denominator if denominator>0 else None
                retention.append(rec)
                deleted=[r for r in fates if (r['report_group'],r['label'],r['stage'])==(group,label,stage) and r['retained']=='False']
                counts=collections.Counter(r['GEO_reason'] for r in deleted)
                for reason,count in sorted(counts.items()):
                    rr=[r for r in deleted if r['GEO_reason']==reason]
                    reasons.append(dict(report_group=group,stage=stage,label=label,reason=reason,deleted_slots=count,
                        request_count=len(subset),raw_reference_area_sum=sum(float(r['raw_area']) for r in rr),
                        pre_geo_reference_area_sum=sum(float(r['pre_geo_area']) for r in rr),weighting='POOLED_SUM_NOT_PARENT_MACRO'))
    csvwrite(DEST/'compact_parent_macro.csv',compact)
    csvwrite(DEST/'area_parent_macro.csv',retention)
    if reasons:csvwrite(DEST/'deletion_reason_counts.csv',reasons)
    write(DEST/'NUMERICAL_SUMMARY.md','\n'.join(lines))
    write(DEST/'AREA_SEMANTICS.md',
        '删槽面积按该原RAW槽在RAW/pre-GEO两套坐标中分别测量；不是删去来源GT的面积。FINAL/RAW面积比包含后处理位移、合并、删面及新增桥面，不能称为完整表面保留率。逐请求比率的父宏均值与总面积相除的pooled比率分别报告，不互相替代。删面原因计数是pool总数，不是父等权均值。\n')
    generated=['compact_parent_macro.csv','area_parent_macro.csv','NUMERICAL_SUMMARY.md','AREA_SEMANTICS.md']
    if reasons:generated.append('deletion_reason_counts.csv')
    result=dict(status='COMPLETE',original_score_cells=414,labels=list(LABELS),primary='M-W',secondary='M-B',
        area_macro_rows=len(retention),protected=protected,files={str(DEST/name):sha(DEST/name) for name in generated},
        code_sha256=sha(__file__),model_calls=0,scoring_calls=0,new_ratings=0)
    write(DEST/'numeric_complete.json',result);return result


def visual_joint():
    protected=prerequisites(visual=True)
    data=rows(OUT/'visual_review/summary_v1/actual_observations.csv')
    by={(r['label'],r['stage'],r['request_id']):r for r in data}
    if len(data)!=len(by) or len(data)!=276 or {r['label'] for r in data}!=set(VISUAL_LABELS):raise ValueError('V2 visual universe differs')
    for r in data:
        r['three_axis_joint']=str(r['overall_shape']=='REASONABLE' and r['C_use']=='PLAUSIBLE' and r['support_retention']=='PRESENT')
        r['joint_value']=float(r['three_axis_joint']=='True')
    summary=[];transitions=[]
    for group in GROUPS:
        for stage in STAGES:
            for label in VISUAL_LABELS:
                subset=[r for r in data if (r['report_group'],r['stage'],r['label'])==(group,stage,label)]
                summary.append(dict(report_group=group,stage=stage,label=label,requests=len(subset),
                    parents=len({r['parent_id'] for r in subset}),joint_count=sum(r['joint_value'] for r in subset),
                    joint_parent_equal_fraction=macro(subset,'joint_value'),
                    plausible=sum(r['C_use']=='PLAUSIBLE' for r in subset),reasonable=sum(r['overall_shape']=='REASONABLE' for r in subset),
                    support_present=sum(r['support_retention']=='PRESENT' for r in subset),
                    fragments_or_gaps_yes=sum(r['visible_fragment_or_gap']=='YES' for r in subset),
                    layering_or_fold_yes=sum(r['layering_or_fold']=='YES' for r in subset),
                    uncertain_any_axis=sum(any(r[k]=='UNCERTAIN' for k in ('overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold')) for r in subset)))
    for r in data:
        if r['label']!='M':continue
        a=by['W',r['stage'],r['request_id']]
        transitions.append(dict(reference='W',candidate='M',request_id=r['request_id'],parent_id=r['parent_id'],
            report_group=r['report_group'],stage=r['stage'],reference_joint=a['three_axis_joint'],candidate_joint=r['three_axis_joint'],
            transition='GAIN' if r['joint_value']>a['joint_value'] else 'LOSS' if r['joint_value']<a['joint_value'] else 'UNCHANGED'))
    for r in data:r.pop('joint_value')
    csvwrite(DEST/'three_axis_per_output.csv',data);csvwrite(DEST/'three_axis_counts.csv',summary);csvwrite(DEST/'three_axis_transitions.csv',transitions)
    write(DEST/'VISUAL_SEMANTICS.md',
        '仅使用本轮同页W/M实际AI_PROXY V2图审，B没有本轮图审，不混入历史标签。三轴交集为REASONABLE且PLAUSIBLE且PRESENT的描述性联合计数；不是新冻结主指标，不认证无碎片、无折层、无相交或拓扑接续。PRESENT按v2指主要支撑表面完整，而非仅残留腿端。UNCERTAIN不算成功且原标签保留。不用数值分数生成图审。\n')
    result=dict(status='COMPLETE',actual_observations=276,visual_labels=list(VISUAL_LABELS),interpretation_version=2,
        quantitative_only_labels=['B'],new_ratings=0,protected=protected,
        files={str(DEST/name):sha(DEST/name) for name in ('three_axis_per_output.csv','three_axis_counts.csv','three_axis_transitions.csv','VISUAL_SEMANTICS.md')})
    write(DEST/'visual_complete.json',result);return result


def archive_relative(path):
    path=resolve(path)
    try:return path.relative_to(ROOT)
    except ValueError:
        try:return Path('_external')/path.relative_to(ROOT.parent)
        except ValueError:return Path('_external')/hashlib.sha256(str(path.parent).encode()).hexdigest()[:16]/path.name


def archive():
    codefiles={}
    maps=[read(OUT/'training_registration.json')['training_code_hashes'],read(OUT/'evaluation/score_registration.json')['source_hashes']]
    for path in sorted((OUT/'evaluation/inference_registrations').glob('*.json')):maps.append(read(path)['code'])
    for mapping in maps:
        for path,digest in mapping.items():
            path=resolve(path)
            if path.suffix!='.py':continue
            if sha(path)!=digest:raise ValueError('Registered source changed: '+str(path))
            codefiles[path]=digest
    for path in CODE.glob('*.py'):codefiles[path]=sha(path)
    # Root-level launch/freeze/recovery helpers are executed source as well.
    for path in OUT.glob('*.py'):codefiles[path]=sha(path)
    records=[]
    for path,digest in sorted(codefiles.items()):
        dest=OUT/'source_archive/final'/archive_relative(path)
        if dest.exists():
            if sha(dest)!=digest:raise ValueError('Source archive differs: '+str(dest))
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as handle:handle.write(path.read_bytes())
        records.append(dict(source=str(path),archive=str(dest.relative_to(OUT)),sha256=digest,bytes=path.stat().st_size))
    write(OUT/'source_archive/final_source_hashes.json',records)
    diff=[]
    for path in sorted(CODE.glob('*.py')):
        previous=PRIOR_CODE/path.name
        before=previous.read_text(encoding='utf-8').splitlines(keepends=True) if previous.exists() else []
        diff.extend(difflib.unified_diff(before,path.read_text(encoding='utf-8').splitlines(keepends=True),
            fromfile=str(previous.relative_to(ROOT)) if previous.exists() else '/dev/null',tofile=str(path.relative_to(ROOT))))
    for path in sorted(OUT.glob('*.py')):
        diff.extend(difflib.unified_diff([],path.read_text(encoding='utf-8').splitlines(keepends=True),
            fromfile='/dev/null',tofile=str(path.relative_to(ROOT))))
    write(OUT/'code.diff',''.join(diff))
    result=dict(status='COMPLETE',actual_code_files=len(records),production_files_modified=0,
        diff_reference='Read-only previous wide HIGH source; all changes are new experiment files',
        files={str(OUT/'source_archive/final_source_hashes.json'):sha(OUT/'source_archive/final_source_hashes.json'),str(OUT/'code.diff'):sha(OUT/'code.diff')})
    write(OUT/'source_archive/complete.json',result);return result


def package():
    if read(OUT/'audits/evaluation_closeout.json')['status']!='PASS':raise ValueError('Complete closeout required')
    for name in ('REPORT.md','RUNBOOK.md','code.diff','source_archive/complete.json','results/delivery_v1/numeric_complete.json','results/delivery_v1/visual_complete.json'):
        if not (OUT/name).is_file():raise ValueError('Missing delivery prerequisite: '+name)
    target=OUT/'R2_BENCH_GROUP_WEIGHT_MATCH_REVIEW.zip'
    if target.exists():raise FileExistsError('Do not overwrite delivered package')
    files=set()
    for path in OUT.iterdir():
        if path.is_file() and path.suffix.lower() in ('.md','.json','.diff'):files.add(path)
    for directory in ('data','results','training_closeout','execution_closeout','visual_review','audits','coefficient_audit','dispatch'):
        files.update(path for path in (OUT/directory).rglob('*') if path.is_file() and path.suffix.lower() in ('.json','.csv','.md'))
    # Fixed complete training-curve sets; never selected by their result.
    for directory in ('training_closeout','execution_closeout'):
        files.update(path for path in (OUT/directory).rglob('*.png') if path.is_file())
    files.update((OUT/'source_archive/final').rglob('*.py'))
    files.add(OUT/'source_archive/final_source_hashes.json');files.add(OUT/'source_archive/complete.json')
    for name in ('generator_step1500.json','checkpoint_step1500.json','status.json'):
        path=OUT/'training/W_MATCHED'/name
        if not path.is_file():raise ValueError('Missing endpoint identity '+str(path))
        files.add(path)
    for path in (OUT/'evaluation/inference_registrations').glob('*.json'):files.add(path)
    write(OUT/'PACKAGE_CONTENTS.md',
        '包含B/W/M完整177字段、逐父配对、W/M本轮v2实际五轴图审、三轴联合描述、面积删除账本、冻结输入清单、预算与权重身份、实际源码及相对前轮code.diff、根目录执行/恢复helper源码、完整execution_closeout收据及全部固定训练曲线PNG。B只有定量参考；不将历史图审混入。小包不放权重、网格/噪声数组、训练逐调用账本或全138页图片；原始资产与图集按RUNBOOK链接保留本地。这里不是可脱离本地大资产重训的发行包。文件选择由类型/目录固定，不按结果好坏选图或选行。\n')
    files.add(OUT/'PACKAGE_CONTENTS.md')
    inventory=[dict(path=str(path.relative_to(OUT)).replace('\\','/'),sha256=sha(path),bytes=path.stat().st_size) for path in sorted(files)]
    inventory_bytes=(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as package_file:
        for path in sorted(files):
            name=str(path.relative_to(OUT)).replace('\\','/')
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            package_file.writestr(info,path.read_bytes(),compresslevel=9)
        info=zipfile.ZipInfo('PACKAGE_FILE_HASHES.json',(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
        package_file.writestr(info,inventory_bytes,compresslevel=9)
    raw=buffer.getvalue()
    if len(raw)>=MAX_PACKAGE_BYTES:
        raise RuntimeError(f'Fixed review package is {len(raw)} bytes; exceeds decimal10MB. No zip written or evidence silently dropped.')
    with zipfile.ZipFile(io.BytesIO(raw)) as package_file:
        if package_file.testzip() is not None:raise ValueError('Zip integrity failure')
    with target.open('xb') as handle:handle.write(raw)
    result=dict(status='COMPLETE',path=str(target),bytes=len(raw),sha256=sha(target),files=len(files)+1,
        bound_bytes=MAX_PACKAGE_BYTES,bound_satisfied=True,images_included=sum(path.suffix.lower()=='.png' for path in files),model_weights_included=False,
        raw_arrays_included=False,files_sha256={entry['path']:entry['sha256'] for entry in inventory})
    write(OUT/'package_receipt.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('numeric','visual','archive','package'))
    action=parser.parse_args().action
    result={'numeric':numeric,'visual':visual_joint,'archive':archive,'package':package}[action]()
    if 'torch' in sys.modules:raise RuntimeError('Delivery imported neural framework')
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','protected','files_sha256')},ensure_ascii=False,indent=2))
