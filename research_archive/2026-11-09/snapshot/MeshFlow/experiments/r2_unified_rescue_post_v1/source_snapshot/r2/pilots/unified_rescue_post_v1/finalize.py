"""Package executed CPU evidence and report; never calls a model or scorer."""
from __future__ import annotations
import collections,csv,difflib,json,shutil
from pathlib import Path
import numpy as np
from . import run as R
C=R.C

def rows(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def number(x):
    if x in ('',None):return None
    if x in ('True','False'):return int(x=='True')
    return float(x)

def finish():
    root=R.ROOT;reg=C.read(root/'registration.json');completed=C.read(root/'completion.json')
    table=C.read(root/'tables/completion.json');R.E.verify(completed['files']);R.E.verify(table['files']);R.E.verify(reg['protected']);contract=R.verify_code(reg)
    out=rows(root/'tables/per_output.csv');sums=rows(root/'tables/summary_parent_equal.csv');edges=rows(root/'tables/connectivity_ledger.csv');guard=rows(root/'tables/request_guard_counts.csv')
    by={(r['panel'],r['request_id'],r['stage']):r for r in out};summary=[];positives=[]
    for panel in sorted({j['panel'] for j in reg['jobs']}):
        jobs=[j for j in reg['jobs'] if j['panel']==panel]
        row=dict(panel=panel,requests=len(jobs),open_C=sum(j['q']['boundary_edges']>0 for j in jobs),
            closed_C=sum(j['q']['boundary_edges']==0 for j in jobs),old_final_chain=jobs[0]['old_final_chain'])
        for stage in ('GUARD_GEO','FINAL'):
            ee=[x for x in edges if x['panel']==panel and x['stage']==stage]
            for field in ('CF_baseline_legal_edges','CF_lost_legal_edges','CF_new_legal_edges','FF_baseline_legal_edges','FF_lost_legal_edges','FF_new_legal_edges',
                'CF_new_illegal_pairs','FF_new_illegal_pairs','CF_illegal_pairs_removed','FF_illegal_pairs_removed','deleted_source_faces'):
                row[stage+'_'+field]=sum(int(number(x[field])) for x in ee)
        gg=[x for x in guard if x['panel']==panel]
        for field in ('floor_triggered_faces','anti_protected_faces','anti_rejected_assignments','protected_deleted_faces'):
            row[field]=sum(int(number(x[field])) for x in gg)
        for stage in ('OLD_BASE','OLD_FINAL','GUARD_GEO','FINAL'):
            rr=[x for x in out if x['panel']==panel and x['stage']==stage]
            valid=[x for x in rr if number(x.get('bulk_reach_ge_0_30_descriptive')) is not None]
            row[stage+'_bulk_ge30']=sum(number(x['bulk_reach_ge_0_30_descriptive'])==1 for x in valid);row[stage+'_bulk_denominator']=len(valid)
        oldpos=keep=newpos=lost=0
        for j in jobs:
            rid=j['q']['request_id'];a,b=by[panel,rid,'OLD_FINAL'],by[panel,rid,'FINAL']
            x,y=number(a.get('bulk_reach_ge_0_30_descriptive')),number(b.get('bulk_reach_ge_0_30_descriptive'))
            status='NA' if x is None or y is None else 'RETAINED' if x and y else 'LOST' if x and not y else 'NEW' if not x and y else 'REMAINS_BELOW'
            oldpos+=x==1;keep+=status=='RETAINED';lost+=status=='LOST';newpos+=status=='NEW'
            positives.append(dict(panel=panel,request_id=rid,parent_id=j['q']['parent_id'],split=j['q']['split'],status=status,
                old_FINAL_bulk_ge30=x,new_FINAL_bulk_ge30=y,old_FINAL_bulk_area_fraction=number(a.get('free_only_reached_bulk_area_fraction')),
                new_FINAL_bulk_area_fraction=number(b.get('free_only_reached_bulk_area_fraction')),
                old_FINAL_legal_CF=number(a['C_F_valid_edge_count']),new_FINAL_legal_CF=number(b['C_F_valid_edge_count']),
                claim='STRUCTURAL_DESCRIPTOR_ONLY; new visual joint review separate'))
        row.update(old_FINAL_positive_count=oldpos,retained_old_FINAL_positives=keep,lost_old_FINAL_positives=lost,new_FINAL_positives=newpos);summary.append(row)
    C.csvwrite(root/'panel_outcomes.csv',summary);C.csvwrite(root/'strong_positive_pairing.csv',positives)
    fields={'FF':'FF_illegal_affected_free_fraction','CF':'CF_illegal_affected_free_fraction','I':'ConflictCov_C','小角':'free_minimum_angle_lt5_fraction',
        '删面':'deleted_free_faces','非流形边':'nonmanifold_edges','bulk可达':'free_only_reached_bulk_area_fraction'}
    slookup={(r['panel'],r['stage'],r['group'],r['metric']):number(r['mean_parent_equal']) for r in sums if r['grouping']=='split'}
    def fmt(v,pct=False):return 'NA' if v is None else f'{v*(100 if pct else 1):.3f}'
    lines=['# 统一CPU保护后处理：140份冻结缓存实测','',
        '**状态：固定140份现成RAW/D0缓存已处理与评分完成；本阶段零模型、零新增D0、零噪声。** 两个保护不是无损修复；以下同时保留恢复面、接续及新增冲突/旧边损失。','',
        '## 范围与因果口径','',
        'Bench两个终点各48份，Chair BACK16份，Chair SEAT28份。`OLD_BASE→GUARD_GEO`隔离两个保护的组合；`OLD_FINAL→FINAL`比较实际旧新终态。Chair旧终态已有HINGE+ALPHA+HOLE；Bench旧终态只有HINGE，所以Bench终态增量还包含ALPHA/HOLE，不能全部归功于两个保护。',
        '闭合C不需要开放接口，边界/SEAM接续列为NA。所有旧评分、面板、阈值、查询点及来源几何保持只读。I仍是C查询冲突覆盖，并非新增非法面对计数。无唯一来源GT款式排名。','',
        '## 完整父等权质量比较','',
        '表内FF/CF/I/小角/bulk可达为百分比；删面、非流形边为数量均值。先噪声、再条件、再父等权；各面板分别汇总。','',
        '| 面板 / 子集 | 阶段 | FF% | CF% | I% | 小角% | 删面 | 非流形边 | bulk可达% |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for panel in sorted({j['panel'] for j in reg['jobs']}):
        for split in ['ALL',*sorted({j['q']['split'] for j in reg['jobs'] if j['panel']==panel})]:
            for stage in ('OLD_BASE','OLD_FINAL','GUARD_GEO','FINAL'):
                vals=[fmt(slookup.get((panel,stage,split,v)),k not in ('删面','非流形边')) for k,v in fields.items()]
                lines.append('| '+panel+'/'+split+' | '+stage+' | '+' | '.join(vals)+' |')
    lines+=['','## 真实旧连接保留与新增冲突','',
        '以下均对原OLD_BASE、按RAW来源面/角身份配对。它们是总条数/总面对数，不是父等权比例。guard本身引入的损失没有被后续HINGE零新增护栏掩盖，也没有触发静默整例回滚。','',
        '| 面板 | 阶段 | 丢旧CF边 | 丢旧FF边 | 新合法CF边 | 新合法FF边 | 新非法CF对 | 新非法FF对 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        for stage in ('GUARD_GEO','FINAL'):
            keys=('CF_lost_legal_edges','FF_lost_legal_edges','CF_new_legal_edges','FF_new_legal_edges','CF_new_illegal_pairs','FF_new_illegal_pairs')
            lines.append('| '+r['panel']+' | '+stage+' | '+' | '.join(str(r[stage+'_'+k]) for k in keys)+' |')
    lines+=['','## 旧结构正例的保留','',
        '这里只把“合法C种子沿free合法共边达到至少30%原BULK面积”作为既有描述标记；不能代替新的视觉质量/用途联合成功。原RAW没有变化，但新的后处理仍可能损坏原有形状。','',
        '| 面板 | 适用请求 | 旧FINAL≥30% | 保留 | 丢失 | 新增 | 新FINAL≥30% |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append('| '+r['panel']+' | '+' | '.join(str(r[k]) for k in ('FINAL_bulk_denominator','old_FINAL_positive_count','retained_old_FINAL_positives','lost_old_FINAL_positives','new_FINAL_positives','FINAL_bulk_ge30'))+' |')
    lost=[r for r in positives if r['status']=='LOST']
    lines+=['','丢失身份：'+('；'.join(r['panel']+'/'+r['request_id'] for r in lost) if lost else '无（按本结构描述标记）。'),'',
        '## 保护、位移与执行账本','',
        '逐面alpha由三条边二次约束首次进入禁区决定，允许切触、禁止跳过中间不安全区；最终FP32写回必要时向RAW方向退alpha ULP，超过原257步时按坐标舍入区间端点解析退步，不假定边长单调。ANTI_SELF_COLLAPSE保留原GEO顺序并预约未处理角点位置。它不拆开进入GEO前已经重合的角点，也不禁止精确重复面清理。','',
        'Hinge/Alpha的1.5%L预算从新的guarded PRE_GEO计算；相对原RAW和原D0提议的总位移另见`displacement/per_output.csv`与逐面表。C及所有留存SEAM硬锚逐字节和索引合同均检查，BRIDGE没有RAW来源身份。','',
        f'修订正式执行：{completed["requests"]}请求，最多{completed["workers"]}个单线程CPU worker，最后一次恢复控制器墙钟{completed["seconds"]:.2f}秒，140个完整格墙钟之和{completed["sum_completed_cell_seconds"]:.2f}秒（并发相加，并非总历时）；280份新完整评分，旧BASE/FINAL不重评分。线程竞争下墙钟不可当孤立算法基准。',
        '初始工程尝试单独保留：6请求启动，4完整、2部分，10份完整评分；修复的是后来角点已经占据精确目标坐标的边界情形。全部旧尝试、源码及注册保留于`engineering_attempt_00`，没有模型/D0/噪声重试。','',
        '第二次数值边界：Chair SEAT `73b369b9_n1/slot179` 在原257次alpha ULP退步后，写入最短边比固定下限少1.56296764e-8模型单位；按一次坐标舍入事件继续至总261 ULP后通过。113个完成格的保护数组逐字节不变且不重算；失败格在GEO/评分之前停止，保留于`engineering_attempt_01`。27格采用显式V2执行注册，科学门槛不变，原V1与V2源码均验hash。','',
        '## 交付与恢复','',
        '- `tables/per_output.csv`：四阶段完整逐例数值，全部原135字段保留，已有179扩展不覆盖旧定义。',
        '- `tables/paired_per_output.csv`、`summary_parent_equal.csv`、`per_parent.csv`：完整配对与分母。',
        '- `tables/connectivity_ledger.csv`及每格JSON：丢失/新增原边与非法面对、删面和面积。',
        '- `tables/floor_per_face.csv`：每个RAW槽的门槛、alpha、位移与保护资格。',
        '- `strong_positive_pairing.csv`：旧结构描述正例保留/丢失身份；图审另列。',
        '- `cells/`：guarded PRE、GEO、HINGE、ALPHA、FINAL及全部候选/拒绝/评分。',
        '- `source_snapshot/`、`source_hashes.json`、`code.diff`：实际CPU实现及原依赖hash。',
        '- 恢复：`python -B -m r2.pilots.unified_rescue_post_v1.resume --workers 4`；已完成阶段校验后跳过，未完成异常目录不静默覆盖。','',
        '本报告不自动晋级后处理、切换模型、训练或提交Git。是否值得保留必须同时考虑完整表、旧结构正例和实际图审，不能从删面数下降单独判断。']
    text='\n'.join(lines)+'\n';report=root/'REPORT.md'
    if report.exists() and report.read_text(encoding='utf-8')!=text:raise FileExistsError('Refusing report overwrite')
    report.write_text(text,encoding='utf-8')
    code=dict(contract['code']);code.update({str(p):C.sha(p) for p in Path(__file__).parent.glob('*.py')})
    for rel in ('r2/pilots/chair_back_transfer_v1/post_cpu.py','r2/pilots/chair_back_transfer_v1/evaluate.py',
        'r2/pilots/seam_slot_c200_v1/postprocess.py','r2/pilots/seam_slot_c200_v1/generate.py',
        'r2/pilots/seam_pool47_couple_v1/endpoint_scoring.py'):
        p=C.ROOT/rel;code[str(p)]=C.sha(p)
    for path,digest in code.items():
        p=C.resolve(path)
        try:rel=p.relative_to(C.ROOT)
        except ValueError:rel=Path('external')/p.name
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert C.sha(dest)==digest
        else:shutil.copy2(p,dest)
    C.write(root/'source_hashes.json',dict(code=code,protected_inputs=reg['protected'],model_calls=0,
        execution_revisions=completed['execution_revisions'],prior_code_snapshots=contract.get('prior_code_snapshots',{})))
    original=(C.ROOT/'r2/src/meshflow_control/geo_c.py').read_text(encoding='utf-8').splitlines(True)
    guardtext=Path(R.guard_geo.__file__).read_text(encoding='utf-8').splitlines(True)
    diff=''.join(difflib.unified_diff(original,guardtext,fromfile='read_only/r2/src/meshflow_control/geo_c.py',tofile='new/unified_rescue_post_v1/guard_geo.py'))
    for p in sorted(Path(__file__).parent.glob('*.py')):
        if p.name!='guard_geo.py':diff+=''.join(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(C.ROOT))))
    (root/'code.diff').write_text(diff,encoding='utf-8')
    if not (root/'resources_after.json').exists():C.write(root/'resources_after.json',C.resources())
    C.write(root/'report_ready.json',dict(status='CPU140_COMPLETE',requests=140,model_calls=0,new_D0=0,new_noise=0,
        report_sha256=C.sha(report),tables_sha256=C.sha(root/'tables/completion.json'),code_diff_sha256=C.sha(root/'code.diff'),
        positive_pairing_sha256=C.sha(root/'strong_positive_pairing.csv'),old_assets_hash_verified=len(reg['protected'])))
    print(json.dumps(summary,indent=2));return summary
if __name__=='__main__':finish()
