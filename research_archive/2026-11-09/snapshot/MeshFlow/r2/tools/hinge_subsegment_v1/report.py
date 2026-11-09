"""Close out the finite CPU test from completed evidence, without new geometry."""
from __future__ import annotations
import collections
import difflib
import hashlib
import json
from pathlib import Path
from .summarize import OUT, ROOT, read, write


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    trials = load(OUT/'trials/all_trials.json')
    execution = load(OUT/'trials/completion.json')
    audit = load(OUT/'independent_split_audit.json')
    metrics_audit = load(OUT/'independent_metrics_audit.json')
    assert 'PASS' in audit['status'] and 'PASS' in metrics_audit['status']
    score = load(OUT/'scores/completion.json')
    census = load(OUT/'census/completion.json')
    stats = {(r['group'], r['metric']):r for r in read(OUT/'metric_summary.csv')}
    rows = {(r['stage'],r['request_id']):r for r in read(OUT/'scores/per_output.csv')}
    stages = ('HINGE_OPP015','SPLIT015','SPLIT025')
    table = []
    shapes = []
    for r in trials:
        children = [x for x in r['affected_face_shapes'] if x['is_new_child']]
        table.append(dict(request_id=r['request_id'], split=r['split'], delta_fraction=r['delta_fraction'],
            candidate_rank=r['rank'], parent_seam_raw_slot=r['split_parent_source_raw_slot'],
            parent_bulk_raw_slot=r['candidate']['bulk_source_raw_slot'], decision=r['decision'],
            all_guard_failures=';'.join(r['all_guard_failures']),
            child_min_angle=min(x['minimum_angle'] for x in children),child_max_q=max(x['q'] for x in children),
            new_illegal_CF=len(r['new_illegal_parent_pairs']['CF']),new_illegal_FF=len(r['new_illegal_parent_pairs']['FF']),
            new_nonmanifold=len(r['new_nonmanifold']),new_same_winding=len(r['new_same_winding']),
            lost_old_legal_edges=len(r['lost_legal_edges']),original_cluster_max_displacement=r['original_PRE_max_displacement'],
            cumulative_displacement_fraction_L=r['original_PRE_max_displacement']/census.get('L', 2*3**.5/.3762),
            open_before=r['open_edges_before'],trial_open_after=r['open_edges_after'],
            area_error=r['split_surface_errors']['area_difference'],plane_error=r['split_surface_errors']['plane_distance']))
        for face in r['affected_face_shapes']:
            shapes.append(dict(request_id=r['request_id'],delta_fraction=r['delta_fraction'],rank=r['rank'],decision=r['decision'],**face))
    write(OUT/'trial_summary.csv',table)
    write(OUT/'trial_affected_face_shapes.csv',shapes)
    review_notes = {
        '85b16941_n0':'已查看三列两外观及原front/side截面。宏观椅形及座面剖面无肉眼可辨显著变化；不能从整体图认证那一条细小新增共边。',
        '2d08a64e_n0':'三列外观与原位置截面相同；座下分段轮廓及局部错层仍在，未观察到新闭口。对应试算已全部回滚。',
        '608b46b0_n3':'三列相同；正截面下方多段短线/间隔、侧截面长段之间缺口仍在。没有把被拒分裂当成成品改善。',
        '610b1621_n3':'三列相同；座下两侧内层短段和主体间的间隔仍可见。仅从截面不能认证三维漂浮或非法相交。'}
    images = load(OUT/'gallery/completion.json')['images']
    review = [dict(request_id=i['request_id'], reviewer='AI_PROXY', blind=False, human_review=False,
                  actual_viewed=True, image_sha256=i['image_sha256'], image=i['path'],
                  note=review_notes[i['request_id']]) for i in images]
    write(OUT/'visual_review.csv',review)

    lines = ['# HINGE seam侧边子段分裂：纯CPU有限验证', '',
        '**结论：完成实施和实际运行，未获得DEV接续突破，不把本候选晋级为新的质量方案。** 两档都只在同一个TRAIN请求 `85/n0` 接受一次局部修补；DEV全部回滚，仍为0/12达到30% bulk可达。保留现有 `PLAIN_S2000 + HINGE_OPP015`，本轮代码和失败证据独立归档。', '',
        '## 范围与合同', '',
        '同一28份冻结产物（16 TRAIN、12 DEV），两档均从已修正绕序守卫的HINGE_OPP015出发；没有用HINGE_OPP025作为压力档起点。δ=1.5%L与2.5%L独立执行，L沿用原评分固定尺度。零训练、零模型前向、零GPU计算、零新噪声、零D0、零来源Y查询。', '',
        '先扫描593个真实共享顶点的边对（两档合计1186行）。正式投影窗t∈[0.15,0.85]、轴角≤20°，并核对全簇原PRE_GEO角点到FP32目标的累计位移。宽窗[0.1,0.9]仅作摸底，未用于放行。正式候选只有3/5个，不支持由82.34%的共点关联面积推断普遍存在可分裂接续。', '',
        '只拆生成seam侧边，不拆C边、不改C坐标/索引/绕序。复用bulk端点的真实顶点ID；子面独立child ID、原POST父面、原RAW槽位、重心角点来源分别保存。新内部子面边单独记账。子面角≥5°且q≤10；受移动影响的所有邻面受形状/法线守卫约束；禁止新增非法CF/FF、未知接触、非流形和同向双面关系，并保留原合法边。守卫失败不清理坏面、不松阈值、不另跑配置。', '',
        '最近的旧路线为R05/R10、整边绑定与HINGE。这里的实质变化是允许生成面做共面细分并匹配短子段；历史整边匹配没有测试这个操作。本轮验证有限条件下的可行性与代价，不是“物理上限”或对全部末端修复的否定。', '',
        '## 候选与实际裁决', '',
        '| 档位 | TRAIN候选/接受 | DEV候选/接受 | 新增达到30% bulk可达的请求 |',
        '|---|---:|---:|---:|',
        '|1.5%L|1 / 1|2 / 0|0|',
        '|2.5%L|1 / 1|4 / 0|0|', '',
        '**两个接受是同一请求的两档结果，不是两份独立正例。** 所有56个终态保存；54格几何与基线逐位一致，旧评分明确继承，只有两格接受产物重新评分。', '',
        '| 请求/源seam槽 | 档位 | 最小子面角 | 最大子面q | 新非法FF | 裁决 |',
        '|---|---|---:|---:|---:|---|']
    for r in table:
        lines.append(f"|{r['request_id']}/s{r['parent_seam_raw_slot']}|{100*r['delta_fraction']:.1f}%L|{r['child_min_angle']:.3f}°|{r['child_max_q']:.3f}|{r['new_illegal_FF']}|{r['decision']}|")
    lines += ['',
        '- `608/n3`：三次试算没有新增非法面对或损失旧合法边，但产生约2.58°—4.14°子面，q最高13.36。它说明有局部子段几何机会，**不满足本轮禁止新增瘦长碎面的合同**；不是“完全找不到附近边”。',
        '- `610/n3@2.5%L`：子面6.82°/10.06°，形状与拓扑守卫通过；整簇移动却新增5对非法FF，因此回滚。仅盯接续三角形会漏掉邻域代价。',
        '- `2d/n0`：最窄子面0.764°、q44.27，另有邻面恶化、新非流形2条、新FF1对、旧合法边丢1条。',
        '- `85/n0`：通过全部守卫，但连接的bulk早已可达。新seam–bulk边1条，长度0.150216模型单位；另1条是细分内部边。开放边38→37，即净减1条，不是附件预期的2条。', '',
        '## 全部指标与分母核对', '',
        '原135字段完整保留，另加15个固定原面分母字段，共84行、8400条配对值。表中比例为逐请求宏均值，ALL按16 TRAIN/12 DEV组成；实际子面分母与原面并集分母分开。', '',
        '| ALL指标 | HINGE_OPP015 | +SPLIT1.5% | +SPLIT2.5% |',
        '|---|---:|---:|---:|']
    definitions = [
        ('FF，实际子面分母','FF_illegal_affected_free_fraction','pct'),
        ('FF，固定原面并集分母','fixed_parent_FF_illegal_affected_fraction','pct'),
        ('CF，固定原面并集分母','fixed_parent_CF_illegal_affected_fraction','pct'),
        ('I，原固定查询','ConflictCov_C','pct'),
        ('小角面，实际子面分母','free_minimum_angle_lt5_fraction','pct'),
        ('小角面，固定原面并集分母','fixed_parent_minimum_angle_lt5_fraction','pct'),
        ('合法seam–bulk边总数','legal_seam_bulk_edge_count','sum'),
        ('合法图全部共边总数（含细分内部边）','graph_valid_edges','sum'),
        ('开放边总数','boundary_edges','sum')]
    for label,key,kind in definitions:
        r=stats['ALL',key]
        values=[float(r[s+('_sum' if kind=='sum' else '_mean')]) for s in stages]
        display=[f'{100*x:.6f}%' if kind=='pct' else f'{x:.0f}' for x in values]
        lines.append('|'+label+'|'+'|'.join(display)+'|')
    lines += ['',
        '**85/n0的FF“下降”完全来自分母变化：受影响原面仍27张，27/148=18.24324%，分裂后27/149=18.12081%；固定原面口径仍18.24324%。没有消除非法FF面对。** 小角面数量也未减少，实际比例变化同样含分母效应。不能把表里的微降包装成降FF/降碎面的新正结果。', '',
        '该例bulk可达面积比例46.277963%→46.279482%，可达bulk面身份未扩大；这是移动造成面积的微变。直接接触C的子面面积减少0.00276227模型平方单位，是把原seam拆成“直接接C子面＋通过内部边可达子面”的记账改变，不是丢C共边。全部字段及不利数值仍保留于all_regressions.csv。', '',
        '| 分组 | ≥30% bulk可达（前→后） | 固定原面FF（前→后） | I（前→后） |',
        '|---|---:|---:|---:|']
    for group,n in [('TRAIN',16),('DEV',12)]:
        reach=stats[group,'bulk_reach_ge_0_30_descriptive'];ff=stats[group,'fixed_parent_FF_illegal_affected_fraction'];ic=stats[group,'ConflictCov_C']
        lines.append(f"|{group}|{float(reach['HINGE_OPP015_sum']):.0f}/{n} → {float(reach['SPLIT015_sum']):.0f}/{n}|{100*float(ff['HINGE_OPP015_mean']):.5f}% → {100*float(ff['SPLIT015_mean']):.5f}%|{100*float(ic['HINGE_OPP015_mean']):.5f}% → {100*float(ic['SPLIT015_mean']):.5f}%|")
    lines += ['',
        '两档终态逐请求相同。DEV全部135字段保持原值，bulk合法可达宏均0、达到30%为0/12；A-VMR、PMR与C/F合法共边没有变化。原4729条合法边逐身份保留，零额外删面，终态无新增非法CF/FF、同向双面边或非流形；原有缺陷未被宣称修复。', '',
        '## 图审与数值精度', '',
        '实际查看了四个候选请求的12列结果：统一原外观相机与front/side截面，无逐结果重新居中。AI_PROXY、非盲、非人类评分：85宏观外观/截面没有明显变化；三份DEV最终图均未改变，608座下分段与间隔、610座下内层短段仍可见。完整记录见visual_review.csv；截面中的线段不能单独认证三维悬片。', '',
        '另外保存并实际查看5个唯一试算的局部科学图（trial_gallery/v2_readable/）：608的细长子面在等比例投影中可见；610的新增非法面对参与邻面用颜色标出。拒绝图明确标为NOT final，不能作为成功输出。局部图的AI_PROXY记录独立保存。', '',
        '接纳例子面面积和相对原seam误差2.59e−9模型平方单位，投影离原平面5.01e−9模型单位。实测误差很小，但不写“FP32数学上100%共面/零面积误差”。原C字节/拓扑严格不变。', '',
        '## 执行和可恢复性', '',
        f"只读census {census['seconds']:.2f}秒；8次完整guard试算/56格保存 {execution['seconds']:.2f}秒；完整指标整理阶段 {score['seconds']:.2f}秒。代码实现、独立审计、图审与归档另计，附件的“五分钟”不是本轮端到端实测承诺。",
        '单个几何进程1线程，实测峰值RSS约113MiB；24逻辑CPU、约61GiB可用内存，独立源码/账本审阅并行。任务很小无需多进程加载；GPU任务始终0。每阶段实际资源记录在resources/与scores/。',
        '独立census逐对核对1186行；独立split重建8次试算并核验全部56格；child-aware评分先在不改动baseline上核对135字段等价。评分收集有两次发生在新增评分之前的工程修正（CSV数值字符串显式转换、资源快照文件名），原日志保留，无扩展科学配置或隐瞒重新采样。', '',
        '恢复入口：`python -B -m r2.tools.hinge_subsegment_v1.resume`。完成资产校验后跳过；失败尝试保留，不覆盖原实验。新文件都在本实验与工具独立目录，没有修改官方模型、原评分器、旧输出、Git默认或上传分支。', '',
        '## 下一步判断', '',
        '**本轮不建议继续扩大这套固定阈值子段分裂作为DEV解法。** 它可完成一个安全的小型拓扑修补，但没有扩大已接通请求覆盖，也没有降低固定分母FF/I。若以后研究更一般的局部重三角化，必须把“在维持子面质量与邻域不自交的条件下接通”作为新的待验证任务；本轮没有执行这条新路线，也没有自动放松5°/q10或零新FF守卫。', '',
        '证据入口：`scores/per_output.csv`（完整逐输出）；`paired_per_output.csv`/`paired_per_parent.csv`；`trial_summary.csv`/`trial_affected_face_shapes.csv`；`trials/all_trials.json`及失败attempt数组；`census/`；三个independent审计回执；`gallery/`；源码hash、code.diff与运行回执。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return dict(report=str(OUT/'REPORT.md'),trials=len(trials),accepted=execution['accepted'])


if __name__=='__main__':
    print(json.dumps(run()),flush=True)
