"""Numerical prose derived from completed paired tables, with no quality gate."""
from __future__ import annotations
from collections import defaultdict
import math
import statistics

DISPLAY={'T1':'Native','Geo':'Native+Geo','Full':'Native+Geo+Joint'}
RMS='remainder_symmetric_RMS_bbox_pct'
COV='fixed_R12_coverage_within_1pct_fraction'
EDGE='Gamma_to_free_edges_RMS_bbox_pct'
STRICT='Gamma_to_free_edges_coverage_0.001_fraction'
CF='CF_proper_affected_free_fraction'
FF='FF_illegal_affected_free_fraction'
ANGLE='free_minimum_angle_lt5_fraction'


def valid(value):
    return isinstance(value,(float,int,bool)) and math.isfinite(float(value))


def fmt(value):
    return f'{value:.6g}' if valid(value) else 'NA'


def pick(rows,**criteria):
    return next((r for r in rows if all(r.get(k)==v for k,v in criteria.items())),{})


def delta(rows,metric,scale=1.,**criteria):
    r=pick(rows,metric=metric,**criteria)
    value=r.get('mean_paired_parent_delta')
    text=fmt(value*scale if valid(value) else None)
    return text+f"（改善{r.get('improved_parents','NA')}/{r.get('paired_valid_parents','NA')}有效父，请求{r.get('requested_parents','NA')}父）"


def mixed_label(rows,criteria):
    good=bad=0
    for metric in [RMS,COV,EDGE,CF,FF,ANGLE]:
        r=pick(rows,metric=metric,**criteria);d=r.get('mean_paired_parent_delta')
        if not valid(d):continue
        v=-d if metric==COV else d
        good+=v<0;bad+=v>0
    if good and bad:return '所列指标收益与代价并存'
    if good:return '所列可用指标均值未出现反向代价，结论仍限于该格与本次单seed训练'
    if bad:return '所列指标没有同向均值增益'
    return '可用差值不足或为零'


def paired_means(cases,comparison,split,metric):
    groups=defaultdict(list)
    for r in cases:
        if r['comparison']==comparison and r['split']==split:
            a,b=r.get(metric+'_before'),r.get(metric+'_after')
            if valid(a) and valid(b):groups[r['uid']].append((a,b))
    before=[];after=[]
    for values in groups.values():
        before.append(statistics.fmean(v[0] for v in values))
        after.append(statistics.fmean(v[1] for v in values))
    return ((statistics.fmean(before),statistics.fmean(after),len(groups))
            if groups else (None,None,0))


def evidence_lines(a_summary,increments,width_summary,width_pairs,width_pair_summary,analytic_summary,holdout_summary):
    lines=['','以下为实际结果的直接回答。差值均先在父内对相同输入配对，再对父等权；“改善父”按原指标方向计数，没有临时成功阈值。']
    lines+=['','新上下文上的Geo增量与Joint增量：']
    for split in ('train','validation'):
        for ratio in (.2,.4):
            for comparison,label in [('Native+Geo_minus_Native','Geo−Native'),('Native+Geo+Joint_minus_Native+Geo','Joint−Geo')]:
                criteria=dict(comparison=comparison,phase_step=3000,table='A',split=split,ratio=ratio)
                lines.append(f"{split}、K={ratio:.0%}、{label}：RMS差 {delta(increments,RMS,**criteria)} %L；GT覆盖差 {delta(increments,COV,100.,**criteria)} 点；Gamma边RMS差 {delta(increments,EDGE,**criteria)} %L；CF面比例差 {delta(increments,CF,100.,**criteria)} 点；FF面比例差 {delta(increments,FF,100.,**criteria)} 点；小角面差 {delta(increments,ANGLE,100.,**criteria)} 点。{mixed_label(increments,criteria)}。")
    lines+=['','free是否随当前C宽度响应：以下斜率与单调统计只来自free，复制的C不进入宽度。斜率1为理想全局缩放响应；绝对误差与几何覆盖仍须共同解释。']
    for arm,label in DISPLAY.items():
        for split in ('train','validation'):
            base=pick(width_summary,arm=arm,phase_step=3000,split=split,alpha=1.)
            low=pick(width_summary,arm=arm,phase_step=3000,split=split,alpha=.9)
            high=pick(width_summary,arm=arm,phase_step=3000,split=split,alpha=1.1)
            lines.append(f"{label} {split}终点：斜率均值 {fmt(base.get('slope'))}，|斜率−1| {fmt(base.get('slope_abs_error'))}，单调比例 {fmt(base.get('monotonic'))}；alpha=.9/1.1的绝对宽度误差/L0分别 {fmt(low.get('X_span_abs_error_L0'))}/{fmt(high.get('X_span_abs_error_L0'))}。斜率有效case {base.get('slope_valid_cases','NA')}/{base.get('requested_cases','NA')}，有效父 {base.get('slope_valid_parents','NA')}/{base.get('requested_parents','NA')}；Y/Z尺度与坐标中位数副作用另列，不把未知项填成理想响应。")
    lines+=['','响应是否在起点存在及新增训练贡献：下列仅使用真正共有的第一噪声、三个alpha，每父等权；不混入终点第二噪声。']
    for arm,label in DISPLAY.items():
        for split in ('train','validation'):
            comparison=label+'_width_endpoint_minus_start'
            wa,wb,n=paired_means(width_pairs,comparison,split,'X_span_abs_error_L0')
            ra,rb,_=paired_means(width_pairs,comparison,split,'response_abs_residual')
            sa,sb,_=paired_means(width_pairs,comparison,split,'slope_abs_error')
            criteria=dict(comparison=comparison,table='B',split=split,ratio=.2)
            lines.append(f"{label} {split}共享输入：宽度绝对误差/L0 {fmt(wa)}→{fmt(wb)}；|响应残差| {fmt(ra)}→{fmt(rb)}；|斜率−1| {fmt(sa)}→{fmt(sb)}。宽度误差配对变化 {delta(width_pair_summary,'X_span_abs_error_L0',**criteria)}；这里有 {n} 个宽度误差可比父，起点已有的响应不归功于本轮训练。")
    lines+=['','相对零前向解析参照：差值为当前模型减参照，在同父、同噪声、同当前alpha上配对，三alpha平均。误差负值、覆盖正值才有利于模型。Post-scale额外知道编辑前C与旧输出，因此只能比较配对编辑场景的实效。']
    for arm,label in DISPLAY.items():
        for split in ('train','validation'):
            for reference in ('No-response','Post-scale'):
                criteria=dict(comparison=label+'_minus_'+reference,phase_step=3000,table='B',split=split,ratio=.2)
                lines.append(f"{label} {split} − {reference}：宽度绝对误差/L0差 {delta(analytic_summary,'X_span_abs_error_L0',**criteria)}；原双向RMS差 {delta(analytic_summary,RMS,**criteria)} %L；GT覆盖差 {delta(analytic_summary,COV,100.,**criteria)} 点；严格Gamma覆盖差 {delta(analytic_summary,STRICT,100.,**criteria)} 点；FF面比例差 {delta(analytic_summary,FF,100.,**criteria)} 点。")
    lines+=['','与Post-scale持平或更差的项目不能被解释为超越简单仿射编辑。即使某些项目更好，单次全形状X向仿射任务仍只支持域内整体几何响应，不能证明语义部件、连接重构或复杂结构推理。']
    lines+=['','20%与40%条件是否不同：以下是各自冻结新C/自身free补集上的终点绝对数值，保留不适用分母。不同K改变了已知面与free目标，不能把这些描述当成仅信息量的严格因果对照。']
    for arm,label in DISPLAY.items():
        for split in ('train','validation'):
            low=pick(a_summary,table='A',arm=arm,phase_step=3000,split=split,ratio=.2)
            high=pick(a_summary,table='A',arm=arm,phase_step=3000,split=split,ratio=.4)
            lines.append(f"{label} {split}：20%/40%的RMS为 {fmt(low.get(RMS))}/{fmt(high.get(RMS))} %L，GT覆盖为 {fmt(low.get(COV))}/{fmt(high.get(COV))}；有效父/请求父分别 {low.get('valid_parents','NA')}/{low.get('requested_parents','NA')} 与 {high.get('valid_parents','NA')}/{high.get('requested_parents','NA')}。实际C面积分层另存A_context_area_strata.csv。")
    lines+=['','已观察留出是否新增退化：以下新累计4000减源1000，使用原实际C/噪声与原评分器。未设置临时“明显”阈值；相交、覆盖、RMS与角质量的方向分别保留。']
    for arm,label in DISPLAY.items():
        criteria=dict(comparison=label+'_new4000_minus_old1000',table='C',split='holdout',ratio=.2)
        lines.append(f"{label}：RMS差 {delta(holdout_summary,RMS,**criteria)} %L；GT覆盖差 {delta(holdout_summary,COV,100.,**criteria)} 点；严格Gamma覆盖差 {delta(holdout_summary,STRICT,100.,**criteria)} 点；CF面比例差 {delta(holdout_summary,CF,100.,**criteria)} 点；FF面比例差 {delta(holdout_summary,FF,100.,**criteria)} 点；小角面差 {delta(holdout_summary,ANGLE,100.,**criteria)} 点。{mixed_label(holdout_summary,criteria)}。")
    return lines
