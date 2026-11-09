"""Add evidence-specific decisions to the completed report; no experiment calls."""
import hashlib
import json
from pathlib import Path
from statistics import mean

OUT = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads((OUT/path).read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert read('RUN_COMPLETE.json')['status'] == 'DONE'
assert read('independent_audit/audit_result.json')['status'] == 'PASS'
assert read('independent_audit/results_arithmetic_audit.json')['status'] == 'PASS'
result = read('paired_results/results.json')
primary = next(r for r in result['primary'] if r['comparison'] == 'OT_HYBRID-OT_BASE')
quality = read('paired_results/quality_paired_summary.json')
def metric(name, table='A', split='validation', ratio=.2):
    return next(r for r in quality if r['comparison']=='OT_HYBRID-OT_BASE' and
                (r['metric'], r['table'], r['split'], r['ratio']) == (name, table, split, ratio))
coverage = metric('coverage_fraction'); tail = metric('fixed_query_p95_L')
angles = metric('free_minimum_angle_lt5_fraction')
hold_cov = metric('coverage_fraction', 'C', 'holdout')
hold_cf = metric('CF_proper_affected_free_fraction', 'C', 'holdout')
train40_cov = metric('coverage_fraction', 'A', 'train', .4)
train40_gamma = metric('Gamma_to_free_edges_RMS_bbox_pct', 'A', 'train', .4)
curves = [r for r in read('paired_results/width_per_curve.json') if r['candidate']=='MODEL' and
          r['split']=='validation' and r['checkpoint_label']=='OT_HYBRID']
assert len(curves) == 8
all_correct = sum(r['all_three_strict_win'] for r in curves)
universal = sum(r['one_input_strictly_wins_all_targets'] for r in curves)
parts = [r for r in read('paired_results/coverage_partition.json') if r['comparison']=='OT_HYBRID-OT_BASE'
         and r['table']=='A' and r['split']=='validation' and r['ratio']==.2 and r['partition']=='both_out']
tail_contribution = mean(r['fixed_query_MSE_contribution'] for r in parts)
priority = ('优先做现有 raw 的小角面收益归因，先核查主几何收益是否依赖新增细长三角面。'
            '假说：HYBRID 新增的最小角<5°面承担了不成比例的固定查询距离下降；'
            '下一步只需用原冻结查询点的最近三角面归属分解误差收益，并按父对象、噪声报告。'
            '若该假说不成立且收益主要落在普通面，再优先安排同协议的独立训练随机流重复。'
            '本轮不执行这项新分析或重复，不更改 loss、默认模型或采样器。')
summary = dict(quality_result=result['quality_result'], stages=result['stages'],
               actual_counts=read('accounting.json')['summary'], primary=primary,
               primary_relative_reduction=-primary['delta_mean']/primary['before_mean'],
               coverage=coverage, p95=tail, small_angles=angles,
               validation_all_three_correct_curves=all_correct,
               validation_one_input_wins_all_targets_curves=universal,
               validation_curve_denominator=8, both_out_net_MSE_contribution=tail_contribution,
               both_out_share_of_net_MSE_reduction=tail_contribution/primary['delta_mean'],
               priority=priority, hypothesis_status='PROPOSED_NOT_EXECUTED',
               independent_ledger_audit='PASS', independent_arithmetic_audit='PASS')
(OUT/'decision_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

paragraphs = [
    '## 六个决策问题', '',
    f"1. **验证 α=1 的自由几何：有开发性改善信号。** 主 MSE 从 {primary['before_mean']:.8f} 降至 {primary['after_mean']:.8f}，降低 {-primary['delta_mean']/primary['before_mean']*100:.2f}%；父均值配对差 {primary['delta_mean']:.8f}，配对差中位数 {primary['paired_delta_median']:.8f}，6/8 父改善。HYBRID 相对共同起点也为6/8父改善。两份噪声相对BASE的主均值均下降，分别7/8与6/8父改善；这是一个训练随机流的开发试验，不是显著性结论。", '',
    f"2. **收益也体现在距离尾部。** 验证A20 coverage 从 {coverage['before_mean']*100:.3f}% 升至 {coverage['after_mean']*100:.3f}%（+{coverage['delta_mean']*100:.3f}个百分点）；p95/L 从 {tail['before_mean']:.6f} 降至 {tail['after_mean']:.6f}，7/8父改善。相同查询点的 both_out 组对净MSE下降的代数贡献为 {tail_contribution/primary['delta_mean']*100:.1f}%，不是仅由跨过覆盖阈值的点产生。注意HYBRID验证覆盖仍略低于共同起点32.715%，不能把相对BASE的改善写成所有比较都改善。", '',
    f"3. **控制只有局部改善，跨噪声不一致。** 验证逐曲线 |b_self−1| 平均从0.691874降至0.638633，平均目标列优势从0.00059752升至0.00097195；但noise0的宽度绝对斜率误差和列优势退化，noise1改善。β=1列优势变差，β=.9严格列胜率下降。HYBRID仍只有 {all_correct}/8 曲线在三个目标上都正确胜出，{universal}/8 曲线由同一个输入对三个目标全部胜出。不能把平均列优势增加理解为条件选择性已解决。144个No-response目标列均为零优势、无严格胜出；Post-scale保留在逐曲线表中。", '',
    f"4. **质量代价明确存在。** 验证A20最小角<5°自由面比例从 {angles['before_mean']*100:.3f}% 升至 {angles['after_mean']*100:.3f}%，6/8父退化。A40训练父覆盖下降 {-train40_cov['delta_mean']*100:.3f}个百分点，接口Gamma RMS上升 {train40_gamma['delta_mean']:.4f}%L（7个适用父）；验证A40 CF受影响面比例也增加。已观察留出覆盖下降 {-hold_cov['delta_mean']*100:.3f}个百分点、CF受影响面比例增加 {hold_cf['delta_mean']*100:.3f}个百分点。另一方面，验证A20接口RMS及CF/FF比例改善，留出MSE也下降，故必须保留各项并列结果。原定义FF非法相交影响面比例在验证A20仍约92.63%，不代表可用的干净网格。", '',
    '5. **响应分解尚不能确定主要机制。** 三检查点的Euler与I/P/S独立重算均通过；局部反向与积分抵消确实存在，但不能由其范数推断控制正确或Jacobian刚性。预选FP32复核中，BASE有1处可比状态的I/P方向判断与BF16不同；因此不把条件注入不足或状态传播抵消写成已证实主因，也不把有限编辑分解写成OT的唯一因果证明。', '',
    '6. **一个优先下一步及检验假说。** '+priority, '',
    '## 最终独立核验与可追溯修正', '',
    '独立账本审计核对两臂各1000 UPDATE_DONE、8000逐例曝光、逐例共同输入hash、32父各250次曝光、每batch四个alpha=1、整体free坐标归一、固定终点实际SHA及调用计数，结果PASS。独立算术审计从414份保存距离向量重算指标，并核对主终点、144个完整3×3矩阵、432个目标列和覆盖分区闭合，结果PASS；两项审计均无模型调用。', '',
    '账本审计首版不能识别CPU评分元数据，修正后显式区分924条CPU记录并保留未知格式拒绝规则；首轮脚本、结果、修正diff和哈希保存在 independent_audit/attempt1 与 parser_correction.*。独立算术审计补正了No-response文本标签匹配，并要求实际覆盖144列；前版和diff保留在 independent_audit/arithmetic_revision1。正式训练、评价、账本和冻结协议均未改动或重跑。', '',
    '冻结报告代码的局部中文编码损坏只在最终文案层恢复；原自动报告、文案修正前后版本及SHA保存在 reporting_corrections。机器可读的最终决策文案以 decision_summary.json 为准，原统计JSON和CSV保持原值。运行/恢复说明见 RUN_AND_RESUME.md；固定任务已完成，当前无继续训练或自动重复任务。', ''
]
report = OUT/'CHAIR_HYBRID_COUPLING_REPORT.md'
before = sha(report)
text = report.read_text(encoding='utf-8')
archive = OUT/'reporting_corrections'/f'CHAIR_HYBRID_COUPLING_REPORT.prose_corrected.{before[:12]}.md'
archive.write_bytes(report.read_bytes())
assert '## 六个决策问题' in text
text = text.split('## 六个决策问题')[0] + '\n'.join(paragraphs)
lead = ('本轮有基准几何改善信号，但尚无联合质量成功：验证主MSE下降约40%，'
        '同时存在小角面增加、部分上下文/留出退化和条件响应跨噪声不一致。'
        '首要后续是核查细长面与几何收益的关系，检验假说见报告末尾；本轮固定预算完成即停止。\n\n')
text = text.replace('## 主终点\n', lead+'## 主终点\n', 1)
report.write_text(text, encoding='utf-8')
audit = dict(status='PASS', action='Evidence-specific prose and proposed next hypothesis only',
             previous_report_sha256=before, final_report_sha256=sha(report),
             prior_report_path=str(archive), final_decision_sha256=sha(OUT/'decision_summary.json'),
             statistical_source_sha256=sha(OUT/'paired_results/results.json'),
             editor_sha256=sha(__file__), actual_model_calls=0)
(OUT/'reporting_corrections/final_decision_edit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
print(json.dumps(audit))
