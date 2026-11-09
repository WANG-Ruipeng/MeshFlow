"""Pure arithmetic supplements from the complete saved-score distributions."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REFS = ('START', 'TEACHER500', 'FM500', 'TR500')
ERRORS = ('D_sym2048', 'fixed_query_MSE', 'fixed_query_p95_L', 'GT_to_F_MSE2048_L0', 'F_to_GT_MSE2048_L0')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(name):
    with (HERE / (name + '.csv')).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    path = HERE / (name + '.csv')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return {'path': str(path), 'rows': len(rows), 'sha256': sha(path)}


def selected(rows, ref, metric):
    return [r for r in rows if r['candidate_label'] == 'CAPPED500' and r['reference_label'] == ref and r['metric'] == metric]


def mean_delta(rows):
    return sum(float(r['delta']) for r in rows) / len(rows)


def run():
    sources = {str(HERE / (name + '.csv')): sha(HERE / (name + '.csv')) for name in ('per_parent', 'per_output', 'summary')}
    parents, outputs, summaries = (read_csv(name) for name in ('per_parent', 'per_output', 'summary'))
    tails = []
    for ref in REFS:
        for metric in ERRORS:
            pp, oo = selected(parents, ref, metric), selected(outputs, ref, metric)
            assert len(pp) == 8 and len(oo) == 16
            worst, best = max(pp, key=lambda r: float(r['delta'])), min(pp, key=lambda r: float(r['delta']))
            wo = max(oo, key=lambda r: float(r['delta']))
            harm = sum(max(0, float(r['delta'])) for r in pp) / 8
            benefit = -sum(min(0, float(r['delta'])) for r in pp) / 8
            delta = mean_delta(pp)
            assert abs(harm - benefit - delta) < 1e-12
            tails.append(dict(reference=ref, metric=metric, delta=delta, gross_harm_mean=harm, gross_benefit_mean=benefit,
                              harmed_parents=sum(float(r['delta']) > 0 for r in pp), harmed_outputs=sum(float(r['delta']) > 0 for r in oo),
                              worst_parent=worst['uid'], worst_parent_delta=float(worst['delta']), worst_parent_net_delta_share_percent=float(worst['net_delta_share_percent']),
                              worst_parent_gross_harm_share_percent=100 * max(0, float(worst['delta'])) / (8 * harm) if harm else None,
                              mean_delta_without_worst_parent=mean_delta([r for r in pp if r['uid'] != worst['uid']]),
                              best_parent=best['uid'], best_parent_delta=float(best['delta']), best_parent_net_delta_share_percent=float(best['net_delta_share_percent']),
                              mean_delta_without_best_parent=mean_delta([r for r in pp if r['uid'] != best['uid']]),
                              worst_output=wo['request_id'], worst_output_uid=wo['uid'], worst_output_noise=int(wo['noise']), worst_output_delta=float(wo['delta']),
                              worst_output_net_delta_share_percent=100 * float(wo['delta']) / (16 * delta) if delta else None))
    paths = {'quality_tail_summary': write_csv('quality_tail_summary', tails)}
    for scope, source in [('parent', parents), ('output', outputs)]:
        keys = ['uid'] if scope == 'parent' else ['uid', 'request_id', 'noise']
        grouped = {}
        for row in source:
            if row['candidate_label'] != 'CAPPED500' or row['role'] != 'CAPPED500_PRIMARY':
                continue
            if row['metric'] not in ('SeamCov', 'ConflictCov_C', 'ExposedRetain', 'D_sym2048', 'fixed_query_MSE', 'fixed_query_p95_L'):
                continue
            key = tuple(row[k] for k in keys)
            outrow = grouped.setdefault(key, {k: row[k] for k in keys})
            outrow[row['metric'] + '_CAPPED500'] = row['candidate_value']
            fields = ['reference_value', 'delta', 'delta_lower', 'delta_upper']
            if scope == 'parent':
                fields += ['noise0_delta', 'noise1_delta', 'noise_delta_pattern']
            for field in fields:
                outrow[f"{row['metric']}__vs_{row['reference_label']}__{field}"] = row[field]
        assert len(grouped) == (8 if scope == 'parent' else 16)
        paths[scope + '_digest'] = write_csv('CAPPED500_' + scope + '_digest', [grouped[k] for k in sorted(grouped)])

    lookup = {(r['reference'], r['metric']): r for r in tails}
    s = {(r['candidate_label'], r['reference_label'], r['metric']): r for r in summaries}
    ms, ds = lookup[('START', 'fixed_query_MSE')], lookup[('START', 'D_sym2048')]
    lines = ['# CAPPED500 尾部与参照范围补充', '',
             '这是已有逐例分数的算术归因，没有重新计算几何，也没有改变原评分门槛。完整 8 父、16 输出仍保留；下述去一父仅作敏感性描述。', '',
             '## 相对 START：收益与代价不是同一组父', '',
             '- 接合 S 改善的两个父是 866e7248 与 c1b8c2f0，父均分别 +43.3967pp、+7.8413pp；其余六父为负。866e7248 占净均值增益 121.26%，去它后 S 均值差为 −1.0869pp。',
             '- 两个噪声的 S 总均值差为 +9.1367pp 与 −0.1897pp；因此不能把固定两噪声的总均值称为各噪声稳定收益。相对 teacher500 则两个噪声均值分别 +3.1878pp、+8.8266pp，但仍有 4/8 父的两个噪声方向相反。',
             f'- 512 MSE 有 {ms["harmed_parents"]}/8 父、{ms["harmed_outputs"]}/16 输出恶化。最大代价父 {ms["worst_parent"][:8]} 的父均增量 {ms["worst_parent_delta"]:+.9f}，占净恶化 {ms["worst_parent_net_delta_share_percent"]:.2f}%，但占所有正代价之和 {ms["worst_parent_gross_harm_share_percent"]:.2f}%。去该父后均值差变为 {ms["mean_delta_without_worst_parent"]:+.9f}；这解释敏感性，不使原固定 8 父质量门通过。',
             f'- 最大单输出代价为 {ms["worst_output_uid"][:8]} 噪声 {ms["worst_output_noise"]}，512 MSE 差 {ms["worst_output_delta"]:+.9f}，占净恶化 {ms["worst_output_net_delta_share_percent"]:.2f}%。完整正负贡献见 quality_tail_summary.csv 与 error_tail.csv。',
             f'- Dsym 略降同时仍有 {ds["harmed_parents"]}/8 父恶化；最佳父 {ds["best_parent"][:8]} 贡献净下降的 {ds["best_parent_net_delta_share_percent"]:.2f}%。去该父后 Dsym 均值差为 {ds["mean_delta_without_best_parent"]:+.9f}，方向反转。它的总体小收益也不能当作普遍形状改善。',
             '- 与 512 MSE 一致，2048 GT→F 单向均方也升高；两组查询在 16/16 输出的差值方向一致。Dsym 降低来自 F→GT 方向下降超过 GT→F 方向上升，不宜归因成单纯 512 查询波动。这里只解释统计方向，具体空间原因由独立几何审计处理。', '',
             '## 相对 teacher500：值得保留的更强信号及限制', '',
             '- S 的 6/8 父改善、中位 +3.4719pp、全部描述性去一父均值仍为正（+3.0383 至 +6.9005pp），确实比“单个父偶然改善”更有支持。只有 2/8 父的两个噪声都严格改善，仍不足以证明未见噪声稳定。',
             '- 相对 teacher500，Dsym 与 512 MSE 的均值均下降，所有去一父均值也保持下降；这说明收益不完全依赖删除某个坏样本。但父级分别仅 5/8 和 4/8 改善，且 START 质量门仍失守。',
             '- I 均值 +0.7080pp 掩盖 534b0e6e 的 +13.9160pp 父均代价；其两个噪声均为代价（+7.0801pp、+20.7520pp）。相对 START 同父均值仅 +4.0039pp，却有一个噪声 +20.8496pp，因此应同时报告逐输出严重尾部。',
             '- E 对 teacher500 +10.3108pp，87.29% 净收益来自 f59627a0；去该父仍 +1.4976pp。它不是完全单父伪信号，但 +0.3101pp 的父中位更能说明典型父增幅。', '',
             '## 500→1000：不是只有对照在漂移', '',
             '- CAPPED 自身 S 下降 5.8677pp（5/8 父下降），E 下降 3.8177pp（6/8 父下降），I 增加 0.6012pp（5/8 父恶化）；两噪声的 S、I、E 总均值方向均更差。S、I、E 的全部去一父均值仍保持这个方向。',
             '- 同期 teacher 的 S 上升 0.9554pp，故相对 teacher 的 S 优势收缩约 6.8231pp，不能解释为 teacher 退化造成 500 步假优势。CAPPED Dsym 自身恶化 21.0144%，teacher 恶化 6.1731%；差分变化的完整单位值见 drift_difference_in_differences_summary.csv。',
             '- 512 MSE 的后半程恶化受 341bcae1 尾部影响，两个噪声均值方向相反，去该父会反向；不应将所有形状指标都写成每父每噪声一致恶化。', '',
             '值得保留的是：500 步对训练匹配 teacher 的接合收益、相对无限幅 TR500 的明显恢复，以及这些信号在全部父保留时仍存在。不能据此宣称 CAPPED500 已可替代 START、全父改善、独立重复成立、早停是唯一修复办法或视觉退化已经消失。', '']
    memo = HERE / 'TAIL_AND_DRIFT_MEMO.md'
    memo.write_text('\n'.join(lines), encoding='utf-8')
    source_control = HERE.parents[3] / 'experiments/mf_h1_local0/chair_boundary_attachment_v1/control_metrics.py'
    source_hash = sha(source_control)
    assert source_hash == 'a6036ccd2f6063a8a863591d4aa79e5358de7ab617449a4c19b133f095ecf9ab'
    receipt = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), source_csv=sources,
                   source_unchanged=all(sha(path) == value for path, value in sources.items()),
                   actual_original_control_source={'path': str(source_control), 'sha256': source_hash},
                   script_sha256=sha(__file__), tables=paths, memo={'path': str(memo), 'sha256': sha(memo)},
                   tail_summary=tails, model_calls=0, rescoring=0, new_generations=0, gate_recomputed=False)
    assert receipt['source_unchanged']
    (HERE / 'supplement_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'tables': len(paths), 'parent_rows': 8, 'output_rows': 16, 'memo': str(memo)}))


if __name__ == '__main__':
    run()
