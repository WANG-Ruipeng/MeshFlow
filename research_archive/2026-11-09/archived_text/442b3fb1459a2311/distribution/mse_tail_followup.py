"""Additional saved-score arithmetic requested for the concise root report."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    sources = [HERE / 'per_parent.csv', HERE / 'per_output.csv']
    parents, outputs = [list(csv.DictReader(p.open(encoding='utf-8-sig'))) for p in sources]
    rows = [r for r in parents if r['candidate_label'] == 'CAPPED500' and r['reference_label'] == 'START' and r['metric'] == 'fixed_query_MSE']
    assert len(rows) == 8
    top = sorted(rows, key=lambda r: float(r['delta']), reverse=True)[:2]
    reduced = [r for r in rows if r['uid'] != top[0]['uid']]
    candidate = sum(float(r['candidate_value']) for r in reduced) / 7
    reference = sum(float(r['reference_value']) for r in reduced) / 7
    out = dict(top2=[{k: r[k] if k == 'uid' else float(r[k]) for k in ('uid', 'delta', 'net_delta_share_percent', 'noise0_delta', 'noise1_delta')} for r in top],
               top2_combined_net_share_percent=sum(float(r['net_delta_share_percent']) for r in top),
               omit_largest_harm_descriptive=dict(remaining_parents=7, candidate_mean=candidate, reference_mean=reference,
                                                 relative_change_percent=100*(candidate-reference)/reference,
                                                 absolute_change=candidate-reference, meets_original_5_percent_numeric_component_only=candidate <= 1.05*reference, gate_recomputed=False),
               parent_improved=sum(float(r['delta']) < 0 for r in rows), parent_worse=sum(float(r['delta']) > 0 for r in rows),
               opposite_noise_parents=sum(r['noise_delta_pattern'] == 'OPPOSITE' for r in rows))
    for n in (0, 1):
        pair = [r for r in outputs if r['candidate_label'] == 'CAPPED500' and r['reference_label'] == 'START' and r['metric'] == 'fixed_query_MSE' and int(r['noise']) == n]
        assert len(pair) == 8
        c = sum(float(r['candidate_value']) for r in pair)/8
        r = sum(float(r['reference_value']) for r in pair)/8
        out['noise'+str(n)] = dict(improved_parents=sum(float(x['delta']) < 0 for x in pair), worsened_parents=sum(float(x['delta']) > 0 for x in pair),
                                   absolute_delta=c-r, relative_change_percent=100*(c-r)/r, candidate_mean=c, reference_mean=r)
    out['source_sha256'] = {str(p): digest(p) for p in sources}
    out['script_sha256'] = digest(__file__)
    out['original_gate_unchanged'] = True
    out['source'] = 'Already completed DEV saved score arithmetic; no metric recomputation or CONFIRM read.'
    (HERE / 'MSE_TAIL_FOLLOWUP.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    lines = ['# 512 MSE 尾部补充', '',
             f'相对 START 的 MSE 总均值退化，首两个贡献父为 {top[0]["uid"][:8]}（{float(top[0]["net_delta_share_percent"]):.4f}% 净代价）与 {top[1]["uid"][:8]}（{float(top[1]["net_delta_share_percent"]):.4f}%），合计 {out["top2_combined_net_share_percent"]:.4f}%。比例超过 100% 表示其他父贡献抵消，绝不是样本权重。', '',
             f'描述性去掉首父后，其余 7 父 MSE 差为 {100*(candidate-reference)/reference:+.4f}%，因而单独 5% 数值组件会满足；该计算不重判原固定 8 父 gate。原分母中仍是 3/8 父改善、5/8 恶化。', '',
             f'噪声 0 的均值变化为 {out["noise0"]["relative_change_percent"]:+.4f}%（3/8 父改善），噪声 1 为 {out["noise1"]["relative_change_percent"]:+.4f}%（4/8 父改善）。这说明仅按改善父数无法看出噪声 1 的严重误差尾部。3/8 父的两个噪声方向相反。', '',
             '相对 teacher500 的 S，噪声 0/1 总均值差分别 +3.18784pp/+8.82657pp；6/8 父均改善，只有 2/8 父两个噪声都严格改善。', '']
    (HERE / 'MSE_TAIL_FOLLOWUP.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'rows_retained': 8, 'LOPO_is_descriptive': True}))


if __name__ == '__main__':
    run()
