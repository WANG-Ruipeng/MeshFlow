"""Read saved CSVs only; write an independently traceable numeric digest."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IDENTIFIERS = {'panel', 'candidate_label', 'reference_label', 'metric', 'uid',
    'request_id', 'label', 'status', 'candidate_status', 'reference_status',
    'omitted_uid', 'minimum_request', 'maximum_request', 'minimum_parent', 'maximum_parent'}
CORE = ('SeamCov', 'GT_to_F_MSE2048_L0', 'D_sym2048', 'ConflictCov_C', 'ExposedRetain')
KEY = CORE + ('fixed_query_MSE', 'F_to_GT_MSE2048_L0', 'fixed_query_p95_L',
    'CF_proper_affected_free_fraction', 'FF_illegal_affected_free_fraction',
    'free_near_degenerate_fraction', 'free_minimum_angle_lt5_fraction', 'free_area_over_GT_free_area')
PANELS = ('OLD', 'FRESH', 'ALL')
CONTRASTS = (('HALF_CAPPED500', 'START'), ('HALF_CAPPED500', 'HIST_CAPPED500'), ('HIST_CAPPED500', 'START'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(key, value):
    if value == '':
        return None
    if key in IDENTIFIERS:
        return value
    if value in ('True', 'False'):
        return value == 'True'
    if value.startswith('{'):
        return json.loads(value)
    try:
        number = float(value)
        return number if math.isfinite(number) else value
    except ValueError:
        return value


def load(name):
    with (ROOT/name).open(encoding='utf-8-sig', newline='') as handle:
        return [{key: convert(key, value) for key, value in row.items()} for row in csv.DictReader(handle)]


def select(rows, **criteria):
    return [row for row in rows if all(row[key] == value for key, value in criteria.items())]


def one(rows, **criteria):
    matched = select(rows, **criteria)
    assert len(matched) == 1, (criteria, len(matched))
    return matched[0]


def build():
    results = json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
    assert results['status'] == 'COMPLETE' and results['completed_cells'] == 96
    names = ('summary.csv', 'per_parent.csv', 'per_output.csv', 'per_noise.csv',
             'leave_one_parent_out.csv', 'output_tail_summary.csv', 'checkpoint_profiles.csv')
    sources = {}
    data = {}
    for name in names:
        expected = results['tables'][name[:-4]]['sha256']
        actual = sha(ROOT/name)
        assert actual == expected, name
        data[name] = load(name)
        sources[name] = dict(path=str(ROOT/name), sha256=actual, rows=len(data[name]))
    summaries, parents, outputs, noises, lopo = [data[name] for name in names[:5]]
    def summary(panel, reference, metric, candidate='HALF_CAPPED500'):
        return one(summaries, panel=panel, candidate_label=candidate, reference_label=reference, metric=metric)
    def father(panel, reference, metric, uid):
        return one(parents, panel=panel, candidate_label='HALF_CAPPED500', reference_label=reference, metric=metric, uid=uid)
    def leave(panel, reference, metric, uid):
        return one(lopo, panel=panel, candidate_label='HALF_CAPPED500', reference_label=reference, metric=metric, omitted_uid=uid)
    groups = {}
    full_parents = {}
    extremes = []
    for panel in PANELS:
        groups[panel] = {}
        for candidate, reference in CONTRASTS:
            key = candidate+'__minus__'+reference
            groups[panel][key] = {metric: summary(panel, reference, metric, candidate) for metric in KEY}
            for metric in KEY:
                high_is_good = metric in ('SeamCov', 'ExposedRetain')
                if metric == 'free_area_over_GT_free_area':
                    continue  # No universal lower/higher preference for an area ratio.
                worst = {}
                for kind, table in (('parent', parents), ('output', outputs)):
                    rows = select(table, panel=panel, candidate_label=candidate, reference_label=reference, metric=metric)
                    valid = [r for r in rows if r['delta'] is not None]
                    if valid:
                        adverse = min(valid, key=lambda r: r['delta']) if high_is_good else max(valid, key=lambda r: r['delta'])
                        worst[kind] = adverse
                extremes.append(dict(panel=panel, candidate_label=candidate, reference_label=reference,
                    metric=metric, scope='Largest observed adverse delta; not a new failure threshold', **worst))
        full_parents[panel] = []
        uids = sorted({r['uid'] for r in select(parents, panel=panel, candidate_label='HALF_CAPPED500', reference_label='START', metric='SeamCov')})
        assert len(uids) == 8
        for uid in uids:
            record = dict(uid=uid, metrics={})
            for metric in CORE:
                record['metrics'][metric] = dict(parent=father(panel, 'START', metric, uid),
                    outputs=select(outputs, panel=panel, candidate_label='HALF_CAPPED500', reference_label='START', metric=metric, uid=uid))
            full_parents[panel].append(record)
    def full_uid(prefix):
        matches = {r['uid'] for r in parents if r['uid'].startswith(prefix)}
        assert len(matches) == 1
        return next(iter(matches))
    f596, p534, p341, p866 = map(full_uid, ('f596', '534b', '341b', '866e'))
    special = {}
    special['OLD_f596_I_E'] = []
    for reference in ('START', 'HIST_CAPPED500'):
        for metric in ('ConflictCov_C', 'ExposedRetain'):
            row = father('OLD', reference, metric, f596)
            special['OLD_f596_I_E'].append(dict(reference_label=reference, metric=metric, parent=row,
                outputs=select(outputs, panel='OLD', candidate_label='HALF_CAPPED500', reference_label=reference, metric=metric, uid=f596),
                existing_severe_threshold=.10, direction='I increase > .10' if metric=='ConflictCov_C' else 'E decrease > .10',
                original_flag=row['original_severe_parent_cost_confirmed'],
                role='Inherited descriptive severe-parent flag; not an added promotion veto'))
    special['FRESH_534b_both_noise_shape_tail'] = {}
    special['FRESH_341b_dominates_gain_vs_historical'] = {}
    for metric in ('fixed_query_MSE', 'GT_to_F_MSE2048_L0', 'F_to_GT_MSE2048_L0', 'D_sym2048'):
        special['FRESH_534b_both_noise_shape_tail'][metric] = dict(parent=father('FRESH', 'START', metric, p534),
            outputs=select(outputs, panel='FRESH', candidate_label='HALF_CAPPED500', reference_label='START', metric=metric, uid=p534),
            leave_out=leave('FRESH', 'START', metric, p534))
        special['FRESH_341b_dominates_gain_vs_historical'][metric] = dict(
            full_summary=summary('FRESH', 'HIST_CAPPED500', metric),
            parent=father('FRESH', 'HIST_CAPPED500', metric, p341),
            outputs=select(outputs, panel='FRESH', candidate_label='HALF_CAPPED500', reference_label='HIST_CAPPED500', metric=metric, uid=p341),
            leave_out=leave('FRESH', 'HIST_CAPPED500', metric, p341))
    h = summary('OLD', 'START', 'SeamCov')
    c = summary('OLD', 'START', 'SeamCov', 'HIST_CAPPED500')
    retention = dict(panel='OLD', numerator_HALF_minus_START=h['delta'], denominator_HIST_minus_START=c['delta'],
        fraction=h['delta']/c['delta'], percent=100*h['delta']/c['delta'],
        scope='Ratio of eight-parent equal mean net S gains only; not per-parent retention, output usability, probability or interpolation linearity')
    special['OLD_866e_S_mean_dependence'] = dict(parent=father('OLD', 'START', 'SeamCov', p866),
        leave_out=leave('OLD', 'START', 'SeamCov', p866))
    digest = dict(status='COMPLETE_NUMERIC_DIGEST', scientific_recomputation=False,
        model_calls=0, geometry_queries=0, rows_rescored=0,
        scope='96 fixed cells; 64 read-only cached baselines and 32 HALF outputs. All 8 parents and all 4 noises retained.',
        units=dict(control_delta='Fractions in JSON; multiply by100 for percentage points.',
            fixed_query_MSE='Original frozen 512-query normalization retained.',
            GT_to_F_MSE2048_L0='Squared distance / L0^2', F_to_GT_MSE2048_L0='Squared distance / L0^2',
            D_sym2048='Arithmetic mean of the two squared-distance directions.', fixed_query_p95_L='Original frozen normalized distance, not squared.'),
        bounds='Exact original saved S/I intervals and E global conservative intervals; PARTIAL/NA not converted to exact or zero.',
        comparison_summaries=groups, all8_parent_HALF_minus_START=full_parents,
        four_noise_means=[r for r in noises if r['panel']=='ALL' and r['candidate_label']=='HALF_CAPPED500' and r['metric'] in KEY],
        worst_observed_deltas=extremes,
        leave_one_parent_out=[r for r in lopo if r['metric'] in KEY],
        absolute_output_tails=[r for r in data['output_tail_summary.csv'] if r['metric'] in KEY],
        inherited_severe_parent_flags=[r for r in parents if r['candidate_label']=='HALF_CAPPED500' and r['original_severe_parent_cost_confirmed'] is True],
        special_cases=special, OLD_mean_S_retention_ratio=retention,
        limitations=['No new threshold, gate, selection, deletion, inference test or model dispatch.',
            'LOPO is descriptive; all actual conclusions keep all8 parents.',
            'Net contribution percentages can exceed100% when other parents offset a tail.',
            'OLD/FRESH are already-observed development panels, not independent new-parent confirmation.',
            'Positive mean S prerequisite alone is not a useful-tradeoff or promotion verdict.'],
        source_csv=sources, source_results_sha256=sha(ROOT/'results.json'), generator_sha256=sha(__file__))
    (ROOT/'decision_numeric_digest.json').write_text(json.dumps(digest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    def pct(panel, ref, metric):
        return f"{summary(panel, ref, metric)['percent_change']:+.4f}%"
    def pp(panel, ref, metric):
        return f"{100*summary(panel, ref, metric)['delta']:+.4f}pp"
    memo = [
        '# 半幅权重实验：数值结论与局限\n',
        '这是对最终CSV的只读摘要：96条固定输出齐全，其中64条基线仅复用、32条HALF新评分。这里没有重评分、新几何查询、训练或模型调用；不新增门槛。完整数字与原区间见 [decision_numeric_digest.json](decision_numeric_digest.json)，父/噪声和留一父完整表仍为本目录原CSV。\n',
        '半幅权重保留了相对START的正接缝均值，并减轻部分历史CAPPED的GT→free误差；改善并不普遍，也没有形成双向质量与控制代价的全面改善。正均值只满足必要描述条件，不能据此晋级。\n',
        '| HALF−START | S均差 | S中位差 | 改善父 | I均差 | E均差 | 512MSE | GT→F2048 | F→GT2048 | Dsym |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for panel in PANELS:
        s = summary(panel, 'START', 'SeamCov')
        memo.append('| '+panel+' | '+pp(panel,'START','SeamCov')+f" | {100*s['median_delta']:+.4f}pp | {int(s['improved_parents'])}/8 | "+
            ' | '.join([pp(panel,'START','ConflictCov_C'),pp(panel,'START','ExposedRetain')]+[pct(panel,'START',m) for m in ('fixed_query_MSE','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','D_sym2048')])+' |')
    memo += ['\nOLD的S正均值只有2/8父改善，中位数为负。866e一父贡献净增益的118.3336%；仅作描述地留出该父，均差变为−0.3768pp。四个固定噪声中，OLD n0/n1全组S均差为+3.6835/−0.0866pp，FRESH n2/n3为+1.9732/+1.0747pp。ALL四噪声合并仅3/8父改善，没有父在四个噪声上全部改善。\n',
        f"OLD的S保留率为{retention['percent']:.4f}%（约40.20%），仅指HALF−START与历史CAPPED−START两个父等权平均增益的比值，不能解释为40.20%的父对象/输出获益。\n",
        '| HALF−历史CAPPED | S均差 | I均差 | E均差 | 512MSE | GT→F2048 | F→GT2048 | Dsym |\n|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for panel in PANELS:
        memo.append('| '+panel+' | '+' | '.join([pp(panel,'HIST_CAPPED500',m) for m in ('SeamCov','ConflictCov_C','ExposedRetain')]+[
            pct(panel,'HIST_CAPPED500',m) for m in ('fixed_query_MSE','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','D_sym2048')])+' |')
    memo += ['\nOLD降低了历史CAPPED的512/2048 GT→free误差，却使反向F→GT误差增加20.7610%，Dsym增加5.9449%。FRESH相对历史CAPPED的平均质量改善也不能称普遍：512/GT2048/Dsym各只有2/8父改善，三者的父中位差都为退化方向。\n',
        '最重要的控制尾部是OLD f596：I相对START和历史CAPPED均增加22.1436pp；相对START的n0/n1分别增加9.2285/35.0586pp。E相对历史CAPPED下降34.1220pp，原保守区间为[−34.2271,−34.0431]pp，两噪声分别下降26.0252/42.2187pp。这确定越过既有的单父I增加或E下降10pp严重代价描述线；没有新增数值否决阈值。E相对START仍改善37.2766pp，不能混淆两个参照。\n',
        'FRESH的形状尾部是534b：相对START，GT→F2048在n2/n3分别增加0.0054328897/0.0020519949；Dsym增加0.0035274403/0.0023518165。两个噪声均退化。Dsym其余7父全部改善，但此父足以使全组均值退化6.8706%；它贡献净退化的241.6345%。比例超过100%来自其他父的抵消，不是发生概率。\n',
        'FRESH相对历史CAPPED的均值改善主要来自341b：其512MSE、GT2048、Dsym分别贡献净均值改善的299.1010%、333.8853%、249.9151%。该父n2反而退化，n3大幅改善。仅作描述地留出341b，三项全组均值均转为退化；详见JSON中的逐输出与LOPO原行。禁止据此删父重新宣布成功。\n',
        'E收益也集中：f596占HALF−START的E净增益OLD约79.9470%、FRESH约70.4952%、ALL约76.2704%；但E的LOPO均值仍全部为正。两种情况应同时报告，不能把集中等同于其余父完全无改善。\n',
        '96条S/I均为PASS；E共72条PARTIAL、24条PASS，继续使用既有global上下界。硬合同PASS仅说明C/噪声/N/有限性和导出要求通过，不是主体质量或人类可用性结论。视觉结论另由全量实际审图给出。\n',
        '本次只检验固定α=0.5与这8个已观察父、原四噪声的表现，不证明缩小幅度能稳定修复CAPPED，也不能外推为新父或新噪声的独立确认。\n',
        '来源校验：所读7份CSV均与冻结分析results.json登记SHA一致；精确路径、SHA和行数记在decision_numeric_digest.json的source_csv。生成脚本不导入评分器或模型。\n',
    ]
    (ROOT/'NUMERIC_FINDINGS.md').write_text('\n'.join(memo), encoding='utf-8')
    for name, record in sources.items():
        assert sha(ROOT/name) == record['sha256'], 'Source changed during digest build'
    print(json.dumps(dict(status='PASS', digest_sha256=sha(ROOT/'decision_numeric_digest.json'),
        memo_sha256=sha(ROOT/'NUMERIC_FINDINGS.md'), source_csv_count=len(sources),
        parent_panels={p:len(r) for p,r in full_parents.items()}, scoring_calls=0, model_calls=0)))


if __name__ == '__main__':
    build()
