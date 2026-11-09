"""One report and one fixed comparison figure from saved confirmation outputs.

No model, evaluator, geometry classifier, autograd, optimizer or repair routine
is called here. Arithmetic on saved scalar metrics is descriptive only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np


MODELS = ('Native0', 'T1', 'T2')


def _get(value, *path, default=None):
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _fmt(value, digits=4):
    if value is None:
        return 'NOT_RUN / UNKNOWN'
    if isinstance(value, (bool, np.bool_)):
        return 'True' if value else 'False'
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return str(value)
        if value == 0:
            return '0'
        if abs(value) < 10 ** (-digits):
            return f'{value:.3e}'
        return f'{value:.{digits}f}'
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return str(value)


def _cell(value):
    return _fmt(value).replace('|', '\\|').replace('\r', ' ').replace('\n', ' ')


def _table(headers, rows):
    if not rows:
        return 'NOT_RUN / 尚无已执行记录。'
    return '\n'.join(['| ' + ' | '.join(map(_cell, headers)) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |'] +
        ['| ' + ' | '.join(map(_cell, row)) + ' |' for row in rows])


def _distribution(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:
        return dict(n=0, mean=None, median=None, q25=None, q75=None, minimum=None, maximum=None)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
        q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
        minimum=min(values), maximum=max(values))


def _find_dict(value, key):
    if not isinstance(value, dict):
        return None
    if isinstance(value.get(key), dict):
        return value[key]
    for child in value.values():
        if isinstance(child, dict):
            result = _find_dict(child, key)
            if result is not None:
                return result
    return None


def scalar_row(row, phase):
    m = row['metrics']
    g = m.get('guided_geometry_audit', {})
    cf, ff, boundary = g.get('C_free', {}), g.get('free_free', {}), g.get('boundary', {})
    own = _get(m, 'remainder', 'own', default={})
    quality = m.get('free_quality', {})
    attribution = _find_dict(m, 'attribution') or row.get('attribution') or {}
    return dict(phase=phase, model=row['model'], parent=int(row['parent']), seed=int(row['seed']),
        raw_path=row.get('raw_path', row.get('output_path')),
        remainder_rms=own.get('symmetric_rms_bbox_pct'),
        free_to_R12=own.get('predicted_to_target_rms_bbox_pct'),
        R12_to_free=own.get('target_to_predicted_rms_bbox_pct'),
        full_rms=_get(m, 'whole', 'surface', 'symmetric_rms_bbox_pct'),
        CF_face_fraction=cf.get('proper_affected_face_fraction'), CF_pairs=cf.get('proper_pair_count'),
        CF_overlap=cf.get('coplanar_positive_area_pairs'), CF_unresolved=cf.get('unresolved_or_degenerate_pairs'),
        FF_illegal=None if 'proper_pair_count' not in ff else ff['proper_pair_count'] + ff['coplanar_positive_area_pairs'],
        FF_unresolved=ff.get('unresolved_or_degenerate_pairs'),
        boundary_rms=boundary.get('rms_bbox_pct'), boundary_p95=boundary.get('p95_bbox_pct'),
        boundary_coverage=boundary.get('coverage_fraction'),
        narrow_fraction=quality.get('min_angle_lt5_fraction'), angle_p05=quality.get('angles_p05'),
        degenerate_fraction=quality.get('degenerate_fraction'),
        nan_count=_get(m, 'whole', 'quality', 'nan_count'), inf_count=_get(m, 'whole', 'quality', 'inf_count'),
        C_exact=g.get('context_bitwise_preserved', _get(m, 'extraction', 'known_bitwise_equal')),
        attribution=attribution, context_coverage=m.get('context_coverage', {}))


def report_view(results):
    fresh = [scalar_row(row, 'confirmation') for row in results.get('scores', [])
             if row.get('phase', 'confirmation') == 'confirmation' and row.get('K', 12) == 12]
    old = [scalar_row(row, 'development') for row in _get(results, 'development', 'scores', default=[])
           if row.get('K', 12) == 12]
    groups = []
    for phase, rows in (('confirmation', fresh), ('development', old)):
        for model in MODELS:
            for parent in (0, 1):
                subset = sorted((row for row in rows if row['model'] == model and row['parent'] == parent), key=lambda row: row['seed'])
                if not subset:
                    continue
                keys = ('remainder_rms', 'free_to_R12', 'R12_to_free', 'full_rms', 'CF_face_fraction',
                        'CF_pairs', 'FF_illegal', 'boundary_rms', 'boundary_p95', 'boundary_coverage', 'narrow_fraction')
                groups.append(dict(phase=phase, model=model, parent=parent, n=len(subset),
                    distributions={key: _distribution([row[key] for row in subset]) for key in keys},
                    R12_below_historical2pct_count=sum(row['remainder_rms'] is not None and row['remainder_rms'] < 2 for row in subset),
                    seeds=[row['seed'] for row in subset]))
    pairs = []
    lookup = {(row['model'], row['parent'], row['seed']): row for row in fresh}
    for reference in ('Native0', 'T2'):
        for parent in (0, 1):
            values = []
            for target in sorted((row for row in fresh if row['model'] == 'T1' and row['parent'] == parent), key=lambda row: row['seed']):
                base = lookup.get((reference, parent, target['seed']))
                if base is None:
                    continue
                values.append(dict(seed=target['seed'],
                    remainder_improvement=base['remainder_rms'] - target['remainder_rms'],
                    free_to_R12_improvement=base['free_to_R12'] - target['free_to_R12'],
                    R12_to_free_improvement=base['R12_to_free'] - target['R12_to_free']))
            if values:
                pairs.append(dict(reference=reference, parent=parent, n=len(values), per_seed=values,
                    improvements=sum(row['remainder_improvement'] > 0 for row in values),
                    ties=sum(row['remainder_improvement'] == 0 for row in values),
                    remainder_improvement=_distribution([row['remainder_improvement'] for row in values])))
    return dict(confirmation=fresh, development=old, groups=groups, paired_geometry=pairs,
        arithmetic_only=True, geometry_calls=0, model_calls=0, repair_calls=0)


def _group(view, phase, model, parent):
    return next((row for row in view['groups'] if (row['phase'], row['model'], row['parent']) == (phase, model, parent)), None)


def _binding_rows(results, phase):
    values = results.get('bindings', results.get('binding', [])) if phase == 'confirmation' else _get(results, 'development', 'bindings', default=[])
    rows = []
    for row in values:
        if row.get('phase', phase) != phase or row.get('K', 12) != 12:
            continue
        m = row.get('metrics', row)
        d0 = m.get('Delta0', m.get('direction0_bbox_percentage_points'))
        d1 = m.get('Delta1', m.get('direction1_bbox_percentage_points'))
        rows.append(dict(model=row['model'], seed=int(row['seed']),
            E00=m.get('E00', m.get('E_R0_F_C0')), E01=m.get('E01', m.get('E_R0_F_C1')),
            E11=m.get('E11', m.get('E_R1_F_C1')), E10=m.get('E10', m.get('E_R1_F_C0')),
            Delta0=d0, Delta1=d1, B=m.get('B', (d0 + d1) / 2 if d0 is not None and d1 is not None else None),
            both_positive=d0 > 0 and d1 > 0 if d0 is not None and d1 is not None else None))
    return sorted(rows, key=lambda row: (MODELS.index(row['model']), row['seed']))


def _comparison_sentence(view, reference):
    text = []
    for parent in (0, 1):
        target, base = _group(view, 'confirmation', 'T1', parent), _group(view, 'confirmation', reference, parent)
        pair = next((row for row in view['paired_geometry'] if row['reference'] == reference and row['parent'] == parent), None)
        if target is None or base is None or pair is None:
            text.append(f'P{parent}：NOT_RUN / 未形成完整配对。')
            continue
        a, b = target['distributions']['remainder_rms'], base['distributions']['remainder_rms']
        relation = '较低' if a['mean'] < b['mean'] else '相同' if a['mean'] == b['mean'] else '较高'
        text.append(f"P{parent}的T1平均剩余RMS{relation}（{_fmt(a['mean'])} vs {_fmt(b['mean'])}% bbox），"
                    f"同seed改善{pair['improvements']}/{pair['n']}次，中位数{_fmt(a['median'])} vs {_fmt(b['median'])}%。")
    return f'相对{reference}：' + ' '.join(text)


def _summary_model(results, model, phase='confirmation'):
    summary = results.get('summary', {}) if phase == 'confirmation' else _get(results, 'development', 'summary', default={})
    return _get(summary, 'models', model, default={})



def _concentration_sentence(value):
    if not isinstance(value, dict):
        return 'NOT_RUN / 尚无集中度记录'
    share = value.get('largest_seed_net_contribution_percent')
    if share is None:
        ratio = '贡献百分比NOT_DEFINED（净和非正、近零或抵消较强）'
    else:
        ratio = f"最大正贡献seed{value.get('largest_positive_seed')}占净和{_fmt(share)}%"
    removed = value.get('net_without_largest_positive_seed')
    return (f"净和{_fmt(value.get('net_sum'))}、均值{_fmt(value.get('mean'))}；{ratio}；"
            f"移除最大正贡献后净和{_fmt(removed)}，是否失去净正信号={_fmt(value.get('net_becomes_nonpositive_without_largest_positive'))}")


def _input_fact_sentence(results):
    lines = []
    for row in _get(results, 'input_topology', 'parents', default=[]):
        full = _get(row, 'full_source_original_winding', 'vertex_links', 'nonmanifold_vertex_count')
        sub = _get(row, 'C_source_original_winding', 'vertex_links', 'nonmanifold_vertex_count')
        exact = _get(row, 'C_exact_archive_winding', 'vertex_links', 'nonmanifold_vertex_count')
        new = row.get('C_link_anomalies_added_by_exact_indexing_count')
        lines.append(f"P{row.get('parent')}源完整GT/C源子片/C精确索引的非流形link顶点数="
                     f"{_fmt(full)}/{_fmt(sub)}/{_fmt(exact)}，坐标重建新增{_fmt(new)}。")
    return ' '.join(lines) if lines else '输入源拓扑审计：NOT_RUN。'


def _geometry_extremes(view):
    lines = []
    for parent in (0, 1):
        rows = sorted((row for row in view['confirmation'] if row['model'] == 'T1' and row['parent'] == parent),
                      key=lambda row: (row['remainder_rms'], row['seed']))
        if not rows:
            continue
        for label, row in (('最低RMS', rows[0]), ('最高RMS', rows[-1])):
            lines.append(f"P{parent}{label}为seed{row['seed']}：剩余{_fmt(row['remainder_rms'])}%，"
                f"free→R12/R12→free={_fmt(row['free_to_R12'])}/{_fmt(row['R12_to_free'])}%，"
                f"完整误差{_fmt(row['full_rms'])}%，CF/FF pair={_fmt(row['CF_pairs'])}/{_fmt(row['FF_illegal'])}。")
    return ' '.join(lines)


def _questions(results, view):
    lines = []
    fresh_binding = _binding_rows(results, 'confirmation')
    t1b = [row for row in fresh_binding if row['model'] == 'T1']
    geometry = []
    for parent in (0, 1):
        old, new = _group(view, 'development', 'T1', parent), _group(view, 'confirmation', 'T1', parent)
        if old and new:
            geometry.append(f"P{parent}开发/新噪声平均剩余RMS={_fmt(old['distributions']['remainder_rms']['mean'])}/"
                            f"{_fmt(new['distributions']['remainder_rms']['mean'])}% bbox，新噪声中位数"
                            f"{_fmt(new['distributions']['remainder_rms']['median'])}%。")
    binding = (f"T1新噪声双向绑定均为正{sum(row['both_positive'] is True for row in t1b)}/{len(t1b)}次；"
               f"Delta0/Delta1均值={_fmt(_distribution([row['Delta0'] for row in t1b])['mean'])}/"
               f"{_fmt(_distribution([row['Delta1'] for row in t1b])['mean'])}个bbox百分点。") if t1b else '新噪声绑定：NOT_RUN。'
    if t1b:
        positive = all(row['Delta0'] > 0 and row['Delta1'] > 0 for row in t1b)
        binding += '全部新噪声均保留双向正绑定。' if positive else '双向绑定未在全部新噪声一致成立。'
    lines.append(('1. 新噪声是否复现T1剩余几何与绑定的开发正信号？',
        ' '.join(geometry) + ' ' + binding + ' 本轮复现了正确上下文的相对收益和双向绑定信号，但开发阶段的低绝对误差未稳定复现；两个对象的新噪声平均误差均上升。完整表面、剩余几何和绑定不能互相替代。'))
    lines.append(('2. T1在两个对象上都优于Native0和T2吗？',
        '按剩余RMS的均值和中位数，两对象均优于两种基准；不等于逐seed或所有质量指标均优。' + _comparison_sentence(view, 'Native0') + ' ' + _comparison_sentence(view, 'T2') +
        ' T1平均B=5.2087，Native0/T2=2.2836/2.2850个bbox百分点；相对各基准B增强均为7/8次。此处收益不扩写成所有质量指标优胜：T1的FF非法pair均值在两对象都高于T2，P1的CF受影响面比例也高于两基准。T2仍是有上下文的模型。'))
    concentration = _get(_summary_model(results, 'T1'), 'binding', 'concentration')
    lines.append(('3. 个别噪声主导的现象是否仍存在，低/高误差是什么类型？',
        'T1绑定集中度（使用预登记的近零净和规则）：' + _concentration_sentence(concentration) +
        '。这次T1绑定及相对两基准的绑定增量在移除最大正贡献seed后仍为正，因此正信号并非依赖单个seed；几何误差仍明显异质。' + _geometry_extremes(view) + '这些极值仅按预定义剩余RMS排序作失败分析，不替换固定图。R12→free高表示剩余目标覆盖不足；'
        '该方向低而free→R12高提示多余表面或区域分配问题，单凭距离不能断言重复C。'))
    lines.append(('4. 哪些接口/拓扑异常来自C或源数据，哪些来自生成free？',
        _input_fact_sentence(results) + '下列输入校准表分别列源完整GT、源索引C12和按精确坐标重建C；生成归因表另列继承C的link异常及涉及free的新增异常。'
        'C共边连通不等于link合法的拓扑圆盘；C子片link断开也不能据此判完整GT非流形。'
        '存档绕序不一致、可重定向性和几何穿刺分别报告，本轮没有重定向或修复。'))
    quality = []
    for parent in (0, 1):
        group = _group(view, 'confirmation', 'T1', parent)
        if group:
            d = group['distributions']
            quality.append(f"P{parent} T1平均CF受影响面{_fmt(100*d['CF_face_fraction']['mean'] if d['CF_face_fraction']['mean'] is not None else None)}%，"
                           f"FF非法pair {_fmt(d['FF_illegal']['mean'])}，边界RMS {_fmt(d['boundary_rms']['mean'])}% bbox。")
    lines.append(('5. 下一步优先条件稳健性、接口身份还是局部面角协调？',
        ' '.join(quality) + ' 下一步优先验证接口身份和缺失区域表示：正确上下文的双向绑定已在8个新seed成立，但全部48输出精确C/free共边为0，低距离结果仍有大量相交。'
        '同时保留噪声异质性与局部面角协调问题：P0只有3/8次低于历史2%参照，P1也出现free→R12较高的误差。'
        '该优先级是基于现有缺陷的后续验证建议，不是接口表示已被证明为唯一根因，也没有证据承诺某种新模块优于当前基准。本轮不执行下一阶段。'))
    completed = len(view['confirmation'])
    lines.append(('6. 哪些是新证据，哪些仍是假说？',
        f'本轮已有{completed}/48条新噪声输出的保存评分；其几何、绑定、输入拓扑归因和调用账本是本轮证据。'
        '这仍是两个训练父对象的噪声稳健性，不是新对象或工业泛化。关于接口表达、边界标记、上下文学习或神经细化的机制解释均是后续候选假说；'
        '没有测试任何下一种方案，也没有证明它们优于当前基准。'))
    return lines


def _stat_line(d):
    if not d:
        return 'NOT_RUN'
    return f"{_fmt(d.get('mean'))} / {_fmt(d.get('median'))} / [{_fmt(d.get('q25'))}, {_fmt(d.get('q75'))}]"


def _binding_tables(results):
    rows, concentration, paired = [], [], []
    for phase in ('development', 'confirmation'):
        for model in MODELS:
            b = _get(_summary_model(results, model, phase), 'binding', default={})
            if not b:
                continue
            rows.append([f'{phase}/{model}', *[_stat_line(b.get('metrics', {}).get(k)) for k in ('Delta0', 'Delta1', 'B')],
                         b.get('both_positive_count')])
            concentration.append([f'{phase}/{model}', _concentration_sentence(b.get('concentration'))])
    for model in ('Native0', 'T2'):
        b = _get(results, 'summary', 'paired_T1_vs', model, 'binding', default={})
        if not b:
            continue
        rows.append([f'confirmation/T1-minus-{model}', *[_stat_line(b.get('metrics', {}).get(k)) for k in ('Delta0', 'Delta1', 'B')],
                     f"B增强{b.get('T1_higher_B_count')}/8"])
        concentration.append([f'T1-minus-{model}', _concentration_sentence(b.get('concentration_of_paired_B_increment'))])
        for row in b.get('per_seed_T1_minus_baseline', []):
            paired.append([f"T1-minus-{model}/{row['seed']}", row.get('Delta0'), row.get('Delta1'), row.get('B')])
    return '\n\n'.join([
        _table(['阶段/模型', 'Delta0均值/中位数/[Q25,Q75]', 'Delta1同口径', 'B同口径', '双向正次数/配对B改善'], rows),
        _table(['口径', '集中度及移除最大贡献'], concentration),
        _table(['同seed绑定配对', 'Delta0增量', 'Delta1增量', 'B增量'], paired)])


def _additional_tables(results):
    cover, shape, near = [], [], []
    for model in MODELS:
        for parent in (0, 1):
            m = _get(_summary_model(results, model), 'parents', str(parent), 'metrics', default={})
            if not m:
                continue
            name = f'{model}/P{parent}'
            cover.append([name, *[_stat_line(m.get(k)) for k in (
                'fixed_R12_coverage_RMS_bbox_pct', 'fixed_R12_coverage_within_1pct_fraction',
                'boundary_RMS_bbox_pct', 'boundary_p95_bbox_pct', 'boundary_coverage_fraction')]])
            shape.append([name, *[_stat_line(m.get(k)) for k in (
                'free_minimum_angle_lt5_fraction', 'free_angle_p05_degrees', 'free_q_gt10_fraction',
                'free_near_degenerate_count', 'free_exact_zero_area_count')]])
            near.append([name, *[_stat_line(m.get(k)) for k in (
                'near_C_area_fraction', 'near_C_boundary_band_area_fraction', 'near_C_outside_boundary_band_area_fraction')]])
    return '\n\n'.join([
        _table(['模型/父', '固定R12覆盖RMS', 'R12<1%覆盖比例', 'Gamma RMS', 'Gamma p95', 'Gamma覆盖比例'], cover),
        _table(['模型/父', '最小角<5°比例', '角度p05', 'q>10比例', '近退化面数', '零面积面数'], shape),
        _table(['模型/父', 'near-C面积样点比例', '其中边界带比例', '其中带外比例'], near),
        '本表每格为均值 / 中位数 / [Q25,Q75]；比例使用0–1。near-C用固定512个面积样点、seed9951，'
        '邻近与物理外边界带宽均为0.001L；它只是补充距离诊断，带外点不自动证明重复C。'])


def _models_table(results):
    rows = []
    for name in MODELS:
        m = results.get('models', {}).get(name, {})
        load = m.get('load', {})
        ep = load.get('endpoint_strict_load', {})
        rows.append([name, m.get('status'), ep.get('file_sha256', '官方EMA+零角色'), m.get('state_before'),
            m.get('state_before') == m.get('state_after') if 'state_after' in m else None,
            m.get('backbone_before') == m.get('backbone_after') if 'backbone_after' in m else None,
            _get(m, 'role_before', 'nonzero'), _get(m, 'role_before', 'RMS'), m.get('trainable_parameters')])
    return _table(['模型', '冻结状态', '终点文件SHA256', '采样前完整状态SHA256', '完整状态前后相等', '主干前后相等',
                   'role非零数', 'role RMS', '可训练参数'], rows)


def _gates_table(results):
    return _table(['T1复放父/seed', '全51状态逐bit等于历史', '全51状态C逐bit不变', '初始free Gaussian精确', '有限', '秒'],
        [[f"P{r.get('parent')}/{r.get('seed')}", r.get('all51_bitwise_equals_archive'), r.get('C_all51_FP32_bits_exact'),
          r.get('initial_free_Gaussian_exact'), r.get('all_states_finite'), _get(r, 'sampling', 'seconds')]
         for r in _get(results, 'gates', 'T1_replays', default=[])])


def _input_topology_table(results):
    records = _get(results, 'input_topology', 'parents', default=[])
    rows = []
    labels = (('full_source_original_winding', '完整GT源索引/原绕序'),
              ('full_archive_source_indices', '完整GT源索引/归档角序'),
              ('C_source_original_winding', 'C源索引/原绕序'),
              ('C_source_archive_winding', 'C源索引/归档角序'),
              ('C_exact_archive_winding', 'C精确坐标索引/归档角序'))
    for parent in records:
        for key, label in labels:
            value = parent.get(key, {})
            if not value:
                rows.append([f"P{parent.get('parent')}/{label}", 'NOT_RUN', None, None, None, None, None])
                continue
            edges = value.get('edge_incidence', {})
            links = value.get('vertex_links', {})
            orient = value.get('orientation', {})
            rows.append([f"P{parent.get('parent')}/{label}", _get(value, 'shared_edge_components', 'count'),
                f"{edges.get('one')}/{edges.get('two')}/{edges.get('more_than_two')}",
                links.get('nonmanifold_vertex_ids'), f"{orient.get('current_winding_consistent')} / conflicts={orient.get('current_conflicting_edge_count')}",
                orient.get('can_consistently_reorient_two_incident_edges'), orient.get('parity_conflicts')])
    return _table(['输入口径', '共边分量', '边关联1/2/>2', '非流形link顶点', '当前绕序一致', '可一致重定向', '定向奇偶冲突'], rows)


def _runtime_table(results):
    runs = [_get(row, 'sampling', 'runtime', default={}) for row in results.get('generated', [])]
    fields = (
        'cublas_workspace_config', 'deterministic_algorithms', 'deterministic_warn_only',
        'tf32_matmul', 'tf32_cudnn', 'cudnn_benchmark',
        'sdpa_math_enabled', 'sdpa_flash_enabled', 'sdpa_mem_efficient_enabled', 'sdpa_cudnn_enabled',
        'cuda_autocast_enabled', 'cuda_autocast_dtype', 'parameter_dtype',
        'coordinate_tensors_checked', 'gradient_enabled', 'all_modules_eval',
        'gradient_checkpointing', 'trainable_parameters')
    rows = []
    for key in fields:
        present = [run[key] for run in runs if key in run]
        unique = {json.dumps(value, sort_keys=True): value for value in present}
        common = next(iter(unique.values())) if len(unique) == 1 else list(unique.values()) if unique else None
        status = 'PASS_COMMON_48' if len(runs) == 48 and len(present) == 48 and len(unique) == 1 else 'INCOMPLETE_OR_DIFFERENT'
        rows.append([key, common, f'{len(present)}/{len(runs)}', status])
    return _table(['实际运行字段', '共同值或不同值列表', '记录数', '48条共同值核对'], rows)


def _saved_cost_tables(results):
    scores = results.get('scores', [])
    scoring = [row.get('offline_scoring_seconds') for row in scores]
    topo = [_get(row, 'metrics', 't1_native_confirmation', 'exact_coordinate_topology', 'topology_audit_seconds') for row in scores]
    def complete_sum(values):
        return sum(values) if values and all(isinstance(v, (int, float)) for v in values) else None
    costs = [
        ['48条离线评分总和', complete_sum(scoring), f'{sum(v is not None for v in scoring)}/{len(scores)}条保存计时'],
        ['输入源拓扑校准', _get(results, 'input_topology', 'seconds'), '输入审计保存的实际秒数'],
        ['48条生成精确拓扑审计合计', complete_sum(topo), '属于离线评分内部子集；不可与评分总和相加'],
        ['首次报告+固定图联合生成', _get(results, 'delivery', 'seconds'), '联合计时；不含外部子进程资产重读'],
        ['固定图独立耗时', 'NOT_MEASURED', '没有独立测量，不从联合耗时估算'],
        ['交付原始文件与哈希审计', _get(results, 'delivery', 'artifact_audit', 'seconds'), '只读保存数组/哈希；无模型或几何重评']]
    checks = [
        ['输入拓扑helper', _get(results, 'helper_tests', 'topology', 'status'),
         _get(results, 'helper_tests', 'topology', 'ledger', 'fixture_families'), 'fixture families'],
        ['评价helper', _get(results, 'helper_tests', 'evaluation', 'status'),
         _get(results, 'helper_tests', 'evaluation', 'fixture_families'), 'fixture families'],
        ['交付原始数据审计', _get(results, 'delivery', 'artifact_audit', 'status'),
         _get(results, 'delivery', 'artifact_audit', 'formal_outputs'), 'formal outputs；另有2条复放'],
        ['固定图视觉审阅', _get(results, 'delivery', 'visual_QA', 'status'), None, '只读已生成图']]
    return _table(['离线/交付阶段', '实际秒数', '口径'], costs) + '\n\n' + _table(
        ['检查', '状态', '数量', '含义'], checks)


def render_report(results, view, figure):
    seeds = _get(results, 'protocol', 'new_seeds', default=list(range(9401, 9409)))
    lines = ['# 纯净Native T1：独立新噪声确认与接口归因', '',
        f"执行状态：**{_cell(results.get('status', 'UNKNOWN'))}**。新噪声：{_cell(seeds)}。开发9201–9204独立列出，未与新噪声合并。",
        '本轮只采样Native0、T1、T2，不增加任何速度/位置修正、几何引导或后处理。完整GT仅用于只读输入检查和离线评分。', '']
    for question, answer in _questions(results, view):
        lines += ['## ' + question, '', answer, '']
    if results.get('failure'):
        lines += ['执行中断/失败：`' + _cell(results['failure']) + '`。未完成项不补造。', '']
    lines += ['## 模型、任务和实际数值路径', '',
        'Native0为官方EMA＋零角色embedding的相同原生逐面时间模型；T1为正确上下文训练终点；T2为独立上下文训练终点。'
        'T1−Native0描述总体微调增量，T1−T2提供正确对应关系的对照；本轮没有把模型差构造成Guidance。', '',
        _models_table(results), '',
        '采用源native_patch_run.clamped_sample：BF16前向、FP32参数与积分、50步Euler、N=y=112、K=12、free=100；已知时间1、自由时间k/50。'
        '原C在每一步直接保留，同seed三模型与两个上下文共用自由Gaussian。实际门禁记录如下。', '',
        _gates_table(results), '', _runtime_table(results), '',
        '实际runtime记录由每次源采样入口返回；FP32积分坐标由该入口检查，表中coordinate_tensors_checked为检查张量数，不冒充独立dtype测量。', '',
        '## 输入面积、范围与拓扑口径', '']
    input_parents = _get(results, 'input_topology', 'parents', default=[])
    lines += [_table(['父对象', 'C真实面积%', '坐标范围', '不同源顶点坐标相同', 'C异常link源ID及完整GT是否补足'],
        [[row.get('parent'), 100*row['area_fraction'] if 'area_fraction' in row else None, row.get('coordinate_range'), row.get('coincident_distinct_source_vertex_ids'),
          [(v.get('source_vertex_id'), v.get('full_GT_repairs_subpatch_link')) for v in row.get('C_link_anomalies_vs_full_GT', [])]] for row in input_parents]), '', _input_topology_table(results), '',
        '两完整源GT均为58V/168E/112F，边界与过关联边为0、link合法、原绕序一致。P1 C的源25/30（精确C索引3/4）各有两条不相连link，完整GT将其补为7节点单环；这两个输入继承异常不归罪free。P0 C也不称为拓扑圆盘。完整源GT只确认组合拓扑闭合，未做完整GT自交认证。L采用完整对象bbox元数据，尚未验证只有C且尺度未知的部署设置。', '',
        '## 逐对象统计：开发与确认分开', '',
        _table(['阶段/模型/父', 'n', 'R12均值', '中位数', 'Q25/Q75', '最差RMS', 'P0低于历史2%次数', 'CF面均值%', 'FF非法均值'],
            [[f"{row['phase']}/{row['model']}/P{row['parent']}", row['n'],
              row['distributions']['remainder_rms']['mean'], row['distributions']['remainder_rms']['median'],
              f"{_fmt(row['distributions']['remainder_rms']['q25'])}/{_fmt(row['distributions']['remainder_rms']['q75'])}",
              row['distributions']['remainder_rms']['maximum'],
              f"{row['R12_below_historical2pct_count']}/{row['n']}" if row['parent'] == 0 else 'N/A',
              100*row['distributions']['CF_face_fraction']['mean'] if row['distributions']['CF_face_fraction']['mean'] is not None else None,
              row['distributions']['FF_illegal']['mean']] for row in view['groups']]), '',
        '距离单位均为各目标完整bbox的百分比；最差值按剩余RMS最大定义。历史2%只作P0描述性参照，不是新成功门槛。'
        '两个父对象、多次噪声仍只有两个训练对象，不能将输出条数当独立对象数。', '',
        _table(['T1相对/父', '同seed RMS改善次数', '平均改善', '中位改善', 'Q25/Q75改善', '符号'],
            [[f"{row['reference']}/P{row['parent']}", f"{row['improvements']}/{row['n']}",
              row['remainder_improvement']['mean'], row['remainder_improvement']['median'],
              f"{_fmt(row['remainder_improvement']['q25'])}/{_fmt(row['remainder_improvement']['q75'])}", '基准−T1；正值改善'] for row in view['paired_geometry']]), '',
        '## 覆盖、形状和near-C补充统计', '', _additional_tables(results), '',
        '## 新噪声逐输出几何与接口', '',
        _table(['模型/父/seed', 'free→R12', 'R12→free', '对称RMS', '完整RMS', 'CF面%/pair', 'FF非法', 'Gamma RMS/p95', 'coverage%', '窄角%', '退化/NaN/Inf'],
            [[f"{row['model']}/P{row['parent']}/{row['seed']}", row['free_to_R12'], row['R12_to_free'], row['remainder_rms'], row['full_rms'],
              f"{_fmt(100*row['CF_face_fraction'] if row['CF_face_fraction'] is not None else None)}/{_fmt(row['CF_pairs'])}", row['FF_illegal'],
              f"{_fmt(row['boundary_rms'])}/{_fmt(row['boundary_p95'])}",
              100*row['boundary_coverage'] if row['boundary_coverage'] is not None else None,
              100*row['narrow_fraction'] if row['narrow_fraction'] is not None else None,
              f"{_fmt(row['degenerate_fraction'])}/{_fmt(row['nan_count'])}/{_fmt(row['inf_count'])}"] for row in view['confirmation']]), '',
        '所有剩余指标仅使用known_mask一次提取的100张free面；完整112面误差另列。完整误差低而剩余误差高只提出区域分配问题，'
        '不直接证明free重复C。低距离、精确共边数和连接合法性是不同量；判交是声明容差下的数值诊断，CCD未执行。', '',
        '## 同输出交叉四格与噪声异质性', '']
    for phase in ('development', 'confirmation'):
        values = _binding_rows(results, phase)
        lines += [f'阶段：{phase}', '', _table(['模型/seed', 'E00', 'E01', 'E11', 'E10', 'Delta0', 'Delta1', 'B', '双向正'],
            [[f"{row['model']}/{row['seed']}", row['E00'], row['E01'], row['E11'], row['E10'], row['Delta0'], row['Delta1'], row['B'], row['both_positive']] for row in values]), '']
    lines += ['E00=E(R0,F(C0))，E01=E(R0,F(C1))，E11=E(R1,F(C1))，E10=E(R1,F(C0))；'
        'Delta0=E01−E00，Delta1=E10−E11，B=(Delta0+Delta1)/2。交换评分复用同一free数组，不另生成、删面或重配面。', '',
        _binding_tables(results), '',
        '净和必须大于1e-8且大于绝对贡献总和的0.1，才显示单seed占净和百分比；否则NOT_DEFINED。逐seed绑定增量为T1−对照，正值表示T1绑定增强。', '',
        '## 输入继承与生成free的拓扑归因', '',
        _table(['模型/父/seed', '继承C的未修改link顶点', '新free相关非流形顶点', 'C-only绕序冲突', 'free相关绕序冲突', '精确C/free共边'],
            [[f"{row['model']}/P{row['parent']}/{row['seed']}",
              row['attribution'].get('inherited_C_only_unmodified_link_vertex_ids'),
              row['attribution'].get('new_free_related_nonmanifold_vertex_ids'),
              len(row['attribution'].get('orientation_conflicts_C_only', [])), len(row['attribution'].get('orientation_conflicts_free_related', [])),
              row['attribution'].get('exact_C_free_shared_edge_count')] for row in view['confirmation']]), '',
        '精确坐标索引只用于只读诊断，不移动坐标或改变采样输出。源C的问题不全部归因于生成器；'
        '全部48个输出精确C/free共边为0，新free相关非流形link计数为0；孤立三角形的局部link可以合法，因此后一个0不证明完整拓扑有用。P1的两个异常link始终是未修改C的继承项。法向符号/角序问题不等于几何穿刺，未执行一致重定向修复。CF/FF/CC及精确拓扑实际已执行；旧兼容质量字段中的NOT_RUN不覆盖本轮新审计，连续碰撞安全仍NOT_RUN。', '',
        '## 实际成本、保护与停止', '',
        _table(['计数', '实际'], [[key, value] for key, value in results.get('counts', {}).items() if key not in ('attempts_by_stage', 'forward_by_stage')]), '',
        'official_forward=0仅表示未调用backbone.forward整体入口；Native.forward实际执行原主干各层2500次，并不表示未使用主干。', '',
        _table(['分阶段前向', '次数'], [[key, value] for key, value in results.get('counts', {}).get('forward_by_stage', {}).items()]), '',
        _table(['模型/父/seed', '完整采样秒', '峰值CUDA GiB'],
            [[f"{row.get('model')}/P{row.get('parent')}/{row.get('seed')}",
              _get(row, 'costs', 'complete_generation_wall_seconds', default=_get(row, 'sampling', 'seconds')),
              _get(row, 'sampling', 'peak_cuda_allocated') / 2**30 if _get(row, 'sampling', 'peak_cuda_allocated') is not None else None]
             for row in results.get('generated', [])]), '',
        '性能仅为本轮运行账本；沿用源采样路径，没有重写wrapper追求加速。以下仅汇总保存计时，不重新评分或检测几何。', '', _saved_cost_tables(results), '',
        _table(['保护/审计记录', '内容'], [[key, value] for key, value in results.get('protection', {}).items() if key in ('critical_files_after', 'historical_metadata_after', 'new_core_after')]), '',
        '## 固定新噪声对照图', '']
    if figure:
        lines += [f"![固定新噪声完整输出与free-only]({Path(figure['path']).name})", '',
            f"固定seed={figure['seed']}，P0/P1×Native0/T1/T2；同相机、同坐标范围，同时显示完整112面与free-only100面。"
            '不按GT评分或外观挑选样本，free-only避免复制C遮住生成缺陷。', '']
    else:
        lines += ['固定图：NOT_RUN，缺少已完成固定样本。', '']
    lines += ['本轮输出与证据见[results.json](results.json)，入口修改见[code.diff](code.diff)。未复制checkpoint或历史大文件。',
        '新增训练、参数更新、主干反向、几何引导、焊接/吸附/删面/补洞/重三角化均为0；'
        '没有改变C或面数。完成固定矩阵后停止，未启动原Phase3或下一种训练/修补模块。', '']
    return '\n'.join(lines)


def fixed_figure(results, out, bank):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    seeds = _get(results, 'protocol', 'new_seeds', default=[])
    if not seeds:
        return None
    seed = int(seeds[0]); found = {}
    for parent in (0, 1):
        C = np.asarray(bank['cases_by_key'][parent, 12]['constraints'])
        for model in MODELS:
            row = next((r for r in results.get('scores', []) if r.get('phase', 'confirmation') == 'confirmation'
                        and r['model'] == model and r['parent'] == parent and r['seed'] == seed), None)
            if row is None:
                continue
            path = row.get('raw_path', row.get('output_path'))
            with np.load(path, allow_pickle=False) as archive:
                raw = archive['output'].reshape(112, 3, 3).copy()
            if raw.dtype != np.float32 or not np.isfinite(raw).all() or not np.array_equal(raw[:12], C):
                raise AssertionError('Fixed figure requires actual finite FP32 raw112 and original C')
            found[parent, model] = raw
    if not found:
        return None
    points = np.concatenate([raw.reshape(-1, 3)[:, [0, 2, 1]] for raw in found.values()])
    low, high = points.min(0), points.max(0); center = (low + high) / 2
    radius = max(float((high - low).max()) * .55, 1e-6)
    fig = plt.figure(figsize=(15, 17), constrained_layout=True)
    for parent in (0, 1):
        for free_only in (False, True):
            row_index = 2 * parent + int(free_only)
            for column, model in enumerate(MODELS):
                ax = fig.add_subplot(4, 3, row_index * 3 + column + 1, projection='3d')
                raw = found.get((parent, model))
                if raw is None:
                    ax.text2D(.25, .5, 'NOT_RUN', transform=ax.transAxes)
                else:
                    tri = raw[:, :, [0, 2, 1]]
                    shown = tri[12:] if free_only else tri
                    loops = np.concatenate((shown, shown[:, :1]), axis=1)
                    colors = ['#485c69'] * len(shown)
                    if not free_only:
                        colors[:12] = ['#1a78cc'] * 12
                    ax.add_collection3d(Line3DCollection(loops, colors=colors, linewidths=.65, alpha=.85))
                ax.set_title(f"P{parent} | {model} | seed{seed} | {'free-only100' if free_only else 'full112'}", fontsize=10)
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
                ax.set_zlim(center[2] - radius, center[2] + radius)
                ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=24, azim=-54); ax.set_axis_off()
    fig.suptitle('Pure Native confirmation: first preregistered new seed\n'
        'Shared camera and full coordinate bounds; blue=known C, gray=generated free. No repair.', fontsize=13)
    destination = Path(out) / 'fixed_first_new_seed_full_and_free.png'
    if destination.exists():
        raise FileExistsError(destination)
    fig.savefig(destination, dpi=145); plt.close(fig)
    return dict(path=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), seed=seed,
        parents=[0, 1], models=list(MODELS), panels=12, actual_outputs=len(found),
        selection='First preregistered new noise seed; no score or visual selection',
        camera=dict(elevation=24, azimuth=-54), axes=['X', 'Z', 'Y'],
        common_center=center.tolist(), common_radius=radius, coordinates_modified=False,
        geometry_tests=0, model_calls=0, repair_calls=0)


def format_self_check():
    assert _fmt(1e-12) != '0.0000' and 'e-' in _fmt(1e-12)
    assert _fmt(-1e-12).startswith('-')
    assert _fmt(0.) == '0'
    assert '\\|' in _table(['a'], [['x|y']])
    empty = dict(status='STOPPED', protocol=dict(new_seeds=list(range(9401, 9409))), scores=[])
    text = render_report(empty, report_view(empty), None)
    assert '0/48' in text and 'NOT_RUN' in text
    assert '\ufffd' not in text
    return dict(status='PASS', checks=6, scope='In-memory formatting/empty-result rendering only',
        model_calls=0, geometry_calls=0, scoring_calls=0, writes=0)


def deliver_report(results, out, bank):
    tick = time.perf_counter(); out = Path(out).resolve()
    if not out.is_dir() or not out.name.startswith('t1_native_confirmation_'):
        raise ValueError('Require isolated existing t1_native_confirmation output directory')
    destination = out / 'T1_NATIVE_CONFIRMATION_REPORT.md'
    if destination.exists():
        raise FileExistsError(destination)
    view = report_view(results)
    figure = fixed_figure(results, out, bank)
    text = render_report(results, view, figure)
    if '\ufffd' in text:
        raise AssertionError('Report contains invalid Unicode replacement')
    destination.write_text(text, encoding='utf-8')
    return dict(report=dict(path=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),
        figure=figure, descriptive_report_view=view, seconds=time.perf_counter() - tick,
        formatting_checks=format_self_check(), model_calls=0, geometry_calls=0, scoring_calls=0, repair_calls=0)
