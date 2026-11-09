"""Summarize all 72 clay slots without repairing, retrying or hiding failures."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from common import load_mesh

ARMS = ('freeze', 'transport', 'refit')
SOURCES = ('sphere', 'thin_u', 'bottle_local_1')
LABELS = {'sphere': 'Sphere squash', 'thin_u': 'Thin-U squeeze', 'bottle_local_1': 'Bottle bend'}
COLORS = {'freeze': '#2875ac', 'transport': '#e1812c', 'refit': '#389157'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def reference_check(root, name, spec):
    directory = root / 'inputs' / name
    result = dict(object=name, source_object=spec['source_object'], frame=spec['frame'], t=spec['t'])
    reasons = []
    required = ['reference.obj', 'identity.json', 'reference_audit.json', 'reference_exact_audit.json']
    missing = [item for item in required if not (directory / item).is_file()]
    if missing:
        result.update(valid=False, failure_reasons=['missing_reference_artifacts:' + ','.join(missing)])
        return result
    identity = read(directory / 'identity.json')
    audit = read(directory / 'reference_audit.json')
    exact = read(directory / 'reference_exact_audit.json')
    digest = sha(directory / 'reference.obj')
    if identity.get('reference_sha256') != digest:
        reasons.append('identity_reference_hash_mismatch')
    if exact.get('source_sha256') != digest:
        reasons.append('exact_audit_reference_hash_mismatch')
    if not audit.get('closed_oriented_2manifold', False):
        reasons.append('reference_not_closed_oriented_2manifold')
    expected = spec['expected_betti']
    betti = [audit.get('beta' + str(i)) for i in range(3)]
    if betti != expected:
        reasons.append('reference_betti_mismatch')
    for key in ['duplicate_faces', 'float_zero_area_faces', 'boundary_edges',
                'nonmanifold_edges', 'nonmanifold_vertices', 'inconsistent_oriented_edges']:
        if audit.get(key) != 0:
            reasons.append('reference_' + key)
    for key in ['exact_intersection_pairs', 'exact_degenerate_faces']:
        if exact.get(key) != 0:
            reasons.append('reference_' + key)
    v, f = load_mesh(directory / 'reference.obj')
    if not np.isfinite(v).all():
        reasons.append('reference_nonfinite_vertices')
    diagonal = float(np.linalg.norm(np.ptp(v, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0:
        reasons.append('reference_invalid_bbox')
    if len(f) != audit.get('faces') or len(v) != audit.get('vertices'):
        reasons.append('reference_audit_shape_mismatch')
    approx = identity.get('reference_approximation')
    normalized = {key + '_pct_bboxdiag': 100 * value / diagonal for key, value in approx.items()} if approx else None
    result.update(valid=not reasons, failure_reasons=reasons, reference_sha256=digest,
                  vertices=len(v), faces=len(f), betti=betti, reference_bbox_diagonal=diagonal,
                  minimum_J_determinant=identity.get('minimum_J_determinant'),
                  corner_margin=identity.get('corner_margin'), reference_approximation_absolute=approx,
                  reference_approximation_pct_bboxdiag=normalized)
    return result


def method_failures(row):
    reasons = []
    if row.get('status') != 'EXTRACTED':
        reasons.append('status:' + str(row.get('status')))
    for key in ['geometry_audited', 'strict_all_vertex_links_simple_cycles',
                'closed_oriented_2manifold', 'betti_match']:
        if row.get(key) is not True:
            reasons.append(key)
    for key in ['exact_intersection_pairs', 'exact_degenerate_faces', 'duplicate_faces']:
        if row.get(key) != 0:
            reasons.append(key)
    graph = row.get('graph_capacity_stats') or {}
    if not graph:
        reasons.append('missing_actual_graph_capacity_audit')
    else:
        if graph.get('nonfinite_count') != 0:
            reasons.append('actual_graph_nonfinite_capacity')
        if graph.get('negative_count') != 0:
            reasons.append('actual_graph_negative_capacity')
    return reasons


def group_statistics(rows):
    counts = collections.Counter(row['status'] for row in rows)
    return dict(requested_slots=len(rows), joint_valid=sum(row['joint_valid'] for row in rows),
                reference_invalid_slots=sum(not row['reference_input_valid'] for row in rows),
                comparable_slots=sum(row['reference_input_valid'] for row in rows),
                method_contract_failures_on_valid_reference=sum(row['reference_input_valid'] and not row['method_contract_pass'] for row in rows),
                status_counts=dict(counts), geometry_audited=sum(bool(row.get('geometry_audited')) for row in rows),
                topology_matching=sum(bool(row.get('betti_match')) for row in rows),
                strict_vertex_links_pass=sum(bool(row.get('strict_all_vertex_links_simple_cycles')) for row in rows),
                arccos_clipped_dot_count=sum(row.get('arccos_clipped_count', 0) for row in rows),
                runs_with_arccos_clamp=sum(row.get('arccos_clipped_count', 0) > 0 for row in rows),
                runs_with_candidate_negative_scores=sum((row.get('candidate_score_stats') or {}).get('negative_count', 0) > 0 for row in rows),
                runs_with_invalid_actual_graph=sum((row.get('graph_capacity_stats') or {}).get('negative_count', 0) > 0 or
                                                  (row.get('graph_capacity_stats') or {}).get('nonfinite_count', 0) > 0 for row in rows))


def paired_statistics(by_key, frames, left, right):
    categories = collections.Counter()
    for source, frame in frames:
        a, b = by_key[source, frame, left], by_key[source, frame, right]
        if not a['reference_input_valid']:
            categories['reference_invalid'] += 1
        elif a['joint_valid'] and b['joint_valid']:
            av, bv = a.get('rms_pct_bboxdiag'), b.get('rms_pct_bboxdiag')
            if av is None or bv is None:
                categories['both_valid_metric_missing'] += 1
            elif av == bv:
                categories['both_valid_equal_rms'] += 1
            else:
                categories['both_valid_' + ('left' if av < bv else 'right') + '_lower_rms'] += 1
        elif a['joint_valid']:
            categories['only_left_valid'] += 1
        elif b['joint_valid']:
            categories['only_right_valid'] += 1
        else:
            categories['both_invalid'] += 1
    return dict(left=left, right=right, requested_frame_pairs=len(frames), counts=dict(categories),
                note='Repeated frames are dependent measurements of three objects; no significance or generalization claim.')


def error_figure(root, rows, references):
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.6), sharex=True)
    for column, source in enumerate(SOURCES):
        for arm in ARMS:
            selected = sorted((row for row in rows if row['source_object'] == source and row['state'] == arm), key=lambda r: r['frame'])
            times = [row['t'] for row in selected]
            for line, metric in enumerate(['rms_pct_bboxdiag', 'p95_pct_bboxdiag']):
                values = [row.get(metric, np.nan) for row in selected]
                axes[line, column].plot(times, values, color=COLORS[arm], marker='o', markersize=3.6, label=arm)
                bad = [i for i, row in enumerate(selected) if not row['joint_valid'] and row.get(metric) is not None]
                axes[line, column].scatter([times[i] for i in bad], [values[i] for i in bad], marker='x',
                                          color='#ac1414', s=65, linewidths=1.8, zorder=5)
        refs = sorted((x for x in references.values() if x['source_object'] == source), key=lambda r: r['frame'])
        for line, metric in enumerate(['rms_pct_bboxdiag', 'p95_pct_bboxdiag']):
            approx = [(x.get('reference_approximation_pct_bboxdiag') or {}).get(metric, np.nan) for x in refs]
            axes[line, column].plot([x['t'] for x in refs], approx, ':', color='#777777', linewidth=1.2,
                                    label='PL point-map discrepancy')
            for x in refs:
                if not x['valid']:
                    axes[line, column].axvline(x['t'], color='#ac1414', alpha=.3, linewidth=3)
            axes[line, column].grid(alpha=.22)
            axes[line, column].set_ylabel(('RMS' if line == 0 else 'p95') + ' (% current reference bbox diagonal)')
            axes[line, column].set_xlim(-.015, 1.015)
        axes[0, column].set_title(LABELS[source])
        axes[1, column].set_xlabel('Frozen path parameter t')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker='x', color='#ac1414', linestyle='None', label='Joint invalid output'))
    labels.append('Joint invalid output')
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9)
    fig.suptitle('Controlled clay paths: all frames, no failed-output filtering', fontsize=13)
    fig.tight_layout(rect=(0, .065, 1, .965))
    fig.savefig(root / 'figures/error_paths.png', dpi=160)
    plt.close(fig)


def contact_figure(root, by_key, references):
    chosen_frames = (0, 4, 7)
    fig = plt.figure(figsize=(24, 10))
    light = np.array([.3, -.4, .85]); light /= np.linalg.norm(light)
    for si, source in enumerate(SOURCES):
        # Same framing for every arm and selected time of one object.
        ref_vertices = [load_mesh(root / 'inputs' / f'{source}_frame{frame:02d}' / 'reference.obj')[0]
                        for frame in chosen_frames]
        all_v = np.concatenate(ref_vertices)
        centre = (all_v.min(0) + all_v.max(0)) / 2
        radius = max(float(np.ptp(all_v, axis=0).max()) * .56, 1e-6)
        for ai, arm in enumerate(ARMS):
            for fi, frame in enumerate(chosen_frames):
                row = by_key[source, frame, arm]
                ax = fig.add_subplot(3, 9, si * 9 + ai * 3 + fi + 1, projection='3d')
                path = root / 'runs' / f"{row['object']}__{arm}" / 'mesh.obj'
                if path.is_file():
                    v, f = load_mesh(path)
                    tri = v[f]
                    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                    lengths = np.linalg.norm(normals, axis=1)
                    normals /= np.maximum(lengths[:, None], 1e-300)
                    shade = .38 + .62 * np.abs(normals @ light)
                    base = np.array([.30, .55, .72]) if row['joint_valid'] else np.array([.78, .36, .29])
                    collection = Poly3DCollection(tri, facecolors=shade[:, None] * base[None, :],
                                                  edgecolors=(.1, .1, .1, .15), linewidths=.14)
                    ax.add_collection3d(collection)
                else:
                    ax.text2D(.5, .45, 'No saved mesh\n' + str(row.get('status')), transform=ax.transAxes,
                              ha='center', fontsize=7, color='#a00000')
                for setter, coordinate in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
                    setter(centre[coordinate] - radius, centre[coordinate] + radius)
                ax.set_box_aspect((1, 1, 1)); ax.set_proj_type('ortho')
                ax.view_init(elev=54 if source == 'thin_u' else 29, azim=-65)
                ax.set_axis_off()
                rms = row.get('rms_pct_bboxdiag')
                rms_text = 'missing' if rms is None else f'{rms:.4f}%'
                validity = 'VALID' if row['joint_valid'] else ('REF INVALID' if not row['reference_input_valid'] else 'INVALID')
                ax.set_title(f"{source}\n{arm} | t={row['t']:.3f}\nF={row.get('faces', '?')} RMS={rms_text} | {validity}",
                             fontsize=7.3, color='#202020' if row['joint_valid'] else '#a00000', pad=1)
    fig.suptitle('All arms at fixed frames 0, 4, 7; red meshes fail the joint contract', fontsize=13)
    fig.subplots_adjust(left=.01, right=.99, bottom=.015, top=.92, hspace=.32, wspace=.02)
    fig.savefig(root / 'figures/contact_sheet.png', dpi=160)
    plt.close(fig)


def summarize(root):
    root = Path(root).resolve()
    protocol = read(root / 'FROZEN_PROTOCOL.json')
    expected_hash = read(root / 'FROZEN_PROTOCOL_SHA256.json')['sha256']
    assert sha(root / 'FROZEN_PROTOCOL.json') == expected_hash, 'Frozen protocol hash mismatch'
    rows = read(root / 'LEDGER.json')
    assert len(rows) == 72, f'Expected 72 slots, received {len(rows)}'
    assert set(protocol['objects']) == {f'{source}_frame{frame:02d}' for source in SOURCES for frame in range(8)}, 'Unexpected frame manifest'
    by_key = {(row['source_object'], int(row['frame']), row['state']): row for row in rows}
    expected = {(source, frame, arm) for source in SOURCES for frame in range(8) for arm in ARMS}
    assert len(by_key) == 72 and set(by_key) == expected, 'Missing, duplicate or unexpected arm slots'
    source_checks = []
    for item in protocol['config']['source_files']:
        path = Path(item['path'])
        actual = sha(path) if path.is_file() else None
        source_checks.append(dict(path=str(path), expected_sha256=item['sha256'], actual_sha256=actual,
                                  passed=actual == item['sha256']))
    assert all(x['passed'] for x in source_checks), 'Archived reference or clean-fit source hash changed'
    references = {name: reference_check(root, name, spec) for name, spec in protocol['objects'].items()}
    frames = [(source, frame) for source in SOURCES for frame in range(8)]
    input_checks = []
    initial_checks = []
    for source, frame in frames:
        group = [by_key[source, frame, arm] for arm in ARMS]
        hashes = [row.get('input_vertex_sha256') for row in group]
        assert hashes[0] and len(set(hashes)) == 1, f'Input vertices differ between arms: {source}/{frame}'
        ts = [row['t'] for row in group]
        assert len(set(ts)) == 1 and ts[0] == protocol['objects'][group[0]['object']]['t'], 'Path parameter mismatch'
        input_checks.append(dict(source=source, frame=frame, input_vertex_sha256=hashes[0], passed=True))
        if frame == 0:
            mesh_hashes = [row.get('mesh_sha256') for row in group]
            assert mesh_hashes[0] and len(set(mesh_hashes)) == 1, f'Initial meshes differ or are missing: {source}'
            initial_checks.append(dict(source=source, mesh_sha256=mesh_hashes[0], passed=True))
    failures = []
    for row in rows:
        ref = references[row['object']]
        reasons = method_failures(row)
        row['reference_input_valid'] = ref['valid']
        row['method_contract_failure_reasons'] = reasons
        row['method_contract_pass'] = not reasons
        row['joint_valid'] = ref['valid'] and not reasons
        previous = by_key.get((row['source_object'], row['frame'] - 1, row['state']))
        current_hash = row.get('surface_input_faces_sha256')
        previous_hash = previous.get('surface_input_faces_sha256') if previous else None
        row['connectivity_changed_from_previous_frame'] = (current_hash != previous_hash) if current_hash and previous_hash else None
        if not row['joint_valid']:
            failures.append(dict(object=row['object'], source_object=row['source_object'], frame=row['frame'],
                                 t=row['t'], arm=row['state'], status=row['status'],
                                 failure_class='METHOD_CONTRACT_FAILED' if ref['valid'] else 'REFERENCE_INPUT_INVALID',
                                 reference_failure_reasons=ref['failure_reasons'], method_failure_reasons=reasons,
                                 mesh_sha256=row.get('mesh_sha256'), error=row.get('error'),
                                 rms_pct_bboxdiag=row.get('rms_pct_bboxdiag'), faces=row.get('faces')))
    wide = []
    selected_fields = ['status', 'joint_valid', 'method_contract_pass', 'method_contract_failure_reasons',
                       'faces', 'vertices', 'rms_pct_bboxdiag', 'p95_pct_bboxdiag', 'rms_absolute', 'p95_absolute',
                       'beta0', 'beta1', 'beta2', 'strict_noncycle_vertex_count', 'nonmanifold_edges',
                       'nonmanifold_vertices', 'exact_intersection_pairs', 'exact_degenerate_faces',
                       'duplicate_faces', 'arccos_clipped_count', 'surface_input_faces_sha256',
                       'connectivity_changed_from_previous_frame', 'surface_faces_added_from_previous',
                       'surface_faces_removed_from_previous', 'previous_successful_frame', 'previous_comparison_is_adjacent', 'candidate_score_stats', 'graph_capacity_stats']
    for source, frame in frames:
        group = [by_key[source, frame, arm] for arm in ARMS]
        ref = references[group[0]['object']]
        line = dict(source_object=source, object=group[0]['object'], frame=frame, t=group[0]['t'],
                    reference_input_valid=ref['valid'], reference_failure_reasons=ref['failure_reasons'],
                    reference_faces=ref.get('faces'), reference_sha256=ref.get('reference_sha256'),
                    reference_approximation_absolute=ref.get('reference_approximation_absolute'),
                    reference_approximation_pct_bboxdiag=ref.get('reference_approximation_pct_bboxdiag'),
                    input_vertex_sha256=group[0]['input_vertex_sha256'],
                    minimum_J_determinant=ref.get('minimum_J_determinant'), corner_margin=ref.get('corner_margin'))
        for row in group:
            for field in selected_fields:
                line[row['state'] + '__' + field] = row.get(field)
        line['all_arm_connectivities_identical'] = len({row.get('surface_input_faces_sha256') for row in group}) == 1 and bool(group[0].get('surface_input_faces_sha256'))
        wide.append(line)
    consistency_path = root / 'audit/POSTRUN_CONSISTENCY_CHECKS.json'
    consistency = read(consistency_path) if consistency_path.is_file() else dict(passed=False, missing=True)
    summary = dict(schema='CONTROLLED_CLAY_SUMMARY_V1', frozen_protocol_sha256=expected_hash,
                   requested_slots=72, object_count=3, frames_per_object=8, arms=list(ARMS),
                   network_training_updates=0, whole_experiment=group_statistics(rows),
                   reference_frames=dict(requested=24, valid=sum(x['valid'] for x in references.values()),
                                         invalid=sum(not x['valid'] for x in references.values())),
                   by_arm={arm: group_statistics([r for r in rows if r['state'] == arm]) for arm in ARMS},
                   by_source_arm={source: {arm: group_statistics([r for r in rows if r['state'] == arm and r['source_object'] == source])
                                           for arm in ARMS} for source in SOURCES},
                   endpoint_by_arm={arm: group_statistics([r for r in rows if r['state'] == arm and r['frame'] == 7]) for arm in ARMS},
                   pairwise=[paired_statistics(by_key, frames, 'freeze', 'transport'),
                             paired_statistics(by_key, frames, 'freeze', 'refit'),
                             paired_statistics(by_key, frames, 'transport', 'refit')],
                   source_identity_checks=source_checks, input_vertex_agreement_checks=input_checks,
                   t0_mesh_agreement_checks=initial_checks, reference_checks=list(references.values()),
                   method_failure_record_count=sum(x['failure_class'] == 'METHOD_CONTRACT_FAILED' for x in failures),
                   reference_failure_record_count=sum(x['failure_class'] == 'REFERENCE_INPUT_INVALID' for x in failures),
                   postrun_consistency_pass=bool(consistency.get('passed')),
                   postrun_consistency_note='Consistency PASS is separate from geometric validity or method success.',
                   decision='CONTROLLED_DIAGNOSTIC_ONLY_NO_TRAINING_OR_FEEDBACK_CLAIM',
                   limitations=['Three existing development objects, one deterministic path each and one sample seed.',
                                'Frames are correlated and are not additional independent objects.',
                                'Refit uses current target reference geometry and is an oracle diagnostic.',
                                'Local affine energy transport is not exact nonlinear Q transport, nor generally identical to unit-normal plane refitting; inverse-map Hessian terms are omitted.',
                                'Arms jointly change Q and n update rules; differences cannot be attributed to A alone.',
                                'PL point-map discrepancy is a sampled material-coordinate diagnostic, not a certified surface distance bound.',
                                'No learned trajectory, material simulation, online feedback mechanism, generalization or speed comparison was tested.',
                                'Validity has no RMS threshold; errors and actual face counts are reported separately.'])
    (root / 'reports').mkdir(exist_ok=True); (root / 'figures').mkdir(exist_ok=True)
    write_json(root / 'CLAY_SUMMARY.json', summary)
    write_json(root / 'FAILURES.json', dict(reference_failures=[x for x in references.values() if not x['valid']],
                                          output_failures=failures, all_72_slots_preserved=True))
    write_csv(root / 'reports/FRAME_TABLE.csv', wide)
    write_csv(root / 'reports/ENDPOINT_TABLE.csv', [row for row in wide if row['frame'] == 7])
    write_csv(root / 'CLAY_LEDGER.csv', rows)
    error_figure(root, rows, references)
    contact_figure(root, by_key, references)
    lines = ['# 受控形变下局部几何信息三路诊断', '',
             '**本轮是三条固定几何路径上的诊断，不是生成训练、反馈机制或材料模拟实验。**', '',
             '固定 sphere squash、thin-U squeeze、bottle bend 三条路径，各8个时间点（包含t=0），每点 freeze / transport / refit 三臂，共72个槽位。所有槽位保留；没有按成功样本过滤。', '',
             f"参考几何审核：{summary['reference_frames']['valid']}/24 合格；参考失败 {summary['reference_frames']['invalid']}/24。参考失败单列，不将其用于方法排名。", '',
             '|实验臂|联合有效 / 全部24槽位|参考失败槽位|合格参考上的方法失败|末帧联合有效 / 3|',
             '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        stats = summary['by_arm'][arm]
        lines.append(f"|{arm}|{stats['joint_valid']}/24|{stats['reference_invalid_slots']}|{stats['method_contract_failures_on_valid_reference']}|{summary['endpoint_by_arm'][arm]['joint_valid']}/3|")
    lines += ['', '联合有效要求：EXTRACTED状态、已经完成几何审计、每个顶点link为单一简单环、闭合定向二维流形、Betti匹配、无精确非法面交/退化/重复面，且实际聚合图容量有限非负。它不包含新设的RMS门槛。', '',
              f"后验一致性检查为 **{'PASS' if summary['postrun_consistency_pass'] else 'FAIL / MISSING'}**；这只代表记录和审计的一致性，不能覆盖上表中的几何失败。", '',
              '以下误差为当前参考包围盒对角线的百分比。失败输出如果具有可测网格，误差仍保留；不能仅依据较小误差忽略失败。', '',
              '|对象末帧|freeze RMS / 面数 / 有效|transport RMS / 面数 / 有效|refit RMS / 面数 / 有效|', '|---|---|---|---|']
    for source in SOURCES:
        cells = []
        for arm in ARMS:
            row = by_key[source, 7, arm]; value = row.get('rms_pct_bboxdiag')
            text = '未测' if value is None else f'{value:.6f}%'
            cells.append(f"{text} / {row.get('faces', '未测')} / {'是' if row['joint_valid'] else '否'}")
        lines.append('|' + source + '|' + '|'.join(cells) + '|')
    lines += ['', '![所有时间点的误差路径](figures/error_paths.png)', '',
              '红叉标记联合无效的可测输出；灰色点线是解析映射样本与声明PL参考重心插值之间的差异，仅作参考离散化诊断，不是另一方法或认证误差上界。', '',
              '![固定初、中、末帧的全部三臂输出](figures/contact_sheet.png)', '',
              '联系图固定显示frame 0、4、7，三臂采用同一对象的相机和显示范围；红色网格表示联合无效。没有挑选更好的时间点。', '',
              '## 协议与可解释范围', '',
              '每帧三臂input_vertex_sha256一致，三条路径的t=0三臂mesh_sha256相同；源reference与256元素fit文件hash已复核。固定顶点输入不意味着输出面数相同，实际面数和按输入顶点ID定义的连接变化均进入宽表。', '',
              'freeze保持局部A/g/c和平均法向，中心随共同顶点路径移动。transport使用局部Jacobian对二次能量做仿射传输，并逆转置法向方向；它不等于非线性曲面的精确Q，也不一般等于重新单位化每个源平面后聚合的Q。', '',
              'refit在当前参考三角面上，使用固定原采样点对应的细分子面重心坐标、固定Voronoi标签和面积变化权重重新聚合单位平面Q/平均法向；不重新定位顶点。它需要当前参考几何，是诊断性参照，不能宣称为现成生成能力。', '',
              f"三臂arccos边界夹取次数分别为 {', '.join(arm + '=' + str(summary['by_arm'][arm]['arccos_clipped_dot_count']) for arm in ARMS)}。夹取只处理协议允许的机器精度越界；实际图容量不做裁剪。负候选评分与非法聚合图容量分开记录。", '',
              '本轮没有训练网络，没有沿中间结果执行反馈修正，没有新增拓扑保障，也没有进行公平速度比较。三条路径的24帧不是24个独立对象；不可据此宣称类别泛化或连续时间内恒成立。', '',
              '## 可核查文件', '',
              '`CLAY_SUMMARY.json`：完整分母、逐对象/逐臂、末帧及配对分类；`CLAY_LEDGER.csv`：72行完整记录；`reports/FRAME_TABLE.csv`：24行三臂宽表；`reports/ENDPOINT_TABLE.csv`：3行末帧宽表；`FAILURES.json`：参考失败与方法失败分开；原始实验结果未被汇总脚本修改。', '']
    (root / 'REPORT_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(requested_slots=72, by_arm=summary['by_arm'], reference_frames=summary['reference_frames'],
                          report=str(root / 'REPORT_zh.md')), ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    summarize(parser.parse_args().root)


if __name__ == '__main__':
    main()
