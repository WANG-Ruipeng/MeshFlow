"""Summarize 48 new Q/n ablations and explicitly reuse 72 parent observations."""
from pathlib import Path
import argparse
import collections
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_clay import read, sha, write_json, write_csv, reference_check, method_failures, SOURCES

ARMS = ('freeze', 'q_only', 'n_only', 'transport', 'refit')
COLORS = dict(freeze='#2875ac', q_only='#9467bd', n_only='#c69a28', transport='#df7023', refit='#389157')


def stats(rows):
    return dict(requested_slots=len(rows), joint_valid=sum(r['joint_valid'] for r in rows),
                reference_invalid_slots=sum(not r['reference_input_valid'] for r in rows),
                method_failures_on_valid_reference=sum(r['reference_input_valid'] and not r['method_contract_pass'] for r in rows),
                status_counts=dict(collections.Counter(r['status'] for r in rows)),
                new_executions=sum(r['execution_origin'] == 'new_ablation' for r in rows),
                reused_parent_observations=sum(r['execution_origin'] == 'reused_parent' for r in rows))


def summarize(root):
    root = Path(root).resolve()
    protocol = read(root / 'FROZEN_PROTOCOL.json')
    digest = sha(root / 'FROZEN_PROTOCOL.json')
    assert digest == read(root / 'FROZEN_PROTOCOL_SHA256.json')['sha256'], 'Protocol hash mismatch'
    parent = Path(protocol['config']['parent_run']).resolve()
    parent_protocol = read(parent / 'FROZEN_PROTOCOL.json')
    assert sha(parent / 'FROZEN_PROTOCOL.json') == read(parent / 'FROZEN_PROTOCOL_SHA256.json')['sha256'], 'Parent protocol hash mismatch'
    fresh, inherited = read(root / 'LEDGER.json'), read(parent / 'LEDGER.json')
    assert len(fresh) == 48 and len(inherited) == 72, 'Expected 48 new and 72 reused slots'
    assert {r['state'] for r in fresh} == {'q_only', 'n_only'}, 'Unexpected new arms'
    assert {r['state'] for r in inherited} == {'freeze', 'transport', 'refit'}, 'Unexpected parent arms'
    rows = []
    for source_rows, origin, directory in [(inherited, 'reused_parent', parent), (fresh, 'new_ablation', root)]:
        for original in source_rows:
            row = dict(original)
            row.update(execution_origin=origin, measurement_run=str(directory))
            rows.append(row)
    keyed = {(r['source_object'], int(r['frame']), r['state']): r for r in rows}
    expected = {(s, f, a) for s in SOURCES for f in range(8) for a in ARMS}
    assert len(keyed) == 120 and set(keyed) == expected, 'Missing or duplicate combined slots'
    references = {}
    for name, spec in parent_protocol['objects'].items():
        old = reference_check(parent, name, spec)
        new = reference_check(root, name, spec)
        assert old.get('reference_sha256') == new.get('reference_sha256') and old.get('reference_sha256'), 'Reference bytes changed: ' + name
        references[name] = dict(valid=old['valid'] and new['valid'], parent=old, current=new)
    hash_checks = []
    for source in SOURCES:
        for frame in range(8):
            group = [keyed[source, frame, arm] for arm in ARMS]
            hashes = [r.get('input_vertex_sha256') for r in group]
            assert hashes[0] and len(set(hashes)) == 1, f'Input geometry changed across arms: {source}/{frame}'
            assert len({r['t'] for r in group}) == 1, 'Different path parameters'
            initial_hash = None
            if frame == 0:
                meshes = [r.get('mesh_sha256') for r in group]
                assert meshes[0] and len(set(meshes)) == 1, 't0 mesh mismatch: ' + source
                initial_hash = meshes[0]
            hash_checks.append(dict(source_object=source, frame=frame, input_vertex_sha256=hashes[0], t0_mesh_sha256=initial_hash, passed=True))
    failures = []
    for row in rows:
        ref = references[row['object']]
        reasons = method_failures(row)
        row.update(reference_input_valid=ref['valid'], method_contract_failure_reasons=reasons,
                   method_contract_pass=not reasons, joint_valid=ref['valid'] and not reasons)
        for arm in ['freeze', 'transport']:
            other = keyed[row['source_object'], row['frame'], arm]
            row['mesh_identical_to_' + arm] = bool(row.get('mesh_sha256')) and row.get('mesh_sha256') == other.get('mesh_sha256')
        if not row['joint_valid']:
            failures.append(dict(object=row['object'], frame=row['frame'], arm=row['state'], execution_origin=row['execution_origin'],
                                 failure_class='METHOD_CONTRACT_FAILED' if ref['valid'] else 'REFERENCE_INPUT_INVALID',
                                 method_failure_reasons=reasons, reference_checks=ref if not ref['valid'] else None,
                                 status=row['status'], error=row.get('error'), mesh_sha256=row.get('mesh_sha256')))
    fields = ['execution_origin', 'status', 'joint_valid', 'method_contract_failure_reasons', 'rms_pct_bboxdiag',
              'p95_pct_bboxdiag', 'rms_absolute', 'p95_absolute', 'vertices', 'faces', 'beta0', 'beta1', 'beta2',
              'strict_all_vertex_links_simple_cycles', 'strict_noncycle_vertex_count', 'nonmanifold_edges',
              'nonmanifold_vertices', 'exact_intersection_pairs', 'exact_degenerate_faces', 'duplicate_faces',
              'graph_capacity_stats', 'arccos_clipped_count', 'mesh_sha256', 'surface_input_faces_sha256',
              'mesh_identical_to_freeze', 'mesh_identical_to_transport']
    wide = []
    for source in SOURCES:
        for frame in range(8):
            base = keyed[source, frame, 'freeze']
            item = dict(source_object=source, frame=frame, t=base['t'], object=base['object'],
                        reference_input_valid=references[base['object']]['valid'],
                        reference_sha256=references[base['object']]['parent'].get('reference_sha256'),
                        input_vertex_sha256=base['input_vertex_sha256'])
            for arm in ARMS:
                for field in fields:
                    item[arm + '__' + field] = keyed[source, frame, arm].get(field)
            four = [keyed[source, frame, a] for a in ['freeze', 'q_only', 'n_only', 'transport']]
            item['factorial_rms_interaction_pct_bboxdiag'] = (four[3]['rms_pct_bboxdiag'] - four[1]['rms_pct_bboxdiag'] -
                four[2]['rms_pct_bboxdiag'] + four[0]['rms_pct_bboxdiag']) if all(r['joint_valid'] and r.get('rms_pct_bboxdiag') is not None for r in four) else None
            wide.append(item)
    endpoints = [r for r in wide if r['frame'] == 7]
    focus = [keyed['thin_u', 6, arm] for arm in ARMS]
    summary = dict(schema='CONTROLLED_CLAY_Q_N_ABLATION_V1', new_extractions=48, reused_parent_observations=72,
                   combined_observations=120, parent_run=str(parent), parent_ledger_sha256=sha(parent / 'LEDGER.json'),
                   current_ledger_sha256=sha(root / 'LEDGER.json'), protocol_sha256=digest,
                   object_count=3, frames_per_object=8, network_training_updates=0,
                   by_arm={a: stats([r for r in rows if r['state'] == a]) for a in ARMS},
                   by_source_arm={s: {a: stats([r for r in rows if r['state'] == a and r['source_object'] == s]) for a in ARMS} for s in SOURCES},
                   endpoints=endpoints, thin_u_frame6=[{k: r.get(k) for k in ['object', 'frame', 't', 'state'] + fields} for r in focus],
                   mesh_identity_counts={a: {target: sum(r['mesh_identical_to_' + target] for r in rows if r['state'] == a)
                                             for target in ['freeze', 'transport']} for a in ARMS},
                   mesh_identity_denominator_per_arm=24, source_reference_checks=references, input_checks=hash_checks,
                   reference_frames=dict(total=24, valid=sum(r['valid'] for r in references.values())),
                   note='72 parent observations are references to prior execution, not new experiments. Joint validity is distinct from consistency PASS.',
                   limitations=['Three known objects and one path per object; frames are not independent subjects.',
                                'Q-only transports full A/g/c and recomputes its eigenvalues; it is not an A-only ablation.',
                                'N-only changes the inverse-transpose normal direction while preserving its original mean magnitude.',
                                'Interaction is descriptive per-frame finite difference, not statistical significance.',
                                'Refit has access to current reference geometry; no neural training or online feedback was tested.'])
    (root / 'reports').mkdir(exist_ok=True); (root / 'figures').mkdir(exist_ok=True)
    write_json(root / 'COMBINED_LEDGER.json', rows); write_csv(root / 'COMBINED_LEDGER.csv', rows)
    write_csv(root / 'reports/FACTORIAL_FRAME_TABLE.csv', wide)
    write_json(root / 'SUMMARY_ABLATION.json', summary)
    write_json(root / 'FAILURES.json', dict(reference_failures={k: r for k, r in references.items() if not r['valid']}, output_failures=failures))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for axis, source in zip(axes, SOURCES):
        for arm in ARMS:
            group = [keyed[source, frame, arm] for frame in range(8)]
            axis.plot([r['t'] for r in group], [r.get('rms_pct_bboxdiag', np.nan) for r in group], color=COLORS[arm], marker='o', markersize=3, label=arm)
            bad = [r for r in group if not r['joint_valid'] and r.get('rms_pct_bboxdiag') is not None]
            axis.scatter([r['t'] for r in bad], [r['rms_pct_bboxdiag'] for r in bad], marker='x', color='#b00000', s=70, zorder=5)
        axis.set_title(source); axis.set_xlabel('Frozen path parameter t'); axis.set_ylabel('RMS (% current reference bbox diagonal)'); axis.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.suptitle('Q/n ablation: 48 new extractions + 72 explicitly reused observations; red x = invalid')
    fig.tight_layout(); fig.savefig(root / 'figures/ablation_rms_paths.png', dpi=150); plt.close(fig)
    lines = ['# 局部Q与法向更新的最小消融', '',
             '**本次实际新增48次提取；旧72条记录从父实验引用复用，组合表120行不是120次新增运行。**', '',
             '同一批3对象、8帧路径，freeze / q_only / n_only / transport / refit每臂24个槽位。所有失败保留。', '',
             '|实验臂|联合有效 / 24|参考失败槽位|合格参考上的方法失败|与freeze mesh字节相同 / 24|与transport相同 / 24|', '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        s = summary['by_arm'][arm]; h = summary['mesh_identity_counts'][arm]
        lines.append(f"|{arm}|{s['joint_valid']}/24|{s['reference_invalid_slots']}|{s['method_failures_on_valid_reference']}|{h['freeze']}/24|{h['transport']}/24|")
    lines += ['', '联合有效沿用完整几何与实际图容量合同，不增加误差阈值。参考失败单列；一致性检查PASS不代表方法有效。相同256输入元素不代表实际输出面数相同。', '',
              '## 薄U的frame 6', '', '|实验臂|Betti|严格顶点link|联合有效|RMS %|实际面数|mesh SHA256|', '|---|---|---|---|---:|---:|---|']
    for r in focus:
        lines.append(f"|{r['state']}|{[r.get('beta'+str(i)) for i in range(3)]}|{r.get('strict_all_vertex_links_simple_cycles')}|{r['joint_valid']}|{r.get('rms_pct_bboxdiag')}|{r.get('faces')}|{r.get('mesh_sha256')}|")
    lines += ['', '## 末帧', '', '|对象|实验臂|联合有效|RMS %|p95 %|实际面数|', '|---|---|---|---:|---:|---:|']
    for source in SOURCES:
        for arm in ARMS:
            r = keyed[source, 7, arm]
            lines.append(f"|{source}|{arm}|{r['joint_valid']}|{r.get('rms_pct_bboxdiag')}|{r.get('p95_pct_bboxdiag')}|{r.get('faces')}|")
    lines += ['', '![五路误差路径](figures/ablation_rms_paths.png)', '',
              'q_only传输完整局部A/g/c，并随Q重新计算eigs；不能简称为仅A矩阵消融。n_only只改变逆转置法向方向并保持原平均法向模长。transport联合执行两者，refit仍是需要当前参考几何的诊断参照。', '',
              '24行宽表在四个因子臂均联合有效且误差可测时报告描述性交互：RMS(transport) − RMS(q_only) − RMS(n_only) + RMS(freeze)。其它帧留空；不能通过删去无效帧宣布普遍收益。帧之间相关，不做显著性推断。', '',
              '结论只涉及这些固定路径上Q单独、n单独及联合更新的影响，不推广为类别泛化、可学习性、新反馈机制或连续时间保证。未训练网络，也不进行速度比较。', '',
              '`COMBINED_LEDGER.json/.csv`保留执行来源与所有失败；`reports/FACTORIAL_FRAME_TABLE.csv`为24行五路宽表；`SUMMARY_ABLATION.json`含逐对象/逐臂、末帧、薄U frame6及hash对照；`FAILURES.json`区分参考失败与方法失败。', '']
    (root / 'REPORT_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(new_extractions=48, reused_parent_observations=72, by_arm=summary['by_arm']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', required=True, type=Path)
    summarize(parser.parse_args().root)
