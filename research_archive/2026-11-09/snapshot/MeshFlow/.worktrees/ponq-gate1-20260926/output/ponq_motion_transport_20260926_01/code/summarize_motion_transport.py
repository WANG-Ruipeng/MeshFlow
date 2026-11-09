"""Report 48 estimated-motion runs plus 48 explicitly reused controls."""
from pathlib import Path
import argparse
import collections
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_clay import read, sha, write_json, write_csv, reference_check, method_failures, SOURCES

ARMS = ('freeze', 'q_only', 'lsq16', 'ridge16')
COLORS = dict(freeze='#2875ac', q_only='#9467bd', lsq16='#df7023', ridge16='#389157')


def stats(rows):
    return dict(requested_slots=len(rows), joint_valid=sum(r['joint_valid'] for r in rows),
                reference_invalid_slots=sum(not r['reference_input_valid'] for r in rows),
                method_failures_on_valid_reference=sum(r['reference_input_valid'] and not r['method_contract_pass'] for r in rows),
                status_counts=dict(collections.Counter(r['status'] for r in rows)),
                new_executions=sum(r['execution_origin'] == 'new_motion_estimation' for r in rows))


def saved_motion_diagnostics(directory):
    """Read named stored diagnostics; never invent a different numerical rank rule."""
    path = directory / 'estimated_motion.npz'
    if not path.is_file():
        return dict(available=False)
    result = dict(available=True, path=str(path), sha256=sha(path), arrays={})
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = archive[key]
            entry = dict(shape=list(value.shape), dtype=str(value.dtype))
            if 'rank' in key.lower() and value.ndim <= 1 and np.issubdtype(value.dtype, np.integer):
                entry['histogram'] = {str(int(k)): int(n) for k, n in zip(*np.unique(value, return_counts=True))}
            if 'fallback' in key.lower() and value.ndim <= 1 and (value.dtype == bool or np.issubdtype(value.dtype, np.integer)):
                if np.all((value == 0) | (value == 1)):
                    entry.update(true_count=int(np.count_nonzero(value)), element_count=int(value.size))
            if key in ['J_raw', 'J_used']:
                entry['nonfinite_entry_count'] = int(np.sum(~np.isfinite(value)))
            result['arrays'][key] = entry
    return result


def summarize(root):
    root = Path(root).resolve()
    protocol = read(root / 'FROZEN_PROTOCOL.json')
    digest = sha(root / 'FROZEN_PROTOCOL.json')
    assert digest == read(root / 'FROZEN_PROTOCOL_SHA256.json')['sha256'], 'Protocol hash mismatch'
    parent = Path(protocol['config']['parent_run']).resolve()
    factorial = Path(protocol['config']['factorial_run']).resolve()
    parent_protocol = read(parent / 'FROZEN_PROTOCOL.json')
    fresh = read(root / 'LEDGER.json')
    frozen = [r for r in read(parent / 'LEDGER.json') if r['state'] == 'freeze']
    analytic = [r for r in read(factorial / 'LEDGER.json') if r['state'] == 'q_only']
    assert len(fresh) == 48 and len(frozen) == len(analytic) == 24, 'Expected 48 new + 24 frozen + 24 analytic controls'
    assert {r['state'] for r in fresh} == {'lsq16', 'ridge16'}, 'Unexpected new methods'
    rows = []
    for source, origin, run in [(frozen, 'reused_clay_freeze', parent), (analytic, 'reused_factorial_analytic_J', factorial),
                                (fresh, 'new_motion_estimation', root)]:
        for original in source:
            row = dict(original); row.update(execution_origin=origin, measurement_run=str(run)); rows.append(row)
    keyed = {(r['source_object'], int(r['frame']), r['state']): r for r in rows}
    assert len(keyed) == 96 and set(keyed) == {(s, f, a) for s in SOURCES for f in range(8) for a in ARMS}, 'Missing/duplicate slots'
    references = {}
    for name, spec in parent_protocol['objects'].items():
        checks = {label: reference_check(run, name, spec) for label, run in [('parent', parent), ('factorial', factorial), ('current', root)]}
        hashes = [r.get('reference_sha256') for r in checks.values()]
        assert hashes[0] and len(set(hashes)) == 1, 'Reference changed: ' + name
        references[name] = dict(valid=all(r['valid'] for r in checks.values()), checks=checks)
    agreement = []
    for source in SOURCES:
        for frame in range(8):
            group = [keyed[source, frame, arm] for arm in ARMS]
            hashes = [r.get('input_vertex_sha256') for r in group]
            assert hashes[0] and len(set(hashes)) == 1, f'Paired input geometry changed: {source}/{frame}'
            assert len({r['t'] for r in group}) == 1, 'Different path parameters'
            mesh_hashes = [r.get('mesh_sha256') for r in group]
            if frame == 0:
                assert mesh_hashes[0] and len(set(mesh_hashes)) == 1, 't0 canonical mesh mismatch: ' + source
            agreement.append(dict(source_object=source, frame=frame, input_vertex_sha256=hashes[0],
                                  initial_mesh_sha256=mesh_hashes[0] if frame == 0 else None, passed=True))
    failures, motion_records = [], []
    for row in rows:
        ref = references[row['object']]; reasons = method_failures(row)
        row.update(reference_input_valid=ref['valid'], method_contract_failure_reasons=reasons,
                   method_contract_pass=not reasons, joint_valid=ref['valid'] and not reasons)
        for arm in ['freeze', 'q_only']:
            control = keyed[row['source_object'], row['frame'], arm]
            row['mesh_identical_to_' + arm] = bool(row.get('mesh_sha256')) and row.get('mesh_sha256') == control.get('mesh_sha256')
        if row['execution_origin'] == 'new_motion_estimation':
            directory = root / 'runs' / (row['object'] + '__' + row['state'])
            motion_records.append(dict(object=row['object'], frame=row['frame'], state=row['state'],
                                       motion_estimate_stats=row.get('motion_estimate_stats'), saved_arrays=saved_motion_diagnostics(directory)))
        if not row['joint_valid']:
            failures.append(dict(object=row['object'], frame=row['frame'], state=row['state'], execution_origin=row['execution_origin'],
                                 failure_class='METHOD_CONTRACT_FAILED' if ref['valid'] else 'REFERENCE_INPUT_INVALID',
                                 method_failure_reasons=reasons, status=row['status'], error=row.get('error'),
                                 motion_estimate_stats=row.get('motion_estimate_stats'), mesh_sha256=row.get('mesh_sha256')))
    fields = ['execution_origin', 'status', 'joint_valid', 'method_contract_failure_reasons', 'rms_pct_bboxdiag', 'p95_pct_bboxdiag',
              'rms_absolute', 'p95_absolute', 'vertices', 'faces', 'beta0', 'beta1', 'beta2', 'strict_all_vertex_links_simple_cycles',
              'strict_noncycle_vertex_count', 'nonmanifold_edges', 'nonmanifold_vertices', 'exact_intersection_pairs',
              'exact_degenerate_faces', 'duplicate_faces', 'graph_capacity_stats', 'mesh_sha256', 'surface_input_faces_sha256',
              'mesh_identical_to_freeze', 'mesh_identical_to_q_only', 'motion_estimate_stats']
    wide = []
    for source in SOURCES:
        for frame in range(8):
            base = keyed[source, frame, 'freeze']
            item = dict(source_object=source, frame=frame, t=base['t'], object=base['object'],
                        reference_input_valid=references[base['object']]['valid'], input_vertex_sha256=base['input_vertex_sha256'])
            for arm in ARMS:
                for field in fields:
                    item[arm + '__' + field] = keyed[source, frame, arm].get(field)
            wide.append(item)
    neighborhoods = []
    for source in SOURCES:
        directory = root / 'motion_inputs' / source
        sp, npz = directory / 'stats.json', directory / 'neighborhood.npz'
        neighborhoods.append(dict(source_object=source, stats=read(sp) if sp.is_file() else None,
                                  stats_sha256=sha(sp) if sp.is_file() else None,
                                  neighborhood_sha256=sha(npz) if npz.is_file() else None))
    diagnostic_ranges = {}
    for arm in ['lsq16', 'ridge16']:
        group = [r.get('motion_estimate_stats') or {} for r in rows if r['state'] == arm]
        keys = sorted(set().union(*(x.keys() for x in group)))
        diagnostic_ranges[arm] = {}
        for key in keys:
            values = [x[key] for x in group if isinstance(x.get(key), (int, float)) and not isinstance(x.get(key), bool) and np.isfinite(x[key])]
            if values:
                diagnostic_ranges[arm][key] = dict(frames_with_finite_value=len(values), minimum=min(values), maximum=max(values))
    summary = dict(schema='LOCAL_MOTION_Q_TRANSPORT_DIAGNOSTIC_V1', new_extractions=48, reused_observations=48,
                   combined_observations=96, parent_run=str(parent), factorial_run=str(factorial), protocol_sha256=digest,
                   source_ledgers={str(run / 'LEDGER.json'): sha(run / 'LEDGER.json') for run in [parent, factorial, root]},
                   object_count=3, frames_per_object=8, fixed_k=16, fixed_ridge=1e-3, tuned_on_results=False, network_training_updates=0,
                   by_arm={a: stats([r for r in rows if r['state'] == a]) for a in ARMS},
                   by_source_arm={s: {a: stats([r for r in rows if r['source_object'] == s and r['state'] == a]) for a in ARMS} for s in SOURCES},
                   endpoints=[r for r in wide if r['frame'] == 7], thin_u_frames_6_7=[r for r in wide if r['source_object'] == 'thin_u' and r['frame'] in [6, 7]],
                   mesh_identity_counts={a: {target: sum(r['mesh_identical_to_' + target] for r in rows if r['state'] == a) for target in ['freeze', 'q_only']} for a in ARMS},
                   mesh_identity_denominator_per_arm=24, reference_checks=references, input_agreement_checks=agreement,
                   neighborhood_records=neighborhoods, motion_estimate_records=motion_records, motion_statistic_ranges=diagnostic_ranges,
                   limitations=['48 controls are reused observations, not new runs.', 'Only three known objects and one path per object; correlated frames.',
                                'Paired initial/current point positions are supplied; this is not a generated-deployment test.',
                                'N/p/v stay fixed to freeze controls; full local Q(A,g,c) and associated eigenvalues change.',
                                'Identity fallback is a recorded return to frozen local Q, not inferred missing deformation.',
                                'Ridge stabilizes rank-deficient fits but cannot recover unobserved motion information.',
                                'k=16 and ridge=1e-3 are frozen, not tuned; no training, feedback mechanism or statistical significance claim.'])
    (root / 'reports').mkdir(exist_ok=True); (root / 'figures').mkdir(exist_ok=True)
    write_json(root / 'COMBINED_LEDGER.json', rows); write_csv(root / 'COMBINED_LEDGER.csv', rows)
    write_csv(root / 'reports/MOTION_FRAME_TABLE.csv', wide); write_json(root / 'SUMMARY_MOTION.json', summary)
    write_json(root / 'FAILURES.json', dict(reference_failures={k: v for k, v in references.items() if not v['valid']}, output_failures=failures))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for axis, source in zip(axes, SOURCES):
        for arm in ARMS:
            group = [keyed[source, frame, arm] for frame in range(8)]
            axis.plot([r['t'] for r in group], [r.get('rms_pct_bboxdiag', np.nan) for r in group], color=COLORS[arm], marker='o', markersize=3, label=arm)
            bad = [r for r in group if not r['joint_valid'] and r.get('rms_pct_bboxdiag') is not None]
            axis.scatter([r['t'] for r in bad], [r['rms_pct_bboxdiag'] for r in bad], marker='x', color='#b00000', s=70, zorder=5)
        axis.set_title(source); axis.set_xlabel('Frozen path parameter t'); axis.set_ylabel('RMS (% current reference bbox diagonal)'); axis.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.suptitle('Estimated local motion: 48 new + 48 reused; q_only uses analytic J; red x = invalid')
    fig.tight_layout(); fig.savefig(root / 'figures/motion_rms_paths.png', dpi=150); plt.close(fig)
    lines = ['# 从配对点运动估计局部Q传输', '', '**新增48次提取，引用复用48条旧对照；96行组合账本不是96次新运行。**', '',
             'freeze来自旧clay实验，q_only来自旧消融并使用解析Jacobian。lsq16与ridge16是本轮新执行，各有3对象×8帧=24个槽位。固定k=16、ridge=1e−3，没有按结果调参。', '',
             '|实验臂|联合有效 / 24|参考失败槽位|方法失败（参考有效）|同解析q_only mesh字节 / 24|同freeze / 24|', '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        s = summary['by_arm'][arm]; h = summary['mesh_identity_counts'][arm]
        lines.append(f"|{arm}|{s['joint_valid']}/24|{s['reference_invalid_slots']}|{s['method_failures_on_valid_reference']}|{h['q_only']}/24|{h['freeze']}/24|")
    lines += ['', '联合有效沿用完整几何、严格link和实际聚合图容量合同，不新增RMS阈值；参考失败另列。一致性检查PASS不能覆盖几何失败。', '',
              '## 薄U关键帧与末帧', '', '|对象|frame|实验臂|有效|Betti|严格link|RMS %|p95 %|面数|同解析q_only|', '|---|---:|---|---|---|---|---:|---:|---:|---|']
    for source, frame in [('thin_u', 6), ('thin_u', 7), ('sphere', 7), ('bottle_local_1', 7)]:
        for arm in ARMS:
            r = keyed[source, frame, arm]
            lines.append(f"|{source}|{frame}|{arm}|{r['joint_valid']}|{[r.get('beta'+str(i)) for i in range(3)]}|{r.get('strict_all_vertex_links_simple_cycles')}|{r.get('rms_pct_bboxdiag')}|{r.get('p95_pct_bboxdiag')}|{r.get('faces')}|{r['mesh_identical_to_q_only']}|")
    lines += ['', '![四路RMS](figures/motion_rms_paths.png)', '', '## 病态、秩亏与回退', '',
              '所有motion_estimate_stats原样进入组合账本。SUMMARY_MOTION保存48帧估计统计、明确命名rank数组的直方图、fallback布尔数组计数及固定邻域统计；没有删掉病态或回退样本。下表是各统计字段在24帧内的范围，不把各帧当作独立对象。', '',
              '|方法|标量诊断字段|有限记录数 / 24|最小值|最大值|', '|---|---|---:|---:|---:|']
    for arm, values in diagnostic_ranges.items():
        for key, v in values.items():
            if any(token in key.lower() for token in ['rank', 'fallback', 'condition', 'singular', 'invalid']):
                lines.append(f"|{arm}|{key}|{v['frames_with_finite_value']}/24|{v['minimum']}|{v['maximum']}|")
    lines += ['', 'ridge在秩亏时提供正则化解，不表示缺失的运动信息已经恢复。回退到I表示回到冻结局部Q；失败和回退仍计入完整分母。', '',
              '## 解释边界', '',
              '各方法使用同帧相同的v、p与法向，变化的是通过估计J传输的完整局部Q(A/g/c)及对应eigs；不是单独A矩阵实验。t=0使用共同初态，四臂mesh hash一致。实际输出面数可不同，完整记录保留。', '',
              '本轮直接给定初态与当前点位置的对应关系，因此检验的是已知配对运动下的局部估计，不是真正生成部署。即使接近解析q_only，也不能据此宣称网络学会了稳定生成。三个已有对象和相关帧不支持总体泛化、统计显著性或连续时间保证。未训练网络，未增加在线反馈，也不比较速度。', '',
              '`COMBINED_LEDGER.json/.csv`明确执行来源；`reports/MOTION_FRAME_TABLE.csv`为24行四路宽表；`SUMMARY_MOTION.json`保存末帧、薄U6/7、meshhash匹配及所有病态统计；`FAILURES.json`区分参考失败与方法失败。', '']
    (root / 'REPORT_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(new_extractions=48, reused_observations=48, by_arm=summary['by_arm']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    summarize(parser.parse_args().root)
