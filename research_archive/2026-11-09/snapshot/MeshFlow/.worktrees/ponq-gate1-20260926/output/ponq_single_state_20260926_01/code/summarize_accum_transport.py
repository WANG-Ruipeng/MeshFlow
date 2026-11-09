"""Report 45 accumulated runs and 72 reused same-position controls."""
from pathlib import Path
import argparse
import collections
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_clay import read, sha, write_json, write_csv, reference_check, method_failures, SOURCES

INC = 'incremental_lsq16'
CONTROLS = ('freeze', 'q_only', 'lsq16')


def stats(rows):
    return dict(requested_slots=len(rows), joint_valid=sum(r['joint_valid'] for r in rows),
                reference_invalid_slots=sum(not r['reference_input_valid'] for r in rows),
                method_failures_on_valid_reference=sum(r['reference_input_valid'] and not r['method_contract_pass'] for r in rows),
                status_counts=dict(collections.Counter(r['status'] for r in rows)))


def summarize(root):
    root = Path(root).resolve(); protocol = read(root/'FROZEN_PROTOCOL.json'); config = protocol['config']
    parent, factorial, motion = (Path(config[k]).resolve() for k in ['parent_run', 'factorial_run', 'motion_run'])
    parent_protocol = read(parent/'FROZEN_PROTOCOL.json')
    assert sha(root/'FROZEN_PROTOCOL.json') == read(root/'FROZEN_PROTOCOL_SHA256.json')['sha256']
    new = read(root/'LEDGER.json'); assert len(new) == 45
    rows = []
    for run, state in [(parent, 'freeze'), (factorial, 'q_only'), (motion, 'lsq16')]:
        selected = [r for r in read(run/'LEDGER.json') if r['state'] == state]; assert len(selected) == 24
        for original in selected:
            row = dict(original); row.update(execution_origin='reused_control', measurement_run=str(run), direction='control', source_frame_object=row['object']); rows.append(row)
    for original in new:
        row = dict(original); row.update(execution_origin='new_incremental', measurement_run=str(root)); rows.append(row)
    assert len(rows) == 117 and {r['state'] for r in new} == {INC}
    expected = {(s,d,f) for s in SOURCES for d, fs in [('forward',range(8)),('reverse',range(6,-1,-1))] for f in fs}
    inc = {(r['source_object'],r['direction'],r['frame']):r for r in rows if r['state']==INC}
    assert len(inc)==45 and set(inc)==expected
    controls = {(r['source_object'],r['frame'],r['state']):r for r in rows if r['state'] in CONTROLS}
    references = {}
    for name, spec in parent_protocol['objects'].items():
        old, current = reference_check(parent,name,spec), reference_check(root,name,spec)
        assert old.get('reference_sha256') and old.get('reference_sha256') == current.get('reference_sha256'), 'Reference changed: '+name
        references[name] = dict(valid=old['valid'] and current['valid'], parent=old, current=current)
    failures = []
    for row in rows:
        ref = references[row['source_frame_object']]; reasons = method_failures(row)
        row.update(reference_input_valid=ref['valid'], method_contract_failure_reasons=reasons,
                   method_contract_pass=not reasons,joint_valid=ref['valid'] and not reasons)
        if not row['joint_valid']:
            failures.append(dict(object=row['object'],source_object=row['source_object'],frame=row['frame'],direction=row['direction'],state=row['state'],
                                 execution_origin=row['execution_origin'],failure_class='METHOD_CONTRACT_FAILED' if ref['valid'] else 'REFERENCE_INPUT_INVALID',
                                 reasons=reasons,status=row['status'],error=row.get('error')))
    fields=['status','joint_valid','rms_pct_bboxdiag','p95_pct_bboxdiag','rms_absolute','p95_absolute','faces','vertices','beta0','beta1','beta2',
            'strict_all_vertex_links_simple_cycles','strict_noncycle_vertex_count','nonmanifold_edges','nonmanifold_vertices',
            'exact_intersection_pairs','exact_degenerate_faces','graph_capacity_stats','mesh_sha256','surface_input_faces_sha256','motion_estimate_stats']
    wide=[]
    for row in sorted([r for r in rows if r['state']==INC],key=lambda r:(SOURCES.index(r['source_object']),r['step_index'])):
        item={k:row[k] for k in ['source_object','direction','frame','step_index','t','object','source_frame_object']}
        group=[row]+[controls[row['source_object'],row['frame'],a] for a in CONTROLS]
        assert len({r.get('input_vertex_sha256') for r in group})==1 and row.get('input_vertex_sha256'), 'Shared positions differ'
        if row['direction']=='forward' and row['frame']==0:
            assert len({r.get('mesh_sha256') for r in group})==1 and row.get('mesh_sha256'), 'Canonical initial meshes differ'
        for r in group:
            for field in fields:item[r['state']+'__'+field]=r.get(field)
        item['incremental_mesh_identical_to_direct_lsq16']=bool(row.get('mesh_sha256')) and row.get('mesh_sha256')==controls[row['source_object'],row['frame'],'lsq16'].get('mesh_sha256')
        wide.append(item)
    f=controls['thin_u',7,'freeze']; d=controls['thin_u',7,'lsq16']; a=inc['thin_u','forward',7]
    gain=dict(definition='(RMS_freeze7 - RMS_incremental_forward7) / (RMS_freeze7 - RMS_direct_lsq16_7)',threshold=.90,
              freeze_rms=f.get('rms_pct_bboxdiag'),direct_lsq16_rms=d.get('rms_pct_bboxdiag'),incremental_rms=a.get('rms_pct_bboxdiag'),
              all_required_outputs_joint_valid=all(r['joint_valid'] for r in [f,d,a]),evaluated=False,ratio=None,meets_90_percent=False)
    if gain['all_required_outputs_joint_valid'] and all(r.get('rms_pct_bboxdiag') is not None for r in [f,d,a]):
        denominator=f['rms_pct_bboxdiag']-d['rms_pct_bboxdiag'];gain['direct_net_gain']=denominator
        gain['incremental_net_gain']=f['rms_pct_bboxdiag']-a['rms_pct_bboxdiag']
        if denominator>0:
            gain.update(evaluated=True,ratio=gain['incremental_net_gain']/denominator)
            gain['meets_90_percent']=gain['ratio']>=.90
        else:gain['not_evaluated_reason']='Direct control has no positive net RMS gain'
    else:gain['not_evaluated_reason']='Required reference/output validity or RMS missing'
    verification=read(root/'audit/ACCUM_VERIFY.json')
    summary=dict(schema='ACCUMULATED_Q_TRANSPORT_DIAGNOSTIC_V1',new_extractions=45,reused_observations=72,combined_observations=117,
                 comparison_table_rows=45,comparison_note='Repeated control references in the 45-row comparison table are not additional observations.',
                 parent_run=str(parent),factorial_run=str(factorial),motion_run=str(motion),network_training_updates=0,
                 source_ledger_hashes={str(r/'LEDGER.json'):sha(r/'LEDGER.json') for r in [parent,factorial,motion,root]},
                 incremental=stats([r for r in rows if r['state']==INC]),
                 by_direction={direction:stats([r for r in rows if r['state']==INC and r['direction']==direction]) for direction in ['forward','reverse']},
                 by_source_direction={s:{direction:stats([r for r in rows if r['state']==INC and r['source_object']==s and r['direction']==direction]) for direction in ['forward','reverse']} for s in SOURCES},
                 controls={state:stats([r for r in rows if r['state']==state]) for state in CONTROLS},
                 thin_u_forward7_net_gain_retention=gain,
                 thin_u_frames6_7=[r for r in wide if r['source_object']=='thin_u' and r['frame'] in [6,7]],
                 forward_endpoints=[r for r in wide if r['direction']=='forward' and r['frame']==7],
                 return_endpoints=[r for r in wide if r['direction']=='reverse' and r['frame']==0],
                 return_parameter_diagnostics=verification['return_diagnostics'],verification_passed=verification['passed'],
                 verification_count=verification['count'],reference_checks=references,
                 incremental_direct_mesh_match_count=sum(r['incremental_mesh_identical_to_direct_lsq16'] for r in wide),mesh_match_denominator=45,
                 limitations=['Known paired positions, fixed initial neighbor IDs/weights/scales; current source X changes at every increment.',
                              'Rank-deficient identity fallback preserves a translated frozen local quadric; it does not reconstruct missing motion.',
                              'Return motion is independently re-estimated; inverse saved-forward-H control is only algebraic and uses no additional extraction.',
                              'Three development objects, correlated frames, no training, feedback mechanism or FM deployment success claim.',
                              '90% criterion concerns only thin-U forward frame7 net RMS gain; validity and the other frames remain separate.'])
    (root/'reports').mkdir(exist_ok=True);(root/'figures').mkdir(exist_ok=True)
    write_json(root/'COMBINED_LEDGER.json',rows);write_csv(root/'COMBINED_LEDGER.csv',rows)
    write_csv(root/'reports/ACCUM_FRAME_TABLE.csv',wide);write_json(root/'SUMMARY_ACCUM.json',summary)
    write_json(root/'FAILURES.json',dict(reference_failures={k:v for k,v in references.items() if not v['valid']},output_failures=failures))
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,source in zip(axes,SOURCES):
        series=[(a,[controls[source,f,a] for f in range(8)],color,'-') for a,color in [('freeze','#2875ac'),('q_only','#9467bd'),('lsq16','#388b4d')]]
        series += [('incremental forward',[inc[source,'forward',f] for f in range(8)],'#dc7525','-'),('incremental reverse',[inc[source,'reverse',f] for f in range(6,-1,-1)],'#b84b71','--')]
        for label,group,color,style in series:
            ax.plot([r['t'] for r in group],[r.get('rms_pct_bboxdiag',np.nan) for r in group],style,color=color,marker='o',markersize=3,label=label)
            bad=[r for r in group if not r['joint_valid'] and r.get('rms_pct_bboxdiag') is not None]
            ax.scatter([r['t'] for r in bad],[r['rms_pct_bboxdiag'] for r in bad],marker='x',color='#b00000',s=70,zorder=5)
        ax.set_title(source);ax.set_xlabel('Frozen path t');ax.set_ylabel('RMS (% current reference bbox diagonal)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('45 new incremental observations + 72 reused controls; red x = joint invalid')
    fig.tight_layout();fig.savefig(root/'figures/accum_rms_paths.png',dpi=150);plt.close(fig)
    lines=['# 全Q逐步传输与返程诊断','','**新增45次提取：每对象正向8帧、返程7帧。72条旧对照明确复用，组合117行不是117次新增运行。**','',
           '|新执行范围|联合有效 / 完整分母|参考失败|方法失败（有效参考）|','|---|---:|---:|---:|']
    for direction in ['forward','reverse']:
        s=summary['by_direction'][direction];lines.append(f"|{direction}|{s['joint_valid']}/{s['requested_slots']}|{s['reference_invalid_slots']}|{s['method_failures_on_valid_reference']}|")
    lines += ['',f"独立存档核验：{'PASS' if verification['passed'] else 'FAIL'}，{verification['count']}项；这是代数/记录核验，不替代几何有效性。",'',
              '## 薄U末帧净增益保留','','定义：(freeze末帧RMS − incremental正向末帧RMS) / (freeze末帧RMS − direct lsq16末帧RMS)。仅在三个输出均联合有效且分母正时评价，使用未四舍五入账本。', '',
              f"本次 ratio={gain['ratio']}；满足≥0.90：{gain['meets_90_percent']}；可评价：{gain['evaluated']}。这不替代frame6或返程的质量审核。",'',
              '|对象|方向|frame|增量有效|增量RMS %|direct RMS %|增量面数|同direct mesh字节|','|---|---|---:|---|---:|---:|---:|---|']
    for r in wide:
        if (r['source_object']=='thin_u' and r['frame'] in [6,7]) or (r['direction']=='reverse' and r['frame']==0):
            lines.append(f"|{r['source_object']}|{r['direction']}|{r['frame']}|{r[INC+'__joint_valid']}|{r[INC+'__rms_pct_bboxdiag']}|{r['lsq16__rms_pct_bboxdiag']}|{r[INC+'__faces']}|{r['incremental_mesh_identical_to_direct_lsq16']}|")
    lines += ['','![正向与返程](figures/accum_rms_paths.png)','','## 返程恢复与解释边界','',
              '|对象|返程顶点逐位恢复|返程mesh逐位恢复|全Q相对漂移|谱归一Q相对漂移|逆保存正向H控制相对残差|','|---|---|---|---:|---:|---:|']
    for r in verification['return_diagnostics']:
        lines.append(f"|{r['source_object']}|{r.get('actual_return_vertex_byte_equal')}|{r.get('actual_return_mesh_identical_to_initial')}|{r.get('actual_return_Q',{}).get('relative_frobenius')}|{r.get('actual_return_spectral_Q',{}).get('relative_frobenius')}|{r['inverse_saved_forward_control'].get('relative_frobenius')}|")
    lines += ['','初始邻居ID、权重、局部尺度固定，但每步回归的源X取当前帧。每步传输完整world Q并重新计算eigs，n/p/v与同位置freeze一致；不做SVD顶点重定位。秩亏和回退全部保留。', '',
              '重新估计返程与逆序使用保存的正向H是不同问题。后者只是无新增提取的代数控制；实际返程的Q漂移、谱归一Q差、mesh字节与规范连接/表面差分别记录，mesh恢复不能替代参数恢复。', '',
              'audit/ACCUM_VERIFY.json包含独立lstsq、逐步/累计H全Q递推、与同位置direct lsq16的局部尺度Q/谱归一Q、语义ID配对图差以及mesh对照。观察到差异不自动等于核验失败。', '',
              '三个已有对象、已知配对位置和有限相关帧不能证明生成部署、FM成功或连续轨迹稳定；没有新增训练或反馈修正，不做统计显著性和速度结论。', '',
              '`COMBINED_LEDGER.json/.csv`保留117条唯一观察的来源；45行`reports/ACCUM_FRAME_TABLE.csv`对每个增量结果引用旧控制，同一旧控制的多次展示不计作新运行。', '']
    (root/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(new_extractions=45,reused_observations=72,by_direction=summary['by_direction'],thin_u_gain=gain),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);summarize(p.parse_args().root)
