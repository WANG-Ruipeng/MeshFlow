"""Read-only aggregation, fixed-view figures and report for the geometry pilot.

No model, training, geometry distance, collision or topology routine is called.
The report consumes registered inputs and already recorded output metrics.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
import numpy as np

from .data import file_sha256
from .geometry_evaluation import load_evaluation

MODELS = ('T0', 'T_base', 'T_geo', 'T_graph')
COMPARISONS = (('T_base', 'T0'), ('T_geo', 'T_base'), ('T_graph', 'T_geo'))
ERROR_KEYS = ('free_rms_pct', 'pred_to_target_pct', 'target_to_pred_pct',
              'fixed_target_rms_pct', 'gamma_rms_pct', 'gamma_p95_pct')
HIGHER_BETTER = {'fixed_target_coverage', 'gamma_coverage'}
NEUTRAL_KEYS = {'free_area'}
STRICT_KEYS = (
    'CF_proper_affected_free_fraction', 'CF_proper_affected_free_count',
    'CF_proper_pair_count', 'CF_coplanar_pair_count', 'CF_unresolved_or_degenerate_pair_count',
    'FF_proper_pair_count', 'FF_coplanar_pair_count', 'FF_illegal_pair_count',
    'FF_unresolved_or_degenerate_pair_count', 'exact_C_free_shared_edge_count',
    'exact_C_free_shared_vertex_count', 'C_only_nonmanifold_vertex_count',
    'inherited_C_only_unmodified_link_count', 'C_inherited_link_anomalies_with_new_free_incidence_count',
    'new_free_related_nonmanifold_vertex_count', 'C_current_winding_conflict_count',
    'free_related_current_winding_conflict_count', 'exact_index_boundary_edge_count',
    'exact_index_overincident_edge_count', 'shared_edge_component_count')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha256(path), bytes=path.stat().st_size)


def _stats(values, higher_better=False):
    original = list(values)
    finite = np.asarray([v for v in original if v is not None and np.isfinite(v)], dtype=np.float64)
    return dict(n=len(finite), missing=len(original)-len(finite),
                mean=float(finite.mean()) if len(finite) else None,
                median=float(np.median(finite)) if len(finite) else None,
                minimum=float(finite.min()) if len(finite) else None,
                maximum=float(finite.max()) if len(finite) else None,
                worst=float(finite.min() if higher_better else finite.max()) if len(finite) else None)


def paired_statistics(rows, newer, reference, key, *, patch_group, parent=None, patch_id=None):
    """Difference is newer-reference; median difference is never median(A)-median(B)."""
    selected = [r for r in rows if r['patch_group'] == patch_group
                and (parent is None or r['parent'] == parent)
                and (patch_id is None or r['patch_id'] == patch_id)]
    def indexed(model):
        found = [r for r in selected if r['model'] == model]
        result = {(r['parent'], r['patch_id'], r['seed']): r for r in found}
        if len(found) != len(result):
            raise ValueError('Duplicate output in paired comparison.')
        return result
    a, b = indexed(newer), indexed(reference)
    if set(a) != set(b):
        raise ValueError('Comparison must have identical parent/patch/noise keys.')
    differences, pairs, missing = [], [], []
    for identity in sorted(a):
        va, vb = a[identity].get(key), b[identity].get(key)
        if va is None or vb is None or not (np.isfinite(va) and np.isfinite(vb)):
            missing.append(list(identity)); continue
        delta = float(va-vb); differences.append(delta)
        pairs.append(dict(parent=identity[0], patch_id=identity[1], seed=identity[2],
                          newer_value=va, reference_value=vb, difference=delta))
    higher = key in HIGHER_BETTER
    statistics=_stats(differences,higher)
    if key in NEUTRAL_KEYS:statistics['worst']=None
    return dict(newer=newer, reference=reference, metric=key, patch_group=patch_group,
        parent=parent, patch_id=patch_id, difference='newer minus reference',
        favorable_direction='NOT_DEFINED_DIAGNOSTIC_ONLY' if key in NEUTRAL_KEYS else 'positive' if higher else 'negative',
        statistics=statistics, paired_differences=pairs,
        improved=None if key in NEUTRAL_KEYS else sum(d > 0 if higher else d < 0 for d in differences),
        tied=sum(d == 0 for d in differences),
        worsened=None if key in NEUTRAL_KEYS else sum(d < 0 if higher else d > 0 for d in differences),
        increased=sum(d>0 for d in differences),decreased=sum(d<0 for d in differences),
        missing_pairs=missing,
        median_definition='Median of samplewise paired differences, not difference of output medians')


def _flatten(row, strict):
    m = row['metrics']; own = m['remainder']['own']; common = m['common_unknown_R12']['own']
    boundary = m['boundary']; quality = m['free_shape_quality']
    result = {key: row[key] for key in ('output_id','model','parent','patch_id','patch_group','seed','raw_path','raw_sha256')}
    result.update(free_rms_pct=own['symmetric_rms_bbox_pct'],
        pred_to_target_pct=own['predicted_to_target_rms_bbox_pct'],
        target_to_pred_pct=own['target_to_predicted_rms_bbox_pct'],
        fixed_target_rms_pct=common['rms_bbox_pct'],
        fixed_target_coverage=common['fraction_target_points_within_1pct_bbox'],
        gamma_rms_pct=boundary['rms_bbox_pct'], gamma_p95_pct=boundary['p95_bbox_pct'],
        gamma_coverage=boundary['coverage_fraction'],
        free_area=quality['total_area'], free_area_normalized_mean=quality['area_normalized']['mean'],
        narrow_q_gt10_fraction=quality['q_actual_gt10_fraction'],
        min_angle_lt5_fraction=quality['min_angle_lt5_fraction'],
        near_degenerate_count=quality['near_degenerate_area_count'],
        exact_zero_area_count=quality['exact_zero_area_count'],
        nan_count=m['whole']['quality']['nan_count'], inf_count=m['whole']['quality']['inf_count'],
        total_faces=m['total_faces'], free_faces=m['free_faces'],
        C_all51_bitwise=row['sampling']['known_all51_bitwise'],
        sampling_seconds=row['sampling']['seconds'],
        sampling_peak_cuda_bytes=row['sampling']['peak_cuda_allocated'],
        strict_audit_status='NOT_RUN' if strict is None else strict.get('status', 'UNKNOWN'))
    scalars = strict.get('scalars', {}) if strict else {}
    result.update({key: scalars.get(key) for key in STRICT_KEYS})
    return result


def summarize(rows):
    """Model distributions and all three preregistered paired comparisons."""
    keys = (*ERROR_KEYS, 'fixed_target_coverage','gamma_coverage','free_area',
            'narrow_q_gt10_fraction','min_angle_lt5_fraction','near_degenerate_count',
            'CF_proper_affected_free_fraction','CF_proper_pair_count','CF_coplanar_pair_count',
            'FF_illegal_pair_count','FF_proper_pair_count','FF_coplanar_pair_count',
            'new_free_related_nonmanifold_vertex_count')
    groups, comparisons = [], []
    for group in ('original', 'new'):
        subsets = [(None, None), (0, None), (1, None)]
        if group == 'new':
            subsets += [(p, j) for p in (0,1) for j in (1,2)]
        for parent, patch_id in subsets:
            subset = [r for r in rows if r['patch_group']==group and (parent is None or r['parent']==parent)
                      and (patch_id is None or r['patch_id']==patch_id)]
            for model in MODELS:
                own = [r for r in subset if r['model']==model]
                distributions={key:_stats([r.get(key) for r in own],key in HIGHER_BETTER) for key in keys}
                for key in NEUTRAL_KEYS:distributions[key]['worst']=None
                groups.append(dict(patch_group=group, parent=parent, patch_id=patch_id, model=model,
                    sample_count=len(own),metrics=distributions,
                    worst_surface_output=max(own,key=lambda r:r['free_rms_pct'])['output_id'] if own else None))
            for newer, reference in COMPARISONS:
                for key in keys:
                    comparisons.append(paired_statistics(rows,newer,reference,key,patch_group=group,parent=parent,patch_id=patch_id))
    return dict(distributions=groups, paired_comparisons=comparisons)


def _rounding_logs(training):
    logs = []
    for arm, report in training.items():
        for entry in report.get('logs', []):
            observation = entry.get('context_observation')
            if not observation:
                continue
            injections = observation.get('diagnostics', {}).get('injections', [])
            nonzero = sum(row['nonzero_delta_elements'] for row in injections)
            swallowed = sum(row['swallowed_nonzero_elements'] for row in injections)
            logs.append(dict(arm=arm, step=entry['step'], microbatch_count=len(injections),
                delta_RMS_known=_stats([row['delta_RMS_known'] for row in injections]),
                effective_RMS_known=_stats([row['effective_RMS_known'] for row in injections]),
                effective_minus_delta_RMS_known=_stats([row['effective_minus_delta_RMS_known'] for row in injections]),
                swallowed_nonzero_elements=swallowed, nonzero_delta_elements=nonzero,
                swallowed_nonzero_fraction=swallowed/nonzero if nonzero else None,
                RMS_summary='Mean/median of per-microbatch known-element RMS; not a pooled coordinate RMS',
                free_direct_delta_exact_zero=all(row['free_direct_delta_exact_zero'] for row in injections),
                free_hidden_unchanged=all(row['free_hidden_unchanged'] for row in injections),
                dtypes=sorted(set((row['hidden_dtype'],row['residual_dtype'],row['after_dtype']) for row in injections)),
                gradients_and_updates=entry.get('encoder_gradient_update', {})))
    return logs


def _counts(gates, training, generation):
    phases = []
    def add(label, value, euler=None):
        phases.append(dict(phase=label,
            forward_attempts=value.get('forward_attempts',value.get('model_forward_attempts',0)),
            forward_returns=value.get('forward_returns',value.get('model_forward_returns',0)),
            backward_attempts=value.get('backward_attempts',value.get('model_backward_calls',0)),
            backward_returns=value.get('backward_returns',value.get('model_backward_calls',0)),
            optimizer_step_attempts=value.get('optimizer_step_attempts',0),optimizer_updates=value.get('optimizer_updates',0),
            complete_rollouts=value.get('complete_rollouts',0), rollout_attempts=value.get('rollout_attempts',0),
            euler_updates=value.get('euler_updates',0) if euler is None else euler,OT_calls=value.get('OT_calls',0)))
    add('gates',gates.get('counts',{}),gates.get('replay_counts',{}).get('euler_updates',0))
    for arm, report in training.items():
        add(arm+'_training',report['counts']);add(arm+'_save_reload',report.get('validation_counts',{}))
    add('formal_generation',generation['formal_counts']);add('endpoint_replays',generation['replay_counts'])
    total={key:sum(row[key] for row in phases) for key in phases[0] if key!='phase'}
    return dict(phases=phases,total=total,scope='Gate forward/backward, three training arms, velocity reload checks, formal and replay sampling; no teacher evaluation')


def _gate_policy_summary(gates):
    failures=list(gates.get('recorded_numerical_failures',[]))
    if not failures:
        for arm,row in gates.get('arms',{}).items():
            numerical=row.get('permutation',{})
            if numerical.get('status')=='FAIL':
                failures.append(dict(arm=arm,site='full_model_bf16_permutation',
                                     blocking=False,**numerical))
    return dict(policy=gates.get('policy','original_strict_permutation_gate'),
        overall_status=gates.get('status'),hard_wiring_status=gates.get('hard_wiring_status',gates.get('status')),
        numerical_diagnostics_status=gates.get('numerical_diagnostics_status',
            'HAS_RECORDED_FAILURES' if failures else 'SEE_ORIGINAL_GATE_RECORD'),
        recorded_numerical_failures=failures,
        baseline_regression_evidence=gates.get('baseline_regression_evidence'),
        interpretation='Overall PASS means the authorized hard wiring contract passed; recorded BF16 permutation FAIL remains a numerical boundary of the baseline, not a hidden PASS.')


def _diagnostic_history(out):
    """Separate prior/supplemental model calls and partial-operator executions."""
    path=out/'diagnostic_history.json'
    if not path.is_file():
        return dict(status='NOT_RECORDED',entries=[],partial_operator_calls={},
                    note='No supplementary diagnostic ledger supplied; main experiment counters do not silently include historical calls.')
    data=_read(path)
    if not isinstance(data,dict) or not isinstance(data.get('entries'),list):
        raise ValueError('Diagnostic history requires an entries list.')
    for entry in data['entries']:
        source=entry.get('source')
        if source:
            if not isinstance(source,dict) or not source.get('path') or not source.get('sha256'):
                raise ValueError('Diagnostic source needs path and SHA256.')
            if file_sha256(source['path'])!=source['sha256']:
                raise ValueError('Supplementary diagnostic evidence changed.')
    result=dict(data)
    result['source']=_record(path)
    result['scope']='Historical stopped gate, separate read-only/FP32 probes and partial operators; excluded from the new run training/generation budget. Partial calls are not full model forwards.'
    keys=('forward_attempts','forward_returns','backward_attempts','backward_returns','optimizer_updates','complete_rollouts','OT_calls')
    result['separate_model_totals']={key:sum(entry.get('counts',{}).get(key,0) for entry in data['entries']) for key in keys}
    return result

def _precision_diagnosis_summary(diagnostic_history):
    """Read the already completed diagnostic; never execute a numerical probe."""
    entry=next((row for row in diagnostic_history.get('entries',[])
                if row.get('label')=='FP32_and_first_block_localization'),None)
    if entry is None or not isinstance(entry.get('source'),dict):
        return dict(status='NOT_RECORDED',reason='No registered FP32/first-block diagnostic source supplied.')
    source=entry['source'];path=Path(source.get('path',''))
    if not path.is_file():
        return dict(status='NOT_RECORDED',reason='Registered precision result file is unavailable.',source=source)
    if file_sha256(path)!=source.get('sha256'):
        raise ValueError('Precision diagnosis source SHA256 differs from the registered history.')
    saved=_read(path);fp32=saved.get('FP32',{});bf16=saved.get('BF16_saved_comparisons',{})
    names=('face_only','corners_only','combined')
    if any(name not in fp32 or name not in bf16 or 'permutation' not in fp32[name] for name in names):
        return dict(status='NOT_RECORDED',reason='Source lacks the required fixed FP32/BF16 comparison records.',source=source)
    comparisons=[dict(permutation=name,BF16=bf16[name],FP32=fp32[name]['permutation']) for name in names]
    block=saved.get('block_replay',{});sites=block.get('sites',[]);first=block.get('first_nonbitwise_site')
    return dict(status='RECORDED_DIAGNOSTIC_ONLY',source=source,original_status=saved.get('status'),
        comparisons=comparisons,first_nonbitwise_site=first,
        first_nonbitwise_record=next((row for row in sites if row.get('site')==first),None),
        both_standalone_blocks_match_original_trace_bitwise=block.get('original_trace_both_bitwise'),
        block_sites=sites,counts=saved.get('counts',{}),
        parameters_unchanged=saved.get('parameters_unchanged'),source_unchanged=saved.get('source_unchanged'),
        state_before=saved.get('state_before'),state_after=saved.get('state_after'),
        production_precision_changed=False,
        internal_SDPA_operator_localization='NOT_PERFORMED',
        scope='Saved FP32 diagnostics and BF16 first-block replay evidence only; no new model/operator calls or production precision change.')

def _audit_input(audit_results, rows):
    if audit_results is None:
        return {}, dict(status='NOT_RUN', reason='No independent strict audit JSON supplied.',
                        CF='NOT_RUN',FF='NOT_RUN',full_topology='NOT_RUN')
    if isinstance(audit_results,(str,Path)):
        audit=_read(audit_results); source=_record(audit_results)
    else:
        audit=audit_results; source=None
    expected={row['output_id']:row for row in rows}; matched={}
    for row in audit.get('rows',[]):
        identity=row['output_id']
        if identity not in expected or identity in matched:
            raise ValueError('External audit contains an unknown or duplicate output ID.')
        original=expected[identity]
        if row['raw_sha256']!=original['raw_sha256'] or any(row[k]!=original[k] for k in ('model','parent','patch_id','seed')):
            raise ValueError('External audit does not refer to the exact registered raw output.')
        matched[identity]=row
    summary=dict(status='COMPLETE' if len(matched)==len(expected) else 'PARTIAL',source=source,
        recorded_status=audit.get('status'),scored_outputs=len(matched),expected_outputs=len(expected),
        total_seconds=audit.get('total_seconds'),endpoint_seconds=audit.get('endpoint_seconds'),
        patch_input_seconds=audit.get('patch_input_seconds'),ledger=audit.get('ledger',{}),
        detector_policy=audit.get('detector_policy'),patch_audits=audit.get('patch_audits',{}),
        exact_predicate_certification='NOT_RUN',continuous_collision_safety='NOT_RUN',
        caveat='Offline sampled endpoint triangle checks and exact-coordinate topology; execution PASS is not mesh validity, exact-predicate certification or continuous collision safety.')
    return matched,summary


def _make_figures(out,rows,bank):
    # Standard offline plotting only. No model or geometric evaluation occurs.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image
    figures=[]
    for j in (0,1,2):
        selected={(p,m):next(r for r in rows if (r['parent'],r['model'],r['patch_id'],r['seed'])==(p,m,j,9701))
                  for p in (0,1) for m in MODELS}
        raw={}
        for key,row in selected.items():
            path=Path(row['raw_path'])
            if file_sha256(path)!=row['raw_sha256']:
                raise ValueError('Raw output changed before rendering.')
            with np.load(path,allow_pickle=False) as saved:
                value=saved['output'].copy()
            if value.dtype!=np.float32 or value.shape!=(112,9) or not np.isfinite(value).all():
                raise ValueError('Figure requires original finite FP32 N112 output.')
            raw[key]=value.reshape(112,3,3)
        fig,axes=plt.subplots(4,4,figsize=(15,13),subplot_kw={'projection':'3d'})
        for p in (0,1):
            geometry=np.concatenate([raw[p,m].reshape(-1,3) for m in MODELS]+
                                    [bank['cases'][p,j].full_target.reshape(-1,3)])[:,[0,2,1]]
            low,high=geometry.min(0),geometry.max(0);center=(low+high)/2;radius=max(high-low)*.55
            for c,m in enumerate(MODELS):
                triangles=raw[p,m][:,:,[0,2,1]]
                for view in (0,1):
                    ax=axes[2*p+view,c]
                    ax.add_collection3d(Poly3DCollection(triangles[12:],facecolors='#74a2c8',edgecolors='#26394a',linewidths=.22,alpha=.92))
                    if view==0:
                        ax.add_collection3d(Poly3DCollection(triangles[:12],facecolors='#dc755d',edgecolors='#6b251a',linewidths=.45,alpha=1.))
                    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius)
                    ax.set_zlim(center[2]-radius,center[2]+radius);ax.set_box_aspect((1,1,1))
                    ax.view_init(elev=20,azim=-60);ax.set_axis_off()
                    metric=selected[p,m]['metrics']['remainder']['own']['symmetric_rms_bbox_pct']
                    ax.set_title(f'{m} | P{p} '+('full' if view==0 else 'free only')+f'\nfree RMS {metric:.3f}% L',fontsize=9,pad=0)
        kind='Original training C12' if j==0 else f'New evaluation patch {j}'
        fig.suptitle(f'{kind} | seed 9701 | same view / shared parent bounds | known C: red, free: blue',fontsize=12)
        fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.94,wspace=.0,hspace=.0)
        path=out/f'comparison_patch{j}_seed9701.png';fig.savefig(path,dpi=160,facecolor='white');plt.close(fig)
        with Image.open(path) as picture:
            picture.verify()
        figures.append(dict(patch_id=j,path=str(path.resolve()),sha256=file_sha256(path),seed=9701,
            elevation=20,azimuth=-60,display_coordinate_order=['x','z','y'],
            shared_bounds='Per parent, common across all four models/full/free; includes all outputs and GT',
            rows=['P0 full','P0 free','P1 full','P1 free'],columns=list(MODELS),
            image_file_integrity='PASS',human_visual_review='PENDING'))
    return figures


def _fmt(value,digits=4):
    if value is None:return 'NOT_RUN'
    if isinstance(value,(float,np.floating)):
        if value and (abs(value)<10**(-digits) or abs(value)>=1e6):
            return f'{value:.{digits}g}'
        return f'{value:.{digits}f}'
    return str(value)


def _table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+[
        '| '+' | '.join(_fmt(v) for v in row)+' |' for row in rows]


def _comparison(summary,newer,reference,group,key='free_rms_pct',parent=None,patch_id=None):
    return next(r for r in summary['paired_comparisons'] if
        (r['newer'],r['reference'],r['patch_group'],r['metric'],r['parent'],r['patch_id'])==
        (newer,reference,group,key,parent,patch_id))


def _conclusions(summary,strict_status):
    result=[]
    titles={('T_base','T0'):'额外500步纯FM',('T_geo','T_base'):'单面几何编码增量',('T_graph','T_geo'):'同参数量的C内部邻接增量'}
    for newer,ref in COMPARISONS:
        text=[];means=[]
        for group,label in (('original','原片'),('new','新片')):
            row=_comparison(summary,newer,ref,group);s=row['statistics'];means.append(s['mean'])
            text.append(f"{label}配对RMS差均值{_fmt(s['mean'])}、中位{_fmt(s['median'])}个百分点，改善{row['improved']}/{s['n']}")
        verdict=('两组平均误差均下降，出现有限条件下的增量信号。' if all(v is not None and v<0 for v in means) else
                 '原片平均误差下降，新片未同向改善，不能扩展为一般上下文收益。' if means[0] is not None and means[0]<0 else
                 '未建立原片与新片一致的平均误差改善。')
        result.append(f"{titles[newer,ref]}（{newer}−{ref}）："+'；'.join(text)+'。'+verdict)
    # Sensitivity summaries consume stored pair scores; retain every primary sample.
    p0_geo=_comparison(summary,'T_geo','T_base','original',parent=0)
    p0_graph=_comparison(summary,'T_graph','T_geo','original',parent=0)
    adverse_pairs=[p for p in p0_geo['paired_differences'] if p['difference']>0]
    if adverse_pairs:
        worst=max(adverse_pairs,key=lambda p:p['difference'])
        graph_pair=next(p for p in p0_graph['paired_differences'] if
            (p['parent'],p['patch_id'],p['seed'])==(worst['parent'],worst['patch_id'],worst['seed']))
        graph_total=sum(p['difference'] for p in p0_graph['paired_differences'])
        others=[p['difference'] for p in p0_graph['paired_differences'] if p is not graph_pair]
        result.append(f"P0原片最明显的Geo退化发生于seed {worst['seed']}：自由面RMS从Base的{_fmt(worst['reference_value'])}% L变为Geo的{_fmt(worst['newer_value'])}% L，"
            f"其余{p0_geo['statistics']['n']-1}例中{sum(p['difference']<0 for p in p0_geo['paired_differences'] if p is not worst)}例改善。"+
            (f"Graph相对Geo的该例改善占该组总配对误差下降的{100*graph_pair['difference']/graph_total:.2f}%；"
             if graph_total<0 and graph_pair['difference']<0 else '')+
            (f"仅作敏感性检查，移除此例后的Graph−Geo配对差均值为{_fmt(float(np.mean(others)))}个百分点。" if others else '')+
            '正式均值、配对表和结论均保留该例，没有剔除或另选终点。')
    new_geo=_comparison(summary,'T_geo','T_base','new')
    sensitivity=[]
    for parent in (0,1):
        diffs=[p['difference'] for p in _comparison(summary,'T_geo','T_base','new',parent=parent)['paired_differences']]
        if len(diffs)>1:
            loo=[float(np.mean(diffs[:i]+diffs[i+1:])) for i in range(len(diffs))]
            sensitivity.append(f"P{parent}留一配对差均值范围[{_fmt(min(loo))}, {_fmt(max(loo))}]")
    result.append(f"Geo相对Base在新片自由面RMS改善{new_geo['improved']}/{new_geo['statistics']['n']}例；"+
        '；'.join(sensitivity)+'个百分点。这只是小样本敏感性检查，不替代全部样本统计或独立新对象验证。')
    result.append('上述新片没有参与本轮微调，但父对象和完整目标已见过；每对象原片4噪声、新片各2噪声，不是新对象泛化或统计显著性结论。')
    if strict_status=='NOT_RUN':
        result.append('严格CF/FF和完整拓扑NOT_RUN；表面/接口距离改善不能证明无自交、水密或流形。')
    else:
        adverse=[]
        for newer,ref in COMPARISONS:
            for group in ('original','new'):
                for key in ('CF_proper_affected_free_fraction','FF_illegal_pair_count'):
                    value=_comparison(summary,newer,ref,group,key)['statistics']['mean']
                    if value is not None and round(value,12)>0:adverse.append(f'{newer}−{ref}/{group}/{key}={value:.5g}')
        result.append(('独立审计观察到质量代价：'+ '；'.join(adverse)+'。' if adverse else
                       '按已执行的独立审计另列CF/FF与拓扑结果；不能把执行通过当作合法网格。')+
                      '未执行精确谓词认证或连续碰撞安全证明。')
        cost=[]
        for parent in (0,1):
            cf=_comparison(summary,'T_geo','T_base','new','CF_proper_affected_free_fraction',parent)
            ff=_comparison(summary,'T_geo','T_base','new','FF_illegal_pair_count',parent)
            gamma=_comparison(summary,'T_geo','T_base','new','gamma_rms_pct',parent)
            coverage=_comparison(summary,'T_geo','T_base','new','gamma_coverage',parent)
            if any(r['statistics']['mean'] is None for r in (cf,ff,gamma,coverage)):
                cost.append(f'P{parent}：独立审计数据不完整，不作质量差异判定')
                continue
            cost.append(f"P{parent}：CF受穿刺free比例差{_fmt(100*cf['statistics']['mean'])}个百分点（{cf['worsened']}/{cf['statistics']['n']}例增加），"
                f"FF非法对均数差{_fmt(ff['statistics']['mean'])}（{ff['improved']}/{ff['statistics']['n']}例减少）；"
                f"Gamma RMS差{_fmt(gamma['statistics']['mean'])}个百分点（{gamma['improved']}/{gamma['statistics']['n']}例改善），"
                f"Gamma覆盖率差{_fmt(100*coverage['statistics']['mean'])}个百分点（{coverage['worsened']}/{coverage['statistics']['n']}例下降）")
        result.append('Geo的新片表面收益与接口质量存在对象差异。'+'；'.join(cost)+'。不能把自由面RMS改善等同于接缝和相交同时改善。')
    result.append('到此停止：已完成用户随后明确授权的原定预算；没有自动扩训、改损失、增层或添加修补模块。')
    return result


def _cpu_test_records(out):
    import re
    records=[]
    for path in sorted(out.glob('*test*.json')):
        data=_read(path)
        if not isinstance(data,dict):continue
        records.append(dict(name=path.name,status=data.get('status','RECORDED'),
            tests=data.get('tests',data.get('test_families',data.get('checks'))),seconds=data.get('seconds'),
            source=_record(path),details=data))
    path=out/'all_cpu_tests.log'
    if path.is_file():
        raw=path.read_bytes()
        encoding='utf-16' if raw.startswith((bytes.fromhex('fffe'),bytes.fromhex('feff'))) else 'utf-8-sig'
        text=raw.decode(encoding);matches=re.findall(r'Ran (\d+) tests in ([0-9.]+)s',text)
        if matches:
            count,seconds=matches[-1]
            records.append(dict(name=path.name,status='PASS' if re.search(r'^OK\s*$',text,re.M) else 'CHECK_LOG',
                tests=int(count),seconds=float(seconds),source=_record(path),
                details=dict(note='Suite count from unittest transcript; may overlap other recorded suites.')))
    return records


def _write_stopped_report(out,registration,gates):
    """Gate failure is a complete stopping outcome, never a fabricated experiment."""
    import re
    tick=time.perf_counter()
    if any((out/arm/'training.json').exists() for arm in MODELS[1:]) or (out/'generation.json').exists():
        raise RuntimeError('Early-gate report cannot hide an already started training/generation phase.')
    tests=_cpu_test_records(out)
    bank,evaluation=load_evaluation(out/'registration/evaluation_inputs.npz',registration['evaluation_inputs_sha256'])
    offline_path=out/'offline_gate_audit.json'
    offline=_read(offline_path) if offline_path.is_file() else dict(status='NOT_RUN')
    proof=gates.get('proofs',{}).get('single_step_velocities',{})
    if proof:
        actual=_record(proof['path'])
        if actual['sha256']!=proof['sha256']:raise ValueError('Recorded failed-gate velocity proof changed.')
    error=gates.get('error','Unspecified gate failure')
    revised_policy=gates.get('policy')=='baseline_regression_and_branch_inheritance_v2'
    encoder_started=any(key in gates.get('arms',{}) for key in ('T_geo','T_graph'))
    numeric={}
    for key in ('max_abs','RMS','atol','rtol'):
        match=re.search(r"'"+key+r"': ([0-9.eE+-]+)",error)
        if match:numeric[key]=float(match.group(1))
    counts=dict(gates.get('counts',{}))
    counts['euler_updates']=gates.get('replay_counts',{}).get('euler_updates',0)
    for key in ('forward_attempts','forward_returns','backward_attempts','backward_returns','optimizer_updates','complete_rollouts','OT_calls'):
        counts.setdefault(key,0)
    phases=dict(training='NOT_RUN',endpoint_generation='NOT_RUN',sampling_replays='NOT_RUN',
        strict_CF_FF_audit='NOT_RUN',full_topology_audit='NOT_RUN',fixed_view_figures='NOT_RUN',
        encoder_GPU_zero_branch_and_gradient_tests='NOT_RUN',encoder_delta_step10_100_500='NOT_RUN',
        endpoint_save_reload='NOT_RUN',comparative_benefit='NOT_EVALUATED')
    source_files={name:_record(out/name) for name in ('registration.json','gates.json')}
    if offline_path.is_file():source_files['offline_gate_audit.json']=_record(offline_path)
    delivery={}
    for filename in ('delivery_integrity.json','code_manifest.json','post_stop_guard_amendment.json'):
        path=out/filename
        delivery[filename]=_read(path) if path.is_file() else dict(status='NOT_RECORDED')
        if path.is_file():source_files[filename]=_record(path)
    conclusions=[
        '本轮在训练前数值门禁停止。纯T0在组合面重排与面内S3角置换后的前向，未满足预登记BF16 allclose容差；失败发生在新增编码器接入GPU之前。',
        '实现与CPU测试已经完成，新增几何/图分支的真实GPU接线、梯度、训练和自由生成尚未通过本轮验证。不能据此判断单面几何或邻接编码有益或无益。',
        'T_base相对T0、T_geo相对T_base、T_graph相对T_geo以及原片/新片收益均未评价。没有新训练终点、生成图、CF/FF成绩或架构优势结论。',
        '门槛、精度和核心数值路径未因失败而放宽；不继续训练，不追加模型前向，不自动进入下一阶段。']
    if revised_policy:
        conclusions[0]='本轮在训练前硬接线门禁停止；具体失败见错误原文。旧容差下的BF16全主干置换FAIL另作非阻断数值诊断，不能把两类检查混为一谈。'
        conclusions[1]='真实GPU接线仅执行到失败位置，已执行子项以gates原始记录为准；未完成的梯度、哈希或C51检查不能计为通过。未进行本轮训练或正式生成。'
        conclusions[3]='不因硬门禁失败放宽容差或继续训练；没有自动追加模型调用。'
        if encoder_started:phases['encoder_GPU_zero_branch_and_gradient_tests']='PARTIAL_OR_FAILED_SEE_GATES'
    result=dict(schema='native_t1_geometry_context_results_v1',status='STOPPED_NUMERICAL_GATE',
        protocol={k:v for k,v in registration.items() if k!='initial_stream_state'},
        evaluation_registration=evaluation,gates=gates,cpu_tests=tests,offline_gate_audit=offline,
        delivery_integrity=delivery['delivery_integrity.json'],code_manifest=delivery['code_manifest.json'],
        post_stop_guard_amendment=delivery['post_stop_guard_amendment.json'],
        peak_cuda_allocated='NOT_RECORDED',
        failure=dict(stage='hard_wiring' if revised_policy else 'pure_T0_combined_face_S3_permutation',error=error,numerical_error=numeric,
                     encoder_attached_at_failure=encoder_started,thresholds=gates.get('thresholds',{})),
        gate_policy=_gate_policy_summary(gates),diagnostic_history=_diagnostic_history(out),
        counts=dict(actual_model_execution=counts,
            scope='Actual MeshFlow model GPU calls; CPU fixture autograd/OT and metadata tests are separate test records, not silently counted as zero.'),
        phases=phases,training={arm:dict(status='NOT_RUN',completed_updates=0) for arm in MODELS[1:]},
        generation=dict(status='NOT_RUN',formal_outputs=0,replay_outputs=0),outputs=[],
        endpoints={arm:dict(status='NOT_RUN',path=None,sha256=None) for arm in MODELS[1:]},
        figures=dict(status='NOT_RUN',files=[],reason='No completed sample exists; no placeholder images generated.'),
        strict_audit=dict(status='NOT_RUN',CF='NOT_RUN',FF='NOT_RUN',full_topology='NOT_RUN'),
        rounding=dict(status='NOT_RUN',steps=[10,100,500]),conclusions=conclusions,
        source_files=source_files,reporter=dict(model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,
                                              geometry_rescoring=0,figures_rendered=0))
    lines=['# GEOMETRY_CONTEXT_REPORT — 数值门禁早停','',
        f"**本轮没有新增训练。** 实际完成{counts['forward_returns']}次模型前向，门禁停止；新增编码器GPU阶段{'已开始，详见逐项记录' if encoder_started else '尚未开始'}。",'']
    lines+=conclusions+['','## 实际执行与预算','',
        '授权预算是三臂各500次更新；实际执行按下表记账。未将授权预算写成已完成工作。','']
    lines+=_table(['项目','实际值'],[[key,counts[key]] for key in
        ('forward_returns','forward_attempts','backward_returns','backward_attempts','optimizer_updates','complete_rollouts','euler_updates','OT_calls')])
    lines+=['',f"门禁耗时：{_fmt(gates.get('seconds'))}秒。CPU测试中的fixture梯度/OT与模型调用分开记录；测试阶段不等于新增训练。",'',
        '## 失败证据','',
        f"共同初始化：`{registration['initialization_path']}`；SHA256 `{registration['initialization_file_sha256']}`，累计1500步纯A。",
        f"严格加载记录：`{gates.get('initialization',{}).get('status','UNKNOWN')}`；初始state SHA `{gates.get('pure_parameter_hash_before','NOT_RECORDED')}`。",
        '固定状态：P0/K12，t=0.5，CPU噪声seed12345，未运行OT；面/角排列seed24680。该状态独立于训练输入流seed1010。',
        '这是本轮新诊断状态，不是历史完全相同case；仅沿用历史容差。置换同时包含面顺序与S3角顺序，因此本次失败不能单独归因到其中一种。',
        '失败前没有记录峰值显存：NOT_RECORDED。没有用事后GPU当前占用替代峰值。','']
    lines+=_table(['量','记录值'],[[key,value] for key,value in numeric.items()])
    lines+=['',f"原错误：`{error}`。",f"单步原始输出：`{proof.get('path','NOT_RUN')}`；SHA256 `{proof.get('sha256','NOT_RUN')}`。",
        '失败比较采用原登记atol/rtol组合判定；不能只凭最大绝对误差单独替代allclose判据。首次门禁尚未完成的后续参数哈希/零分支/梯度/重放检查不计为通过。','',
        '## CPU测试记录','',
        '各套件有重叠，下列测试数量不能相加解释为互不重复的测试总数。','']
    lines+=_table(['记录','状态','测试/家族数','耗时秒'],[[r['name'],r['status'],r['tests'],r['seconds']] for r in tests])
    lines+=['','## 已冻结的评价输入','',
        f"评价注册：`{evaluation['registration']['path']}`；SHA256 `{evaluation['registration']['sha256']}`。",
        '原两对象C12与每对象两块新片已在训练前完成选择及hash登记；64条正式生成计划未执行。新片按原始源面ID的共享整边BFS选择，不根据生成结果挑片；各片保留自己的100面自由补集与固定评价点。','']
    lines+=_table(['片','旧片重合面','新增面','面积占比','边界边数'],[
        [r['case_id'],r['original_overlap_faces'],r['new_face_count'],r['patch_area_fraction'],r['trusted_source_boundary_edges']] for r in evaluation['cases']])
    lines+=['','## 离线门禁审计','',
        f"状态：{offline.get('status','RECORDED')}。此记录只读取既存速度数组/排列/代码证据，不是新的模型运行。"]
    if offline_path.is_file():
        lines += [f"完整证据：[offline_gate_audit.json](offline_gate_audit.json)；完整内容另保留在results.json。",
                  f"同输入重复逐位一致：{offline.get('reference_repeat',{}).get('bitwise_equal','NOT_RECORDED')}。离线复现成功不代表原数值门禁通过。",'']
        comparison=offline.get('permutation_comparison',{})
        lines+=_table(['区域','元素数','越阈元素','涉及面','最大绝对差','RMS','allclose'],[
            [label,comparison.get(key,{}).get('element_count'),comparison.get(key,{}).get('violating_elements'),
             comparison.get(key,{}).get('violating_faces'),comparison.get(key,{}).get('max_abs'),
             comparison.get(key,{}).get('RMS'),comparison.get(key,{}).get('allclose')]
            for key,label in (('all','全部'),('known','known'),('free','free'))])
        lines += ['NumPy与既存Torch比较的阈值mask及排列对账见离线原记录。离线审计未追加模型前向、反向、OT或更新；没有将失败归因于尚未接入GPU的新编码器。']
    else:
        lines += ['未提供offline_gate_audit.json，保持NOT_RUN；不从缺失证据推断根因。']
    lines+=['','## 交付与完整性','']
    for filename,data in delivery.items():
        status=data.get('status','RECORDED') if isinstance(data,dict) else 'RECORDED'
        source=source_files.get(filename)
        lines += [f"- {filename}：{status}；"+(f"SHA256 `{source['sha256']}`。" if source else '未记录。')]
    lines+=['完整源文件清单、代码diff与资产保护验证以相应原始JSON为准，并保留在results.json。',
        '实际失败GPU调用采用原registration记录的执行源码。交付代码随后增加仅入口的fail-closed保护，阻止直接调用generate阶段绕过失败门禁；该修订单独记录，未改变数学或容差，也没有追加GPU执行。原registration pins和gates证据保持原样。']
    lines+=['','## 未执行事项','']+_table(['阶段','状态'],list(phases.items()))
    lines+=['','三个500步终点：T_base、T_geo、T_graph均NOT_RUN；没有对应新checkpoint路径或hash。',
        '固定首噪声full/free图：NOT_RUN。没有采样数据，因此不创建空图或伪图。',
        '本轮不能回答几何编码/邻接是否增强补全；下一步首先需要澄清纯T0置换门禁的数值契约与失败原因。没有擅自放宽容差或启动后续实验。',
        '交付为已实现代码diff、实际CPU记录、该报告、results.json、test_summary.csv及既存门禁原始速度证据。没有新增训练权重或生成输出，也没有复制官方大权重。','']
    (out/'GEOMETRY_CONTEXT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    with (out/'test_summary.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['record','status','tests_or_families','seconds','source_path','source_sha256'])
        writer.writeheader()
        writer.writerows(dict(record=r['name'],status=r['status'],tests_or_families=r['tests'],seconds=r['seconds'],
                              source_path=r['source']['path'],source_sha256=r['source']['sha256']) for r in tests)
    result['reporter']['seconds']=time.perf_counter()-tick
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result

def write_report(out,audit_results=None):
    """Write one final report/JSON/CSV and three fixed-view PNGs after completion."""
    tick=time.perf_counter();out=Path(out).resolve()
    registration=_read(out/'registration.json');gates=_read(out/'gates.json')
    if gates.get('status') == 'STOPPED':
        return _write_stopped_report(out,registration,gates)
    generation=_read(out/'generation.json')
    training={arm:_read(out/arm/'training.json') for arm in MODELS[1:]}
    if gates.get('status')!='PASS' or generation.get('status')!='PASS' or len(generation.get('rows',[]))!=64:
        raise RuntimeError('Report requires completed gates and the entire 64-output fixed matrix.')
    if any(r.get('status')!='PASS' or r.get('completed_updates')!=500 or r.get('update_in_progress') for r in training.values()):
        raise RuntimeError('Each of the three arms must finish exactly 500 updates.')
    bank,evaluation=load_evaluation(out/'registration/evaluation_inputs.npz',registration['evaluation_inputs_sha256'])
    expected={(m,p,j,s) for m in MODELS for p in (0,1) for j in (0,1,2) for s in ((9701,9702,9703,9704) if j==0 else (9701,9702))}
    identities=[(r['model'],r['parent'],r['patch_id'],r['seed']) for r in generation['rows']]
    if len(identities)!=len(set(identities)) or set(identities)!=expected:
        raise ValueError('Formal outputs differ from the preregistered model/parent/patch/noise matrix.')
    streams=[]
    for arm,report in training.items():
        actual=file_sha256(out/arm/'input_stream.jsonl')
        if actual!=report['input_stream_sha256']:raise ValueError('Saved training stream log changed.')
        streams.append(actual)
    if len(set(streams))!=1 or not all(r.get('input_stream_matches_T_base') for r in training.values()):
        raise ValueError('Three training streams are not identical.')
    if len({r['pure_initial_state_sha256'] for r in training.values()})!=1:
        raise ValueError('Original model initialization differs across arms.')
    if training['T_geo']['encoder_initial_sha256']!=training['T_graph']['encoder_initial_sha256']:
        raise ValueError('Geometry encoder initializations differ.')
    audit_rows,strict=_audit_input(audit_results,generation['rows'])
    rows=[_flatten(row,audit_rows.get(row['output_id'])) for row in generation['rows']]
    summary=summarize(rows);rounding=_rounding_logs(training);counts=_counts(gates,training,generation)
    gate_policy=_gate_policy_summary(gates);diagnostic_history=_diagnostic_history(out);cpu_tests=_cpu_test_records(out)
    precision_summary=_precision_diagnosis_summary(diagnostic_history)
    if counts['total']['optimizer_updates']!=1500 or counts['total']['complete_rollouts']>68:
        raise ValueError('Recorded update/rollout budget differs from the authorized budget.')
    figures=_make_figures(out,generation['rows'],bank)
    conclusions=_conclusions(summary,strict['status'])
    if gate_policy['recorded_numerical_failures']:
        conclusions.insert(0,'本轮整体PASS仅表示硬接线门禁与授权预算执行完成；BF16全主干置换在原容差下的FAIL仍保留为非阻断数值诊断，没有改记为通过，也没有改主干精度。')
    arm_keys=('status','completed_updates','counts','validation_counts','timing','seconds','parameters',
              'all_parameters_trainable','optimizer_parameter_count','initialization','pure_initial_state_sha256',
              'context','encoder_initial_sha256','input_stream_sha256','input_stream_matches_T_base','peak_cuda_allocated','reload','checkpoints','logs')
    results=dict(schema='native_t1_geometry_context_results_v1',status='PASS_EXECUTION',
        protocol={k:v for k,v in registration.items() if k!='initial_stream_state'},
        evaluation_registration=evaluation,gates=gates,gate_policy=gate_policy,diagnostic_history=diagnostic_history,cpu_tests=cpu_tests,
        precision_diagnosis_summary=precision_summary,
        training={arm:{k:v for k,v in report.items() if k in arm_keys} for arm,report in training.items()},
        generation={k:v for k,v in generation.items() if k!='rows'},outputs=generation['rows'],
        compact_outputs=rows,summary=summary,rounding=rounding,counts=counts,strict_audit=strict,
        strict_output_scalars={key:dict(status=value.get('status'),scalars=value.get('scalars'),
            embedded_geometry_status=value.get('embedded_geometry_status'),seconds=value.get('seconds')) for key,value in audit_rows.items()},
        endpoints={arm:report['checkpoints'][-1] for arm,report in training.items()},
        figures=figures,conclusions=conclusions,
        source_files={name:_record(out/name) for name in ('registration.json','gates.json','generation.json')},
        reporter=dict(model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,geometry_rescoring=0,
            scope='Statistics, integrity checks and Matplotlib rendering only; no checkpoint copies or geometry repair'))
    lines=['# GEOMETRY_CONTEXT_REPORT','',
        '本轮新增训练：三臂各500步，共1500次优化器更新；共同纯A初始累计1500步，终点累计2000步。结果只覆盖两个见过的父对象。','']
    lines+=conclusions+['','## 固定协议与执行','',
        f"初始化：`{registration['initialization_path']}`；SHA256 `{registration['initialization_file_sha256']}`。",
        '三臂fresh AdamW，输入seed1010；effective batch8 / microbatch1，原free-only masked FM与nested OT。没有FC/FF辅助损失、Guidance或后处理。',
        '主干BF16、参数/积分FP32；新增几何与MLP固定FP32。T_geo与T_graph参数和初始化相同，仅P=I与C内exact-FP32共享边图不同。',
        f"同流日志：`{streams[0]}`；三臂500个effective batch逐步一致。原提交CPU测试与本轮新增测试记录保存在本run目录；GPU门禁 `{gates['status']}`。",'']
    lines+=_table(['训练臂','总参数','新增参数','optimizer参数','全部可训练'],[
        [arm,report['parameters'],report['parameters']-training['T_base']['parameters'],report['optimizer_parameter_count'],report['all_parameters_trainable']]
        for arm,report in training.items()])+['']
    lines+=['','## 分层门禁与已知数值边界','',
        f"policy=`{gate_policy['policy']}`；硬接线状态 `{gate_policy['hard_wiring_status']}`；完整BF16数值诊断状态 `{gate_policy['numerical_diagnostics_status']}`。",
        '整体PASS只覆盖硬接线契约：原版T0回归、同输入重复、零分支逐位匹配对应T0、已知特征/图/h1置换、梯度、参数hash和C的51状态保持。',
        '原BF16完整主干面/角置换使用原atol=0.015625、rtol=0.03125记录误差；以下FAIL原样保留。其非阻断处理来自随后明确授权和已核验的基线继承关系，不表示容差放宽或数值问题已消失。','']
    lines+=_table(['臂','诊断','原状态','阻断','越阈/元素','最大绝对差','RMS','atol','rtol'],[
        [r.get('arm'),r.get('site'),r.get('status'),r.get('blocking'),f"{r.get('violating_elements','NOT_RECORDED')}/{r.get('elements','NOT_RECORDED')}",r.get('max_abs'),r.get('RMS'),r.get('atol'),r.get('rtol')]
        for r in gate_policy['recorded_numerical_failures']])
    evidence=gate_policy.get('baseline_regression_evidence') or {}
    if evidence:
        lines += ['',f"原版基线回归证据：{evidence.get('status','RECORDED')}；`{evidence.get('results_path','见gates.json')}`。checkpoint、输入、代码与速度参考身份核对保留在results.json，不将历史不同诊断状态冒称同一case。"]
    lines+=['','先前与补充诊断单独记账，不计入本次1500更新、64正式/4重放预算。局部first-block或其他算子调用不换算成完整模型前向。','']
    if diagnostic_history.get('entries'):
        lines+=_table(['诊断记录','完整前向返回/尝试','反向返回/尝试','更新','完整采样','局部算子调用'],[
            [r.get('label',r.get('name','recorded')),f"{r.get('counts',{}).get('forward_returns',0)}/{r.get('counts',{}).get('forward_attempts',0)}",
             f"{r.get('counts',{}).get('backward_returns',0)}/{r.get('counts',{}).get('backward_attempts',0)}",
             r.get('counts',{}).get('optimizer_updates',0),r.get('counts',{}).get('complete_rollouts',0),
             json.dumps(r.get('partial_operator_calls',{}),ensure_ascii=False,sort_keys=True)] for r in diagnostic_history['entries']])
    else:
        lines += ['补充诊断账本NOT_RECORDED；没有把未提供的记录静默计为零。']
    lines+=['','### FP32与第一层补充诊断（已有证据）','']
    if precision_summary['status']=='RECORDED_DIAGNOSTIC_ONLY':
        lines += ['以下只读取已验SHA的旧诊断结果，不新增模型调用。BF16栏来自保存的原精度结果；FP32栏是独立诊断，生产主干仍使用BF16。','']
        lines+=_table(['置换','BF16 max abs','BF16 RMS','原BF16 allclose','FP32 max abs','FP32 RMS','FP32诊断allclose'],[
            [row['permutation'],row['BF16']['max_abs'],row['BF16']['RMS'],row['BF16']['allclose'],
             row['FP32']['max_abs'],row['FP32']['RMS'],row['FP32']['FP32_diagnostic_allclose_1e5']]
            for row in precision_summary['comparisons']])
        first=precision_summary.get('first_nonbitwise_record') or {}
        lines += ['',
            f"第一层最早被观测到的非逐位一致点：`{precision_summary.get('first_nonbitwise_site','NOT_RECORDED')}`；max abs={_fmt(first.get('max_abs'))}，RMS={_fmt(first.get('RMS'))}。此前记录的池化、归一化、QKV及Q/K归一化逐位一致。",
            f"两次独立第一层重放均逐位复现各自完整前向trace：{precision_summary.get('both_standalone_blocks_match_original_trace_bitwise')}。尚未定位SDPA内部更细的矩阵乘、softmax或归约步骤，不能声称已确定唯一内部算子根因。",
            'FP32诊断判据为 `|a-b| <= 1e-5 + 1e-5*|b|`；通过该联合判据不等于逐位一致，也不会把原BF16容差FAIL改为PASS。四次FP32完整前向、两次BF16第一层及三次embedding独立调用已在上方历史账本单列，不计为新增训练或采样。',
            f"源结果：`{precision_summary['source']['path']}`；SHA256 `{precision_summary['source']['sha256']}`。"]
    else:
        lines += [f"NOT_RECORDED：{precision_summary.get('reason','缺少已验证精度诊断记录')}。未据缺失记录推断任何数值根因。"]
    lines+=['','本次执行阶段账本：','']
    lines+=_table(['阶段','前向返回/尝试','反向返回/尝试','优化器更新','完整采样','OT'],[
        [r['phase'],f"{r['forward_returns']}/{r['forward_attempts']}",f"{r['backward_returns']}/{r['backward_attempts']}",r['optimizer_updates'],r['complete_rollouts'],r['OT_calls']] for r in counts['phases']])
    total=counts['total'];lines+=['',f"本次新run合计：前向{total['forward_returns']}/{total['forward_attempts']}，反向{total['backward_returns']}/{total['backward_attempts']}，更新{total['optimizer_updates']}，完整采样{total['complete_rollouts']}（正式64，其余重放），Euler更新{total['euler_updates']}。",'',
        '## 评价输入与解释范围','',
        '新片仅按原始NPZ源面ID顺序进行共享整边BFS；保留raw→canonical映射。没有依据模型输出挑片。每片均有自己的100面补集与固定检查点。9701–9704在本地可检索历史元数据中未发现生成用途；9702的旧匹配是计数，不是噪声使用记录。','']
    lines+=_table(['条件','原片重合面','新增面','片面积/完整面积','重合面积/片面积','边界边数','边界点数'],[
        [r['case_id'],r['original_overlap_faces'],r['new_face_count'],r['patch_area_fraction'],r['overlap_area_fraction_of_patch'],r['trusted_source_boundary_edges'],r['boundary_point_count']] for r in evaluation['cases']])
    lines+=['','表面误差单位均为完整目标bbox对角线的百分数；排除C后双向surface RMS使用原512点口径。固定目标覆盖阈值1%L；Gamma包含C所有incidence-one边，以弧长加权，覆盖阈值0.1%L。Gamma最大值/分位数来自固定采样点，不能视作精确Hausdorff。',
        '## 自由表面结果','',
        '“输出中位”是单模型输出误差的中位数；下表“配对差中位”是同父对象/片/噪声下逐样本相减后的中位数。两者不混用。','']
    for group,title in (('original','原固定片'),('new','新上下文片')):
        lines+=['### '+title,'']
        groups=[r for r in summary['distributions'] if r['patch_group']==group and r['parent'] is not None and r['patch_id'] is None]
        lines+=_table(['对象','模型','n','RMS均值','输出中位','最差','预测→目标均值','目标→预测均值','固定目标覆盖'],[
            [r['parent'],r['model'],r['sample_count'],r['metrics']['free_rms_pct']['mean'],r['metrics']['free_rms_pct']['median'],r['metrics']['free_rms_pct']['worst'],r['metrics']['pred_to_target_pct']['mean'],r['metrics']['target_to_pred_pct']['mean'],r['metrics']['fixed_target_coverage']['mean']] for r in groups])
        lines+=['']
        pair_rows=[]
        for p in (0,1):
            for newer,ref in COMPARISONS:
                r=_comparison(summary,newer,ref,group,parent=p);s=r['statistics']
                pair_rows.append([p,newer+'−'+ref,s['mean'],s['median'],s['worst'],f"{r['improved']}/{s['n']}"])
        lines+=_table(['对象','预定比较','配对差均值','配对差中位','最差配对差','改善数'],pair_rows)+['']
    lines+=['新片分别保留如下，不能仅报告合并均值：','']
    groups=[r for r in summary['distributions'] if r['patch_group']=='new' and r['patch_id'] is not None]
    lines+=_table(['对象','新片','模型','n','RMS均值','输出中位','最差'],[[r['parent'],r['patch_id'],r['model'],r['sample_count'],*[r['metrics']['free_rms_pct'][k] for k in ('mean','median','worst')]] for r in groups])
    lines+=['','新片逐块配对差：','']
    detail_rows=[]
    for p in (0,1):
        for j in (1,2):
            for newer,ref in COMPARISONS:
                r=_comparison(summary,newer,ref,'new',parent=p,patch_id=j);stats=r['statistics']
                detail_rows.append([p,j,newer+'−'+ref,stats['mean'],stats['median'],stats['worst'],f"{r['improved']}/{stats['n']}"])
    lines+=_table(['对象','新片','预定比较','配对差均值','配对差中位','最差配对差','改善数'],detail_rows)
    lines+=['','## 接口与质量','']
    groups=[r for r in summary['distributions'] if r['parent'] is not None and r['patch_id'] is None]
    lines+=_table(['片组','对象','模型','Gamma RMS','Gamma p95','Gamma覆盖','窄长面比例','退化面均数','CF穿刺free比例','FF非法对均数'],[
        [r['patch_group'],r['parent'],r['model'],*[r['metrics'][k]['mean'] for k in ('gamma_rms_pct','gamma_p95_pct','gamma_coverage','narrow_q_gt10_fraction','near_degenerate_count','CF_proper_affected_free_fraction','FF_illegal_pair_count')]] for r in groups])
    lines+=['',f"独立严格审计状态：{strict['status']}。CF穿刺比例仅统计proper穿刺；coplanar接触/重叠及不确定/退化候选另列于JSON/CSV。FF非法对是verified proper+coplanar，未将全部候选视为已证实穿刺。", 
        '完整拓扑若执行，保留C继承link异常与新增free相关异常的独立字段。当前绕序冲突不直接等于不可重定向；精确谓词认证与连续碰撞安全NOT_RUN。C恒等保持来自clamping，不是新增编码器学习成绩。','',
        '## 同噪声跨上下文绑定','',
        '四格行目标为各自patch的自由补集，列为P0/P1上下文；仅在同模型、同seed、同patch index内配对。B>0表示该固定配对方向有条件绑定收益。','']
    binding_rows=[]
    for model in MODELS:
        for j in (0,1,2):
            values=[r for r in generation['binding'] if r['model_name']==model and r['patch_index']==j]
            s=_stats([r['B'] for r in values],True)
            binding_rows.append([model,j,s['n'],s['mean'],s['median'],s['worst'],sum(r['both_directions_positive'] for r in values)])
    lines+=_table(['模型','patch配对','n','B均值','B中位','最差B','双向均正数'],binding_rows)
    lines+=['','## 新分支学习与舍入观测','',
        '下表RMS是各微批known元素RMS的均值；吞没比例按所有非零delta元素加权。部分舍入不能单独解释学习成败，非零delta也不等于有效条件收益。','']
    lines+=_table(['臂','步','delta RMS','有效变化RMS','有效−原delta RMS','吞没比例','free直接为0'],[
        [r['arm'],r['step'],r['delta_RMS_known']['mean'],r['effective_RMS_known']['mean'],r['effective_minus_delta_RMS_known']['mean'],'N/A (zero delta)' if r['nonzero_delta_elements']==0 else r['swallowed_nonzero_fraction'],r['free_direct_delta_exact_zero']] for r in rounding])
    gradient_rows=[]
    for r in rounding:
        for name,v in r['gradients_and_updates'].items():
            gradient_rows.append([r['arm'],r['step'],name,v['gradient_RMS_after_global_clip'],v['actual_update_RMS'],v['nonzero_gradient_elements'],v['nonzero_update_elements']])
    lines+=['']+_table(['臂','步','参数组','裁剪后梯度RMS','实际更新RMS','非零梯度数','非零更新数'],gradient_rows)
    lines+=['','## 资源与时间','',
        '采样时间包含原51状态CPU拷贝、同步及逐步记录；NPZ压缩写盘单列。条件分支wall/CUDA-event计时覆盖C提取检查、几何、图和FP32 MLP，事件时间包含期间同步空隙，不能解释为纯算子内核耗时。条件分支开销已包含训练/采样总时间，不能再相加。','']
    timing_rows=[]
    for arm,r in training.items():
        t=r['timing'];timing_rows.append([arm,r.get('seconds'),t['input_OT_seconds'],t['training_step_seconds'],t['checkpoint_seconds'],t['context_cuda_seconds'],t['context_wall_seconds'],r.get('peak_cuda_allocated')])
    lines+=_table(['训练臂','总秒','输入OT秒','训练步秒','写权重秒','分支event秒','分支wall秒','峰值bytes'],timing_rows)
    t=generation['timing'];lines+=['',f"正式生成：采样{t['sampling_seconds']:.3f}s；轻量几何评分{t['light_geometry_seconds']:.3f}s；NPZ写入{t['archive_write_seconds']:.3f}s；模型读取{t['model_load_seconds']:.3f}s；分支event/wall={t['context_cuda_seconds']:.3f}/{t['context_wall_seconds']:.3f}s。重放时间在各sampling记录中保留，未混入正式采样合计。",
        f"独立严格审计：total={_fmt(strict.get('total_seconds'))}s，endpoint={_fmt(strict.get('endpoint_seconds'))}s，input={_fmt(strict.get('patch_input_seconds'))}s；判交/索引账本保留在results.json。",'',
        '## 三个固定终点','']
    for arm in MODELS[1:]:
        cp=training[arm]['checkpoints'][-1]
        lines+=[f"- {arm}：`{cp['path']}`；SHA256 `{cp['sha256']}`；state `{cp['model_state_sha256']}`；累计{cp['cumulative_updates']}步。"]
    lines+=['','## CPU验证记录','', '下列套件可能重叠，不能相加声称独立测试数。','']
    lines+=_table(['记录','状态','测试/家族数','秒'],[[r['name'],r['status'],r['tests'],r['seconds']] for r in cpu_tests])
    lines+=['','## 固定首噪声图','',
        '图中仅固定seed9701；各图同一视角，同一父对象所有模型/full/free共用坐标范围。显示坐标顺序x,z,y使原y轴竖直。红色为已知C，蓝色为free。PNG完整性已核验，视觉验收由调用方执行。','']
    for figure in figures:
        lines += [f"![patch{figure['patch_id']}](comparison_patch{figure['patch_id']}_seed9701.png)",'']
    lines+=['代码diff单独保存在本轮交付目录。results.json保留完整逐输出轻量指标、四格绑定、配对差数组、调用账本及独立审计标量；compact_per_output.csv为逐输出紧凑表。未复制官方权重或历史数据包。','']
    reproduce_path=out/'reproduce.md'
    if reproduce_path.is_file():
        reproduction=reproduce_path.read_text(encoding='utf-8-sig').splitlines()
        if reproduction and reproduction[0].startswith('# '):reproduction=reproduction[1:]
        reproduction=[('#'+line if line.startswith('#') else line) for line in reproduction]
        lines+=['','## 附录：复现入口与门禁边界','']+reproduction+['']
        results['reproduction_notes']=dict(status='INCLUDED_IN_MAIN_REPORT',source=_record(reproduce_path))
    else:
        lines+=['','## 附录：复现入口与门禁边界','','NOT_RECORDED：本run未提供reproduce.md。','']
        results['reproduction_notes']=dict(status='NOT_RECORDED')
    (out/'GEOMETRY_CONTEXT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    with (out/'compact_per_output.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows([{k:('NOT_RUN' if v is None else v) for k,v in row.items()} for row in rows])
    results['reporter']['seconds']=time.perf_counter()-tick
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return results