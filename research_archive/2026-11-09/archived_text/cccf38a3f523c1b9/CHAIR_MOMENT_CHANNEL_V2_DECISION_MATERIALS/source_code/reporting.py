"""CPU-only registered reductions for the two-arm V2 learning experiment."""
from __future__ import annotations

import argparse
import importlib.util
import statistics
from pathlib import Path

from .scoring import ROOT, CHAIR, OUT, LABELS, ARMS, DEFINITION, records, read_json, write_json, write_csv, sha, missing_width
from experiments.mf_h1_local0.chair_surface_attention_v1 import reporting as historical

COMPARISONS=(('H_CONT_END','MOMENT_XY_END'),('START','H_CONT_END'),('START','MOMENT_XY_END'))
_source=ROOT/'experiments/mf_h1_local0/chair_hybrid_coupling_v1/reporting.py'
_spec=importlib.util.spec_from_file_location('experiments.mf_h1_local0.chair_hybrid_coupling_v1._moment_v2_reductions',_source)
helper=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(helper)
helper.LABELS=LABELS;helper.COMPARISONS=COMPARISONS
METRICS=dict(historical.METRICS)
# Matching the GT angle distribution is descriptive, not a mesh-quality gate.
METRICS['small_angle_excess_over_same_GT']=dict(direction='descriptive',unit='fraction')
METRICS['small_angle_abs_gap_to_same_GT']=dict(direction='descriptive',unit='fraction')
WIDTH_METRICS=dict(helper.WIDTH_METRICS)
KEYS=dict(per_output=('case_id',),width_per_output=('uid','noise_index','candidate','alpha'),
    width_per_curve=('uid','noise_index','candidate'),cross_matrix=('uid','noise_index','candidate','alpha','target_beta'),
    target_column_audit=('uid','noise_index','candidate','target_beta'))


def placeholders(label,registered,reason):
    result=missing_width(label,registered,reason);result['per_output']=[]
    for rec in registered:
        row={k:rec.get(k) for k in ('case_id','table','uid','split','ratio','alpha','noise_index','task_id','patch_id','input_sha256')}
        row.update(checkpoint_label=label,status='BLOCKED' if rec['status']=='READY' else 'NOT_APPLICABLE',
            reason=reason if rec['status']=='READY' else rec.get('reason','Original registered NA'))
        result['per_output'].append(row)
    return result


def load_tables(out,registered,gate):
    result={k:[] for k in KEYS};availability={}
    for label in LABELS:
        folder=out/'evaluation'/label/'scoring'
        summary=folder/'summary.json'
        availability[label]=read_json(summary) if summary.exists() else dict(status='NOT_RUN')
        defaults=placeholders(label,registered,gate.get('status','NOT_RUN')+'; registered score unavailable')
        for name,fields in KEYS.items():
            path=folder/(name+'.json');saved=read_json(path) if path.exists() else []
            index={tuple(r.get(k) for k in fields):r for r in saved}
            allowed={tuple(r.get(k) for k in fields) for r in defaults[name]}
            if len(index)!=len(saved) or not set(index).issubset(allowed):raise ValueError('Unregistered or duplicate score rows: '+label+'/'+name)
            full=[index.get(tuple(r.get(k) for k in fields),r) for r in defaults[name]]
            if any(r['checkpoint_label']!=label for r in full):raise ValueError('Checkpoint label mismatch')
            result[name]+=full
    return result,availability


def _summary(pairs,fields,metrics):
    rows=helper.pair_summary(pairs,fields,metrics)
    for row in rows:
        if row['valid_pairs']==0:
            for key in ('improved_count','worsened_count','tied_count'):row[key]=None
    return rows


def primary_results(cases,parents):
    subset=[r for r in parents if r['table']=='A' and r['split']=='validation' and r['ratio']==.2]
    pairs=helper.pair_rows(subset,['table','split','ratio','uid'],['fixed_query_MSE'])
    summaries=_summary(pairs,['comparison'],{'fixed_query_MSE':'lower'})
    for row in summaries:
        complete=row['requested_pairs']==row['valid_pairs']==8
        row.update(registered_parent_denominator=8,complete_8_parents=complete,
            geometry_improvement_candidate=bool(row['delta_mean']<0 and row['paired_delta_median']<0 and row['improved_count']>=6) if complete else None)
    cases=[r for r in cases if r['table']=='A' and r['split']=='validation' and r['ratio']==.2]
    noise=helper.pair_rows(cases,['table','split','ratio','uid','case_id','noise_index'],['fixed_query_MSE'])
    return pairs,summaries,noise,_summary(noise,['comparison','noise_index'],{'fixed_query_MSE':'lower'})


def control_counts(curves,columns):
    rows=[]
    for label in LABELS:
        for split in ('train','validation'):
            for candidate in ('MODEL','No-response','Post-scale'):
                for noise in ('all',0,1):
                    def matches(r):
                        return r['checkpoint_label']==label and r['split']==split and r['candidate']==candidate and (noise=='all' or r['noise_index']==noise)
                    selected=[r for r in curves if matches(r)];cols=[r for r in columns if matches(r)]
                    good=[r for r in selected if r['status']=='PASS'];cg=[r for r in cols if r['status']=='PASS']
                    denominator=8 if noise=='all' else 4
                    if len(selected)!=denominator or len(cols)!=3*denominator:raise ValueError('Original B registered denominator changed')
                    complete=len(good)==denominator and len(cg)==3*denominator
                    row=dict(checkpoint_label=label,split=split,candidate=candidate,noise_index=noise,
                        status='COMPLETE' if complete else 'BLOCKED',registered_curves=denominator,valid_curves=len(good),
                        registered_columns=3*denominator,valid_columns=len(cg),
                        strict_column_win_count=sum(bool(r['column_strict_win']) for r in cg) if complete else None,
                        all_three_strict_win_curve_count=sum(bool(r['all_three_strict_win']) for r in good) if complete else None,
                        one_input_strictly_wins_all_targets_curve_count=sum(bool(r['one_input_strictly_wins_all_targets']) for r in good) if complete else None,
                        mean_A_column=helper.mean(r.get('A_column') for r in cg) if complete else None)
                    for metric in WIDTH_METRICS:row[metric+'_curve_mean']=helper.mean(r.get(metric) for r in good) if complete else None
                    for beta in (.9,1.,1.1):
                        vals=[r for r in cg if r['target_beta']==beta]
                        row['beta_'+str(beta)+'_strict_win_count']=sum(bool(r['column_strict_win']) for r in vals) if complete else None
                        row['beta_'+str(beta)+'_mean_A_column']=helper.mean(r.get('A_column') for r in vals) if complete else None
                    rows.append(row)
    return rows


def external_facts(out):
    protocol=read_json(out/'protocol.json') if (out/'protocol.json').exists() else {}
    paths=dict(precheck='precheck/result.json',initial_replay='initial_replay/result.json',
        numerical_diagnostics='precheck/numerical_diagnostics.json',cost='precheck/cost_results.json')
    paths.update(protocol.get('report_inputs',{}));facts={}
    for key,value in paths.items():
        path=Path(value);path=path if path.is_absolute() else out/path
        facts[key]=dict(path=str(path),sha256=sha(path),data=read_json(path)) if path.exists() else dict(path=str(path),data=dict(status='NOT_RUN'))
    return facts


def status_facts(out,gate,availability):
    stages={p.stem:read_json(p) for p in (out/'stages').glob('*.json')};arms={}
    for arm in ARMS:
        stage=stages.get('train_'+arm,{})
        endpoints=[read_json(p) for p in (out/'checkpoints').glob(arm+'*1000*.json')]
        endpoints=[r for r in endpoints if r.get('metadata',{}).get('moment_step')==1000 and r.get('metadata',{}).get('conditional_total_step')==6000]
        arms[arm]=dict(stage_status=stage.get('status','NOT_RUN'),fixed_endpoint_present=bool(endpoints),
            status='DONE' if stage.get('status')=='DONE' and endpoints else 'BLOCKED' if stage.get('status')=='BLOCKED' else 'NOT_RUN',
            endpoint=endpoints[0] if endpoints else None)
    trained=all(r['status']=='DONE' for r in arms.values())
    evaluated=all(availability[label].get('status')=='COMPLETE' for label in LABELS)
    return dict(IMPLEMENTED='DONE' if gate.get('status')=='PASS' else gate.get('status','NOT_RUN'),
        TRAINED='DONE' if trained else 'INCOMPLETE' if any(r['stage_status']!='NOT_RUN' for r in arms.values()) else 'NOT_RUN',
        EVALUATED='DONE' if evaluated else 'INCOMPLETE' if any(r.get('status')!='NOT_RUN' for r in availability.values()) else 'NOT_RUN',
        arms=arms,raw_stage_status=stages)


def decision(primary,noise,quality,width,width_noise,counts,gate,status):
    by={r['comparison']:r for r in primary}
    complete=status['TRAINED']=='DONE' and status['EVALUATED']=='DONE' and len(by)==3 and all(r['complete_8_parents'] for r in primary)
    base=dict(status='INCOMPLETE',QUALITY_RESULT='NA_NOT_EVALUATED',geometry_signal=None,control_signal=None,
        primary_noise_consistent=None,control_noise_consistent=None,tradeoffs=[],retention=None,
        rule_note='Geometry uses the registered mean/median/6-of-8 rule. Control is a descriptive concordance screen, not a significance test or permission for quality loss.')
    if gate.get('status')!='PASS' or not complete:
        base.update(status=gate.get('status','NOT_RUN') if gate.get('status')!='PASS' else 'EVALUATION_INCOMPLETE',
            recommendation='先完成本轮已授权且尚缺失的门禁或固定终点评估。未训练或缺失的三比较保持 NA；不据此判断方法质量。',
            test_hypothesis='只有完整固定面板及配对证据，才能检验关系通道是否超过等预算续训并保留共同起点质量。')
        return base
    comparison='MOMENT_XY_END-H_CONT_END';geom=by[comparison]['geometry_improvement_candidate']
    selected=[r for r in width if r['comparison']==comparison and r['candidate']=='MODEL' and r['split']=='validation' and r['metric'] in ('b_self_abs_error','b_GT_abs_error')]
    slopes=len(selected)==2 and all(helper.usable(r['delta_mean']) and r['delta_mean']<=0 for r in selected)
    def counter(label,noise):return next(r for r in counts if r['checkpoint_label']==label and r['candidate']=='MODEL' and r['split']=='validation' and r['noise_index']==noise)
    def concordant(noise,require_gain=False):
        b=counter('H_CONT_END',noise);a=counter('MOMENT_XY_END',noise)
        if b['status']!='COMPLETE' or a['status']!='COMPLETE':return False
        keys=('strict_column_win_count','all_three_strict_win_curve_count')
        okay=all(a[k]>=b[k] for k in keys) and a['one_input_strictly_wins_all_targets_curve_count']<=b['one_input_strictly_wins_all_targets_curve_count']
        return okay and (not require_gain or any(a[k]>b[k] for k in keys))
    wn=[r for r in width_noise if r['comparison']==comparison and r['candidate']=='MODEL' and r['split']=='validation' and r['metric'] in ('b_self_abs_error','b_GT_abs_error')]
    control_noise=len(wn)==4 and all(helper.usable(r['delta_mean']) and r['delta_mean']<=0 for r in wn) and concordant(0) and concordant(1)
    control=slopes and concordant('all',True) and control_noise
    primary_noise=[r for r in noise if r['comparison']==comparison]
    noise_ok=len(primary_noise)==2 and all(helper.usable(r['delta_mean']) and r['delta_mean']<0 for r in primary_noise)
    guards=('F_to_GT_original_RMS_L','fixed_query_p95_L','coverage_fraction','Gamma_to_free_edges_RMS_bbox_pct',
        'Gamma_to_free_edges_coverage_0.001_fraction','CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction',
        'CF_unresolved_or_degenerate_pair_count','FF_unresolved_or_degenerate_pair_count','free_near_degenerate_fraction')
    losses=[r for r in quality if r['comparison']==comparison and r['metric'] in guards and helper.usable(r['delta_mean']) and r['delta_mean']*(1 if r['direction']=='lower' else -1)>0]
    retention=by['MOMENT_XY_END-START']
    regression=retention['delta_mean']>0
    adverse_control=not slopes or not concordant('all')
    tradeoff=bool(losses or regression or (geom and adverse_control))
    result='SIGNAL_WITH_TRADEOFF' if (geom or control) and tradeoff else 'JOINT_POSITIVE_PILOT_SIGNAL' if geom and control else 'RELATION_CHANNEL_GEOMETRY_SIGNAL' if geom else 'RELATION_CHANNEL_CONTROL_SIGNAL' if control else 'NO_CLEAR_LEARNING_SIGNAL'
    recommendation='接受当前有限预算内未见清晰学习收益；下一步优先离线定位逐父失败模式，检验是否存在一致、可复现的关系通道受益子集。'
    hypothesis='若通道有稳定收益，同一父的两噪声应出现一致的主终点与正确目标列改善；否则当前证据支持暂不扩大训练。'
    if geom or control:
        recommendation='下一步优先独立复现本轮配对信号，并预先保留已经暴露的覆盖尾部、接口、相交及起点保真代价作为守门项。'
        hypothesis='若收益来自可学习关系通道，独立训练随机流应重复超过 H_CONT 的配对信号，并保留 START 质量；仅击败退化续训对照不足。'
    base.update(status='COMPLETE',QUALITY_RESULT=result,geometry_signal=bool(geom),control_signal=bool(control),
        primary_noise_consistent=noise_ok,control_noise_consistent=control_noise,tradeoffs=losses,
        retention=retention,relative_START_primary_regression=regression,adverse_control=adverse_control,
        recommendation=recommendation,test_hypothesis=hypothesis)
    return base


def cost_rows(out,status):
    per_output=[];summary=[]
    for label in LABELS:
        path=out/'evaluation'/label/'generated_rows.json';raw=read_json(path) if path.exists() else {}
        items=raw if isinstance(raw,list) else raw.get('rows',[])
        for row in items:
            meta=row.get('metadata',row.get('audit',{}))
            per_output.append(dict(checkpoint_label=label,case_id=row['case_id'],status=row['status'],
                sampling_seconds=meta.get('wall_seconds',meta.get('elapsed_seconds',row.get('sampling_seconds'))),
                model_calls=row.get('model_calls'),peak_cuda_allocated_bytes=meta.get('peak_cuda_allocated_bytes'),
                peak_cuda_reserved_bytes=meta.get('peak_cuda_reserved_bytes'),metadata=meta))
        arm=label.removesuffix('_END');stage=status['raw_stage_status'].get('train_'+arm,{})
        training_path=out/'training'/(arm+'_summary.json');training=read_json(training_path) if training_path.exists() else {}
        measured=[r['sampling_seconds'] for r in per_output if r['checkpoint_label']==label and r['status']=='SAMPLED' and helper.usable(r['sampling_seconds'])]
        summary.append(dict(checkpoint_label=label,sampled=sum(r['status']=='SAMPLED' for r in items),cached=sum(r['status']=='CACHED' for r in items),
            NA=sum(r['status']=='NOT_APPLICABLE' for r in items),sampling_timing_count=len(measured),
            sampling_seconds_mean=helper.mean(measured),sampling_seconds_median=statistics.median(measured) if measured else None,
            training_summary=training,stage=stage,timing_scope='Full sampling rows only; historical cache latency is excluded; training summaries retain their measured scope.'))
    return per_output,summary


def finalize(out,source_out=CHAIR,gate=None):
    out=Path(out).resolve();folder=out/'paired_results';folder.mkdir(parents=True,exist_ok=True)
    gate=gate if gate is not None else read_json(out/'gate_results.json') if (out/'gate_results.json').exists() else dict(status='NOT_RUN')
    registered=records(source_out);tables,availability=load_tables(out,registered,gate)
    cases=tables['per_output'];curves=tables['width_per_curve']
    parents=helper.parent_rows(cases,['checkpoint_label','table','split','ratio'],METRICS)
    ac=[r for r in parents if r['table'] in ('A','C')]
    qp=helper.pair_rows(ac,['table','split','ratio','uid'],METRICS);qs=_summary(qp,['comparison','table','split','ratio'],METRICS)
    pp,ps,npairs,ns=primary_results(cases,parents)
    wp=helper.parent_rows(curves,['checkpoint_label','candidate','split'],WIDTH_METRICS)
    wpaired=helper.pair_rows(wp,['candidate','split','uid'],WIDTH_METRICS);ws=_summary(wpaired,['comparison','candidate','split'],WIDTH_METRICS)
    wcp=helper.pair_rows(curves,['candidate','split','uid','noise_index'],WIDTH_METRICS)
    wn=_summary(wcp,['comparison','candidate','split','noise_index'],WIDTH_METRICS)
    cm={'A_column':'higher','column_strict_win':'higher','diagonal_RMS_L0':'lower'}
    cp=helper.parent_rows(tables['target_column_audit'],['checkpoint_label','candidate','split','target_beta'],cm)
    cpaired=helper.pair_rows(cp,['candidate','split','target_beta','uid'],cm)
    cs=_summary(cpaired,['comparison','candidate','split','target_beta'],cm)
    cnp=helper.pair_rows(tables['target_column_audit'],['candidate','split','target_beta','uid','noise_index'],cm)
    cns=_summary(cnp,['comparison','candidate','split','target_beta','noise_index'],cm)
    counts=control_counts(curves,tables['target_column_audit'])
    accounting=read_json(out/'accounting.json') if (out/'accounting.json').exists() else dict(summary={},status='NOT_RUN')
    statuses=status_facts(out,gate,availability);verdict=decision(ps,ns,qs,ws,wn,counts,gate,statuses)
    costs,costsummary=cost_rows(out,statuses)
    exports=dict(per_output=cases,per_parent=parents,paired_per_output=helper.pair_rows(cases,['case_id','table','split','ratio','uid','noise_index','alpha'],METRICS),
        paired_per_parent=qp,quality_paired_summary=qs,primary_paired_parents=pp,primary_summary=ps,primary_noise_pairs=npairs,primary_noise_summary=ns,
        width_per_output=tables['width_per_output'],width_per_curve=curves,width_per_parent=wp,width_paired_parents=wpaired,
        width_paired_summary=ws,width_paired_curves=wcp,width_noise_summary=wn,noise_pairs_per_parent=helper.noise_pairs(curves),
        cross_matrix=tables['cross_matrix'],target_column_audit=tables['target_column_audit'],target_column_per_parent=cp,
        target_column_pairs=cpaired,target_column_summary=cs,target_column_noise_pairs=cnp,target_column_noise_summary=cns,
        control_curve_counts=counts,coverage_partition=helper.coverage_partitions(cases),sampling_cost_per_output=costs,cost_summary=costsummary)
    for name,data in exports.items():write_json(folder/(name+'.json'),data);write_csv(folder/(name+'.csv'),data)
    result=dict(experiment='NATIVE_CHAIR_MOMENT_CHANNEL_LEARNING_V2',stages=statuses,gate_results=gate,availability=availability,
        actual_accounting=accounting,actual_counts=accounting.get('summary',accounting),primary=ps,primary_noise=ns,decision=verdict,
        external=external_facts(out),definition=DEFINITION,source_report_helper_sha256=sha(_source),
        row_counts={k:len(v) for k,v in exports.items()},I_P_extra_forwards=0,default_model_changed=False)
    write_json(folder/'results.json',result);write_json(out/'results/summary.json',result)
    from .report import render
    render(out,result)
    write_json(folder/'artifact_manifest.json',[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.glob('*')) if p.is_file() and p.name!='artifact_manifest.json'])
    return result


def cpu_tests(out):
    """Schema/fixed-denominator and non-cancelling paired arithmetic tests."""
    out=Path(out);fixture=out/'tests/reporting_blocked_fixture';fixture.mkdir(parents=True,exist_ok=True)
    result=finalize(fixture,gate=dict(status='IMPLEMENTATION_BLOCKED',reason='Synthetic missing-output schema test; no model calls'))
    assert result['row_counts']['per_output']==432 and result['row_counts']['cross_matrix']==1296
    assert all(r['valid_pairs']==0 and r['requested_pairs']==8 and r['delta_mean'] is None for r in result['primary'])
    assert result['decision']['QUALITY_RESULT']=='NA_NOT_EVALUATED'
    synthetic=[]
    for label,offset in (('START',0.),('H_CONT_END',.2),('MOMENT_XY_END',-.1)):
        for parent in range(8):
            for noise in (0,1):synthetic.append(dict(checkpoint_label=label,uid=str(parent),case_id=str(parent)+'_'+str(noise),noise_index=noise,
                table='A',split='validation',ratio=.2,status='PASS',fixed_query_MSE=1+parent+noise+offset))
    parents=helper.parent_rows(synthetic,['checkpoint_label','table','split','ratio'],['fixed_query_MSE'])
    _,ps,_,_=primary_results(synthetic,parents)
    main=next(r for r in ps if r['comparison']=='MOMENT_XY_END-H_CONT_END')
    assert abs(main['delta_mean']+.3)<1e-12 and main['improved_count']==8 and main['geometry_improvement_candidate']
    result=dict(status='PASS',blocked_registered_outputs=432,blocked_matrix_cells=1296,all_three_missing_comparisons_NA=True,
        paired_parent_noise_arithmetic=True,synthetic_geometry_delta=main['delta_mean'],model_calls=0,
        old_module_globals_unchanged=historical.LABELS==('START','H_CONT','MOMENT_XY','SURFACE_XY'))
    assert result['old_module_globals_unchanged'];write_json(out/'tests/v2_reporting_cpu_tests.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));p.add_argument('--cpu-tests',action='store_true');a=p.parse_args()
    print(cpu_tests(a.out)['status'] if a.cpu_tests else finalize(a.out)['decision']['QUALITY_RESULT'])
