"""Bounded final-velocity / injection pilot evaluation, with no training.

Only raw NPZ files are created. The caller persists the returned evidence and
freezes the pure selection result before invoking confirmation generation.
"""
from __future__ import annotations
from .stable_runtime import stable_execution, assert_stable_runtime

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import re
import time
import traceback
import numpy as np
import torch

from .inference_guidance import SETTINGS, EXPECTED_NFE, alpha_table, guided_euler_sample
from .nxc112_evaluation import (_cases, _fingerprint, _file_hash, _assert_base,
    _count_base, _counts, _save_raw, _score, _bindings, _load_base_results,
    _rng_snapshot, _rng_equal, array_hash, EXPECTED_BASE, TEACHER_TIMES)
from .official_wrapper import state_sha256
from .prompt_dataset import collate_prompts, collate_samples
from .sample import euler_sample

DEVELOPMENT_SEEDS = (7101,7102,7103,7104)
CONFIRMATION_SEEDS = tuple(range(8101,8117))


def _distribution(values):
    values=np.asarray(list(values),dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        return dict(count=len(values),finite=False,mean=None,median=None,q25=None,q75=None,min=None,max=None)
    return dict(count=len(values),finite=True,mean=float(values.mean()),median=float(np.median(values)),
                q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),
                min=float(values.min()),max=float(values.max()))


def summarize_outputs(outputs, bindings):
    """Saved-score arithmetic only, including cancellation-safe contribution."""
    parents=[]
    is_base=all(r['input_condition'] is None for r in outputs)
    for parent in range(2):
        selected=[r for r in outputs if is_base or r['input_condition']==parent]
        parents.append(dict(parent=parent,
            guide=_distribution(r['guide_bbox_pct'][parent] for r in selected),
            completion=_distribution(r['surface_bbox_pct'][parent] for r in selected),
            narrow=_distribution(r['quality']['min_angle_lt5_fraction'] for r in selected),
            angle_p05=_distribution(r['quality']['angles_p05'] for r in selected),
            degenerate=_distribution(r['quality']['degenerate_fraction'] for r in selected),
            nan=sum(r['quality']['nan_count'] for r in selected),
            inf=sum(r['quality']['inf_count'] for r in selected),
            nearest_own_count=sum(r['nearest_target_common_units']==parent for r in selected)))
    vals=[r['mean_binding_improvement'] for r in bindings]
    net=float(sum(vals));gross=float(sum(abs(v) for v in vals))
    near_zero=abs(net)<=max(1e-8,.1*gross)
    largest=None if not bindings else max(bindings,key=lambda r:abs(r['mean_binding_improvement']))
    contribution=dict(net_binding_sum=net,sum_absolute_binding=gross,
        near_zero_net=near_zero,near_zero_rule='abs(net) <= max(1e-8 bbox pp, 0.1 * sum(abs(B)))',
        largest_absolute_noise=None if largest is None else largest['noise_seed'],
        largest_absolute_binding=None if largest is None else largest['mean_binding_improvement'],
        fraction_of_net=None if largest is None or near_zero else largest['mean_binding_improvement']/net,
        fraction_suppressed=near_zero,
        note='No percentage of a near-cancelled net total; all individual B values remain available.')
    return dict(parents=parents,binding=_distribution(vals),
        direction0=_distribution(r['a_binding_improvement'] for r in bindings),
        direction1=_distribution(r['b_binding_improvement'] for r in bindings),
        both_improve=sum(r['both_improve'] for r in bindings),binding_noise_count=len(bindings),
        contribution=contribution,all_finite=all(p['nan']==p['inf']==0 and
            all(p[k]['finite'] for k in ('guide','completion','narrow','angle_p05','degenerate')) for p in parents),
        independence='Two fixed training parents. Noise draws assess sampling robustness, not new-object generalization.')


def paired_comparison(candidate, reference):
    """Per-parent/noise deltas; accepts a shared Base or conditional reference."""
    rows=reference.get('outputs',reference.get('rows'))
    indexed={(r['noise_seed'],r['input_condition']):r for r in rows}
    result=[]
    for row in candidate['outputs']:
        parent=row['input_condition'];seed=row['noise_seed']
        if parent is None: raise ValueError('Paired comparisons require a conditional candidate')
        other=indexed.get((seed,parent),indexed.get((seed,None)))
        if other is None: raise ValueError('Missing paired reference draw')
        g,bg=row['guide_bbox_pct'][parent],other['guide_bbox_pct'][parent]
        c,bc=row['surface_bbox_pct'][parent],other['surface_bbox_pct'][parent]
        result.append(dict(parent=parent,noise_seed=seed,guide=g,reference_guide=bg,
            guide_improvement=bg-g,guide_improvement_pct=100*(bg-g)/bg if bg else None,
            completion=c,reference_completion=bc,completion_change=c-bc,
            completion_change_pct=100*(c-bc)/bc if bc else None,
            narrow_change_pp=100*(row['quality']['min_angle_lt5_fraction']-other['quality']['min_angle_lt5_fraction']),
            nearest_target_changed=row['nearest_target_common_units']!=other['nearest_target_common_units']))
    summary=[]
    for parent in range(2):
        selected=[r for r in result if r['parent']==parent]
        summary.append(dict(parent=parent,guide_improvement=_distribution(r['guide_improvement'] for r in selected),
            guide_improvement_pct=_distribution(r['guide_improvement_pct'] for r in selected),
            completion_change=_distribution(r['completion_change'] for r in selected),
            completion_change_pct=_distribution(r['completion_change_pct'] for r in selected),
            nearest_target_changed=sum(r['nearest_target_changed'] for r in selected)))
    return dict(rows=result,parents=summary)


def select_candidate(stage_a):
    """The complete preregistered development rule; performs no model/file I/O."""
    if stage_a.get('status')!='PASS' or set(stage_a['settings'])!=set(SETTINGS):
        raise ValueError('Selection requires all six completed development settings')
    if not stage_a['historical_A0_replay']['passed']:
        raise ValueError('Selection requires historical Post identity reproduction')
    base=summarize_outputs(stage_a['base_results']['rows'],[])['parents']
    post=stage_a['settings']['A0'];ps=post['summary']['parents']
    post_j=.5*sum(ps[p]['guide']['mean']/base[p]['guide']['mean'] for p in range(2))
    records=[]
    for name in SETTINGS:
        evaluation=stage_a['settings'][name];summary=evaluation['summary'];parents=summary['parents']
        if len(evaluation['outputs'])!=8 or {r['noise_seed'] for r in evaluation['outputs']}!=set(DEVELOPMENT_SEEDS):
            raise ValueError('Development selection requires exactly two parents x four registered noises')
        j=.5*sum(parents[p]['guide']['mean']/base[p]['guide']['mean'] for p in range(2))
        oldrows={(r['noise_seed'],r['input_condition']):r for r in post['outputs']}
        gates=dict(all_finite=summary['all_finite'],
            degenerate_not_increased=all(r['quality']['degenerate_fraction']<=oldrows[(r['noise_seed'],r['input_condition'])]['quality']['degenerate_fraction'] for r in evaluation['outputs']),
            both_parent_mean_guides_better_base=all(parents[p]['guide']['mean']<base[p]['guide']['mean'] for p in range(2)),
            both_direction_binding_at_least3of4=summary['both_improve']>=3,
            binding_median_positive=summary['binding']['median']>0,
            each_parent_completion_worsening_at_most5pct=all(parents[p]['completion']['mean']<=1.05*ps[p]['completion']['mean'] for p in range(2)),
            each_parent_narrow_increase_at_most2pp=all(parents[p]['narrow']['mean']<=ps[p]['narrow']['mean']+.02 for p in range(2)))
        records.append(dict(setting=name,J=j,nfe_per_trajectory=EXPECTED_NFE[name],gates=gates,
            passes_all_protection_lines=all(gates.values()),better_J_than_original_post=j<post_j,
            eligible=name!='A0' and all(gates.values()) and j<post_j,
            summary=summary,alpha=alpha_table(name)))
    eligible=[r for r in records if r['eligible']]
    eligible.sort(key=lambda r:(r['J'],r['nfe_per_trajectory'],SETTINGS.index(r['setting'])))
    selected=eligible[0] if eligible else None
    return dict(selected_setting=None if selected is None else selected['setting'],
        selected_parameters=None if selected is None else dict(alpha=selected['alpha'],internal_control_scale=1.,steps=50,N=112),
        candidates=records,post_J=post_j,selected_J=None if selected is None else selected['J'],
        rule='All protection lines, strictly lower J than A0; then minimum J, exact-J tie smaller NFE, then A1..A5 order.',
        degenerate_rule='Every paired output must not increase its degenerate-face fraction against A0.',
        decision='No inference setting worth promotion' if selected is None else 'One frozen inference candidate',
        source_noise_seeds=list(DEVELOPMENT_SEEDS),confirmation_results_consulted=False)


class InferenceEvaluator:
    """An evaluator owns no controller hooks; root supplies exactly one wrapper."""
    def __init__(self,base,bank,output_dir,old_report,progress_cb=None):
        self.base=base;self.bank=bank;self.output=Path(output_dir).resolve()
        self.output.mkdir(parents=True,exist_ok=True)
        self.old=old_report;self.progress_cb=progress_cb;self.cases=_cases(bank,2)
        if any(c['target'].reshape(-1,9).shape[0]!=112 for c in self.cases):
            raise ValueError('Both parents must retain exactly112 original faces')
        if set(bank['eval_noises'])!=set(DEVELOPMENT_SEEDS):raise ValueError('Changed development noises')
        self.bank_hash=_fingerprint(bank)
        self.historical=next(e for e in self.old['stage_b']['evaluations'] if e['step']==500)
        if (self.old['status']!='PASS_EXECUTION' or self.historical['status']!='PASS'
                or self.historical['bank_fingerprint']!=self.bank_hash
                or self.historical['base_state_sha256_before']!=EXPECTED_BASE):
            raise ValueError('Historical step500 identity or input bank mismatch')
        raw_parent=Path(self.old['stage_b']['base_results']['raw']['path']).parent
        possible_sources=(raw_parent/'results.json',raw_parent.parent/'results.json')
        matching=[p for p in possible_sources if p.is_file() and json.loads(p.read_text())==old_report]
        if len(matching)!=1:
            raise ValueError('Cannot identify the unique immutable historical results file')
        self.source_path=matching[0]
        self.source_sha256=_file_hash(self.source_path)

    def _emit(self,report,event,**extra):
        if self.progress_cb is not None:
            self.progress_cb(dict(stage=report['stage'],event=event,model=report.get('model'),
                setting=report.get('setting'),tag=report.get('tag'),counts=dict(report['actual_counts']),**extra))

    def _checked_raw(self,record):
        if _file_hash(record['path'])!=record['sha256']:raise RuntimeError('Historical raw file changed')
        with np.load(record['path'],allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
        if set(arrays)!=set(record['arrays']):raise RuntimeError('Historical raw keys changed')
        for key,value in arrays.items():
            if array_hash(value)!=record['arrays'][key]['sha256'] or not np.isfinite(value).all():
                raise RuntimeError('Historical raw tensor hash/finiteness failed')
        return arrays

    def load_historical_base(self):
        saved=copy.deepcopy(self.old['stage_b']['base_results'])
        _load_base_results(saved,self.bank_hash,EXPECTED_BASE)
        self._checked_raw(saved['raw'])
        saved['reuse']=dict(new_official_forwards=0,new_rollouts=0,historical_independent_draws=4,
            source_results_path=str(self.source_path),source_results_sha256=self.source_sha256,
            scoring='Reuse the archived independent geometric scores; no rescoring')
        return saved

    def historical_post_teacher(self):
        if _file_hash(self.source_path)!=self.source_sha256:raise RuntimeError('Historical results changed')
        value=copy.deepcopy(self.historical['teacher'])
        if tuple(value['times'])!=TEACHER_TIMES or len(value['rows'])!=5:
            raise ValueError('Historical teacher time strata incomplete')
        return dict(status='PASS_REUSED',teacher=value,new_official_forwards=0,new_teacher_examples=0,
            historical_teacher_forwards=15,historical_teacher_examples=30,
            source_results_path=str(self.source_path),source_results_sha256=self.source_sha256,
            adapter_state_sha256=self.historical['adapter_state_sha256_before'],bank_fingerprint=self.bank_hash)

    @contextmanager
    def _session(self,wrapper,report):
        if wrapper is not None and (wrapper.backbone is not self.base or wrapper._context is not None):
            raise ValueError('Evaluator requires this base and its inactive caller-owned wrapper')
        adapter=None if wrapper is None else wrapper.adapter
        modes=None if wrapper is None else (wrapper.training,adapter.training,self.base.training)
        base_before=_assert_base(self.base);adapter_before=None if adapter is None else state_sha256(adapter)
        rng_before=_rng_snapshot();prior=None if wrapper is None else dict(wrapper.counts)
        start=time.perf_counter()
        try:
            if wrapper is not None:wrapper.eval()
            with _count_base(self.base,report['actual_counts']),torch.inference_mode(),stable_execution(self.base,adapter) as facts:
                report['actual_runtime']=facts
                yield
            report['base_state_sha256']=_assert_base(self.base)
            report['adapter_state_sha256']=None if adapter is None else state_sha256(adapter)
            report['bank_unchanged']=_fingerprint(self.bank)==self.bank_hash
            report['rng_unchanged']=_rng_equal(rng_before,_rng_snapshot())
            if (report['base_state_sha256']!=base_before or report['adapter_state_sha256']!=adapter_before
                    or not report['bank_unchanged'] or not report['rng_unchanged']):
                raise RuntimeError('Inference changed weights, fixed inputs, or RNG')
        finally:
            report['wall_seconds']=time.perf_counter()-start
            if wrapper is not None:
                report['wrapper_count_delta']={k:wrapper.counts[k]-prior.get(k,0) for k in wrapper.counts}
                report['actual_counts']['condition_encodes']=report['wrapper_count_delta']['condition_encoding']
                wrapper.train(modes[0]);adapter.train(modes[1]);self.base.train(modes[2])

    def _noise(self,value):
        if not torch.is_tensor(value):value=torch.from_numpy(np.asarray(value))
        if value.shape!=(1,112,9) or value.dtype!=torch.float32 or value.device.type!='cpu':
            raise ValueError('Required archived CPU FP32 noise shape is [1,112,9]')
        return value

    def _report(self,stage,model,setting,tag,seeds):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',tag):raise ValueError('Unsafe output tag')
        path=self.output/(tag+'.npz')
        if path.exists():raise FileExistsError(path)
        counts=_counts();counts['euler_updates']=0
        return dict(stage=stage,model=model,setting=setting,tag=tag,status='RUNNING',N=112,
            noise_seeds=list(seeds),bank_fingerprint=self.bank_hash,actual_counts=counts,
            outputs=[],bindings=[],formal_rollouts=0,diagnostic_rollouts=0),path

    def evaluate(self,wrapper,model,setting,noises,tag,historical_compare=False):
        if setting not in SETTINGS:raise ValueError('Only registered settings are allowed')
        seeds=tuple(sorted(noises))
        if seeds not in (DEVELOPMENT_SEEDS,CONFIRMATION_SEEDS):raise ValueError('Unregistered noise collection')
        reference=None
        if historical_compare:
            if seeds!=DEVELOPMENT_SEEDS or setting!='A0':raise ValueError('Only original development A0 has historical counterpart')
            if state_sha256(wrapper.adapter)!=self.historical['adapter_state_sha256_before']:
                raise ValueError('A0 is not the protected Post step500 checkpoint')
            reference=self._checked_raw(self.historical['raw'])
        report,path=self._report('development' if seeds==DEVELOPMENT_SEEDS else 'confirmation',model,setting,tag,seeds)
        arrays={};counts=report['actual_counts'];before_noise={s:array_hash(self._noise(noises[s])) for s in seeds}
        try:
            with self._session(wrapper,report):
                device=next(self.base.parameters()).device
                for seed in seeds:
                    for parent,case in enumerate(self.cases):
                        source=self._noise(noises[seed]);z=source.to(device).clone()
                        prompts=collate_prompts([case['prompt']],device)
                        start_forward=counts['official_forward_attempts'];counts['sampler_calls']+=1
                        try:
                            value,stats=guided_euler_sample(wrapper,z,prompts,setting)
                            counts['complete_rollouts']+=1;counts['adapted_rollouts']+=1
                            counts['euler_updates']+=stats['euler_updates'];report['formal_rollouts']+=1
                        except BaseException as failure:
                            partial=getattr(failure,'guidance_stats',None)
                            if partial is not None:
                                report['failed_guidance_stats']=partial
                                counts['euler_updates']+=partial['euler_updates']
                                if partial['euler_updates']==50:
                                    counts['complete_rollouts']+=1;counts['adapted_rollouts']+=1
                            raise
                        finally:
                            counts['sampling_official_forwards']+=counts['official_forward_attempts']-start_forward
                        key=f'C{parent}_seed{seed}';tokens=value[0].detach().cpu().numpy();arrays[key]=tokens
                        if stats['status']!='PASS' or not np.isfinite(tokens).all():raise RuntimeError('Nonfinite generation; no repair/resample')
                        row=dict(model=model,setting=setting,N=112,noise_seed=seed,input_condition=parent,raw_key=key,
                            sampling=dict(stats=stats,initial_noise_sha256=before_noise[seed],actual_shape=[1,112,9],y=112,mask_all_true=True),
                            **_score(tokens,self.cases,report))
                        if reference is not None:
                            oldrow=next(r for r in self.historical['outputs'] if r['raw_key']==key)
                            if oldrow['sampling']['initial_noise_sha256']!=before_noise[seed]:
                                raise RuntimeError('Historical A0 initial-noise identity differs')
                            old=reference[key];row['historical_replay']=dict(bitwise_equal=tokens.dtype==old.dtype and np.array_equal(tokens,old),
                                max_abs=float(np.max(np.abs(tokens.astype(np.float64)-old.astype(np.float64)))))
                            if not row['historical_replay']['bitwise_equal']:
                                report['outputs'].append(row);raise RuntimeError('A0 historical full-generation gate failed; stop scan')
                        report['outputs'].append(row)
                        self._emit(report,'trajectory',noise_seed=seed,input_condition=parent)
                report['bindings']=_bindings(report['outputs'],2,(112,),seeds,(model,))
                report['summary']=summarize_outputs(report['outputs'],report['bindings'])
                report['raw']=_save_raw(path,arrays)
            report['noise_unchanged']=all(array_hash(noises[s])==before_noise[s] for s in seeds)
            rolls=2*len(seeds)
            report['expected_counts']=dict(official_forward_attempts=rolls*EXPECTED_NFE[setting],
                official_forward_returns=rolls*EXPECTED_NFE[setting],complete_rollouts=rolls,
                sampler_calls=rolls,adapted_rollouts=rolls,condition_encodes=rolls,euler_updates=50*rolls,
                guide_scores=2*rolls,surface_scores=2*rolls,quality_scores=rolls,optimizer_updates=0)
            if not report['noise_unchanged'] or any(counts[k]!=v for k,v in report['expected_counts'].items()):
                raise RuntimeError('Sampling count or immutable noise check failed')
            report['status']='PASS'
        except BaseException as error:
            report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
            if arrays and not path.exists():report['partial_raw']=_save_raw(path.with_name(path.stem+'_partial.npz'),arrays)
            error.evaluation_report=report;self._emit(report,'failed',error=repr(error));raise
        self._emit(report,'complete')
        return report

    def evaluate_base(self,noises,tag):
        seeds=tuple(sorted(noises))
        if seeds!=CONFIRMATION_SEEDS:raise ValueError('New Base generation is only the sixteen confirmation noises')
        report,path=self._report('confirmation','Base','OFF',tag,seeds)
        arrays={};counts=report['actual_counts'];before={s:array_hash(self._noise(noises[s])) for s in seeds}
        try:
            with self._session(None,report):
                device=next(self.base.parameters()).device
                for seed in seeds:
                    z=self._noise(noises[seed]).to(device).clone()
                    y=torch.tensor([112],device=device,dtype=torch.long);mask=torch.ones((1,112),device=device,dtype=torch.bool)
                    assert_stable_runtime(self.base,coordinates=z)
                    counts['sampler_calls']+=1;prior=counts['official_forward_attempts']
                    try:
                        value,stats=euler_sample(self.base,z,y,mask,precision='bf16')
                        counts['complete_rollouts']+=1;counts['base_rollouts']+=1;counts['euler_updates']+=50;report['formal_rollouts']+=1
                    finally:counts['sampling_official_forwards']+=counts['official_forward_attempts']-prior
                    tokens=value[0].detach().cpu().numpy();key=f'seed{seed}';arrays[key]=tokens
                    if stats['status']!='PASS' or not np.isfinite(tokens).all():raise RuntimeError('Nonfinite Base generation')
                    if array_hash(z)!=before[seed]:raise RuntimeError('Base sampler mutated its initial noise')
                    report['outputs'].append(dict(model='Base',setting='OFF',N=112,noise_seed=seed,input_condition=None,
                        raw_key=key,sampling=dict(stats=stats,initial_noise_sha256=before[seed]),**_score(tokens,self.cases,report)))
                    self._emit(report,'trajectory',noise_seed=seed)
                report['summary']=summarize_outputs(report['outputs'],[]);report['raw']=_save_raw(path,arrays)
            report['expected_counts']=dict(official_forward_attempts=800,official_forward_returns=800,
                complete_rollouts=16,base_rollouts=16,condition_encodes=0,euler_updates=800,guide_scores=32,surface_scores=32,quality_scores=16)
            if any(counts[k]!=v for k,v in report['expected_counts'].items()) or any(array_hash(noises[s])!=before[s] for s in seeds):
                raise RuntimeError('Confirmation Base count/input mismatch')
            report['status']='PASS'
        except BaseException as error:
            report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
            if arrays and not path.exists():report['partial_raw']=_save_raw(path.with_name(path.stem+'_partial.npz'),arrays)
            error.evaluation_report=report;raise
        return report

    def evaluate_stage_a(self,post):
        report=dict(stage='A',status='RUNNING',base_results=self.load_historical_base(),settings={},actual_counts={})
        try:
            for setting in SETTINGS:
                current=self.evaluate(post,'Post',setting,self.bank['eval_noises'],'development_Post_'+setting,historical_compare=setting=='A0')
                report['settings'][setting]=current
                for key,value in current['actual_counts'].items():report['actual_counts'][key]=report['actual_counts'].get(key,0)+value
                if setting=='A0':
                    report['historical_A0_replay']=dict(passed=all(r['historical_replay']['bitwise_equal'] for r in current['outputs']),
                        compared_arrays=8,source_raw=self.historical['raw'],max_abs=max(r['historical_replay']['max_abs'] for r in current['outputs']))
                current['vs_base']=paired_comparison(current,report['base_results'])
                current['vs_original_post']=paired_comparison(current,report['settings']['A0'])
            report['status']='PASS'
            if report['actual_counts']['official_forward_returns']!=4072 or report['actual_counts']['complete_rollouts']!=48:
                raise RuntimeError('Six-setting development budget mismatch')
        except BaseException as error:
            report.update(status='FAIL',error=repr(error))
            if getattr(error,'evaluation_report',None) is not report:report['failed_evaluation']=getattr(error,'evaluation_report',None)
            error.evaluation_report=report;raise
        return report

    def teacher(self,wrapper,model):
        report=dict(stage='teacher',model=model,status='RUNNING',actual_counts=_counts(),rows=[],times=list(TEACHER_TIMES))
        counts=report['actual_counts'];samples=self.bank['teacher_samples']
        if len(samples)!=2 or [s['parent_uid'] for s in samples]!=list(self.bank['parent_uids']):
            raise ValueError('Teacher requires the two archived independent real OT pairs')
        try:
            with self._session(wrapper,report):
                device=next(self.base.parameters()).device
                for timepoint in TEACHER_TIMES:
                    selected=[dict(s,t=timepoint,prompt=self.cases[p]['prompt']) for p,s in enumerate(samples)]
                    batch=collate_samples(selected,device)
                    x1,x0,t,mask=batch['tokens'],batch['noise'],batch['t'],batch['masks']
                    if x1.shape!=(2,112,9) or not mask.all() or not (batch['num_faces']==112).all():
                        raise RuntimeError('Teacher N/shape/mask changed')
                    xt=(1-t[:,None,None])*x0+t[:,None,None]*x1;target=x1-x0
                    before={k:array_hash(v) for k,v in dict(xt=xt,target=target,t=t,mask=mask,x0=x0,x1=x1).items()}
                    swapped=collate_prompts([self.cases[1]['prompt'],self.cases[0]['prompt']],device)
                    predictions={}
                    for mode in ('off','correct','swapped'):
                        assert_stable_runtime(self.base,wrapper.adapter,coordinates=xt)
                        prompts=None if mode=='off' else batch['prompts'] if mode=='correct' else swapped
                        counts['teacher_forward_attempts']+=1
                        value=wrapper(xt,t,y=batch['num_faces'],mask=mask,prompts=prompts,control_scale=1.,inference_geometry_cache=False)
                        counts['teacher_forward_returns']+=1;counts['teacher_examples']+=2
                        if not torch.isfinite(value).all():raise RuntimeError('Nonfinite teacher')
                        predictions[mode]=value.detach().double()
                    errors={m:target.double()-v for m,v in predictions.items()};e=errors['off'];d=predictions['correct']-predictions['off']
                    rows=[]
                    for parent in range(2):
                        valid=mask[parent];denom=int(valid.sum())*9
                        loss={m:float(v[parent][valid].square().sum())/denom for m,v in errors.items()}
                        d2=float(d[parent][valid].square().sum())/denom
                        two_ed=float((2*e[parent][valid]*d[parent][valid]).sum())/denom
                        delta=float((errors['correct'][parent][valid].square()-e[parent][valid].square()).sum())/denom
                        rows.append(dict(parent_index=parent,valid_coordinates=denom,mse=loss,mean_d2=d2,
                            two_mean_ed=two_ed,Lcorrect_minus_Loff=delta,algebra_absolute_error=abs(delta-(d2-two_ed))))
                    row=dict(t=timepoint,per_parent=rows,valid_coordinates=2016,
                        mse={m:sum(r['mse'][m] for r in rows)/2 for m in errors},
                        mean_d2=sum(r['mean_d2'] for r in rows)/2,two_mean_ed=sum(r['two_mean_ed'] for r in rows)/2,
                        Lcorrect_minus_Loff=sum(r['Lcorrect_minus_Loff'] for r in rows)/2,
                        algebra_max_absolute_error=max(r['algebra_absolute_error'] for r in rows),input_hashes=before)
                    if row['algebra_max_absolute_error']>1e-10:raise RuntimeError('Teacher loss decomposition failed')
                    if before!={k:array_hash(v) for k,v in dict(xt=xt,target=target,t=t,mask=mask,x0=x0,x1=x1).items()}:
                        raise RuntimeError('Teacher mutated fixed inputs')
                    report['rows'].append(row);self._emit(report,'teacher_time',t=timepoint)
            report['expected_counts']=dict(official_forward_attempts=15,official_forward_returns=15,
                teacher_forward_attempts=15,teacher_forward_returns=15,teacher_examples=30,condition_encodes=10,optimizer_updates=0)
            if any(counts[k]!=v for k,v in report['expected_counts'].items()):raise RuntimeError('Teacher actual count mismatch')
            report['status']='PASS'
        except BaseException as error:
            report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());error.evaluation_report=report;raise
        return report
