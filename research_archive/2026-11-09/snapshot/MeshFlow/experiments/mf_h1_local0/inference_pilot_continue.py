"""Continue only the zero-update interruption caused by baseline observation classification.

The original entry and failed precheck remain immutable. This entry never repeats
stage A, changes selection, resumes an optimizer, or waives a Pre numerical gate.
"""
from .stable_runtime import configure_stable_runtime
import argparse
import copy
import json
import time
import traceback
from pathlib import Path
import torch
from .inference_pilot_run import (PROJECT,REPO,EMA,OLD,BASE_SHA,load_adapter,make_post,train_pre,atomic_json,file_record,frozen)
from .official_wrapper import load_official,state_sha256
from .prompt_dataset import file_sha256
from .inference_pre_wrapper import PreFFNWrapper


def run(output):
    from .inference_pilot_data import prepare_inputs
    from .inference_pilot_evaluation import InferenceEvaluator
    from .inference_precheck_contract import run_prechecks
    out=Path(output).resolve();path=out/'results.json'
    report=json.loads(path.read_text())
    assert report['status']=='STOPPED' and report['counts']['optimizer_step_attempts']==0
    assert report['counts']['optimizer_updates']==0 and not report['stage_c']
    assert report['stage_a']['status']=='PASS'
    prior=report['stage_b']['precheck']
    failures=[x['name'] for x in prior['tests'] if x['status']!='PASS']
    assert failures==['base_own_face_permutation_roundoff']
    assert all(prior[k] for k in ('formal_initialization_unchanged','base_unchanged','backbone_gradients_absent','hooks_and_methods_restored','rng_unchanged'))
    assert report['protected_unchanged'] and 'continuation' not in report
    selection_path=out/'selection.json';selection_sha=report['selection']['sha256']
    assert file_sha256(selection_path)==selection_sha
    selection=json.loads(selection_path.read_text())
    assert selection['selected_setting'] is None
    # No prior optimizer state exists in this new arm, and no new-noise model ran.
    assert not (out/'stage_b/checkpoints').exists()
    for name,record in report['protected_after'].items():
        if file_record(Path(name))!=record:raise RuntimeError('Protected asset changed before continuation: '+name)
    torch.set_num_threads(8);runtime=configure_stable_runtime()
    interruption={k:copy.deepcopy(report[k]) for k in ('status','error','traceback','elapsed_seconds','active_stage','counts','backbone_at_stop','protected_unchanged')}
    report['execution_interruptions']=[interruption]
    report['stage_b']['precheck_attempts']=[copy.deepcopy(prior)]
    for k in ('error','traceback'):report.pop(k,None)
    new_sources=[Path(__file__).resolve(),PROJECT/'inference_precheck_contract.py']
    extra_protected={str(p):file_record(p) for p in new_sources}
    report['continuation']=dict(reason='Fix a test classification error: the user requires recording Base permutation error, not treating it as a Pre tolerance gate.',
        changed_numerical_tolerances=False,changed_model_or_forward=False,changed_selection=False,
        repeated_development_rollouts=0,prior_failure_preserved=True,optimizer_resume=False,
        runtime=runtime,additional_protected_before=extra_protected)
    report.update(status='RUNNING',active_stage='continuation_asset_audit')
    def write():atomic_json(path,report)
    def progress(event):report['latest_progress']=event;write();print(json.dumps(event,allow_nan=False),flush=True)
    started=time.perf_counter();base=None;write()
    try:
        bank,old,audit,saved0,saved500=prepare_inputs(out,persist=False)
        if audit['status']!='PASS':raise RuntimeError('Continuation input identity audit failed')
        report['continuation']['input_audit']=audit
        base,loaded=load_official(EMA,REPO/'configs/snet/base-120m-ot-v-chair.yaml','cuda')
        report['continuation']['strict_load']=loaded
        assert frozen(base,True)==BASE_SHA
        def count_pre(module,args):
            c=report['counts'];c['official_forward_attempts']+=1
            stage=report['active_stage'];c['official_by_stage'][stage]=c['official_by_stage'].get(stage,0)+1
        def count_post(module,args,value):
            c=report['counts'];c['official_forward_returns']+=1
            stage=report['active_stage'];c['official_returns_by_stage'][stage]=c['official_returns_by_stage'].get(stage,0)+1
        base.register_forward_pre_hook(count_pre);base.register_forward_hook(count_post)
        evaluator=InferenceEvaluator(base,bank,out,old,progress_cb=progress)
        stage_b_started=time.perf_counter();pre_adapter=load_adapter(saved0,True);initial_sha=state_sha256(pre_adapter)
        report['active_stage']='B_precheck_contract';write()
        precheck=run_prechecks(base,pre_adapter,bank,counters=report['counts'])
        report['stage_b']['precheck_attempts'].append(copy.deepcopy(precheck))
        report['stage_b']['precheck']=precheck
        for key in ('backward_attempts','backward_completed'):report['counts'][key]+=precheck['actual_counts'][key]
        write()
        if precheck['status']!='PASS' or state_sha256(pre_adapter)!=initial_sha:raise RuntimeError('Actual Pre correctness gate failed; stop without further retry')
        progress(dict(stage='B_precheck_contract',status='PASS',counts=report['counts']))
        pre_adapter=train_pre(base,pre_adapter,bank,old,saved0,saved500,out,report,write,progress)
        pre=PreFFNWrapper(base,pre_adapter).eval()
        try:
            report['active_stage']='B_development';write()
            report['stage_b']['development']=evaluator.evaluate(pre,'Pre500','A0',bank['eval_noises'],'B_Pre500_development')
            report['active_stage']='B_teacher';write()
            report['stage_b']['teacher']=evaluator.teacher(pre,'Pre500')
            report['stage_b']['old_post_teacher_reference']=evaluator.historical_post_teacher()
        finally:pre.close()
        report['stage_b']['stage_wall_seconds_including_contract_precheck']=time.perf_counter()-stage_b_started
        if file_sha256(selection_path)!=selection_sha:raise RuntimeError('Selection changed before confirmation')
        assert report['counts']['optimizer_updates']==500 and report['counts']['optimizer_step_attempts']==500
        report['stage_c']['settings_frozen_before_any_new_noise_forward']=True
        stage_c_started=time.perf_counter();confirmation=bank['confirmation_noises']
        report['active_stage']='C_Base';write()
        report['stage_c']['Base']=evaluator.evaluate_base(confirmation,'C_Base')
        post_adapter=load_adapter(saved500,False);post=make_post(base,post_adapter)
        try:
            report['active_stage']='C_Post500_A0';write()
            report['stage_c']['Post500_A0']=evaluator.evaluate(post,'Post500','A0',confirmation,'C_Post500_A0')
        finally:post.close()
        pre=PreFFNWrapper(base,pre_adapter).eval()
        try:
            report['active_stage']='C_Pre500_A0';write()
            report['stage_c']['Pre500_A0']=evaluator.evaluate(pre,'Pre500','A0',confirmation,'C_Pre500_A0')
        finally:pre.close()
        report['stage_c']['stage_wall_seconds']=time.perf_counter()-stage_c_started
        report['selection']['sha256_after_confirmation']=file_sha256(selection_path)
        assert report['selection']['sha256_after_confirmation']==selection_sha
        report['backbone_after']=frozen(base,True)
        report['protected_after']={name:file_record(Path(name)) for name in report['protected_before']}
        report['protected_unchanged']=report['protected_before']==report['protected_after']
        report['continuation']['additional_protected_after']={name:file_record(Path(name)) for name in extra_protected}
        assert report['protected_unchanged'] and report['continuation']['additional_protected_after']==extra_protected
        report['continuation']['elapsed_seconds']=time.perf_counter()-started
        report.update(status='PASS_EXECUTION',active_stage='complete',elapsed_seconds=interruption['elapsed_seconds']+report['continuation']['elapsed_seconds'])
        write();progress(dict(stage='complete',status='PASS_EXECUTION',counts=report['counts']))
    except BaseException as error:
        if hasattr(error,'evaluation_report'):report['failed_evaluation']=error.evaluation_report
        if hasattr(error,'input_audit'):report['failed_input_audit']=error.input_audit
        report['continuation']['elapsed_seconds']=time.perf_counter()-started
        report.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=interruption['elapsed_seconds']+report['continuation']['elapsed_seconds'])
        if base is not None:report['backbone_at_stop']=state_sha256(base)
        report['protected_after']={name:file_record(Path(name)) for name in report['protected_before']}
        report['protected_unchanged']=report['protected_before']==report['protected_after']
        write();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
