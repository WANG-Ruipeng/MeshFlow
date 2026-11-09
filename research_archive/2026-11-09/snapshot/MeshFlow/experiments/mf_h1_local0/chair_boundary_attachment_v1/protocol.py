"""Freeze before first call/update; verify only actual protected dependencies."""
from pathlib import Path
import difflib
import shutil
from .common import *
from .ledger import LIMITS

SCIENCE=('common.py','ledger.py','method.py','stream.py','checkpoint.py','training.py','precheck.py','sampling.py','worker.py','resources.py','protocol.py','control_metrics.py','metric_tests.py','run_all.py')
DEPENDENCIES=('native_t1/runtime.py','native_t1/model.py','native_t1/context_geometry.py','native_t1/training.py',
    'native_t1/data.py','native_t1/checkpoint.py','native_t1/portable_checkpoint.py','models/equidit.py','models/attention.py',
    'utils/ot_utils.py','experiments/mf_h1_local0/control_mvp48/model_mvp.py',
    'experiments/mf_h1_local0/control_mvp48/train_stream.py','experiments/mf_h1_local0/control_mvp48/train_io.py',
    'experiments/mf_h1_local0/control_mvp48/eval_runtime.py','experiments/mf_h1_local0/chair_hybrid_coupling_v1/checkpoint.py',
    'experiments/mf_h1_local0/chair_hybrid_coupling_v1/coupling.py','experiments/mf_h1_local0/chair_hybrid_coupling_v1/sampling.py',
    'experiments/mf_h1_local0/chair_domain_control_v1/prepare.py',
    'experiments/mf_h1_local0/evaluate.py','experiments/mf_h1_local0/native_inpainting_metrics.py',
    'experiments/mf_h1_local0/native_patch_seams.py','experiments/mf_h1_local0/guided_native_geometry_audit.py',
    'experiments/mf_h1_local0/guided_native_geometry_geometry.py')


def verify(out=OUT,training=False):
    out=Path(out)
    for name in ('precheck_sources.json','protected_sources_before.json'):
        for path,expected in read_json(out/'protocol'/name).items():
            if digest(resolve(path))!=expected:raise RuntimeError('Frozen source changed: '+path)
    for path,expected in read_json(out/'registration/input_file_hashes.json').items():
        if digest(resolve(path))!=expected:raise RuntimeError('Frozen input changed: '+path)
    if training:
        p=read_json(out/'protocol.json')
        assert p['train_plan_sha256']==digest(out/'train_plan.json')
        assert p['requests_sha256']==digest(out/'requests.json')
        assert p['lambda_calibration_sha256']==digest(out/'lambda_calibration.json')
    return dict(status='PASS',training=training)


def freeze_precheck(out=OUT):
    out=Path(out)
    if (out/'protocol/precheck_sources.json').exists():return verify(out)
    assert read_json(out/'tests/method/results.json')['status']=='PASS'
    assert read_json(out/'registration/summary.json')['status']=='PASS'
    files=[CODE/name for name in SCIENCE]
    for p in files:compile(p.read_text(encoding='utf-8'),str(p),'exec')
    freeze_json(out/'protocol/precheck_sources.json',{str(p.relative_to(ROOT)):digest(p) for p in files})
    freeze_json(out/'protocol/protected_sources_before.json',{p:digest(ROOT/p) for p in DEPENDENCIES})
    spec=dict(task='CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_V1',seed=SEED,arms=list(ARMS),
        start_sha256=START_SHA,source_conditional_total_step=5000,updates_per_arm=1000,
        budget_limits=LIMITS,total_forward_limit=52196,GPU_workers=1,threads_per_worker=2,
        training_stream=dict(plan_sha256=digest(out/'train_plan.json'),effective_batch=8,microbatch=1,distinct_parents=8,
            unit_alpha=4,uniform_alpha=4,alpha_bounds=[.9,1.1],hybrid_lambda=.25),
        optimizer=dict(name='AdamW',lr=1e-5,betas=[.9,.95],weight_decay=0,clip_grad_norm=1,
            old_moments_restored=True,old_step=5000,new_boundary_step=0),
        calibration=dict(first_effective_batch=0,t=[.5,.8],max_model_forwards=16,
            lambda_rule='.25 * median(||g_FM|| / ||g_Eseam_div8||)',minimum_states=8,minimum_parents=4),
        precheck=dict(planned_forwards=44,planned_backwards=16,optimizer_updates=0,
            cost_tasks='first batch ordinary nearest median N and largest N; t .5 and .8 once each per arm',
            cost_limit_relative_A=2,nonzero_boundary_perm_einf=5e-5,nonzero_boundary_perm_erms=1e-5),
        sampling=dict(solver='original_Euler50',BF16_forward=True,FP32_integration=True,expected_new=304,replays=2,
            cached=76,requests=400,original_NA=20),
        controls=['SeamCov','ConflictCov_C','ExposedRetain'],
        sensitivity=dict(units=4,primary=[64,2048,26],dense=[256,8192,74]),
        decision=dict(unit='P1 parent, average all four candidates before across 8 parents',
            seam_gain_vs_A=.05,median_delta_gt=0,improving_parents_min=6,mean_not_below_START=True,
            conflict_mean_increase_max=.02,exposed_mean_drop_max=.02,severe_parent_regression=.10,
            D_sym2048_relative_worse_max=.05,fixed_query_MSE_relative_worse_max=.05,
            fixed_query_p95_relative_worse_max=.10,FF_affected_free_increase_max=.02,
            quality_comparators=['A_FM_END','START_HYBRID'],incomplete_uncertainty_never_pass=True),
        boundaries=dict(default_changed=False,commit=False,push=False,extra_loss=False,extra_generation=False,old_sources_read_only=True))
    freeze_json(out/'protocol/precheck_protocol.json',spec)
    snapshot=out/'code_snapshot/precheck';snapshot.mkdir(parents=True,exist_ok=True)
    for p in files:
        dst=snapshot/p.name
        if dst.exists():assert digest(dst)==digest(p)
        else:shutil.copyfile(p,dst)
    return verify(out)


def freeze_training(out=OUT):
    out=Path(out);verify(out)
    pre=read_json(out/'precheck/results.json');cal=read_json(out/'lambda_calibration.json')
    replay=read_json(out/'replays/summary.json');assert replay['status']=='PASS'
    metric=read_json(out/'metric_tests_v1r1/metric_test_results.json');assert metric['status']=='PASS'
    protocol=dict(read_json(out/'protocol/precheck_protocol.json'),
        precheck_results_sha256=digest(out/'precheck/results.json'),lambda_calibration_sha256=digest(out/'lambda_calibration.json'),
        lambda_seam=pre['lambda_seam'],arm_status=pre['arm_status'],
        precheck_sources_sha256=digest(out/'protocol/precheck_sources.json'),
        metric_tests_sha256=digest(out/'metric_tests_v1r1/metric_test_results.json'),
        metric_source_sha256=digest(CODE/'control_metrics.py'),
        train_plan_sha256=digest(out/'train_plan.json'),requests_sha256=digest(out/'requests.json'))
    freeze_json(out/'protocol.json',protocol)
    freeze_json(out/'protocol/protocol_identity.json',dict(file_sha256=digest(out/'protocol.json'),semantic_sha256=json_hash(protocol)))
    return protocol
