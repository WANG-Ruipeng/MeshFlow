"""Freeze the user-specified method, stage gate and actual execution identities."""
from .common import *


def freeze(out=OUT):
    out=Path(out)
    from .project import FIXED_CONFIG
    tests=read_json(out/'tests/project/attempt1/results.json')
    assert tests['status']=='PASS' and tests['configurations_completed']==8
    bases=read_json(out/'base_requests.json');requests=read_json(out/'requests.json')
    assert len(bases)==16 and len(requests)==48
    assert sorted(set(x['candidate'] for x in bases))==[0,1]
    assert len({r['uid'] for r in bases})==8
    for row in requests:
        if row['label']=='RAW':
            r=read_json(result_path(out,row));assert r['status']=='CACHED'
            assert digest(resolve(r['raw_path']))==r['raw_sha256']
    config=read_json(out/'projection_config.json')
    assert config==dict(FIXED_CONFIG)
    source_files=['common.py','project.py','project_tests.py','endpoint.py','ledger.py','protocol.py','registration.py']
    sources={str(CODE/name):digest(CODE/name) for name in source_files}
    freeze_json(out/'protocol/core_sources.json',sources)
    old=read_json(out/'tests/project/source_freeze.json')
    value=dict(task='CHAIR_BOUNDED_ATTACHMENT_CORRECTION_PILOT_V1',version=1,
        model=dict(path=str(START),sha256=START_SHA,arm='OT_HYBRID',pilot_step=1000,conditional_total_step=5000,lambda_hybrid=.25),
        inputs=dict(base_requests_sha256=digest(out/'base_requests.json'),requests_sha256=digest(out/'requests.json'),parents=8,candidates=[0,1]),
        projection_config=config,projection_config_sha256=digest(out/'projection_config.json'),
        CPU_tests=dict(configurations=8,source_receipt=old,result_sha256=digest(out/'tests/project/attempt1/results.json')),
        metric_version='chair_boundary_attachment_control_metrics_v1r1',
        seam_taus=[.001,.005,.01,.02],seam_angle_degrees=20,
        new_strict_tau_reason='Preregistered fine attachment point for bounded small displacement; old 1% and .5%/2% remain unchanged',
        phase1_gate=dict(strict_seam_mean_gain=.02,strict_median_gain_positive=True,minimum_improving_parents=4,
            expected_parents=8,old_seam_max_mean_drop=.005,conflict_max_mean_rise=.01,exposure_max_mean_drop=.01,
            maximum_parent_conflict_rise_or_exposure_drop=.05,D_sym_max_relative_worsening=.05,
            fixed_MSE_max_relative_worsening=.05,fixed_p95_max_relative_worsening=.10,
            complete_all_16=True,conservative_uncertainty=True,all_geometry_contracts=True),
        phase2=dict(conditional_on='phase1_gate.json status PASS only',steps=[40,45,49],Euler_steps=50,
            formula='F_euler + (t+h)*(Ycorr-Yhat); no accepted correction returns original F_euler directly',
            last_step='Yhat=original F_euler; return project coordinates directly',
            cumulative_endpoint_budget='sum norm(Delta) per corner <= delta; not reset; not an end-to-end RAW drift guarantee',
            final_RAW_drift_limit_delta=3.,same_corrector=True,all_16_requests=True),
        budgets=dict(training=0,model_backward=0,optimizer=0,RAW_replenishment_max=16,trajectory_max=16,
            replay_max=2,rollout_attempts_max=34,sampling_NFE_max=1700,extra_forward_max=8,total_forward_max=1708),
        replay_registration=[dict(uid=bases[0]['uid'],candidate=0),dict(uid=bases[-1]['uid'],candidate=1)],
        dispatch=dict(GPU_workers=1,phase1_CPU_workers=2,CPU_threads_per_worker=2,
            measured_cpu_cores=24,available_RAM_MiB_at_start=62749,
            rationale='Two independent pure-geometry processes, disjoint outputs and a single parent ledger writer; no GPU before gate'),
        forbidden=['training','GT-aware per-output choice','moving C','welding indices','hyperparameter search','automatic default switch'],
        fixed_task_stop=True)
    freeze_json(out/'protocol.json',value)
    freeze_json(out/'protocol_hash.json',dict(sha256=digest(out/'protocol.json')))
    return value


def verify(out=OUT):
    out=Path(out)
    assert digest(out/'protocol.json')==read_json(out/'protocol_hash.json')['sha256']
    for path,sha in read_json(out/'protocol/core_sources.json').items():assert digest(resolve(path))==sha,path
    p=read_json(out/'protocol.json')
    for filename,key in [('base_requests.json','base_requests_sha256'),('requests.json','requests_sha256')]:
        assert digest(out/filename)==p['inputs'][key]
    assert digest(out/'projection_config.json')==p['projection_config_sha256']
    return p
