"""Write an honest Session C dependency report; never create formal identities."""
from __future__ import annotations
import csv
import time
from pathlib import Path
from .eval_common import ROOT, RUN, EVALUATION, REGISTRATION, read_json, write_json, sha256, dependencies


def main():
    deps=dependencies()
    if deps['DATA_READY.json']['exists'] or deps['TRAIN_READY.json']['exists']:
        raise RuntimeError('Readiness changed: inspect before writing blocked report')
    blocked=read_json(RUN/'DATA_BLOCKED.json')
    if blocked.get('status')!='BLOCKED':
        raise RuntimeError('Expected explicit upstream blocked status')
    shortage=read_json(RUN/'data/selection_shortfall.json')
    evidence_paths=[REGISTRATION/'eval_scoring_fixtures.json', REGISTRATION/'eval_teacher_fixtures.json',
                    REGISTRATION/'eval_preparation_fixture.json',
                    EVALUATION/'eval_reporting_fixture/eval_reporting_fixture.json']
    evidence=[dict(path=str(p.relative_to(RUN)),sha256=sha256(p),record=read_json(p)) for p in evidence_paths]
    if any(e['record']['status']!='PASS' for e in evidence):
        raise RuntimeError('CPU preparation evidence is incomplete')
    from .eval_scoring import METRIC_SPECS,source_identity
    code=ROOT/'experiments/mf_h1_local0/control_mvp48'
    code_hashes={str(p.relative_to(ROOT)):sha256(p) for p in sorted(code.glob('eval_*.py'))}
    timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    result=dict(schema='control_mvp48_evaluation_status_v1',owner='Session C',status='BLOCKED_DATA',
        complete=False,recorded_utc=timestamp,dependency=blocked,
        dependency_sha256=sha256(RUN/'DATA_BLOCKED.json'),dependencies=deps,
        shortages=shortage['shortages'],protocol_status='NOT_FROZEN_NO_DATA_READY',
        model_identity_status='NOT_RUN_NO_TRAIN_READY_OR_THREE_FINAL_ENDPOINTS',
        planned=dict(test_parents=8,conditions_per_parent=2,noises_per_condition=2,models=3,
                     formal_requests=96,formal_NFE=4800,teacher_requests=24,teacher_NFE=24,max_optional_repeats=2),
        actual=dict(registered_formal_requests=0,formal_request_attempts=0,formal_completed=0,
                    formal_failed_outputs=0,formal_valid_denominator=0,formal_model_forward_attempts=0,
                    formal_model_forward_returns=0,formal_NFE=0,teacher_forward_attempts=0,
                    teacher_forward_returns=0,teacher_valid_denominator=0,repeat_attempts=0,
                    real_teacher_OT_attempts=0,real_teacher_OT_returns=0,formal_scoring_attempts=0,
                    formal_scoring_returns=0,optimizer_updates=0,training_loss_calls=0,
                    model_backward_calls=0,geometry_gradient_calls=0,geometry_repairs=0,
                    CUDA_models_created=0,GPU_workers=0,GPU_lock_acquisitions=0),
        budget_remaining=dict(formal_requests=96,formal_NFE=4800,teacher_requests=24),
        cpu_preparation=dict(status='PASS',evidence=evidence,code_sha256=code_hashes,
            scorer_identity=source_identity(),runtime_GPU_test='NOT_RUN',
            preparation_full_data_integration='NOT_RUN',teacher_synthetic_OT_calls=2,
            note='Synthetic fixtures are preparation evidence only, never formal model observations.'),
        results=[],per_parent=[],teacher_rows=[],comparisons={},all_failed_outputs=[],
        table_status=dict(per_output='HEADER_ONLY_0_ROWS_UNREGISTERED',per_parent='HEADER_ONLY_0_ROWS_UNREGISTERED',
                          teacher='HEADER_ONLY_0_ROWS_NOT_RUN',
                          explanation='No frozen 8-parent task identities exist. Do not fabricate UID/seed or 96 observation rows.'),
        exact_topology='NOT_RUN',watertight_certification='NOT_RUN',
        control=dict(status='NOT_RUN',reason='No frozen formal test matrix or outputs; geometry editing control not established'),
        fixed_views=dict(status='NOT_RUN',formal_images=0,noise_id=0,
                         synthetic_fixture='evaluation/eval_reporting_fixture/parent_00_noise0.png'),
        conclusions=dict(q1='NOT_EVALUABLE',q2='NOT_EVALUABLE',q3='NOT_EVALUABLE',
                         q4='Resolve Session A eligible-test-group shortfall, then await DATA_READY and all Session B final endpoints.'),
        follow_up=['Session A resolves its reported 6-of-8 eligible independent test-group shortfall under authorized data rules.',
                   'Read final DATA_READY and freeze exact evaluation identities/protocol; create8 real teacher pairs once.',
                   'Await TRAIN_READY and T1/Geo/Full step1000; verify unified loader, common official initialization, hashes and branch lineage.',
                   'Run original96 Euler50 requests plus24 teacher forwards with one GPU worker, score offline and replace this blocked status report.'],
        limitations=['No model effect or 1000-step convergence conclusion is possible from an unrun matrix.',
                     'No training or data-selection rules changed; no six-parent substitute matrix.',
                     'No automatic background GPU dispatch has been scheduled.'])
    write_json(EVALUATION/'results.json',result)
    write_json(REGISTRATION/'eval_dependency_snapshot.json',dict(recorded_utc=timestamp,
        dependencies=deps,blocker=blocked,shortfall=shortage,
        initial_resource_snapshot=read_json(REGISTRATION/'eval_preparation_status.json'),
        actual_counts=result['actual']))
    write_json(REGISTRATION/'eval_protocol_draft.json',dict(status='DRAFT_NOT_FROZEN',owner='Session C',
        prerequisite='DATA_READY before freezing; TRAIN_READY plus all final endpoints before any GPU',
        planned=result['planned'],metric_specs=METRIC_SPECS,scorer_identity=source_identity(),
        integration='original Euler50; 51 saved FP32 states; BF16 forward; strict deterministic TF32-off SDPA math',
        input='C,N,free Gaussian/current state,t,valid/known masks only; no GT for generation',
        scoring='own free complement per C; original512 area-weighted samples; same fixed GT queries across models; copied C excluded',
        aggregation='within-parent2C*2noise mean then8 equal parents; paired Geo-T1,Full-Geo,Full-T1; no best noise',
        fixed_view='noise0; two C per object; GT,C,T1,Geo,Full; elevation20 azimuth-55 vertical y; shared range per parent',
        execution='shared GPU lock, one worker; CPU scoring may overlap only after measured resources; no automatic launch'))
    tables={
        'per_output.csv':['request_id','parent_id','condition_id','noise_id','model','status','nfe','failure_reason']+list(METRIC_SPECS),
        'per_parent.csv':['parent_id','status','registered_cases','valid_cases'],
        'teacher.csv':['parent_id','condition_id','model','status','t','FM_error','nfe','failure_reason'],
    }
    for name,fields in tables.items():
        with (EVALUATION/name).open('w',encoding='utf-8',newline='') as stream:
            csv.DictWriter(stream,fieldnames=fields).writeheader()
    figures=RUN/'figures';figures.mkdir(exist_ok=True)
    (figures/'eval_NOT_RUN.md').write_text('Formal fixed views: NOT_RUN. DATA_READY is absent; no formal mesh was generated.\nThe synthetic reporting fixture is CPU layout validation only.\n',encoding='utf-8')
    print('BLOCKED_DATA report data and honest header-only tables written; formal NFE=0, teacher NFE=0')


if __name__=='__main__':
    main()
