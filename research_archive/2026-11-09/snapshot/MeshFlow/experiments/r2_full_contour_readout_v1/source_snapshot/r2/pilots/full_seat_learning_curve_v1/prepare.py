"""Freeze the one-arm FULL task before any actual model execution."""
import json
from . import common as C

def prepare():
    target=C.OUT/'registration.json'
    if target.exists():
        r=C.read(target)
        for p,h in {**r['training_code_hashes'],**r['evaluation_code_hashes']}.items():
            if C.sha(p)!=h:raise ValueError('Frozen source changed: '+p)
        return {'status':'VERIFIED_FROZEN','sha256':C.sha(target)}
    dpath=C.OUT/'inputs/training_data.json';d=C.read(dpath)
    line=C.read(C.OUT/'inputs/checkpoint_lineage.json')
    cleanup=C.OUT/'cleanup/completion.json';cl=C.read(cleanup)
    if cl['status']!='COMPLETE' or cl['training_started']:raise ValueError('Cleanup must complete first')
    if d['status']!='FROZEN' or len(d['train_task_ids'])!=47:raise ValueError('Actual coherent TRAIN FULL pool changed')
    old=C.read(C.BASE/'registration.json')
    ca=C.read(C.OUT/'inputs/evaluation_cache_manifest.json')
    if ca['status']!='PASS':raise ValueError('Cache audit incomplete')
    runtime=sorted((C.ROOT/'r2/src/meshflow_control').rglob('*.py'))
    train_extra=['r2/pilots/user_data_scaling_32_67_v1/training_data.py','r2/pilots/user_condition_finetune_v1/trainer.py']
    train_paths=runtime+[C.CODE/p for p in ('common.py','data.py','train.py','prepare.py')]+[C.ROOT/p for p in train_extra]
    eval_extra=['r2/pilots/regular_patch_control_v1/inference.py','r2/pilots/regular_patch_control_v1/score.py',
      'r2/pilots/seat_failure_diagnosis_v1/gpu.py','r2/pilots/condition_width_response_v1/evaluate.py',
      'r2/pilots/user_condition_finetune_v1/evaluation.py','r2/tools/full_boundary_attribution_v1/geometry.py',
      'r2/tools/cref_layer_audit_v1/geometry.py','r2/pilots/regular_patch_control_v1/depth_gallery.py',
      'r2/pilots/c_reference_frame_v1/model.py']
    eval_paths=runtime+[C.CODE/p for p in ('common.py','evaluate.py','render.py')]+[C.ROOT/p for p in eval_extra]
    for x in train_paths+eval_paths:
        if not x.is_file():raise FileNotFoundError(x)
    inf=C.OLD/'inference/USER_EXPANDED_67_generator_step6000.pt'
    ir=C.read(inf.with_suffix('.json'))
    # Reuse the completed cleanup's actual retained-file hashes, not filename trust.
    keep=C.csvread(C.OUT/'cleanup/retained_verified.csv')
    hashes={str(C.resolve(x['path'])):x['sha256'].lower() for x in keep}
    for p,h in [(C.SOURCE,line['checkpoint_expected_sha256']),(inf,ir['sha256'])]:
        if hashes.get(str(p))!=h:raise ValueError('Retained actual checkpoint hash differs: '+str(p))
    cache=ca['cache'];cached0=len(cache.get('FULL_S0000',{}));new_rollouts=96-cached0
    r=dict(version=C.VERSION,status='FROZEN',utc=C.utc(),
       source_checkpoint=str(C.SOURCE),source_checkpoint_sha256=line['checkpoint_expected_sha256'],
       source_generator_sha256=line['generator_state_sha256'],source_optimizer_sha256=line['optimizer_state_sha256'],source_rng_sha256=line['rng_sha256'],
       source_global_step=6000,source_data=str(C.DATA),source_data_sha256=C.sha(C.DATA),
       step0_inference_path=str(inf),step0_inference_sha256=ir['sha256'],
       interface_manifest_path=str(dpath),interface_manifest_sha256=C.sha(dpath),train_task_ids=d['train_task_ids'],
       cleanup_receipt_path=str(cleanup),cleanup_receipt_sha256=C.sha(cleanup),
       updates_per_arm=6000,batch=8,microbatch=1,diagnostic_time_bin_edges=[0,.2,.4,.6,.8,1],
       evaluation_labels=C.EVALUATION_LABELS,requests=old['requests'],queries=old['queries'],conditions=old['conditions'],
       d0_checkpoint=old['d0_checkpoint'],models=old['models'],cache=cache,
       resources={**old['resources'],'training_incremental_estimate_MiB':4096,'CPU_OT_workers':2},
       training_code_hashes={str(p):C.sha(p) for p in train_paths},evaluation_code_hashes={str(p):C.sha(p) for p in eval_paths},
       budget=dict(additional_updates=6000,train_forward=48000,train_backward=48000,optimizer_update=6000,
           checkpoints_local=[0,1000,3000,6000],checkpoints_global=[6000,7000,9000,12000],
           evaluation_requests_per_checkpoint=24,logical_RAW=96,logical_POST=96,new_rollouts=new_rollouts,
           new_sampling_forward=50*new_rollouts,new_D0_forward=new_rollouts,reference_cached_RAW=48,reference_cached_POST=48),
       training_contract=dict(model='Native/T1+role+Geo',coordinates='WORLD',objective='original free-only FM',
           coupling='free-only OT then HYBRID0.25',precision='BF16 forward, FP32 parameters',
           augmentation='4 clean + 4 x-width U(0.9,1.1) per effective batch',
           optimizer='restore matched live EXP67 AdamW and RNG; no EMA mixing; no reset',
           sampling='balanced parents, one coherent original SEAT_FULL each',diagnostics='detached source legal C/Y corner partition; no loss change; endpoint=(1-t)^2 velocity SSE',
           no_extra_forward=True,no_new_loss=True,main_endpoint=6000),
       evaluation_contract=dict(no_new_noise=True,solver='Euler50',post='one official D0 / GEO_C / original source winding / free orientation',
           fixed_panel='original six FULL parents, four noises; all24, TRAIN/DEV separate',
           references='frozen SMALL_C200 and CREF_C200, cached only',
           source_GT_rank=False,required_interface='NA_NO_REQUIRED_INTERFACE',review='actual AI_PROXY, whole shape and C adoption separately',
           metrics='unchanged CF/FF/I, degeneracy/fragments, A-VMR/PMR, actual legal shared edges/length and finite boundary distances',
           boundary_config='unchanged full_boundary_attribution_v1.DEFAULTS',no_best_checkpoint=True),
       failure_policy='No silent replay. Preserve failure logs and partial calls. Safe segment checkpoints resume exact generator/AdamW/RNG/cursor. Engineering-only repairs documented with prior source and registration retained.',
       final_actions='No default switch, automatic commit or push; stop after fixed task')
    C.write(target,r)
    C.event('REGISTRATION_FROZEN',sha256=C.sha(target),parents=47,updates=6000,new_rollouts=new_rollouts)
    return dict(status='FROZEN',sha256=C.sha(target),parents=47,new_rollouts=new_rollouts)

if __name__=='__main__':print(json.dumps(prepare(),indent=2))
