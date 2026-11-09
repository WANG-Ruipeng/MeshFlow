"""Freeze the one-arm scientific contract before any candidate GPU call."""
from __future__ import annotations
import json,copy
from . import common as C

def prepare():
    source=C.BASELINE/'registration.json';old=C.read(source)
    topo=C.OUT/'inputs/topology_manifest.json'
    if not topo.exists():raise RuntimeError('Actual FULL topology must be completed first')
    tests=C.OUT/'audits/model_unit_tests.json'
    if not tests.exists() or C.read(tests).get('status')!='PASS':raise RuntimeError('CPU geometry/readout tests required')
    from . import topology
    t=topology.TopologyInputs()
    if len(old['train_task_ids'])!=47 or len(old['requests'])!=24:raise ValueError('Frozen pool/panel changed')
    plan=C.BASELINE/'plans/FULL_SEAT.json'
    planhash=C.sha(plan)
    if planhash!='52a48be0987c11511dea149829935e6af9baceb4ac70317052653b95c557f4c9':raise ValueError('Original replay plan hash differs')
    r=copy.deepcopy(old)
    r.update(version=C.VERSION,utc=C.utc(),status='FROZEN',cache={},models={},evaluation_labels=C.EVALUATION_LABELS,
        baseline_registration_path=str(source),baseline_registration_sha256=C.sha(source),
        replay_plan_path=str(plan),replay_plan_sha256=planhash,
        replay_exposure_path=str(C.BASELINE/'training/FULL_SEAT/exposure.jsonl'),
        replay_exposure_sha256=C.sha(C.BASELINE/'training/FULL_SEAT/exposure.jsonl'),
        topology_manifest_path=str(topo),topology_manifest_sha256=C.sha(topo),new_module_seed=20261014,
        pairing='Original 48000 plan records unchanged including original FULL_SEAT labels/seeds; existing6000 OT batches read-only; compare original input/label/interface hashes at each exposure',
        hypothesis='C finite-surface plus per-face source contour readout may improve RAW adoption AND POST legal interface on same outputs, beyond fully trained original FULL_S6000.',
        main_comparator='FULL_S6000',main_endpoint='CONTOUR_S6000',
        judging_contract=dict(joint_baseline=6,denominator=24,uncertain_not_yes=True,more_than6_not_sufficient=True,
            preserve_existing_positive_requests=True,report_new_lost_by_parent_noise=True,
            report_legal_length_and_long_edge_midpoints=True,report_RAW_contour_and_body=True,
            report_CF_FF_I_and_DEV_costs=True,no_single_composite_metric=True,no_posthoc_best=True,
            generalization='Repeatedly inspected three DEV parents; not a new independent test'),
        budget=dict(additional_updates=6000,train_forward=48000,train_backward=48000,optimizer_update=6000,
            checkpoints_local=[0,1000,3000,6000],checkpoints_global=[6000,7000,9000,12000],
            evaluation_requests_per_checkpoint=24,logical_RAW=96,logical_POST=96,
            step0_cached_RAW=24,step0_cached_POST=24,new_rollouts=72,new_sampling_forward=3600,new_D0_forward=72,
            baseline_cached_RAW=96,baseline_cached_POST=96,new_noise_arrays=0,new_OT=0,extra_model_probes=0),
        module_contract=dict(after_complete_block=6,dimension=128,heads=4,head_dimension=32,features=17,extra_parameters=412032,
            fixed_L=C.L,MLP=[17,128,128],MLP_activation='SiLU',qkv_bias=False,output_bias=False,MLP_bias=True,
            layernorm_affine=False,layernorm_eps=1e-5,qkv_init='xavier_uniform',MLP_init='torch Linear default',output_init='zero',
            memory='current H0 coordinate+role+Geo; FP32 face mean',pair_value='V(memory_j)+phi(relation_ij)',
            boundary_domain='per_face incidence1 edges from actual FULL C IDs',tie='geometry-canonical symmetric mean of fixed FP32 equal-distance candidates',
            IDs_features=False,only_free_write=True,all_C_attention=True,no_topk=True,no_geometry_attention_bias=True,
            detached_memory=False,FP32_add_before_original_dtype_cast=True),
        optimizer_contract=dict(old_clock='6000+k',new_clock='actual AdamW per-parameter state from empty; expect k because every parameter receives a gradient tensor',
            restoration='load old194parameter AdamW state first; append separate new group; verify names/order/shapes/values',
            lr=1e-5,betas=[.9,.95],eps=1e-8,weight_decay=0,clip='single global norm 1, old/new norms recorded; no groupwise clipping'),
        initialization_check='CPU geometry checks plus first8ordinary training forwards before first update; exact zero residual/unchanged hidden and original per-micro FM; no extra full model forward',
        failure_policy='No silent model retry or replay. Record failures and preserve partial calls. Engineering repairs only with evidence, old sources and repair record. Exact safe checkpoints can resume.')
    r['training_contract']=dict(old['training_contract'],model='Native/T1+role+Geo plus after-block6 finite surface/source contour readout',optimizer='Restore old matching AdamW/RNG, append empty same-hyperparameter branch group')
    r['evaluation_contract']=dict(old['evaluation_contract'],references='Original FULL curve four points, read-only cache; fixed 24 requests')
    r['resources']=dict(old['resources'],training_incremental_estimate_MiB=4608,generator_incremental_estimate_MiB=2800,
        initial_GPU_workers=1,CPU_workers_per_pipeline=2,reason='single dependent training arm; user ~6GiB experiment envelope, uncertain new branch process peak; incremental reassessment for evaluation')
    hashes=dict(old['training_code_hashes'])
    for name in ('common.py','train.py','model.py','relations.py','topology.py','prepare.py'):
        p=C.CODE/name;hashes[str(p)]=C.sha(p)
    r['training_code_hashes']=hashes
    evalhash=dict(old['evaluation_code_hashes'])
    for name in ('common.py','model.py','relations.py','topology.py','evaluation.py'):
        p=C.CODE/name;evalhash[str(p)]=C.sha(p)
    r['evaluation_code_hashes']=evalhash
    C.write(C.OUT/'registration.json',r)
    C.write(C.OUT/'plans/replay_binding.json',dict(path=str(plan),sha256=planhash,byte_identical_original=True,records=48000,regenerated=False))
    for p,h in {**hashes,**evalhash}.items():
        dest=C.OUT/'source_snapshot'/C.resolve(p).relative_to(C.ROOT)
        C._write_bytes(dest,C.resolve(p).read_bytes())
    C.event('REGISTRATION_FROZEN',sha256=C.sha(C.OUT/'registration.json'),train_parents=47,updates=6000,new_sampling_forwards=3600,new_D0=72)
    return dict(status='FROZEN',registration_sha256=C.sha(C.OUT/'registration.json'),topology_manifest_sha256=C.sha(topo),replay_plan_sha256=planhash)

if __name__=='__main__':print(json.dumps(prepare(),indent=2))
