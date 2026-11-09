"""Fixed-budget experiment control, registration and read-only resource receipts."""
from pathlib import Path
import os, json, hashlib, time, subprocess, sys, argparse
REPO=Path("/mnt/e/MeshFlow")
OUT=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1")
DATA=OUT.parent/"chair_r2_user_conditions_v1"
BASE=OUT.parent/"chair_condition_recipe_factorial_v1/checkpoints/R2_MIX_H_ALL_step6000.pt"
CODE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
    os.replace(tmp,p)
def event(event,**fields):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"controller_events.jsonl").open("a") as f:
        f.write(json.dumps(dict(time_unix=time.time(),event=event,**fields))+"\n");f.flush();os.fsync(f.fileno())
def capture(cmd):
    p=subprocess.run(cmd,text=True,capture_output=True)
    return dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def resources(stage):
    obj=dict(stage=stage,time_unix=time.time(),cpu_cores=os.cpu_count(),
       gpu=capture(["nvidia-smi","--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu","--format=csv"]),
       gpu_processes=capture(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv"]),
       memory={l.split(":")[0]:l.split(":")[1].strip() for l in Path("/proc/meminfo").read_text().splitlines() if l.split(":")[0] in ("MemTotal","MemAvailable","SwapTotal","SwapFree")},
       cpu_load=capture(["ps","-eo","pid,pcpu,pmem,rss,args","--sort=-pcpu"]),
       disk=capture(["df","-B1",str(OUT.parent)]))
    write(OUT/"resources"/(str(time.time_ns())+"_"+stage+".json"),obj)
    print(json.dumps({k:v for k,v in obj.items() if k!="cpu_load"},indent=2))
    return obj
def registration():
    p=OUT/"registration.json"
    if p.exists():
        print(json.dumps(dict(status="REUSE_IMMUTABLE_REGISTRATION",sha256=sha(p))));return
    required={"active_conditions.json":"e3922f61cddc37fe8c8ee5b0224f2b943c9158ea9bc9a68f983652d8a856da7a",
      "dataset_handoff_active.json":"7efceb63258d95c9707120e3ce9ba96cfb107b7cc6b3f585b76536e4b72c249e",
      "old_regression_selection.json":"9a4f7618b41edc5f637edeeb4ad690bc96d13f6dbda9e3d8f243d785619a9842"}
    for f,h in required.items(): assert sha(DATA/f)==h,f
    expected="89f49e014615eb4a41e052ed90719a2580ead46418997c268e35e09ff37a7541"
    assert sha(BASE)==expected,"main R2 full checkpoint"
    sources={str(p):sha(p) for p in sorted((REPO/"r2/src").rglob("*.py"))}
    obj=dict(schema="r2_user_condition_finetune_registration_v1",experiment="R2_USER_CONDITION_FINETUNE_PILOT_V1",
      created_unix=time.time(),authorization="User: 行，那你就这样进行实验吧; supersedes preparation-only training_authorized=false without altering original handoff",
      out=str(OUT),data_root=str(DATA),base_checkpoint=str(BASE),base_checkpoint_sha256=expected,
      authorized_arms=["R2_CONT_OLD","R2_CONT_USER"],frozen_label="R2_FROZEN",
      base_global_step=6000,updates_per_arm=1000,endpoint_global_step=7000,
      effective_batch=8,microbatch=1,preflight_limits=dict(forward=16,backward=8,optimizer_update=0),
      train_limits=dict(optimizer_update=2000,train_forward=16000,train_backward=16000,sample_exposures=16000),
      evaluation=dict(new_conditions=6,new_parents=6,old_conditions=8,noises_per_condition=2,models=3,total_outputs=84,
        normal_new_rollouts=68,normal_sampling_forwards=3400,max_new_rollouts_if_old_identity_invalid=84,
        invalid_cache_policy="Same original inputs/noises only; scoring version mismatch triggers CPU rescore, never regeneration",
        queries_per_C=2048,integrator="Euler50",postprocess=False,GT_reconstruction=False,S="NA: no registered interface",E="NA: no exposure contract",
        primary_comparison="USER minus OLD",secondary_comparison="USER minus FROZEN",panels_separate=True,
        qualitative="All84; common cameras and neutral/C blue F orange views; AI_PROXY with uncertainty, independent overall quality and C-use"),
      model=dict(architecture="Native/T1 + role + Geo",B=False,teacher=False,readout=False,full_parameter_finetune=True,
        FM="free-only, denominator 9*sum(N-K) over effective batch",HYBRID_lambda=0.25,order="free-only OT against original Gaussian, then mix independent epsilon; no rematching",
        inference_initial_noise="standard Gaussian",forward="BF16",parameters_and_integration="FP32",TF32=False,SDPA="math",deterministic_warn_only=False),
      optimizer=dict(type="AdamW",lr=1e-5,betas=[0.9,0.95],weight_decay=0,gradient_clip=1,
        inheritance="same complete main step6000 parameters, moments, per-param steps and RNG; fresh branch stream identity",scheduler=None),
      data=dict(common_parents=25,old_conditions=395,old_C20=200,old_C40=195,user_conditions=117,
        parent_sampling="Uniform; balanced randomized 25-batch blocks, 8 distinct parents/batch, 320 exposures per parent per arm",
        old_sampling="2 C20 clean + 2 C20 width + 2 C40 clean + 2 C40 width per batch; within-parent/ratio uniform",
        user_sampling="within parent active unique C uniform; no seat/back/tier quota",
        augment="4 clean alpha=1, 4 shared X-width alpha uniform[0.9,1.1]; same source M and N",
        pairing="parent/N/source M/alpha/time/random generation rule; K-dependent array shapes and OT differ",
        known_risk="14 source defect parents retained; HOLD excluded; no source defect subtraction from generated scores"),
      checkpoints=dict(recovery_steps=[500,1000],evaluated_steps=[1000],exports=2,recoverable_endpoints=2),
      failure_policy=dict(silent_retry=False,completed_optimizer_replay=False,extra_updates=False,
        action="Stop additional GPU dispatch on OOM/numerical failure; retain attempt/error ledger and latest safely saved state; no automatic seed/batch/precision/model changes",
        resumability="Only verified exact next cursor and no unresolved consumed update; no rollback to step500 if later updates occurred",
        engineering_fix_before_calls="Allowed with preserved source revisions; consumed calls count toward fixed limits; no automatic model-call retry"),
      dispatch=dict(cpu_threads_per_process=2,headroom="max(2048MiB,10%total)",incremental_slots="floor(max(0,free-headroom)/(1.2*measured_worker_peak))",
        workers="Measured separately per training/sampling wave; incremental launch; account compute utilization and CPU/OT load",
        unrelated_processes="Do not stop"),
      stop="After two1000-step branches,84 uniform scores and full review; no additional arms or default-model switch, no git commit/push",
      protected_inputs={str(DATA/f):h for f,h in required.items()},source_hashes=sources)
    write(p,obj);event("REGISTERED",sha256=sha(p),base_checkpoint_sha256=expected)
    print(json.dumps(dict(status="REGISTERED",sha256=sha(p),source_files=len(sources),base_checkpoint_sha256=expected)))
def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["register","resources"]);p.add_argument("--name",default="manual")
    a=p.parse_args()
    if a.stage=="register":registration()
    else: resources(a.name)
if __name__=="__main__":main()
