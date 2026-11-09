#!/usr/bin/env python3
"""Recoverable, finite first-stage recipe experiment. Optional stages require a recorded decision."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key,"1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"));sys.path.insert(0,str(REPO/"tools"))
from meshflow_control.artifacts import atomic_json,file_sha256
from dispatch_recipe_factorial import freeze,event,snapshot
ART=Path("/mnt/e/MeshFlow-Control-artifacts")
OUT=ART/"experiments/chair_condition_recipe_factorial_v1"
OLD=ART/"experiments/chair_early_alignment_schedule40_v1"
OFFICIAL=Path("/mnt/e/MeshFlow/native_t1/runs/control_mvp48_v1/checkpoints/official_chair_last.pt")
RECIPES={"R2_MIX_H_ALL":"all","R3_40_H_LATE":"late","R4_MIX_H_LATE":"late"}
def read(path):return json.loads(Path(path).read_text())
def checked(path,expected):
    actual=file_sha256(path)
    if actual!=expected:raise ValueError("Protected identity differs: "+str(path))
    return actual
def source_registration():
    dest=OUT/"registrations/source_hashes.json"
    if dest.exists():
        sources=read(dest)
        for name,h in sources.items():checked(REPO/name,h)
    else:
        files=sorted(list((REPO/"src").rglob("*.py"))+list((REPO/"tools").glob("*.py"))+list((REPO/"tests").glob("*recipe_factorial*.py")))
        sources={str(p.relative_to(REPO)):file_sha256(p) for p in files}
        freeze(dest,sources)
        for name in sources:
            target=OUT/"source_snapshot"/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(REPO/name,target)
    return sources
def r1_reuse_audit():
    done=read(OLD/"training/FM_ONLY/completed.json")
    common=read(OLD/"training/COMMON/completed.json")
    plan=read(OUT/"streams/recipe_plan.json")
    if done["status"]!="COMPLETE" or done["step"]!=6000 or common["step"]!=5000:raise ValueError("R1 trajectory incomplete")
    expected={"data_manifest_sha256":"9ec06813f8bfb3a51bd6ad0b9b6b4be33e5466064e0026f4b010177da596d6b2",
              "plan_sha256":"7f4804c9bc24c320c71e2e5bac676372572ac2b39a74c1f9eb41ddce8bd7e0e1",
              "paired_inputs_sha256":"97b01c6b74f394d585720d6b2383a181728152ca5f502b8c3ad2cb3034167869"}
    for k,h in expected.items():
        if done["identities"][k]!=h:raise ValueError("Old R1 input identity differs: "+k)
    if plan["base_plan_sha256"]!=expected["plan_sha256"] or plan["base_paired_inputs_sha256"]!=expected["paired_inputs_sha256"]:raise ValueError("New experiment changed R1 basis")
    checked(ART/"data/train_manifest.json",expected["data_manifest_sha256"])
    checked(OLD/"stream/train_plan.json",expected["plan_sha256"])
    checked(OLD/"stream/paired_inputs.json",expected["paired_inputs_sha256"])
    checked(OFFICIAL,"bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d")
    for info in (done["checkpoint"],done["inference_export"],common["checkpoint"]):checked(info["path"],info["sha256"])
    lineage=done["lineage"]
    if lineage["generator_initial_sha256"]!="792ad8d6249ed4ccd1b5c5a81dfba0e527fc61911be96144729f8a2fc7f4811a" or not lineage["fresh_AdamW"]:raise ValueError("R1 initialization differs")
    result=dict(status="PASS",label="R1_40_H_ALL",old_label="FM_ONLY",new_training_updates=0,reused_training_updates=6000,
        immutable_old_receipts={str(p):file_sha256(p) for p in [OLD/"training/COMMON/completed.json",OLD/"training/FM_ONLY/completed.json",OLD/"protocol.json"]},
        input_identities=expected,lineage=lineage,inference_export=done["inference_export"],
        rationale="Original COMMON1..5000 plus uninterrupted FM_ONLY5001..6000; same conditional initialization, optimizer state, input40 stream, HYBRID and FM arithmetic.")
    freeze(OUT/"audit/R1_reuse.json",result);return result
def run_worker(script,args):
    cmd=[sys.executable,"-B",str(REPO/"tools"/script),*map(str,args)]
    event(OUT,"STAGE_COMMAND",command=cmd)
    start=time.perf_counter()
    env=dict(os.environ)
    if script=="evaluate_recipe_factorial.py" and "score" in args:
        env.update(OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2")
    r=subprocess.run(cmd,cwd=REPO,env=env)
    event(OUT,"STAGE_RETURN",command=cmd,returncode=r.returncode,wall_seconds=time.perf_counter()-start)
    if r.returncode:raise RuntimeError("Stage failed: "+script+" code="+str(r.returncode))
def prepare():
    from meshflow_control.data.recipe_factorial import prepare as data_prepare,precompute
    from evaluate_recipe_factorial import prepare_inputs,register_phase
    snapshot(OUT,"before_CPU_prepare")
    source_registration()
    data_prepare(ART/"data",OLD,OUT/"streams")
    receipt=OUT/"streams/precompute_receipt.json"
    if not receipt.exists():
        progress=OUT/"streams/precompute_progress.json"
        if progress.exists() and read(progress).get("status")=="RUNNING":
            raise RuntimeError("Existing CPU OT owner/attempt needs completion review; refuse duplicate precompute")
        precompute(ART/"data",OUT/"streams",workers=4)
    pre=read(receipt)
    if pre["status"]!="PASS" or pre["plan_sha256"]!=file_sha256(OUT/"streams/recipe_plan.json") or pre["slots"]!=48000:raise ValueError("Incomplete precompute")
    r1_reuse_audit()
    tables={str(p.relative_to(OUT)):file_sha256(p) for p in sorted((OUT/"streams/paired_inputs").glob("*.json"))}
    freeze(OUT/"registrations/paired_hashes.json",tables)
    prepare_inputs(OUT,ART)
    models=[
        dict(label="START5000",loader="start",checkpoint_path="/mnt/e/MeshFlow/native_t1/checkpoints/chair_hybrid_start_total5000.pt"),
        dict(label="HIST_FM6000",loader="historical",checkpoint_path=str(ART/"experiments/chair_contextual_alignment_sie_v1/inference/A_generator_step1000.pt")),
        dict(label="R1_40_H_ALL",loader="schedule40",checkpoint_path=str(OLD/"inference/FM_ONLY_generator_step6000.pt"))]
    models += [dict(label=r,loader="recipe_factorial",checkpoint_path=str(OUT/"inference"/(r+"_generator_step6000.pt"))) for r in RECIPES]
    freeze(OUT/"registrations/models_main.json",models);register_phase(OUT,"main",models)
    return dict(status="COMPLETE",new_OT=pre["actual_new20_OT_returns"],registered_main_outputs=384)
def register_training():
    sources=source_registration()
    registration=dict(schema="chair_condition_recipe_factorial_training_authorization_v1",
        experiment="CHAIR_CONDITION_RECIPE_FACTORIAL_V1",namespace=read(OUT/"streams/recipe_plan.json")["namespace"],
        out=str(OUT),experiment_root=str(OUT),authorized_recipes=list(RECIPES),hybrid_schedules=RECIPES,
        updates_per_recipe=6000,effective_batch=8,microbatch=1,source_hashes=sources)
    path=OUT/"registrations/stage1_training.json";freeze(path,registration)
    jobs=[]
    for label,schedule in RECIPES.items():
        folder=OUT/"training"/label
        jobs.append(dict(name=label,phase="training",status_root=str(folder),progress_path=str(folder/"progress.json"),
            completed_path=str(folder/"completed.json"),latest_checkpoint=str(folder/"latest.pt"),latest_receipt=str(folder/"latest.json"),
            required_completion_files=[str(OUT/"inference"/(label+"_generator_step6000.pt"))],
            command=[sys.executable,"-B",str(REPO/"tools/train_recipe_factorial.py"),"--recipe",label,"--hybrid-schedule",schedule,
                "--official",str(OFFICIAL),"--data-manifest",str(ART/"data/train_manifest.json"),"--stream-root",str(OUT/"streams"),
                "--out",str(OUT),"--registration",str(path)]))
    wave=OUT/"registrations/main_training_jobs.json"
    freeze(wave,dict(schema="recipe_factorial_dispatch_wave_v1",wave="main_training",max_workers=2,jobs=jobs))
    return wave
def verify_training():
    expected=read(OUT/"audit/R1_reuse.json")["lineage"]["generator_initial_sha256"]
    for label in RECIPES:
        d=read(OUT/"training"/label/"completed.json")
        if d["status"]!="COMPLETE" or d["step"]!=6000 or d["lineage"]["generator_initial_sha256"]!=expected:raise ValueError("New recipe completion/initialization differs")
        for key in ("checkpoint","inference_export"):checked(d[key]["path"],d[key]["sha256"])
    return dict(status="COMPLETE",new_unique_updates=18000,reused_R1_updates=6000)
def train():
    prepare()
    wave=register_training()
    run_worker("dispatch_recipe_factorial.py",["--out",OUT,"--jobs",wave])
    return verify_training()
def sample():
    verify_training();source_registration()
    from evaluate_recipe_factorial import bind_checkpoint,reuse_outputs,LABELS
    historical=[]
    for label in ("START5000","HIST_FM6000","FM_ONLY"):
        for p in sorted((OLD/"outputs"/label).glob("*/receipt.json")):historical.append(read(p))
    cache=OUT/"registrations/historical_raw_candidates.json";freeze(cache,dict(rows=historical))
    for label,old_label in (("START5000","START5000"),("HIST_FM6000","HIST_FM6000"),("R1_40_H_ALL","FM_ONLY")):
        reuse_outputs(OUT,"main",label,cache,old_label)
    jobs=[]
    for label in LABELS:
        bind_checkpoint(OUT,"main",label)
        folder=OUT/"phases/main/sampling"/label
        jobs.append(dict(name=label,phase="sampling",status_root=str(folder),progress_path=str(folder/"progress.json"),
            completed_path=str(folder/"completed.json"),attempt_option="--attempt",
            command=[sys.executable,"-B",str(REPO/"tools/evaluate_recipe_factorial.py"),"--stage","sample","--out",str(OUT),
                "--phase","main","--label",label,"--controller-lock-owned"]))
    wave=OUT/"registrations/main_sampling_jobs.json"
    freeze(wave,dict(schema="recipe_factorial_dispatch_wave_v1",wave="main_sampling",max_workers=2,jobs=jobs))
    run_worker("dispatch_recipe_factorial.py",["--out",OUT,"--jobs",wave])
    from evaluate_recipe_factorial import completed_rows
    return dict(status="COMPLETE",registered_outputs=len(completed_rows(OUT,"main")))
def evaluate():
    source_registration();snapshot(OUT,"before_CPU_score_render")
    run_worker("evaluate_recipe_factorial.py",["--stage","collect","--out",OUT])
    run_worker("evaluate_recipe_factorial.py",["--stage","score","--out",OUT,"--workers","4"])
    run_worker("evaluate_recipe_factorial.py",["--stage","render","--out",OUT])
    return dict(status="NUMERICS_AND_GALLERY_COMPLETE",visual_review="PENDING",optional_decision="PENDING")
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",required=True,choices=("prepare","train","sample","evaluate","main"))
    a=p.parse_args()
    stages=("prepare","train","sample","evaluate") if a.stage=="main" else (a.stage,)
    for stage in stages:
        event(OUT,"TOTAL_ENTRY_STAGE_BEGIN",stage=stage)
        result=globals()[stage]()
        atomic_json(OUT/"audit"/(stage+"_completion.json"),result)
        event(OUT,"TOTAL_ENTRY_STAGE_END",stage=stage,result=result)
        print(json.dumps(dict(stage=stage,**result)),flush=True)
    return 0
if __name__=="__main__":raise SystemExit(main())