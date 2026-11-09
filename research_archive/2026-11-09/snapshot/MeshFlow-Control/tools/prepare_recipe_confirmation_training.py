#!/usr/bin/env python3
"""Register/finalize the approved confirmation training pair; never execute workers."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
VERSION = "recipe_confirmation_training_preparation_v1"
EXPERIMENT = "CHAIR_CONDITION_RECIPE_FACTORIAL_V1"
PAIR = ["C20_40_MIX", "R2_MIX_H_ALL"]
NAMESPACE = EXPERIMENT + "/confirmation/input_repeat1"
OFFICIAL_SHA = "bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d"
INITIAL_SHA = "792ad8d6249ed4ccd1b5c5a81dfba0e527fc61911be96144729f8a2fc7f4811a"
MIN_FREE_BYTES = 64 * 1024**3

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError("Frozen registration differs: " + str(path))
        return digest(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    return digest(path)

def validate_decision(value, path):
    expected = dict(schema="recipe_confirmation_decision_v1", status="APPROVED",
        experiment=EXPERIMENT, stage="independent_confirmation", namespace=NAMESPACE,
        selected_pair=PAIR, hybrid_schedules={r:"all" for r in PAIR},
        updates_per_recipe=6000, effective_batch=8, microbatch=1, max_training_recipes=2,
        max_workers=2, checkpoint_interval=100, max_resume_per_recipe=1,
        max_replay_completed_updates_per_failed_attempt=99,
        evaluation_noises=[8,9,10,11], default_changed=False, automatic_git_operation=False)
    for key, val in expected.items():
        if value.get(key) != val:
            raise ValueError("Decision differs: " + key)
    root, out = Path(value["experiment_root"]).resolve(), Path(value["out"]).resolve()
    if out != root / "confirmation" or Path(value["stream_root"]).resolve() != out / "streams":
        raise ValueError("Independent confirmation output/stream path differs")
    if Path(path).resolve() != root / "registrations/confirmation_decision.json":
        raise ValueError("Decision must be the registered experiment decision")
    if Path(value["decision_markdown"]).resolve() != root / "registrations/CONFIRMATION_DECISION.md":
        raise ValueError("Decision Markdown binding differs")
    # The executable may be a venv symlink; resolve both without changing argv.
    if Path(value["python"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("Decision interpreter differs from current interpreter")
    for key in ("official", "data_manifest", "decision_markdown"):
        if not Path(value[key]).is_file():
            raise ValueError("Decision input missing: " + key)
    return value

def registration_paths(decision):
    root = Path(decision["experiment_root"])
    return (root/"registrations/confirmation_training.json",
            root/"registrations/confirmation_training_jobs.json",
            root/"registrations/confirmation_training_preparation.json",
            root/"registrations/confirmation_training_ready.json")

def prior_frozen_sources(root):
    """Check legacy freezes without adding or changing their contents."""
    files = [root/"registrations/source_hashes.json",
        root/"registrations/curriculum_additional_source_hashes.json",
        root/"registrations/all_baselines_analysis_sources.json",
        root/"registrations/all_baselines_analysis_v2_sources.json"]
    merged = {}
    for p in files:
        value = read(p)
        if "sources" in value:
            values=value["sources"]
        elif "tool_path" in value:
            values={"tools/"+Path(value["tool_path"].replace("\\","/")).name:value["tool_sha256"],
                    "tests/"+Path(value["test_path"].replace("\\","/")).name:value["test_sha256"]}
        else:
            values=value
        for name, sha in values.items():
            source = (REPO/name).resolve()
            if not source.is_relative_to(REPO) or not isinstance(sha,str) or digest(source) != sha:
                raise ValueError("Prior frozen source changed: " + name)
            if name in merged and merged[name] != sha:
                raise ValueError("Conflicting source freeze: " + name)
            merged[name] = sha
    return merged, {str(p):digest(p) for p in files}

def source_hashes():
    # Import metadata validators only. Never construct/load a model or create an optimizer.
    from meshflow_control.training import recipe_confirmation as worker
    values = {"src/meshflow_control/"+k:v for k,v in worker.source_identity().items()}
    for name in ("tools/train_recipe_confirmation.py", "tools/dispatch_recipe_factorial.py",
                 "tools/prepare_recipe_confirmation_training.py"):
        values[name] = digest(REPO/name)
    return values

def authorization(decision, decision_path, sources):
    return dict(schema="chair_recipe_confirmation_training_authorization_v1",
        experiment=EXPERIMENT, stage="independent_confirmation", namespace=NAMESPACE,
        selected_pair=PAIR, authorized_recipes=PAIR, hybrid_schedules={r:"all" for r in PAIR},
        experiment_root=decision["experiment_root"], out=decision["out"],
        updates_per_recipe=6000, effective_batch=8, microbatch=1, max_training_recipes=2,
        source_hashes=sources, decision_path=str(Path(decision_path).resolve()),
        decision_sha256=digest(decision_path), decision_markdown_sha256=digest(decision["decision_markdown"]),
        official_file_sha256=OFFICIAL_SHA, expected_generator_initial_sha256=INITIAL_SHA,
        initialization="Official EMA; Geo1010; identical initial model; fresh AdamW step0",
        recovery_contract=dict(checkpoint_interval=100, maximum_resumes_per_recipe=1,
            maximum_replayed_completed_updates_per_failed_attempt=99,
            eligibility="Only parallel-exposed OOM; downgrade once; serial OOM or numerical/code failure blocks",
            partial_failed_calls_always_counted=True),
        checkpoint_policy="Atomic latest only; final hard link and generator-only export; no extra long-term intermediate copies",
        unique_updates=12000, unique_microbatch_forwards=96000, unique_microbatch_backwards=96000,
        evaluation_noises=[8,9,10,11],
        evaluation_registration_by_separate_tool=True)

def storage(decision):
    root=Path(decision["experiment_root"])
    usage=shutil.disk_usage(root)
    cp=root/"training/C20_40_MIX/latest.pt"
    exp=root/"inference/C20_40_MIX_generator_step6000.pt"
    cb=cp.stat().st_size if cp.is_file() else 1565060843
    eb=exp.stat().st_size if exp.is_file() else 521669499
    # Four checkpoint-size allocations allow both latest+atomic tmp simultaneously.
    # Final endpoints are hard links; reserve duplicate export/temp space conservatively.
    estimate=4*cb+4*eb
    result=dict(free_bytes=usage.free,total_bytes=usage.total,used_bytes=usage.used,
        minimum_required_free_bytes=MIN_FREE_BYTES,comparable_checkpoint_bytes=cb,
        comparable_generator_export_bytes=eb,conservative_training_peak_new_bytes=estimate,
        additional_OT_logs_sampling_headroom_bytes=MIN_FREE_BYTES-estimate,
        final_checkpoint_hardlinked=True,no_extra_weight_copy=True)
    if usage.free < MIN_FREE_BYTES:
        raise ValueError("Insufficient storage headroom: " + str(result))
    return result

def register(decision_path):
    decision_path=Path(decision_path).resolve()
    d=validate_decision(read(decision_path),decision_path)
    old, freezes=prior_frozen_sources(Path(d["experiment_root"]))
    sources=source_hashes()
    for name,sha in sources.items():
        if name in old and old[name] != sha:
            raise ValueError("Training dependency differs from prior freeze: "+name)
    if digest(d["official"]) != OFFICIAL_SHA:
        raise ValueError("Official EMA file identity differs")
    ap,jp,pp,rp=registration_paths(d)
    auth=authorization(d,decision_path,sources)
    freeze(ap,auth)
    from meshflow_control.training.recipe_confirmation import check_authorization
    for recipe in PAIR:
        check_authorization(ap,recipe,"all",NAMESPACE,d["out"])
    registration=dict(version=VERSION,status="REGISTERED_AWAITING_PAIRED_INPUTS",
        authorization_path=str(ap),authorization_sha256=digest(ap),
        decision_path=str(decision_path),decision_sha256=digest(decision_path),
        old_freeze_registry_hashes=freezes,prior_frozen_sources_checked=len(old),
        source_hashes=sources,official_sha256=OFFICIAL_SHA,
        data_manifest_sha256=digest(d["data_manifest"]),
        stream_root=d["stream_root"],worker_out=d["out"],
        dispatcher_out=d["experiment_root"],jobs_path=str(jp),
        worker_construction_or_training_calls=0,OT_calls=0)
    freeze(pp,registration)
    return dict(status="PREPARED_NOT_DISPATCHED",authorization=str(ap),
        pending="CPU confirmation plan/paired inputs and finalize" if not rp.exists() else None,
        jobs_exist=jp.exists(),storage=storage(d),registration=registration)

def pair_contract(document, label, plan, plan_sha, data_sha, implementation_sha, records, total=6000):
    expected=dict(schema="meshflow_recipe_confirmation_paired_inputs_v1",recipe=label,
        namespace=NAMESPACE,selected_pair=PAIR,hybrid_schedule="all",
        plan_sha256=plan_sha,data_manifest_sha256=data_sha,
        implementation_sha256=implementation_sha,historical_paired_inputs_reused=False)
    for k,v in expected.items():
        if document.get(k)!=v:raise ValueError("Paired metadata differs: "+k)
    rows=document.get("records",[])
    if len(rows)!=total*8 or len(records)!=total*8:
        raise ValueError("Paired sample denominator differs")
    if sorted(r["sample_index"] for r in rows)!=list(range(total*8)):
        raise ValueError("Paired origin multiset differs")
    signatures={}
    for row,record in zip(rows,records):
        for k in ("step","uid","task_id","N","K","ratio","alpha","origin_sample_index"):
            if row.get(k)!=record.get(k):raise ValueError("Paired order/input differs: "+k)
        if row["sample_index"]!=record["origin_sample_index"] or row.get("lambda_value")!=.25:
            raise ValueError("Paired origin or H_ALL differs")
        if row.get("free_coordinate_count")!=(row["N"]-row["K"])*9:
            raise ValueError("Free scalar exposure differs")
        for k in ("shared_input_sha256","input_bytehash","label_bytehash","ot_cache_key"):
            s=row.get(k)
            if not isinstance(s,str) or len(s)!=64 or any(c not in "0123456789abcdef" for c in s):
                raise ValueError("Invalid paired digest: "+k)
        # step intentionally omitted: the course is a reordering of the same inputs.
        signatures[row["sample_index"]]={k:v for k,v in row.items() if k!="step"}
    return signatures

def ready_inputs(d,auth):
    from meshflow_control.data.recipe_confirmation import ConfirmationStream, records_for_recipe
    from meshflow_control.training.recipe_confirmation import validate_confirmation_plan
    sr=Path(d["stream_root"]);plan_path=sr/"confirmation_plan.json"
    pre_path=sr/"precompute_receipt.json"
    required=[plan_path,pre_path]+[sr/"paired_inputs"/(r+".json") for r in PAIR]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise FileNotFoundError("PENDING confirmation inputs: "+", ".join(missing))
    # Constructor reproduces plan from metadata only: no sample()/OT/model call.
    stream=ConfirmationStream(Path(d["data_manifest"]).parent,sr,PAIR[0],hybrid_schedule="all",require_cache=True)
    plan=stream.plan
    for recipe in PAIR:validate_confirmation_plan(plan,auth,recipe)
    if plan.get("sample_count")!=48000 or plan.get("no_historical_pair_or_OT_reuse") is not True:
        raise ValueError("Confirmation sample/reuse contract differs")
    pre=read(pre_path)
    expected=dict(status="PASS",namespace=NAMESPACE,selected_pair=PAIR,
        plan_sha256=digest(plan_path),registered_unique_OT_inputs=48000,
        completed_unique_OT_inputs=48000,historical_OT_reused=0,
        paired_input_rows_per_recipe={r:48000 for r in PAIR},model_calls=0,optimizer_updates=0)
    for k,v in expected.items():
        if pre.get(k)!=v:raise ValueError("CPU precompute receipt differs: "+k)
    if pre.get("actual_OT_attempts")!=pre.get("actual_OT_returns") or pre["actual_OT_returns"]+pre["OT_cache_hits"]!=48000:
        raise ValueError("OT attempted/returned/cache-hit accounting differs")
    reference=None
    for label in PAIR:
        path=sr/"paired_inputs"/(label+".json")
        records=records_for_recipe(plan,label)
        current=pair_contract(read(path),label,plan,digest(plan_path),digest(d["data_manifest"]),stream.implementation_sha256,records)
        if reference is None:reference=current
        elif current!=reference:raise ValueError("Selected pair does not share the exact input multiset")
    orders=plan["curriculum_orders"][PAIR[0]]
    if orders[32000:]!=list(range(32000,48000)):
        raise ValueError("Final2000 MIX suffix changed")
    return {str(p):digest(p) for p in required},dict(status="PASS",paired_rows_per_recipe=48000,
        identical_input_multiset=True,final2000_suffix_identical=True,
        first2000="20%",middle2000="40%",last2000="MIX",lambda_all=.25,
        cache_payloads_not_recomputed=True,precompute_authority="Frozen receipt plus paired byte hashes; worker requires its registered cache")

def wave(d,auth_path,identities):
    jobs=[]
    for label in PAIR:
        folder=Path(d["out"])/"training"/label
        jobs.append(dict(name=label,phase="training",status_root=str(folder),
            progress_path=str(folder/"progress.json"),completed_path=str(folder/"completed.json"),
            latest_checkpoint=str(folder/"latest.pt"),latest_receipt=str(folder/"latest.json"),
            required_completion_files=[str(Path(d["out"])/"checkpoints"/(label+"_step6000.pt")),
                str(Path(d["out"])/"inference"/(label+"_generator_step6000.pt"))],
            source_hashes=identities,command=[sys.executable,"-B",str(REPO/"tools/train_recipe_confirmation.py"),
                "--recipe",label,"--hybrid-schedule","all","--official",d["official"],
                "--data-manifest",d["data_manifest"],"--stream-root",d["stream_root"],
                "--out",d["out"],"--registration",str(auth_path)]))
    return dict(schema="recipe_factorial_dispatch_wave_v1",wave="confirmation_training_input_repeat1",
        max_workers=2,jobs=jobs)

def finalize(decision_path,verify=False):
    prepared=register(decision_path)
    d=validate_decision(read(decision_path),decision_path)
    ap,jp,pp,rp=registration_paths(d);auth=read(ap)
    inputs,pair_audit=ready_inputs(d,auth)
    identities={str(REPO/k):v for k,v in auth["source_hashes"].items()}
    identities.update(inputs)
    identities.update({str(Path(decision_path).resolve()):digest(decision_path),
        str(ap):digest(ap),str(pp):digest(pp),d["data_manifest"]:digest(d["data_manifest"]),
        d["decision_markdown"]:digest(d["decision_markdown"])})
    value=wave(d,ap,identities)
    readiness=dict(version=VERSION,status="READY_NOT_DISPATCHED",
        authorization_path=str(ap),authorization_sha256=digest(ap),
        input_and_source_hashes=identities,pair_audit=pair_audit,
        dispatcher_out=d["experiment_root"],worker_out=d["out"],
        model_calls=0,training_updates=0,OT_calls=0)
    if verify:
        if read(jp)!=value or read(rp)!=readiness:raise ValueError("Finalized preparation changed")
    else:
        # No callable dispatch job is published until every prerequisite has passed.
        freeze(rp,readiness);freeze(jp,value)
    return dict(status="VERIFIED_READY_NOT_DISPATCHED" if verify else "READY_NOT_DISPATCHED",
        authorization=str(ap),jobs=str(jp),pair_audit=pair_audit,storage=prepared["storage"],
        dispatcher_command=[sys.executable,"-B",str(REPO/"tools/dispatch_recipe_factorial.py"),
            "--out",d["experiment_root"],"--jobs",str(jp)])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decision",required=True)
    p.add_argument("--stage",choices=("register","finalize","verify"),default="register")
    a=p.parse_args(argv)
    result=register(a.decision) if a.stage=="register" else finalize(a.decision,a.stage=="verify")
    print(json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
