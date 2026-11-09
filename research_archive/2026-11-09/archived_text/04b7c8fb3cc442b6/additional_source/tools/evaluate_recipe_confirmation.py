#!/usr/bin/env python3
"""Future selected-pair confirmation adapter; imports authorize no work.

Only an existing explicit training authorization can materialize n8..11.
The original sampler, per-cell ledger, output checks, scoring and gallery remain
unchanged. Same eight DEV parents with held-out noise are not new-parent evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_recipe_factorial as base

VERSION = "chair_recipe_confirmation_evaluation_adapter_v1"
AUTH_SCHEMA = "chair_recipe_confirmation_training_authorization_v1"
GENERATOR_SCHEMA = "chair_recipe_confirmation_generator_v1"
EXPERIMENT = "CHAIR_CONDITION_RECIPE_FACTORIAL_V1"
NOISE_NAMESPACE = EXPERIMENT + "/confirmation_noise/v1"
NOISES = (8, 9, 10, 11)
RECIPES = ("R1_40_H_ALL", "R2_MIX_H_ALL", "R3_40_H_LATE", "R4_MIX_H_LATE", "C20_40_MIX", "C40_20_MIX")
LOCAL_DEPENDENCIES = (
    "tools/evaluate_recipe_confirmation.py", "tools/evaluate_recipe_factorial.py",
    "tools/score_recipe_factorial_conflict.py", "tools/evaluate_schedule40.py",
    "src/meshflow_control/sampling.py", "src/meshflow_control/runtime.py",
    "src/meshflow_control/mesh_io.py", "src/meshflow_control/evaluation/io.py",
    "src/meshflow_control/evaluation/hashing.py",
)


def check_authorization(out, path):
    """Read the future explicit registration without importing a model module."""
    out = base.root(out); path = base.resolve(path).resolve(); a = base.read_json(path)
    expected = dict(schema=AUTH_SCHEMA, experiment=EXPERIMENT, stage="independent_confirmation",
                    updates_per_recipe=6000, effective_batch=8, microbatch=1, max_training_recipes=2)
    for key, value in expected.items():
        if a.get(key) != value:
            raise ValueError("Confirmation authorization differs: " + key)
    pair = a.get("selected_pair")
    if not isinstance(pair, list) or len(pair) != 2 or len(set(pair)) != 2 or any(r not in RECIPES for r in pair):
        raise ValueError("Exactly two explicitly selected confirmation recipes required")
    if a.get("authorized_recipes") != pair:
        raise ValueError("Selected and authorized pair differ")
    namespace = a.get("namespace", "")
    prefix = EXPERIMENT + "/confirmation/"
    # Same contract as data.recipe_confirmation.validate_namespace, without
    # importing its training-input dependencies in this evaluation adapter.
    if not isinstance(namespace, str) or not namespace.startswith(prefix):
        raise ValueError("A distinct confirmation input namespace is required")
    suffix = namespace[len(prefix):]
    if not suffix or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_/-]*", suffix):
        raise ValueError("Invalid confirmation input namespace suffix")
    schedules = a.get("hybrid_schedules", {})
    if set(schedules) != set(pair):
        raise ValueError("Registered HYBRID schedules must cover pair")
    for recipe in pair:
        fixed = "all" if recipe in RECIPES[:2] else "late" if recipe in RECIPES[2:4] else None
        if schedules[recipe] not in ("all", "late") or (fixed and schedules[recipe] != fixed):
            raise ValueError("Registered HYBRID schedule differs")
    if base.resolve(a["experiment_root"]).resolve() != out:
        raise ValueError("Training and evaluation experiment roots differ")
    training_out = base.resolve(a["out"]).resolve()
    if training_out == out or not training_out.is_relative_to(out):
        raise ValueError("Training confirmation requires its own child directory")
    sources = a.get("source_hashes", {})
    required = {"src/meshflow_control/training/recipe_confirmation.py",
                "src/meshflow_control/data/recipe_confirmation.py", "tools/train_recipe_confirmation.py"}
    if not required.issubset(sources):
        raise ValueError("Training confirmation source registration missing")
    for relative, value in sources.items():
        source = (base.REPO / relative).resolve()
        if not source.is_relative_to(base.REPO) or base.digest(source) != value:
            raise ValueError("Registered confirmation source changed: " + relative)
    return a


def validate_plan(plan, main_rows):
    base.validate_requests(main_rows)
    if plan.get("namespace") != NOISE_NAMESPACE or plan.get("expected_outputs") != 128 or plan.get("candidate_model_identity_in_seed") is not False:
        raise ValueError("Noise preregistration differs")
    if plan.get("new_parent_generalization") is not False:
        raise ValueError("Confirmation uses the same DEV parents")
    main = {(q["parent_id"], q["ratio"]): q for q in main_rows}
    planned = {}
    for q in plan["rows"]:
        key = q["parent_id"], q["ratio"], q["candidate"]
        if key in planned or q["candidate"] not in NOISES or key[:2] not in main:
            raise ValueError("Duplicate or unregistered confirmation noise")
        old = main[key[:2]]
        if any(q[k] != old[k] for k in ("N", "K", "condition_id", "C_sha256")):
            raise ValueError("Confirmation condition differs from main")
        seed_text = NOISE_NAMESPACE + "/" + q["parent_id"] + "/n" + str(q["candidate"]-8)
        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4], "little")
        if q["seed_text"] != seed_text or q["seed"] != seed:
            raise ValueError("Confirmation seed differs from frozen namespace")
        planned[key] = q
    if set(planned) != {(p,r,n) for p,r in main for n in NOISES}:
        raise ValueError("Confirmation requires exactly16 conditions x4 noises")
    for parent in {q["parent_id"] for q in main_rows}:
        if main[parent,.2]["N"] != main[parent,.4]["N"]:
            raise ValueError("Cross-ratio full Gaussian N differs")
    return planned


def collision_audit(new_rows, new_banks, main_rows, main_banks):
    """Compare actual bytes, including absolute free suffixes across K."""
    details = []
    seen = {}
    for q in new_rows:
        z = new_banks[q["request_id"]]["z"]
        if z.dtype != base.np.float32 or z.shape != (1,q["N"],9) or base.np.any(z[:,:q["K"]]):
            raise ValueError("Confirmation actual Gaussian contract differs")
        comparisons = 0
        for old in main_rows:
            if old["N"] != q["N"]:
                continue
            oz = main_banks[old["request_id"]]["z"]
            k = max(q["K"], old["K"])
            comparisons += 1
            if base.byte_equal(z, oz) or base.byte_equal(z[:,k:], oz[:,k:]):
                raise ValueError("Confirmation actual Gaussian collides with main history")
        full_hash = base.array_sha(z)
        if full_hash in seen and seen[full_hash] != (q["parent_id"],q["candidate"]):
            raise ValueError("Distinct holdout draws have identical actual arrays")
        seen[full_hash] = q["parent_id"],q["candidate"]
        details.append(dict(request_id=q["request_id"], Gaussian_sha256=full_hash,
                            full_N_Gaussian_sha256=q.get("noise_full_N_Gaussian_sha256"),
                            full_N_source_definition=q.get("noise_generator"),
                            main_same_N_comparisons=comparisons, status="PASS"))
    return dict(status="PASS", method="actual FP32 full array and common absolute free-slot suffix byte comparisons",
                main_input_files=[dict(request_id=q["request_id"],input_sha256=q["input_sha256"],
                                       Gaussian_sha256=base.array_sha(main_banks[q["request_id"]]["z"])) for q in main_rows],
                rows=details, new_parent_generalization=False)


def materialize_arrays(plan, main_rows):
    """Pure small-array function; called only after explicit pair validation."""
    planned = validate_plan(plan, main_rows)
    main_banks = {q["request_id"]:base.read_input(q) for q in main_rows}
    original = {(q["parent_id"],q["ratio"]):q for q in sorted(main_rows,key=lambda q:q["candidate"],reverse=True)}
    draws = {}; rows = []; banks = {}
    for key,q in sorted(planned.items()):
        parent,ratio,candidate = key; shared = parent,candidate
        if shared not in draws:
            draws[shared] = base.np.random.RandomState(q["seed"]).standard_normal((1,q["N"],9)).astype(base.np.float32)
        z = draws[shared].copy(); z[:,:q["K"]] = 0
        old = original[parent,ratio]; b = main_banks[old["request_id"]]
        row = {k:v for k,v in old.items() if not k.startswith("noise_")}
        rid = parent + "__C" + str(int(ratio*100)) + "__n" + str(candidate)
        row.update(request_id=rid,candidate=candidate,noise_seed=q["seed"],noise_seed_text=q["seed_text"],
                   noise_provenance="REGISTERED_MT19937_FULL_N_THEN_KNOWN_PREFIX_ZERO",
                   noise_generator="numpy RandomState MT19937 standard_normal full[1,N,9] float64 cast FP32 then zero known prefix",
                   noise_full_N_Gaussian_sha256=base.array_sha(draws[shared]),
                   source_main_request_id=old["request_id"],Gaussian_sha256=base.array_sha(z),
                   new_parent_generalization=False)
        rows.append(row); banks[rid] = dict(C=b["C"],z=z,N=b["N"],K=b["K"])
    audit = collision_audit(rows,banks,main_rows,main_banks)
    return rows,banks,audit


def frozen_queries(out, main_requests, main_rows):
    """No query sampler is called: require and verify the existing16 banks."""
    import score_recipe_factorial_conflict as scorer
    saved = scorer.load_registration(out, main_requests)
    by = {r["condition_id"]:r for r in saved["rows"]}
    expected = {q["condition_id"]:q for q in main_rows}
    if len(saved["rows"]) != 16 or set(by) != set(expected):
        raise ValueError("Exactly16 pre-existing C query banks required")
    for cid,row in by.items():
        if row["C_sha256"] != expected[cid]["C_sha256"]:
            raise ValueError("Original C query identity differs")
        scorer.load_queries(row)
    return saved, scorer.query_registration_path(out,main_requests)


def require_completed_selection_panel(out):
    p = base.phase_root(out,"main")
    receipt = base.read_json(p/"evaluation/scoring_receipt.json")
    gallery = base.read_json(p/"gallery/gallery.json")
    if receipt.get("status") != "COMPLETE" or receipt.get("scored") != 384 or receipt.get("registered") != 384:
        raise ValueError("Completed main scoring required before confirmation noise")
    if gallery.get("status") != "COMPLETE" or gallery.get("page_count") != 128 or gallery.get("all_registered_outputs_shown") is not True:
        raise ValueError("Complete main gallery required before confirmation noise")
    return (p/"evaluation/scoring_receipt.json",p/"gallery/gallery.json")


def prepare(out, authorization, phase="confirm"):
    out = base.root(out); p = base.phase_root(out,phase)
    auth = check_authorization(out,authorization)
    selection_files = require_completed_selection_panel(out)
    main_requests = out/"evaluation/requests.json"; plan_path = out/"evaluation/confirmation_plan.json"
    main_rows = base.read_json(main_requests)["rows"]; plan = base.read_json(plan_path)
    validate_plan(plan,main_rows)
    query,query_path = frozen_queries(out,main_requests,main_rows)
    prior = [base.read_json(f) for f in (out/"phases").glob("*/registration.json") if f.parent != p]
    if any(s["kind"] == "confirm" for s in prior) or sum(s["expected_outputs"] for s in prior)+128 > 640:
        raise ValueError("Only one128-output confirmation pair within run ceiling640")
    files = dict(authorization=base.resolve(authorization).resolve(),main_requests=main_requests,
                 noise_plan=plan_path,main_queries=query_path,
                 main_scoring=selection_files[0],main_gallery=selection_files[1])
    binding = dict(version=VERSION,phase=phase,selected_pair=auth["selected_pair"],training_namespace=auth["namespace"],
                   sources={r:base.digest(base.REPO/r) for r in LOCAL_DEPENDENCIES},
                   files={k:dict(path=str(v),sha256=base.digest(v)) for k,v in files.items()})
    base.write_json(p/"preparation_binding.json",binding,freeze=True)
    if (p/"registration.json").exists():
        _,registration,_ = phase_registration(out,phase)
        return registration
    rows,banks,audit = materialize_arrays(plan,main_rows)
    for q in rows:
        path = p/"evaluation/inputs"/(q["request_id"]+".npz")
        base.save_npz(path,banks[q["request_id"]]); q.update(input_path=str(path),input_sha256=base.digest(path))
        base.read_input(q)
    rp = p/"evaluation/requests.json"
    base.write_json(p/"evaluation/holdout_noise_collision_audit.json",audit,freeze=True)
    base.write_json(rp,dict(version=VERSION,rows=rows,holdout_noise_collision_audit_passed=True,
                            noise_plan_path=str(plan_path),noise_plan_sha256=base.digest(plan_path),
                            authorization_sha256=base.digest(authorization),new_parent_generalization=False,
                            model_calls=0,Gaussian_full_N_draws=32),freeze=True)
    import score_recipe_factorial_conflict as scorer
    qr = dict(query); qr["identity"] = dict(query["identity"],requests_sha256=base.digest(rp))
    base.write_json(scorer.query_registration_path(out,rp),qr,freeze=True)
    models = [dict(label=r,recipe=r,loader="recipe_confirmation",
                   checkpoint_path=str(base.resolve(auth["out"])/"inference"/(r+"_generator_step6000.pt")))
              for r in auth["selected_pair"]]
    registration = dict(version=base.VERSION,adapter_version=VERSION,phase=phase,kind="confirm",
        models=models,labels=auth["selected_pair"],requests_path=str(rp),requests_sha256=base.digest(rp),
        registered_per_model=64,expected_outputs=128,nfe_per_output=50,max_cell_attempts=1,max_label_dispatch_attempts=2,
        actual_checkpoint_hashes="Separately frozen before first sample",new_parent_generalization=False,
        preparation_binding_sha256=base.digest(p/"preparation_binding.json"),
        query_registration_sha256=base.digest(scorer.query_registration_path(out,rp)),
        collision_audit_sha256=base.digest(p/"evaluation/holdout_noise_collision_audit.json"))
    base.write_json(p/"registration.json",registration,freeze=True)
    phase_registration(out,phase)
    return registration


def phase_registration(out, phase):
    p,s,qs = base.phase_registration(out,phase)
    if s.get("adapter_version") != VERSION or s["kind"] != "confirm" or s["expected_outputs"] != 128:
        raise ValueError("Not this independent confirmation registration")
    binding_path = p/"preparation_binding.json"; binding = base.read_json(binding_path)
    if base.digest(binding_path) != s["preparation_binding_sha256"]:
        raise ValueError("Confirmation preparation binding changed")
    for relative,h in binding["sources"].items():
        if base.digest(base.REPO/relative) != h:
            raise ValueError("Confirmation evaluation source changed: " + relative)
    for item in binding["files"].values():
        if base.digest(item["path"]) != item["sha256"]:
            raise ValueError("Frozen confirmation source input changed")
    auth = check_authorization(out,binding["files"]["authorization"]["path"])
    expected_models = [dict(label=r,recipe=r,loader="recipe_confirmation",
                            checkpoint_path=str(base.resolve(auth["out"])/"inference"/(r+"_generator_step6000.pt")))
                       for r in auth["selected_pair"]]
    if s["models"] != expected_models:
        raise ValueError("Confirmation generator export registration differs")
    if s["labels"] != auth["selected_pair"] or len(qs) != 64 or {q["candidate"] for q in qs} != set(NOISES):
        raise ValueError("Confirmation pair/noise coverage differs")
    main_rows = base.read_json(binding["files"]["main_requests"]["path"])["rows"]
    plan = base.read_json(binding["files"]["noise_plan"]["path"])
    planned = validate_plan(plan,main_rows)
    if {(q["parent_id"],q["ratio"],q["candidate"]) for q in qs} != set(planned):
        raise ValueError("Confirmation request identities differ")
    for q in qs:
        expected = planned[q["parent_id"],q["ratio"],q["candidate"]]
        if any(q[k] != expected[k] for k in ("N","K","condition_id","C_sha256")) or q["noise_seed"] != expected["seed"]:
            raise ValueError("Confirmation input identity differs")
        base.read_input(q)
    import score_recipe_factorial_conflict as scorer
    qr_path = scorer.query_registration_path(out,s["requests_path"])
    if base.digest(qr_path) != s["query_registration_sha256"]:
        raise ValueError("Confirmation query registration changed")
    current = scorer.load_registration(out,s["requests_path"])
    old,_ = frozen_queries(out,binding["files"]["main_requests"]["path"],main_rows)
    if current["rows"] != old["rows"]:
        raise ValueError("Confirmation must exactly reuse original query files and arrays")
    if base.digest(p/"evaluation/holdout_noise_collision_audit.json") != s["collision_audit_sha256"]:
        raise ValueError("Holdout actual-array collision audit changed")
    return p,s,qs


def validate_generator_identity(identity, auth, recipe, authorization_sha256):
    if identity.get("schema") != GENERATOR_SCHEMA or identity.get("experiment") != EXPERIMENT or identity.get("global_step") != 6000:
        raise ValueError("Wrong confirmation generator schema/endpoint")
    if identity.get("recipe") != recipe or identity.get("segment") != recipe or recipe not in auth["selected_pair"]:
        raise ValueError("Confirmation generator recipe differs")
    protocol = identity["protocol"]; ids = identity["identities"]
    if (protocol.get("input_namespace") != auth["namespace"] or protocol.get("training_stage") != "independent_confirmation"
        or protocol.get("independent_training_input_stream") is not True
        or protocol.get("independent_model_initialization") is not False
        or protocol.get("hybrid_schedule") != auth["hybrid_schedules"][recipe]):
        raise ValueError("Confirmation generator training protocol differs")
    if ids.get("authorization_sha256") != authorization_sha256 or ids.get("selected_pair") != auth["selected_pair"] or ids.get("confirmation_namespace") != auth["namespace"]:
        raise ValueError("Confirmation generator authorization lineage differs")
    return identity


def sample_label(out, phase, label, attempt="initial", controller_lock_owned=False):
    if not controller_lock_owned:
        raise RuntimeError("Sampling requires resource-aware controller owning shared GPU lock")
    if attempt not in ("initial","serial_retry","oom_serial_retry"):
        raise ValueError("At most initial and one serial retry dispatch")
    p,s,qs = phase_registration(out,phase)
    if label not in s["labels"]:
        raise ValueError("Unregistered confirmation label")
    binding = base.bind_checkpoint(out,phase,label)
    preparation = base.read_json(p/"preparation_binding.json")
    auth = check_authorization(out,preparation["files"]["authorization"]["path"])
    folder = p/"sampling"/label; folder.mkdir(parents=True,exist_ok=True)
    if attempt != "initial" and not (folder/"initial_failure.json").exists():
        raise ValueError("Serial retry requires previous failed dispatch")
    dispatch_key = "initial" if attempt == "initial" else "serial_retry"
    with (folder/(dispatch_key+"_dispatch.json")).open("x",encoding="utf8") as f:
        json.dump(dict(binding,attempt=attempt,started=time.time()),f); f.flush(); os.fsync(f.fileno())
    completed=[];blocked=[];pending=[];tick=time.perf_counter()
    try:
        for q in qs:
            dst=p/"outputs"/label/q["request_id"]
            if (dst/"receipt.json").exists():
                r=base.read_json(dst/"receipt.json");base.verify_output(q,r,binding["checkpoint_sha256"]);completed.append(r)
            elif (dst/"calls.jsonl").exists():
                blocked.append(q["request_id"])
            else:
                base.read_input(q);pending.append(q)
        if pending:
            import torch
            from meshflow_control import runtime
            from meshflow_control.sampling import sample
            from meshflow_control.training.recipe_confirmation import load_generator
            torch.set_num_threads(2);runtime.configure_stable_runtime()
            model,identity=load_generator(base.resolve(binding["checkpoint_path"]),device="cuda",expected_sha256=binding["checkpoint_sha256"])
            validate_generator_identity(identity,auth,label,preparation["files"]["authorization"]["sha256"])
            base.write_json(p/"checkpoints"/(label+"_generator_identity.json"),identity,freeze=True)
            for q in pending:
                dst=p/"outputs"/label/q["request_id"];b=base.read_input(q)
                context=dict(label=label,request_id=q["request_id"],attempt=attempt,phase=phase,
                             checkpoint_sha256=binding["checkpoint_sha256"],input_sha256=q["input_sha256"])
                budget=base.CellBudget(dst/"calls.jsonl",context)
                _,_,audit=sample(model,b["C"],q["N"],q["noise_seed"],b["z"],budget=budget,out=dst,context=context)
                r=dict(q,label=label,status="COMPLETE",checkpoint_sha256=binding["checkpoint_sha256"],
                       generation_status="NEW_EULER50",new_rollouts=1,attempt=attempt,
                       **{k:v for name,file in (("raw","raw.npz"),("ply","raw.ply"),("result","run.json"))
                          for k,v in ((name+"_path",str(dst/file)),(name+"_sha256",base.digest(dst/file)))})
                _,facts=base.verify_output(q,r,binding["checkpoint_sha256"]);r["contracts"]=facts
                base.write_json(dst/"receipt.json",r,freeze=True);completed.append(r)
                base.write_json(folder/"progress.json",dict(status="RUNNING",label=label,attempt=attempt,completed=len(completed),blocked=blocked,
                    elapsed_seconds=time.perf_counter()-tick,**{k:audit[k] for k in ("cuda_peak_allocated_bytes","cuda_peak_reserved_bytes","peak_scope")}))
        result=dict(status="COMPLETE" if len(completed)==len(qs) else "PARTIAL_BLOCKED",label=label,rows=completed,
                    blocked=blocked,registered=len(qs),seconds=time.perf_counter()-tick)
        base.write_json(folder/"completed.json",result);return result
    except BaseException as exc:
        result=dict(status="OOM" if "out of memory" in str(exc).lower() else "FAILED",label=label,attempt=attempt,
                    completed=len(completed),blocked=blocked,error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-tick)
        base.write_json(folder/(attempt+"_failure.json"),result);raise


def after_raw(out, phase, stage, workers=2):
    phase_registration(out,phase)
    rows=base.completed_rows(out,phase)  # Both models x64 before any score/render.
    if stage=="collect":
        return dict(status="COMPLETE",outputs=len(rows))
    if stage=="render":
        return base.render(out,phase)
    if stage=="score":
        from score_recipe_factorial_conflict import score
        return score(out,phase,workers)
    raise ValueError("Unknown completed-RAW stage")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",required=True,choices=("prepare","bind-checkpoint","sample","collect","score","render"))
    p.add_argument("--out",default=base.DEFAULT_OUT);p.add_argument("--phase",default="confirm")
    p.add_argument("--authorization");p.add_argument("--label");p.add_argument("--attempt",default="initial")
    p.add_argument("--workers",type=int,default=2);p.add_argument("--controller-lock-owned",action="store_true")
    args=p.parse_args(argv)
    if args.stage=="prepare":
        if not args.authorization:p.error("--authorization is required for prepare")
        result=prepare(args.out,args.authorization,args.phase)
    elif args.stage=="sample":
        result=sample_label(args.out,args.phase,args.label,args.attempt,args.controller_lock_owned)
    elif args.stage=="bind-checkpoint":
        phase_registration(args.out,args.phase);result=base.bind_checkpoint(args.out,args.phase,args.label)
    else:result=after_raw(args.out,args.phase,args.stage,args.workers)
    print(json.dumps(result),flush=True)
    return 2 if result.get("status") in ("PARTIAL_BLOCKED","FAILED") else 0


if __name__=="__main__":
    raise SystemExit(main())

