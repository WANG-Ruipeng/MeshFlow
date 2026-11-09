"""Two authorized 200-update branches from EXPOSURE global25500.

CPU preparation freezes an exact replay of its first 200 training batches.
FLIP200 alternates whole CLEAN/X_WIDTH pairs per condition; PLAIN200 does not
flip. No new RNG stream, model probe, head update, or automatic retry exists.
GPU training additionally requires a parent resource-dispatch GO for that arm.
Imports remain CPU-only without Torch until an incomplete authorized arm launches.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise RuntimeError("Conflicting CUBLAS_WORKSPACE_CONFIG")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

from .common import ROOT, OUT, read, resolve, sha, write

ARMS = ("PLAIN200", "FLIP200")
UPDATES = 200
BASE_STEP = 25500
NATIVE_BASE = 13500
STAGE = "NATIVE"
EXPERIMENT = "R2_BENCH_C_EDGEFLIP_TRAIN200_V1"
SCHEMA = "bench_c_edgeflip_training_v1"
INFERENCE_SCHEMA = "bench_c_edgeflip_generator_v1"
OLD = ROOT / "experiments/r2_bench_four_routes_v1/exposure"
SOURCE = OLD / "training/M_PLUS3000/checkpoint_step3000.pt"
SOURCE_IDENTITIES = dict(
    source_checkpoint_sha256="3c1e9e1db0f9a462632bd70a91b5acaa4bef1dd40ad20ad90a194ab17bbd49fd",
    source_generator_sha256="76c42158b9016e58cc235b859f6cd5bc68ff1f4e19f3c70e4f94f676447170f4",
    source_optimizer_sha256="52d3145ba8012c556e4c627fe88c3b7b6c2ae7b9c6f7c177207cd7dec3640beb",
    source_rng_sha256="824e3e71a9ca6cd66d3f042a4c5b0746b63968c8f185d377af10f044e5de66e8")
LOSS_NORMALIZATION = "narrow_reference_group_MSE_coefficients_v1"
REGISTRATION = OUT / "training_registration.json"
FEASIBILITY = OUT / "training_preparation/feasibility.json"
CPU_GATE = OUT / "training_preparation/runtime_cpu_gate.json"


def limits():
    return dict(train_forward=1600, train_backward=1600, optimizer_update=200,
                head_forward=0, head_backward=0, head_optimizer_update=0, probe_forward=0)


def cursor(step):
    if type(step) is not int or not 0 <= step <= UPDATES:
        raise ValueError("Branch cursor is outside the exact 200-update budget")
    return dict(stage=STAGE, stage_step=step, local_step=step,
                curriculum_step=7500 + step, global_step=BASE_STEP + step,
                native_total_step=NATIVE_BASE + step, seam_total_step=0)


def _verify(book):
    for path, expected in book.items():
        if sha(path) != expected:
            raise ValueError("Frozen file changed: " + str(path))


def _stream_values(old, feasibility):
    """Use the same scientific sample rows; changes are branch metadata only."""
    source_spec = old["schedules"]["M_PLUS3000"]
    _verify({source_spec["path"]: source_spec["sha256"],
             old["coefficient_schedule"]: old["coefficient_schedule_sha256"]})
    original = read(source_spec["path"])
    original_coeff = read(old["coefficient_schedule"])
    m = read(old["manifest"])
    byid = {r["condition_id"]: r for r in m["train"]}
    eligible = {r["condition_id"]: r["status"] == "PASS" for r in feasibility["rows"]}
    if set(eligible) != set(byid):
        raise ValueError("Feasibility inventory is not the original TRAIN pool")
    steps, coeffs, plan = [], [], []
    occurrence, exposures, grouped = Counter(), Counter(), Counter()
    for local, (source, coefficient) in enumerate(zip(original["steps"][:UPDATES], original_coeff["steps"][:UPDATES]), 1):
        if source["stage_step"] != local or coefficient["stage_step"] != local or len(source["samples"]) != 8:
            raise ValueError("The first 200 source batches/coefficient rows differ")
        entry = copy.deepcopy(source)
        entry.update(cursor(local), step=local, source_stage_step=source["stage_step"])
        slot_plan = []
        for offset in (0, 2, 4, 6):
            pair = entry["samples"][offset:offset+2]
            cid = pair[0]["condition_id"]
            if pair[1]["condition_id"] != cid or [r["augmentation"] for r in pair] != ["CLEAN", "X_WIDTH"]:
                raise ValueError("Original CLEAN/X_WIDTH condition pair differs")
            if float(pair[0]["alpha"]) != 1. or cid not in byid:
                raise ValueError("Original clean pair or TRAIN identity differs")
            occurrence[cid] += 1
            requested = occurrence[cid] % 2 == 0
            applied = bool(eligible[cid] and requested)
            for micro, row in enumerate(pair, offset):
                src = byid[cid]
                if (row["parent_id"], row["replay_group"]) != (src["parent_id"], src["replay_group"]):
                    raise ValueError("Condition parent/group differs")
                exposures[cid] += 1
                slot = dict(condition_id=cid, parent_id=row["parent_id"], microbatch=micro,
                            replay_group=row["replay_group"], augmentation=row["augmentation"],
                            condition_pair_occurrence=occurrence[cid], condition_exposure=exposures[cid],
                            eligible=eligible[cid], flip_requested=requested, flip_apply=applied,
                            blocked_fallback=bool(requested and not eligible[cid]))
                slot_plan.append(slot)
                grouped[(row["parent_id"], row["replay_group"], row["augmentation"], applied)] += 1
        counts = [9 * (int(byid[r["condition_id"]]["N"]) - int(byid[r["condition_id"]]["K"])) for r in entry["samples"]]
        if counts != coefficient["micro_free_scalar_counts"] or sum(counts) != coefficient["D_W"]:
            raise ValueError("Inherited group-coefficient denominator differs")
        steps.append(entry)
        coeffs.append(copy.deepcopy(coefficient))
        plan.append(dict(stage_step=local, slots=slot_plan))
    if len(steps) != UPDATES or sum(exposures.values()) != 1600 or set(exposures) != set(byid):
        raise ValueError("All 30 original TRAIN conditions must remain in the 1600-slot stream")
    schedule = dict(schema=SCHEMA + "_shared_stream", updates=UPDATES, batch=8, arms=list(ARMS),
                    source_schedule=source_spec, steps=steps,
                    stream_policy="Exact first200 EXPOSURE batches/random values, new branch cursor; not a fresh random stream")
    coefficients = dict(schema=SCHEMA + "_coefficients", updates=UPDATES, steps=coeffs,
                        loss_normalization=LOSS_NORMALIZATION,
                        source_path=old["coefficient_schedule"], source_sha256=old["coefficient_schedule_sha256"])
    flip_plan = dict(schema=SCHEMA + "_flip_plan", policy="even 1-based condition pair occurrence; CLEAN/X_WIDTH together",
                     first_pair="ORIGINAL", blocked_policy="original replay; never drop or replace parent", steps=plan,
                     rejected_draft="Single-exposure parity was not executed: it would confound flip with X_WIDTH. Replaced before registration.")
    slots = [s for e in plan for s in e["slots"]]
    summary = dict(instances_per_arm=1600, conditions=len(byid),
                   group_exposure=dict(Counter(s["replay_group"] for s in slots)),
                   parent_exposure=dict(sorted(Counter(s["parent_id"] for s in slots).items())),
                   PLAIN200_flipped=0, FLIP200_flipped=sum(s["flip_apply"] for s in slots),
                   requested_flips=sum(s["flip_requested"] for s in slots),
                   blocked_fallbacks=sum(s["blocked_fallback"] for s in slots),
                   group_flipped=dict(Counter(s["replay_group"] for s in slots if s["flip_apply"])),
                   variant_width_counts=[dict(parent_id=p, replay_group=g, augmentation=a, flip_apply=f, instances=n)
                                         for (p, g, a, f), n in sorted(grouped.items())])
    return schedule, coefficients, flip_plan, summary


def prepare_training():
    """Freeze metadata only; no Torch import, model load, OT, or RNG draw."""
    if REGISTRATION.exists():
        checked()
        return read(REGISTRATION)
    from r2.pilots.bench_four_routes_v1.resources import snapshot
    resources = OUT / "training_preparation/resources_registration.json"
    if not resources.exists():
        write(resources, snapshot())
    old = read(OLD / "training_registration.json")
    feasibility, gate = read(FEASIBILITY), read(CPU_GATE)
    if feasibility["status"] != "PREPARED_NOT_TRAINED" or gate["status"] != "PASS":
        raise ValueError("Feasibility and actual Native CPU packing gates must pass")
    if feasibility["manifest_sha256"] != old["manifest_sha256"]:
        raise ValueError("TRAIN pool changed between feasibility and old EXPOSURE")
    if (gate["actual_CPU_OT_calls"], gate["conditions"], gate["model_constructions"],
            gate["tensor_weight_loads"], gate["model_forwards"], gate["optimizer_updates"], gate["CUDA_initialized"]) != (60, 30, 0, 0, 0, 0, False):
        raise ValueError("Registered CPU-only 60-OT preparation scope differs")
    _verify({str(Path(__file__).with_name("training_data_checks.py")): gate["source_code_sha256"],
             str(Path(__file__).with_name("training_data.py")): gate["adapter_sha256"],
             str(ROOT / "r2/pilots/bench_natural_n_recipe_v1/runtime.py"): gate["runtime_source_sha256"],
             str(FEASIBILITY): gate["feasibility_sha256"]})
    _verify(feasibility["protected_inputs"])
    _verify(old["training_code_hashes"])
    if sha(SOURCE) != SOURCE_IDENTITIES["source_checkpoint_sha256"]:
        raise ValueError("Retained full EXPOSURE endpoint changed")
    receipt = read(SOURCE.with_suffix(".json"))
    for key, source_key in (("generator_state_sha256", "source_generator_sha256"),
                            ("optimizer_state_sha256", "source_optimizer_sha256"),
                            ("rng_sha256", "source_rng_sha256"), ("sha256", "source_checkpoint_sha256")):
        if receipt[key] != SOURCE_IDENTITIES[source_key]:
            raise ValueError("Source checkpoint receipt identity differs: " + key)
    schedule, coefficients, plan, summary = _stream_values(old, feasibility)
    paths = dict(input_schedule=OUT / "training_preparation/shared_schedule.json",
                 coefficient_schedule=OUT / "training_preparation/coefficient_schedule.json",
                 flip_plan=OUT / "training_preparation/flip_plan.json")
    for key, value in zip(paths, (schedule, coefficients, plan)):
        write(paths[key], value)
    paths.update(manifest=resolve(old["manifest"]), eval_manifest=resolve(old["eval_manifest"]), source_checkpoint=SOURCE)
    sources = {resolve(p) for p in old["training_code_hashes"]}
    sources.update(Path(__file__).with_name(name) for name in ("training.py", "training_data.py", "training_data_checks.py", "geometry.py", "common.py"))
    # common.py imports the old CPU-only common; freeze those dependencies too.
    sources.update(ROOT / name for name in ("r2/pilots/bench_c_tessellation_v1/common.py", "r2/pilots/bench_c_tessellation_v1/geometry.py",
                                           "r2/pilots/bench_four_routes_v1/resources.py"))
    if sha(Path(__file__).with_name("training_data.py")) != feasibility["source_code_sha256"] or sha(Path(__file__).with_name("geometry.py")) != feasibility["geometry_source_sha256"]:
        raise ValueError("CPU-tested data or geometry source changed")
    reg = dict(schema=SCHEMA + "_registration", status="FROZEN", experiment=EXPERIMENT,
               arms=list(ARMS), updates=UPDATES, full_steps=[200], checkpoint_every=50,
               source_global_step=BASE_STEP, source_native_total_step=NATIVE_BASE,
               final_global_step=BASE_STEP + UPDATES, final_native_total_step=NATIVE_BASE + UPDATES,
               **SOURCE_IDENTITIES, model_config=old["model_config"], batch=8, microbatch=1, hybrid_lambda=.25,
               runtime_version=old["runtime_version"], N_contract=[128, 799], loss_normalization=LOSS_NORMALIZATION,
               optimizer="Inherited full EXPOSURE AdamW/RNG; lr1e-5 betas(.9,.95) wd0 clip1; no reset",
               trainable_parameters="all inherited generator parameters; no new parameters", EMA=False,
               per_arm_budget=limits(), total_budget={k: 2*v for k, v in limits().items()},
               head_forward_budget=0, head_optimizer_updates=0, probe_forward_budget=0,
               source_registration=str(OLD / "training_registration.json"), source_registration_sha256=sha(OLD / "training_registration.json"),
               schedules={a: dict(path=str(paths["input_schedule"]), sha256=sha(paths["input_schedule"])) for a in ARMS},
               training_code_hashes={str(p): sha(p) for p in sorted(sources)},
               gate_hashes={str(p): sha(p) for p in (FEASIBILITY, CPU_GATE, resources)},
               input_case_hashes={str(resolve(r["case_path"])): r["case_sha256"] for r in read(paths["manifest"])["train"]},
               exposure=summary, stream_policy=schedule["stream_policy"], flip_policy=plan["policy"],
               scientific_boundary="Same-stream 200-update development comparison; reused early training random values, no new randomness/generalization claim",
               checkpoint_policy="One rolling latest at50/100/150; immutable full and inference endpoints at200; no intermediate evaluation",
               failure_contract="No silent replay or rollback. Explicit resume only from safe checkpoint with exact ledger/log cursor; failed or pending calls block.",
               evaluation_contract="Separate parent-owned evaluation registration required; this module runs training only",
               no_cleanup_commit_push_default_change=True)
    for key, path in paths.items():
        reg[key], reg[key + "_sha256"] = str(path), sha(path)
    write(REGISTRATION, reg)
    checked()
    return reg


def checked():
    """Strict CPU registration/hash/stream checks, before any model import."""
    reg = read(REGISTRATION)
    expected = dict(status="FROZEN", experiment=EXPERIMENT, arms=list(ARMS), updates=UPDATES,
                    batch=8, microbatch=1, hybrid_lambda=.25, source_global_step=BASE_STEP,
                    final_global_step=BASE_STEP+UPDATES, source_native_total_step=NATIVE_BASE,
                    checkpoint_every=50, full_steps=[200], per_arm_budget=limits(),
                    total_budget={k: 2*v for k, v in limits().items()},
                    head_forward_budget=0, head_optimizer_updates=0, probe_forward_budget=0,
                    loss_normalization=LOSS_NORMALIZATION, **SOURCE_IDENTITIES)
    if any(reg.get(k) != v for k, v in expected.items()):
        raise ValueError("Authorized paired 200-update contract differs")
    if resolve(reg["source_checkpoint"]).resolve() != SOURCE.resolve():
        raise ValueError("Only the full global25500 EXPOSURE source is accepted")
    for book in (reg["training_code_hashes"], reg["gate_hashes"], reg["input_case_hashes"]):
        _verify(book)
    _verify({reg[k]: reg[k+"_sha256"] for k in ("source_registration", "source_checkpoint", "manifest", "eval_manifest", "input_schedule", "coefficient_schedule", "flip_plan")})
    schedule, coeffs, plan, summary = _stream_values(read(reg["source_registration"]), read(FEASIBILITY))
    if (read(reg["input_schedule"]), read(reg["coefficient_schedule"]), read(reg["flip_plan"]), reg["exposure"]) != (schedule, coeffs, plan, summary):
        raise ValueError("Frozen same-stream or pairwise flip plan differs")
    if any(reg["schedules"][a] != dict(path=reg["input_schedule"], sha256=reg["input_schedule_sha256"]) for a in ARMS):
        raise ValueError("Both arms must reference the exact same stream")
    m = read(reg["manifest"])
    train_parents = {r["parent_id"] for r in m["train"]}
    heldout = {r["parent_id"] for key in ("dev", "high_holdout") for r in m.get(key, [])}
    for row in read(reg["eval_manifest"])["requests"]:
        group = row.get("eval_group", row.get("report_group", row.get("group", "")))
        if group in ("DEV", "HOLDOUT", "HIGH_HOLDOUT", "NEW_DEV", "INDEPENDENT_DEV", "NEW_PARENT_HOLDOUT"):
            heldout.add(row["parent_id"])
    if train_parents & heldout:
        raise ValueError("Held-out parent entered training")
    return reg, m, schedule["steps"], coeffs["steps"], plan["steps"]


def completed(arm):
    """Return hash-verified completed state without importing Torch/model code."""
    if arm not in ARMS:
        raise ValueError(arm)
    path = OUT / "training" / arm / "status.json"
    if not path.exists():
        return None
    state = read(path)
    if state["status"] != "COMPLETE":
        return None
    if state["registration_sha256"] != sha(REGISTRATION) or state["stage_step"] != UPDATES or state["global_step"] != BASE_STEP+UPDATES:
        raise ValueError("Completed branch identity differs")
    for key in ("checkpoint", "inference_export"):
        _verify({state[key]["path"]: state[key]["sha256"]})
    expected = {k: v for k, v in limits().items() if v}
    calls = state["calls"]
    if (calls["limits"] != limits() or calls["failed"] or calls["pending_call_ids"]
            or any(calls["attempted"].get(k, 0) != n or calls["successful"].get(k, 0) != n for k, n in expected.items())):
        raise ValueError("Completed work differs from exact budget")
    return state


def _dependencies():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError("Conflicting CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    sys.path.insert(0, str(ROOT / "r2/src"))
    from r2.pilots.bench_four_routes_v1.exposure import training as E
    from . import training_data as A
    return E, A


def require_cursor(ledger, step):
    cursor(step)
    expected = dict(limits(), train_forward=8*step, train_backward=8*step, optimizer_update=step)
    if ledger.pending or ledger.failed or ledger.limits != limits():
        raise RuntimeError("Pending/failed calls or changed budget forbid replay")
    if any(ledger.attempted[k] != n or ledger.successful[k] != n for k, n in expected.items()):
        raise RuntimeError("Saved cursor differs from attempted/completed work; no rollback")
    for kind in ("train_forward", "train_backward", "optimizer_update"):
        rows = [r["context"] for r in ledger.rows if r["event"] == "CALL_BEGIN" and r["kind"] == kind]
        wanted = [(s, m) for s in range(1, step+1) for m in (range(8) if kind != "optimizer_update" else (None,))]
        if [(r["stage_step"], r.get("microbatch")) for r in rows] != wanted or any(r["global_step"] != BASE_STEP+r["stage_step"] or r["stage"] != STAGE for r in rows):
            raise RuntimeError("Exact step/microbatch sequence differs")


def _validate_payload(E, p, reg, step=None, arm=None):
    for key, hk, function in (("generator_state", "generator_state_sha256", E.state_hash),
                               ("optimizer", "optimizer_state_sha256", E.T.base.tree_hash),
                               ("rng", "rng_sha256", E.T.base.tree_hash)):
        if function(p[key]) != p[hk]:
            raise ValueError("Checkpoint state hash differs: " + key)
    E._check_generator(p["generator_state"])
    E.validate_head_state(p["head_state"], p["head_state_sha256"])
    if p["model_config"] != reg["model_config"] or p["runtime_version"] != reg["runtime_version"] or p["EMA"] or p["update_in_progress"]:
        raise ValueError("Checkpoint model/runtime/update identity differs")
    if p["head_optimizer"] is not None or p["head_step"] or p["head_calls"]:
        raise ValueError("Source head must remain frozen and unused")
    if step is None:
        if (p["schema"], p["arm"], p["stage_step"], p["global_step"], p["native_total_step"], p["seam_total_step"]) != ("bench_four_routes_exposure_training_v1", "M_PLUS3000", 3000, BASE_STEP, NATIVE_BASE, 0):
            raise ValueError("Source is not the retained EXPOSURE global25500 endpoint")
        for hk, rk in (("generator_state_sha256", "source_generator_sha256"), ("optimizer_state_sha256", "source_optimizer_sha256"), ("rng_sha256", "source_rng_sha256")):
            if p[hk] != reg[rk]:
                raise ValueError("Source inherited state differs: " + hk)
    else:
        if any(p.get(k) != v for k, v in cursor(step).items()) or p["schema"] != SCHEMA or p["arm"] != arm or p["registration_sha256"] != sha(REGISTRATION):
            raise ValueError("Resume branch identity differs")
        expected = dict(batch_index=step, next_sample_index=8*step, arm=arm, schedule_sha256=reg["input_schedule_sha256"], flip_plan_sha256=reg["flip_plan_sha256"])
        if p["stream_state"] != expected:
            raise ValueError("Resume stream cursor differs")


def train(arm, resume=False):
    done = completed(arm)
    if done is not None:
        return done
    reg, m, steps, coefficients, plan = checked()
    reg_sha = sha(REGISTRATION)
    go = read(OUT / "dispatch" / (arm + "_go.json"))
    if not go.get("authorized") or go.get("training_registration_sha256") != reg_sha or go.get("arm") != arm:
        raise RuntimeError("Parent resource-aware training dispatch GO is missing")
    if "retained_headroom_mib" not in go or "minimum_free_mib" not in go:
        raise RuntimeError("Explicit parent headroom and minimum-free thresholds required")
    headroom = max(2048., float(go["retained_headroom_mib"]))
    E, A = _dependencies()
    T, torch, np = E.T, E.torch, E.np
    if reg["model_config"] != E.MODEL_CONFIG or reg["runtime_version"] != E.RUNTIME_VERSION:
        raise ValueError("Inherited model/runtime contract differs")
    T.install_io(OUT, arm)
    out = OUT / "training" / arm
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    with T.historical.exclusive_writer(out):
        previous = read(status_path) if status_path.exists() else None
        if previous and (not resume or previous["status"] != "INTERRUPTED_SAFE"):
            raise RuntimeError("Unresolved prior attempt; no automatic replay")
        if resume and not previous:
            raise ValueError("No exact safe cursor exists")
        step = int(previous["stage_step"]) if previous else 0
        attempt = "attempt_" + str(time.time_ns())
        adir = out / "attempts" / attempt
        adir.mkdir(parents=True, exist_ok=False)
        ledger = T.historical.CallLedger(out / "calls.jsonl", limits(), arm, attempt)
        require_cursor(ledger, step)
        E._log_cursor(out, step)
        initial = T.resource()
        T.write(adir / "resources_dispatch.json", initial)
        if initial["free_mib"] is None or initial["free_mib"] < max(headroom, float(go["minimum_free_mib"])):
            raise RuntimeError("RESOURCE_WAIT before model load")
        state = dict(status="LOADING", arm=arm, **cursor(step), target_local_step=UPDATES,
                     registration_sha256=reg_sha, pid=os.getpid(), attempt=attempt,
                     head_step=0, head_calls=0, probe_calls=0)
        if previous:
            state["checkpoint"] = previous["checkpoint"]
        started = time.perf_counter()
        observer = T.historical.GPUObserver(EXPERIMENT + "_" + arm, interval=2.)
        stop = {"signal": None}
        def handler(sig, unused):
            stop["signal"] = sig
        handlers = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
        T.write(status_path, state)
        try:
            with observer:
                torch.set_num_threads(2)
                E.runtime.configure_stable_runtime()
                source_path = resolve(previous["checkpoint"]["path"]) if previous else SOURCE
                source_sha = previous["checkpoint"]["sha256"] if previous else reg["source_checkpoint_sha256"]
                _verify({str(source_path): source_sha})
                p = torch.load(source_path, map_location="cpu", weights_only=True, mmap=True)
                if previous and not previous["checkpoint"].get("source_reference_only", False):
                    _validate_payload(E, p, reg, step, arm)
                    if p["execution"] != ledger.snapshot():
                        raise ValueError("Saved scientific ledger differs")
                    initialization = p["initialization"]
                else:
                    if step != 0:
                        raise ValueError("Source endpoint can initialize only local step0")
                    _validate_payload(E, p, reg)
                    initialization = dict(source_path=str(SOURCE), **SOURCE_IDENTITIES,
                        source_global_step=BASE_STEP, source_native_total_step=NATIVE_BASE,
                        inherited_optimizer=True, inherited_rng=True, fresh_AdamW=False,
                        new_generator_parameters=0, head_used=False, head_calls=0, arm=arm,
                        source_stream_resumed=False, new_random_stream=False,
                        schedule_sha256=reg["input_schedule_sha256"], flip_plan_sha256=reg["flip_plan_sha256"])
                head_state = p["head_state"]
                model, optimizer = E._build_restore(p, BASE_STEP + step, "cuda")
                E.restore_rng(p["rng"])
                if T.base.tree_hash(E.rng_state()) != p["rng_sha256"]:
                    raise ValueError("Inherited RNG restoration differs")
                del p
                T.write(adir / "initialization.json", initialization)
                cases = {r["condition_id"]: A.load_train_case(r, reg["manifest"]) for r in m["train"]}
                adapted, adaptation = {}, {}
                frozen_feasibility = {r["condition_id"]: r for r in read(FEASIBILITY)["rows"]}
                if arm == "FLIP200":
                    for cid, case in cases.items():
                        adapted[cid], adaptation[cid] = A.apply_flip_to_case(case)
                        expected = frozen_feasibility[cid]
                        if adaptation[cid]["status"] != expected["status"] or adaptation[cid]["geometry"] != expected["geometry"]:
                            raise ValueError("TRAIN geometry eligibility changed after registration: " + cid)

                def checkpoint_save():
                    require_cursor(ledger, step)
                    E._log_cursor(out, step)
                    names = E.validate_optimizer(model, optimizer, BASE_STEP + step)
                    g, o, rng = E._cpu_copy(model.state_dict()), E._cpu_copy(optimizer.state_dict()), E._cpu_copy(E.rng_state())
                    E._check_generator(g)
                    E.validate_head_state(head_state)
                    payload = dict(schema=SCHEMA, experiment=EXPERIMENT, arm=arm, **cursor(step),
                        source_global_step=BASE_STEP, model_config=copy.deepcopy(E.MODEL_CONFIG), runtime_version=E.RUNTIME_VERSION,
                        EMA=False, update_in_progress=False, generator_state=g, generator_state_sha256=E.state_hash(g),
                        optimizer=o, optimizer_state_sha256=T.base.tree_hash(o), optimizer_parameter_names=names,
                        rng=rng, rng_sha256=T.base.tree_hash(rng), registration_sha256=reg_sha,
                        head_state=head_state, head_state_sha256=E.HEAD_STATE_SHA256, head_config=E.HEAD_CONFIG,
                        head_optimizer=None, head_step=0, head_calls=0, initialization=initialization,
                        execution=ledger.snapshot(), stream_state=dict(batch_index=step, next_sample_index=8*step, arm=arm,
                            schedule_sha256=reg["input_schedule_sha256"], flip_plan_sha256=reg["flip_plan_sha256"]))
                    immutable = step == UPDATES
                    dest = out / ("checkpoint_step200.pt" if immutable else "checkpoint_latest.pt")
                    if immutable and dest.exists():
                        receipt = read(dest.with_suffix(".json"))
                        _verify({str(dest): receipt["sha256"]})
                        E._receipt_matches(receipt, payload)
                    else:
                        receipt = E._save(dest, payload, replace=not immutable)
                        E.append_event(out / "checkpoint_history.jsonl", receipt)
                    state["checkpoint"] = receipt
                    if immutable:
                        keys = ("experiment", "arm", "stage", "stage_step", "local_step", "curriculum_step", "global_step",
                                "native_total_step", "seam_total_step", "generator_state", "generator_state_sha256",
                                "model_config", "runtime_version", "EMA", "head_state", "head_state_sha256", "head_config", "head_step")
                        export = {k: payload[k] for k in keys}
                        export.update(schema=INFERENCE_SCHEMA, state_sha256=payload["generator_state_sha256"],
                            training_checkpoint_sha256=receipt["sha256"], registration_sha256=reg_sha,
                            model_arm="PLAIN", representation="NATIVE", head_frozen=True, head_used=False)
                        ep = out / "generator_step200.pt"
                        if ep.exists():
                            er = read(ep.with_suffix(".json"))
                            _verify({str(ep): er["sha256"]})
                            if er["generator_state_sha256"] != payload["generator_state_sha256"]:
                                raise ValueError("Immutable inference export differs")
                        else:
                            er = E._save(ep, export)
                        state["inference_export"] = er
                        T.write(out / "rolling_checkpoint_retirement.json", dict(status="SUPERSEDED_BY_FORMAL_ENDPOINT",
                            old_rolling_receipt=read(out / "checkpoint_latest.json") if (out / "checkpoint_latest.json").exists() else None,
                            active_checkpoint=receipt, copied_endpoint_to_latest=False, physical_deletion=False))
                    state.update(cursor(step))
                    T.write(status_path, state)

                state["status"] = "TRAINING"
                if not previous:
                    state["checkpoint"] = dict(path=str(SOURCE), sha256=reg["source_checkpoint_sha256"], source_reference_only=True)
                    T.write(out / "branch_initialization.json", dict(**initialization, registration_sha256=reg_sha))
                for entry, allocation, flip_entry in zip(steps[step:], coefficients[step:], plan[step:]):
                    if stop["signal"] is not None:
                        checkpoint_save()
                        state.update(status="INTERRUPTED_SAFE", signal=stop["signal"])
                        break
                    tick = time.perf_counter()
                    current = entry["stage_step"]
                    if current % 25 == 1:
                        resource = T.resource()
                        E.append_event(out / "resource_checks.jsonl", dict(stage_step_before=step, **resource))
                        if resource["free_mib"] is None or resource["free_mib"] < headroom:
                            checkpoint_save()
                            state.update(status="INTERRUPTED_SAFE", reason="VRAM_headroom_below_dispatch_margin")
                            break
                    actual = [dict(s, flip_applied=bool(arm == "FLIP200" and s["flip_apply"])) for s in flip_entry["slots"]]
                    samples = [E.sample_for_training(adapted[row["condition_id"]] if s["flip_applied"] else cases[row["condition_id"]], row, "PLAIN")
                               for row, s in zip(entry["samples"], actual)]
                    audits = [E.verify_sample(sample, "NATIVE") for sample in samples]
                    denominator = sum(int(s["free_coordinate_count"]) for s in samples)
                    allocation = E.allocation_for_update(allocation, samples, entry)
                    input_seconds = time.perf_counter() - tick
                    begin = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    optimizer.zero_grad(set_to_none=True)
                    fm, objective, stats = 0., 0., []
                    for mi, (sample, row, mechanism, flip) in enumerate(zip(samples, entry["samples"], audits, actual)):
                        context = dict(arm=arm, **cursor(current), microbatch=mi, condition_id=row["condition_id"])
                        b = T.tensors(sample)
                        with E.native_execution(model, coordinates=(b["x"], b["t"]), autocast=True):
                            pred, _ = ledger.call("train_forward", lambda: model(b["x"], b["t"], b["y"], valid_mask=b["valid"], known_mask=b["known"]).float(), context)
                            base_loss = E.corner_masked_fm_loss(pred, b["target"], b["valid"], b["fixed"], denominator=denominator)
                            gain = allocation["micro_gains"][mi]
                            loss = E.apply_loss_gain(base_loss, allocation, mi)
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError("Nonfinite weighted FM objective")
                            ledger.call("train_backward", loss.backward, context)
                        values = E.statistics(pred, b["target"], sample)
                        if not np.isclose(values["all_free_velocity_sse"]/denominator, float(base_loss.detach()), rtol=4e-6, atol=1e-7):
                            raise ValueError("Whole-batch FM accounting differs")
                        if not np.isclose(values["all_free_velocity_sse"]/denominator*gain, float(loss.detach()), rtol=4e-6, atol=1e-7):
                            raise ValueError("Group coefficient objective accounting differs")
                        fm += float(base_loss.detach())
                        objective += float(loss.detach())
                        stats.append(values)
                        E.append_event(out / "exposure.jsonl", {**row, **context, "t": float(sample["t"]), "N": int(sample["N"]), "K": int(sample["K"]),
                            "loss_contribution": float(loss.detach()), "unweighted_loss_contribution": float(base_loss.detach()), "loss_gain": gain,
                            "actual_batch_denominator_D_W": denominator, "reference_denominator_D_B": allocation["D_B"], "reference_HIGH_ratio_rho": allocation["rho"],
                            **values, "mechanism": mechanism, "flip": flip, "head_forward_called": False})
                        del pred, loss, base_loss, b
                    if any(p.grad is None for p in model.parameters()):
                        raise RuntimeError("Unused inherited generator parameter")
                    grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                    ledger.call("optimizer_update", optimizer.step, dict(arm=arm, **cursor(current)))
                    step = current
                    E.validate_optimizer(model, optimizer, BASE_STEP + step)
                    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                        raise FloatingPointError("Nonfinite parameter")
                    sums = {k: sum(r[k] for r in stats) for k in stats[0]}
                    result = dict(arm=arm, **cursor(step), FM=fm, unweighted_FM=fm, objective=objective,
                        free_coordinate_denominator=denominator, loss_normalization=LOSS_NORMALIZATION, group_allocation=allocation,
                        group_velocity_sse={g: sum(v["all_free_velocity_sse"] for v, r in zip(stats, entry["samples"]) if r["replay_group"] == g) for g in ("OLD", "FOCUS", "HIGH")},
                        flip_applied=sum(s["flip_applied"] for s in actual),
                        group_flip_applied=dict(Counter(s["replay_group"] for s in actual if s["flip_applied"])),
                        blocked_flip_fallbacks=sum(s["blocked_fallback"] for s in actual) if arm == "FLIP200" else 0,
                        **sums, generator_gradient_norm=grad, head_step=0, head_calls=0, input_seconds=input_seconds,
                        update_seconds=time.perf_counter()-begin, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        peak_scope="One formal update including backward/inherited AdamW; allocator is not process VRAM")
                    E.append_event(out / "training_log.jsonl", result)
                    state.update(cursor(step), last=result, elapsed_seconds=time.perf_counter()-started)
                    if step % 50 == 0:
                        checkpoint_save()
                    T.write(status_path, state)
                    if step in (1, 5, 25) or step % 50 == 0:
                        T.write(out / f"resources_step{step}.json", T.resource())
                        print(json.dumps(result), flush=True)
                    if step == 1:
                        T.write(out / "first_formal_backward.json", dict(status="PASS", no_extra_forward=True, **result))
                else:
                    state["status"] = "COMPLETE"
                require_cursor(ledger, step)
                E._log_cursor(out, step)
                T.write(status_path, state)
        except BaseException as exc:
            state.update(status="OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "FAILED",
                         error=repr(exc), traceback=traceback.format_exc(), no_automatic_replay=True)
            T.write(adir / "failure.json", state)
            raise
        finally:
            for sig, old_handler in handlers.items():
                signal.signal(sig, old_handler)
            state.update(cursor(step), elapsed_seconds=time.perf_counter()-started, calls=ledger.snapshot(), resources=observer.summary())
            T.write(adir / "resources.json", dict(samples=observer.samples, summary=observer.summary()))
            T.write(status_path, state)
            T.write(adir / "final_status.json", state)
        return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-training", action="store_true")
    parser.add_argument("--check-cpu", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.prepare_training:
        value = prepare_training()
        print(json.dumps(dict(status=value["status"], registration_sha256=sha(REGISTRATION), exposure=value["exposure"], model_calls=0)))
    elif args.check_cpu:
        value, *_ = checked()
        print(json.dumps(dict(status="PASS", registration_sha256=sha(REGISTRATION), exposure=value["exposure"], model_calls=0)))
    elif args.arm:
        value = train(args.arm, args.resume)
        print(json.dumps({k: value[k] for k in ("status", "arm", "stage_step", "global_step", "calls", "checkpoint", "inference_export") if k in value}))
    else:
        parser.error("Choose --prepare-training, --check-cpu or --arm")
