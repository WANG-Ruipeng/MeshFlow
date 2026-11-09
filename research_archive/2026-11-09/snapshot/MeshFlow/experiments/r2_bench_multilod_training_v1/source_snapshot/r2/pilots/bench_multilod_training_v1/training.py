"""Paired 1500-update fixed-C multi-LOD continuations from EXPOSURE25500.

Only the registered mesh version differs. Generator/live AdamW/RNG are inherited;
Native/T1 + role + Geo, free-only FM and OT-then-HYBRID.25 remain unchanged.
Scientific execution requires a separately recorded resource-dispatch GO.
Import and --check-cpu do not construct models, load tensors, or initialize CUDA.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
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

from .common import (ROOT, OUT, read, resolve, sha, write, ARMS, UPDATES, BASE_STEP,
                     NATIVE_BASE, EXPERIMENT, SOURCE, SOURCE_IDENTITIES,
                     LOSS_NORMALIZATION, REGISTRATION)

STAGE = "NATIVE"
SCHEMA = "bench_multilod_training_v1"
INFERENCE_SCHEMA = "bench_multilod_generator_v1"


def limits():
    return dict(train_forward=12000, train_backward=12000, optimizer_update=1500,
                head_forward=0, head_backward=0, head_optimizer_update=0, probe_forward=0)


def cursor(step):
    if type(step) is not int or not 0 <= step <= UPDATES:
        raise ValueError("Branch cursor is outside the exact 1500-update budget")
    return dict(stage=STAGE, stage_step=step, local_step=step,
                curriculum_step=7500 + step, global_step=BASE_STEP + step,
                native_total_step=NATIVE_BASE + step, seam_total_step=0)


def _verify(book):
    for path, expected in book.items():
        if sha(path) != expected:
            raise ValueError("Frozen file changed: " + str(path))


def allocation_for_update(coefficient, samples, entry):
    """Preserve the original group-MSE weights under each arm's actual N.

    Let a_g be the frozen original effective group weight.  The unchanged
    masked loss returns SSE_i / D_actual; multiplying it by
    a_g * D_actual / m_g_actual yields a_g * SSE_i / m_g_actual.
    This function is CPU-only and never accepts stale original-N denominators.
    """
    counts = [int(s["free_coordinate_count"]) for s in samples]
    groups = [r["replay_group"] for r in entry["samples"]]
    if (coefficient["stage_step"] != entry["stage_step"] or len(counts) != 8 or
            coefficient["micro_free_scalar_counts"] != counts or
            coefficient["micro_groups"] != groups or coefficient["D_W"] != sum(counts)):
        raise ValueError("Actual samples/masks differ from frozen coefficient schedule")
    for sample, count in zip(samples, counts):
        if count != 9 * (int(sample["N"]) - int(sample["K"])) or count <= 0:
            raise ValueError("Native free scalar denominator differs")
    actual = {g: sum(c for c, group in zip(counts, groups) if group == g)
              for g in ("OLD", "FOCUS", "HIGH")}
    if coefficient["actual_group_scalar_counts"] != actual:
        raise ValueError("Actual per-group scalar denominator differs")
    weights = coefficient["reference_group_mse_weights"]
    if set(weights) != set(actual) or any(not math.isfinite(float(a)) or a < 0 for a in weights.values()):
        raise ValueError("Invalid frozen reference group weights")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0., abs_tol=1e-12):
        raise ValueError("Reference group weights must sum to one")
    if any((actual[g] == 0) != (weights[g] == 0) for g in actual):
        raise ValueError("Absent group has nonzero reference weight or conversely")
    expected = [weights[g] * sum(counts) / actual[g] for g in groups]
    if any(not math.isfinite(float(x)) or not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)
           for x, y in zip(coefficient["micro_gains"], expected)) or len(coefficient["micro_gains"]) != 8:
        raise ValueError("Effective microbatch gains differ from reference group MSE")
    effective = {g: sum(c * gain for c, group, gain in zip(counts, groups, expected) if group == g) / sum(counts)
                 for g in actual}
    if any(not math.isclose(effective[g], weights[g], rel_tol=1e-12, abs_tol=1e-12) for g in actual):
        raise ValueError("Actual micro loss gains changed a reference group weight")
    return coefficient


def checked(arm):
    """Hash-verify registration and the selected stream before Torch import."""
    from . import data as A
    if arm not in ARMS:
        raise ValueError(arm)
    reg = read(REGISTRATION)
    expected = dict(status="FROZEN", experiment=EXPERIMENT, arms=list(ARMS), updates=UPDATES,
                    batch=8, microbatch=1, hybrid_lambda=.25, source_global_step=BASE_STEP,
                    final_global_step=BASE_STEP + UPDATES, source_native_total_step=NATIVE_BASE,
                    final_native_total_step=NATIVE_BASE + UPDATES,
                    checkpoint_every=250, full_steps=[UPDATES], per_arm_budget=limits(),
                    total_budget={k: 2 * v for k, v in limits().items()},
                    head_forward_budget=0, head_optimizer_updates=0, probe_forward_budget=0,
                    loss_normalization=LOSS_NORMALIZATION, N_contract=[128, 799], **SOURCE_IDENTITIES)
    if any(reg.get(k) != v for k, v in expected.items()):
        raise ValueError("Authorized paired multi-LOD 1500-update contract differs")
    if resolve(reg["source_checkpoint"]).resolve() != SOURCE.resolve():
        raise ValueError("Only full global25500 EXPOSURE initialization is accepted")
    for book in (reg["training_code_hashes"], reg["gate_hashes"], reg["input_case_hashes"]):
        _verify(book)
    _verify({reg[k]: reg[k + "_sha256"] for k in ("source_checkpoint", "manifest", "eval_manifest")})
    schedule_spec = reg["schedules"][arm]
    coefficients_spec = reg["coefficient_schedules"][arm]
    _verify({schedule_spec["path"]: schedule_spec["sha256"],
             coefficients_spec["path"]: coefficients_spec["sha256"]})
    m = read(reg["manifest"])
    schedule = read(schedule_spec["path"])
    A.validate_schedule(schedule, m, arm)
    coefficients = A.checked_coefficients(coefficients_spec["path"], schedule, m)
    if len(schedule["steps"]) != UPDATES or len(coefficients) != UPDATES:
        raise ValueError("Exactly1500 frozen schedule and allocation rows required")
    train_parents = {r["parent_id"] for r in m["train"]}
    heldout = {r["parent_id"] for key in ("dev", "high_holdout") for r in m.get(key, [])}
    for row in read(reg["eval_manifest"])["requests"]:
        group = row.get("eval_group", row.get("report_group", row.get("group", "")))
        if group in ("DEV", "HOLDOUT", "HIGH_HOLDOUT", "NEW_DEV", "INDEPENDENT_DEV", "NEW_PARENT_HOLDOUT"):
            heldout.add(row["parent_id"])
    if train_parents & heldout:
        raise ValueError("Held-out parent entered training")
    for entry, coefficient in zip(schedule["steps"], coefficients):
        if any(entry.get(k) != v for k, v in cursor(entry["stage_step"]).items()):
            raise ValueError("Frozen step metadata differs from branch cursor")
        samples = [dict(N=r["N"], K=r["K"], free_coordinate_count=9 * (int(r["N"]) - int(r["K"])))
                   for r in entry["samples"]]
        allocation_for_update(coefficient, samples, entry)
    return reg, m, schedule["steps"], coefficients


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
    expected = limits()
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
    from . import data as A
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
    if p["head_optimizer"] is not None or p["head_step"] or p["head_calls"] or p["head_config"] != E.HEAD_CONFIG:
        raise ValueError("Source head must remain frozen and unused")
    if step is None:
        if (p["schema"], p["arm"], p["stage"], p["stage_step"], p["global_step"], p["native_total_step"], p["seam_total_step"]) != ("bench_four_routes_exposure_training_v1", "M_PLUS3000", STAGE, 3000, BASE_STEP, NATIVE_BASE, 0):
            raise ValueError("Source is not the retained EXPOSURE global25500 endpoint")
        for hk, rk in (("generator_state_sha256", "source_generator_sha256"), ("optimizer_state_sha256", "source_optimizer_sha256"), ("rng_sha256", "source_rng_sha256")):
            if p[hk] != reg[rk]:
                raise ValueError("Source inherited state differs: " + hk)
    else:
        if any(p.get(k) != v for k, v in cursor(step).items()) or p["schema"] != SCHEMA or p["arm"] != arm or p["registration_sha256"] != sha(REGISTRATION):
            raise ValueError("Resume branch identity differs")
        expected = dict(batch_index=step, next_sample_index=8*step, arm=arm, schedule_sha256=reg["schedules"][arm]["sha256"], coefficient_schedule_sha256=reg["coefficient_schedules"][arm]["sha256"])
        if p["stream_state"] != expected:
            raise ValueError("Resume stream cursor differs")


def train(arm, resume=False):
    done = completed(arm)
    if done is not None:
        return done
    reg, m, steps, coefficients = checked(arm)
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
                        source_stream_resumed=False, new_random_stream=True,
                        schedule_sha256=reg["schedules"][arm]["sha256"], coefficient_schedule_sha256=reg["coefficient_schedules"][arm]["sha256"])
                head_state = p["head_state"]
                model, optimizer = E._build_restore(p, BASE_STEP + step, "cuda")
                E.restore_rng(p["rng"])
                if T.base.tree_hash(E.rng_state()) != p["rng_sha256"]:
                    raise ValueError("Inherited RNG restoration differs")
                del p
                T.write(adir / "initialization.json", initialization)
                used_cases = {r["case_id"]: r for entry in steps for r in entry["samples"]}
                cases = {case_id: A.load_case(row) for case_id, row in used_cases.items()}

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
                            schedule_sha256=reg["schedules"][arm]["sha256"], coefficient_schedule_sha256=reg["coefficient_schedules"][arm]["sha256"]))
                    immutable = step == UPDATES
                    dest = out / ("checkpoint_step1500.pt" if immutable else "checkpoint_latest.pt")
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
                        ep = out / "generator_step1500.pt"
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
                for entry, allocation in zip(steps[step:], coefficients[step:]):
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
                    samples = [E.sample_for_training(cases[row["case_id"]], row, "PLAIN")
                               for row in entry["samples"]]
                    audits = [E.verify_sample(sample, "NATIVE") for sample in samples]
                    denominator = sum(int(s["free_coordinate_count"]) for s in samples)
                    allocation = allocation_for_update(allocation, samples, entry)
                    input_seconds = time.perf_counter() - tick
                    begin = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    optimizer.zero_grad(set_to_none=True)
                    fm, objective, stats = 0., 0., []
                    for mi, (sample, row, mechanism) in enumerate(zip(samples, entry["samples"], audits)):
                        context = dict(arm=arm, **cursor(current), microbatch=mi, condition_id=row["condition_id"], case_id=row["case_id"], variant=row["variant"])
                        b = T.tensors(sample)
                        with E.native_execution(model, coordinates=(b["x"], b["t"]), autocast=True):
                            pred, _ = ledger.call("train_forward", lambda: model(b["x"], b["t"], b["y"], valid_mask=b["valid"], known_mask=b["known"]).float(), context)
                            base_loss = E.corner_masked_fm_loss(pred, b["target"], b["valid"], b["fixed"], denominator=denominator)
                            gain = allocation["micro_gains"][mi]
                            loss = base_loss * gain
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
                            "actual_batch_denominator_D_W": denominator, "reference_denominator_D_B": allocation["D_B"], "reference_group_mse_weights": allocation["reference_group_mse_weights"],
                            **values, "mechanism": mechanism, "head_forward_called": False})
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
                        variant_instances=dict(Counter(r["variant"] for r in entry["samples"])),
                        variant_velocity_sse={variant: sum(values["all_free_velocity_sse"] for values, row in zip(stats, entry["samples"]) if row["variant"] == variant)
                                              for variant in sorted({r["variant"] for r in entry["samples"]})},
                        variant_free_scalar_counts={variant: sum(int(sample["free_coordinate_count"]) for sample, row in zip(samples, entry["samples"]) if row["variant"] == variant)
                                                    for variant in sorted({r["variant"] for r in entry["samples"]})},
                        **sums, generator_gradient_norm=grad, head_step=0, head_calls=0, input_seconds=input_seconds,
                        update_seconds=time.perf_counter()-begin, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        peak_scope="One formal update including backward/inherited AdamW; allocator is not process VRAM")
                    E.append_event(out / "training_log.jsonl", result)
                    state.update(cursor(step), last=result, elapsed_seconds=time.perf_counter()-started)
                    if step % 250 == 0:
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
    parser.add_argument("--check-cpu", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.check_cpu:
        arms = (args.arm,) if args.arm else ARMS
        for arm in arms:
            reg, _, steps, coefficients = checked(arm)
            print(json.dumps(dict(status="PASS", arm=arm, registration_sha256=sha(REGISTRATION),
                                  steps=len(steps), coefficients=len(coefficients), model_calls=0)))
    elif args.arm:
        value = train(args.arm, args.resume)
        print(json.dumps({k: value[k] for k in ("status", "arm", "stage_step", "global_step", "calls", "checkpoint", "inference_export") if k in value}))
    else:
        parser.error("Choose --check-cpu or --arm")
