"""Two isolated, precommitted Bench mechanisms, 1500 updates each.

Imports the proven IO/checkpoint utilities, never the old experiment state machine.
No fixed neural probes: native training losses are not directly comparable.
"""
from __future__ import annotations
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
import copy
import json
import signal
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "r2/src"))
import numpy as np
import torch
from meshflow_control import runtime
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash, rng_state, restore_rng
from meshflow_control.models.native import build_model, native_execution
from meshflow_control.seam.model_adapter import seam_forward, corner_masked_fm_loss
from meshflow_control.training.context_checkpoint import _cpu_copy, _check_generator
from meshflow_control.data.io import digest, append_event
from r2.pilots.bench_adaptation_strength_v1 import training as T
from r2.pilots.bench_clean_multipart_seam_v1 import data as D
from .common import OUT, ARMS, EXPERIMENT, resolve

SCHEMA = "bench_state_schedule_training_v1"
UPDATES = 1500


def checked():
    reg = T.read(OUT / "training_registration.json")
    if (reg["status"], reg["experiment"], reg["arms"], reg["updates"],
        reg["batch"], reg["microbatch"], reg["hybrid_lambda"]) != (
            "FROZEN", EXPERIMENT, list(ARMS), 1500, 8, 1, .25):
        raise ValueError("Frozen definition differs")
    for path, expected in reg["training_code_hashes"].items():
        if digest(resolve(path)) != expected:
            raise ValueError("Frozen training code differs: " + path)
    for key in ("manifest", "input_schedule", "source_checkpoint"):
        if digest(resolve(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("Frozen input differs: " + key)
    manifest = T.read(resolve(reg["manifest"]))
    steps = T.read(resolve(reg["input_schedule"]))["steps"]
    if (len(manifest["rows"]), len(manifest["train"]), len(manifest["dev"])) != (316, 195, 39):
        raise ValueError("Original pool differs")
    if len({r["parent_id"] for r in manifest["train"]}) != 58:
        raise ValueError("Training parents differ")
    if [e["step"] for e in steps] != list(range(1, 1501)) or any(len(e["samples"]) != 8 for e in steps):
        raise ValueError("Original 1500x8 stream differs")
    ids = {r["condition_id"] for r in manifest["train"]}
    if any(r["condition_id"] not in ids for e in steps for r in e["samples"]):
        raise ValueError("Non-TRAIN row")
    for row in manifest["rows"]:
        if digest(resolve(row["case_path"])) != row["case_sha256"]:
            raise ValueError("Case changed: " + row["condition_id"])
    return reg, manifest, steps


def statistics(pred, b, sample, arm):
    error = (pred.detach().float() - b["target"]).reshape(int(sample["N"]), 3, 3).square().cpu().numpy()
    fixed = np.asarray(sample["anchor_ids"]) >= 0
    valid = np.asarray(sample["valid_mask"], bool)
    free = valid[:, None] & ~fixed
    tip = free & (fixed.sum(1) == 2)[:, None]
    bulk = free & (fixed.sum(1) == 0)[:, None]
    active = free.copy()
    t = float(sample["t"])
    s = float(sample["seam_time"]) if arm == "ASYNC_LEAD_SEAM" else t
    if arm == "ASYNC_LEAD_SEAM":
        active = np.asarray(sample["active_free_corner_mask"], bool)
    tip_active = tip & active
    endpoint_factor = np.full(free.shape, 1. - t, np.float32)
    endpoint_factor[tip] = 1. - s
    endpoint_error = error * endpoint_factor[..., None] ** 2
    global_error = error.copy()
    if arm == "ASYNC_LEAD_SEAM":
        global_error[tip_active] /= .7 ** 2
    return dict(
        all_free_velocity_sse=float(error[active].sum(dtype=np.float64)),
        all_free_scalar_count=int(free.sum()) * 3,
        active_scalar_count=int(active.sum()) * 3,
        seam_tip_velocity_sse=float(error[tip_active].sum(dtype=np.float64)),
        seam_tip_scalar_count=int(tip_active.sum()) * 3,
        inactive_tip_scalar_count=int((tip & ~active).sum()) * 3,
        bulk_velocity_sse=float(error[bulk].sum(dtype=np.float64)),
        bulk_scalar_count=int(bulk.sum()) * 3,
        active_endpoint_sse=float(endpoint_error[active].sum(dtype=np.float64)),
        seam_tip_active_endpoint_sse=float(endpoint_error[tip_active].sum(dtype=np.float64)),
        bulk_endpoint_sse=float(endpoint_error[bulk].sum(dtype=np.float64)),
        global_equivalent_velocity_sse=float(global_error[active].sum(dtype=np.float64)),
    )


def import_start(reg, arm):
    payload = torch.load(resolve(reg["source_checkpoint"]), map_location="cpu", weights_only=True, mmap=True)
    if (payload["arm"], payload["local_step"], payload["global_step"], payload["update_in_progress"]) != (
            "A_COMMON_LR1E5", 0, 12000, False):
        raise ValueError("Must import exact A step0")
    for key, fn, expected in (
        ("generator_state", state_hash, reg["source_generator_sha256"]),
        ("optimizer", T.base.tree_hash, reg["source_optimizer_sha256"]),
        ("rng", T.base.tree_hash, reg["source_rng_sha256"]),
    ):
        if fn(payload[key]) != expected:
            raise ValueError("A step0 internal identity differs: " + key)
    if payload["optimizer"]["state"]:
        raise ValueError("A step0 must be fresh AdamW")
    model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode="none", trainable=True).eval()
    model.load_state_dict(payload["generator_state"], strict=True)
    model.to("cuda")
    optimizer = T.make_optimizer(model, "A_COMMON_LR1E5")
    if payload["optimizer_parameter_names"] != T.group_names(model):
        raise ValueError("Parameter names/order differ")
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"])
    if T.base.tree_hash(rng_state()) != reg["source_rng_sha256"]:
        raise ValueError("Imported runtime RNG differs")
    audit = dict(source_path=reg["source_checkpoint"], source_sha256=reg["source_checkpoint_sha256"],
                 generator_initial_sha256=reg["source_generator_sha256"],
                 imported_optimizer_sha256=reg["source_optimizer_sha256"],
                 imported_rng_sha256=reg["source_rng_sha256"], fresh_AdamW=True,
                 source_global_step=12000, arm=arm, new_parameters=0, head_calls=0,
                 generator_parameters=sum(p.numel() for p in model.parameters()))
    return model, optimizer, audit


def train(arm, resume=False):
    if arm not in ARMS:
        raise ValueError(arm)
    T.install_io(OUT, arm)
    reg, manifest, steps = checked()
    reg_sha = digest(OUT / "training_registration.json")
    go = T.read(OUT / "training_dispatch_go.json")
    if not go.get("authorized") or go["training_registration_sha256"] != reg_sha or arm not in go["arms"]:
        raise RuntimeError("Parent resource-aware release missing")
    out = OUT / "training" / arm
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    previous = T.read(status_path) if status_path.exists() else None
    if previous and previous["status"] == "COMPLETE":
        if digest(resolve(previous["checkpoint"]["path"])) != previous["checkpoint"]["sha256"]:
            raise ValueError("Final checkpoint changed")
        return previous
    if previous and (not resume or previous["status"] != "INTERRUPTED_SAFE"):
        raise RuntimeError("Unresolved prior attempt; no silent replay")
    if resume and not previous:
        raise ValueError("No exact saved cursor")
    step = int(previous["local_step"]) if previous else 0
    attempt = "attempt_" + str(time.time_ns())
    adir = out / "attempts" / attempt
    adir.mkdir(parents=True, exist_ok=False)
    ledger = T.historical.CallLedger(out / "calls.jsonl",
        dict(train_forward=12000, train_backward=12000, optimizer_update=1500), arm, attempt)
    T.require_cursor(ledger, step)
    state = dict(status="LOADING", arm=arm, local_step=step, global_step=12000 + step,
                 target_local_step=1500, registration_sha256=reg_sha, pid=os.getpid(),
                 attempt=attempt, head_calls=0, probe_calls=0)
    if previous:
        state["checkpoint"] = previous["checkpoint"]
    initial = T.resource()
    T.write(adir / "resources_dispatch.json", initial)
    if initial["free_mib"] is None or initial["free_mib"] < 2048:
        raise RuntimeError("RESOURCE_WAIT before model load")
    observer = T.historical.GPUObserver("bench_state_schedule_" + arm, interval=2.)
    started = time.perf_counter()
    stop = {"signal": None}
    def handler(sig, unused):
        stop["signal"] = sig
    handlers = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    owned = False
    try:
        with T.historical.exclusive_writer(out), observer:
            owned = True
            T.write(status_path, state)
            torch.set_num_threads(2)
            runtime.configure_stable_runtime()
            if previous:
                path = resolve(previous["checkpoint"]["path"])
                if digest(path) != previous["checkpoint"]["sha256"]:
                    raise ValueError("Resume SHA differs")
                p = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if (p["schema"], p["arm"], p["local_step"], p["registration_sha256"],
                    p["update_in_progress"]) != (SCHEMA, arm, step, reg_sha, False):
                    raise ValueError("Wrong exact-cursor checkpoint")
                for key, fn, hk in (("generator_state", state_hash, "generator_state_sha256"),
                                   ("optimizer", T.base.tree_hash, "optimizer_state_sha256"),
                                   ("rng", T.base.tree_hash, "rng_sha256")):
                    if fn(p[key]) != p[hk]:
                        raise ValueError("Resume hash mismatch: " + key)
                model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode="none", trainable=True).eval()
                model.load_state_dict(p["generator_state"], strict=True)
                model.to("cuda")
                optimizer = T.make_optimizer(model, "A_COMMON_LR1E5")
                optimizer.load_state_dict(p["optimizer"])
                restore_rng(p["rng"])
                if T.base.tree_hash(rng_state()) != p["rng_sha256"]:
                    raise ValueError("Resume RNG mismatch")
                audit = p["initialization"]
                del p
            else:
                model, optimizer, audit = import_start(reg, arm)
            T.validate_optimizer(model, optimizer, "A_COMMON_LR1E5", step)
            T.write(adir / "initialization.json", audit)
            cases = {r["condition_id"]: D.load_case(r) for r in manifest["train"]}
            reference_payload = torch.load(resolve(reg["source_checkpoint"]), map_location="cpu", weights_only=True, mmap=True)
            reference = reference_payload["generator_state"]

            def checkpoint_save():
                T.require_cursor(ledger, step)
                names = T.validate_optimizer(model, optimizer, "A_COMMON_LR1E5", step)
                g = _cpu_copy(model.state_dict())
                _check_generator(g)
                o, rng = _cpu_copy(optimizer.state_dict()), _cpu_copy(rng_state())
                p = dict(schema=SCHEMA, experiment=EXPERIMENT, arm=arm, local_step=step,
                    global_step=12000+step, source_global_step=12000, seam_total_step=step,
                    EMA=False, update_in_progress=False, generator_state=g,
                    generator_state_sha256=state_hash(g), optimizer=o,
                    optimizer_state_sha256=T.base.tree_hash(o), optimizer_parameter_names=names,
                    rng=rng, rng_sha256=T.base.tree_hash(rng), initialization=audit,
                    model_config=copy.deepcopy(MODEL_CONFIG), registration_sha256=reg_sha,
                    execution=ledger.snapshot(), stream_state=dict(batch_index=step, next_sample_index=8*step),
                    parameter_drift=T.parameter_drift(g, reference, names), head_calls=0)
                immutable = step in (0, 1500)
                dest = out / (f"checkpoint_step{step}.pt" if immutable else "rolling.pt")
                if dest.exists() and immutable:
                    receipt = T.read(dest.with_suffix(".json"))
                    if any(receipt[k] != p[k] for k in ("generator_state_sha256", "optimizer_state_sha256", "rng_sha256")):
                        raise ValueError("Immutable checkpoint differs")
                    state["checkpoint"] = receipt
                else:
                    state["checkpoint"] = T.save_payload(dest, p, replace=not immutable)
                    append_event(out / "checkpoint_history.jsonl", state["checkpoint"])
                T.write(out / "parameter_drift" / f"step{step}.json", p["parameter_drift"])
                if step == 1500:
                    export = {k: p[k] for k in ("experiment", "arm", "local_step", "global_step",
                        "seam_total_step", "generator_state", "generator_state_sha256", "model_config", "EMA")}
                    export.update(schema="bench_state_schedule_generator_v1", state_sha256=p["generator_state_sha256"],
                                  training_checkpoint_sha256=state["checkpoint"]["sha256"])
                    state["inference_export"] = T.save_payload(out / "generator_step1500.pt", export)
                T.write(status_path, state)

            state["status"] = "TRAINING"
            if not previous:
                checkpoint_save()
            from . import async_core, perturbed_core
            for entry in steps[step:]:
                if stop["signal"] is not None:
                    checkpoint_save()
                    state.update(status="INTERRUPTED_SAFE", signal=stop["signal"])
                    break
                current = int(entry["step"])
                tick = time.perf_counter()
                if current == step+1 and (current == 1 or current % 25 == 1):
                    rr = T.resource()
                    append_event(out / "resource_checks.jsonl", dict(local_step_before=step, **rr))
                    if rr["free_mib"] is None or rr["free_mib"] < 2048:
                        checkpoint_save()
                        state.update(status="INTERRUPTED_SAFE", reason="VRAM_headroom_below_2048MiB")
                        break
                samples, audits = [], []
                for row in entry["samples"]:
                    original = D.sample_for_training(cases[row["condition_id"]], row)
                    if arm == "PERTURBED_STATE":
                        s, audit_row = perturbed_core.apply_perturbation(original, row)
                    else:
                        s = async_core.transform_sample(original)
                        audit_row = dict(tip_clock=float(s["seam_time"]),
                            tip_active=float(s["t"]) < float(np.float32(.7)))
                    samples.append(s)
                    audits.append(audit_row)
                denominator = sum(int(s["free_coordinate_count"]) for s in samples)
                input_seconds = time.perf_counter()-tick
                batch_start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                optimizer.zero_grad(set_to_none=True)
                total_loss, rows = 0., []
                for mi, (s, row, audit_row) in enumerate(zip(samples, entry["samples"], audits)):
                    context = dict(local_step=current, global_step=12000+current,
                                   microbatch=mi, condition_id=row["condition_id"])
                    b = T.tensors(s)
                    forward = async_core.forward if arm == "ASYNC_LEAD_SEAM" else seam_forward
                    with native_execution(model, coordinates=(b["x"], b["t"]), autocast=True):
                        pred, _ = ledger.call("train_forward", lambda: forward(
                            model, b["x"], b["t"], b["y"], b["valid"], b["known"], b["fixed"]).float(), context)
                        if arm == "ASYNC_LEAD_SEAM":
                            loss, _ = async_core.loss_and_stats(pred, b, denominator=denominator)
                        else:
                            loss = corner_masked_fm_loss(pred, b["target"], b["valid"], b["fixed"], denominator=denominator)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("Nonfinite objective")
                        ledger.call("train_backward", loss.backward, context)
                    st = statistics(pred, b, s, arm)
                    if not np.isclose(st["all_free_velocity_sse"]/denominator, float(loss.detach()), rtol=4e-6, atol=1e-7):
                        raise ValueError("Loss accounting differs")
                    total_loss += float(loss.detach())
                    append_event(out / "exposure.jsonl", {**row, **context, "t": float(s["t"]),
                        "N": int(s["N"]), "K": int(s["K"]), "loss_contribution": float(loss.detach()),
                        **st, "mechanism": audit_row})
                    rows.append(st)
                    del pred, loss, b
                if any(p.grad is None for p in model.parameters()):
                    raise RuntimeError("Unused generator parameter")
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                ledger.call("optimizer_update", optimizer.step, dict(local_step=current, global_step=12000+current))
                step = current
                T.validate_optimizer(model, optimizer, "A_COMMON_LR1E5", step)
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError("Nonfinite updated parameter")
                sums = {key: sum(r[key] for r in rows) for key in rows[0]}
                result = dict(local_step=step, global_step=12000+step, objective=total_loss,
                    free_coordinate_denominator=denominator, **sums, generator_gradient_norm=gn,
                    input_seconds=input_seconds, update_seconds=time.perf_counter()-batch_start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_scope="real formal batch including backward/AdamW; allocator not process VRAM")
                append_event(out / "training_log.jsonl", result)
                state.update(local_step=step, global_step=12000+step, last=result, elapsed_seconds=time.perf_counter()-started)
                if step % 250 == 0:
                    checkpoint_save()
                T.write(status_path, state)
                if step in (1, 5, 25) or step % 100 == 0:
                    T.write(out / f"resources_step{step}.json", T.resource())
                    print(json.dumps(dict(arm=arm, **result)), flush=True)
                if step == 1:
                    T.write(out / "first_formal_backward.json", dict(status="PASS", no_extra_forward=True, **result))
            else:
                state["status"] = "COMPLETE"
            T.require_cursor(ledger, step)
            T.write(status_path, state)
    except BaseException as exc:
        state.update(status="OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "FAILED",
                     error=repr(exc), traceback=traceback.format_exc(), no_automatic_replay=True)
        T.write(adir / "failure.json", state)
        raise
    finally:
        for sig, old in handlers.items():
            signal.signal(sig, old)
        state.update(local_step=step, global_step=12000+step, elapsed_seconds=time.perf_counter()-started,
                     calls=ledger.snapshot(), resources=observer.summary())
        T.write(adir / "resources.json", dict(samples=observer.samples, summary=observer.summary()))
        if owned:
            T.write(status_path, state)
        T.write(adir / "final_status.json", state)
    return state


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-cpu", action="store_true")
    args = p.parse_args()
    if args.check_cpu:
        r, m, s = checked()
        print(json.dumps(dict(status="PASS", updates=1500, train_conditions=len(m["train"]),
                              new_neural_probes=0, CUDA_initialized=torch.cuda.is_initialized())))
    else:
        if not args.arm:
            p.error("--arm required")
        print(json.dumps(train(args.arm, args.resume), indent=2))
