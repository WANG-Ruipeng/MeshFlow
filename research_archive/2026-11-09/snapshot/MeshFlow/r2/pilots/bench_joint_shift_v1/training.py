"""One frozen Bench joint translation arm, 1500 updates.

Imports the proven IO/checkpoint utilities, never the old experiment state machine.
No extra fixed neural probes; original SEAM loss and A0 initialization.
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
from . import data as D
from .common import OUT, ARMS, ARM, EXPERIMENT, BASELINE_ROOT, resolve

SCHEMA = "bench_joint_shift_training_v1"
UPDATES = 1500


def checked():
    """Frozen metadata and actual case/schedule checks, with zero model calls."""
    reg = T.read(OUT / "training_registration.json")
    if (reg["status"], reg["experiment"], reg["arms"], reg["updates"],
        reg["batch"], reg["microbatch"], reg["hybrid_lambda"]) != (
            "FROZEN", EXPERIMENT, list(ARMS), 1500, 8, 1, .25):
        raise ValueError("Frozen joint translation definition differs")
    if reg["model_config"] != MODEL_CONFIG or reg["training_seed"] != 20261020:
        raise ValueError("Same A architecture and initial RNG contract required")
    if not reg["training_code_hashes"]:
        raise ValueError("Frozen source hashes required")
    for path, expected in reg["training_code_hashes"].items():
        if digest(resolve(path)) != expected:
            raise ValueError("Frozen training code differs: " + path)
    for key in ("manifest", "input_schedule", "source_checkpoint", "eval_manifest"):
        if digest(resolve(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("Frozen input differs: " + key)
    for path, expected in reg["paired_reference_inputs"].items():
        if digest(resolve(path)) != expected:
            raise ValueError("Paired reference input changed: " + path)
    expected_start = BASELINE_ROOT / "training/A_COMMON_LR1E5/checkpoint_step0.pt"
    if resolve(reg["source_checkpoint"]).resolve() != expected_start.resolve():
        raise ValueError("Only exact retained A step0 can initialize this arm")
    fixed = dict(
        source_checkpoint_sha256="75d54c15f3ff2ee1328dddca1269352db48f5ecbca4d459eaff8c603c2297398",
        source_generator_sha256="7f96d80f8ed4cd6f803a4d5b94fb91560497cb1ea134de69eebf5916de57e5ba",
        source_optimizer_sha256="6e0db9a3226cd9a460d9f87f8b3ac89b217958a590e2b3704fd3656752278ffe",
        source_rng_sha256="5ab3c61d3c7e9e9578bfa19a293c7f33ba4f5cdc81ae59a014027adb8e5a30b4",
    )
    if any(reg.get(key) != value for key, value in fixed.items()):
        raise ValueError("A step0 identity changed")
    manifest = T.read(resolve(reg["manifest"]))
    steps = T.read(resolve(reg["input_schedule"]))["steps"]
    D.validate_schedule(steps)
    actual = dict(rows=len(manifest["rows"]), train=len(manifest["train"]),
                  dev=len(manifest["dev"]),
                  train_parents=len({r["parent_id"] for r in manifest["train"]}))
    if actual != reg["data_counts"] or actual["train_parents"] < 4 or actual["train"] < 1:
        raise ValueError("Actual new pool differs from frozen realized counts")
    ids = {r["condition_id"] for r in manifest["train"]}
    if len(ids) != actual["train"]:
        raise ValueError("Duplicate training condition ID")
    by_id = {r["condition_id"]: r for r in manifest["train"]}
    train_parents = {r["parent_id"] for r in manifest["train"]}
    dev_parents = {r["parent_id"] for r in manifest["dev"]}
    panel = T.read(resolve(reg["eval_manifest"]))
    old_panel = T.read(BASELINE_ROOT / "eval_manifest.json")
    if panel != old_panel or len(panel["requests"]) != 48:
        raise ValueError("Fixed historical A48 evaluation must remain identical")
    dev_parents |= {r["parent_id"] for r in panel["requests"] if r["group"] == "DEV"}
    if train_parents & dev_parents:
        raise ValueError("Held-out parent enters training")
    for row in manifest["rows"]:
        if digest(resolve(row["case_path"])) != row["case_sha256"]:
            raise ValueError("Frozen case changed: " + row["condition_id"])
    if [e["step"] for e in steps] != list(range(1, 1501)) or any(len(e["samples"]) != 8 for e in steps):
        raise ValueError("Exactly 1500x8 registered exposures required")
    for entry in steps:
        rows = entry["samples"]
        if len({r["parent_id"] for r in rows}) != 4:
            raise ValueError("Four distinct parents per effective batch required")
        for row in rows:
            if row["condition_id"] not in ids or row["parent_id"] != by_id[row["condition_id"]]["parent_id"]:
                raise ValueError("Unknown training identity")
            scale = np.asarray(row["scale_xyz"], np.float64)
            if scale.shape != (3,) or not np.isfinite(scale).all() or (scale < .92).any() or (scale > 1.08).any():
                raise ValueError("Frozen xyz augmentation range differs")
            for key in ("noise_seed", "epsilon_seed", "time_seed", "permutation_seed"):
                if not isinstance(row[key], int):
                    raise ValueError("Explicit integer sample RNG identity required")
        for parent in {r["parent_id"] for r in rows}:
            paired = [r for r in rows if r["parent_id"] == parent]
            if (len(paired) != 2 or len({r["condition_id"] for r in paired}) != 1
                    or not any(np.array_equal(np.asarray(r["scale_xyz"]), np.ones(3)) for r in paired)):
                raise ValueError("Each selected condition needs identity plus paired xyz augmentation")
    return reg, manifest, steps


def statistics(pred, b, sample, arm):
    error = (pred.detach().float() - b["target"]).reshape(int(sample["N"]), 3, 3).square().cpu().numpy()
    fixed = np.asarray(sample["anchor_ids"]) >= 0
    valid = np.asarray(sample["valid_mask"], bool)
    free = valid[:, None] & ~fixed
    tip = free & (fixed.sum(1) == 2)[:, None]
    bulk = free & (fixed.sum(1) == 0)[:, None]
    endpoint_error = error * np.float32(1. - float(sample["t"])) ** 2
    return dict(all_free_velocity_sse=float(error[free].sum(dtype=np.float64)),
        all_free_scalar_count=int(free.sum()) * 3,
        seam_tip_velocity_sse=float(error[tip].sum(dtype=np.float64)),
        seam_tip_scalar_count=int(tip.sum()) * 3,
        bulk_velocity_sse=float(error[bulk].sum(dtype=np.float64)),
        bulk_scalar_count=int(bulk.sum()) * 3,
        endpoint_sse=float(endpoint_error[free].sum(dtype=np.float64)),
        seam_tip_endpoint_sse=float(endpoint_error[tip].sum(dtype=np.float64)),
        bulk_endpoint_sse=float(endpoint_error[bulk].sum(dtype=np.float64)))

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
    observer = T.historical.GPUObserver("bench_joint_shift_" + arm, interval=2.)
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
                immutable = step in (0, 500, 1500)
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
                    export.update(schema="bench_joint_shift_generator_v1", state_sha256=p["generator_state_sha256"],
                                  training_checkpoint_sha256=state["checkpoint"]["sha256"])
                    state["inference_export"] = T.save_payload(out / "generator_step1500.pt", export)
                T.write(status_path, state)

            state["status"] = "TRAINING"
            if not previous:
                checkpoint_save()
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
                    s = D.sample_for_training(cases[row["condition_id"]], row)
                    if not 128 <= int(s["N"]) <= 256:
                        raise ValueError("Frozen N128..256 exceeded")
                    audit_row = dict(policy="JOINT_C_Y_TRANSLATION_STANDARD_GAUSSIAN_ORIGINAL_SEAM_FM",
                        source_owner_count=int(np.asarray(s["occupied_labels"], bool).sum()),
                        C_boundary_edges=int(len(s["occupied_labels"])),
                        closed_C=not len(s["occupied_labels"]))
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
                    forward = seam_forward
                    with native_execution(model, coordinates=(b["x"], b["t"]), autocast=True):
                        pred, _ = ledger.call("train_forward", lambda: forward(
                            model, b["x"], b["t"], b["y"], b["valid"], b["known"], b["fixed"]).float(), context)
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
    p.add_argument("--arm", choices=ARMS, default=ARM)
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
