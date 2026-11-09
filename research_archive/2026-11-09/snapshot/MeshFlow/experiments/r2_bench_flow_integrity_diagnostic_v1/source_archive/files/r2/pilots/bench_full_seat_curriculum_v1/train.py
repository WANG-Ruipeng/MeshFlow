"""Registered Bench full-seat Native -> SEAM training with continuous AdamW.

The registration owns every budget. No probes, occupancy-head forward or head
training are performed. Stage changes preserve the live generator, its original
single-group AdamW, and RNG. GPU work additionally needs a resource dispatch GO.
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
from meshflow_control import runtime
import torch
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash, rng_state, restore_rng
from meshflow_control.models.native import build_model, native_execution
from meshflow_control.seam.model_adapter import seam_forward, corner_masked_fm_loss
from meshflow_control.training.context_checkpoint import _cpu_copy, _check_generator
from meshflow_control.training.schedule40 import make_optimizer
from meshflow_control.data.io import digest, append_event
from r2.pilots.bench_adaptation_strength_v1 import training as T
from r2.pilots.bench_parts_convergence_v1 import data as legacy_data
from .common import OUT, STAGES, EXPERIMENT, SOURCE, HEAD, HEAD_SHA256, resolve
from . import training_state as S


def checked():
    """Validate frozen metadata/case bytes; no OT, model construction or forward."""
    reg = T.read(OUT / "training_registration.json")
    if (reg["status"], reg["experiment"], reg["batch"], reg["microbatch"],
            reg["hybrid_lambda"], reg["model_config"]) != (
            "FROZEN", EXPERIMENT, 8, 1, .25, MODEL_CONFIG):
        raise ValueError("Frozen curriculum contract differs")
    if reg.get("stages", list(STAGES)) != list(STAGES):
        raise ValueError("Stage order differs")
    for stage in STAGES:
        S.stage_spec(reg, stage)
    if any(reg.get(key) != value for key, value in S.SOURCE_IDENTITIES.items()):
        raise ValueError("Only original COMMON12000 with matching AdamW/RNG accepted")
    if resolve(reg["source_checkpoint"]).resolve() != SOURCE.resolve():
        raise ValueError("COMMON checkpoint path differs; fresh A-step0 is not accepted")
    if (resolve(reg["head_source_checkpoint"]).resolve() != HEAD.resolve()
            or reg["head_source_sha256"] != HEAD_SHA256
            or reg["head_state_sha256"] != S.HEAD_STATE_SHA256):
        raise ValueError("Only the frozen retained OLD_S15000 head is accepted")
    for key in ("head_forward_budget", "head_optimizer_updates", "probe_forward_budget"):
        if reg.get(key, 0) != 0:
            raise ValueError("No head/probe activity may be added to training")
    if not reg["training_code_hashes"]:
        raise ValueError("Frozen source manifest is required")
    for path, expected in reg["training_code_hashes"].items():
        if digest(resolve(path)) != expected:
            raise ValueError("Frozen training code changed: " + path)
    for key in ("manifest", "input_schedule", "source_checkpoint"):
        if digest(resolve(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("Frozen asset changed: " + key)
    if digest(resolve(reg["head_source_checkpoint"])) != reg["head_source_sha256"]:
        raise ValueError("Frozen head file changed")
    manifest = T.read(resolve(reg["manifest"]))
    steps = T.read(resolve(reg["input_schedule"]))["steps"]
    train = manifest["train"]
    parents = {r["parent_id"] for r in train}
    by_id = {r["condition_id"]: r for r in train}
    if len(parents) != len(train) or len(by_id) != len(train) or len(train) < 4:
        raise ValueError("One distinct condition per TRAIN parent; at least four parents required")
    if parents & {r["parent_id"] for r in manifest.get("dev", [])}:
        raise ValueError("DEV parent entered training")
    if "eval_manifest" in reg:
        if digest(resolve(reg["eval_manifest"])) != reg["eval_manifest_sha256"]:
            raise ValueError("Evaluation manifest changed")
        requests = T.read(resolve(reg["eval_manifest"]))["requests"]
        if parents & {r["parent_id"] for r in requests if r.get("eval_group", r.get("group")) == "DEV"}:
            raise ValueError("Evaluation DEV parent entered training")
    for row in manifest.get("rows", train):
        if digest(resolve(row["case_path"])) != row["case_sha256"]:
            raise ValueError("Frozen case changed: " + row["condition_id"])
    total = sum(reg["stage_updates"].values())
    if [r["step"] for r in steps] != list(range(1, total + 1)):
        raise ValueError("Continuous input schedule cursor differs from both stage budgets")
    seen = {stage: set() for stage in STAGES}
    for entry in steps:
        stage = "NATIVE" if entry["step"] <= reg["stage_updates"]["NATIVE"] else "SEAM"
        spec = S.stage_spec(reg, stage)
        if entry["stage"] != stage or entry["stage_step"] != entry["step"] - spec["offset"]:
            raise ValueError("Stage boundary differs from the frozen schedule")
        batch = entry["samples"]
        if len(batch) != 8 or len({r["parent_id"] for r in batch}) != 4:
            raise ValueError("Four distinct parents, each paired clean/width, per effective batch")
        for row in batch:
            if row["condition_id"] not in by_id or row["parent_id"] != by_id[row["condition_id"]]["parent_id"]:
                raise ValueError("Unregistered training identity")
            alpha = float(row["alpha"])
            if not np.isfinite(alpha) or not .9 <= alpha <= 1.1:
                raise ValueError("Width augmentation must be U[.9,1.1]")
            if "scale_xyz" in row and not np.array_equal(np.asarray(row["scale_xyz"]), [alpha, 1., 1.]):
                raise ValueError("Only x-width augmentation is registered")
            if any(type(row[key]) is not int for key in ("noise_seed", "epsilon_seed", "time_seed", "permutation_seed")):
                raise ValueError("All sample RNG identifiers must be explicit integers")
            seen[stage].add(row["condition_id"])
        for parent in {r["parent_id"] for r in batch}:
            pair = [r for r in batch if r["parent_id"] == parent]
            if len(pair) != 2 or len({r["condition_id"] for r in pair}) != 1 or not any(float(r["alpha"]) == 1. for r in pair):
                raise ValueError("Parent must have the same C at identity plus x-width augmentation")
    if any(ids != set(by_id) for ids in seen.values()):
        raise ValueError("Both stages must use exactly the same complete frozen TRAIN pool")
    return reg, manifest, steps


def _head_state(reg):
    p = torch.load(resolve(reg["head_source_checkpoint"]), map_location="cpu", weights_only=True, mmap=True)
    head_state = _cpu_copy(p["head_state"])
    S.validate_head_state(head_state, reg["head_state_sha256"])
    return head_state


def _common_payload(reg):
    p = torch.load(resolve(reg["source_checkpoint"]), map_location="cpu", weights_only=True, mmap=True)
    if (p["schema"], p["arm"], p["step"], p["global_step"], p["EMA"],
            p["head_state"], p["head_optimizer"], p["model_config"]) != (
            "bench_parts_training_v1", "COMMON", 12000, 12000, False, None, None, MODEL_CONFIG):
        raise ValueError("Original COMMON source identity differs")
    for key, fn, hk in (("generator_state", state_hash, "source_generator_sha256"),
            ("optimizer", T.base.tree_hash, "source_optimizer_sha256"),
            ("rng", T.base.tree_hash, "source_rng_sha256")):
        if fn(p[key]) != reg[hk]:
            raise ValueError("Original COMMON state identity differs: " + key)
    _check_generator(p["generator_state"])
    return p


def _build_restore(payload, global_step, device):
    model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode="none", trainable=True).eval()
    model.load_state_dict(payload["generator_state"], strict=True)
    model.to(device)
    optimizer = make_optimizer(model, None)
    optimizer.load_state_dict(payload["optimizer"])
    names = S.validate_optimizer(model, optimizer, global_step)
    if "optimizer_parameter_names" in payload and payload["optimizer_parameter_names"] != names:
        raise ValueError("Single-group AdamW parameter names/order differ")
    if state_hash(model.state_dict()) != payload["generator_state_sha256"]:
        raise ValueError("Restored generator content differs")
    # Hash before any update; restore may transfer moments but must preserve values.
    expected_optimizer = payload["optimizer_state_sha256"] if "optimizer_state_sha256" in payload else T.base.tree_hash(payload["optimizer"])
    if T.base.tree_hash(optimizer.state_dict()) != expected_optimizer:
        raise ValueError("Original AdamW moments were reset, regrouped or changed")
    return model, optimizer


def _log_cursor(out, step):
    """Refuse stale/partial scientific logs as well as stale call ledgers."""
    for name, per_update in (("training_log.jsonl", 1), ("exposure.jsonl", 8)):
        path = out / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        expected = [s for s in range(1, step + 1) for _ in range(per_update)]
        if [r["stage_step"] for r in rows] != expected:
            raise RuntimeError("Incomplete/stale scientific log; explicit recovery required: " + name)
        if per_update == 8 and [r["microbatch"] for r in rows] != list(range(8)) * step:
            raise RuntimeError("Exposure microbatch sequence differs")


def _receipt_matches(receipt, payload):
    for key in ("generator_state_sha256", "optimizer_state_sha256", "rng_sha256", "head_state_sha256"):
        if receipt.get(key) != payload[key]:
            raise ValueError("Immutable checkpoint differs: " + key)


def _save(path, payload, replace=False):
    receipt = T.save_payload(path, payload, replace=replace)
    receipt.update({k: payload[k] for k in ("stage", "stage_step", "curriculum_step", "head_state_sha256") if k in payload})
    T.write(path.with_suffix(".json"), receipt)
    return receipt


def train(stage="NATIVE", resume=False):
    if stage not in STAGES:
        raise ValueError(stage)
    T.install_io(OUT, stage)
    reg, manifest, steps = checked()
    spec = S.stage_spec(reg, stage)
    reg_sha = digest(OUT / "training_registration.json")
    go = T.read(OUT / "dispatch" / (stage + "_go.json"))
    if (not go.get("authorized") or go["training_registration_sha256"] != reg_sha
            or go.get("stage") != stage):
        raise RuntimeError("Parent resource-aware dispatch release missing")
    headroom = max(2048., float(go.get("retained_headroom_mib", 2048.)))
    out = OUT / "training" / stage
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    # Hold ownership before inspecting or creating status, ledger or attempts.
    with T.historical.exclusive_writer(out):
        previous = T.read(status_path) if status_path.exists() else None
        if previous and previous["status"] == "COMPLETE":
            if previous["registration_sha256"] != reg_sha or previous["stage_step"] != spec["updates"]:
                raise ValueError("Completed stage identity differs")
            for key in ("checkpoint", "inference_export"):
                if digest(resolve(previous[key]["path"])) != previous[key]["sha256"]:
                    raise ValueError("Final artifact changed: " + key)
            return previous
        if previous and (not resume or previous["status"] != "INTERRUPTED_SAFE"):
            raise RuntimeError("Unresolved prior attempt; no automatic scientific replay")
        if resume and not previous:
            raise ValueError("Exact saved cursor required for resume")
        step = int(previous["stage_step"]) if previous else 0
        attempt = "attempt_" + str(time.time_ns())
        adir = out / "attempts" / attempt
        adir.mkdir(parents=True, exist_ok=False)
        ledger = T.historical.CallLedger(out / "calls.jsonl", S.limits(reg, stage), stage, attempt)
        S.require_cursor(ledger, reg, stage, step)
        _log_cursor(out, step)
        state = dict(status="LOADING", arm=stage, **S.cursor(reg, stage, step),
                     target_local_step=spec["updates"], registration_sha256=reg_sha,
                     pid=os.getpid(), attempt=attempt, head_step=0, head_calls=0, probe_calls=0)
        if previous:
            state["checkpoint"] = previous["checkpoint"]
        initial = T.resource()
        T.write(adir / "resources_dispatch.json", initial)
        if initial["free_mib"] is None or initial["free_mib"] < headroom:
            raise RuntimeError("RESOURCE_WAIT before model load")
        started = time.perf_counter()
        observer = T.historical.GPUObserver("bench_full_seat_" + stage, interval=2.)
        stop = {"signal": None}
        def handler(sig, unused):
            stop["signal"] = sig
        handlers = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
        T.write(status_path, state)
        try:
            with observer:
                torch.set_num_threads(2)
                runtime.configure_stable_runtime()
                if previous:
                    source = resolve(previous["checkpoint"]["path"])
                    if digest(source) != previous["checkpoint"]["sha256"]:
                        raise ValueError("Resume file hash differs")
                    p = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
                    S.validate_payload(p, reg, stage, step, reg_sha)
                    if p["execution"] != ledger.snapshot():
                        raise ValueError("Saved call ledger differs from actual completed work")
                    audit = p["initialization"]
                    head_state = p["head_state"]
                elif stage == "NATIVE":
                    p = _common_payload(reg)
                    source = resolve(reg["source_checkpoint"])
                    head_state = _head_state(reg)
                    audit = dict(source_path=str(source), source_sha256=reg["source_checkpoint_sha256"],
                        generator_initial_sha256=reg["source_generator_sha256"],
                        optimizer_initial_sha256=reg["source_optimizer_sha256"],
                        rng_initial_sha256=reg["source_rng_sha256"], source_global_step=12000,
                        inherited_optimizer=True, inherited_rng=True, fresh_AdamW=False,
                        new_generator_parameters=0, head_frozen=True, head_calls=0,
                        head_source_checkpoint=reg["head_source_checkpoint"],
                        head_source_sha256=reg["head_source_sha256"], head_state_sha256=reg["head_state_sha256"])
                else:
                    native = T.read(OUT / "training/NATIVE/status.json")
                    if (native["status"], native["registration_sha256"], native["stage_step"]) != (
                            "COMPLETE", reg_sha, reg["stage_updates"]["NATIVE"]):
                        raise ValueError("SEAM requires the complete registered Native endpoint")
                    source = resolve(native["checkpoint"]["path"])
                    if digest(source) != native["checkpoint"]["sha256"]:
                        raise ValueError("Native endpoint hash differs")
                    p = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
                    S.validate_payload(p, reg, "NATIVE", reg["stage_updates"]["NATIVE"], reg_sha)
                    audit = copy.deepcopy(p["initialization"])
                    audit["stage_transition"] = dict(source_path=str(source), source_sha256=native["checkpoint"]["sha256"],
                        generator_state_sha256=p["generator_state_sha256"], optimizer_state_sha256=p["optimizer_state_sha256"],
                        rng_sha256=p["rng_sha256"], optimizer_reset=False, head_frozen=True,
                        same_manifest_sha256=reg["manifest_sha256"], source_global_step=spec["global_start"])
                    head_state = p["head_state"]
                model, optimizer = _build_restore(p, state["global_step"], "cuda")
                S.validate_head_state(head_state)
                restore_rng(p["rng"])
                expected_rng = p["rng_sha256"] if "rng_sha256" in p else T.base.tree_hash(p["rng"])
                if T.base.tree_hash(rng_state()) != expected_rng:
                    raise ValueError("Restored RNG differs")
                del p
                T.write(adir / "initialization.json", audit)
                cases = {r["condition_id"]: legacy_data.load_case(r) for r in manifest["train"]}

                def checkpoint_save():
                    S.require_cursor(ledger, reg, stage, step)
                    _log_cursor(out, step)
                    position = S.cursor(reg, stage, step)
                    names = S.validate_optimizer(model, optimizer, position["global_step"])
                    g, o, rng = _cpu_copy(model.state_dict()), _cpu_copy(optimizer.state_dict()), _cpu_copy(rng_state())
                    _check_generator(g)
                    S.validate_head_state(head_state)
                    payload = dict(schema=S.SCHEMA, experiment=EXPERIMENT, arm=stage, **position,
                        source_global_step=12000, model_config=copy.deepcopy(MODEL_CONFIG),
                        EMA=False, update_in_progress=False, generator_state=g, generator_state_sha256=state_hash(g),
                        optimizer=o, optimizer_state_sha256=T.base.tree_hash(o), optimizer_parameter_names=names,
                        rng=rng, rng_sha256=T.base.tree_hash(rng), registration_sha256=reg_sha,
                        head_state=head_state, head_state_sha256=S.HEAD_STATE_SHA256,
                        head_config=S.HEAD_CONFIG, head_optimizer=None, head_step=0, head_calls=0,
                        initialization=audit, execution=ledger.snapshot(),
                        stream_state=dict(batch_index=position["curriculum_step"], stage_batch_index=step,
                                          next_sample_index=8 * position["curriculum_step"]))
                    immutable = step in spec["full_steps"]
                    dest = out / (f"checkpoint_step{step}.pt" if immutable else "checkpoint_latest.pt")
                    if immutable and dest.exists():
                        receipt = T.read(dest.with_suffix(".json"))
                        if digest(dest) != receipt["sha256"]:
                            raise ValueError("Immutable checkpoint file changed")
                        _receipt_matches(receipt, payload)
                    else:
                        receipt = _save(dest, payload, replace=not immutable)
                        append_event(out / "checkpoint_history.jsonl", receipt)
                    state["checkpoint"] = receipt
                    # Native step zero is the exact COMMON live reference export.
                    # Export is a CPU serialization of already loaded state: zero forward.
                    if step == spec["updates"] or (stage == "NATIVE" and step == 0):
                        keys = ("experiment", "arm", "stage", "stage_step", "local_step", "curriculum_step", "global_step",
                            "native_total_step", "seam_total_step", "generator_state", "generator_state_sha256",
                            "model_config", "EMA", "head_state", "head_state_sha256", "head_config", "head_step")
                        export = {k: payload[k] for k in keys}
                        export.update(schema=S.INFERENCE_SCHEMA, state_sha256=payload["generator_state_sha256"],
                            training_checkpoint_sha256=receipt["sha256"], threshold_logit=0.,
                            feature_schema="C_edge24_world_chair_v1", registration_sha256=reg_sha,
                            model_arm="PLAIN" if stage == "NATIVE" else "SEAM", head_frozen=True)
                        export_path = out / f"generator_step{step}.pt"
                        if export_path.exists():
                            er = T.read(export_path.with_suffix(".json"))
                            if digest(export_path) != er["sha256"] or er["generator_state_sha256"] != payload["generator_state_sha256"] or er["head_state_sha256"] != S.HEAD_STATE_SHA256:
                                raise ValueError("Immutable endpoint export differs")
                            state["inference_export"] = er
                        else:
                            state["inference_export"] = _save(export_path, export)
                    state.update(position)
                    T.write(status_path, state)

                state["status"] = "TRAINING"
                if not previous:
                    checkpoint_save()
                selected = steps[spec["offset"] + step:spec["offset"] + spec["updates"]]
                for entry in selected:
                    if stop["signal"] is not None:
                        checkpoint_save()
                        state.update(status="INTERRUPTED_SAFE", signal=stop["signal"])
                        break
                    tick = time.perf_counter()
                    current = int(entry["stage_step"])
                    if current % 25 == 1:
                        resource = T.resource()
                        append_event(out / "resource_checks.jsonl", dict(stage_step_before=step, **resource))
                        if resource["free_mib"] is None or resource["free_mib"] < headroom:
                            checkpoint_save()
                            state.update(status="INTERRUPTED_SAFE", reason="VRAM_headroom_below_dispatch_margin")
                            break
                    # The audited parts adapter performs no owner restriction in PLAIN,
                    # source-owner SEAM packing in SEAM; exactly one required OT/sample.
                    samples = [legacy_data.sample_for_training(cases[r["condition_id"]], r,
                        "PLAIN" if stage == "NATIVE" else "SEAM") for r in entry["samples"]]
                    audits = [S.verify_sample(s, stage) for s in samples]
                    denominator = sum(int(s["free_coordinate_count"]) for s in samples)
                    input_seconds = time.perf_counter() - tick
                    begin = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    optimizer.zero_grad(set_to_none=True)
                    fm, statistics = 0., []
                    for mi, (sample, row, audit_row) in enumerate(zip(samples, entry["samples"], audits)):
                        context = dict(**S.cursor(reg, stage, current), microbatch=mi, condition_id=row["condition_id"])
                        b = T.tensors(sample)
                        with native_execution(model, coordinates=(b["x"], b["t"]), autocast=True):
                            if stage == "SEAM":
                                call = lambda: seam_forward(model, b["x"], b["t"], b["y"], b["valid"], b["known"], b["fixed"]).float()
                            else:
                                call = lambda: model(b["x"], b["t"], b["y"], valid_mask=b["valid"], known_mask=b["known"]).float()
                            pred, _ = ledger.call("train_forward", call, context)
                            loss = corner_masked_fm_loss(pred, b["target"], b["valid"], b["fixed"], denominator=denominator)
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError("Nonfinite original FM objective")
                            ledger.call("train_backward", loss.backward, context)
                        values = S.statistics(pred, b["target"], sample)
                        if not np.isclose(values["all_free_velocity_sse"] / denominator, float(loss.detach()), rtol=4e-6, atol=1e-7):
                            raise ValueError("Loss accounting differs from original whole-batch free scalars")
                        fm += float(loss.detach())
                        statistics.append(values)
                        append_event(out / "exposure.jsonl", {**row, **context, "t": float(sample["t"]),
                            "N": int(sample["N"]), "K": int(sample["K"]), "loss_contribution": float(loss.detach()),
                            **values, "mechanism": audit_row, "head_forward_called": False})
                        del pred, loss, b
                    if any(p.grad is None for p in model.parameters()):
                        raise RuntimeError("Unused generator parameter")
                    grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                    ledger.call("optimizer_update", optimizer.step, S.cursor(reg, stage, current))
                    step = current
                    S.validate_optimizer(model, optimizer, spec["global_start"] + step)
                    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                        raise FloatingPointError("Nonfinite updated parameter")
                    sums = {k: sum(r[k] for r in statistics) for k in statistics[0]}
                    result = dict(**S.cursor(reg, stage, step), FM=fm, objective=fm,
                        free_coordinate_denominator=denominator, **sums, generator_gradient_norm=grad,
                        head_step=0, head_calls=0, input_seconds=input_seconds,
                        update_seconds=time.perf_counter() - begin,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        peak_scope="one formal batch including generator backward and inherited AdamW update; allocator not process VRAM")
                    append_event(out / "training_log.jsonl", result)
                    state.update(S.cursor(reg, stage, step), last=result, elapsed_seconds=time.perf_counter() - started)
                    if step % 250 == 0 or step in spec["full_steps"]:
                        checkpoint_save()
                    T.write(status_path, state)
                    if step in (1, 5, 25) or step % 100 == 0:
                        T.write(out / f"resources_step{step}.json", T.resource())
                        print(json.dumps(dict(arm=stage, **result)), flush=True)
                    if step == 1:
                        T.write(out / "first_formal_backward.json", dict(status="PASS", no_extra_forward=True, **result))
                else:
                    state["status"] = "COMPLETE"
                S.require_cursor(ledger, reg, stage, step)
                _log_cursor(out, step)
                T.write(status_path, state)
        except BaseException as exc:
            state.update(status="OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "FAILED",
                error=repr(exc), traceback=traceback.format_exc(), no_automatic_replay=True)
            T.write(adir / "failure.json", state)
            raise
        finally:
            for sig, old in handlers.items():
                signal.signal(sig, old)
            state.update(S.cursor(reg, stage, step), elapsed_seconds=time.perf_counter() - started,
                         calls=ledger.snapshot(), resources=observer.summary())
            T.write(adir / "resources.json", dict(samples=observer.samples, summary=observer.summary()))
            T.write(status_path, state)
            T.write(adir / "final_status.json", state)
        return state


def check_cpu():
    """Load/validate real source weights and moments on CPU; no neural/OT call."""
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU check must start without a CUDA context")
    reg, manifest, steps = checked()
    p = _common_payload(reg)
    model, optimizer = _build_restore(p, S.BASE_STEP, "cpu")
    head = _head_state(reg)
    result = dict(status="PASS", schema="bench_full_seat_curriculum_cpu_preflight_v1",
        source_checkpoint_sha256=reg["source_checkpoint_sha256"],
        generator_state_sha256=state_hash(model.state_dict()), optimizer_state_sha256=T.base.tree_hash(optimizer.state_dict()),
        source_rng_sha256=T.base.tree_hash(p["rng"]), optimizer_parameter_count=len(optimizer.state),
        optimizer_step_values=sorted({float(s["step"]) for s in optimizer.state.values()}),
        head_state_sha256=state_hash(head), stage_updates=reg["stage_updates"],
        stage_full_steps=reg["stage_full_steps"], train_parents=len(manifest["train"]),
        registered_exposures=len(steps) * 8, generator_forward_calls=0, backward_calls=0,
        optimizer_updates=0, head_forward_calls=0, OT_calls=0, CUDA_initialized=torch.cuda.is_initialized())
    if result["CUDA_initialized"]:
        raise RuntimeError("CPU check initialized CUDA")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="NATIVE")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-cpu", action="store_true")
    args = parser.parse_args()
    value = check_cpu() if args.check_cpu else train(args.stage, args.resume)
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
