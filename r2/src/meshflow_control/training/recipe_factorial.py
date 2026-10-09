"""Independent FM-only condition-recipe worker; no automatic GPU dispatch.

The old schedule40 COMMON update is reused verbatim as an arithmetic adapter.
All recipe, schedule, checkpoint, authorization and input identities are new.
"""
from __future__ import annotations
from .. import runtime
import copy
import csv
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from ..checkpoints import state_hash, rng_state, restore_rng
from ..config import MODEL_CONFIG
from ..data.io import atomic_json, append_event, digest, json_hash, read_json
from ..models.native import build_model
from .context_checkpoint import _cpu_copy, _check_generator
from . import schedule40 as baseline

EXPERIMENT = "CHAIR_CONDITION_RECIPE_FACTORIAL_V1"
SCHEMA = "chair_condition_recipe_factorial_training_v1"
INFERENCE_SCHEMA = "chair_condition_recipe_factorial_generator_v1"
AUTH_SCHEMA = "chair_condition_recipe_factorial_training_authorization_v1"
TOTAL_UPDATES = 6000
CHECKPOINT_INTERVAL = 100
RECIPES = {
    "R1_40_H_ALL": ("FORTY", "all"),
    "R2_MIX_H_ALL": ("MIX", "all"),
    "R3_40_H_LATE": ("FORTY", "late"),
    "R4_MIX_H_LATE": ("MIX", "late"),
    "C20_40_MIX": ("C20_40_MIX", None),
    "C40_20_MIX": ("C40_20_MIX", None),
}
tree_hash = baseline.tree_hash
make_optimizer = lambda model: baseline.make_optimizer(model, None)
optimizer_names = lambda model: baseline.optimizer_names(model, None)
CallLedger = baseline.CallLedger


def resolved_schedule(recipe, hybrid_schedule=None):
    if recipe not in RECIPES:
        raise ValueError("Unknown recipe")
    fixed = RECIPES[recipe][1]
    if hybrid_schedule is None:
        hybrid_schedule = fixed
    if hybrid_schedule not in ("all", "late") or (fixed is not None and hybrid_schedule != fixed):
        raise ValueError("Recipe HYBRID strategy differs")
    return hybrid_schedule


def lambda_at_step(recipe, step, hybrid_schedule=None):
    if type(step) is not int or not 1 <= step <= TOTAL_UPDATES:
        raise ValueError("Invalid optimizer update index")
    schedule = resolved_schedule(recipe, hybrid_schedule)
    return 0.25 if schedule == "all" or step > 4000 else 0.0


def protocol(recipe, hybrid_schedule, namespace):
    schedule = resolved_schedule(recipe, hybrid_schedule)
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("Explicit frozen input namespace required")
    return dict(experiment=EXPERIMENT, recipe=recipe, input_namespace=namespace,
        condition_schedule=RECIPES[recipe][0], hybrid_schedule=schedule,
        hybrid_late_inclusive=[4001, 6000], hybrid_lambda=0.25, pure_OT_lambda=0.0,
        updates=TOTAL_UPDATES, batch=8, microbatch=1, clean_per_batch=4,
        width_augmentation=[0.9, 1.1], model=copy.deepcopy(MODEL_CONFIG),
        model_seed=baseline.MODEL_SEED, Geo_seed=1010, generator_lr=1e-5,
        betas=[0.9, 0.95], weight_decay=0.0, generator_clip=1.0,
        FM_denominator="whole effective batch valid free scalar count",
        condition_interface="Native/T1+role+Geo; readout none; C hard preservation",
        alignment=False, teacher=False, precision="BF16 forward; FP32 parameters and integration",
        deterministic_strict=True, tf32=False, sdpa="math",
        checkpoint_interval=CHECKPOINT_INTERVAL,
        official_file_sha256=baseline.OFFICIAL_FILE_SHA256,
        official_backbone_sha256=baseline.OFFICIAL_BACKBONE_SHA256,
        update_arithmetic="unchanged schedule40.update COMMON path; no head or teacher",
        no_optimizer_reset_at_lambda_switch=True)


def source_identity():
    paths = [Path(__file__), Path(baseline.__file__),
             Path(__file__).with_name("losses.py"),
             Path(__file__).parents[1] / "data/recipe_factorial.py",
             Path(__file__).parents[1] / "models/native.py",
             Path(__file__).parents[1] / "runtime.py"]
    return {str(p.relative_to(Path(__file__).parents[1])): digest(p) for p in paths}


def check_authorization(path, recipe, hybrid_schedule, namespace, out):
    registration = read_json(path)
    expected = dict(schema=AUTH_SCHEMA, experiment=EXPERIMENT,
                    namespace=namespace, out=str(Path(out).resolve()))
    for key, value in expected.items():
        if registration.get(key) != value:
            raise ValueError("Dispatch authorization differs: " + key)
    if recipe not in registration.get("authorized_recipes", []):
        raise ValueError("Recipe has not been authorized for dispatch")
    if registration.get("hybrid_schedules", {}).get(recipe) != hybrid_schedule:
        raise ValueError("Dispatch authorization HYBRID strategy differs")
    for key, expected_value in (("updates_per_recipe", TOTAL_UPDATES), ("effective_batch", 8), ("microbatch", 1)):
        if registration.get(key) != expected_value:
            raise ValueError("Dispatch scientific budget differs: " + key)
    if "experiment_root" in registration:
        if not Path(out).resolve().is_relative_to(Path(registration["experiment_root"]).resolve()):
            raise ValueError("Dispatch output escapes the registered experiment root")
    sources = registration.get("source_hashes")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Frozen source hashes required")
    repo = Path(__file__).resolve().parents[3]
    required = {"src/meshflow_control/training/recipe_factorial.py",
                "src/meshflow_control/data/recipe_factorial.py",
                "src/meshflow_control/training/schedule40.py",
                "tools/train_recipe_factorial.py"}
    if not required.issubset(sources):
        raise ValueError("Training dependency source registration incomplete")
    for relative, expected_hash in sources.items():
        source = (repo / relative).resolve()
        if not source.is_relative_to(repo) or digest(source) != expected_hash:
            raise ValueError("Registered source changed: " + relative)
    return registration


def validate_optimizer(model, optimizer, step):
    if type(step) is not int or not 0 <= step <= TOTAL_UPDATES:
        raise ValueError("Optimizer cursor outside recipe")
    return baseline.validate_optimizer(model, None, optimizer, "COMMON", step)


def initialize_official(path, device="cpu"):
    baseline._seed_worker()
    model, head, optimizer, audit = baseline.initialize_official(path, "COMMON", device)
    if head is not None:
        raise AssertionError("FM-only recipe constructed an auxiliary head")
    baseline._seed_worker()
    return model, optimizer, dict(audit, arithmetic_source="schedule40 COMMON",
        no_auxiliary_head_constructed=True, teacher_loaded=False)


def validate_samples(samples, recipe, step, hybrid_schedule):
    if len(samples) != 8 or len({s["object_id"] for s in samples}) != 8:
        raise ValueError("Eight distinct parent microbatches required")
    if sum(s["alpha"] == 1.0 for s in samples) != 4:
        raise ValueError("Four exact clean samples required")
    expected_lambda = lambda_at_step(recipe, step, hybrid_schedule)
    condition = RECIPES[recipe][0]
    expected_ratio = 0.4 if condition == "FORTY" else None
    if condition in ("C20_40_MIX", "C40_20_MIX") and step <= 4000:
        first = 0.2 if condition == "C20_40_MIX" else 0.4
        expected_ratio = first if step <= 2000 else 0.6 - first
    for sample in samples:
        if int(sample["step"]) != step or float(sample["lambda_value"]) != expected_lambda:
            raise ValueError("Actual sample step/lambda differs")
        n, k, ratio = int(sample["y"]), int(sample["K"]), float(sample["ratio"])
        if ratio not in (0.2, 0.4) or k != round(ratio * n) or not 128 <= n <= 256:
            raise ValueError("Condition face-count contract differs")
        if expected_ratio is not None and abs(ratio - expected_ratio) > 1e-12:
            raise ValueError("Condition schedule differs")
        if sample["free_coordinate_count"] != 9 * (n-k):
            raise ValueError("Sample free-coordinate denominator differs")
    if expected_ratio is None:
        for ratio in (0.2, 0.4):
            for clean in (False, True):
                if sum(s["ratio"] == ratio and (s["alpha"] == 1.0) == clean for s in samples) != 2:
                    raise ValueError("MIX requires two clean and two augmented examples per ratio")
    return expected_lambda


def update(model, optimizer, samples, ledger, recipe, step, hybrid_schedule):
    lam = validate_samples(samples, recipe, step, hybrid_schedule)
    # This is exactly the inherited no-head COMMON arithmetic, including its
    # whole-batch free denominator, microbatch accumulation, clipping and AdamW.
    result = baseline.update(model, None, optimizer, samples, None, ledger, "COMMON", step)
    if result["alignment_active"] or result["aligned_samples"] or result["head_adamw_step"]:
        raise AssertionError("Unexpected auxiliary training")
    result["lambda_value"] = lam
    return result


def _payload(model, optimizer, *, recipe, hybrid_schedule, namespace, step,
             stream_state, identities, lineage, receipt):
    validate_optimizer(model, optimizer, step)
    if stream_state["batch_index"] != step:
        raise ValueError("Incomplete stream/update checkpoint")
    p = protocol(recipe, hybrid_schedule, namespace)
    generator = _cpu_copy(model.state_dict())
    _check_generator(generator)
    opt, rng = _cpu_copy(optimizer.state_dict()), rng_state()
    return dict(schema=SCHEMA, experiment=EXPERIMENT, recipe=recipe, segment=recipe,
        global_step=step, update_in_progress=False, protocol=p, protocol_sha256=json_hash(p),
        model_config=copy.deepcopy(MODEL_CONFIG), generator_state=generator,
        generator_state_sha256=state_hash(generator), optimizer=opt,
        optimizer_state_sha256=tree_hash(opt), optimizer_parameter_names=optimizer_names(model),
        rng=rng, rng_sha256=tree_hash(rng), stream_state=copy.deepcopy(stream_state),
        stream_state_sha256=json_hash(stream_state), identities=copy.deepcopy(identities),
        lineage=copy.deepcopy(lineage), execution_receipt=copy.deepcopy(receipt),
        aligner_state=None, teacher_state=None,
        next_lambda=None if step == TOTAL_UPDATES else lambda_at_step(recipe, step+1, hybrid_schedule))


def validate_payload(value):
    if value.get("schema") != SCHEMA or value.get("experiment") != EXPERIMENT or value.get("update_in_progress") is not False:
        raise ValueError("Not a completed recipe checkpoint")
    recipe, step, p = value["recipe"], value["global_step"], value["protocol"]
    if type(step) is not int or not 0 <= step <= TOTAL_UPDATES or value["segment"] != recipe:
        raise ValueError("Checkpoint recipe/cursor differs")
    expected = protocol(recipe, p["hybrid_schedule"], p["input_namespace"])
    if p != expected or value["protocol_sha256"] != json_hash(expected) or value["model_config"] != MODEL_CONFIG:
        raise ValueError("Checkpoint scientific protocol differs")
    if value["stream_state"]["batch_index"] != step:
        raise ValueError("Checkpoint input cursor differs")
    if value["aligner_state"] is not None or value["teacher_state"] is not None:
        raise ValueError("FM-only checkpoint has auxiliary tensors")
    expected_next = None if step == TOTAL_UPDATES else lambda_at_step(recipe, step+1, p["hybrid_schedule"])
    if value["next_lambda"] != expected_next:
        raise ValueError("Checkpoint next HYBRID value differs")
    _check_generator(value["generator_state"])
    for field, hasher in (("generator_state", state_hash), ("optimizer", tree_hash),
                          ("rng", tree_hash), ("stream_state", json_hash)):
        key = "optimizer_state_sha256" if field == "optimizer" else field+"_sha256"
        if hasher(value[field]) != value[key]:
            raise ValueError("Checkpoint internal state hash differs: " + field)
    return value


def _write_checkpoint(path, value, immutable=False):
    # Uses the proven same-filesystem atomic serializer. segment is an explicit
    # compatibility field equal to the new recipe; it never means old COMMON.
    receipt = baseline._write_checkpoint(path, value, immutable=immutable)
    return dict(receipt, recipe=value["recipe"], protocol_sha256=value["protocol_sha256"])


def load_checkpoint(path, *, recipe=None, hybrid_schedule=None, namespace=None, device="cpu"):
    runtime.configure_stable_runtime()
    value = validate_payload(torch.load(path, map_location="cpu", weights_only=True, mmap=True))
    p = value["protocol"]
    for key, expected in (("recipe", recipe), ("hybrid_schedule", hybrid_schedule),
                          ("input_namespace", namespace)):
        actual = value["recipe"] if key == "recipe" else p[key]
        if expected is not None and actual != expected:
            raise ValueError("Resume identity differs: " + key)
    model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode="none", trainable=True).eval()
    model.load_state_dict(value["generator_state"], strict=True)
    model.to(device)
    optimizer = make_optimizer(model)
    if value["optimizer_parameter_names"] != optimizer_names(model):
        raise ValueError("Optimizer parameter mapping differs")
    optimizer.load_state_dict(value["optimizer"])
    validate_optimizer(model, optimizer, value["global_step"])
    return model, optimizer, value


def export_generator(checkpoint, out):
    value = validate_payload(torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True))
    if value["global_step"] != TOTAL_UPDATES:
        raise ValueError("Only final recipe endpoints export")
    result = dict(schema=INFERENCE_SCHEMA, experiment=EXPERIMENT,
        recipe=value["recipe"], segment=value["recipe"], global_step=TOTAL_UPDATES,
        model_config=value["model_config"], generator_state=value["generator_state"],
        generator_state_sha256=value["generator_state_sha256"], protocol=value["protocol"],
        protocol_sha256=value["protocol_sha256"], identities=value["identities"], lineage=value["lineage"],
        training_checkpoint_sha256=digest(checkpoint),
        inference_contract=dict(readout_mode="none", teacher=False, aligner=False, capture=False,
                                forward="BF16", parameters="FP32", integration="FP32"))
    return _write_checkpoint(out, result, immutable=True)


def load_generator(path, device="cpu", expected_sha256=None):
    runtime.configure_stable_runtime()
    if expected_sha256 is not None and digest(path) != expected_sha256:
        raise ValueError("Generator file hash differs")
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if value.get("schema") != INFERENCE_SCHEMA or value.get("experiment") != EXPERIMENT:
        raise ValueError("Not a recipe generator export")
    p = value["protocol"]
    if value["model_config"] != MODEL_CONFIG or value["global_step"] != TOTAL_UPDATES:
        raise ValueError("Generator model/cursor differs")
    if p != protocol(value["recipe"], p["hybrid_schedule"], p["input_namespace"]) or value["protocol_sha256"] != json_hash(p):
        raise ValueError("Generator protocol differs")
    if any(key in value for key in ("optimizer", "rng", "aligner_state", "teacher_state")):
        raise ValueError("Inference export contains training state")
    expected_contract = dict(readout_mode="none", teacher=False, aligner=False, capture=False,
                             forward="BF16", parameters="FP32", integration="FP32")
    if value["inference_contract"] != expected_contract:
        raise ValueError("Generator inference contract differs")
    _check_generator(value["generator_state"])
    if state_hash(value["generator_state"]) != value["generator_state_sha256"]:
        raise ValueError("Generator tensor hash differs")
    model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode="none", trainable=False).eval()
    model.load_state_dict(value["generator_state"], strict=True)
    model.to(device).set_trainable(False)
    return model, {k: v for k, v in value.items() if k != "generator_state"}


LOG_COLUMNS = ("recipe", "attempt_id", "step", "lambda_value", "FM", "total_loss",
    "free_coordinate_denominator", "generator_gradient_norm", "generator_adamw_step",
    "ratio20_samples", "ratio40_samples", "free20_coordinates", "free40_coordinates",
    "shared_batch_sha256", "input_batch_sha256", "label_batch_sha256",
    "input_seconds", "forward_seconds", "backward_seconds", "optimizer_seconds", "update_seconds",
    "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
EXPOSURE_COLUMNS = ("recipe", "attempt_id", "step", "sample_index", "uid", "task_id",
    "origin_sample_index", "N", "K", "ratio", "alpha", "t", "lambda_value", "free_coordinates",
    "shared_input_sha256", "input_bytehash", "label_bytehash", "ot_cache_key", "ot_cache_status")


def train_recipe(recipe, official, data_manifest, stream_root, out, attempt_id,
                 registration, hybrid_schedule=None, resume=None, device="cuda"):
    """Execute one already-authorized trajectory; never dispatch or self-retry."""
    from ..data.recipe_factorial import RecipeFactorialStream
    hybrid_schedule = resolved_schedule(recipe, hybrid_schedule)
    if not attempt_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in attempt_id):
        raise ValueError("Safe explicit attempt_id required")
    if not str(device).startswith("cuda"):
        raise ValueError("Formal training requires an owned CUDA worker")
    torch.set_num_threads(2)
    runtime.configure_stable_runtime()
    out, data_manifest, stream_root = Path(out).resolve(), Path(data_manifest).resolve(), Path(stream_root).resolve()
    folder = out / "training" / recipe
    attempt = folder / "attempts" / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    state = dict(status="LOADING", recipe=recipe, segment=recipe, attempt_id=attempt_id,
        pid=os.getpid(), step=0, stop_step=TOTAL_UPDATES, checkpoint=None,
        physical_optimizer_updates_this_attempt=0, updates_this_attempt=0,
        torch_peak_scope="reset before each effective update; 8 microbatch F/B plus optimizer; excludes CUDA context/desktop",
        alignment_active=False, teacher_forwards=0, head_forwards=0)
    status_path, events = folder / "status.json", attempt / "events.jsonl"
    atomic_json(status_path, state)
    ledger = CallLedger(attempt / "calls.jsonl", recipe, attempt_id)
    started, step, first_step = time.perf_counter(), 0, 0
    signal_state = {"number": None}
    def interrupt(signum, frame):
        signal_state["number"] = int(signum)
    handlers = {s: signal.signal(s, interrupt) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        stream = RecipeFactorialStream(data_manifest.parent, stream_root,
            recipe=recipe, hybrid_schedule=hybrid_schedule, require_cache=True)
        namespace = stream.plan["namespace"]
        check_authorization(registration, recipe, hybrid_schedule, namespace, out)
        scientific = protocol(recipe, hybrid_schedule, namespace)
        paired_path = stream.paired_path
        paired = read_json(paired_path)
        if paired["plan_sha256"] != stream.plan_sha256:
            raise ValueError("Paired input plan identity differs")
        expected_inputs = {r["sample_index"]: r for r in paired["records"]}
        if len(expected_inputs) != TOTAL_UPDATES*8:
            raise ValueError("Expected 48000 registered sample inputs")
        identities = dict(data_manifest_sha256=digest(data_manifest), plan_sha256=stream.plan_sha256,
            paired_inputs_sha256=digest(paired_path), authorization_sha256=digest(registration),
            training_sources=source_identity(), stream_implementation_sha256=stream.implementation_sha256)
        state.update(identities=identities, protocol_sha256=json_hash(scientific),
                     namespace=namespace, hybrid_schedule=hybrid_schedule)
        baseline._seed_worker()
        if resume:
            model, optimizer, payload = load_checkpoint(resume, recipe=recipe,
                hybrid_schedule=hybrid_schedule, namespace=namespace, device=device)
            if payload["identities"] != identities:
                raise ValueError("Resume data/stream/authorization/source identities differ")
            stream.load_state_dict(payload["stream_state"])
            step = payload["global_step"]
            first_step = step
            restore_rng(payload["rng"])
            lineage = copy.deepcopy(payload["lineage"])
            append_event(events, dict(event="RESUME", source=str(resume),
                source_sha256=digest(resume), step=step, optimizer_reset=False))
            del payload
        else:
            model, optimizer, lineage = initialize_official(official, device)
            atomic_json(folder / "initialization_audit.json", lineage)
        if stream.batch_index != step:
            raise ValueError("Restored stream cursor differs")
        validate_optimizer(model, optimizer, step)
        state.update(status="TRAINING", step=step, inherited_updates=first_step, lineage=lineage)
        atomic_json(status_path, state)

        def save(current, final=False):
            append_event(events, dict(event="CHECKPOINT_BEGIN", step=current))
            receipt_calls = dict(attempt_id=attempt_id, recipe=recipe, first_step=first_step,
                completed_updates=current-first_step, successful_calls=dict(ledger.counts), events_path=str(events))
            payload = _payload(model, optimizer, recipe=recipe, hybrid_schedule=hybrid_schedule,
                namespace=namespace, step=current, stream_state=stream.state_dict(),
                identities=identities, lineage=lineage, receipt=receipt_calls)
            receipt = _write_checkpoint(folder / "latest.pt", payload)
            atomic_json(folder / "latest.json", receipt)
            if final:
                immutable = out / "checkpoints" / (recipe+"_step6000.pt")
                immutable.parent.mkdir(parents=True, exist_ok=True)
                if immutable.exists():
                    raise FileExistsError("Immutable endpoint exists")
                os.link(folder / "latest.pt", immutable)
                receipt = dict(receipt, path=str(immutable.resolve()))
                atomic_json(immutable.with_suffix(".json"), receipt)
            append_event(events, dict(event="CHECKPOINT_DONE", step=current, checkpoint=receipt))
            state["checkpoint"] = receipt
            return receipt

        max_allocated, max_reserved = 0, 0
        with (attempt/"training_log.csv").open("x", newline="", encoding="utf-8") as logfile, (attempt/"exposure.csv").open("x", newline="", encoding="utf-8") as exposure:
            writer = csv.DictWriter(logfile, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            exposure_writer = csv.DictWriter(exposure, fieldnames=EXPOSURE_COLUMNS)
            exposure_writer.writeheader()
            for current in range(step+1, TOTAL_UPDATES+1):
                if signal_state["number"] is not None:
                    save(step)
                    state.update(status="INTERRUPTED_SAFE", signal=signal_state["number"])
                    break
                append_event(events, dict(event="INPUT_BEGIN", step=current))
                tick = time.perf_counter()
                samples = stream.next_effective_batch()
                input_seconds = time.perf_counter()-tick
                lam = validate_samples(samples, recipe, current, hybrid_schedule)
                for sample in samples:
                    expected = expected_inputs[sample["sample_index"]]
                    for key in ("shared_input_sha256", "input_bytehash", "label_bytehash",
                                "ot_cache_key", "free_coordinate_count"):
                        if sample[key] != expected[key]:
                            raise ValueError("Registered input differs: " + key)
                audit = stream.last_audit
                append_event(events, dict(audit, event="INPUT_DONE", step=current, lambda_value=lam))
                for sample in samples:
                    exposure_writer.writerow(dict(recipe=recipe, attempt_id=attempt_id, step=current,
                        sample_index=sample["sample_index"], origin_sample_index=sample["origin_sample_index"],
                        uid=sample["object_id"], task_id=sample["task_id"],
                        N=int(sample["y"]), K=sample["K"], ratio=sample["ratio"], alpha=sample["alpha"],
                        t=float(sample["t"]), lambda_value=lam, free_coordinates=sample["free_coordinate_count"],
                        **{k: sample[k] for k in ("shared_input_sha256", "input_bytehash", "label_bytehash", "ot_cache_key", "ot_cache_status")}))
                exposure.flush()
                os.fsync(exposure.fileno())
                append_event(events, dict(event="UPDATE_BEGIN", step=current, lambda_value=lam,
                    shared_batch_sha256=audit["shared_batch_sha256"]))
                result = update(model, optimizer, samples, ledger, recipe, current, hybrid_schedule)
                step = current
                counts = dict(ratio20_samples=sum(s["ratio"] == 0.2 for s in samples),
                    ratio40_samples=sum(s["ratio"] == 0.4 for s in samples),
                    free20_coordinates=sum(s["free_coordinate_count"] for s in samples if s["ratio"] == 0.2),
                    free40_coordinates=sum(s["free_coordinate_count"] for s in samples if s["ratio"] == 0.4))
                append_event(events, dict(event="UPDATE_DONE", step=step, generator_adamw_step=step,
                    optimizer_updates=1, lambda_value=lam, **counts, shared_batch_sha256=audit["shared_batch_sha256"]))
                append_event(attempt/"diagnostics.jsonl", dict(event="SAMPLE_LOSSES", step=step, lambda_value=lam, values=result["sample_losses"]))
                writer.writerow(dict(recipe=recipe, attempt_id=attempt_id, step=step,
                    input_seconds=input_seconds, **counts,
                    **{k: audit[k] for k in ("shared_batch_sha256", "input_batch_sha256", "label_batch_sha256")},
                    **{k: result[k] for k in LOG_COLUMNS if k in result}))
                logfile.flush()
                os.fsync(logfile.fileno())
                free, total = torch.cuda.mem_get_info()
                max_allocated = max(max_allocated, result["cuda_peak_allocated_bytes"])
                max_reserved = max(max_reserved, result["cuda_peak_reserved_bytes"])
                state.update(step=step, physical_optimizer_updates_this_attempt=ledger.counts.get("optimizer_update", 0),
                    updates_this_attempt=ledger.counts.get("optimizer_update", 0), last_FM=result["FM"],
                    lambda_value=lam, update_seconds=result["update_seconds"], cuda_free_bytes=free,
                    cuda_total_bytes=total, cuda_peak_allocated_bytes=max_allocated,
                    cuda_peak_reserved_bytes=max_reserved, cuda_current_allocated_bytes=torch.cuda.memory_allocated(),
                    cuda_current_reserved_bytes=torch.cuda.memory_reserved(), successful_calls=dict(ledger.counts),
                    wall_seconds=time.perf_counter()-started, **counts)
                if step % CHECKPOINT_INTERVAL == 0 or step == TOTAL_UPDATES:
                    save(step, final=step == TOTAL_UPDATES)
                atomic_json(status_path, state)
                atomic_json(folder/"progress.json", state)
                if step == first_step+1 or step % 25 == 0:
                    print(recipe, step, "lambda", lam, "FM", result["FM"], flush=True)
            else:
                state["status"] = "COMPLETE"
        if state["status"] == "COMPLETE":
            # A resumed already-final endpoint is supported without any update.
            if state["checkpoint"] is None:
                immutable = out/"checkpoints"/(recipe+"_step6000.pt")
                if not immutable.exists():
                    save(step, final=True)
                else:
                    state["checkpoint"] = read_json(immutable.with_suffix(".json"))
            state["inference_export"] = export_generator(state["checkpoint"]["path"],
                out/"inference"/(recipe+"_generator_step6000.pt"))
    except BaseException as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(error)
        state.update(status="OOM" if is_oom else "BLOCKED", error=repr(error),
            traceback=traceback.format_exc(), step=step,
            stream_cursor=stream.batch_index if "stream" in locals() else None,
            successful_calls=dict(ledger.counts),
            physical_optimizer_updates_this_attempt=ledger.counts.get("optimizer_update", 0),
            recoverable_under_controller_contract=is_oom)
        append_event(events, dict(event="TRAIN_FAILED", step=step, status=state["status"],
                                 error=repr(error), successful_calls=dict(ledger.counts)))
        atomic_json(attempt/"failure.json", state)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        state.update(wall_seconds=time.perf_counter()-started, step=step)
        atomic_json(attempt/"final_status.json", state)
        atomic_json(status_path, state)
        atomic_json(folder/"progress.json", state)
        if state["status"] == "COMPLETE":
            atomic_json(folder/"completed.json", state)
    return state
