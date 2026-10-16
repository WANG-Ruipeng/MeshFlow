"""Independent R2 main6000 ->7000 continuation; never dispatches a GPU worker.

The FM update below preserves schedule40.py:276-342 COMMON arithmetic while
removing auxiliary schedule dispatch and its historical 6000-step ceiling.
Old source files remain unchanged. Every attempted call consumes its budget.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
import sys
from pathlib import Path
PILOT_DIR = Path(__file__).resolve().parent
R2_ROOT = PILOT_DIR.parents[1]
sys.path.insert(0, str(R2_ROOT / "src"))
from meshflow_control import runtime  # CUBLAS policy precedes torch import.
import torch
import numpy as np
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import json
import signal
import subprocess
import threading
import time
import traceback

from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash, rng_state, restore_rng
from meshflow_control.data.io import atomic_json, append_event, digest, json_hash, read_json
from meshflow_control.data.dataset import collate, model_inputs
from meshflow_control.models.native import build_model, native_execution
from meshflow_control.training import recipe_factorial as historical
from meshflow_control.training import schedule40 as baseline
from meshflow_control.training.context_checkpoint import _cpu_copy, _check_generator
from meshflow_control.training.losses import masked_fm_loss

EXPERIMENT = "R2_USER_CONDITION_FINETUNE_PILOT_V1"
TRAIN_SCHEMA = "r2_user_condition_finetune_training_v1"
INFERENCE_SCHEMA = "r2_user_condition_finetune_generator_v1"
REGISTRATION_SCHEMA = "r2_user_condition_finetune_registration_v1"
ARMS = ("R2_CONT_OLD", "R2_CONT_USER")
BASE_GLOBAL_STEP, UPDATES, BATCH = 6000, 1000, 8
BASE_CHECKPOINT = Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1/checkpoints/R2_MIX_H_ALL_step6000.pt")
BASE_SHA256 = "89f49e014615eb4a41e052ed90719a2580ead46418997c268e35e09ff37a7541"
BASE_GENERATOR_SHA256 = "0e988aec51a7b9026d0a8bcf1312f2b3ff03873d09a3e75552ddab77af92563f"
BASE_OPTIMIZER_SHA256 = "900ceec2b3ef6458b0a17868a58b2bbfb39a9f61a94859acb35da8ac16c4ccd6"
BASE_RNG_SHA256 = "4edcac4143f8d44c823b7a8b6787e9330cbb5da697bf32e53ab94e087576305a"
BASE_STREAM_SHA256 = "00cbc3bf394cf1505d8b5c61d9979c72d2516ba7afcd62ded6359b1eac40d923"
BASE_NAMESPACE = "CHAIR_CONDITION_RECIPE_FACTORIAL_V1/stage1"
HISTORICAL_UPDATE_SHA256 = "702e30848602bee0aa53a8950da83e62bab338741fb65383b0ca4d04009132ac"
TRAIN_LIMITS = dict(train_forward=8000, train_backward=8000, optimizer_update=1000)
PREFLIGHT_LIMITS = dict(preflight_forward=16, preflight_backward=8)


def protocol():
    return dict(experiment=EXPERIMENT, base_global_step=6000, additional_updates=1000,
        global_steps=[6001, 7000], batch=8, microbatch=1, distinct_parents_per_batch=8,
        clean_per_batch=4, width_augmentation=[.9, 1.1], generator_lr=1e-5,
        betas=[.9, .95], weight_decay=0., generator_clip=1., hybrid_lambda=.25,
        hybrid_schedule="all", FM_denominator="whole effective batch valid free scalar count",
        model=copy.deepcopy(MODEL_CONFIG), precision="BF16 forward; FP32 parameters and integration",
        deterministic_strict=True, tf32=False, sdpa="math", readout="none",
        teacher=False, auxiliary_loss=False, routine_checkpoint_local_steps=[500, 1000],
        selection_endpoint_local_step=1000, recovery="No replay of attempted optimizer updates",
        historical_update_source_sha256=HISTORICAL_UPDATE_SHA256)


def safe_attempt_id(value):
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("Explicit safe attempt_id required")
    return value


def check_registration(path, out, arm=None):
    value = read_json(path)
    required = dict(schema=REGISTRATION_SCHEMA, experiment=EXPERIMENT,
        out=str(Path(out).resolve()), updates_per_arm=1000, effective_batch=8,
        microbatch=1, base_checkpoint_sha256=BASE_SHA256)
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError("Registration identity differs: " + key)
    if set(value.get("authorized_arms", [])) != set(ARMS) or (arm is not None and arm not in ARMS):
        raise ValueError("Exactly two registered continuation arms required")
    if value.get("preflight_limits") != dict(forward=16, backward=8, optimizer_update=0):
        raise ValueError("Preflight budget must be 16F/8B maximum")
    sources = value.get("source_hashes", {})
    code_manifest = read_json(Path(out) / "code_manifest.json")
    pilot_sources = code_manifest.get("source_hashes", {})
    for required_name in ("trainer.py", "preflight.py", "data.py"):
        if not any(Path(p).name == required_name for p in pilot_sources):
            raise ValueError("Missing frozen implementation: " + required_name)
    for path, expected in {**sources, **pilot_sources, **value.get("protected_inputs", {})}.items():
        source = Path(path)
        source = source if source.is_absolute() else PILOT_DIR / source
        if digest(source) != expected:
            raise ValueError("Registered source differs: " + str(source))
    if digest(baseline.__file__) != HISTORICAL_UPDATE_SHA256:
        raise ValueError("Historical update source changed")
    return value


@contextmanager
def exclusive_writer(folder):
    """Per-stage/arm lock only; independent arms may run concurrently."""
    import fcntl
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "writer.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class CallLedger:
    """Durable attempt-first budget, across all attempts in one owned arm."""
    def __init__(self, path, limits, arm, attempt_id, synchronize=True):
        self.path, self.limits = Path(path), dict(limits)
        self.arm, self.attempt_id, self.synchronize = arm, attempt_id, synchronize
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        self.attempted, self.successful, self.failed = Counter(), Counter(), Counter()
        self.pending = {}
        self.next_id = 1
        begun_ids = set()
        for row in self.rows:
            if row.get("arm") != arm or row["kind"] not in limits:
                raise ValueError("Call ledger ownership/category differs")
            ident, kind = row["call_id"], row["kind"]
            self.next_id = max(self.next_id, ident + 1)
            if row["event"] == "CALL_BEGIN":
                if ident in begun_ids:
                    raise ValueError("Duplicate call reservation")
                begun_ids.add(ident)
                self.pending[ident] = row
                self.attempted[kind] += 1
            elif row["event"] in ("CALL_DONE", "CALL_FAILED"):
                if ident not in self.pending or self.pending[ident]["kind"] != kind:
                    raise ValueError("Call completion without matching reservation")
                del self.pending[ident]
                (self.successful if row["event"] == "CALL_DONE" else self.failed)[kind] += 1
            else:
                raise ValueError("Unknown call ledger event")
        if any(self.attempted[k] > limit for k, limit in limits.items()):
            raise ValueError("Recorded budget already exceeds limit")

    def snapshot(self):
        return dict(attempted=dict(self.attempted), successful=dict(self.successful),
                    failed=dict(self.failed), pending_call_ids=sorted(self.pending), limits=self.limits)

    def call(self, kind, function, context):
        if self.pending:
            raise RuntimeError("Unclosed call requires explicit accounting recovery")
        if kind not in self.limits or self.attempted[kind] >= self.limits[kind]:
            raise RuntimeError("Attempt budget exhausted: " + kind)
        ident = self.next_id
        self.next_id += 1
        row = dict(event="CALL_BEGIN", call_id=ident, kind=kind, arm=self.arm,
                   attempt_id=self.attempt_id, context=context)
        append_event(self.path, row)
        self.rows.append(row)
        self.pending[ident] = row
        self.attempted[kind] += 1
        started = time.perf_counter()
        try:
            result = function()
            if self.synchronize:
                torch.cuda.synchronize()
        except BaseException as exc:
            failed = dict(row, event="CALL_FAILED", error=repr(exc), seconds=time.perf_counter() - started)
            append_event(self.path, failed)
            self.rows.append(failed)
            self.failed[kind] += 1
            del self.pending[ident]
            raise
        seconds = time.perf_counter() - started
        done = dict(row, event="CALL_DONE", seconds=seconds)
        append_event(self.path, done)
        self.rows.append(done)
        self.successful[kind] += 1
        del self.pending[ident]
        return result, seconds

    def require_exact_cursor(self, local_step):
        """Refuse stale checkpoints, ambiguous calls, and unbudgeted replay."""
        expected = dict(train_forward=8 * local_step, train_backward=8 * local_step,
                        optimizer_update=local_step)
        if self.pending or self.failed:
            raise ValueError("Failed/unclosed calls require explicit recovery; no automatic replay")
        if any(self.attempted[k] != n or self.successful[k] != n for k, n in expected.items()):
            raise ValueError("Checkpoint trails attempted/completed work; rollback is forbidden")
        optimizer_steps = [row["context"]["global_step"] for row in self.rows
                           if row["kind"] == "optimizer_update" and row["event"] == "CALL_BEGIN"]
        if optimizer_steps != list(range(6001, 6001 + local_step)):
            raise ValueError("Optimizer update sequence is not exact; replay/skip forbidden")


class GPUObserver:
    """Observe real work, including full-process memory if nvidia-smi exposes it."""
    def __init__(self, phase, interval=1.):
        self.phase, self.interval = phase, interval
        self.stop_event = threading.Event()
        self.samples = []
        self.thread = None

    def sample(self):
        record = dict(time_unix=time.time(), phase=self.phase, pid=os.getpid())
        commands = dict(gpu=["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            processes=["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
        for key, command in commands.items():
            try:
                proc = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
                record[key] = dict(returncode=proc.returncode, stdout=proc.stdout.strip(), stderr=proc.stderr.strip())
            except Exception as exc:
                record[key] = dict(error=repr(exc))
        self.samples.append(record)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop_event.set()
        self.thread.join(timeout=8)
        self.sample()

    def summary(self):
        own, total_used = [], []
        for record in self.samples:
            for line in record.get("processes", {}).get("stdout", "").splitlines():
                fields = [s.strip() for s in line.split(",")]
                if len(fields) >= 3 and fields[0] == str(os.getpid()):
                    try:
                        own.append(float(fields[-1]))
                    except ValueError:
                        pass
            for line in record.get("gpu", {}).get("stdout", "").splitlines():
                fields = [s.strip() for s in line.split(",")]
                if len(fields) >= 5:
                    try:
                        total_used.append(float(fields[2]))
                    except ValueError:
                        pass
        return dict(sample_count=len(self.samples), process_peak_mib=max(own) if own else None,
                    process_memory_status="OBSERVED" if own else "UNAVAILABLE_NVIDIA_SMI",
                    total_device_used_peak_mib=max(total_used) if total_used else None,
                    attribution="Device memory includes background processes; only matching PID is attributed to this worker",
                    sampling_interval_seconds=self.interval)


def optimizer_hyperparameters(optimizer):
    return [{key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups]


def validate_optimizer(model, optimizer, global_step, expected_hyperparameters=None):
    if type(global_step) is not int or not 6000 <= global_step <= 7000:
        raise ValueError("Continuation AdamW step outside 6000..7000")
    baseline.validate_optimizer(model, None, optimizer, "COMMON", global_step)
    if expected_hyperparameters is not None and optimizer_hyperparameters(optimizer) != expected_hyperparameters:
        raise ValueError("Saved AdamW group hyperparameters changed")


def load_base(path=BASE_CHECKPOINT, device="cuda"):
    path = Path(path).resolve()
    if digest(path) != BASE_SHA256:
        raise ValueError("Only registered MAIN R2 full step6000 may initialize this pilot")
    model, optimizer, payload = historical.load_checkpoint(path, recipe="R2_MIX_H_ALL",
        hybrid_schedule="all", namespace=BASE_NAMESPACE, device=device)
    expected = dict(global_step=6000, generator_state_sha256=BASE_GENERATOR_SHA256,
        optimizer_state_sha256=BASE_OPTIMIZER_SHA256, rng_sha256=BASE_RNG_SHA256,
        stream_state_sha256=BASE_STREAM_SHA256)
    if any(payload.get(k) != value for k, value in expected.items()):
        raise ValueError("Full main checkpoint lineage differs")
    if state_hash(model.state_dict()) != BASE_GENERATOR_SHA256 or baseline.tree_hash(optimizer.state_dict()) != BASE_OPTIMIZER_SHA256:
        raise ValueError("Loaded model/AdamW bytes differ from full base checkpoint")
    restore_rng(payload["rng"])
    if baseline.tree_hash(rng_state()) != BASE_RNG_SHA256:
        raise ValueError("Inherited RNG bytes differ")
    hyper = optimizer_hyperparameters(optimizer)
    validate_optimizer(model, optimizer, 6000, hyper)
    lineage = dict(base_checkpoint=str(path), base_checkpoint_sha256=BASE_SHA256,
        **expected, optimizer_parameter_names=baseline.optimizer_names(model, None),
        optimizer_hyperparameters=hyper, base_stream_state=copy.deepcopy(payload["stream_state"]),
        inherited_rng=True, old_stream_is_provenance_only=True, new_stream_cursor=0)
    del payload
    return model, optimizer, lineage


def validate_samples(samples, arm=None, local_step=None, effective_batch=True):
    if len(samples) != 8:
        raise ValueError("Exactly eight registered samples required")
    if effective_batch and (len({s["object_id"] for s in samples}) != 8 or sum(s["alpha"] == 1. for s in samples) != 4):
        raise ValueError("Require eight distinct parents and four clean samples")
    denominator = 0
    for sample in samples:
        n, k = int(sample["y"]), int(sample["K"])
        valid, known = sample["valid_mask"], sample["known_mask"]
        if not 128 <= n <= 256 or not 1 <= k < n or valid.dtype != bool or known.dtype != bool:
            raise ValueError("Invalid N/K or mask dtype")
        if valid.shape != (n,) or known.shape != (n,) or not valid.all() or int(known.sum()) != k:
            raise ValueError("Original full target and exact known mask required")
        if sample["free_coordinate_count"] != 9 * (n-k) or float(sample["lambda_value"]) != .25:
            raise ValueError("Free denominator or HYBRID lambda differs")
        for name in ("xt", "u", "context"):
            array = sample[name]
            if array.dtype != np.float32 or array.shape != (n, 9) or not np.isfinite(array).all():
                raise ValueError("Invalid finite FP32 sample array: " + name)
        if sample["xt"][known].tobytes() != sample["context"][known].tobytes():
            raise ValueError("Known C changed in training state")
        if local_step is not None and (int(sample["step"]) != local_step or int(sample["global_step"]) != 6000 + local_step):
            raise ValueError("Sample local/global clock differs")
        if arm == "R2_CONT_OLD" and (sample["ratio"] not in (.2, .4) or k != round(n * sample["ratio"])):
            raise ValueError("OLD must retain historical C20/C40")
        denominator += 9 * int((valid & ~known).sum())
    if effective_batch and arm == "R2_CONT_OLD":
        observed = Counter((s["ratio"], s["alpha"] == 1.) for s in samples)
        if observed != Counter({(.2, True):2, (.2, False):2, (.4, True):2, (.4, False):2}):
            raise ValueError("OLD ratio/width balance differs")
    return denominator


def update(model, optimizer, samples, ledger, arm, local_step, expected_hyperparameters):
    """Arithmetic adapted from frozen schedule40.update COMMON; no head path."""
    global_step = 6000 + local_step
    denominator = validate_samples(samples, arm, local_step)
    validate_optimizer(model, optimizer, global_step - 1, expected_hyperparameters)
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("Formal training requires controller-owned CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    result = dict(FM=0., total_loss=0., free_coordinate_denominator=denominator,
        forward_seconds=0., backward_seconds=0., optimizer_seconds=0., sample_losses=[])
    for micro, sample in enumerate(samples):
        batch = collate([sample], device)
        context = dict(local_step=local_step, global_step=global_step, microbatch=micro,
            sample_index=int(sample["sample_index"]), input_bytehash=sample["input_bytehash"], label_bytehash=sample["label_bytehash"])
        with native_execution(model, coordinates=(batch["xt"], batch["t"]), autocast=True):
            velocity, seconds = ledger.call("train_forward", lambda: model(*model_inputs(batch)).float(), context)
            result["forward_seconds"] += seconds
            fm = masked_fm_loss(velocity, batch["u"], batch["valid_mask"], batch["known_mask"], denominator=denominator)
            if not bool(torch.isfinite(fm)):
                raise FloatingPointError("Nonfinite FM loss")
            fm_value = float(fm.detach())
            _, seconds = ledger.call("train_backward", fm.backward, context)
            result["backward_seconds"] += seconds
        result["FM"] += fm_value
        result["total_loss"] += fm_value
        result["sample_losses"].append(dict(sample_index=int(sample["sample_index"]),
            FM_contribution=fm_value, batch_free_denominator=denominator))
        del velocity, fm, batch
    parameters = list(model.parameters())
    if any(p.grad is None for p in parameters):
        raise RuntimeError("Unused generator parameter")
    result["generator_gradient_norm"] = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
    _, result["optimizer_seconds"] = ledger.call("optimizer_update", optimizer.step,
        dict(local_step=local_step, global_step=global_step, expected_generator_adamw_step=global_step))
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise FloatingPointError("Nonfinite generator after update")
    validate_optimizer(model, optimizer, global_step, expected_hyperparameters)
    result.update(generator_adamw_step=global_step, update_seconds=time.perf_counter() - start,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


def validate_payload(value):
    if value.get("schema") != TRAIN_SCHEMA or value.get("experiment") != EXPERIMENT or value.get("arm") not in ARMS:
        raise ValueError("Not this independent continuation schema")
    local = value["finetune_step"]
    if type(local) is not int or not 0 <= local <= 1000 or value["base_global_step"] != 6000 or value["global_step"] != 6000 + local:
        raise ValueError("Continuation step contract differs")
    if value.get("update_in_progress") is not False or value["stream_state"]["batch_index"] != local:
        raise ValueError("Incomplete continuation checkpoint")
    if value["protocol"] != protocol() or value["protocol_sha256"] != json_hash(protocol()) or value["model_config"] != MODEL_CONFIG:
        raise ValueError("Continuation mathematical protocol differs")
    if value["lineage"]["base_checkpoint_sha256"] != BASE_SHA256:
        raise ValueError("Wrong base checkpoint lineage")
    _check_generator(value["generator_state"])
    for field, hasher, key in (("generator_state", state_hash, "generator_state_sha256"),
        ("optimizer", baseline.tree_hash, "optimizer_state_sha256"), ("rng", baseline.tree_hash, "rng_sha256"),
        ("stream_state", json_hash, "stream_state_sha256")):
        if hasher(value[field]) != value[key]:
            raise ValueError("Checkpoint hash differs: " + field)
    if value.get("aligner_state") is not None or value.get("teacher_state") is not None:
        raise ValueError("Auxiliary state forbidden")
    return value


def checkpoint_payload(model, optimizer, arm, local_step, stream_state, identities, lineage, ledger):
    validate_optimizer(model, optimizer, 6000 + local_step, lineage["optimizer_hyperparameters"])
    ledger.require_exact_cursor(local_step)
    generator, opt, rng = _cpu_copy(model.state_dict()), _cpu_copy(optimizer.state_dict()), _cpu_copy(rng_state())
    value = dict(schema=TRAIN_SCHEMA, experiment=EXPERIMENT, arm=arm, base_global_step=6000,
        finetune_step=local_step, global_step=6000 + local_step, update_in_progress=False,
        protocol=protocol(), protocol_sha256=json_hash(protocol()), model_config=copy.deepcopy(MODEL_CONFIG),
        generator_state=generator, generator_state_sha256=state_hash(generator),
        optimizer=opt, optimizer_state_sha256=baseline.tree_hash(opt),
        optimizer_parameter_names=baseline.optimizer_names(model, None), rng=rng, rng_sha256=baseline.tree_hash(rng),
        stream_state=copy.deepcopy(stream_state), stream_state_sha256=json_hash(stream_state),
        identities=copy.deepcopy(identities), lineage=copy.deepcopy(lineage), execution_receipt=ledger.snapshot(),
        aligner_state=None, teacher_state=None)
    if stream_state["batch_index"] != local_step:
        raise ValueError("Checkpoint stream has advanced beyond completed update")
    return value


def write_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Immutable checkpoint already exists: " + str(path))
    temporary = path.with_name(path.name + ".partial." + str(os.getpid()))
    if temporary.exists():
        raise FileExistsError("Unresolved checkpoint serialization")
    with temporary.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    keys = ("schema", "arm", "base_global_step", "finetune_step", "global_step", "generator_state_sha256",
            "optimizer_state_sha256", "rng_sha256", "stream_state_sha256", "protocol_sha256")
    receipt = dict(path=str(path.resolve()), sha256=digest(path), **{k:value[k] for k in keys if k in value})
    atomic_json(path.with_suffix(".json"), receipt)
    return receipt


def load_checkpoint(path, device="cuda", expected_sha256=None, expected_arm=None, expected_identities=None):
    if expected_sha256 is None or digest(path) != expected_sha256:
        raise ValueError("Explicit matching continuation file SHA256 required")
    value = validate_payload(torch.load(path, map_location="cpu", weights_only=True, mmap=True))
    if expected_arm is not None and value["arm"] != expected_arm:
        raise ValueError("Continuation arm differs")
    if expected_identities is not None and value["identities"] != expected_identities:
        raise ValueError("Continuation dataset/registration/source identity differs")
    runtime.configure_stable_runtime()
    model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode="none", trainable=True).eval()
    model.load_state_dict(value["generator_state"], strict=True)
    model.to(device)
    optimizer = baseline.make_optimizer(model, None)
    if value["optimizer_parameter_names"] != baseline.optimizer_names(model, None):
        raise ValueError("Continuation AdamW parameter name/order differs")
    optimizer.load_state_dict(value["optimizer"])
    validate_optimizer(model, optimizer, value["global_step"], value["lineage"]["optimizer_hyperparameters"])
    if state_hash(model.state_dict()) != value["generator_state_sha256"] or baseline.tree_hash(optimizer.state_dict()) != value["optimizer_state_sha256"]:
        raise ValueError("Restored continuation tensors differ")
    restore_rng(value["rng"])
    if baseline.tree_hash(rng_state()) != value["rng_sha256"]:
        raise ValueError("Restored continuation RNG differs")
    return model, optimizer, value


def export_generator(checkpoint, out, expected_sha256):
    if digest(checkpoint) != expected_sha256:
        raise ValueError("Final full checkpoint hash differs")
    value = validate_payload(torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True))
    if value["finetune_step"] != 1000 or value["global_step"] != 7000:
        raise ValueError("Only fixed local1000 endpoint exports")
    result = {key:value[key] for key in ("experiment", "arm", "base_global_step", "finetune_step", "global_step",
        "model_config", "generator_state", "generator_state_sha256", "protocol", "protocol_sha256", "identities", "lineage")}
    result.update(schema=INFERENCE_SCHEMA, training_checkpoint_sha256=expected_sha256,
        inference_contract=dict(readout_mode="none", teacher=False, aligner=False, forward="BF16", parameters="FP32", integration="FP32"))
    return write_checkpoint(out, result)


def load_generator(checkpoint, device="cuda", expected_sha256=None):
    actual = digest(checkpoint)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("Generator file hash differs")
    value = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if value.get("schema") != INFERENCE_SCHEMA or value.get("experiment") != EXPERIMENT or value.get("arm") not in ARMS:
        raise ValueError("Wrong continuation generator schema")
    if value["global_step"] != 7000 or value["finetune_step"] != 1000 or value["base_global_step"] != 6000:
        raise ValueError("Wrong fixed continuation endpoint")
    if value["protocol"] != protocol() or value["protocol_sha256"] != json_hash(protocol()) or value["model_config"] != MODEL_CONFIG:
        raise ValueError("Generator protocol differs")
    if any(k in value for k in ("optimizer", "rng", "aligner_state", "teacher_state")):
        raise ValueError("Generator export contains training tensors")
    if value["lineage"]["base_checkpoint_sha256"] != BASE_SHA256:
        raise ValueError("Generator base lineage differs")
    _check_generator(value["generator_state"])
    if state_hash(value["generator_state"]) != value["generator_state_sha256"]:
        raise ValueError("Generator tensor hash differs")
    runtime.configure_stable_runtime()
    model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode="none", trainable=False).eval()
    model.load_state_dict(value["generator_state"], strict=True)
    model.to(device).set_trainable(False)
    identity = {k:v for k,v in value.items() if k != "generator_state"}
    identity["file_sha256"] = actual
    return model, identity


def train(*, arm, out, registration, attempt_id, data_root=None, base_checkpoint=BASE_CHECKPOINT, resume=None, device="cuda"):
    if not str(device).startswith("cuda"):
        raise ValueError("Formal training requires controller-owned CUDA")
    safe_attempt_id(attempt_id)
    out = Path(out).resolve()
    registered = check_registration(registration, out, arm)
    preflight = read_json(out / "preflight/completed.json")
    if (preflight.get("status") != "PASS"
            or preflight.get("source_registration_sha256") != digest(registration)
            or preflight.get("pilot_code_manifest_sha256") != digest(out / "code_manifest.json")):
        raise ValueError("Current frozen implementation has no matching passed preflight")
    if data_root is not None and Path(data_root).resolve() != Path(registered["data_root"]).resolve():
        raise ValueError("Prepared dataset root differs from registration")
    torch.set_num_threads(2)
    runtime.configure_stable_runtime()
    try:
        from . import data as pilot_data
    except ImportError:
        import data as pilot_data
    folder = out / "training" / arm
    with exclusive_writer(folder):
        stream = pilot_data.Stream(out, arm, require_cache=True)
        identities = dict(registration_sha256=digest(registration), data=copy.deepcopy(stream.identities),
            training_sources=read_json(registration)["source_hashes"], protocol_sha256=json_hash(protocol()),
            pilot_code_manifest_sha256=digest(out / "code_manifest.json"))
        ledger = CallLedger(folder / "calls.jsonl", TRAIN_LIMITS, arm, attempt_id)
        previous = read_json(folder / "status.json") if (folder / "status.json").exists() else None
        if resume is None:
            if previous is not None or ledger.rows:
                raise ValueError("Existing arm requires explicit valid resume; no restart from base")
        else:
            if previous is None or previous["status"] != "INTERRUPTED_SAFE":
                raise ValueError("Only latest safely interrupted continuation can resume")
            receipt = previous["checkpoint"]
            if Path(resume).resolve() != Path(receipt["path"]).resolve() or digest(resume) != receipt["sha256"]:
                raise ValueError("Resume must use latest safe checkpoint")
            ledger.require_exact_cursor(receipt["finetune_step"])
        attempt = folder / "attempts" / attempt_id
        attempt.mkdir(parents=True, exist_ok=False)
        events = attempt / "events.jsonl"
        state = dict(status="LOADING", arm=arm, attempt_id=attempt_id, pid=os.getpid(),
            base_global_step=6000, finetune_step=0, global_step=6000, checkpoint=None,
            identities=identities, alignment_active=False, teacher_forwards=0,
            torch_peak_scope="reset per effective update; eight F/B plus AdamW; full-process nvidia-smi sampled separately")
        atomic_json(folder / "status.json", state)
        started, local_step, first_step = time.perf_counter(), 0, 0
        signal_state = {"number":None}
        handlers = {}
        def interrupt(signum, frame):
            signal_state["number"] = int(signum)
        observer = GPUObserver("train_" + arm)
        try:
            handlers = {sig:signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
            if resume is None:
                model, optimizer, lineage = load_base(base_checkpoint, device)
                ledger.require_exact_cursor(0)
                atomic_json(attempt / "initialization_audit.json", lineage)
            else:
                model, optimizer, value = load_checkpoint(resume, device=device,
                    expected_sha256=receipt["sha256"], expected_arm=arm, expected_identities=identities)
                local_step = first_step = value["finetune_step"]
                stream.load_state_dict(value["stream_state"])
                lineage = value["lineage"]
                state["checkpoint"] = receipt
                del value
            if stream.state_dict()["batch_index"] != local_step:
                raise ValueError("Independent stream cursor differs from local step")
            state.update(status="TRAINING", finetune_step=local_step, global_step=6000+local_step, lineage=lineage)
            atomic_json(folder / "status.json", state)
            def save(current, emergency=False):
                value = checkpoint_payload(model, optimizer, arm, current, stream.state_dict(), identities, lineage, ledger)
                name = (f"{arm}_local{current:04d}_global{6000+current}" + (f"_safe_{attempt_id}" if emergency else "") + ".pt")
                receipt = write_checkpoint(out / "checkpoints" / name, value)
                del value
                state["checkpoint"] = receipt
                atomic_json(folder / "latest.json", receipt)
                append_event(events, dict(event="CHECKPOINT_DONE", checkpoint=receipt))
                return receipt
            peak_allocated = peak_reserved = 0
            with observer:
                for current in range(local_step + 1, 1001):
                    if signal_state["number"] is not None:
                        save(local_step, emergency=True)
                        state.update(status="INTERRUPTED_SAFE", signal=signal_state["number"])
                        break
                    tick = time.perf_counter()
                    samples = stream.next_effective_batch()
                    input_seconds = time.perf_counter() - tick
                    validate_samples(samples, arm, current)
                    for sample in samples:
                        append_event(attempt / "exposure.jsonl", dict(event="EXPOSURE", local_step=current,
                            global_step=6000+current, sample_index=int(sample["sample_index"]), uid=sample["object_id"],
                            task_id=sample["task_id"], N=int(sample["y"]), K=int(sample["K"]), alpha=float(sample["alpha"]),
                            region_type=sample.get("region_type"), tier=sample.get("tier"), ratio=float(sample["ratio"]),
                            t=float(sample["t"]), free_coordinates=int(sample["free_coordinate_count"]),
                            input_bytehash=sample["input_bytehash"], label_bytehash=sample["label_bytehash"],
                            ot_cache_key=sample["ot_cache_key"]))
                    append_event(events, dict(event="UPDATE_BEGIN", local_step=current, global_step=6000+current))
                    result = update(model, optimizer, samples, ledger, arm, current, lineage["optimizer_hyperparameters"])
                    local_step = current
                    append_event(attempt / "training_log.jsonl", dict(event="UPDATE_DONE", local_step=current,
                        global_step=6000+current, input_seconds=input_seconds, **result))
                    peak_allocated = max(peak_allocated, result["cuda_peak_allocated_bytes"])
                    peak_reserved = max(peak_reserved, result["cuda_peak_reserved_bytes"])
                    free, total = torch.cuda.mem_get_info()
                    state.update(finetune_step=current, global_step=6000+current, last_FM=result["FM"],
                        update_seconds=result["update_seconds"], input_seconds=input_seconds,
                        cuda_peak_allocated_bytes=peak_allocated, cuda_peak_reserved_bytes=peak_reserved,
                        cuda_free_bytes=free, cuda_total_bytes=total, calls=ledger.snapshot(),
                        updates_this_attempt=current-first_step, wall_seconds=time.perf_counter()-started)
                    if current in (500, 1000):
                        save(current)
                    atomic_json(folder / "status.json", state)
                    atomic_json(folder / "progress.json", state)
                    if current == first_step+1 or current % 25 == 0:
                        print(arm, current, "global",6000+current,"FM",result["FM"],flush=True)
                else:
                    state["status"] = "TRAIN_UPDATES_COMPLETE"
            if state["status"] == "TRAIN_UPDATES_COMPLETE":
                final_receipt = state["checkpoint"]
                state["inference_export"] = export_generator(final_receipt["path"],
                    out / "inference" / (arm + "_generator_step7000.pt"), final_receipt["sha256"])
                state["status"] = "COMPLETE"
        except BaseException as exc:
            state.update(status="OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "BLOCKED",
                error=repr(exc), traceback=traceback.format_exc(), calls=ledger.snapshot(),
                no_automatic_retry=True, no_update_rollback=True)
            atomic_json(attempt / "failure.json", state)
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            seconds = time.perf_counter() - started
            state.update(finetune_step=local_step, global_step=6000+local_step,
                calls=ledger.snapshot(), wall_seconds=seconds,
                updates_per_second_this_attempt=(local_step-first_step)/seconds,
                physical_update_attempts=ledger.attempted["optimizer_update"],
                physical_successful_updates=ledger.successful["optimizer_update"], resource_observation=observer.summary())
            atomic_json(attempt / "resources.json", dict(summary=observer.summary(), samples=observer.samples))
            atomic_json(attempt / "final_status.json", state)
            atomic_json(folder / "status.json", state)
            atomic_json(folder / "progress.json", state)
            if state["status"] == "COMPLETE":
                atomic_json(folder / "completed.json", state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--resume", type=Path)
    options = vars(parser.parse_args())
    result = train(**options)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0 if result["status"] == "COMPLETE" else 75


if __name__ == "__main__":
    raise SystemExit(main())
