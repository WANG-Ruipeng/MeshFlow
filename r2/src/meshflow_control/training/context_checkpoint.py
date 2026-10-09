"""Strict independent schemas for alignment training and generator-only export."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import torch
from .. import checkpoints as original
from ..artifacts import file_sha256
from ..config import MODEL_CONFIG, config_hash
from .context_contract import ARMS, HEAD_CONFIG, HEAD_PARAMETERS, validate_config

TRAIN_SCHEMA = "meshflow_contextual_alignment_training_v1"
INFERENCE_SCHEMA = "meshflow_contextual_alignment_generator_v1"
GENERATOR_TENSORS = 194
GENERATOR_PARAMETERS = 130401155


def _generator_factory(trainable):
    return original._make("none", trainable)


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def _check_generator(state):
    if len(state) != GENERATOR_TENSORS or sum(v.numel() for v in state.values()) != GENERATOR_PARAMETERS:
        raise ValueError("Generator capacity differs from START")
    if any(n.startswith("readouts.") or n.startswith("aligner.") for n in state):
        raise ValueError("Inference/training generator cannot contain auxiliary/readout parameters")
    if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError("Generator state must be finite FP32")


def named_parameters(model, head):
    names = {"generator." + n: p for n, p in model.named_parameters()}
    if head is not None:
        names.update({"aligner." + n: p for n, p in head.named_parameters()})
    return names


def validate_optimizer(model, head, optimizer, step):
    if not isinstance(optimizer, torch.optim.AdamW) or model.training or model.readout_mode != "none":
        raise ValueError("Restored eval-mode no-readout generator and AdamW required")
    named = named_parameters(model, head)
    if any(not p.requires_grad or p.dtype != torch.float32 for p in named.values()):
        raise ValueError("All generator/head parameters must be trainable FP32")
    groups = [[p for _, p in model.named_parameters()]]
    if head is not None:
        groups.append(list(head.parameters()))
        if head.training or sum(p.numel() for p in head.parameters()) != HEAD_PARAMETERS:
            raise ValueError("Head mode/capacity differs")
    if len(optimizer.param_groups) != len(groups):
        raise ValueError("Optimizer group count differs")
    for index, (g, expected) in enumerate(zip(optimizer.param_groups, groups)):
        if [id(p) for p in g["params"]] != [id(p) for p in expected]:
            raise ValueError("Optimizer parameter name/order mapping differs")
        if (g["lr"], tuple(g["betas"]), g["weight_decay"]) != (1e-5 if index == 0 else 1e-4, (.9, .95), 0.):
            raise ValueError("Optimizer recipe differs")
    for n, p in named.items():
        s = optimizer.state[p]
        expected = step if n.startswith("aligner.") else 5000 + step
        if set(s) != {"step", "exp_avg", "exp_avg_sq"} or float(s["step"]) != expected:
            raise ValueError("AdamW cursor differs: " + n)
        if any(s[k].shape != p.shape or s[k].dtype != torch.float32 for k in ("exp_avg", "exp_avg_sq")):
            raise ValueError("AdamW moments differ: " + n)
    return named


def load_start(path, config, device="cpu"):
    config = validate_config(config)
    model, optimizer, audit = original.load_start(path, mode="none", device=device, trainable=True)
    from .context_alignment import AlignmentHead
    head = None if config["arm"] == "A" else AlignmentHead().to(device).eval()
    if head is not None:
        optimizer.add_param_group(dict(params=list(head.parameters()), lr=1e-4, betas=(.9, .95), weight_decay=0.))
        for p in head.parameters():
            optimizer.state[p] = dict(step=torch.tensor(0.), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
    validate_optimizer(model, head, optimizer, 0)
    return model, head, optimizer, dict(audit, alignment_arm=config["arm"],
        aligner_parameter_count=0 if head is None else HEAD_PARAMETERS,
        aligner_initial_sha256=None if head is None else original.state_hash(head.state_dict()))


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Checkpoint output already exists: " + str(path))
    temp = path.with_suffix(path.suffix + ".partial")
    if temp.exists():
        raise FileExistsError("Unresolved checkpoint write: " + str(temp))
    with temp.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return dict(path=str(path.resolve()), sha256=file_sha256(path),
        pilot_step=payload["pilot_step"], generator_state_sha256=payload["generator_state_sha256"],
        schema=payload["schema"])


def save(path, model, head, optimizer, *, step, stream_state, config, data_manifest_hash, teacher_manifest_hash, extra=None):
    config = validate_config(config)
    if type(step) is not int or not 0 <= step <= 1000 or stream_state["batch_index"] != step:
        raise ValueError("Checkpoint must cover a whole completed update")
    if (head is None) != (config["arm"] == "A"):
        raise ValueError("Arm/head mismatch")
    named = validate_optimizer(model, head, optimizer, step)
    generator = _cpu_copy(model.state_dict())
    _check_generator(generator)
    aligner = None if head is None else _cpu_copy(head.state_dict())
    reverse = {id(p): n for n, p in named.items()}
    payload = dict(schema=TRAIN_SCHEMA, generator_state=generator,
        generator_state_sha256=original.state_hash(generator), aligner_state=aligner,
        aligner_state_sha256=None if aligner is None else original.state_hash(aligner),
        model_config=copy.deepcopy(MODEL_CONFIG), alignment_config=copy.deepcopy(HEAD_CONFIG),
        config=config, config_sha256=config_hash(config), arm=config["arm"],
        pilot_step=step, conditional_total_step=5000 + step,
        source_start_sha256=original.START_FILE_SHA256, source_start_state_sha256=original.START_STATE_SHA256,
        optimizer=_cpu_copy(optimizer.state_dict()),
        optimizer_parameter_names=[[reverse[id(p)] for p in g["params"]] for g in optimizer.param_groups],
        rng=original.rng_state(), stream_state=copy.deepcopy(stream_state),
        data_manifest_sha256=data_manifest_hash, teacher_manifest_sha256=teacher_manifest_hash,
        update_in_progress=False, extra=extra or {})
    return _write(path, payload)


_TRAIN_KEYS = {"schema", "generator_state", "generator_state_sha256", "aligner_state", "aligner_state_sha256",
    "model_config", "alignment_config", "config", "config_sha256", "arm", "pilot_step", "conditional_total_step",
    "source_start_sha256", "source_start_state_sha256", "optimizer", "optimizer_parameter_names", "rng", "stream_state",
    "data_manifest_sha256", "teacher_manifest_sha256", "update_in_progress", "extra"}


def _validate_common(payload):
    if payload["source_start_sha256"] != original.START_FILE_SHA256 or payload["source_start_state_sha256"] != original.START_STATE_SHA256:
        raise ValueError("Checkpoint START provenance differs")
    if payload["model_config"] != MODEL_CONFIG or type(payload["pilot_step"]) is not int or not 0 <= payload["pilot_step"] <= 1000:
        raise ValueError("Model/step identity differs")
    if payload["conditional_total_step"] != 5000 + payload["pilot_step"] or payload["arm"] not in ARMS:
        raise ValueError("Checkpoint step/arm differs")
    _check_generator(payload["generator_state"])
    if original.state_hash(payload["generator_state"]) != payload["generator_state_sha256"]:
        raise ValueError("Generator state hash differs")


def load(path, device="cpu", trainable=True, expected_sha256=None):
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("Checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if set(payload) != _TRAIN_KEYS or payload["schema"] != TRAIN_SCHEMA or payload["update_in_progress"] is not False:
        raise ValueError("Alignment training schema differs")
    _validate_common(payload)
    config = validate_config(payload["config"])
    if config_hash(config) != payload["config_sha256"] or config["arm"] != payload["arm"] or payload["alignment_config"] != HEAD_CONFIG:
        raise ValueError("Alignment configuration differs")
    if payload["stream_state"]["batch_index"] != payload["pilot_step"]:
        raise ValueError("Checkpoint stream cursor differs")
    model = _generator_factory(trainable)
    model.load_state_dict(payload["generator_state"], strict=True)
    model.to(device).set_trainable(trainable).eval()
    from .context_alignment import AlignmentHead
    head = None if config["arm"] == "A" else AlignmentHead()
    if head is None:
        if payload["aligner_state"] is not None or payload["aligner_state_sha256"] is not None:
            raise ValueError("FM-only arm contains an aligner")
    else:
        if original.state_hash(payload["aligner_state"]) != payload["aligner_state_sha256"]:
            raise ValueError("Aligner state hash differs")
        if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in payload["aligner_state"].values()):
            raise ValueError("Aligner state must be finite FP32")
        head.load_state_dict(payload["aligner_state"], strict=True)
        head.to(device).requires_grad_(trainable).eval()
    optimizer = None
    if trainable:
        named = named_parameters(model, head)
        expected = [["generator." + n for n, _ in model.named_parameters()]]
        if head is not None:
            expected.append(["aligner." + n for n, _ in head.named_parameters()])
        if payload["optimizer_parameter_names"] != expected:
            raise ValueError("Saved optimizer parameter mapping differs")
        optimizer = torch.optim.AdamW([dict(params=[named[n] for n in names]) for names in expected],
                                      lr=1e-5, betas=(.9, .95), weight_decay=0.)
        optimizer.load_state_dict(payload["optimizer"])
        validate_optimizer(model, head, optimizer, payload["pilot_step"])
    return model, head, optimizer, payload


_INFERENCE_KEYS = {"schema", "generator_state", "generator_state_sha256", "model_config", "arm",
    "pilot_step", "conditional_total_step", "source_start_sha256", "source_start_state_sha256",
    "source_training_checkpoint_sha256", "training_config_sha256", "inference_contract"}


def export_generator(training_checkpoint, out):
    # CPU-only export does not instantiate the model or auxiliary head.
    payload = torch.load(training_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if set(payload) != _TRAIN_KEYS or payload["schema"] != TRAIN_SCHEMA or payload["update_in_progress"] is not False:
        raise ValueError("Only completed alignment checkpoints can be exported")
    _validate_common(payload)
    config = validate_config(payload["config"])
    if config_hash(config) != payload["config_sha256"] or config["arm"] != payload["arm"]:
        raise ValueError("Training configuration hash/arm differs")
    result = {k: payload[k] for k in ("generator_state", "generator_state_sha256", "model_config", "arm", "pilot_step",
        "conditional_total_step", "source_start_sha256", "source_start_state_sha256")}
    result.update(schema=INFERENCE_SCHEMA, source_training_checkpoint_sha256=file_sha256(training_checkpoint),
        training_config_sha256=payload["config_sha256"],
        inference_contract=dict(readout_mode="none", capture=False, aligner=False, teacher=False,
                                precision="LEGACY_BF16", parameters="FP32", integration="FP32"))
    return _write(out, result)


def load_generator(path, device="cpu", expected_sha256=None):
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("Inference file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if set(payload) != _INFERENCE_KEYS or payload["schema"] != INFERENCE_SCHEMA:
        raise ValueError("Not the independent alignment generator schema")
    _validate_common(payload)
    if payload["inference_contract"] != dict(readout_mode="none", capture=False, aligner=False, teacher=False,
                                             precision="LEGACY_BF16", parameters="FP32", integration="FP32"):
        raise ValueError("Inference contract differs")
    model = _generator_factory(False)
    model.load_state_dict(payload["generator_state"], strict=True)
    model.to(device).set_trainable(False).eval()
    return model, {k: v for k, v in payload.items() if k != "generator_state"}
