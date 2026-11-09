"""CPU-verifiable state contract for a continuous Native -> SEAM curriculum.

No model construction, forward, optimizer update or CUDA initialization occurs
on import. Recovery only accepts a fully saved cursor with exact call records.
"""
from __future__ import annotations

import copy
import numpy as np
import torch

from meshflow_control.checkpoints import state_hash
from meshflow_control.training import schedule40
from meshflow_control.training.context_checkpoint import _check_generator
from r2.pilots.bench_adaptation_strength_v1 import training as io
from .common import STAGES

SCHEMA = "bench_full_seat_curriculum_training_v1"
INFERENCE_SCHEMA = "bench_full_seat_curriculum_generator_v1"
BASE_STEP = 12000
HEAD_CONFIG = dict(features_dim=24, hidden_dim=32, parameters=833,
                   threshold_logit=0., feature_schema="C_edge24_world_chair_v1")
SOURCE_IDENTITIES = dict(
    source_checkpoint_sha256="40aecccd1838a5b04dadeac13e72b9cf7315f3721c16e180b97844a04c2f8032",
    source_generator_sha256="7f96d80f8ed4cd6f803a4d5b94fb91560497cb1ea134de69eebf5916de57e5ba",
    source_optimizer_sha256="0d55263e81a89c9ddd351db52110647d833486767f18f73e85ba42c2fe597da5",
    source_rng_sha256="824e3e71a9ca6cd66d3f042a4c5b0746b63968c8f185d377af10f044e5de66e8")
HEAD_STATE_SHA256 = "21229acce25e3e070ae3adfa580798e1534e704cc6512e006de1654d2fefb9c4"


def stage_spec(reg, stage):
    if stage not in STAGES or set(reg["stage_updates"]) != set(STAGES):
        raise ValueError("Exactly NATIVE followed by SEAM required")
    updates = reg["stage_updates"]
    if any(type(updates[s]) is not int or updates[s] <= 0 for s in STAGES):
        raise ValueError("Positive integer registered stage budgets required")
    offset = 0 if stage == "NATIVE" else updates["NATIVE"]
    full = reg["stage_full_steps"][stage]
    if (full != sorted(set(full)) or not full or full[0] != 0
            or full[-1] != updates[stage]
            or any(type(x) is not int or not 0 <= x <= updates[stage] for x in full)):
        raise ValueError("Immutable checkpoints must include stage zero and endpoint within budget")
    return dict(stage=stage, updates=updates[stage], offset=offset,
                global_start=BASE_STEP + offset, full_steps=full)


def cursor(reg, stage, step):
    spec = stage_spec(reg, stage)
    if type(step) is not int or not 0 <= step <= spec["updates"]:
        raise ValueError("Stage cursor outside registered budget")
    return dict(stage=stage, stage_step=step, local_step=step,
                curriculum_step=spec["offset"] + step,
                global_step=spec["global_start"] + step,
                native_total_step=step if stage == "NATIVE" else reg["stage_updates"]["NATIVE"],
                seam_total_step=step if stage == "SEAM" else 0)


def limits(reg, stage):
    count = stage_spec(reg, stage)["updates"]
    return dict(train_forward=count * 8, train_backward=count * 8,
                optimizer_update=count, head_forward=0, head_backward=0,
                head_optimizer_update=0, probe_forward=0)


def require_cursor(ledger, reg, stage, step):
    current = cursor(reg, stage, step)
    if ledger.pending or ledger.failed:
        raise RuntimeError("Failed or pending calls prohibit automatic replay")
    expected = dict(train_forward=step * 8, train_backward=step * 8,
                    optimizer_update=step, head_forward=0, head_backward=0,
                    head_optimizer_update=0, probe_forward=0)
    if ledger.limits != limits(reg, stage):
        raise ValueError("Call ledger budget differs from registration")
    if any(ledger.attempted[k] != n or ledger.successful[k] != n for k, n in expected.items()):
        raise RuntimeError("Saved cursor trails attempted/completed work; rollback forbidden")
    starts = [r for r in ledger.rows if r["event"] == "CALL_BEGIN"]
    for kind in ("train_forward", "train_backward", "optimizer_update"):
        rows = [r["context"] for r in starts if r["kind"] == kind]
        wanted = [(s, m) for s in range(1, step + 1)
                  for m in (range(8) if kind != "optimizer_update" else (None,))]
        actual = [(r["stage_step"], r.get("microbatch")) for r in rows]
        if actual != wanted or any(r["stage"] != stage or
                r["global_step"] != current["global_step"] - step + r["stage_step"] for r in rows):
            raise RuntimeError("Exact stage/global/microbatch call sequence differs: " + kind)


def validate_optimizer(model, optimizer, global_step):
    schedule40.validate_optimizer(model, None, optimizer, "COMMON", global_step)
    if type(optimizer) is not torch.optim.AdamW or optimizer.param_groups[0]["eps"] != 1e-8:
        raise ValueError("Original AdamW implementation/epsilon required")
    if len(optimizer.state) != len(list(model.parameters())):
        raise ValueError("Every generator parameter needs its inherited AdamW moments")
    return [[name for name, _ in model.named_parameters()]]


def validate_head_state(head_state, expected=HEAD_STATE_SHA256):
    if (not isinstance(head_state, dict) or state_hash(head_state) != expected
            or expected != HEAD_STATE_SHA256
            or sum(x.numel() for x in head_state.values()) != 833
            or any(x.device.type != "cpu" or x.dtype != torch.float32 or x.requires_grad
                   or not bool(torch.isfinite(x).all()) for x in head_state.values())):
        raise ValueError("Frozen retained OLD_S15000 head state differs")


def validate_payload(payload, reg, stage, step, registration_sha256):
    expected = cursor(reg, stage, step)
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError("Checkpoint curriculum/stage cursor differs")
    if (payload.get("schema"), payload.get("arm"), payload.get("registration_sha256"),
            payload.get("EMA"), payload.get("update_in_progress")) != (
            SCHEMA, stage, registration_sha256, False, False):
        raise ValueError("Checkpoint identity or update boundary differs")
    if payload["stream_state"] != dict(batch_index=expected["curriculum_step"],
            stage_batch_index=step, next_sample_index=8 * expected["curriculum_step"]):
        raise ValueError("Saved stream cursor differs")
    for key, hash_key, fn in (("generator_state", "generator_state_sha256", state_hash),
            ("optimizer", "optimizer_state_sha256", io.base.tree_hash),
            ("rng", "rng_sha256", io.base.tree_hash)):
        if fn(payload[key]) != payload[hash_key]:
            raise ValueError("Checkpoint state content/hash differs: " + key)
    _check_generator(payload["generator_state"])
    validate_head_state(payload["head_state"], payload["head_state_sha256"])
    if (payload.get("head_optimizer") is not None or payload.get("head_step") != 0
            or payload.get("head_calls") != 0 or payload.get("head_config") != HEAD_CONFIG):
        raise ValueError("Frozen inference head must have zero training activity")
    if payload["model_config"] != reg["model_config"]:
        raise ValueError("Model architecture differs")


def statistics(prediction, target, sample):
    error = (prediction.detach().float() - target).reshape(int(sample["N"]), 3, 3).square().cpu().numpy()
    fixed = np.asarray(sample["anchor_ids"]) >= 0
    valid = np.asarray(sample["valid_mask"], bool)
    free = valid[:, None] & ~fixed
    tip = free & (fixed.sum(1) == 2)[:, None]
    bulk = free & (fixed.sum(1) == 0)[:, None]
    result = {}
    for name, mask in (("all_free", free), ("seam_tip", tip), ("bulk", bulk)):
        result[name + "_velocity_sse"] = float(error[mask].sum(dtype=np.float64))
        result[name + "_scalar_count"] = int(mask.sum()) * 3
        result[name + "_endpoint_sse"] = float((error[mask] *
            np.float32(1. - float(sample["t"])) ** 2).sum(dtype=np.float64))
    return result


def verify_sample(sample, stage):
    """Check role invariants on an already required training sample; no extra OT."""
    n, k = int(sample["N"]), int(sample["K"])
    if not 128 <= n <= 256 or not 2 <= k < n:
        raise ValueError("Frozen N128..256/K>=2 completion contract exceeded")
    anchor = np.asarray(sample["anchor_ids"])
    fixed = anchor >= 0
    known = np.asarray(sample["known_mask"], bool)
    labels = np.asarray(sample["occupied_labels"], bool)
    active = np.asarray(sample["occupied_for_FM"], bool)
    wanted = labels if stage == "SEAM" else np.zeros_like(labels)
    if not np.array_equal(active, wanted):
        raise ValueError("NATIVE has no owner constraint; SEAM uses source owners only")
    if known.sum() != k or not np.array_equal(known, fixed.all(1)) or not np.asarray(sample["valid_mask"]).all():
        raise ValueError("Known/valid role masks differ")
    seam_count = int((fixed.sum(1) == 2).sum())
    expected = 9 * (n - k) - 6 * int(wanted.sum())
    if (seam_count != int(wanted.sum()) or int(sample["free_coordinate_count"]) != expected
            or int((~fixed).sum()) * 3 != expected or expected <= 0):
        raise ValueError("Whole-batch free scalar denominator differs")
    if stage == "NATIVE" and np.any(fixed[~known]):
        raise ValueError("Native free corners must have no owner anchors")
    return dict(source_owner_count=int(labels.sum()), active_seam_slots=seam_count,
                C_boundary_edges=len(labels), closed_C=not len(labels),
                free_coordinate_count=expected,
                policy="NATIVE_ALL_FREE_OT" if stage == "NATIVE" else "SOURCE_OWNER_SEAM_OT")
