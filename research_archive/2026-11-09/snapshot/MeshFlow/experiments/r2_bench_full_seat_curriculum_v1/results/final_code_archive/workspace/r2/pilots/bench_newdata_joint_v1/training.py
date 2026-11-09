"""One frozen Bench new-data/generator/occupancy adaptation, 1500 updates.

Imports the proven IO/checkpoint utilities, never the old experiment state machine.
No extra neural probes; original SEAM FM and exact A0 generator initialization.
The retained 833-parameter C-only head has an independent fresh optimizer.
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
from meshflow_control.seam.occupancy import OccupancyHead, occupancy_bce_loss
from meshflow_control.training.context_checkpoint import _cpu_copy, _check_generator
from meshflow_control.data.io import digest, append_event
from r2.pilots.bench_adaptation_strength_v1 import training as T
from . import data as D
from .common import OUT, ARMS, ARM, EXPERIMENT, BASELINE_ROOT, resolve

SCHEMA = "bench_newdata_joint_training_v1"
UPDATES = 1500
HEAD_CONFIG = dict(features_dim=24, hidden_dim=32, parameters=833, threshold_logit=0.)
HEAD_LR = .001
# Native checkpoint state_hash includes tensor key/dtype/shape framing. The
# historical HEAD_PROTOCOL used its own hash framing for these same tensors.
HEAD_SOURCE_STATE_SHA256 = "21229acce25e3e070ae3adfa580798e1534e704cc6512e006de1654d2fefb9c4"


def static_head_plan(manifest, steps):
    """Read existing label arrays only; no feature, OT, RNG or neural work."""
    edges, positive, negative = {}, 0, 0
    for row in manifest["train"]:
        labels = np.asarray(D.load_case(row)["occupied_labels"])
        if labels.ndim != 1 or not np.isin(labels, (False, True)).all():
            raise ValueError("Binary one-dimensional source owner labels required")
        edges[row["condition_id"]] = len(labels)
        positive += int(np.count_nonzero(labels))
        negative += int(len(labels) - np.count_nonzero(labels))
    weights = [(positive + negative) / (2 * negative), (positive + negative) / (2 * positive)] if positive and negative else [1., 1.]
    plan = []
    cumulative_calls = cumulative_updates = 0
    for entry in steps:
        nonempty = [mi for mi, row in enumerate(entry["samples"]) if edges[row["condition_id"]]]
        edge_count = sum(edges[row["condition_id"]] for row in entry["samples"])
        cumulative_calls += len(nonempty)
        cumulative_updates += int(bool(edge_count))
        plan.append(dict(step=int(entry["step"]), nonempty_micros=nonempty, edges=edge_count,
                         cumulative_calls=cumulative_calls, cumulative_updates=cumulative_updates))
    return plan, weights, dict(owner=positive, inactive=negative)


def require_cursor(ledger, step, plan, head_step):
    T.require_cursor(ledger, step)
    expected_calls = plan[step - 1]["cumulative_calls"] if step else 0
    expected_updates = plan[step - 1]["cumulative_updates"] if step else 0
    if head_step != expected_updates:
        raise RuntimeError("Head optimizer cursor differs from the frozen schedule")
    for kind, count in (("head_forward", expected_calls), ("head_backward", expected_calls),
                        ("head_optimizer_update", expected_updates)):
        if ledger.attempted[kind] != count or ledger.successful[kind] != count:
            raise RuntimeError("Head call cursor trails actual work; rollback forbidden: " + kind)
    wanted = [p["step"] for p in plan[:step] if p["edges"]]
    actual = [r["context"]["local_step"] for r in ledger.rows
              if r["event"] == "CALL_BEGIN" and r["kind"] == "head_optimizer_update"]
    if actual != wanted:
        raise RuntimeError("Head optimizer update sequence differs")


def make_head_optimizer(head):
    return torch.optim.AdamW(head.parameters(), lr=HEAD_LR, betas=(.9, .95), eps=1e-8, weight_decay=0.)


def validate_head(head, optimizer, step):
    names = [name for name, _ in head.named_parameters()]
    parameters = list(head.parameters())
    if (head.training or head.features_dim != 24 or head.hidden_dim != 32
            or sum(p.numel() for p in parameters) != 833
            or type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1):
        raise ValueError("Expected original eval-mode 833-parameter occupancy head and independent AdamW")
    group = optimizer.param_groups[0]
    if (group["lr"], tuple(group["betas"]), group["eps"], group["weight_decay"]) != (HEAD_LR, (.9, .95), 1e-8, 0.):
        raise ValueError("Head optimizer hyperparameters differ")
    if [id(p) for p in group["params"]] != [id(p) for p in parameters]:
        raise ValueError("Head optimizer parameter identity/order differs")
    for name, p in head.named_parameters():
        if p.dtype != torch.float32 or not p.requires_grad:
            raise ValueError("Trainable FP32 head parameter required: " + name)
        state = optimizer.state.get(p, {})
        if step == 0 and not state:
            continue
        if set(state) != {"step", "exp_avg", "exp_avg_sq"} or float(state["step"]) != step:
            raise ValueError("Head AdamW local cursor differs: " + name)
        if any(state[k].shape != p.shape or state[k].dtype != torch.float32 for k in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Head FP32 moment shape differs: " + name)
    return names


def save_payload(path, payload, *, replace=False):
    receipt = T.save_payload(path, payload, replace=replace)
    receipt.update({k: payload[k] for k in ("head_state_sha256", "head_optimizer_state_sha256", "head_step") if k in payload})
    T.write(Path(path).with_suffix(".json"), receipt)
    return receipt


def checked():
    """Frozen metadata and actual case/schedule checks, with zero model calls."""
    reg = T.read(OUT / "training_registration.json")
    if (reg["status"], reg["experiment"], reg["arms"], reg["updates"],
        reg["batch"], reg["microbatch"], reg["hybrid_lambda"]) != (
            "FROZEN", EXPERIMENT, list(ARMS), 1500, 8, 1, .25):
        raise ValueError("Frozen joint-adaptation definition differs")
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
    if digest(resolve(reg["head_source_checkpoint"])) != reg["head_source_sha256"]:
        raise ValueError("Frozen source head file changed")
    if reg["head_state_sha256"] != HEAD_SOURCE_STATE_SHA256:
        raise ValueError("Only the retained OLD_S15000_HEAD may initialize this adaptation")
    expected_head_optimizer = dict(lr=HEAD_LR, betas=[.9, .95], eps=1e-8, weight_decay=0, clip=1., fresh=True)
    if reg.get("head_optimizer", expected_head_optimizer) != expected_head_optimizer or reg.get("head_threshold", 0.) != 0.:
        raise ValueError("Frozen independent head optimizer/threshold differs")
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
    actual = dict(rows=len(manifest["rows"]), train=len(manifest["train"]),
                  dev=len(manifest["dev"]),
                  train_parents=len({r["parent_id"] for r in manifest["train"]}))
    if actual != reg["data_counts"] or (actual["train_parents"], actual["train"]) != (27, 92):
        raise ValueError("Actual new pool differs from frozen realized counts")
    ids = {r["condition_id"] for r in manifest["train"]}
    if len(ids) != actual["train"]:
        raise ValueError("Duplicate training condition ID")
    by_id = {r["condition_id"]: r for r in manifest["train"]}
    train_parents = {r["parent_id"] for r in manifest["train"]}
    dev_parents = {r["parent_id"] for r in manifest["dev"]}
    panel = T.read(resolve(reg["eval_manifest"]))
    eval_groups = [r.get("eval_group", {"TRAIN": "CURRENT_TRAIN", "HOLDOUT": "SAME_PARENT_HELDOUT",
                                      "DEV": "DEV"}.get(r.get("group"))) for r in panel["requests"]]
    if len(panel["requests"]) != 28 or any(group not in ("CURRENT_TRAIN", "SAME_PARENT_HELDOUT", "DEV") for group in eval_groups):
        raise ValueError("Exactly 28 frozen TRAIN/HOLDOUT/DEV requests required")
    dev_parents |= {r["parent_id"] for r, group in zip(panel["requests"], eval_groups) if group == "DEV"}
    if train_parents & dev_parents:
        raise ValueError("DEV parent enters training; same-parent HOLDOUT C remains permitted")
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
    plan, weights, counts = static_head_plan(manifest, steps)
    if not np.array_equal(np.asarray(reg["head_class_weights"], np.float64), np.asarray(weights, np.float64)):
        raise ValueError("Head class weights must be frozen from the actual full TRAIN pool")
    if any(not p["edges"] for p in plan):
        raise ValueError("Frozen four-parent batches must contain at least one open C")
    if (reg["head_forward_budget"], reg["head_optimizer_updates"]) != (plan[-1]["cumulative_calls"], plan[-1]["cumulative_updates"]):
        raise ValueError("Registered head budgets differ from the frozen schedule")
    if "head_counts" in reg and reg["head_counts"] != counts:
        raise ValueError("Registered head class counts differ")
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
    source_head = torch.load(resolve(reg["head_source_checkpoint"]), map_location="cpu", weights_only=True, mmap=True)
    if state_hash(source_head["head_state"]) != reg["head_state_sha256"]:
        raise ValueError("Source head internal state identity differs")
    head = OccupancyHead(24).eval()
    head.load_state_dict(source_head["head_state"], strict=True)
    head.to("cuda")
    hopt = make_head_optimizer(head)
    del source_head
    restore_rng(payload["rng"])
    if T.base.tree_hash(rng_state()) != reg["source_rng_sha256"]:
        raise ValueError("Imported runtime RNG differs")
    audit = dict(source_path=reg["source_checkpoint"], source_sha256=reg["source_checkpoint_sha256"],
                 generator_initial_sha256=reg["source_generator_sha256"],
                 imported_optimizer_sha256=reg["source_optimizer_sha256"],
                 imported_rng_sha256=reg["source_rng_sha256"], fresh_AdamW=True,
                 source_global_step=12000, arm=arm, new_parameters=0, head_calls=0,
                 generator_parameters=sum(p.numel() for p in model.parameters()),
                 head_source_checkpoint=reg["head_source_checkpoint"], head_source_sha256=reg["head_source_sha256"],
                 head_initial_state_sha256=state_hash(head.state_dict()), head_source_global_step=15000,
                 head_source_updates=2996, head_optimizer_fresh=True, head_local_step=0,
                 head_config=HEAD_CONFIG, head_optimizer_parameter_names=validate_head(head, hopt, 0),
                 head_independent_generator_gradient=True)
    return model, optimizer, head, hopt, audit


def train(arm, resume=False):
    if arm not in ARMS:
        raise ValueError(arm)
    T.install_io(OUT, arm)
    reg, manifest, steps = checked()
    head_plan, _, _ = static_head_plan(manifest, steps)
    reg_sha = digest(OUT / "training_registration.json")
    go = T.read(OUT / "training_dispatch_go.json")
    if not go.get("authorized") or go["training_registration_sha256"] != reg_sha or arm not in go["arms"]:
        raise RuntimeError("Parent resource-aware release missing")
    headroom_mib = max(2048., float(go.get("retained_headroom_mib", 2048.)))
    out = OUT / "training" / arm
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    previous = T.read(status_path) if status_path.exists() else None
    if previous and previous["status"] == "COMPLETE":
        if previous["registration_sha256"] != reg_sha or previous["local_step"] != UPDATES:
            raise ValueError("Completed training identity changed")
        for key in ("checkpoint", "inference_export"):
            if digest(resolve(previous[key]["path"])) != previous[key]["sha256"]:
                raise ValueError("Final checkpoint/export changed: " + key)
        return previous
    if previous and (not resume or previous["status"] != "INTERRUPTED_SAFE"):
        raise RuntimeError("Unresolved prior attempt; no silent replay")
    if resume and not previous:
        raise ValueError("No exact saved cursor")
    step = int(previous["local_step"]) if previous else 0
    head_step = int(previous["head_step"]) if previous else 0
    attempt = "attempt_" + str(time.time_ns())
    adir = out / "attempts" / attempt
    adir.mkdir(parents=True, exist_ok=False)
    ledger = T.historical.CallLedger(out / "calls.jsonl",
        dict(train_forward=12000, train_backward=12000, optimizer_update=1500,
             head_forward=reg["head_forward_budget"], head_backward=reg["head_forward_budget"],
             head_optimizer_update=reg["head_optimizer_updates"]), arm, attempt)
    require_cursor(ledger, step, head_plan, head_step)
    state = dict(status="LOADING", arm=arm, local_step=step, global_step=12000 + step,
                 target_local_step=1500, registration_sha256=reg_sha, pid=os.getpid(),
                 attempt=attempt, head_step=head_step, head_calls=ledger.successful["head_forward"], probe_calls=0)
    if previous:
        state["checkpoint"] = previous["checkpoint"]
    initial = T.resource()
    T.write(adir / "resources_dispatch.json", initial)
    if initial["free_mib"] is None or initial["free_mib"] < headroom_mib:
        raise RuntimeError("RESOURCE_WAIT before model load")
    observer = T.historical.GPUObserver("bench_newdata_joint_" + arm, interval=2.)
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
                                   ("rng", T.base.tree_hash, "rng_sha256"),
                                   ("head_state", state_hash, "head_state_sha256"),
                                   ("head_optimizer", T.base.tree_hash, "head_optimizer_state_sha256")):
                    if fn(p[key]) != p[hk]:
                        raise ValueError("Resume hash mismatch: " + key)
                model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode="none", trainable=True).eval()
                model.load_state_dict(p["generator_state"], strict=True)
                model.to("cuda")
                optimizer = T.make_optimizer(model, "A_COMMON_LR1E5")
                optimizer.load_state_dict(p["optimizer"])
                if p["optimizer_parameter_names"] != T.group_names(model):
                    raise ValueError("Resume generator parameter names/order differ")
                head = OccupancyHead(24).eval()
                head.load_state_dict(p["head_state"], strict=True)
                head.to("cuda")
                hopt = make_head_optimizer(head)
                hopt.load_state_dict(p["head_optimizer"])
                if (p["head_step"] != head_step or p["head_config"] != HEAD_CONFIG
                        or p["head_class_weights"] != reg["head_class_weights"]
                        or p["head_optimizer_parameter_names"] != [name for name, _ in head.named_parameters()]):
                    raise ValueError("Resume head cursor/config/parameter names differ")
                restore_rng(p["rng"])
                if T.base.tree_hash(rng_state()) != p["rng_sha256"]:
                    raise ValueError("Resume RNG mismatch")
                audit = p["initialization"]
                del p
            else:
                model, optimizer, head, hopt, audit = import_start(reg, arm)
            T.validate_optimizer(model, optimizer, "A_COMMON_LR1E5", step)
            validate_head(head, hopt, head_step)
            if {id(p) for p in model.parameters()} & {id(p) for p in head.parameters()}:
                raise ValueError("Generator and head must own disjoint parameters")
            T.write(adir / "initialization.json", audit)
            cases = {r["condition_id"]: D.load_case(r) for r in manifest["train"]}
            reference_payload = torch.load(resolve(reg["source_checkpoint"]), map_location="cpu", weights_only=True, mmap=True)
            reference = reference_payload["generator_state"]

            def checkpoint_save():
                require_cursor(ledger, step, head_plan, head_step)
                names = T.validate_optimizer(model, optimizer, "A_COMMON_LR1E5", step)
                head_names = validate_head(head, hopt, head_step)
                g = _cpu_copy(model.state_dict())
                _check_generator(g)
                o, rng = _cpu_copy(optimizer.state_dict()), _cpu_copy(rng_state())
                hs, ho = _cpu_copy(head.state_dict()), _cpu_copy(hopt.state_dict())
                p = dict(schema=SCHEMA, experiment=EXPERIMENT, arm=arm, local_step=step,
                    global_step=12000+step, source_global_step=12000, seam_total_step=step,
                    EMA=False, update_in_progress=False, generator_state=g,
                    generator_state_sha256=state_hash(g), optimizer=o,
                    optimizer_state_sha256=T.base.tree_hash(o), optimizer_parameter_names=names,
                    head_state=hs, head_state_sha256=state_hash(hs), head_optimizer=ho,
                    head_optimizer_state_sha256=T.base.tree_hash(ho), head_optimizer_parameter_names=head_names,
                    head_config=HEAD_CONFIG, head_step=head_step, head_class_weights=reg["head_class_weights"],
                    rng=rng, rng_sha256=T.base.tree_hash(rng), initialization=audit,
                    model_config=copy.deepcopy(MODEL_CONFIG), registration_sha256=reg_sha,
                    execution=ledger.snapshot(), stream_state=dict(batch_index=step, next_sample_index=8*step),
                    parameter_drift=T.parameter_drift(g, reference, names), head_calls=ledger.successful["head_forward"])
                immutable = step in (0, 500, 1500)
                dest = out / (f"checkpoint_step{step}.pt" if immutable else "rolling.pt")
                if dest.exists() and immutable:
                    receipt = T.read(dest.with_suffix(".json"))
                    if digest(dest) != receipt["sha256"] or any(receipt[k] != p[k] for k in (
                            "generator_state_sha256", "optimizer_state_sha256", "rng_sha256",
                            "head_state_sha256", "head_optimizer_state_sha256", "head_step")):
                        raise ValueError("Immutable checkpoint differs")
                    state["checkpoint"] = receipt
                else:
                    state["checkpoint"] = save_payload(dest, p, replace=not immutable)
                    append_event(out / "checkpoint_history.jsonl", state["checkpoint"])
                T.write(out / "parameter_drift" / f"step{step}.json", p["parameter_drift"])
                if step == 1500:
                    export = {k: p[k] for k in ("experiment", "arm", "local_step", "global_step",
                        "seam_total_step", "generator_state", "generator_state_sha256", "model_config", "EMA",
                        "head_state", "head_state_sha256", "head_config", "head_step", "head_class_weights")}
                    export.update(schema="bench_newdata_joint_generator_head_v1", state_sha256=p["generator_state_sha256"],
                                  training_checkpoint_sha256=state["checkpoint"]["sha256"], threshold_logit=0.,
                                  feature_schema="C_edge24_world_chair_v1", registration_sha256=reg_sha)
                    state["inference_export"] = save_payload(out / "generator_step1500.pt", export)
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
                    if rr["free_mib"] is None or rr["free_mib"] < headroom_mib:
                        checkpoint_save()
                        state.update(status="INTERRUPTED_SAFE", reason="VRAM_headroom_below_frozen_dispatch_margin",
                                     required_free_mib=headroom_mib)
                        break
                samples, audits = [], []
                for row in entry["samples"]:
                    s = D.sample_for_training(cases[row["condition_id"]], row)
                    if not 128 <= int(s["N"]) <= 256:
                        raise ValueError("Frozen N128..256 exceeded")
                    audit_row = dict(policy="NEWDATA_SOURCE_OWNER_ORIGINAL_SEAM_FM_INDEPENDENT_HEAD_BCE",
                        source_owner_count=int(np.asarray(s["occupied_labels"], bool).sum()),
                        C_boundary_edges=int(len(s["occupied_labels"])),
                        closed_C=not len(s["occupied_labels"]))
                    samples.append(s)
                    audits.append(audit_row)
                denominator = sum(int(s["free_coordinate_count"]) for s in samples)
                edge_denominator = sum(len(s["occupied_labels"]) for s in samples)
                nonempty_micros = [mi for mi, s in enumerate(samples) if len(s["occupied_labels"])]
                planned = head_plan[current - 1]
                if edge_denominator != planned["edges"] or nonempty_micros != planned["nonempty_micros"]:
                    raise ValueError("Actual head calls/edge denominator differ from the frozen schedule")
                input_seconds = time.perf_counter()-tick
                batch_start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                optimizer.zero_grad(set_to_none=True)
                hopt.zero_grad(set_to_none=True)
                total_loss, total_bce, rows = 0., 0., []
                head_counts = dict(TP=0, FN=0, FP=0, TN=0)
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
                    head_log = dict(head_edge_count=int(len(s["occupied_labels"])),
                                    head_forward_called=False, BCE_contribution=None)
                    if len(s["occupied_labels"]):
                        features = torch.as_tensor(np.asarray(s["edge_features"]), device="cuda", dtype=torch.float32)
                        labels = torch.as_tensor(np.asarray(s["occupied_labels"]), device="cuda", dtype=torch.float32)
                        mask = torch.ones_like(labels, dtype=torch.bool)
                        with runtime.stable_execution(head):
                            logits, _ = ledger.call("head_forward", lambda: head(features, mask).float(), context)
                            bce = occupancy_bce_loss(logits, labels, mask,
                                class_weights=reg["head_class_weights"]) * len(labels) / edge_denominator
                            if not bool(torch.isfinite(bce)):
                                raise FloatingPointError("Nonfinite independent head BCE")
                            ledger.call("head_backward", bce.backward, context)
                        contribution = float(bce.detach())
                        total_bce += contribution
                        prediction = logits.detach() >= 0
                        truth = labels.bool()
                        confusion = {key: int(value.sum()) for key, value in (
                            ("TP", prediction & truth), ("FN", ~prediction & truth),
                            ("FP", prediction & ~truth), ("TN", ~prediction & ~truth))}
                        for key, value in confusion.items():
                            head_counts[key] += value
                        head_log.update(head_forward_called=True, BCE_contribution=contribution,
                                        head_owner_count=int(truth.sum()), head_confusion=confusion)
                        del features, labels, mask, logits, bce, prediction, truth
                    append_event(out / "exposure.jsonl", {**row, **context, "t": float(s["t"]),
                        "N": int(s["N"]), "K": int(s["K"]), "loss_contribution": float(loss.detach()),
                        **st, **head_log, "mechanism": audit_row})
                    rows.append(st)
                    del pred, loss, b
                if any(p.grad is None for p in model.parameters()):
                    raise RuntimeError("Unused generator parameter")
                if edge_denominator and any(p.grad is None for p in head.parameters()):
                    raise RuntimeError("Unused occupancy head parameter")
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                hn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)) if edge_denominator else None
                ledger.call("optimizer_update", optimizer.step, dict(local_step=current, global_step=12000+current))
                if edge_denominator:
                    ledger.call("head_optimizer_update", hopt.step, dict(local_step=current, head_step=head_step+1))
                    head_step += 1
                step = current
                T.validate_optimizer(model, optimizer, "A_COMMON_LR1E5", step)
                validate_head(head, hopt, head_step)
                if any(not bool(torch.isfinite(p).all()) for p in list(model.parameters()) + list(head.parameters())):
                    raise FloatingPointError("Nonfinite updated parameter")
                sums = {key: sum(r[key] for r in rows) for key in rows[0]}
                result = dict(local_step=step, global_step=12000+step, objective=total_loss, FM=total_loss,
                    occupancy_BCE=total_bce if edge_denominator else None, head_step=head_step,
                    head_edge_denominator=edge_denominator, head_condition_count=len(nonempty_micros),
                    head_gradient_norm=hn, head_confusion=head_counts,
                    free_coordinate_denominator=denominator, **sums, generator_gradient_norm=gn,
                    input_seconds=input_seconds, update_seconds=time.perf_counter()-batch_start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_scope="one formal batch including generator/head forward/backward and both AdamW updates; allocator not process VRAM")
                append_event(out / "training_log.jsonl", result)
                state.update(local_step=step, global_step=12000+step, head_step=head_step,
                             head_calls=ledger.successful["head_forward"], last=result, elapsed_seconds=time.perf_counter()-started)
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
            require_cursor(ledger, step, head_plan, head_step)
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
                     head_step=head_step, head_calls=ledger.successful["head_forward"],
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
                              head_forward_budget=r["head_forward_budget"], head_optimizer_updates=r["head_optimizer_updates"],
                              new_neural_probes=0, CUDA_initialized=torch.cuda.is_initialized())))
    else:
        if not args.arm:
            p.error("--arm required")
        print(json.dumps(train(args.arm, args.resume), indent=2))
