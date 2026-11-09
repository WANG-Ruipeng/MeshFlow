"""Three fixed 1500-update Bench arms; no model construction on import.

Fresh AdamW in every arm. Official and COMMON are generator initializations,
not optimizer continuations. Source owner layouts are training labels only;
the frozen C-only inference layout is owned by the evaluation entrypoint.
"""
from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
from pathlib import Path
import argparse
import copy
import json
import random
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "r2/src"))
from meshflow_control import runtime
import numpy as np
import torch
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash, rng_state, restore_rng
from meshflow_control.models.native import build_model, native_execution
from meshflow_control.training import schedule40 as base
from meshflow_control.training.context_checkpoint import _cpu_copy, _check_generator
from meshflow_control.seam.model_adapter import seam_forward, corner_masked_fm_loss
from meshflow_control.data.io import digest, append_event
from r2.pilots.user_condition_finetune_v1 import trainer as historical
from r2.pilots.bench_parts_convergence_v1.worker_entry import ScopedIO
from r2.pilots.seam_pool47_couple_v1.training import resource
from r2.pilots.bench_clean_multipart_seam_v1 import data as D
from r2.pilots.bench_clean_multipart_seam_v1.training import stat, tensors

EXPERIMENT = "R2_BENCH_ADAPTATION_STRENGTH_V1"
SCHEMA = "bench_adaptation_strength_training_v1"
DEFAULT_ROOT = ROOT / "experiments/r2_bench_adaptation_strength_v1"
ARMS = ("A_COMMON_LR1E5", "B_COMMON_LR2E6", "C_OFFICIAL_LR2E6")
UPDATES = 1500
PROBE_STEPS = (0, 500, 1000, 1500)
EXPORT_STEPS = (0, 500, 1500)
FULL_STEPS = (0, 500, 1000, 1500)
_IO = None


def resolve(path):
    path = str(path)
    if os.name != "nt" and len(path) > 2 and path[1] == ":":
        path = "/mnt/" + path[0].lower() + path[2:].replace("\\", "/")
    return Path(path) if Path(path).is_absolute() else ROOT / path


def _plain_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read(path):
    return _IO.metadata_reader(_plain_read)(path) if _IO else _plain_read(path)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    (_IO.replace if _IO else os.replace)(tmp, path)


def install_io(root, arm=None):
    global _IO
    log = Path(root) / "training" / (arm or "cpu") / "io_guard.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    _IO = ScopedIO(root, os.replace, log_path=log)


def source_global(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return 0 if arm == "C_OFFICIAL_LR2E6" else 12000


def group_names(model):
    names = [name for name, _ in model.named_parameters()]
    theta = [name for name in names if name.startswith("backbone.")]
    phi = [name for name in names if name == "role_embedding" or name.startswith("context_encoder.")]
    if not theta or not phi or set(theta + phi) != set(names) or len(theta + phi) != len(names):
        raise ValueError("Unregistered generator parameter outside backbone / role / Geo")
    return [theta, phi]


def make_optimizer(model, arm):
    names = group_names(model)
    named = dict(model.named_parameters())
    theta_lr = 1e-5 if arm == "A_COMMON_LR1E5" else 2e-6
    return torch.optim.AdamW([
        dict(params=[named[n] for n in names[0]], lr=theta_lr, name="theta_backbone"),
        dict(params=[named[n] for n in names[1]], lr=1e-5, name="phi_condition"),
    ], betas=(.9, .95), eps=1e-8, weight_decay=0.)


def validate_optimizer(model, optimizer, arm, step):
    if arm not in ARMS or not 0 <= step <= UPDATES:
        raise ValueError("Arm / completed local cursor differs")
    if type(optimizer) is not torch.optim.AdamW or model.training or model.readout_mode != "none":
        raise ValueError("Expected eval-mode native model with fresh grouped AdamW")
    names = group_names(model)
    named = dict(model.named_parameters())
    if len(optimizer.param_groups) != 2:
        raise ValueError("Exactly two generator groups required")
    for index, (group, ns) in enumerate(zip(optimizer.param_groups, names)):
        lr = (1e-5 if arm == "A_COMMON_LR1E5" else 2e-6) if index == 0 else 1e-5
        if [id(p) for p in group["params"]] != [id(named[n]) for n in ns]:
            raise ValueError("Optimizer parameter identity / order changed")
        if (group["lr"], tuple(group["betas"]), group["eps"], group["weight_decay"]) != (lr, (.9, .95), 1e-8, 0.):
            raise ValueError("Registered optimizer hyperparameters changed")
        for name in ns:
            p = named[name]
            if p.dtype != torch.float32 or not p.requires_grad:
                raise ValueError("Trainable FP32 parameter required: " + name)
            state = optimizer.state.get(p, {})
            if step == 0 and not state:
                continue
            if set(state) != {"step", "exp_avg", "exp_avg_sq"} or float(state["step"]) != step:
                raise ValueError("AdamW local cursor differs: " + name)
            if any(state[k].shape != p.shape or state[k].dtype != torch.float32 for k in ("exp_avg", "exp_avg_sq")):
                raise ValueError("FP32 moment shape differs: " + name)
    return names


def checked(root):
    root = resolve(root).resolve()
    reg = read(root / "training_registration.json")
    if root != DEFAULT_ROOT.resolve() or reg["status"] != "FROZEN" or reg["experiment"] != EXPERIMENT:
        raise ValueError("Only the frozen independent experiment root is accepted")
    if reg["arms"] != list(ARMS) or (reg["updates"], reg["batch"], reg["microbatch"], reg["hybrid_lambda"]) != (1500, 8, 1, .25):
        raise ValueError("Registered arm / update / batch definition differs")
    if reg["model_config"] != MODEL_CONFIG or reg["fixed_probe_local_steps"] != list(PROBE_STEPS):
        raise ValueError("Registered model / probes differ")
    if reg["training_seed"] != 20261020:
        raise ValueError("Shared runtime seed differs")
    if not reg["training_code_hashes"]:
        raise ValueError("Frozen training code hashes required")
    for path, expected in reg["training_code_hashes"].items():
        if digest(resolve(path)) != expected:
            raise ValueError("Frozen training code changed: " + path)
    for key in ("manifest", "input_schedule", "probe_manifest"):
        if digest(resolve(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("Frozen input changed: " + key)
    manifest = read(resolve(reg["manifest"]))
    steps = read(resolve(reg["input_schedule"]))["steps"]
    if (len(manifest["rows"]), len(manifest["train"]), len(manifest["dev"])) != (316, 195, 39):
        raise ValueError("Expected existing 316 / eligible195 / DEV39 conditions")
    if len({row["parent_id"] for row in manifest["train"]}) != 58:
        raise ValueError("Expected 58 eligible TRAIN parents")
    if len(steps) != UPDATES or [int(s["step"]) for s in steps] != list(range(1, UPDATES + 1)):
        raise ValueError("Exactly original 1500-update schedule required")
    train_ids = {row["condition_id"] for row in manifest["train"]}
    for entry in steps:
        if len(entry["samples"]) != 8 or any(s["condition_id"] not in train_ids for s in entry["samples"]):
            raise ValueError("Frozen batch must contain eight eligible TRAIN examples")
    for row in manifest["rows"]:
        if digest(resolve(row["case_path"])) != row["case_sha256"]:
            raise ValueError("Frozen source case changed: " + row["condition_id"])
    if len(read(resolve(reg["probe_manifest"]))["rows"]) != 96:
        raise ValueError("Original 96-input probe required")
    return reg, manifest, steps


def require_cursor(ledger, step):
    if ledger.pending or ledger.failed:
        raise RuntimeError("Failed or pending call requires explicit recovery; never silently replay")
    expected = dict(train_forward=step * 8, train_backward=step * 8, optimizer_update=step)
    if any(ledger.attempted[k] != n or ledger.successful[k] != n for k, n in expected.items()):
        raise RuntimeError("Saved cursor trails actual completed work; rollback forbidden")
    sequence = [r["context"]["local_step"] for r in ledger.rows
                if r["event"] == "CALL_BEGIN" and r["kind"] == "optimizer_update"]
    if sequence != list(range(1, step + 1)):
        raise RuntimeError("Optimizer update sequence differs")


def initialize_generator(reg, arm):
    """CPU-only weight import; called once inside an authorized GPU worker."""
    if arm == "C_OFFICIAL_LR2E6":
        from r2.pilots.bench_control_transfer_v1.train import initialize
        model, obsolete_optimizer, audit = initialize(reg, device="cpu")
        del obsolete_optimizer
        if audit["generator_initial_sha256"] != reg["official_initial_generator_sha256"]:
            raise ValueError("Rebuilt official + original condition initialization hash differs")
    else:
        source = resolve(reg["source_checkpoint"])
        if digest(source) != reg["source_checkpoint_sha256"]:
            raise ValueError("Protected COMMON checkpoint hash differs")
        payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        if (payload["arm"], payload["global_step"], payload["step"], payload.get("EMA")) != ("COMMON", 12000, 12000, False):
            raise ValueError("Only COMMON12000 live generator accepted")
        if payload["model_config"] != MODEL_CONFIG or payload.get("head_state") is not None:
            raise ValueError("COMMON source architecture / head differs")
        if state_hash(payload["generator_state"]) != reg["source_generator_sha256"]:
            raise ValueError("COMMON live generator hash differs")
        model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode="none", trainable=True).eval()
        model.load_state_dict(payload["generator_state"], strict=True)
        audit = dict(source_path=str(source), source_sha256=reg["source_checkpoint_sha256"],
                     generator_initial_sha256=reg["source_generator_sha256"], inherited_optimizer=False)
        del payload
    _check_generator(model.state_dict())
    audit.update(arm=arm, fresh_AdamW=True, inherited_optimizer=False, inherited_rng=False,
                 source_global_step=source_global(arm), generator_parameters=sum(p.numel() for p in model.parameters()),
                 optimizer_parameter_names=group_names(model), head_calls=0, SEAM_new_generator_parameters=0)
    return model, audit


def save_payload(path, payload, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError("Immutable checkpoint already exists: " + str(path))
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    _IO.replace(tmp, path)
    receipt = dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size,
                   **{k: payload[k] for k in ("schema", "arm", "local_step", "global_step",
                      "generator_state_sha256", "optimizer_state_sha256", "rng_sha256") if k in payload})
    write(path.with_suffix(".json"), receipt)
    return receipt


def parameter_drift(current, reference, names):
    result = {}
    for label, group in zip(("theta_backbone", "phi_condition"), names):
        delta2 = ref2 = 0.
        max_abs = 0.
        count = 0
        for name in group:
            a, b = current[name].detach().cpu(), reference[name].detach().cpu()
            delta = a - b
            delta2 += float(delta.square().sum(dtype=torch.float64))
            ref2 += float(b.square().sum(dtype=torch.float64))
            max_abs = max(max_abs, float(delta.abs().max()))
            count += a.numel()
        result[label] = dict(parameters=count, delta_l2=float(np.sqrt(delta2)),
                             delta_rms=float(np.sqrt(delta2 / count)), reference_l2=float(np.sqrt(ref2)),
                             relative_l2=float(np.sqrt(delta2 / ref2)) if ref2 else None, max_abs_delta=max_abs)
    return result


def fixed_probes(root, arm, step, model, reg, cases, ledger):
    """Original supervised inputs; A owns the one shared COMMON step0 call set."""
    from r2.pilots.bench_parts_convergence_v1.train import _rng_snapshot, _assert_same_rng
    out = root / "training" / arm
    path = out / "fixed_loss" / f"step{step}.json"
    shared = root / "shared/fixed_loss/COMMON_step0.json"
    current_hash = state_hash(model.state_dict())
    if arm == "B_COMMON_LR2E6" and step == 0:
        value = read(shared)
        if value["generator_state_sha256"] != current_hash or value["probe_manifest_sha256"] != reg["probe_manifest_sha256"]:
            raise ValueError("Shared A/B step0 probe identity differs")
        receipt = dict(value, arm=arm, actual_generator_forwards=0, inherited_probe=str(shared),
                       inherited_probe_sha256=digest(shared))
        write(path, receipt)
        return receipt
    if path.exists():
        value = read(path)
        if value["generator_state_sha256"] != current_hash:
            raise ValueError("Existing fixed probe does not match current generator")
        return value
    calls = [r for r in ledger.rows if r["event"] == "CALL_BEGIN" and r["kind"] == "probe_forward"
             and r["context"]["local_step"] == step]
    if calls:
        raise RuntimeError("Incomplete fixed probe encountered: no silent forward replay")
    before = _rng_snapshot()
    started = time.perf_counter()
    rows = []
    for row in read(resolve(reg["probe_manifest"]))["rows"]:
        if digest(resolve(row["noise_npz"])) != row["noise_sha256"]:
            raise ValueError("Frozen probe noise changed")
        noise = D.arrays(row["noise_npz"])
        z, eps = noise.get("z", noise.get("noise")), noise.get("eps", noise.get("epsilon"))
        if z is None or eps is None:
            raise ValueError("Original probe noise missing")
        sample = D.sample_for_probe(cases[row["condition_id"]], z, eps, row["t"], row["permutation_seed"])
        b = tensors(sample)
        context = dict(local_step=step, global_step=source_global(arm) + step, probe_id=row["probe_id"])
        with torch.no_grad(), native_execution(model, coordinates=(b["x"], b["t"]), autocast=True):
            pred, _ = ledger.call("probe_forward", lambda: seam_forward(
                model, b["x"], b["t"], b["y"], b["valid"], b["known"], b["fixed"]).float(), context)
        error = (pred - b["target"]).reshape(int(sample["N"]), 3, 3).square().cpu().numpy()
        if not np.isfinite(error).all():
            raise FloatingPointError("Nonfinite fixed probe")
        rows.append(dict(probe_id=row["probe_id"], condition_id=row["condition_id"], parent_id=row["parent_id"],
                         split=row["group"], part=row["part_type"], t=row["t"], **stat(error, sample)))
        del pred, b
    _assert_same_rng(before)
    def aggregate(xs):
        result = {}
        for key, label in (("all_free", "ALL_free"), ("bulk", "bulk"), ("seam_tip", "seam_tip")):
            sse = sum(r[key + "_velocity_sse"] for r in xs)
            count = sum(r[key + "_scalar_count"] for r in xs)
            result[label] = dict(sse=sse, count=count, MSE=sse / count if count else None)
        return result
    summary = {}
    for split in ("TRAIN", "DEV"):
        xs = [r for r in rows if r["split"] == split]
        summary[split.lower()] = aggregate(xs)
        summary[split.lower()]["parts"] = {part: aggregate([r for r in xs if r["part"] == part])
                                          for part in sorted({r["part"] for r in xs})}
    value = dict(schema="BENCH_ADAPTATION_ORIGINAL_PROBE_V1", arm=arm, local_step=step,
                 global_step=source_global(arm) + step, generator_state_sha256=current_hash,
                 probe_manifest_sha256=reg["probe_manifest_sha256"], rows=rows, **summary,
                 actual_generator_forwards=96, head_forwards=0, backward_calls=0, RNG_unchanged=True,
                 wall_seconds=time.perf_counter() - started)
    write(path, value)
    if arm == "A_COMMON_LR1E5" and step == 0:
        if shared.exists() and read(shared) != value:
            raise ValueError("Immutable shared probe already differs")
        if not shared.exists():
            write(shared, value)
    return value


def train(root, arm, *, resume=False, target=UPDATES):
    root = resolve(root).resolve()
    install_io(root, arm)
    reg, manifest, steps = checked(root)
    reg_sha = digest(root / "training_registration.json")
    go = read(root / "training_dispatch_go.json")
    if not go.get("authorized") or go.get("training_registration_sha256") != reg_sha or arm not in go.get("arms", []):
        raise ValueError("Parent per-arm GPU dispatch release missing")
    if arm not in ARMS or not 0 < target <= UPDATES:
        raise ValueError("Unknown arm / target exceeds fixed budget")
    out = root / "training" / arm
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    previous = read(status_path) if status_path.exists() else None
    if previous and previous["status"] == "COMPLETE":
        if digest(resolve(previous["checkpoint"]["path"])) != previous["checkpoint"]["sha256"]:
            raise ValueError("Completed checkpoint changed")
        return previous
    if previous and (not resume or previous["status"] not in ("INTERRUPTED_SAFE", "SAFE_SEGMENT")):
        raise RuntimeError("Unresolved previous attempt; no silent retry")
    if resume and previous is None:
        raise ValueError("No safe exact-cursor state to resume")
    step = int(previous["local_step"]) if previous else 0
    if step >= target:
        raise ValueError("Target must advance the exact saved cursor")
    attempt = "attempt_" + str(time.time_ns())
    adir = out / "attempts" / attempt
    adir.mkdir(parents=True, exist_ok=False)
    limits = dict(train_forward=12000, train_backward=12000, optimizer_update=1500,
                  probe_forward=288 if arm == "B_COMMON_LR2E6" else 384)
    ledger = historical.CallLedger(out / "calls.jsonl", limits, arm, attempt)
    require_cursor(ledger, step)
    state = dict(status="LOADING", arm=arm, local_step=step, global_step=source_global(arm) + step,
                 target_local_step=target, registration_sha256=reg_sha, pid=os.getpid(), attempt=attempt,
                 head_calls=0, fresh_optimizer=True)
    if previous:
        state["checkpoint"] = previous["checkpoint"]
    initial_resource = resource()
    write(adir / "resources_dispatch.json", initial_resource)
    if initial_resource["free_mib"] is None or initial_resource["free_mib"] < 2048:
        raise RuntimeError("RESOURCE_WAIT: no model loaded")
    if arm == "B_COMMON_LR2E6" and step == 0 and not (root / "shared/fixed_loss/COMMON_step0.json").exists():
        raise RuntimeError("DEPENDENCY_WAIT: A must publish the shared step0 probe before B")
    observer = historical.GPUObserver("bench_adaptation_formal_training_" + arm, interval=2.)
    started = time.perf_counter()
    stop = {"signal": None}
    owned = False
    def handler(sig, unused):
        stop["signal"] = sig
    handlers = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with historical.exclusive_writer(out), observer:
            owned = True
            write(status_path, state)
            torch.set_num_threads(2)
            runtime.configure_stable_runtime()
            if previous:
                path = resolve(previous["checkpoint"]["path"])
                if digest(path) != previous["checkpoint"]["sha256"]:
                    raise ValueError("Resume checkpoint SHA differs")
                payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if (payload["schema"], payload["arm"], payload["local_step"], payload["registration_sha256"],
                    payload.get("update_in_progress")) != (SCHEMA, arm, step, reg_sha, False):
                    raise ValueError("Resume completed-state identity differs")
                for key, fn, hashkey in (("generator_state", state_hash, "generator_state_sha256"),
                                         ("optimizer", base.tree_hash, "optimizer_state_sha256"),
                                         ("rng", base.tree_hash, "rng_sha256")):
                    if fn(payload[key]) != payload[hashkey]:
                        raise ValueError("Resume internal hash differs: " + key)
                model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode="none", trainable=True).eval()
                model.load_state_dict(payload["generator_state"], strict=True)
                model.to("cuda")
                optimizer = make_optimizer(model, arm)
                if payload["optimizer_parameter_names"] != group_names(model):
                    raise ValueError("Resume parameter grouping differs")
                optimizer.load_state_dict(payload["optimizer"])
                restore_rng(payload["rng"])
                if base.tree_hash(rng_state()) != payload["rng_sha256"]:
                    raise ValueError("Resume RNG is not exact")
                audit = payload["initialization"]
                del payload
            else:
                model, audit = initialize_generator(reg, arm)
                model.to("cuda")
                optimizer = make_optimizer(model, arm)
                # Equal post-initialization RNG. Every data noise/time/permutation
                # is additionally seeded by the identical frozen sample row.
                random.seed(reg["training_seed"])
                np.random.seed(reg["training_seed"])
                torch.manual_seed(reg["training_seed"])
                torch.cuda.manual_seed_all(reg["training_seed"])
                audit["fresh_runtime_rng_sha256"] = base.tree_hash(rng_state())
            validate_optimizer(model, optimizer, arm, step)
            write(adir / "initialization.json", audit)
            cases = {row["condition_id"]: D.load_case(row) for row in manifest["rows"]}
            reference_payload = None
            if previous:
                reference_path = out / "checkpoint_step0.pt"
                receipt = read(reference_path.with_suffix(".json"))
                if digest(reference_path) != receipt["sha256"]:
                    raise ValueError("Arm step0 reference changed")
                reference_payload = torch.load(reference_path, map_location="cpu", weights_only=True, mmap=True)
                reference = reference_payload["generator_state"]
            else:
                reference = _cpu_copy(model.state_dict())
            if state_hash(reference) != audit["generator_initial_sha256"]:
                raise ValueError("Parameter drift reference is not the arm starting generator")

            def checkpoint_save():
                require_cursor(ledger, step)
                names = validate_optimizer(model, optimizer, arm, step)
                g = _cpu_copy(model.state_dict())
                _check_generator(g)
                o, rng = _cpu_copy(optimizer.state_dict()), _cpu_copy(rng_state())
                payload = dict(schema=SCHEMA, experiment=EXPERIMENT, arm=arm, local_step=step,
                               global_step=source_global(arm) + step, source_global_step=source_global(arm),
                               seam_total_step=step, EMA=False, update_in_progress=False,
                               generator_state=g, generator_state_sha256=state_hash(g),
                               optimizer=o, optimizer_state_sha256=base.tree_hash(o), optimizer_parameter_names=names,
                               rng=rng, rng_sha256=base.tree_hash(rng), initialization=audit,
                               model_config=copy.deepcopy(MODEL_CONFIG), registration_sha256=reg_sha,
                               execution=ledger.snapshot(), stream_state=dict(batch_index=step, next_sample_index=8 * step),
                               parameter_drift=parameter_drift(g, reference, names), head_calls=0)
                dest = out / (f"checkpoint_step{step}.pt" if step in FULL_STEPS else "rolling.pt")
                if dest.exists() and step in FULL_STEPS:
                    old = read(dest.with_suffix(".json"))
                    if old["generator_state_sha256"] != payload["generator_state_sha256"] or old["optimizer_state_sha256"] != payload["optimizer_state_sha256"]:
                        raise ValueError("Existing immutable checkpoint differs")
                    state["checkpoint"] = old
                else:
                    state["checkpoint"] = save_payload(dest, payload, replace=step not in FULL_STEPS)
                    append_event(out / "checkpoint_history.jsonl", state["checkpoint"])
                write(out / "parameter_drift" / f"step{step}.json", dict(local_step=step, **payload["parameter_drift"]))
                if step in EXPORT_STEPS:
                    export_path = root / "inference" / f"{arm}_S{step}_generator.pt"
                    export = {k: payload[k] for k in ("experiment", "arm", "local_step", "global_step", "seam_total_step",
                                                     "generator_state", "generator_state_sha256", "model_config", "EMA")}
                    export.update(schema="bench_adaptation_strength_generator_v1",
                                  state_sha256=payload["generator_state_sha256"],
                                  training_checkpoint_sha256=state["checkpoint"]["sha256"])
                    if not export_path.exists():
                        state["inference_export"] = save_payload(export_path, export)
                    else:
                        state["inference_export"] = read(export_path.with_suffix(".json"))
                        if state["inference_export"]["generator_state_sha256"] != payload["generator_state_sha256"]:
                            raise ValueError("Immutable inference export differs")
                write(status_path, state)

            state["status"] = "TRAINING"
            if not previous:
                checkpoint_save()
            if step in PROBE_STEPS:
                fixed_probes(root, arm, step, model, reg, cases, ledger)
            write(status_path, state)
            resume_step = step
            for entry in steps[step:target]:
                if stop["signal"] is not None:
                    checkpoint_save()
                    state.update(status="INTERRUPTED_SAFE", signal=stop["signal"])
                    break
                current = int(entry["step"])
                tick = time.perf_counter()
                if current == resume_step + 1 or current % 25 == 1:
                    rr = resource()
                    append_event(out / "resource_checks.jsonl", dict(local_step_before=step, **rr))
                    if rr["free_mib"] is None or rr["free_mib"] < 2048:
                        checkpoint_save()
                        state.update(status="INTERRUPTED_SAFE", reason="measured_vram_headroom_below_2048MiB_or_unknown")
                        break
                samples = [D.sample_for_training(cases[row["condition_id"]], row) for row in entry["samples"]]
                denominator = sum(int(s["free_coordinate_count"]) for s in samples)
                input_seconds = time.perf_counter() - tick
                batch_start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                optimizer.zero_grad(set_to_none=True)
                fm, rows = 0., []
                for mi, (sample, row) in enumerate(zip(samples, entry["samples"])):
                    context = dict(local_step=current, global_step=source_global(arm) + current,
                                   microbatch=mi, condition_id=row["condition_id"])
                    b = tensors(sample)
                    with native_execution(model, coordinates=(b["x"], b["t"]), autocast=True):
                        pred, _ = ledger.call("train_forward", lambda: seam_forward(
                            model, b["x"], b["t"], b["y"], b["valid"], b["known"], b["fixed"]).float(), context)
                        loss = corner_masked_fm_loss(pred, b["target"], b["valid"], b["fixed"], denominator=denominator)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("Nonfinite FM")
                        ledger.call("train_backward", loss.backward, context)
                    err = (pred.detach() - b["target"]).reshape(int(sample["N"]), 3, 3).square().cpu().numpy()
                    statistics = stat(err, sample)
                    if not np.isclose(statistics["all_free_velocity_sse"] / denominator, float(loss.detach()), rtol=4e-6, atol=1e-7):
                        raise ValueError("FM SSE / scalar denominator accounting differs")
                    fm += float(loss.detach())
                    tm = float(sample["t"])
                    time_bin = min(4, int(tm * 5))
                    append_event(out / "exposure.jsonl", {**row, **context, "t": tm,
                        "time_bin": f"{time_bin/5:.1f}-{(time_bin+1)/5:.1f}", "N": int(sample["N"]), "K": int(sample["K"]),
                        "FM_contribution": float(loss.detach()), **statistics})
                    rows.append(statistics)
                    del pred, loss, b
                if any(p.grad is None for p in model.parameters()):
                    raise RuntimeError("Unused generator parameter")
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                ledger.call("optimizer_update", optimizer.step,
                            dict(local_step=current, global_step=source_global(arm) + current))
                step = current
                validate_optimizer(model, optimizer, arm, step)
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError("Nonfinite updated generator")
                sums = {key: sum(row[key] for row in rows) for key in rows[0]}
                result = dict(local_step=step, global_step=source_global(arm) + step, optimizer_local_step=step,
                              FM=fm, free_coordinate_denominator=denominator, **sums,
                              generator_gradient_norm=gradient_norm, input_seconds=input_seconds,
                              update_seconds=time.perf_counter() - batch_start,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                              peak_scope="one real formal batch including backward and both-group AdamW; not process VRAM",
                              theta_lr=optimizer.param_groups[0]["lr"], phi_lr=optimizer.param_groups[1]["lr"], head_calls=0)
                append_event(out / "training_log.jsonl", result)
                state.update(local_step=step, global_step=source_global(arm) + step, last=result,
                             elapsed_seconds=time.perf_counter() - started)
                if step % 250 == 0:
                    checkpoint_save()
                if step in PROBE_STEPS:
                    fixed_probes(root, arm, step, model, reg, cases, ledger)
                write(status_path, state)
                if step in (1, 5, 25) or step % 100 == 0:
                    write(out / f"resources_step{step}.json", resource())
                    print(json.dumps(dict(arm=arm, **result)), flush=True)
                if step == 1:
                    write(out / "first_formal_backward.json", dict(status="PASS", no_extra_model_forward=True, **result))
            else:
                if step % 250:
                    checkpoint_save()
                state["status"] = "COMPLETE" if step == UPDATES else "SAFE_SEGMENT"
            require_cursor(ledger, step)
            expected_probe_count = 96 * sum(s <= step for s in PROBE_STEPS) - (96 if arm == "B_COMMON_LR2E6" else 0)
            if ledger.successful["probe_forward"] != expected_probe_count:
                raise RuntimeError("Completed fixed-probe budget/cursor differs")
            write(status_path, state)
    except BaseException as exc:
        state.update(status="OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "FAILED",
                     error=repr(exc), traceback=traceback.format_exc(), no_automatic_retry=True)
        write(adir / "failure.json", state)
        raise
    finally:
        for sig, old in handlers.items():
            signal.signal(sig, old)
        state.update(local_step=step, global_step=source_global(arm) + step,
                     elapsed_seconds=time.perf_counter() - started, calls=ledger.snapshot(), resources=observer.summary())
        write(adir / "resources.json", dict(samples=observer.samples, summary=observer.summary()))
        if owned:
            write(status_path, state)
        write(adir / "final_status.json", state)
    return state


def preflight_cpu(root):
    """One official CPU rebuild, no forward/update or CUDA initialization."""
    root = resolve(root).resolve()
    install_io(root)
    reg, manifest, steps = checked(root)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU initialization preflight requires an untouched CUDA context")
    path = root / "training_audit/initialization_preflight.json"
    if path.exists():
        old = read(path)
        if old["training_registration_sha256"] != digest(root / "training_registration.json"):
            raise ValueError("Existing preflight registration differs")
        return old
    torch.set_num_threads(2)
    model, audit = initialize_generator(reg, "C_OFFICIAL_LR2E6")
    names = group_names(model)
    named = dict(model.named_parameters())
    groups = []
    for label, ns in zip(("theta_backbone", "phi_condition"), names):
        groups.append(dict(name=label, tensors=len(ns), parameters=sum(named[n].numel() for n in ns),
                           parameter_names=ns, state_sha256=state_hash({n: named[n] for n in ns})))
    optimizers = []
    for arm in ARMS:
        optimizer = make_optimizer(model, arm)
        validate_optimizer(model, optimizer, arm, 0)
        if optimizer.state:
            raise ValueError("Fresh AdamW has inherited moments")
        optimizers.append(dict(arm=arm, groups=[{k: v for k, v in group.items() if k != "params"}
                                                for group in optimizer.param_groups],
                               state_entries=0, local_optimizer_step=0))
        del optimizer
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
    value = dict(status="PASS", schema="bench_adaptation_initialization_cpu_preflight_v1",
                 training_registration_sha256=digest(root / "training_registration.json"),
                 initialization=audit, groups=groups, group_disjoint_complete=True, optimizers=optimizers,
                 train_conditions=len(manifest["train"]), train_parents=58, updates_per_arm=len(steps),
                 model_constructions=1, model_forwards=0, backward_calls=0, optimizer_updates=0,
                 CUDA_initialized=False, official_generator_initial_hash_matches=True)
    write(path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--arm", choices=ARMS)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-cpu", action="store_true")
    group.add_argument("--preflight-cpu", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target", type=int, default=UPDATES)
    args = parser.parse_args()
    if args.preflight_cpu:
        result = preflight_cpu(args.root)
    elif args.check_cpu:
        reg, manifest, steps = checked(args.root)
        result = dict(status="PASS", updates_per_arm=len(steps), train_conditions=len(manifest["train"]),
                      train_parents=58, actual_probe_forwards_total=1056, head_calls=0,
                      model_constructions=0, model_forwards=0, CUDA_initialized=torch.cuda.is_initialized())
    else:
        if not args.arm:
            parser.error("--execute requires --arm")
        result = train(args.root, args.arm, resume=args.resume, target=args.target)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
