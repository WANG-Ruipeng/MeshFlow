"""One resumable contextual-alignment trainer; caller owns the GPU lock."""
from __future__ import annotations
from contextlib import nullcontext
import csv
import os
from pathlib import Path
import signal
import time
import traceback
import torch
from ..data.dataset import collate, model_inputs
from ..data.contextual_stream import HybridStream
from ..data.io import read_json, rows, atomic_json, append_event, digest
from ..config import config_hash
from ..checkpoints import restore_rng
from .losses import masked_fm_loss
from .trainer import validate_recovery, _call, _norm
from .context_alignment import AttentionCapture, DOMAINS, alignment_loss, select_messages, validate_config
from . import context_checkpoint as checkpoint

LOG_COLUMNS = ("arm", "step", "conditional_total_step", "FM", "raw_align", "weighted_align", "total_loss",
    "free_coordinate_denominator", "alignment_denominator", "aligned_samples",
    "prediction_norm_mean", "teacher_norm_mean", "cosine_mean",
    "generator_gradient_norm", "aligner_gradient_norm", "generator_clip_triggered", "aligner_clip_triggered",
    "generator_actual_update_norm", "aligner_actual_update_norm", "unused_generator_parameters", "unused_aligner_parameters",
    "old_adamw_step", "aligner_adamw_step", "shared_batch_sha256", "input_batch_sha256", "label_batch_sha256",
    "input_seconds", "forward_seconds", "aligner_forward_seconds", "backward_seconds", "optimizer_seconds",
    "update_seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
EXPOSURE_COLUMNS = ("arm", "step", "sample_index", "uid", "task_id", "N", "K", "ratio", "alpha", "t",
    "alignment_eligible", "free_coordinates", "shared_input_sha256", "input_bytehash", "label_bytehash", "ot_cache_key", "ot_cache_status")


def observation_parameters(model):
    # Fixed before training: coordinate projection and all layer-four attention parameters.
    return {n: p for n, p in model.named_parameters()
            if n == "backbone.x_embedder.mlp.0.weight" or n.startswith("backbone.layers.3.attn.")}


def gradient_observation(fm, weighted_aux, model, budget, context, device):
    named = observation_parameters(model)
    if not named:
        raise ValueError("Registered early parameter set missing")
    values = list(named.values())
    gf, _ = _call(budget, "gradient_observation_backward",
        lambda: torch.autograd.grad(fm, values, retain_graph=True, allow_unused=True),
        dict(context, objective="FM", parameter_names=list(named)), device)
    ga, _ = _call(budget, "gradient_observation_backward",
        lambda: torch.autograd.grad(weighted_aux, values, retain_graph=True, allow_unused=True),
        dict(context, objective="weighted_aux", parameter_names=list(named)), device)
    nf, na = _norm(gf), _norm(ga)
    dot = sum(float((f.detach().double() * a.detach().double()).sum())
              for f, a in zip(gf, ga) if f is not None and a is not None)
    result = dict(parameter_names=list(named), FM_norm=nf, weighted_aux_norm=na,
                  cosine=None if nf == 0 or na == 0 else dot / (nf * na),
                  FM_unused=[n for n, g in zip(named, gf) if g is None],
                  auxiliary_unused=[n for n, g in zip(named, ga) if g is None],
                  effect_on_accumulated_parameter_gradients="none; autograd.grad return only")
    del gf, ga
    return result


def update(model, head, optimizer, samples, targets, budget, arm, step):
    """Eight genuine F/B calls; one optimizer step across separately clipped groups."""
    from ..models.native import native_execution
    if len(samples) != 8 or len({s["object_id"] for s in samples}) != 8 or sum(s["alpha"] == 1. for s in samples) != 4:
        raise ValueError("Require eight distinct parents and four exact unaugmented samples")
    named = checkpoint.validate_optimizer(model, head, optimizer, step - 1)
    if (head is None) != (arm == "A"):
        raise ValueError("Arm/head identity differs")
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("Formal updates require an explicitly owned CUDA worker")
    denominator = sum(s["free_coordinate_count"] for s in samples)
    if denominator != 9 * sum(int((s["valid_mask"] & ~s["known_mask"]).sum()) for s in samples):
        raise ValueError("Whole-batch FM free scalar denominator differs")
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    tick = time.perf_counter()
    result = dict(FM=0., raw_align=0., weighted_align=0., total_loss=0.,
        free_coordinate_denominator=denominator, alignment_denominator=4,
        aligned_samples=0, forward_seconds=0., aligner_forward_seconds=0., backward_seconds=0.,
        optimizer_seconds=0., sample_losses=[], gradient_observations=[])
    observations = []
    observed = False
    for micro, sample in enumerate(samples):
        batch = collate([sample], device)
        ctx = dict(arm=arm, step=step, microbatch=micro, sample_index=sample["sample_index"],
                   input_bytehash=sample["input_bytehash"], label_bytehash=sample["label_bytehash"])
        eligible = head is not None and sample["alpha"] == 1.
        capture = AttentionCapture(model) if eligible else nullcontext(None)
        with capture as captured, native_execution(model, coordinates=(batch["xt"], batch["t"]), autocast=True):
            velocity, seconds = _call(budget, "train_forward", lambda: model(*model_inputs(batch)).float(), ctx, device)
            result["forward_seconds"] += seconds
            fm = masked_fm_loss(velocity, batch["u"], batch["valid_mask"], batch["known_mask"], denominator=denominator)
            raw_aux = None
            statistics = None
            if eligible:
                a = captured.get()
                tokens, selected = select_messages(a, batch["known_mask"], batch["valid_mask"], DOMAINS[arm])
                teacher = torch.as_tensor(targets.for_sample(sample), dtype=torch.float32, device=device)
                prediction, seconds = _call(budget, "aligner_forward", lambda: head(tokens[0]), ctx, device)
                result["aligner_forward_seconds"] += seconds
                raw_aux, statistics = alignment_loss(prediction, teacher)
                weighted_aux = (.1 / 4.) * raw_aux
                total = fm + weighted_aux
                observations.append(statistics)
                result["raw_align"] += float(raw_aux.detach()) / 4.
                result["weighted_align"] += float(weighted_aux.detach())
                result["aligned_samples"] += 1
                if step in (1, 100, 1000) and not observed:
                    result["gradient_observations"].append(dict(microbatch=micro,
                        **gradient_observation(fm, weighted_aux, model, budget, ctx, device)))
                    observed = True
            else:
                total = fm
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("Nonfinite combined loss")
            fm_value, total_value = float(fm.detach()), float(total.detach())
            # Nested accounting describes the same single combined autograd call.
            def backward():
                if eligible:
                    return budget.call("aligner_backward",
                        lambda: budget.call("train_backward", total.backward, context=dict(ctx, combined_backward=True)),
                        context=dict(ctx, combined_backward=True, generator_call_shared=True))
                return budget.call("train_backward", total.backward, context=ctx)
            torch.cuda.synchronize(device)
            btime = time.perf_counter()
            backward()
            torch.cuda.synchronize(device)
            result["backward_seconds"] += time.perf_counter() - btime
        result["FM"] += fm_value
        result["total_loss"] += total_value
        result["sample_losses"].append(dict(sample_index=sample["sample_index"], alpha=sample["alpha"],
            FM_contribution=fm_value, raw_align=None if raw_aux is None else float(raw_aux.detach()),
            weighted_align=0. if raw_aux is None else .025 * float(raw_aux.detach()),
            batch_free_denominator=denominator, alignment_denominator=4,
            selected_token_count=None if not eligible else int(selected.sum()), **(statistics or {})))
        if eligible:
            del a, tokens, selected, teacher, prediction, raw_aux, weighted_aux
        del velocity, fm, total, batch
    if result["aligned_samples"] != (0 if head is None else 4):
        raise AssertionError("Auxiliary denominator/eligibility differs")
    generator = dict(model.named_parameters())
    auxiliary = {} if head is None else dict(head.named_parameters())
    unused_g = [n for n, p in generator.items() if p.grad is None]
    unused_h = [n for n, p in auxiliary.items() if p.grad is None]
    if unused_g or unused_h:
        raise RuntimeError("Unused trainable parameters: " + str((unused_g, unused_h)))
    result["unused_generator_parameters"], result["unused_aligner_parameters"] = len(unused_g), len(unused_h)
    result["generator_gradient_norm"] = float(torch.nn.utils.clip_grad_norm_(list(generator.values()), 1., error_if_nonfinite=True))
    result["aligner_gradient_norm"] = 0. if head is None else float(torch.nn.utils.clip_grad_norm_(list(auxiliary.values()), 1., error_if_nonfinite=True))
    result["generator_clip_triggered"] = result["generator_gradient_norm"] > 1.
    result["aligner_clip_triggered"] = result["aligner_gradient_norm"] > 1.
    # Full generator copy only at these four preregistered updates.
    previous_g = {n: p.detach().clone() for n, p in generator.items()} if step in (1, 100, 500, 1000) else {}
    previous_h = {n: p.detach().clone() for n, p in auxiliary.items()}
    _, result["optimizer_seconds"] = _call(budget, "optimizer_update", optimizer.step,
        dict(arm=arm, step=step, conditional_total_step=5000 + step,
             generator_lr=1e-5, aligner_lr=None if head is None else 1e-4), device)
    result["generator_actual_update_norm"] = _norm([p.detach() - previous_g[n] for n, p in generator.items()]) if previous_g else None
    result["aligner_actual_update_norm"] = _norm([p.detach() - previous_h[n] for n, p in auxiliary.items()])
    if any(not bool(torch.isfinite(p).all()) for p in named.values()):
        raise FloatingPointError("Nonfinite parameter after update")
    checkpoint.validate_optimizer(model, head, optimizer, step)
    for name in ("prediction_norm", "teacher_norm", "cosine"):
        result[name + "_mean"] = sum(r[name] for r in observations) / len(observations) if observations else None
    result.update(old_adamw_step=5000 + step, aligner_adamw_step=None if head is None else step,
        update_seconds=time.perf_counter() - tick, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


def train(config, data_manifest, stream_root, start_checkpoint, out, budget, teacher_manifest,
          stop_step=1000, resume=None, device="cuda"):
    from ..teachers.siglip_targets import TargetStore
    config = validate_config(config)
    arm = config["arm"]
    if stop_step not in (500, 1000):
        raise ValueError("Only fixed 500/1000 boundaries")
    data_manifest, teacher_manifest = Path(data_manifest).resolve(), Path(teacher_manifest).resolve()
    out = Path(out).resolve()
    folder = out / "training" / arm
    folder.mkdir(parents=True, exist_ok=True)
    status_path, events_path = folder / "status.json", folder / "events.jsonl"
    if status_path.exists() and read_json(status_path).get("status") == "BLOCKED":
        raise RuntimeError("Prior incomplete attempt is BLOCKED; explicit engineering recovery required")
    stream = HybridStream(data_manifest.parent, stream_root, arm)
    paired = read_json(Path(stream_root) / "paired_inputs.json")
    if paired["plan_sha256"] != stream.plan_sha256 or paired["data_manifest_sha256"] != digest(data_manifest):
        raise ValueError("Frozen paired input identity differs")
    expected_inputs = {r["sample_index"]: r for r in paired["records"]}
    teacher_hash = digest(teacher_manifest)
    targets = None if arm == "A" else TargetStore(teacher_manifest.parent)
    if resume:
        model, head, optimizer, payload = checkpoint.load(resume, device=device, trainable=True)
        if payload["config_sha256"] != config_hash(config) or payload["data_manifest_sha256"] != digest(data_manifest) or payload["teacher_manifest_sha256"] != teacher_hash:
            raise ValueError("Resume config/data/teacher identity differs")
        stream.load_state_dict(payload["stream_state"])
        step = payload["pilot_step"]
        restore_rng(payload["rng"])
    else:
        model, head, optimizer, audit = checkpoint.load_start(start_checkpoint, config, device)
        atomic_json(folder / "start_import_audit.json", audit)
        step = 0
    if stream.batch_index != step or step > stop_step:
        raise ValueError("Stream/checkpoint cursor differs")
    validate_recovery(rows(events_path), step)
    if budget.snapshot()["updates_by_arm"].get(arm, 0) != step:
        raise ValueError("Actual optimizer ledger is not covered by checkpoint")
    checkpoint.validate_optimizer(model, head, optimizer, step)
    if step == stop_step:
        return dict(status="COMPLETE", arm=arm, step=step, actual_updates_this_call=0,
                    checkpoint=dict(path=str(resume), sha256=digest(resume)))
    stop = {"signal": None}
    def interrupt(signum, frame):
        stop["signal"] = int(signum)
    handlers = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    started, first_step = time.perf_counter(), step
    state = dict(status="TRAINING", arm=arm, label=config["label"], step=step, stop_step=stop_step,
        config_sha256=config_hash(config), data_manifest_sha256=digest(data_manifest),
        teacher_manifest_sha256=teacher_hash, plan_sha256=stream.plan_sha256, budget_before=budget.snapshot())
    def save(current):
        latest = folder / "latest.json"
        if latest.exists():
            existing = read_json(latest)
            if existing["pilot_step"] == current and digest(existing["path"]) == existing["sha256"]:
                return existing
        append_event(events_path, dict(event="CHECKPOINT_BEGIN", arm=arm, step=current))
        receipt = checkpoint.save(out / "checkpoints" / f"{arm}_step{current:04d}.pt",
            model, head, optimizer, step=current, stream_state=stream.state_dict(), config=config,
            data_manifest_hash=digest(data_manifest), teacher_manifest_hash=teacher_hash,
            extra=dict(plan_sha256=stream.plan_sha256, budget=budget.snapshot()))
        append_event(events_path, dict(event="CHECKPOINT_DONE", arm=arm, step=current, checkpoint=receipt))
        atomic_json(latest, receipt)
        return receipt
    try:
        atomic_json(status_path, state)
        with (folder / "training_log.csv").open("a", newline="", encoding="utf-8") as log, (folder / "exposure.csv").open("a", newline="", encoding="utf-8") as exposure:
            writer, ewriter = csv.DictWriter(log, fieldnames=LOG_COLUMNS), csv.DictWriter(exposure, fieldnames=EXPOSURE_COLUMNS)
            if log.tell() == 0:
                writer.writeheader()
            if exposure.tell() == 0:
                ewriter.writeheader()
            for current in range(step + 1, stop_step + 1):
                if stop["signal"] is not None:
                    state.update(status="INTERRUPTED_SAFE", signal=stop["signal"], checkpoint=save(step))
                    break
                append_event(events_path, dict(event="INPUT_BEGIN", arm=arm, step=current))
                tick = time.perf_counter()
                samples = stream.next_effective_batch()
                input_seconds = time.perf_counter() - tick
                for sample in samples:
                    for key in ("shared_input_sha256", "input_bytehash", "label_bytehash", "ot_cache_key", "free_coordinate_count"):
                        if sample[key] != expected_inputs[sample["sample_index"]][key]:
                            raise ValueError("Actual paired input differs: " + key)
                audit = stream.last_audit
                append_event(events_path, dict(event="INPUT_DONE", arm=arm, step=current, **audit))
                for s in samples:
                    ewriter.writerow(dict(arm=arm, step=current, sample_index=s["sample_index"], uid=s["object_id"],
                        task_id=s["task_id"], N=int(s["y"]), K=s["K"], ratio=s["ratio"], alpha=s["alpha"], t=float(s["t"]),
                        alignment_eligible=s["alpha"] == 1., free_coordinates=s["free_coordinate_count"],
                        **{k: s[k] for k in ("shared_input_sha256", "input_bytehash", "label_bytehash", "ot_cache_key", "ot_cache_status")}))
                exposure.flush()
                os.fsync(exposure.fileno())
                append_event(events_path, dict(event="UPDATE_BEGIN", arm=arm, step=current, shared_batch_sha256=audit["shared_batch_sha256"]))
                result = update(model, head, optimizer, samples, targets, budget, arm, current)
                step = current
                append_event(events_path, dict(event="UPDATE_DONE", arm=arm, step=step, train_forward=8, train_backward=8,
                    aligner_forward=result["aligned_samples"], aligner_backward=result["aligned_samples"],
                    optimizer_updates=1, old_adamw_step=5000+step, aligner_adamw_step=result["aligner_adamw_step"],
                    shared_batch_sha256=audit["shared_batch_sha256"]))
                append_event(folder / "diagnostics.jsonl", dict(event="TRAIN_DIAGNOSTICS", arm=arm, step=step,
                    sample_losses=result["sample_losses"], gradient_observations=result["gradient_observations"]))
                writer.writerow(dict(arm=arm, step=step, conditional_total_step=5000+step, input_seconds=input_seconds,
                    **{k: audit[k] for k in ("shared_batch_sha256", "input_batch_sha256", "label_batch_sha256")},
                    **{k: result[k] for k in LOG_COLUMNS if k in result}))
                log.flush()
                os.fsync(log.fileno())
                state.update(step=step, conditional_total_step=5000+step, last_FM=result["FM"],
                    last_raw_align=result["raw_align"], last_weighted_align=result["weighted_align"], budget=budget.snapshot())
                if step in (500, 1000):
                    state["checkpoint"] = save(step)
                if step % 10 == 0 or step in (1, 500, 1000):
                    atomic_json(status_path, state)
                if step % 25 == 0 or step == 1:
                    print(arm, step, "FM", result["FM"], "align", result["raw_align"], flush=True)
            else:
                state["status"] = "COMPLETE"
        if stop["signal"] is not None and state["status"] == "TRAINING":
            state.update(status="INTERRUPTED_SAFE", signal=stop["signal"], checkpoint=save(step))
    except BaseException as exc:
        state.update(status="BLOCKED", error=repr(exc), traceback=traceback.format_exc(), step=step,
                     stream_cursor=stream.batch_index, budget=budget.snapshot())
        append_event(events_path, dict(event="TRAIN_FAILED", arm=arm, step=step, error=repr(exc)))
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        state.update(wall_seconds=time.perf_counter()-started, actual_updates_this_call=step-first_step,
                     budget_after=budget.snapshot())
        atomic_json(status_path, state)
    return state
