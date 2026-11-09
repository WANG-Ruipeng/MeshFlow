"""Bounded Phase 2 training only: 16 parents, Q/G, seed 10, 200 updates each.

The caller must first pass all official and standalone prechecks. This module
never changes losses, adds a scheduler, extends training, or launches Phase 3.
"""
from __future__ import annotations
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch

from .adapters import ControlAdapter
from .official_wrapper import OfficialWrapper, state_sha256
from .prompt_dataset import RealChairStream, collate_samples, sample_input_hash


def pack_numpy_state(state):
    return [state[0], torch.from_numpy(state[1].astype(np.int64)), *state[2:]]


def unpack_numpy_state(state):
    return (state[0], state[1].cpu().numpy().astype(np.uint32), *state[2:])


def rng_state(stream):
    return dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all(),
                python=random.getstate(), numpy=pack_numpy_state(np.random.get_state()),
                stream={k: pack_numpy_state(v) for k, v in stream.state_dict().items()})


def restore_rng(state, stream):
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    random.setstate(state["python"])
    np.random.set_state(unpack_numpy_state(state["numpy"]))
    stream.load_state_dict({k: unpack_numpy_state(v) for k, v in state["stream"].items()})


def tensor_hash(batch):
    h = hashlib.sha256()
    for key in ("tokens", "noise", "num_faces", "masks", "t"):
        v = batch[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str((v.dtype, tuple(v.shape))).encode())
        h.update(v.numpy().tobytes())
    for key, v in batch["prompts"].items():
        v = v.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str((v.dtype, tuple(v.shape))).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def velocity_loss_sum(wrapper, batch, precision):
    t = batch["t"]
    x1, x0 = batch["tokens"], batch["noise"]
    xt = t[:, None, None] * x1 + (1-t[:, None, None]) * x0
    with torch.autocast(device_type=x1.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        pred = wrapper(xt, t, y=batch["num_faces"], mask=batch["masks"], prompts=batch["prompts"])
    return ((pred.float() - (x1-x0)).square() * batch["masks"].unsqueeze(-1)).sum()


@torch.no_grad()
def teacher_mse(wrapper, samples, device, precision, microbatch):
    denominator = sum(int(s["num_faces"]) * 9 for s in samples)
    total = 0.
    for start in range(0, len(samples), microbatch):
        batch = collate_samples(samples[start:start+microbatch], device)
        total += velocity_loss_sum(wrapper, batch, precision).item()
    return total / denominator


def _unused_path(path):
    """Preserve earlier stopped attempts and failure logs instead of overwriting."""
    if not path.exists():
        return path
    for number in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_attempt{number:04d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(path)


def train_phase2_arm(base, arm, parents, initial_state, output, base_sha, data_hash,
                     prompt_hash, *, microbatch=2, precision="bf16", device="cuda",
                     reference_input_hashes=None, reference_tensor_hashes=None,
                     resume=None, gates=None):
    if arm not in ("Q_center", "G_closest") or len(parents) != 16:
        raise ValueError("Phase 2 authorizes only Q/G and 16 distinct parents")
    if len({parent.get("parent_uid", parent["uid"]) for parent in parents}) != 16:
        raise ValueError("Phase 2 requires 16 distinct parent UIDs")
    if microbatch not in (1, 2) or precision not in ("fp32", "bf16"):
        raise ValueError("Unsupported shared resource setting")
    if not gates or not all(gates.get(k) == "PASS" for k in ("strict_load", "standalone", "official", "sampler")):
        raise RuntimeError("Training is gated on all required prechecks")
    if reference_input_hashes is not None and len(reference_input_hashes) != 200:
        raise ValueError("Reference input stream must contain exactly 200 completed updates")
    if reference_tensor_hashes is not None and (len(reference_tensor_hashes) != 200 or
            any(len(values) != 8 // microbatch for values in reference_tensor_hashes)):
        raise ValueError("Reference tensor stream must match all 200 updates and the shared microbatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume is not None)
    if (output / "training_report.json").exists():
        raise FileExistsError("A completed run cannot be resumed or overwritten")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=resume is not None)
    torch.manual_seed(10); np.random.seed(10); random.seed(10)
    stream = RealChairStream(parents, seed=10)
    adapter = ControlAdapter(arm).to(device)
    adapter.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=0)
    hashes, tensor_hashes, rolling = [], [], hashlib.sha256()
    start_step = completed_step = 0
    attempted_step = None
    optimizer_in_progress = False
    optimizer_step_attempts = 0
    boundary_rng = None
    initial_teacher_mse = None
    phase_forward_counts = dict(training=0, teacher=0)
    initial_identity = state_sha256_from_state(initial_state)
    protocol = dict(arm=arm, seed=10, effective_batch=8, microbatch=microbatch,
                    precision=precision, lr=1e-4, betas=[.9,.95], weight_decay=0,
                    grad_clip=1., optimizer_updates=200, scheduler=None, adapter_ema=None,
                    loss="original masked FM velocity MSE", base_sha256=base_sha,
                    initial_state_sha256=initial_identity,
                    data_hash=data_hash, prompt_hash=prompt_hash, gates=gates)
    base_before = state_sha256(base)
    saved = None
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("checkpoint_schema") != 2 or saved.get("recoverable") is not True:
            raise ValueError("Checkpoint is not a verified recoverable update boundary")
        if saved["config"] != protocol:
            raise ValueError("Resume configuration, initialization or source identity changed")
        if saved["backbone_state_sha256"] != base_before:
            raise ValueError("Resume backbone state hash differs from the saved official EMA")
        start_step = completed_step = int(saved["step"])
        if not 0 <= start_step <= 200 or len(saved["input_hashes"]) != start_step or len(saved["tensor_hashes"]) != start_step:
            raise ValueError("Checkpoint step and completed input history disagree")
        # Never silently replay already logged optimizer updates from an older
        # checkpoint, which would invalidate the actual 400-update total cap.
        existing_log = output / "train.jsonl"
        if existing_log.exists():
            for line in existing_log.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line)["step"] > start_step:
                    raise ValueError("Checkpoint is older than a logged completed update; replay is prohibited")
        adapter.load_state_dict(saved["adapter"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        hashes, tensor_hashes = list(saved["input_hashes"]), list(saved["tensor_hashes"])
        if reference_input_hashes is not None and hashes != list(reference_input_hashes[:start_step]):
            raise ValueError("Saved raw input stream differs from the reference arm")
        if reference_tensor_hashes is not None and tensor_hashes != list(reference_tensor_hashes[:start_step]):
            raise ValueError("Saved tensor input stream differs from the reference arm")
        for digest in hashes:
            rolling.update(digest.encode())
        phase_forward_counts = dict(saved["phase_forward_counts"])
        optimizer_step_attempts = int(saved["optimizer_step_attempts"])
        initial_teacher_mse = saved["initial_teacher_mse"]
        restore_rng(saved["rng"], stream)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in base.parameters()) or base.training:
        raise ValueError("Official backbone must already be frozen, gradient-free and in eval mode")
    wrapper = OfficialWrapper(base, adapter).train()
    if saved:
        wrapper.counts.update(saved["counts"])
        adapter.call_count = saved["adapter_internal_calls"]
        adapter.encoder_call_count = saved["encoder_internal_calls"]
    session_start_counts = dict(wrapper.counts)
    session_start_phase_counts = dict(phase_forward_counts)
    session_start_optimizer_attempts = optimizer_step_attempts
    input_log = output / "train.jsonl"
    start_time = time.perf_counter()

    def measured_training_loss(batch):
        before = wrapper.counts["wrapper_forward"]
        try:
            return velocity_loss_sum(wrapper, batch, precision)
        finally:
            phase_forward_counts["training"] += wrapper.counts["wrapper_forward"] - before

    def measured_teacher():
        before = wrapper.counts["wrapper_forward"]
        try:
            return teacher_mse(wrapper, fixed_teacher, device, precision, microbatch)
        finally:
            phase_forward_counts["teacher"] += wrapper.counts["wrapper_forward"] - before

    def count_report():
        return dict(counts=dict(wrapper.counts),
                    counts_this_session={key: value-session_start_counts[key] for key, value in wrapper.counts.items()},
                    phase_forward_counts=dict(phase_forward_counts),
                    phase_forward_counts_this_session={key: value-session_start_phase_counts[key]
                                                       for key, value in phase_forward_counts.items()},
                    adapter_internal_calls=adapter.call_count, encoder_internal_calls=adapter.encoder_call_count,
                    optimizer_step_attempts=optimizer_step_attempts,
                    optimizer_step_attempts_this_session=optimizer_step_attempts-session_start_optimizer_attempts)

    def save_checkpoint(label, *, recoverable=True, recovery_reason="completed optimizer update boundary"):
        path = _unused_path(checkpoints / f"adapter_{label}.pt")
        torch.save(dict(checkpoint_schema=2, recoverable=recoverable, recovery_reason=recovery_reason,
                        adapter=adapter.state_dict(), optimizer=optimizer.state_dict(),
                        step=completed_step, attempted_step=attempted_step,
                        optimizer_step_in_progress=optimizer_in_progress,
                        rng=rng_state(stream), config=protocol,
                        input_hashes=hashes, tensor_hashes=tensor_hashes,
                        backbone_state_sha256=base_before, initial_teacher_mse=initial_teacher_mse,
                        **count_report()), path)
        return path

    try:
        fixed_teacher = RealChairStream(parents, seed=9010).next_effective_batch(8)
        teacher_start = measured_teacher()
        if initial_teacher_mse is None:
            initial_teacher_mse = teacher_start
        with input_log.open("a" if resume else "x", encoding="utf-8") as log:
            for attempted_step in range(start_step+1, 201):
                tick = time.perf_counter()
                # Snapshot before fetching any sample, prompt or time. An OOM or
                # backward failure before optimizer.step will replay this input.
                boundary_rng = rng_state(stream)
                samples = stream.next_effective_batch(8)
                input_digest = sample_input_hash(samples)
                if reference_input_hashes is not None and input_digest != reference_input_hashes[attempted_step-1]:
                    raise RuntimeError(f"Q/G input sequence mismatch at step {attempted_step}")
                denominator = sum(int(item["num_faces"]) * 9 for item in samples)
                optimizer.zero_grad(set_to_none=True)
                total_loss, per_micro_hashes = 0., []
                for start in range(0, 8, microbatch):
                    batch = collate_samples(samples[start:start+microbatch], device)
                    digest = tensor_hash(batch)
                    per_micro_hashes.append(digest)
                    if reference_tensor_hashes is not None and digest != reference_tensor_hashes[attempted_step-1][start // microbatch]:
                        raise RuntimeError(f"Q/G tensor input mismatch at step {attempted_step}, microbatch {start // microbatch}")
                    loss = measured_training_loss(batch) / denominator
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    total_loss += loss.item()
                grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True).item()
                if any(p.grad is not None or p.requires_grad for p in base.parameters()) or base.training:
                    raise RuntimeError("Backbone freezing/eval invariant violated")
                next_hashes = hashes + [input_digest]
                next_tensor_hashes = tensor_hashes + [per_micro_hashes]
                next_rolling = rolling.copy(); next_rolling.update(input_digest.encode())
                optimizer_step_attempts += 1
                optimizer_in_progress = True
                optimizer.step()
                # GPU kernels are asynchronous. Synchronize before declaring the
                # update complete so an asynchronous optimizer error is unsafe.
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                hashes, tensor_hashes, rolling, completed_step = next_hashes, next_tensor_hashes, next_rolling, attempted_step
                optimizer_in_progress = False
                boundary_rng = None
                record = dict(step=completed_step, loss=total_loss, grad_norm=grad_norm,
                              valid_coordinates=denominator, input_hash=input_digest,
                              tensor_hashes=per_micro_hashes, rolling_hash=rolling.hexdigest(),
                              seconds=time.perf_counter()-tick)
                if completed_step in (1, 50, 100, 150, 200):
                    record["fixed_teacher_mse"] = measured_teacher()
                    record["backbone_state_sha256"] = state_sha256(base)
                    if record["backbone_state_sha256"] != base_before:
                        raise RuntimeError("Backbone state changed")
                record.update(count_report())
                log.write(json.dumps(record)+"\n"); log.flush()
                print(json.dumps(dict(stage="phase2", arm=arm, **record)), flush=True)
                if completed_step in (100, 200):
                    save_checkpoint(f"step{completed_step:04d}")
        base_after = state_sha256(base)
        if base_after != base_before:
            raise RuntimeError("Frozen backbone hash mismatch")
        teacher_after = measured_teacher()
        if sum(phase_forward_counts.values()) != wrapper.counts["wrapper_forward"]:
            raise RuntimeError("Training/teacher forward counters do not sum to actual wrapper calls")
        result = dict(status="PASS", arm=arm, updates=completed_step-start_step,
                      updates_this_session=completed_step-start_step, completed_updates_total=completed_step,
                      start_step=start_step, input_hashes=hashes, tensor_hashes=tensor_hashes,
                      input_sequence_sha256=rolling.hexdigest(), initial_state_sha256=initial_identity,
                      teacher_mse_before=initial_teacher_mse, teacher_mse_at_session_start=teacher_start,
                      teacher_mse_after=teacher_after, base_before=base_before, base_after=base_after,
                      seconds=time.perf_counter()-start_time,
                      peak_cuda_allocated=torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else 0,
                      **count_report())
        (output / "training_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return wrapper, result
    except BaseException as exc:
        # An optimizer exception may have partially written parameters/moments.
        # Such a forensic checkpoint must NEVER claim to be resumable.
        recoverable = not optimizer_in_progress and len(hashes) == completed_step and len(tensor_hashes) == completed_step
        reason = "completed optimizer update boundary"
        if optimizer_in_progress:
            reason = "optimizer.step or its CUDA synchronization failed; parameters/moments may be partially written"
        elif not recoverable:
            reason = "completed-step metadata and input history disagree"
        elif boundary_rng is not None:
            restore_rng(boundary_rng, stream)
            reason = "failed before optimizer.step; RNG restored to start of the uncompleted input batch"
        failure = dict(error=repr(exc), completed_updates=completed_step,
                       updates_this_session=completed_step-start_step, attempted_step=attempted_step,
                       recoverable=recoverable, recovery_reason=reason, **count_report())
        try:
            current_base_hash = state_sha256(base)
            failure["backbone_state_sha256_actual"] = current_base_hash
            if current_base_hash != base_before or any(p.requires_grad or p.grad is not None for p in base.parameters()) or base.training:
                recoverable = False
                reason = "backbone hash/frozen-state invariant failed"
            failure.update(recoverable=recoverable, recovery_reason=reason)
            optimizer.zero_grad(set_to_none=True)
            label = f"stopped_step{completed_step:04d}" if recoverable else f"unrecoverable_attempt{attempted_step or completed_step:04d}"
            failure["checkpoint"] = str(save_checkpoint(label, recoverable=recoverable, recovery_reason=reason))
        except BaseException as checkpoint_error:
            failure.update(recoverable=False, checkpoint_error=repr(checkpoint_error))
        finally:
            wrapper.close()
            failure_path = _unused_path(output / "failure.json")
            failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise


def state_sha256_from_state(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()