"""Independent-Gaussian NFT-style / SELF_REFLOW learning for this pilot.

MeshFlow time is noise=0, endpoint=1: x_t=(1-t)*noise+t*endpoint and
v_target=endpoint-noise. We use the official NFT self-normalized endpoint
regression, with branch-specific stop-gradient MAE denominators (floor 1e-5).
The scalar r is already the registered [0,1] optimality probability: this
module never recalibrates rewards or silently drops UNKNOWN samples.

SELF_FM is an API label for SELF_REFLOW: same generated endpoints, independent
noise, times, masks, and endpoint normalization, with no preference signal.
Reference regularization is velocity MSE; it is NOT an exact distribution KL.
"""
from __future__ import annotations

from pathlib import Path
import copy
import os
import uuid

from .model import VERSION, nft_execution, file_sha256
import torch
from meshflow_control.checkpoints import rng_state, restore_rng, state_hash

SCHEMA = "terminal_reward_nft_training_v1"
BETA_NFT = 0.1
LAMBDA_REF = 0.01
MAE_FLOOR = 1e-5
MICROBATCHES = 8
CLIP_NORM = 1.0


def optimizer_names(policy):
    return policy.current_parameter_names()


def make_optimizer(policy):
    return torch.optim.AdamW(policy.current_parameters(), lr=1e-5,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
        amsgrad=False, foreach=False, fused=False)


def validate_optimizer(policy, optimizer, *, step=None):
    if len(optimizer.param_groups) != 1:
        raise ValueError("Exactly one current-LoRA AdamW parameter group is required")
    group = optimizer.param_groups[0]
    params = policy.current_parameters()
    if [id(p) for p in group["params"]] != [id(p) for p in params]:
        raise ValueError("Optimizer does not own exactly the ordered current adapters")
    if (group["lr"], tuple(group["betas"]), group["eps"], group["weight_decay"],
            group["amsgrad"], group["foreach"], group["fused"]) != (
            1e-5, (0.9, 0.999), 1e-8, 0.0, False, False, False):
        raise ValueError("Frozen optimizer hyperparameters changed")
    for p in params:
        state = optimizer.state.get(p, {})
        if not state:
            if step not in (None, 0):
                raise ValueError("Missing optimizer history")
            continue
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Unexpected AdamW state")
        if step is not None and float(state["step"]) != step:
            raise ValueError("AdamW step differs from completed update count")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if value.dtype != torch.float32 or value.shape != p.shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid FP32 optimizer moments")


def free_mean(value, free_mask):
    """Mean per output over valid free coordinates only (not over padding/C)."""
    if value.ndim != 3 or value.shape[-1] != 9:
        raise ValueError("Expected values[B,N,9]")
    if free_mask.dtype != torch.bool or free_mask.shape != value.shape[:2]:
        raise ValueError("Expected bool free_mask[B,N]")
    denominator = free_mask.sum(1) * value.shape[-1]
    if bool((denominator <= 0).any()):
        raise ValueError("Every output needs at least one valid free face")
    selected = torch.where(free_mask[..., None], value, torch.zeros_like(value))
    return selected.sum((1, 2)) / denominator.to(value.dtype)


def endpoint_regression(prediction, endpoint, free_mask):
    error = prediction.float() - endpoint.float()
    # Official implementation computes the branch MAE in float64, detached.
    with torch.no_grad():
        weight = free_mean(error.double().abs(), free_mask).clamp_min(MAE_FLOOR)
    loss = free_mean(error.square(), free_mask) / weight
    return loss, weight


def compute_losses(v_current, v_old, v_reference, xt, endpoint, t, free_mask,
                   *, arm, r=None, known_reward=None, beta_nft=BETA_NFT,
                   lambda_ref=LAMBDA_REF):
    """Pure differentiable loss; usable in small CPU algebra/gradient checks."""
    if arm not in ("NFT", "SELF_FM", "SELF_REFLOW"):
        raise ValueError("Unknown learning arm")
    if beta_nft != BETA_NFT or lambda_ref != LAMBDA_REF:
        raise ValueError("NFT beta and reference coefficient are frozen")
    batch = endpoint.shape[0]
    if t.shape != (batch,) or v_current.shape != endpoint.shape or v_reference.shape != endpoint.shape:
        raise ValueError("Prediction/time shapes differ")
    scale = (1.0 - t.float())[:, None, None]
    current = v_current.float()
    reference = v_reference.detach().float()
    ref_per_output = free_mean((current - reference).square(), free_mask)
    statistics = dict(reference_velocity_mse=ref_per_output.mean().detach(),
                      reference_is_exact_kl=False)
    if arm == "NFT":
        if v_old is None or v_old.shape != endpoint.shape:
            raise ValueError("NFT requires a frozen old prediction")
        if r is None or known_reward is None or r.shape != (batch,) or known_reward.shape != (batch,):
            raise ValueError("NFT requires reward probabilities and known flags[B]")
        if known_reward.dtype != torch.bool:
            raise ValueError("Reward known flags must be bool")
        if bool((known_reward & (~torch.isfinite(r) | (r < 0) | (r > 1))).any()):
            raise ValueError("Known reward probabilities must be finite and in [0,1]")
        reward = torch.where(known_reward, r.float(), torch.full_like(r.float(), 0.5))
        old = v_old.detach().float()
        positive = (1.0 - beta_nft) * old + beta_nft * current
        negative = (1.0 + beta_nft) * old - beta_nft * current
        positive_loss, positive_weight = endpoint_regression(xt + scale * positive, endpoint, free_mask)
        negative_loss, negative_weight = endpoint_regression(xt + scale * negative, endpoint, free_mask)
        pref_per_output = (reward * positive_loss + (1.0 - reward) * negative_loss) / beta_nft
        # Denominator remains the registered batch, including UNKNOWN entries;
        # UNKNOWN removes its preference contribution, not its reference term.
        pref_per_output = torch.where(known_reward, pref_per_output, torch.zeros_like(pref_per_output))
        preference = pref_per_output.mean()
        statistics.update(preference_loss=preference.detach(),
            positive_endpoint_loss=positive_loss.mean().detach(),
            negative_endpoint_loss=negative_loss.mean().detach(),
            positive_weight=positive_weight.mean().detach(),
            negative_weight=negative_weight.mean().detach(),
            old_velocity_mse=free_mean((current - old).square(), free_mask).mean().detach(),
            reward_known=int(known_reward.sum().item()), reward_unknown=int((~known_reward).sum().item()))
    else:
        values, weights = endpoint_regression(xt + scale * current, endpoint, free_mask)
        preference = values.mean()
        statistics.update(self_reflow_loss=preference.detach(), endpoint_weight=weights.mean().detach(),
                          reward_used=False, reward_known=0, reward_unknown=0)
    loss = preference + lambda_ref * ref_per_output.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite NFT/SELF_REFLOW loss; no retry is allowed")
    statistics["loss"] = loss.detach()
    return loss, statistics


def loss_for_batch(policy, batch, *, arm, beta_nft=BETA_NFT, lambda_ref=LAMBDA_REF, budget=None, context=None):
    """Run only budgeted training forwards; randomness is provided by caller."""
    endpoint, noise, t = batch["endpoint"], batch["noise"], batch["t"]
    valid, known, y = batch["valid_mask"], batch["known_mask"], batch["y"]
    if any(v.dtype != torch.float32 for v in (endpoint, noise, t)):
        raise ValueError("Training endpoint/noise/time must be FP32")
    if noise.shape != endpoint.shape or t.shape != (endpoint.shape[0],):
        raise ValueError("Training endpoint/noise/time shapes differ")
    if not bool(torch.isfinite(endpoint).all() and torch.isfinite(noise).all() and torch.isfinite(t).all()):
        raise ValueError("Nonfinite training batch")
    if bool(((t < .001) | (t > .999)).any()):
        raise ValueError("Training time must use frozen [.001,.999] clipping")
    if endpoint.requires_grad or noise.requires_grad or t.requires_grad:
        raise ValueError("Generated endpoints and sampled noise/time must be detached")
    free = valid & ~known
    mixed = (1.0 - t[:, None, None]) * noise + t[:, None, None] * endpoint
    xt = torch.where(free[..., None], mixed, endpoint)
    if not torch.equal(xt[known], endpoint[known]):
        raise RuntimeError("Known C changed during independent Gaussian noising")
    args = (xt, t, y)
    kwargs = dict(valid_mask=valid, known_mask=known)
    old = None
    def forward(branch):
        if budget is None:
            return policy(*args, **kwargs)
        return budget.call("training_forward", lambda: policy(*args, **kwargs),
                           context={**(context or {}), "arm": arm, "branch": branch})
    with nft_execution(policy, coordinates=(xt, t)):
        if arm == "NFT":
            with policy.activate("old"), torch.no_grad():
                old = forward("old").detach()
        with policy.activate("reference"), torch.no_grad():
            reference = forward("reference").detach()
        with policy.activate("current"):
            current = forward("current")
        loss, statistics = compute_losses(current, old, reference, xt, endpoint, t, free,
            arm=arm, r=batch.get("r"), known_reward=batch.get("known_reward"),
            beta_nft=beta_nft, lambda_ref=lambda_ref)
        statistics.update(free_velocity_target_mse=free_mean(
            (current.float() - (endpoint - noise)).square(), free).mean().detach(),
            actual_model_forwards=3 if arm == "NFT" else 2,
            examples=int(endpoint.shape[0]), free_faces=int(free.sum().item()))
    return loss, statistics


def step_microbatches(policy, optimizer, batches, *, arm, budget=None, context=None):
    """Exactly eight one-output microbatches, one clipped AdamW update.

The caller records START before calling and COMPLETE only after return. Failure
does not clear/restore state or retry. No old snapshot update occurs here.
"""
    if len(batches) != MICROBATCHES or any(b["endpoint"].shape[0] != 1 for b in batches):
        raise ValueError("Registered effective batch is 8 with microbatch=1")
    validate_optimizer(policy, optimizer)
    optimizer.zero_grad(set_to_none=True)
    rows = []
    for index, batch in enumerate(batches):
        loss, stats = loss_for_batch(policy, batch, arm=arm, budget=budget,
            context={**(context or {}), "microbatch": index})
        (loss / MICROBATCHES).backward()
        rows.append({k: (float(v.item()) if isinstance(v, torch.Tensor) else v) for k, v in stats.items()})
    params = policy.current_parameters()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
        raise FloatingPointError("Missing/nonfinite current adapter gradient; no retry")
    gradient_norm = torch.nn.utils.clip_grad_norm_(params, CLIP_NORM, error_if_nonfinite=True, foreach=False)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in params):
        raise FloatingPointError("Nonfinite adapter after AdamW; no retry")
    optimizer.zero_grad(set_to_none=True)
    return dict(microbatches=MICROBATCHES, examples=MICROBATCHES,
                gradient_norm_before_clip=float(gradient_norm.item()), clip_norm=CLIP_NORM,
                actual_model_forwards=sum(row["actual_model_forwards"] for row in rows), rows=rows)


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


def save_checkpoint(path, policy, optimizer, *, step, round_index, arm, metadata=None):
    """Save adapter + optimizer/RNG; frozen source base stays external/protected."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite completed pilot checkpoint: " + str(path))
    validate_optimizer(policy, optimizer, step=int(step))
    if policy.active_adapter != "current":
        raise RuntimeError("Checkpoint only with current adapter active")
    if any(p.grad is not None for p in policy.current_parameters()):
        raise RuntimeError("Checkpoint only after completed update and cleared gradients")
    adapters = policy.adapter_state_dict()
    payload = dict(schema=SCHEMA, model_version=VERSION, arm=arm, step=int(step),
        round_index=int(round_index), update_in_progress=False,
        adapters=adapters, adapters_sha256=state_hash(adapters),
        optimizer=_cpu_copy(optimizer.state_dict()), optimizer_names=optimizer_names(policy),
        model_metadata=policy.metadata(), metadata=_cpu_copy(metadata or {}), rng=_cpu_copy(rng_state()),
        loss_contract=dict(beta_nft=BETA_NFT, lambda_ref=LAMBDA_REF,
            normalization="branch_free_only_stopgrad_endpoint_MAE", mae_floor=MAE_FLOOR,
            old_update="exact_round_start_snapshot", reference_penalty="velocity_MSE_not_exact_KL"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            raise FileExistsError("Checkpoint appeared during save")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(path=str(path), sha256=file_sha256(path), adapters_sha256=payload["adapters_sha256"],
                step=int(step), round_index=int(round_index))


def load_checkpoint(path, policy, optimizer=None, *, expected_sha256=None, restore_random_state=True):
    path = Path(path)
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("Pilot checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("model_version") != VERSION or payload.get("update_in_progress") is not False:
        raise ValueError("Not a completed registered NFT pilot checkpoint")
    if payload["model_metadata"] != policy.metadata():
        raise ValueError("Base/LoRA/runtime identity differs")
    if payload["optimizer_names"] != optimizer_names(policy):
        raise ValueError("Optimizer parameter order differs")
    if state_hash(payload["adapters"]) != payload["adapters_sha256"]:
        raise ValueError("Adapter checkpoint internal hash differs")
    policy.load_adapter_state_dict(payload["adapters"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
        validate_optimizer(policy, optimizer, step=payload["step"])
    if restore_random_state:
        restore_rng(payload["rng"])
    return payload
