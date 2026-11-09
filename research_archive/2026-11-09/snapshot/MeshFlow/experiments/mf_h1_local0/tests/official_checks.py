"""Real-checkpoint Phase 1 checks; the caller owns loading and GPU scheduling.

This module never constructs a replacement backbone, starts a sampler, or
performs formal Phase 2 training. Small face subsets are interface probes only;
their losses and outputs are never represented as full-object training/results.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..adapters import ControlAdapter
from ..official_wrapper import OfficialWrapper, state_sha256


def _rng_state():
    numpy = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"kind": numpy[0], "keys": torch.from_numpy(numpy[1].copy()),
                      "pos": numpy[2], "has_gauss": numpy[3], "cached_gaussian": numpy[4]},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy["kind"], numpy["keys"].cpu().numpy(), numpy["pos"],
                         numpy["has_gauss"], numpy["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tree_digest(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(str((tuple(item.shape), item.dtype)).encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def _difference(reference, actual, atol, rtol):
    reference, actual = reference.detach().double(), actual.detach().double()
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    delta = (reference - actual).abs()
    return {"status": "PASS" if finite and torch.allclose(reference, actual, atol=atol, rtol=rtol) else "FAIL",
            "bitwise_equal": bool(torch.equal(reference, actual)),
            "max_abs": (float(delta.max().item()) if delta.numel() else 0.0) if finite else None,
            "max_rel": (float((delta / reference.abs().clamp_min(1e-12)).max().item()) if delta.numel() else 0.0) if finite else None,
            "atol": atol, "rtol": rtol, "finite": finite}


@contextmanager
def _deterministic_fp32(device):
    old = {"deterministic": torch.are_deterministic_algorithms_enabled(),
           "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
           "tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn_tf32": torch.backends.cudnn.allow_tf32,
           "benchmark": torch.backends.cudnn.benchmark,
           "cudnn_deterministic": torch.backends.cudnn.deterministic}
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.use_deterministic_algorithms(old["deterministic"], warn_only=old["warn_only"])
        torch.backends.cuda.matmul.allow_tf32 = old["tf32"]
        torch.backends.cudnn.allow_tf32 = old["cudnn_tf32"]
        torch.backends.cudnn.benchmark = old["benchmark"]
        torch.backends.cudnn.deterministic = old["cudnn_deterministic"]


def _prepare_probe(real_batch, device):
    required = ("tokens", "noise", "num_faces", "masks", "prompts")
    for key in required:
        if key not in real_batch:
            raise ValueError(f"Real collated batch is missing {key}")
    # Prefer a mixed prompt for nontrivial key gradients. Selection is solely
    # by declared input token count, never by a model output or loss.
    index = int(real_batch["prompts"]["primitive_mask"].sum(-1).argmax())
    valid = real_batch["masks"][index].nonzero().flatten()
    if len(valid) < 2:
        raise ValueError("At least two real face tokens are needed for interface probes")
    selected = valid[:16]
    tokens = real_batch["tokens"][index:index+1, selected].detach().to(device=device, dtype=torch.float32)
    noise = real_batch["noise"][index:index+1, selected].detach().to(device=device, dtype=torch.float32)
    y = real_batch["num_faces"][index:index+1].detach().to(device=device, dtype=torch.long).reshape(1)
    prompts = {key: value[index:index+1].detach().to(device) for key, value in real_batch["prompts"].items()}
    prompts = {key: value.float() if value.is_floating_point() else value for key, value in prompts.items()}
    mask = torch.ones(tokens.shape[:2], device=device, dtype=torch.bool)
    t = torch.full((1,), 0.375, device=device, dtype=torch.float32)
    return {"tokens": tokens, "noise": noise, "y": y, "mask": mask, "t": t,
            "x_t": (1-t[:, None, None])*noise + t[:, None, None]*tokens,
            "velocity": tokens-noise, "prompts": prompts,
            "source_batch_index": index, "source_face_indices": selected.cpu().tolist(),
            "source_parent_face_count": int(y.item()), "probe_face_slots": len(selected)}


def run_official_checks(base, device, output_dir, real_batch):
    """Run real-backbone checks and persist every completed result.

    ``base`` must already hold strictly loaded official EMA FP32 parameters and
    have no other OfficialWrapper attached. ``real_batch`` is the result of
    prompt_dataset.collate_samples, including real tokens/noise/masks/prompts.
    Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before the process initializes CUDA.

    Returns a JSON-serializable report. Failures remain FAIL and prevent the
    caller from treating this as a formal training authorization. Expected
    optimizer probe budget: 5 Q + 5 G + (10 vs 5+5 restore) = 30 updates.
    """
    device = torch.device(device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "official_unit_tests.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checks: {report_path}")
    report = {"status": "RUNNING", "tests": [], "precision": "float32",
              "attention_backend": "PyTorch SDPA math for all comparisons",
              "deterministic_algorithms": True, "tf32": False,
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "scope": "official EMA interface probes only; no formal training or full sampling",
              "probe_optimizer_updates": 0, "formal_optimizer_updates": 0,
              "complete_sampling_calls": 0, "wrapper_runs": [],
              "counts": {"actual_backbone_forward": 0, "direct_base_forward": 0}}
    previous_rng = _rng_state()
    started = time.perf_counter()
    current = [None]
    counter_handle = None

    def write():
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    def record(name, result):
        result = dict(result, name=name)
        report["tests"].append(result)
        write()
        return result["status"] == "PASS"

    def attach(arm, initial):
        if current[0] is not None:
            raise RuntimeError("A wrapper is still attached")
        adapter = ControlAdapter(arm).to(device=device, dtype=torch.float32)
        adapter.load_state_dict(initial, strict=True)
        wrapper = OfficialWrapper(base, adapter).train()
        current[0] = wrapper
        return wrapper

    def close(label):
        wrapper = current[0]
        if wrapper is not None:
            report["wrapper_runs"].append({"label": label, "arm": wrapper.adapter.arm,
                                            **dict(wrapper.counts),
                                            "adapter_internal_calls": wrapper.adapter.call_count,
                                            "encoder_internal_calls": wrapper.adapter.encoder_call_count})
            wrapper.close()
            current[0] = None

    try:
        if any(p.dtype != torch.float32 for p in base.parameters()):
            raise ValueError("Official checks require FP32 checkpoint parameters; no silent conversion")
        if any(p.requires_grad for p in base.parameters()) or base.training:
            raise ValueError("Caller must freeze the official backbone and put it in eval mode")
        if any(p.grad is not None for p in base.parameters()):
            raise ValueError("Backbone unexpectedly has pre-existing gradients")
        if any(layer._forward_hooks for layer in base.layers):
            raise ValueError("Remove existing backbone layer hooks before official checks")
        probe = _prepare_probe(real_batch, device)
        report["input"] = {key: probe[key] for key in ("source_batch_index", "source_face_indices",
                           "source_parent_face_count", "probe_face_slots")}
        report["input"]["subset_warning"] = "16 or fewer real faces for interface tests only; no full-object quality claim"
        report["input"]["valid_primitives"] = int(probe["prompts"]["primitive_mask"].sum())
        torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else
                    {k: v.cpu() for k, v in value.items()} if key == "prompts" else value
                    for key, value in probe.items()}, output / "official_probe_inputs.pt")
        report["base_sha256_before"] = state_sha256(base)

        def count_base(module, args):
            report["counts"]["actual_backbone_forward"] += 1
        counter_handle = base.register_forward_pre_hook(count_base)
        _seed(20260928)
        initial_adapter = ControlAdapter("Q_center")
        initial = copy.deepcopy(initial_adapter.state_dict())
        del initial_adapter
        report["adapter_initial_sha256"] = _tree_digest(initial)
        x, t, y, mask, prompts = (probe[key] for key in ("x_t", "t", "y", "mask", "prompts"))

        with _deterministic_fp32(device):
            with torch.no_grad():
                report["counts"]["direct_base_forward"] += 1
                reference = base(x, t, y, mask=mask)
            identities_ok = True
            for arm in ("Q_center", "G_closest"):
                wrapper = attach(arm, initial)
                with torch.no_grad():
                    off = wrapper(x, t, y, mask=mask, prompts=prompts, control_scale=0)
                    zero = wrapper(x, t, y, mask=mask, prompts=prompts)
                identities_ok &= record(f"{arm}/control_off_identity", _difference(reference, off, 1e-6, 1e-5))
                identities_ok &= record(f"{arm}/zero_output_identity", _difference(reference, zero, 1e-6, 1e-5))
                close(f"{arm}/identities")
            if not identities_ok:
                raise RuntimeError("Identity checks failed; stopped before optimizer probes")

            # Separate nonzero test projections; these weights never enter a
            # gradient probe, Phase 2 initialization, or generated result.
            for arm in ("Q_center", "G_closest"):
                wrapper = attach(arm, initial)
                with torch.no_grad():
                    for layer in wrapper.adapter.layers:
                        layer.output_projection.weight.normal_(std=0.001)
                        layer.output_projection.bias.fill_(0.001)
                    nonzero = wrapper(x, t, y, mask=mask, prompts=prompts)
                    finite = bool(torch.isfinite(nonzero).all())
                    strength = float((nonzero-reference).abs().max()) if finite else None
                    record(f"{arm}/nonzero_test_projection", {"status": "PASS" if finite and strength > 0 else "FAIL",
                           "max_output_change": strength, "finite": finite, "threshold": 0.0})
                    permutation = torch.arange(x.shape[1]-1, -1, -1, device=device)
                    changed = wrapper(x[:, permutation].contiguous(), t, y, mask=mask[:, permutation], prompts=prompts)
                    record(f"{arm}/face_permutation", _difference(nonzero[:, permutation], changed, 1e-5, 1e-5))
                    # Different permutations per face, not just a global corner rotation.
                    corner_order = torch.tensor([[2, 0, 1], [1, 0, 2]], device=device)
                    corner_order = corner_order[torch.arange(x.shape[1], device=device) % 2]
                    gather = corner_order[None, :, :, None].expand(1, -1, -1, 3)
                    changed_x = x.reshape(1, -1, 3, 3).gather(2, gather).reshape_as(x).contiguous()
                    changed = wrapper(changed_x, t, y, mask=mask, prompts=prompts)
                    expected = nonzero.reshape(1, -1, 3, 3).gather(2, gather).reshape_as(nonzero)
                    record(f"{arm}/within_face_corner_permutation", _difference(expected, changed, 1e-5, 1e-5))
                    padded_x = torch.cat((x, torch.full((1, 3, 9), 7.25, device=device)), 1)
                    padded_mask = torch.cat((mask, torch.zeros((1, 3), device=device, dtype=torch.bool)), 1)
                    padded = wrapper(padded_x, t, y, mask=padded_mask, prompts=prompts)
                    record(f"{arm}/face_padding", _difference(nonzero, padded[:, :x.shape[1]], 1e-5, 1e-5))
                close(f"{arm}/nonzero_permutations")

            def update(wrapper, optimizer):
                optimizer.zero_grad(set_to_none=True)
                # Exercise every saved RNG stream. This probe distribution is
                # explicitly separate from official Phase 2 data/t sampling.
                random.random()
                np.random.random()
                torch.rand((), device="cpu")
                probe_t = torch.sigmoid(torch.randn((1,), device=device))
                current_x = (1-probe_t[:, None, None])*probe["noise"] + probe_t[:, None, None]*probe["tokens"]
                prediction = wrapper(current_x, probe_t, y, mask=mask, prompts=prompts)
                loss = (prediction-probe["velocity"]).square()[mask].mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite official gradient-probe loss")
                loss.backward()
                gradients = {name: {"max_abs": float(parameter.grad.abs().max()) if parameter.grad is not None and torch.isfinite(parameter.grad).all() else None,
                                     "finite": bool(torch.isfinite(parameter.grad).all()) if parameter.grad is not None else False}
                             for name, parameter in wrapper.adapter.named_parameters()}
                if any(parameter.grad is not None for parameter in base.parameters()):
                    raise RuntimeError("Frozen backbone acquired a gradient")
                torch.nn.utils.clip_grad_norm_(wrapper.adapter.parameters(), 1.0)
                optimizer.step()
                report["probe_optimizer_updates"] += 1
                return float(loss.detach()), gradients

            def optimizer_for(wrapper):
                return torch.optim.AdamW(wrapper.adapter.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0)

            for arm in ("Q_center", "G_closest"):
                wrapper = attach(arm, initial)
                optimizer = optimizer_for(wrapper)
                _seed(10)
                losses, first_grads, final_grads = [], None, None
                for step in range(5):
                    loss, final_grads = update(wrapper, optimizer)
                    losses.append(loss)
                    if step == 0:
                        first_grads = final_grads
                output_names = [f"layers.{i}.output_projection.weight" for i in range(4)]
                key_names = ["condition_encoder.0.weight", "layers.0.query_projection.weight",
                             "layers.0.dynamic_mlp.0.weight", "layers.0.dynamic_mlp.2.weight", *output_names]
                if report["input"]["valid_primitives"] > 1:
                    key_names.append("layers.0.key_projection.weight")
                has_gradient = lambda values, names: all(values[name]["finite"] and values[name]["max_abs"] > 0 for name in names)
                record(f"{arm}/five_update_gradient_probe", {
                    "status": "PASS" if has_gradient(first_grads, output_names) and has_gradient(final_grads, key_names) else "FAIL",
                    "optimizer_updates": 5, "losses": losses,
                    "first_step_output_gradients": {name: first_grads[name] for name in output_names},
                    "first_step_nonoutput_max_gradient": max(value["max_abs"] or 0 for name, value in first_grads.items() if "output_projection" not in name),
                    "fifth_step_required_gradients": {name: final_grads[name] for name in key_names},
                    "fifth_step_all_gradients": final_grads,
                    "earliest_adapter_gradient_traverses_frozen_layers_4_through_12": True,
                    "backbone_gradients_absent": all(p.grad is None for p in base.parameters()),
                    "backbone_eval": not base.training,
                    "single_token_key_gradient_not_required": report["input"]["valid_primitives"] == 1})
                close(f"{arm}/5_updates")
                del wrapper, optimizer

            # Both paths use the same exact initialization, data subset and RNG.
            continuous = attach("Q_center", initial)
            optimizer = optimizer_for(continuous)
            _seed(7710)
            continuous_losses = [update(continuous, optimizer)[0] for _ in range(10)]
            with torch.no_grad():
                continuous_output = continuous(x, t, y, mask=mask, prompts=prompts).detach().clone()
            continuous_model = _tree_digest(continuous.adapter.state_dict())
            continuous_optimizer = _tree_digest(optimizer.state_dict())
            continuous_rng = _tree_digest(_rng_state())
            close("Q_center/restore_continuous_10")
            del continuous, optimizer

            split = attach("Q_center", initial)
            optimizer = optimizer_for(split)
            _seed(7710)
            split_losses = [update(split, optimizer)[0] for _ in range(5)]
            checkpoint_path = output / "official_probe_resume_step5.pt"
            torch.save({"adapter": split.adapter.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": 5, "rng": _rng_state(), "base_sha256": report["base_sha256_before"],
                        "scope": "disposable Phase 1 probe, never Phase 2 initialization"}, checkpoint_path)
            close("Q_center/restore_before_save_5")
            del split, optimizer
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            resumed = attach("Q_center", saved["adapter"])
            optimizer = optimizer_for(resumed)
            optimizer.load_state_dict(saved["optimizer"])
            _restore_rng(saved["rng"])
            split_losses.extend(update(resumed, optimizer)[0] for _ in range(5))
            with torch.no_grad():
                resumed_output = resumed(x, t, y, mask=mask, prompts=prompts)
            model_equal = continuous_model == _tree_digest(resumed.adapter.state_dict())
            optimizer_equal = continuous_optimizer == _tree_digest(optimizer.state_dict())
            rng_equal = continuous_rng == _tree_digest(_rng_state())
            difference = _difference(continuous_output, resumed_output, 1e-6, 1e-5)
            record("Q_center/official_backbone_10_vs_5_save_restore_5", {
                **difference, "status": "PASS" if difference["status"] == "PASS" and model_equal and optimizer_equal and rng_equal and continuous_losses == split_losses else "FAIL",
                "adapter_exact_equal": model_equal, "optimizer_exact_equal": optimizer_equal,
                "all_rng_exact_equal": rng_equal, "losses_exact_equal": continuous_losses == split_losses,
                "continuous_losses": continuous_losses, "resumed_losses": split_losses,
                "optimizer_updates_executed": 20, "checkpoint": str(checkpoint_path),
                "backbone_frozen_after_restore": all(not p.requires_grad and p.grad is None for p in base.parameters()),
                "backbone_eval_after_restore": not base.training})
            close("Q_center/restore_after_load_5")
            del resumed, optimizer
    except Exception as error:
        record("execution_exception", {"status": "FAIL", "type": type(error).__name__,
                                      "message": str(error), "traceback": traceback.format_exc()})
    finally:
        close("interrupted_or_failed_check")
        if counter_handle is not None:
            counter_handle.remove()
        if "base_sha256_before" in report:
            report["base_sha256_after"] = state_sha256(base)
            record("official_backbone_frozen_state", {
                "status": "PASS" if report["base_sha256_before"] == report["base_sha256_after"] and
                all(not p.requires_grad and p.grad is None for p in base.parameters()) and not base.training else "FAIL",
                "hash_equal": report["base_sha256_before"] == report["base_sha256_after"],
                "no_backbone_gradients": all(p.grad is None for p in base.parameters()),
                "no_trainable_backbone_parameters": all(not p.requires_grad for p in base.parameters()),
                "eval": not base.training})
        totals = report["counts"]
        totals["wrapper_forward"] = sum(run["wrapper_forward"] for run in report["wrapper_runs"])
        totals["adapter_layer"] = sum(run["adapter_layer"] for run in report["wrapper_runs"])
        totals["condition_encoding"] = sum(run["condition_encoding"] for run in report["wrapper_runs"])
        totals["adapter_internal_calls"] = sum(run["adapter_internal_calls"] for run in report["wrapper_runs"])
        record("actual_call_counters", {"status": "PASS" if
            totals["actual_backbone_forward"] == totals["wrapper_forward"] + totals["direct_base_forward"] and
            totals["adapter_layer"] == totals["adapter_internal_calls"] else "FAIL", **totals})
        report["status"] = "PASS" if report["tests"] and all(test["status"] == "PASS" for test in report["tests"]) and report["probe_optimizer_updates"] == 30 else "FAIL"
        _restore_rng(previous_rng)
        write()
    return report
