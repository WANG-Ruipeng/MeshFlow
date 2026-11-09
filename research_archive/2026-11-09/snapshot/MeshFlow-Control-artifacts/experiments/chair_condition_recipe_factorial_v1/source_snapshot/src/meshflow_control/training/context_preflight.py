"""Bounded real preflight: 6 generator F, 10 zero-update B; explicit dispatch only."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import time
import traceback
import torch
from ..data.dataset import collate, model_inputs
from ..data.contextual_stream import HybridStream
from ..data.io import atomic_json, digest, read_json
from ..checkpoints import state_hash, START_STATE_SHA256
from .context_alignment import AlignmentHead, AttentionCapture, select_messages, alignment_loss
from .context_contract import validate_config, HEAD_PARAMETERS
from .context_checkpoint import load_start, validate_optimizer
from .context_trainer import observation_parameters
from .trainer import _call, _norm
from .losses import masked_fm_loss


def run(out, start_checkpoint, data_manifest, stream_root, teacher_manifest, budget, device="cuda"):
    """Caller owns shared GPU lock; no warmup, update, generation or retries."""
    from ..models.native import native_execution
    from ..teachers.siglip_targets import TargetStore
    out = Path(out)
    folder = out
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "preflight.json"
    if result_path.exists():
        raise FileExistsError("Preflight evidence exists; no silent rerun")
    result = dict(status="RUNNING", planned_generator_forward=6, planned_backward=10,
        optimizer_updates=0, rows=[], budget_before=budget.snapshot(),
        capture_site="backbone.layers[3].attn actual module output before external gate")
    atomic_json(result_path, result)
    began = time.perf_counter()
    try:
        stream = HybridStream(Path(data_manifest).parent, stream_root, "A")
        batch_samples = stream.next_effective_batch()
        expected = {r["sample_index"]: r for r in read_json(Path(stream_root)/"paired_inputs.json")["records"]}
        for s in batch_samples:
            for key in ("shared_input_sha256", "input_bytehash", "label_bytehash", "ot_cache_key"):
                if expected[s["sample_index"]][key] != s[key]:
                    raise ValueError("Preflight input differs from frozen shared identity")
        chosen = [s for s in batch_samples if s["alpha"] == 1.][:2]
        denominator = sum(s["free_coordinate_count"] for s in batch_samples)
        target_store = TargetStore(Path(teacher_manifest).parent)
        model, _, optimizer, audit = load_start(start_checkpoint, validate_config({"arm": "A"}), device)
        actual_device = next(model.parameters()).device
        if actual_device.type != "cuda":
            raise ValueError("Real preflight must be explicitly dispatched on owned GPU")
        head = AlignmentHead().to(device).eval()
        head_initial = state_hash(head.state_dict())
        twin = AlignmentHead()
        identical_head = head_initial == state_hash(twin.state_dict())
        del twin
        if not identical_head:
            raise AssertionError("B/C initialization differs")
        result.update(start_import=audit, identical_B_C_head=identical_head, head_parameters=HEAD_PARAMETERS,
                      teacher_manifest_sha256=digest(teacher_manifest), sample_indices=[s["sample_index"] for s in chosen],
                      denominator=denominator)
        for sample in chosen:
            batch = collate([sample], device)
            teacher = torch.as_tensor(target_store.for_sample(sample), device=device, dtype=torch.float32)
            ctx = dict(stage="contextual_alignment_preflight", sample_index=sample["sample_index"])
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(actual_device)
            with native_execution(model, coordinates=(batch["xt"], batch["t"]), autocast=True):
                baseline, fs = _call(budget, "preflight_forward", lambda: model(*model_inputs(batch)).float(),
                                     dict(ctx, arm="A", capture=False), actual_device)
                fm = masked_fm_loss(baseline, batch["u"], batch["valid_mask"], batch["known_mask"], denominator=denominator)
                mask = batch["valid_mask"] & ~batch["known_mask"]
                direct = ((baseline - batch["u"]).square()[mask]).sum() / denominator
                if not torch.equal(fm, direct):
                    raise AssertionError("Aux-off FM differs from original whole-batch formula")
                base_cpu, base_fm = baseline.detach().cpu(), float(fm.detach())
                _, bs = _call(budget, "preflight_backward", fm.backward, dict(ctx, arm="A", objective="FM"), actual_device)
            row = dict(sample_index=sample["sample_index"], input_sha256=sample["input_bytehash"],
                baseline_FM=base_fm, baseline_forward_seconds=fs, baseline_backward_seconds=bs,
                baseline_generator_grad_norm=_norm([p.grad for p in model.parameters()]),
                original_FM_formula_equal=True, domains={})
            del baseline, fm, direct
            previous_message = None
            for arm, domain in (("B", "free"), ("C", "known")):
                model.zero_grad(set_to_none=True)
                head.zero_grad(set_to_none=True)
                with AttentionCapture(model) as capture, native_execution(model, coordinates=(batch["xt"], batch["t"]), autocast=True):
                    velocity, fs = _call(budget, "preflight_forward", lambda: model(*model_inputs(batch)).float(),
                                         dict(ctx, arm=arm, capture=True), actual_device)
                    a = capture.get()
                    current_message = a.detach().cpu().clone()
                    if previous_message is not None and not torch.equal(previous_message, current_message):
                        raise AssertionError("B/C did not capture the same actual attention tensor")
                    previous_message = current_message
                    if not torch.equal(velocity.detach().cpu(), base_cpu):
                        raise AssertionError("Read-only capture/head changed generator prediction")
                    fm = masked_fm_loss(velocity, batch["u"], batch["valid_mask"], batch["known_mask"], denominator=denominator)
                    if float(fm.detach()) != base_fm:
                        raise AssertionError("FM changed with capture")
                    selected, mask = select_messages(a, batch["known_mask"], batch["valid_mask"], domain)
                    prediction = head(selected[0])
                    aux, statistics = alignment_loss(prediction, teacher)
                    early = observation_parameters(model)
                    late = {n:p for n,p in model.named_parameters() if n.startswith("backbone.layers.4.attn.")}
                    sources = [a, *early.values(), *late.values()]
                    grads, ab = _call(budget, "preflight_backward",
                        lambda: torch.autograd.grad(.025 * aux, sources, retain_graph=True, allow_unused=True),
                        dict(ctx, arm=arm, objective="auxiliary_connectivity"), actual_device)
                    grad_a = grads[0]
                    early_grads = grads[1:1+len(early)]
                    late_grads = grads[1+len(early):]
                    if grad_a is None or not bool((grad_a[~mask] == 0).all()) or not bool(torch.isfinite(grad_a).all()):
                        raise AssertionError("Unselected/padding message gradient is not zero")
                    if _norm(early_grads) == 0. or any(g is not None for g in late_grads):
                        raise AssertionError("Auxiliary early/late generator path differs")
                    total = fm + .025 * aux
                    _, cb = _call(budget, "preflight_backward", total.backward,
                        dict(ctx, arm=arm, objective="combined_FM_auxiliary"), actual_device)
                    head_norm = _norm([p.grad for p in head.parameters()])
                    generator_norm = _norm([p.grad for p in model.parameters()])
                    if head_norm <= 0. or generator_norm <= 0. or not math.isfinite(head_norm + generator_norm):
                        raise AssertionError("First combined backward has no head/generator gradient")
                    row["domains"][domain] = dict(capture_calls=capture.calls,
                        captured_shape=list(a.shape), captured_dtype=str(a.dtype),
                        message_sha256=state_hash({"message": current_message}),
                        generator_prediction_bitwise_equal=True, FM_bitwise_equal=True,
                        selected_tokens=int(mask.sum()), direct_unselected_gradient_zero=True,
                        auxiliary_early_gradient_norm=_norm(early_grads), auxiliary_late_all_unused=True,
                        selected_message_gradient_norm=float(grad_a[mask].float().norm()),
                        head_gradient_norm=head_norm, combined_generator_gradient_norm=generator_norm,
                        auxiliary_loss=float(aux.detach()), head_prediction_dtype=str(prediction.dtype),
                        forward_seconds=fs, auxiliary_backward_seconds=ab, combined_backward_seconds=cb,
                        **statistics)
                    del velocity, a, selected, prediction, aux, fm, total, grads, grad_a, early_grads, late_grads, sources
                if capture.tensor is not None or capture.handle is not None:
                    raise AssertionError("Capture graph/hook was not released")
            row.update(B_C_message_identical=True, capture_released=True,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(actual_device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(actual_device))
            result["rows"].append(row)
            atomic_json(result_path, result)
            del batch, teacher, previous_message
        validate_optimizer(model, None, optimizer, 0)
        if state_hash(model.state_dict()) != START_STATE_SHA256 or state_hash(head.state_dict()) != head_initial:
            raise AssertionError("Zero-update preflight changed a parameter")
        result.update(status="PASS", generator_parameters_unchanged=True, aligner_parameters_unchanged=True,
            old_optimizer_step=5000, optimizer_updates=0, auxiliary_head_preflight_forward=4,
            auxiliary_head_preflight_backward=8, all_generator_calls_accounted=True)
    except BaseException as error:
        result.update(status="IMPLEMENTATION_BLOCKED", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(wall_seconds=time.perf_counter()-began, budget_after=budget.snapshot())
        atomic_json(result_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explicit bounded contextual alignment preflight (GPU lock required)")
    for name in ("out", "start", "data", "stream-root", "teacher-manifest", "ledger"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    from .context_budget import Budget
    from ..runtime import gpu_worker
    with gpu_worker(args.ledger, "contextual_alignment_preflight"):
        run(args.out, args.start, str(Path(args.data) / "train_manifest.json") if Path(args.data).is_dir() else args.data, args.stream_root, args.teacher_manifest, Budget(args.ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
