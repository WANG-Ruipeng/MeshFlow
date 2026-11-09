"""Budgeted single-pass evaluation; no generator, training, new noise or GEO.

Controller dispatches five sequential jobs after checking resources:
  python -B -m r2.pilots.anchor_aware_denoiser_v1.inference synthetic --arm D0_OFFICIAL
  python -B -m r2.pilots.anchor_aware_denoiser_v1.inference real --arm D_CONT_NO_ROLE
  python -B -m r2.pilots.anchor_aware_denoiser_v1.inference synthetic --arm D_CONT_NO_ROLE
  python -B -m r2.pilots.anchor_aware_denoiser_v1.inference real --arm D_ANCHOR
  python -B -m r2.pilots.anchor_aware_denoiser_v1.inference synthetic --arm D_ANCHOR

Exactly 48+48 real and 18+18+18 synthetic authorized forward slots. D0 real
outputs are reused by the separate CPU evaluator and never run here. Completed
receipts are verified/skipped; unresolved forward starts are never replayed.
Each process owns one model and uses gpu_run.Phase's cumulative three-hour cap.
"""
from __future__ import annotations
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"
import argparse
import gc
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from . import common as C
from .gpu_run import Phase, memory, rng_hash, runtime
from .model import load_official_wrapper

EVAL_ARMS = ("D0_OFFICIAL", *C.ARMS)
EVAL_VERSION = "anchor_denoiser_singlepass_evaluation_v1"
LEDGER = "inference_ledger.jsonl"


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
        raise ValueError("Unsafe registered output identity")
    return value


def _checkpoint(arm: str) -> tuple[Path, str]:
    if arm == "D0_OFFICIAL":
        return C.ASSET, C.EXPECTED
    path = C.OUT / "checkpoints" / arm / "inference.pt"
    receipt = C.read(path.with_suffix(".json"))
    if receipt["status"] != "PASS" or receipt["step"] != 1000 or C.sha(path) != receipt["sha256"]:
        raise ValueError("Final inference checkpoint receipt/hash mismatch")
    return path, receipt["sha256"]


def _registered(stage: str, arm: str, checkpoint_path: Path, checkpoint_sha: str) -> dict:
    registration_path = C.OUT / "registration.json"
    main = C.read(registration_path)
    if stage == "real":
        rows = main["records"]
        if len(rows) != 48 or arm == "D0_OFFICIAL":
            raise ValueError("Only 48 real inputs per adapted arm are authorized")
        if len({(row["label"], row["request_id"]) for row in rows}) != 48:
            raise ValueError("Real evaluation identity duplicated")
    else:
        manifest_path = C.OUT / "stream/synthetic/manifest.json"
        if C.sha(manifest_path) != main["synthetic_manifest_sha256"]:
            raise ValueError("Synthetic inputs changed after registration")
        rows = C.read(manifest_path)["rows"]
        if len(rows) != 18 or len({row["synthetic_id"] for row in rows}) != 18:
            raise ValueError("Synthetic identity/count differs")
    reg = dict(version=EVAL_VERSION, stage=stage, arm=arm, records=rows,
        main_registration_sha256=C.sha(registration_path),
        inference_source_sha256=C.sha(Path(__file__)), wrapper_source_sha256=C.sha(C.CODE / "model.py"),
        phase_source_sha256=C.sha(C.CODE / "gpu_run.py"), common_source_sha256=C.sha(C.CODE / "common.py"),
        model_checkpoint_path=str(checkpoint_path), model_checkpoint_sha256=checkpoint_sha,
        forward_limit=48 if stage == "real" else 18,
        scale=C.SCALE, sigma_denoiser_units=C.SIGMA,
        math="BF16 PE/project/decoder, FP32 parameters and one FP32 input+residual; fixed inverse scale",
        known_mask="First K real/source faces; no padding, decoder mask=None",
        C_projection="Write original FP32 source C directly after inverse coordinate transform",
        postprocessing="None here; separate fixed GEO_C stage",
        source_reconstruction="Synthetic same-shape inputs only; never real generated RAW",
        inference_rng="No new draws; RNG state must remain unchanged",
        endpoint="Official EMA or exact adapted step1000; no intermediate selection")
    path = C.OUT / "eval_registrations" / f"{stage}_{arm}.json"
    if path.exists():
        if C.read(path) != reg:
            raise ValueError("Evaluation registration changed on resume")
    else:
        C.write(path, reg)
    reg["registration_path"] = str(path)
    reg["registration_sha256"] = C.sha(path)
    return reg


def _load(arm: str, checkpoint_path: Path, checkpoint_sha: str, stage: str):
    model, spec = load_official_wrapper(anchor=arm == "D_ANCHOR", activation_checkpointing=True,
                                       face_count_dropout=.1, verify_checkpoint_hash=True)
    if arm != "D0_OFFICIAL":
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        if saved["schema"] != "anchor_denoiser_inference_v1" or saved["arm"] != arm or saved["step"] != 1000:
            raise ValueError("Inference endpoint identity mismatch")
        identity = saved["identity"]
        if identity["official_sha256"] != C.EXPECTED or identity["registration_sha256"] != C.sha(C.OUT / "registration.json"):
            raise ValueError("Adapted model initialization/task identity mismatch")
        if identity["stream_sha256"] != C.sha(C.OUT / "stream/manifest.json"):
            raise ValueError("Adapted model stream mismatch")
        loaded = model.load_state_dict(saved["model"], strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError("Adapted wrapper load was not strict")
        spec["adapted_identity"] = identity
        del saved
    model.validate_parameters()
    model.eval().requires_grad_(False)
    spec.update(evaluation_arm=arm, evaluation_stage=stage, final_checkpoint_sha256=checkpoint_sha,
                final_checkpoint_path=str(checkpoint_path), eval=True, no_optimizer=True)
    C.write(C.OUT / "eval_load_checks" / f"{stage}_{arm}.json", spec)
    gc.collect()
    return model


def _dest(stage: str, arm: str, row: dict) -> Path:
    if stage == "real":
        return C.OUT / "inference" / arm / _safe_name(row["label"]) / _safe_name(row["request_id"])
    return C.OUT / "synthetic_results" / arm / _safe_name(row["synthetic_id"])


def _identity(stage: str, arm: str, row: dict, reg: dict) -> dict:
    common = dict(denoiser=arm, model_checkpoint_sha256=reg["model_checkpoint_sha256"],
                  evaluation_registration_sha256=reg["registration_sha256"])
    if stage == "real":
        return dict(**common, label=row["label"], request_id=row["request_id"], raw_sha256=row["raw_sha256"])
    return dict(**common, synthetic_id=row["synthetic_id"], condition_id=row["condition_id"],
                synthetic_input_sha256=row["sha256"])


def _key(stage: str, arm: str, row: dict) -> tuple[str, str, str]:
    return (stage, arm, row["label"] + "/" + row["request_id"] if stage == "real" else row["synthetic_id"])


def _verify_done(stage: str, arm: str, row: dict, reg: dict) -> dict | None:
    dest = _dest(stage, arm, row)
    receipt_path = dest / "receipt.json"
    output_path = dest / ("denoised.npz" if stage == "real" else "prediction.npz")
    if not receipt_path.exists():
        if output_path.exists():
            raise ValueError("Evaluation output exists without completed receipt; do not overwrite")
        return None
    receipt = C.read(receipt_path)
    if receipt["status"] != "PASS" or any(receipt.get(k) != v for k, v in _identity(stage, arm, row, reg).items()):
        raise ValueError("Completed evaluation identity differs")
    if C.sha(output_path) != receipt["output_sha256"]:
        raise ValueError("Completed evaluation output hash differs")
    with np.load(output_path, allow_pickle=False) as saved:
        k, n = int(row["K"]), int(row["N"])
        if saved["output"].shape != (n, 3, 3) or saved["output"].dtype != np.float32:
            raise ValueError("Completed output shape/precision differs")
        if saved["output"][:k].tobytes() != saved["C"].tobytes():
            raise ValueError("Completed output C bytes differ")
    return receipt


def _forward(phase: Phase, model, x_np: np.ndarray, k: int, official: bool, *, input_model_coordinates: bool = False):
    n = len(x_np)
    x = torch.from_numpy(x_np.reshape(1, n, 9)).cuda()
    if input_model_coordinates:
        x = x * C.SCALE  # preserve audited D0 GPU FP32 input scaling
    known = torch.arange(n, device="cuda").reshape(1, n) < k
    before = rng_hash()
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.inference_mode():
        residual = phase.call(model, x, known, official=official)
        proposal_tensor_d = x.float() + residual.float()
        # Keep inverse scale on the same GPU FP32 PyTorch path as audited D0.
        proposal_model = (proposal_tensor_d / C.SCALE).reshape(n, 3, 3).cpu().numpy().copy()
        proposal_d = proposal_tensor_d.reshape(n, 3, 3).cpu().numpy().copy()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    if before != rng_hash():
        raise ValueError("Eval forward consumed RNG")
    if proposal_d.dtype != np.float32 or not np.isfinite(proposal_d).all():
        raise ValueError("Nonfinite/non-FP32 inference proposal")
    residual_dtype = str(residual.dtype)
    del x, known, residual, proposal_tensor_d
    return proposal_d, proposal_model, elapsed, residual_dtype


def _real(phase: Phase, model, arm: str, row: dict, reg: dict) -> dict:
    if C.sha(row["raw_path"]) != row["raw_sha256"]:
        raise ValueError("Protected RAW changed")
    n, k = int(row["N"]), int(row["K"])
    with np.load(row["raw_path"], allow_pickle=False) as saved:
        raw, known_c = saved["raw"].reshape(n, 3, 3).copy(), saved["C"].reshape(k, 3, 3).copy()
    if raw.dtype != np.float32 or known_c.dtype != np.float32 or raw[:k].tobytes() != known_c.tobytes():
        raise ValueError("Original RAW/C precision or identity differs")
    identity = _identity("real", arm, row, reg)
    C.event("EVAL_FORWARD_STARTED", ledger=LEDGER, stage="real", arm=arm, key=list(_key("real", arm, row)), **identity)
    proposal_d, proposal, elapsed, dtype = _forward(phase, model, raw, k, official=False, input_model_coordinates=True)
    output = proposal.copy()
    output[:k] = known_c
    displacement = output.astype(np.float64) - raw.astype(np.float64)
    dest = _dest("real", arm, row)
    output_path = dest / "denoised.npz"
    C.save_npz(output_path, output=output, C=known_c, proposal=proposal,
               displacement_from_RAW=displacement, known_faces=np.asarray(k), input_faces=np.asarray(n),
               coordinate_scale=np.asarray(C.SCALE))
    with np.load(output_path, allow_pickle=False) as reopened:
        if reopened["output"].tobytes() != output.tobytes() or reopened["output"][:k].tobytes() != known_c.tobytes():
            raise ValueError("Real output roundtrip/C byte contract failed")
    c_move = np.linalg.norm(proposal[:k].astype(np.float64) - known_c.astype(np.float64), axis=-1)
    free_move = np.linalg.norm(displacement[k:], axis=-1)
    receipt = dict(status="PASS", **identity, N=n, K=k, output_path=str(output_path),
        output_sha256=C.sha(output_path), seconds=elapsed, C_original_bits_restored=True,
        no_rng_consumed=True, output_FP32=True, residual_dtype=dtype,
        C_proposal_mean_displacement_model=float(c_move.mean()), C_proposal_max_displacement_model=float(c_move.max()),
        mean_free_displacement_model=float(free_move.mean()), max_free_displacement_model=float(free_move.max()),
        source_reconstruction_metric="NOT_APPLICABLE_GENERATED_RAW", vram=memory())
    C.write(dest / "receipt.json", receipt)
    C.event("EVAL_FORWARD_COMPLETE", ledger=LEDGER, stage="real", arm=arm, key=list(_key("real", arm, row)), **receipt)
    return receipt


def _error_stats(delta: np.ndarray) -> dict:
    count = int(delta.size)
    if count == 0:
        return dict(sse=None, mse=None, coordinate_rmse=None, corner_l2_mean=None,
                    corner_l2_median=None, corner_l2_p95=None, corner_l2_max=None, scalars=0, corners=0)
    d = delta.astype(np.float64)
    distance = np.linalg.norm(d, axis=-1)
    sse = float(np.square(d).sum())
    return dict(sse=sse, mse=sse/count, coordinate_rmse=float(np.sqrt(sse/count)),
        corner_l2_mean=float(distance.mean()), corner_l2_median=float(np.median(distance)),
        corner_l2_p95=float(np.percentile(distance, 95)), corner_l2_max=float(distance.max()),
        scalars=count, corners=int(distance.size))


def _synthetic(phase: Phase, model, arm: str, row: dict, reg: dict) -> dict:
    if C.sha(row["path"]) != row["sha256"]:
        raise ValueError("Frozen synthetic input changed")
    with np.load(row["path"], allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    n, k = int(row["N"]), int(row["K"])
    x_d, target_d, known_c = data["x_d"], data["target_d"], data["C"]
    if x_d.dtype != np.float32 or x_d.shape != (n, 3, 3) or x_d[:k].tobytes() != target_d[:k].tobytes():
        raise ValueError("Synthetic coordinate/known input contract differs")
    identity = _identity("synthetic", arm, row, reg)
    C.event("EVAL_FORWARD_STARTED", ledger=LEDGER, stage="synthetic", arm=arm, key=list(_key("synthetic", arm, row)), **identity)
    proposal_d, proposal, elapsed, dtype = _forward(phase, model, x_d, k, official=arm == "D0_OFFICIAL")
    restored_d = proposal_d.copy()
    restored_d[:k] = x_d[:k]
    output = proposal.copy()
    output[:k] = known_c
    # FP64 evaluation of saved FP32 predictions; this is the same unweighted
    # scalar-MSE definition, not necessarily bit-identical training reduction.
    delta = proposal_d.astype(np.float64) - target_d.astype(np.float64)
    input_delta = x_d.astype(np.float64) - target_d.astype(np.float64)
    free = _error_stats(delta[k:])
    original = _error_stats(input_delta[k:])
    indices = data["supported_Y_local_corner_indices"].astype(np.int64)
    interface = _error_stats(delta[k:][indices[:, 0], indices[:, 1]])
    interface_original = _error_stats(input_delta[k:][indices[:, 0], indices[:, 1]])
    movement = _error_stats(proposal_d[k:].astype(np.float64) - x_d[k:].astype(np.float64))
    c_movement = _error_stats(proposal_d[:k].astype(np.float64) - x_d[:k].astype(np.float64))
    metrics = dict(denoiser=arm, synthetic_id=row["synthetic_id"], condition_id=row["condition_id"],
        uid=row["uid"], region_type=row["region_type"], tier=row["tier"], N=n, K=k,
        clean=row["clean"], noise_index=row["noise_index"], sigma_denoiser_units=C.SIGMA,
        C_original_bits_restored=True, units="denoiser coordinates", scope="synthetic_same_shape_only",
        free_sse=free["sse"], free_mse=free["mse"], free_scalar_denominator=free["scalars"],
        free_coordinate_rmse=free["coordinate_rmse"], free_corner_l2_mean=free["corner_l2_mean"],
        free_corner_l2_p95=free["corner_l2_p95"], input_free_mse=original["mse"],
        supported_interface_corners=interface["corners"], supported_interface_sse=interface["sse"],
        supported_interface_mse=interface["mse"], supported_interface_corner_l2_mean=interface["corner_l2_mean"],
        supported_interface_corner_l2_p95=interface["corner_l2_p95"],
        input_supported_interface_mse=interface_original["mse"],
        free_movement_corner_l2_mean=movement["corner_l2_mean"], free_movement_corner_l2_max=movement["corner_l2_max"],
        clean_unnecessary_movement_mse=movement["mse"] if row["clean"] else None,
        clean_unnecessary_movement_corner_l2_mean=movement["corner_l2_mean"] if row["clean"] else None,
        C_unconstrained_movement_corner_l2_mean=c_movement["corner_l2_mean"],
        C_unconstrained_movement_corner_l2_max=c_movement["corner_l2_max"],
        interface_status="SOURCE_SUPPORTED_REFERENCE" if interface["corners"] else "NA_NO_VERIFIED_SOURCE_INTERFACE")
    dest = _dest("synthetic", arm, row)
    output_path = dest / "prediction.npz"
    C.save_npz(output_path, output=output, C=known_c, proposal=proposal,
               proposal_denoiser=proposal_d, restored_denoiser=restored_d,
               free_error_denoiser=delta[k:], supported_interface_error_denoiser=delta[k:][indices[:, 0], indices[:, 1]])
    with np.load(output_path, allow_pickle=False) as reopened:
        if reopened["output"].tobytes() != output.tobytes() or reopened["output"][:k].tobytes() != known_c.tobytes():
            raise ValueError("Synthetic output roundtrip/C byte contract failed")
    receipt = dict(status="PASS", **identity, N=n, K=k, output_path=str(output_path),
        output_sha256=C.sha(output_path), seconds=elapsed, residual_dtype=dtype, metrics=metrics,
        no_rng_consumed=True, C_original_bits_restored=True, vram=memory())
    C.write(dest / "receipt.json", receipt)
    C.event("EVAL_FORWARD_COMPLETE", ledger=LEDGER, stage="synthetic", arm=arm, key=list(_key("synthetic", arm, row)),
            output_sha256=receipt["output_sha256"], seconds=elapsed, **identity)
    return receipt


def summarize_synthetic() -> dict:
    rows = []
    for arm in EVAL_ARMS:
        arm_rows = [C.read(path)["metrics"] for path in sorted((C.OUT / "synthetic_results" / arm).glob("*/receipt.json"))]
        if arm_rows:
            C.csvwrite(C.OUT / "synthetic_results" / arm / "per_input.csv", arm_rows)
            rows.extend(arm_rows)
    if not rows:
        return dict(status="NOT_RUN", outputs=0)
    C.csvwrite(C.OUT / "synthetic_results/per_input.csv", rows)
    summaries = []
    for arm in EVAL_ARMS:
        for clean in (True, False):
            group = [row for row in rows if row["denoiser"] == arm and row["clean"] == clean]
            if not group:
                continue
            interface_rows = [row for row in group if row["supported_interface_corners"]]
            summaries.append(dict(denoiser=arm, input_kind="clean" if clean else "noisy", count=len(group),
                free_mse_macro=float(np.mean([row["free_mse"] for row in group])),
                free_mse_pooled=sum(row["free_sse"] for row in group)/sum(row["free_scalar_denominator"] for row in group),
                input_free_mse_macro=float(np.mean([row["input_free_mse"] for row in group])),
                supported_interface_mse_macro=float(np.mean([row["supported_interface_mse"] for row in interface_rows])) if interface_rows else None,
                supported_interface_mse_pooled=sum(row["supported_interface_sse"] for row in interface_rows)/(3*sum(row["supported_interface_corners"] for row in interface_rows)) if interface_rows else None,
                free_movement_corner_l2_mean_macro=float(np.mean([row["free_movement_corner_l2_mean"] for row in group])),
                C_byte_exact_count=sum(row["C_original_bits_restored"] for row in group)))
    C.csvwrite(C.OUT / "synthetic_results/summary.csv", summaries)
    status = dict(status="COMPLETE" if len(rows) == 54 else "PARTIAL", outputs=len(rows),
                  expected_outputs=54, source_GT_scope="synthetic same-shape denoising only")
    C.write(C.OUT / "synthetic_results/aggregation.json", status)
    return status


def run(stage: str, arm: str) -> dict:
    if stage not in ("real", "synthetic") or arm not in EVAL_ARMS or (stage == "real" and arm == "D0_OFFICIAL"):
        raise ValueError("Unauthorized evaluation stage/arm")
    checkpoint_path, checkpoint_sha = _checkpoint(arm)
    reg = _registered(stage, arm, checkpoint_path, checkpoint_sha)
    existing, pending = [], []
    for row in reg["records"]:
        receipt = _verify_done(stage, arm, row, reg)
        (existing if receipt is not None else pending).append(receipt if receipt is not None else row)
    if not pending:
        if stage == "synthetic":
            summarize_synthetic()
        return dict(status="SKIP_VERIFIED_COMPLETE", stage=stage, arm=arm, outputs=len(existing), new_forwards=0)
    history_path = C.OUT / LEDGER
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line] if history_path.exists() else []
    starts = {tuple(event["key"]) for event in history if event["event"] == "EVAL_FORWARD_STARTED"}
    if any(_key(stage, arm, row) in starts for row in pending):
        raise RuntimeError("Unresolved evaluation forward START; no automatic replay")
    prior_attempts = sum(1 for event in history if event["event"] == "EVAL_FORWARD_STARTED" and event["stage"] == stage and event["arm"] == arm)
    if prior_attempts != len(existing) or prior_attempts + len(pending) != reg["forward_limit"]:
        raise ValueError("Evaluation slots cannot be reconciled with complete receipts")
    prefix = f"{stage}_{arm}"
    previous_phases = sorted((C.OUT / "gpu_phases").glob(prefix + "*.json"))
    if previous_phases:
        if any(C.read(path)["status"] not in ("GPU_TIME_LIMIT", "PAUSED_BOUNDARY") for path in previous_phases):
            raise RuntimeError("Prior evaluation phase was not cleanly paused; do not retry")
        phase_name = prefix + "__resume" + str(len(existing))
    else:
        phase_name = prefix
    phase = Phase(phase_name)
    model = None
    try:
        before = C.read(C.OUT / "resources" / f"{phase_name}_before.json")
        fields = before["gpu"]["stdout"].strip().splitlines()[0].split(",")
        if float(fields[3]) < phase.headroom + 2400:
            raise RuntimeError("RESOURCE_BLOCKED: measured inference estimate plus headroom unavailable")
        model = _load(arm, checkpoint_path, checkpoint_sha, stage)
        torch.cuda.reset_peak_memory_stats()
        model.cuda()
        torch.cuda.synchronize()
        C.write(C.OUT / "resources" / f"{phase_name}_resident.json", C.resources())
        for index, row in enumerate(pending):
            if phase.remaining() < 30:
                status = dict(status="GPU_TIME_LIMIT", stage=stage, arm=arm, outputs=len(existing),
                              new_forwards=phase.f, expected_outputs=reg["forward_limit"])
                C.write(C.OUT / "eval_completion" / f"{stage}_{arm}.json", status)
                phase.finish("GPU_TIME_LIMIT")
                return status
            receipt = _real(phase, model, arm, row, reg) if stage == "real" else _synthetic(phase, model, arm, row, reg)
            existing.append(receipt)
            if index == 0:
                C.write(C.OUT / "resources" / f"{phase_name}_first_actual_forward.json",
                        dict(resources=C.resources(), vram=memory(), seconds=receipt["seconds"], workers=1,
                             scope="Resident FP32 denoiser, one BF16 forward, no optimizer/generator"))
        model.validate_parameters()
        if len(existing) != reg["forward_limit"] or phase.f != len(pending):
            raise ValueError("Evaluation forward/output budget differs")
        if stage == "synthetic":
            summarize_synthetic()
        result = dict(status="COMPLETE", stage=stage, arm=arm, outputs=len(existing), new_forwards=phase.f,
            total_attempts=prior_attempts+phase.f, expected_outputs=reg["forward_limit"],
            model_checkpoint_sha256=checkpoint_sha, evaluation_registration_sha256=reg["registration_sha256"],
            all_C_byte_exact=True, all_no_rng_consumed=True, GPU_workers=1, vram=memory(),
            source_reconstruction_metrics=("synthetic_only" if stage == "synthetic" else "NOT_APPLICABLE"))
        C.write(C.OUT / "eval_completion" / f"{stage}_{arm}.json", result)
        phase.finish("COMPLETE")
        return result
    except BaseException as error:
        C.event("EVALUATION_FAILED", ledger=LEDGER, stage=stage, arm=arm, error=repr(error),
                completed_outputs=len(existing), phase_forward_attempts=phase.f)
        phase.finish("FAILED", repr(error))
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("real", "synthetic", "summarize"))
    parser.add_argument("--arm", choices=EVAL_ARMS)
    args = parser.parse_args()
    if args.stage == "summarize":
        result = summarize_synthetic()
    else:
        if args.arm is None:
            parser.error("--arm is required for inference")
        runtime()
        result = run(args.stage, args.arm)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
