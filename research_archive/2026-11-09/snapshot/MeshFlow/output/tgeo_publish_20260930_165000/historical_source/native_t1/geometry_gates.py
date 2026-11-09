"""Bounded pre-training gates for the registered geometry-context pilot.

Importing performs no IO, model construction, CUDA work, or training. The
explicit runner uses 13 single-step forwards, two backwards with no optimizer,
and one 50-step complete rollout. It never retries with relaxed precision.
"""
from .runtime import configure_stable_runtime
from pathlib import Path
import time
import json
import numpy as np
import torch

from .artifacts import atomic_json, array_hash, file_sha256, state_sha256
from .baseline import BASELINE_FILE_SHA256
from .data import load_archive, native_collate, model_inputs
from .model import native_execution, assert_native_runtime
from .training import masked_fm_loss
from .sampling import clamped_sample, make_noise, new_counts
from .context_geometry import (attach_context_encoder, ContextGeometryObserver,
                               FORMULA_ATOL, FORMULA_RTOL)
from .portable_checkpoint import validate_training_metadata

REPO = Path(__file__).resolve().parents[1]
HISTORICAL_REPLAY = REPO / "experiments/mf_h1_local0/outputs/a_then_fc_20260930_001700/raw/A_continue_P0_K12_seed9601.npz"
BF16_ATOL, BF16_RTOL = .015625, .03125
FORWARD_CAP, BACKWARD_CAP, ROLLOUT_CAP = 63, 2, 1
GATE_POLICY = "baseline_regression_and_branch_inheritance_v2"
STRICT_GATE_POLICY = "strict_permutation_v1"
BASELINE_EVIDENCE_RESULTS = REPO / "native_t1/runs/geometry_gate_code_audit_20260930_150400/results.json"
BASELINE_EVIDENCE_VELOCITIES = BASELINE_EVIDENCE_RESULTS.with_name("velocities.npz")


def load_baseline_regression_evidence(contract, identity, inputs_sha256):
    """Read and verify the explicitly registered evidence; no model calls.

    A numerical failure is never exempted because its magnitude seems small.
    Authorization requires the fixed external code-regression case, matching
    assets/code/state, and its saved exact baseline velocities.
    """
    if contract.get("gate_policy") != GATE_POLICY:
        raise ValueError("Explicit baseline-regression gate policy is required")
    declared = contract.get("baseline_regression_evidence")
    required = ("results_path", "results_sha256", "velocities_path", "velocities_sha256")
    if not isinstance(declared, dict) or any(not declared.get(key) for key in required):
        raise ValueError("Registered baseline-regression evidence is missing")
    results_path, velocities_path = Path(declared["results_path"]), Path(declared["velocities_path"])
    if (results_path.resolve() != BASELINE_EVIDENCE_RESULTS.resolve()
            or velocities_path.resolve() != BASELINE_EVIDENCE_VELOCITIES.resolve()):
        raise ValueError("Baseline evidence is not the fixed authorized code audit")
    for path, key in ((results_path, "results_sha256"), (velocities_path, "velocities_sha256")):
        if file_sha256(path) != declared[key]:
            raise ValueError("Baseline evidence file SHA256 differs: " + key)
    evidence = json.loads(results_path.read_text(encoding="utf-8-sig"))
    if evidence.get("status") != "DIAGNOSIS_COMPLETED_EXPERIMENT_REMAINS_STOPPED":
        raise ValueError("External baseline audit is not a completed diagnosis")
    registration = evidence.get("registration", {})
    current_model_sha = file_sha256(REPO / "native_t1/model.py")
    if (registration.get("source_commit") != "2502d5f89c54706cc4700d3cdea4a12fd6792529"
            or registration.get("checkpoint_sha256") != identity.get("checkpoint_sha256")
            or registration.get("inputs_sha256") != inputs_sha256
            or registration.get("source_pins", {}).get("native_t1/model.py") != current_model_sha):
        raise ValueError("External baseline checkpoint/input/model.py identity differs")
    if evidence.get("initialization", {}).get("checkpoint_sha256") != identity.get("checkpoint_sha256"):
        raise ValueError("External loaded checkpoint identity differs")
    state = identity.get("state_sha256")
    if not state or any(evidence.get(key) != state for key in (
            "state_before", "state_after", "base_wrapper_state_sha256", "base_wrapper_state_after")):
        raise ValueError("External baseline parameter state differs")
    if any(evidence.get(key) is not True for key in (
            "parameters_unchanged", "source_unchanged", "fixed_batch_hashes_match_failed_gate")):
        raise ValueError("External baseline audit did not preserve source/state/fixed inputs")
    proof = evidence.get("velocity_proof", {})
    if (proof.get("sha256") != declared["velocities_sha256"]
            or Path(proof.get("path", "")).resolve() != velocities_path.resolve()):
        raise ValueError("External velocity_proof does not match registered NPZ")
    for label in ("reference", "combined"):
        comparison = evidence.get("current_vs_base_commit", {}).get(label, {})
        if comparison.get("bitwise") is not True or comparison.get("finite") is not True:
            raise ValueError("External current/base-commit regression is not exact: " + label)
    values = {}
    with np.load(velocities_path, allow_pickle=False) as stored:
        for label in ("reference", "combined"):
            base, current = "base_commit_"+label, "current_"+label
            if base not in stored or current not in stored:
                raise ValueError("Required baseline/current velocity is absent")
            a, b = stored[base], stored[current]
            if (a.shape != (1,112,9) or b.shape != a.shape or a.dtype != np.float32
                    or b.dtype != np.float32 or not np.isfinite(a).all() or not np.isfinite(b).all()
                    or not np.array_equal(a.view(np.uint32), b.view(np.uint32))):
                raise ValueError("Saved current/base-commit velocity tensors are not finite and exact")
            values[label] = torch.from_numpy(a.copy())
    audit = dict(status="VERIFIED", policy=GATE_POLICY, **declared,
                 checkpoint_sha256=identity["checkpoint_sha256"], inputs_sha256=inputs_sha256,
                 model_source_sha256=current_model_sha, model_state_sha256=state,
                 current_vs_base_commit=evidence["current_vs_base_commit"],
                 saved_current_vs_base_bitwise=True, existing_diagnostic_counts=evidence.get("counts"),
                 external_counts_included_in_this_gate_budget=False)
    return values, audit


def fixed_batch(archive, *, device="cuda"):
    """Fixed no-OT P0/K12 diagnostic/reload state, independent of training RNG.

    Ordered free GT is used solely as a gate's FM target. The model receives
    only model_inputs(batch), never this supervision or source identities.
    """
    case = archive.cases[0, 12]
    noise = torch.randn((100, 9), generator=torch.Generator(device="cpu").manual_seed(12345),
                        dtype=torch.float32).numpy()
    x0 = np.zeros((112, 9), dtype=np.float32)
    x1 = np.zeros_like(x0)
    x0[12:] = noise
    x1[12:] = case.free_target
    context = np.zeros_like(x0)
    context[:12] = case.constraints.reshape(12, 9)
    known = np.arange(112) < 12
    sample = dict(x0=x0, x1=x1, u=x1-x0, t=np.asarray(.5, dtype=np.float32),
                  y=np.asarray(112, dtype=np.int64), known_mask=known,
                  valid_mask=np.ones(112, dtype=bool), context=context)
    return native_collate([sample], device=device)


def _permuted_batch(batch):
    """A fixed face permutation and all six S3 corner permutations, no OT."""
    generator = torch.Generator(device="cpu").manual_seed(24680)
    order = torch.randperm(112, generator=generator).to(batch["xt"].device)
    six = torch.tensor([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]],
                       device=order.device)
    corner = six[torch.arange(112, device=order.device) % 6]
    def permute(value):
        selected = value[:, order].reshape(1, 112, 3, 3)
        return selected.gather(2, corner[None, :, :, None].expand(1,112,3,3)).reshape(1,112,9)
    result = dict(batch)
    for key in ("x0", "x1", "u", "xt", "context"):
        result[key] = permute(batch[key])
    for key in ("known_mask", "valid_mask", "loss_mask"):
        result[key] = batch[key][:, order]
    return result, order, corner, permute


def _comparison_metrics(actual, expected, *, bitwise=False, atol=BF16_ATOL, rtol=BF16_RTOL):
    from .geometry_pilot import exact_tensor
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = actual.float() - expected.float()
    exact = exact_tensor(actual, expected)
    close = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
    return dict(finite=finite, bitwise=exact, allclose=close,
                max_abs=float(difference.abs().max()), RMS=float(difference.double().square().mean().sqrt()),
                violating_elements=int((difference.abs() > (atol+rtol*expected.float().abs())).sum()),
                elements=actual.numel(), atol=0. if bitwise else atol, rtol=0. if bitwise else rtol,
                status="PASS" if finite and (exact if bitwise else close) else "FAIL")


def _comparison(actual, expected, *, bitwise=False, atol=BF16_ATOL, rtol=BF16_RTOL):
    row = _comparison_metrics(actual, expected, bitwise=bitwise, atol=atol, rtol=rtol)
    if row["status"] != "PASS":
        raise RuntimeError("Fixed numerical gate failed: " + str(row))
    return row


def _permutation_diagnostic(actual, expected, *, evidence):
    """Keep the original BF16 tolerance result; finite remains a hard gate."""
    if (not isinstance(evidence, dict) or evidence.get("status") != "VERIFIED"
            or evidence.get("policy") != GATE_POLICY):
        raise ValueError("Verified baseline evidence is required for numerical diagnostics")
    row = _comparison_metrics(actual, expected)
    if not row["finite"]:
        raise RuntimeError("Nonfinite full-model permutation remains a hard failure")
    row.update(site="full_model_bf16_permutation", blocking=False,
               classification="baseline_numerical_behavior_with_exact_branch_inheritance",
               evidence_results_sha256=evidence["results_sha256"])
    return row


def _permutation_check(actual, expected, *, policy=STRICT_GATE_POLICY, evidence=None):
    """Default is the original hard BF16 threshold; v2 must be explicit."""
    if policy == STRICT_GATE_POLICY:
        row = _comparison(actual, expected)
        row.update(site="full_model_bf16_permutation", blocking=True,
                   classification="strict_historical_permutation_gate")
        return row
    if policy == GATE_POLICY:
        return _permutation_diagnostic(actual, expected, evidence=evidence)
    raise ValueError("Unrecognized geometry gate policy")


class _Observer:
    def __init__(self, model, batch):
        self.model, self.batch = model, batch
        self.diagnostics = ContextGeometryObserver()
        self.geometry = []
        self.runtime = None
        self.injection = None

    def __call__(self, record):
        if record["site"] == "geometry_start":
            self.runtime = assert_native_runtime(self.model, coordinates=self.batch["xt"])
        elif record["site"] == "geometry":
            self.geometry.append(record)
        elif record["site"] == "geometry_injection":
            self.injection = record
        self.diagnostics(record)

    def report(self):
        if self.runtime is None or self.injection is None or len(self.geometry) != 1:
            raise RuntimeError("Expected exactly one complete geometry observation")
        result = self.diagnostics.as_dict()
        for row in result["injections"]:
            if not row["free_direct_delta_exact_zero"] or not row["free_hidden_unchanged"]:
                raise RuntimeError("Direct context injection reached a free face")
            if row["residual_dtype"] != "torch.float32":
                raise RuntimeError("Geometry residual left the fixed FP32 path")
        result["runtime"] = self.runtime
        result["known_slots"] = self.geometry[0]["known_indices"].cpu().tolist()
        result["feature_shape"] = list(self.geometry[0]["phi"].shape)
        return result


def _backward(model, batch, counts):
    """One counted real FM backward inside the same strict execution context."""
    model.zero_grad(set_to_none=True)
    model.set_trainable(True).eval()
    if counts["forward_attempts"] >= 13 or counts["backward_attempts"] >= BACKWARD_CAP:
        raise RuntimeError("Single-step gate budget exhausted")
    with native_execution(model, coordinates=batch["xt"]) as runtime:
        counts["forward_attempts"] += 1
        velocity = model(*model_inputs(batch)).float()
        counts["forward_returns"] += 1
        loss = masked_fm_loss(velocity, batch["u"], batch["valid_mask"], batch["known_mask"])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite zero-update FM gate")
        counts["backward_attempts"] += 1
        loss.backward()
        counts["backward_returns"] += 1
    gradients = {}
    for name, parameter in model.context_encoder.named_parameters():
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Missing/nonfinite geometry gradient: " + name)
        row = dict(norm=float(gradient.double().norm()), nonzero=int(torch.count_nonzero(gradient)),
                   elements=gradient.numel(), dtype=str(gradient.dtype))
        if name == "W_out.weight":
            if row["nonzero"] == 0:
                raise RuntimeError("Zero outlet receives no real FM gradient")
        elif row["nonzero"] != 0:
            raise RuntimeError("Unexpected upstream encoder gradient at exact zero outlet")
        gradients[name] = row
    present = [p.grad for p in model.parameters() if p.grad is not None]
    if not present or not bool(torch.stack([torch.isfinite(g).all() for g in present]).all()):
        raise RuntimeError("Nonfinite full-model gradient")
    row = dict(status="PASS", loss=float(loss.detach()), free_coordinate_denominator=900,
               encoder_gradients=gradients, full_model_gradient_tensors=len(present),
               full_model_gradient_elements=sum(g.numel() for g in present),
               all_present_gradients_finite=True, optimizer_constructed=False,
               optimizer_updates=0, runtime=runtime)
    model.zero_grad(set_to_none=True)
    model.set_trainable(False).eval()
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Gate gradients were not cleared")
    del velocity, loss, present, gradient
    return row


def run_gates(args):
    from .geometry_pilot import load_initial, recorded_forward, exact_tensor, free_memory, read_json
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "gates.json"
    if destination.exists():
        raise FileExistsError("Existing gate run is never overwritten or retried: " + str(destination))
    contract = read_json(out / "registration.json")
    policy = contract.get("gate_policy", STRICT_GATE_POLICY)
    if policy not in (STRICT_GATE_POLICY, GATE_POLICY):
        raise ValueError("Unrecognized registered geometry gate policy")
    if policy == STRICT_GATE_POLICY and contract.get("baseline_regression_evidence") is not None:
        raise ValueError("External evidence requires the explicitly registered v2 policy")
    if (contract.get("bf16_permutation_atol") != BF16_ATOL
            or contract.get("bf16_permutation_rtol") != BF16_RTOL
            or contract.get("zero_repeat_bitwise_required") is not True):
        raise ValueError("Registered gate thresholds differ; no fallback is allowed")
    single = dict(forward_attempts=0, forward_returns=0, backward_attempts=0,
                  backward_returns=0, optimizer_updates=0)
    sample_counts = new_counts()
    result = dict(status="RUNNING", schema="native_t1_geometry_gate_v1",
                  thresholds=dict(bf16_atol=BF16_ATOL, bf16_rtol=BF16_RTOL,
                                  feature_atol=FORMULA_ATOL, feature_rtol=FORMULA_RTOL,
                                  zero_repeat_observer_bitwise=True),
                  budget=dict(single_step_forwards=13, full_rollouts=1, total_forwards=63,
                              model_backwards=2, optimizer_updates=0, OT_calls=0),
                  diagnostic_counts=single, replay_counts=sample_counts,
                  counts={}, arms={}, proofs={}, OT_calls=0, policy=policy,
                  hard_wiring_status="RUNNING", recorded_numerical_failures=[],
                  numerical_diagnostics_status="NOT_RUN",
                  pass_meaning=("Hard baseline-regression/branch-wiring gates passed; recorded BF16 permutation failures remain failures"
                                if policy == GATE_POLICY else "All original hard wiring and BF16 permutation gates passed"))
    model = None
    proof = {}
    tick = time.perf_counter()
    def persist():
        if result["recorded_numerical_failures"]:
            result["numerical_diagnostics_status"] = "HAS_RECORDED_FAILURES"
        result["counts"] = dict(
            forward_attempts=single["forward_attempts"]+sample_counts["model_forward_attempts"],
            forward_returns=single["forward_returns"]+sample_counts["model_forward_returns"],
            backward_attempts=single["backward_attempts"], backward_returns=single["backward_returns"],
            optimizer_updates=0, rollout_attempts=sample_counts["rollout_attempts"],
            complete_rollouts=sample_counts["complete_rollouts"], OT_calls=0)
        result["seconds"] = time.perf_counter()-tick
        atomic_json(destination, result)
    def forward(batch, label):
        if single["forward_attempts"] >= 13:
            raise RuntimeError("13 single-step forward cap exceeded")
        value = recorded_forward(model, batch, single)
        proof[label] = value.detach().cpu().numpy().copy()
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("Nonfinite single-step gate velocity")
        return value
    persist()
    try:
        archive = load_archive(args.inputs, expected_sha256=contract["inputs_sha256"])
        model, identity = load_initial(args)
        if identity["checkpoint_sha256"] != contract["initialization_file_sha256"]:
            raise RuntimeError("Initialization changed after registration")
        result["initialization"] = identity
        external_velocities = None
        if policy == GATE_POLICY:
            external_velocities, external_audit = load_baseline_regression_evidence(contract, identity, archive.sha256)
        else:
            external_audit = dict(status="NOT_SUPPLIED", policy=STRICT_GATE_POLICY,
                reason="Strict historical permutation gating; no external baseline exemption is used")
        result["baseline_regression_evidence"] = external_audit
        before = state_sha256(model)
        result["pure_parameter_hash_before"] = before
        batch = fixed_batch(archive)
        permuted, order, corners, permute = _permuted_batch(batch)
        result["fixed_batch"] = dict(parent=0, K=12, time=.5, CPU_noise_seed=12345, OT=False,
            ordered_free_target=True, independent_of_training_stream=True,
            hashes={key:array_hash(value) for key,value in batch.items() if torch.is_tensor(value)},
            permutation_seed=24680, face_permutation=order.cpu().tolist(),
            corner_permutations=corners.cpu().tolist())
        # Schema-only rejection does not construct a model or read another checkpoint.
        try:
            validate_training_metadata(dict(schema="native_t1_geom_context_v1"))
        except ValueError:
            result["original_portable_loader_rejects_geometry_schema"] = True
        else:
            raise RuntimeError("Original portable identity validator accepted the new schema")
        original = forward(batch, "T0_reference")
        repeat = forward(batch, "T0_repeat")
        base_permuted = forward(permuted, "T0_permuted")
        result["arms"]["T0"] = dict(
            repeat=_comparison(repeat, original, bitwise=True),
            context_encoder_is_none=model.context_encoder is None)
        if external_velocities is not None:
            result["arms"]["T0"]["baseline_regression"] = dict(
                reference=_comparison(original.cpu(), external_velocities["reference"], bitwise=True),
                repeat=_comparison(repeat.cpu(), external_velocities["reference"], bitwise=True),
                combined=_comparison(base_permuted.cpu(), external_velocities["combined"], bitwise=True))
        else:
            result["arms"]["T0"]["baseline_regression"] = dict(status="NOT_SUPPLIED", reason=external_audit["reason"])
        result["arms"]["T0"]["permutation"] = _permutation_check(
            base_permuted, permute(original), policy=policy, evidence=external_audit)
        if result["arms"]["T0"]["permutation"]["status"] == "FAIL":
            result["recorded_numerical_failures"].append(dict(arm="T0", **result["arms"]["T0"]["permutation"]))
        persist()
        initial_encoder_hash = None
        for mode in ("geo", "graph"):
            arm = "T_"+mode
            row = result["arms"].setdefault(arm, {})
            row["context"] = attach_context_encoder(model, mode, seed=1010)
            encoder_hash = state_sha256(model.context_encoder)
            if initial_encoder_hash is None:
                initial_encoder_hash = encoder_hash
            if encoder_hash != initial_encoder_hash:
                raise RuntimeError("Geo and graph seeded parameter tensors differ")
            row["encoder_initial_sha256"] = encoder_hash
            branch_before = state_sha256(model)
            row["parameter_hash_before"] = branch_before
            first = forward(batch, arm+"_zero")
            second = forward(batch, arm+"_repeat")
            row["zero_vs_T0"] = _comparison(first, original, bitwise=True)
            row["repeat"] = _comparison(second, first, bitwise=True)
            observation = _Observer(model, batch)
            model.context_observer = observation
            observed = forward(batch, arm+"_observed")
            model.context_observer = None
            row["observer_on_off"] = _comparison(observed, first, bitwise=True)
            row["observation"] = observation.report()
            shuffled = _Observer(model, permuted)
            model.context_observer = shuffled
            permuted_value = forward(permuted, arm+"_permuted")
            model.context_observer = None
            row["zero_vs_T0_permuted"] = _comparison(permuted_value, base_permuted, bitwise=True)
            row["permutation"] = _permutation_check(permuted_value, permute(first), policy=policy, evidence=external_audit)
            if row["permutation"]["status"] == "FAIL":
                result["recorded_numerical_failures"].append(dict(arm=arm, **row["permutation"]))
            row["scattered_slots_observation"] = shuffled.report()
            before_geom, after_geom = observation.geometry[0], shuffled.geometry[0]
            original_known = order[after_geom["known_indices"]]
            if bool((original_known >= 12).any()):
                raise RuntimeError("Scattered geometry used a free face")
            row["scattered_feature_equivariance"] = _comparison(
                after_geom["phi"], before_geom["phi"][original_known],
                atol=FORMULA_ATOL, rtol=FORMULA_RTOL)
            row["scattered_h1_equivariance"] = _comparison(
                after_geom["h1"], before_geom["h1"][original_known],
                atol=FORMULA_ATOL, rtol=FORMULA_RTOL)
            row["scattered_P_equivariance"] = _comparison(
                after_geom["P"], before_geom["P"][original_known][:, original_known],
                atol=FORMULA_ATOL, rtol=FORMULA_RTOL)
            row["backward"] = _backward(model, batch, single)
            branch_after = state_sha256(model)
            row["parameter_hash_after"] = branch_after
            if branch_after != branch_before:
                raise RuntimeError("A zero-update branch gate changed model state")
            row["parameters_unchanged"] = True
            model.context_encoder = None
            model.context_observer = None
            model.set_trainable(False).eval()
            if state_sha256(model) != before:
                raise RuntimeError("Removing the disposable branch did not recover pure T1 state")
            del first, second, observed, permuted_value, observation, shuffled, before_geom, after_geom
            free_memory()
            persist()
        if single["forward_attempts"] != 13 or single["backward_attempts"] != 2:
            raise RuntimeError("Diagnostic gate execution count differs")
        # Exactly one default-path complete rollout. Historical comparison is
        # valid only for the corresponding exact A_continue checkpoint identity.
        historical = identity["checkpoint_sha256"] == BASELINE_FILE_SHA256
        if historical and not HISTORICAL_REPLAY.is_file():
            raise FileNotFoundError(HISTORICAL_REPLAY)
        c = torch.from_numpy(archive.cases[0,12].constraints.copy()).to("cuda")
        z = make_noise(9601).to("cuda")
        raw, path, sampling = clamped_sample(model, z, c, sample_counts)
        if sample_counts["rollout_attempts"] != ROLLOUT_CAP:
            raise RuntimeError("Gate replay count differs")
        known = np.broadcast_to(archive.cases[0,12].constraints.reshape(1,12,9), (51,12,9))
        known_bitwise = np.array_equal(path[:,:12].copy().view(np.uint32), known.copy().view(np.uint32))
        if not known_bitwise:
            raise RuntimeError("Complete rollout did not preserve C bitwise")
        replay = dict(status="PASS", model="T0", parent=0, K=12, seed=9601,
                      sampling=sampling, known_all51_uint32_equal=True,
                      history_comparison="NOT_APPLICABLE", history_reason="A different explicit portable initialization")
        if historical:
            with np.load(HISTORICAL_REPLAY, allow_pickle=False) as stored:
                old_path, old_raw = stored["path"], stored["output"]
                same = (old_path.dtype == path.dtype == np.float32 and old_path.shape == path.shape
                        and np.array_equal(path.view(np.uint32), old_path.view(np.uint32))
                        and np.array_equal(raw.view(np.uint32), old_raw.view(np.uint32)))
            if not same:
                raise RuntimeError("Default no-encoder rollout differs from all 51 historical states")
            replay.update(history_comparison="PASS", history_reason=None,
                          historical_path=str(HISTORICAL_REPLAY.resolve()),
                          historical_sha256=file_sha256(HISTORICAL_REPLAY), all51_uint32_equal=True)
        replay_file = out / "gate_T0_P0_K12_seed9601.npz"
        if replay_file.exists():
            raise FileExistsError(replay_file)
        np.savez_compressed(replay_file, output=raw, path=path)
        replay["raw_proof"] = dict(path=str(replay_file.resolve()), sha256=file_sha256(replay_file),
                                   trajectory_sha256=array_hash(path), output_sha256=array_hash(raw))
        result["replay"] = replay
        after = state_sha256(model)
        result["pure_parameter_hash_after"] = after
        if after != before:
            raise RuntimeError("Gate execution changed pure T1 parameters")
        result["parameters_unchanged"] = True
        proof_file = out / "gate_single_step_proof.npz"
        if proof_file.exists():
            raise FileExistsError(proof_file)
        np.savez_compressed(proof_file, **proof)
        result["proofs"]["single_step_velocities"] = dict(path=str(proof_file.resolve()),
                                                         sha256=file_sha256(proof_file), arrays=sorted(proof))
        if (single["forward_returns"]+sample_counts["model_forward_returns"] != FORWARD_CAP
                or single["backward_returns"] != BACKWARD_CAP or sample_counts["complete_rollouts"] != 1):
            raise RuntimeError("Completed gate budget accounting differs")
        result["status"] = "PASS"
        result["hard_wiring_status"] = "PASS"
        result["numerical_diagnostics_status"] = ("HAS_RECORDED_FAILURES" if result["recorded_numerical_failures"]
                                                    else "PASS_AT_REGISTERED_TOLERANCE")
    except BaseException as error:
        result.update(status="STOPPED", hard_wiring_status="STOPPED", error=repr(error))
        result["numerical_diagnostics_status"] = ("HAS_RECORDED_FAILURES" if result["recorded_numerical_failures"] else "INCOMPLETE")
        raise
    finally:
        # Keep any completed diagnostic outputs even when a later gate fails.
        partial_proof = out / "gate_single_step_proof.npz"
        if proof and not partial_proof.exists():
            np.savez_compressed(partial_proof, **proof)
            result["proofs"]["single_step_velocities"] = dict(
                path=str(partial_proof.resolve()), sha256=file_sha256(partial_proof), arrays=sorted(proof))
        persist()
        if model is not None:
            model.context_observer = None
            model.zero_grad(set_to_none=True)
            del model
        free_memory()
    return result
