"""Read-only CPU proof of inactive loss caps on the actual teacher control.

This neither registers a recipe nor claims an unexecuted candidate exists.
PASS requires all 1000 actual teacher updates and strictly positive numerical
margin. A future implementation must preserve the listed execution premises.
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from .common import OUT, START_SHA, atomic_json
from .training_diagnostics import Checks, _csv, _expected_aux, _jsonl, _pool

ARM = "REPLAY_FM_CONTROL"
PHASE = "explore"
FIXED_TEACHER_TRAINING_SHA256 = "3e03d25a71668db6495eabcb002e80a816b79de4f0797a1759ac3b8e30f07096"
CAP = .1
EPS = 1e-30


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def cap_scalar(fm, auxiliary):
    """Frozen counterfactual: Python binary64 scalars, exact inactive branch.

    The one-ULP interval is deliberately stricter than just no activation. It
    uses one representable step down for FM, up for denominator, and down for
    division. This is an audit uncertainty envelope, not a changed loss rule.
    """
    fm, auxiliary = float(fm), float(auxiliary)
    if not all(math.isfinite(x) and x >= 0 for x in (fm, auxiliary)):
        return dict(status="INVALID_NONFINITE_OR_NEGATIVE", cap_active=None,
            FM_hex=fm.hex(), aux_MSE_hex=auxiliary.hex())
    denominator = max(auxiliary, EPS)
    quotient = fm/denominator
    weight = min(CAP, quotient)
    margin_exact = Fraction.from_float(fm)-Fraction.from_float(CAP)*Fraction.from_float(denominator)
    fm_lower = max(0., math.nextafter(fm, -math.inf))
    den_upper = math.nextafter(denominator, math.inf)
    quotient_lower = math.nextafter(fm_lower/den_upper, -math.inf)
    active = weight.hex() != CAP.hex()
    safe = not active and margin_exact > 0 and quotient_lower > CAP
    if fm == 0 and auxiliary == 0:
        zero_case = "BOTH_ZERO_FLOOR_ACTIVATES_CAP_NO_EQUIVALENCE_WAIVER"
    elif fm == 0:
        zero_case = "PRIMARY_ZERO"
    elif auxiliary == 0:
        zero_case = "AUX_ZERO_STILL_APPLY_DENOMINATOR_FLOOR"
    else:
        zero_case = "NONE"
    return dict(status="CAP_ACTIVE" if active else "INACTIVE_WITH_STRICT_MARGIN" if safe else "INACTIVE_BOUNDARY_INCONCLUSIVE",
        FM=fm, aux_MSE=auxiliary, denominator=denominator,
        FM_hex=fm.hex(), aux_MSE_hex=auxiliary.hex(), denominator_hex=denominator.hex(),
        quotient=quotient if math.isfinite(quotient) else None, quotient_hex=quotient.hex(),
        lambda_eff=weight, lambda_eff_hex=weight.hex(), original_lambda_hex=CAP.hex(),
        cap_active=active, exact_positive_margin=margin_exact > 0,
        margin=float(margin_exact), margin_exact_fraction=str(margin_exact),
        relative_margin=float(margin_exact)/max(fm, CAP*denominator),
        conservative_quotient_lower=quotient_lower if math.isfinite(quotient_lower) else None,
        conservative_quotient_lower_hex=quotient_lower.hex(),
        strict_numerical_margin=safe, denominator_floor_active=auxiliary < EPS, zero_case=zero_case)


def self_test():
    cases = [
        (1., 2., "INACTIVE_WITH_STRICT_MARGIN"),
        (.1, 2., "CAP_ACTIVE"),
        (0., 0., "CAP_ACTIVE"),
        (0., 1., "CAP_ACTIVE"),
        (1., 0., "INACTIVE_WITH_STRICT_MARGIN"),
        (1e-32, 0., "CAP_ACTIVE"),
        (.1, 1., "INACTIVE_BOUNDARY_INCONCLUSIVE"),
        (math.nextafter(.1, math.inf), 1., "INACTIVE_BOUNDARY_INCONCLUSIVE"),
        (.11, 1., "INACTIVE_WITH_STRICT_MARGIN"),
        (float("nan"), 1., "INVALID_NONFINITE_OR_NEGATIVE"),
        (1., -1., "INVALID_NONFINITE_OR_NEGATIVE"),
    ]
    rows = []
    for fm, aux, expected in cases:
        actual = cap_scalar(fm, aux)
        assert actual["status"] == expected, (fm, aux, actual, expected)
        rows.append(dict(FM_hex=fm.hex(), aux_hex=aux.hex(), expected=expected, actual=actual["status"], passed=True))
    return dict(status="PASS", cases=rows, model_calls=0, matching_calls=0)


def run(out=OUT, *, live=False):
    out = Path(out)
    destination = out/"cap_control_equivalence"/("live" if live else "final")
    destination.mkdir(parents=True, exist_ok=True)
    check, snapshots, missing, hashes = Checks(), [], [], {}
    test_receipt = self_test()
    marker_path = out/"training"/f"{PHASE}_{ARM}_complete.json"
    marker = json.loads(marker_path.read_text()) if marker_path.exists() else None
    metric_path = out/"training"/f"{PHASE}_{ARM}_metrics.jsonl"
    rows = _jsonl(metric_path, live, snapshots) if metric_path.exists() else []
    if not rows:
        missing.append("No actual teacher metric rows yet")
    check("teacher_metric_steps_unique_contiguous", [r["pilot_step"] for r in rows] == list(range(1, len(rows)+1)))
    check("teacher_metric_step_budget", len(rows) <= 1000)
    complete = len(rows) == 1000 and marker is not None and marker.get("status") == "PASS" and marker.get("steps") == 1000
    if not complete:
        missing.append("Require actual teacher 1000-step PASS completion marker and all 1000 metric rows")
    if marker and marker.get("steps") == 1000:
        check("completed_teacher_exact_1000_rows", len(rows) == 1000)
        check("completed_teacher_cursor", marker["stream_state"]["batch_index"] == 1000 and marker["stream_state"]["sample_index"] == 8000)
    baseline_path = out/"training/explore_FM_CONTROL_metrics.jsonl"
    baseline = _jsonl(baseline_path, live, snapshots) if rows and baseline_path.exists() else []
    baseline = {r["pilot_step"]: r for r in baseline}
    pool = _pool(out, check) if rows else None
    csv_rows = []
    for r in rows:
        step = r["pilot_step"]
        check("actual_teacher_arm_phase", r["arm"] == ARM and r["phase"] == PHASE, step)
        accumulated = 0.
        for sample in r["sample_diagnostics"]:
            accumulated += float(sample["FM_contribution"])
        check("FM_original_python_sequential_sum_bit_exact", accumulated.hex() == float(r["FM"]).hex(), step)
        check("each_primary_contribution_is_original_FP32", all(float(np.float32(s["FM_contribution"])) == s["FM_contribution"] for s in r["sample_diagnostics"]), step)
        check("aux_unweighted_is_original_FP32", float(np.float32(r["aux_unweighted"])) == r["aux_unweighted"], step)
        check("original_teacher_weighted_loss_exact", float(np.float32(CAP)*np.float32(r["aux_unweighted"])) == r["aux_weighted"], step)
        check("main_8_unique_parents_4unit", len(r["sample_diagnostics"]) == 8 and len({s["uid"] for s in r["sample_diagnostics"]}) == 8 and sum(s["alpha"] == 1 for s in r["sample_diagnostics"]) == 4, step)
        check("main_original_free_denominator", r["free_coordinate_denominator"] == sum(9*(s["N"]-s["K"]) for s in r["sample_diagnostics"]), step)
        b = baseline.get(step)
        check("main_stream_matches_original_FM_control", b is not None and all(r[k] == b[k] for k in ("shared_batch_sha256", "coupled_batch_sha256", "free_coordinate_denominator"))
            and [s["shared_input_sha256"] for s in r["sample_diagnostics"]] == [s["shared_input_sha256"] for s in b["sample_diagnostics"]], step)
        expected, _ = _expected_aux(step, ARM, 20261008, pool)
        check("teacher_aux_matches_frozen_pool_and_selection", all(r["aux_audit"].get(k) == v for k, v in expected.items()), step)
        check("teacher_aux_denominator_t_match", r["aux_free_denominator"] == expected["denominator"] and r["aux_t"] == expected["t"], step)
        scalar = cap_scalar(r["FM"], r["aux_unweighted"])
        check("cap_scalars_valid", scalar["status"] != "INVALID_NONFINITE_OR_NEGATIVE", step)
        csv_rows.append(dict(step=step, **scalar, FM_reconstructed_hex=accumulated.hex(),
            original_weighted_aux=r["aux_weighted"], original_weighted_aux_hex=float(r["aux_weighted"]).hex(),
            queue=expected["queue"], uid=expected["uid"], request_id=expected["request_id"], t=expected["t"],
            xt_sha256=expected["xt_sha256"], u_sha256=expected["u_sha256"], target_sha256=expected["target_sha256"],
            shared_batch_sha256=r["shared_batch_sha256"], aux_payload_sha256=expected["payload_sha256"]))

    counts, begins, ends = defaultdict(Counter), {}, {}
    if rows:
        for path in sorted((out/"ledger").glob(f"calls_{PHASE}_{ARM}_*train.jsonl")):
            for entry in _jsonl(path, live, snapshots):
                ctx = entry.get("context") or {}
                step = ctx.get("pilot_step")
                if ctx.get("arm") != ARM or ctx.get("phase") != PHASE or not isinstance(step, int) or not 1 <= step <= len(rows):
                    continue
                event, kind = entry.get("event"), entry.get("kind")
                if event not in ("CALL_BEGIN", "CALL_RETURN", "CALL_FAIL"):
                    continue
                holder = begins if event == "CALL_BEGIN" else ends
                check("teacher_call_unique", entry["call_id"] not in holder, entry["call_id"])
                holder[entry["call_id"]] = entry
                counts[step][(event, kind, ctx.get("microbatch"))] += 1
        check("teacher_completed_step_no_pending_or_orphan", set(begins) == set(ends))
        for cid, end in ends.items():
            check("teacher_actual_call_returned_successfully", end["event"] == "CALL_RETURN", cid)
            if cid in begins:
                check("teacher_call_begin_return_identity", end["kind"] == begins[cid]["kind"] and end["context"] == begins[cid]["context"], cid)
        for step in range(1, len(rows)+1):
            expected_count = Counter()
            for event in ("CALL_BEGIN", "CALL_RETURN"):
                for kind in ("training_forward", "training_backward"):
                    for micro in list(range(8))+["aux"]:
                        expected_count[(event, kind, micro)] = 1
                expected_count[(event, "optimizer_update", None)] = 1
            check("teacher_exact_9F_9B_1update_per_step", counts[step] == expected_count, step)
        starts_path = out/"decisions/training_starts.jsonl"
        starts = _jsonl(starts_path, live, snapshots) if starts_path.exists() else []
        starts = [r for r in starts if r.get("arm") == ARM and r.get("phase") == PHASE]
        check("teacher_start_from_same_START", any(r.get("start_step") == 0 and r.get("source_checkpoint_sha256") == START_SHA for r in starts))
        for start in starts:
            check("all_teacher_stages_retain_START_origin", start.get("source_checkpoint_sha256") == START_SHA)
        loads = sorted((out/"audit").glob(f"{PHASE}_{ARM}_*train_load.json"))
        check("teacher_actual_optimizer_load_receipt_present", bool(loads))
        initial_loads = []
        for path in loads:
            data = json.loads(path.read_text())
            hashes[str(path)] = file_hash(path)
            check("teacher_full_AdamW_restored", data.get("optimizer_restored") is True and data["metadata"].get("optimizer_reset") is False, str(path))
            if data["metadata"].get("feedback_step") == 0:
                initial_loads.append(data)
                check("teacher_initial_checkpoint_is_START", data["hashes"]["checkpoint_sha256"] == START_SHA, str(path))
        check("teacher_initial_load_receipt_present", bool(initial_loads))
        source_manifests = sorted((out/"code_snapshots").glob(f"{PHASE}_{ARM}_*train_*/manifest.json"))
        check("teacher_executed_code_snapshots_present", bool(source_manifests))
        for path in source_manifests:
            manifest = json.loads(path.read_text())
            hashes[str(path)] = file_hash(path)
            for name, expected_sha in manifest.items():
                saved = path.parent/Path(name).name
                check("teacher_executed_source_snapshot_hash", saved.exists() and file_hash(saved) == expected_sha, str(saved))
                if Path(name).name == "training.py":
                    check("teacher_executed_constant_weight_trainer_identity", expected_sha == FIXED_TEACHER_TRAINING_SHA256, str(saved))

    endpoint = None
    if complete:
        side_path = out/"checkpoints/explore_REPLAY_FM_CONTROL_1000.json"
        if not side_path.exists():
            missing.append("Endpoint checkpoint sidecar absent")
        else:
            side = json.loads(side_path.read_text())
            endpoint_path = Path(side["path"])
            hashes[str(side_path)] = file_hash(side_path)
            if not endpoint_path.exists():
                missing.append("Endpoint checkpoint bytes absent")
            else:
                actual_sha = file_hash(endpoint_path)
                check("teacher_endpoint_file_sha256", actual_sha == side["checkpoint_sha256"])
                meta = side["metadata"]
                check("teacher_endpoint_full_state_identity", meta["source_checkpoint_sha256"] == START_SHA and meta["feedback_step"] == 1000 and meta["conditional_total_step"] == 6000 and meta["full_recovery_state"] and not meta["optimizer_reset"])
                endpoint = dict(path=str(endpoint_path), checkpoint_sha256=actual_sha,
                    model_state_sha256=meta["model_state_sha256"], optimizer_state_sha256=meta["optimizer_state_sha256"],
                    rng_sha256=meta["rng_sha256"], stream_state_sha256=meta["stream_state_sha256"],
                    tensor_hash_scope="Stored verified sidecar metadata; this CPU verifier hashes checkpoint bytes without loading model/optimizer tensors.")
    active = [r["step"] for r in csv_rows if r.get("cap_active") is True]
    boundary = [r["step"] for r in csv_rows if r["status"] == "INACTIVE_BOUNDARY_INCONCLUSIVE"]
    if check.errors:
        status = "FAIL_EVIDENCE"
    elif active:
        status = "FAIL_CAP_ACTIVE"
    elif boundary or missing:
        status = "INCOMPLETE"
    else:
        status = "PASS_CAP_INACTIVE_1000"
    for path in [Path(__file__), Path(__file__).with_name("training_diagnostics.py"), out/"protocol.json",
        out/"streams/explore/train_plan.json", out/"feedback/pool_v1/manifest.json"]:
        if path.exists():
            hashes[str(path)] = file_hash(path)
    _csv(destination/"per_step.csv", csv_rows, fields=list(csv_rows[0]) if csv_rows else ["step", "status", "FM_hex", "aux_MSE_hex", "lambda_eff_hex", "margin", "strict_numerical_margin"])
    receipt = dict(status=status, live=live, completed_teacher_steps=len(rows), required_steps=1000,
        actual_marker_complete=complete, cap_active_steps=active, boundary_inconclusive_steps=boundary,
        safe_inactive_steps=sum(r["status"] == "INACTIVE_WITH_STRICT_MARGIN" for r in csv_rows),
        minimum_positive_margin=min((r["margin"] for r in csv_rows if r.get("margin", 0) > 0), default=None),
        minimum_relative_margin=min((r["relative_margin"] for r in csv_rows if "relative_margin" in r), default=None),
        checks=sum(check.counts.values()), checks_by_name=dict(check.counts), errors=check.errors, missing=missing,
        snapshots=snapshots, source_and_evidence_hashes=hashes, endpoint=endpoint,
        counterfactual_rule="lambda_eff=min(0.1, Python_float_sequential_FM_sum/max(Python_float_FP32_aux_MSE,1e-30)); coefficient detached",
        margin_rule="Exact rational margin FM-binary64(0.1)*max(aux,binary64(1e-30)) > 0 and lower quotient after downward-FM/upward-denominator/downward-division one-ULP envelope > binary64(0.1).",
        implication="PASS establishes zero activation on the entire actual teacher path. Reusing its weights for a future capped teacher is conditional on preserving all execution premises below; no new teacher training is claimed.",
        execution_premises=["Identical START model, full AdamW and RNG state", "Identical eight-primary and teacher-extra inputs, targets, queues, times and denominators",
            "Identical Python sequential primary loss sum and scalar cap arithmetic", "Inactive branch retains exact original loss=.1*raw expression",
            "No gradient through coefficient; same backward accumulation, clipping, optimizer, precision, deterministic runtime and ordering",
            "New rule consumes no RNG and changes no mutable training state", "Explore proof does not automatically prove repeat seed equivalence"],
        zero_case_policy="No activation waiver for zero loss: the denominator floor still applies, and a rounded-zero MSE does not establish a zero gradient.",
        budget_policy="Any active step or inconclusive evidence prevents this reuse claim. A new matched control cannot be silently added beyond the registered recipe allowance.",
        candidate_registration="NONE; bounded mathematical/CPU audit only", model_calls=0, backward_calls=0, optimizer_updates=0,
        self_tests=test_receipt, output_directory=str(destination))
    atomic_json(destination/"receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    result = run(args.out, live=args.live)
    print(json.dumps({k: result[k] for k in ("status", "completed_teacher_steps", "cap_active_steps", "boundary_inconclusive_steps", "missing", "output_directory")}, indent=2))
    raise SystemExit(1 if result["status"].startswith("FAIL") else 0)
