"""Independent CPU repeat-teacher cap proof; original exploration proof is read-only.

No model, matching, new noise, or repeat dispatch occurs here. Full PASS
requires the authorized candidate and teacher to each have 1000 real updates.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from .common import OUT, START_SHA, atomic_json, json_hash
from .cap_control_equivalence import cap_scalar, self_test, file_hash, CAP, FIXED_TEACHER_TRAINING_SHA256
from .training_diagnostics import Checks, _csv, _expected_aux, _jsonl, _pool

ARM = "REPLAY_FM_CONTROL"
CANDIDATE = "TRAJECTORY_REPLAY_CAPPED"
PHASE = "repeat"
SEED = 20261009


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def authorization_issues(authorization, recipe):
    issues = []
    pair = {CANDIDATE, ARM}
    if recipe.get("status") != "FROZEN" or recipe.get("arm") != CANDIDATE or recipe.get("START_sha256") != START_SHA:
        issues.append("H3_RECIPE_IDENTITY")
    if not recipe.get("sources", {}).get("training.py"):
        issues.append("H3_TRAINER_HASH_MISSING")
    if authorization.get("status") != "FROZEN" or authorization.get("START_sha256") != START_SHA:
        issues.append("REPEAT_AUTHORIZATION_IDENTITY")
    if authorization.get("candidate_arm") != CANDIDATE or authorization.get("matched_arm") != ARM:
        issues.append("REPEAT_ARM_ROLES")
    arms = authorization.get("arms", [])
    models = authorization.get("exploration_checkpoints", {})
    if not isinstance(arms, list) or len(arms) != 2 or set(arms) != pair or set(models) != pair:
        issues.append("EXACT_TWO_ARMS_AND_MODELS")
    if not all(m.get("path") and m.get("sha256") for m in models.values()):
        issues.append("EXPLORATION_MODELS_INCOMPLETE")
    if not authorization.get("DEV_gate_path") or not authorization.get("DEV_gate_sha256"):
        issues.append("REVIEWED_DEV_GATE_MISSING")
    return issues


def completion_issues(marker, arm, plan_sha):
    issues = []
    if (marker.get("status"), marker.get("steps"), marker.get("arm"), marker.get("phase"), marker.get("seed")) != ("PASS", 1000, arm, PHASE, SEED):
        issues.append("COMPLETE_ROLE_STEP_SEED")
    stream = marker.get("stream_state", {})
    source_arm = "TRAJECTORY_REPLAY" if arm == CANDIDATE else ARM
    expected = dict(seed=SEED, batch_index=1000, sample_index=8000, arm=source_arm,
        plan_sha256=plan_sha, lambda_value=.25)
    if any(stream.get(k) != v for k,v in expected.items()):
        issues.append("COMPLETE_STREAM_IDENTITY")
    return issues


def _early(destination, live, status, missing, errors=None):
    r = dict(status=status, live=live, phase=PHASE, training_seed=SEED, teacher_arm=ARM,
        matched_candidate_arm=CANDIDATE, completed_teacher_steps=0, completed_candidate_steps=0,
        required_steps=1000, actual_marker_complete=False, candidate_marker_complete=False,
        safe_inactive_steps=0, cap_active_steps=[], boundary_inconclusive_steps=[],
        errors=errors or [], missing=missing, model_calls=0, backward_calls=0, optimizer_updates=0,
        candidate_registration="NONE", output_directory=str(destination))
    atomic_json(destination/"receipt.json",r)
    return r


class _TeacherBranch(ast.NodeTransformer):
    """Specialize only literal arm tests, never execute a trainer function."""
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in ("arm", "aux_arm", "stream_arm"):
            return ast.copy_location(ast.Constant(ARM), node)
        return node

    @staticmethod
    def truth(node):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            a, b = node.left, node.comparators[0]
            if isinstance(a, ast.Constant) and isinstance(b, ast.Constant):
                if isinstance(node.ops[0], ast.Eq): return a.value == b.value
                if isinstance(node.ops[0], ast.NotEq): return a.value != b.value
        return None

    def visit_If(self, node):
        node = self.generic_visit(node)
        choice = self.truth(node.test)
        return node if choice is None else node.body if choice else node.orelse

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        choice = self.truth(node.test)
        return node if choice is None else node.body if choice else node.orelse

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("aux_arm", "stream_arm"):
            if not isinstance(node.value, ast.Constant) or node.value.value != ARM:
                raise ValueError("Teacher alias is not an exact identity")
            return None
        return node


def teacher_branch_equivalent(old_source, new_source):
    def functions(source):
        return {n.name:n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    old, new = functions(old_source), functions(new_source)
    def specialized(node):
        return _TeacherBranch().visit(copy.deepcopy(node))
    a, b = specialized(old["update"]), specialized(new["update"])
    def stream_call(node):
        matches = [n for n in ast.walk(specialized(node)) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "HybridStream"]
        if len(matches) != 1: raise ValueError("Require exactly one unchanged primary stream construction")
        return ast.dump(matches[0], include_attributes=False)
    return dict(update_equal=ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),
        gradient_helper_equal=ast.dump(old["global_grad_norm"],include_attributes=False)==ast.dump(new["global_grad_norm"],include_attributes=False),
        stream_call_equal=stream_call(old["train"])==stream_call(new["train"]))


def run(out=OUT, *, live=False):
    out = Path(out).resolve()
    destination = out/"cap_control_equivalence"/"repeat"/("live" if live else "final")
    destination.mkdir(parents=True, exist_ok=True)
    check, snapshots, missing, hashes = Checks(), [], [], {}
    auth_path, recipe_path = out/"decisions/repeat_authorization.json", out/"decisions/H3_recipe.json"
    if not auth_path.exists() or not recipe_path.exists():
        return _early(destination, live, "INCOMPLETE", ["Frozen repeat authorization and H3 recipe are required"])
    authorization, recipe = _json(auth_path), _json(recipe_path)
    issues = authorization_issues(authorization, recipe)
    if issues:
        return _early(destination, live, "FAIL_EVIDENCE", [], issues)
    candidate = authorization["candidate_arm"]
    expected_trainer_sha = recipe["sources"]["training.py"]
    hashes[str(auth_path)], hashes[str(recipe_path)] = file_hash(auth_path), file_hash(recipe_path)
    gate_path = Path(authorization["DEV_gate_path"])
    if not gate_path.exists():
        return _early(destination, live, "INCOMPLETE", ["Authorized reviewed DEV gate is absent"])
    gate = _json(gate_path)
    check("repeat_positive_reviewed_DEV_gate", gate.get("status") == gate.get("numeric_status") == "PASS")
    check("repeat_DEV_gate_hash", file_hash(gate_path) == authorization["DEV_gate_sha256"])
    hashes[str(gate_path)] = file_hash(gate_path)
    check("H3_protocol_pool_still_bound", recipe["parent_protocol_sha256"] == file_hash(out/"protocol.json")
        and recipe["feedback_pool_sha256"] == file_hash(out/"feedback/pool_v1/manifest.json"))
    proof_path = Path(recipe["exploration_matched_control_equivalence"]["path"])
    check("immutable_exploration_proof_bound", file_hash(proof_path) == recipe["exploration_matched_control_equivalence"]["sha256"])
    check("immutable_exploration_proof_pass", _json(proof_path)["status"] == "PASS_CAP_INACTIVE_1000")
    hashes[str(proof_path)] = file_hash(proof_path)
    plan_path = out/"streams/repeat/train_plan.json"
    plan, explore_plan = _json(plan_path), _json(out/"streams/explore/train_plan.json")
    plan_sha = file_hash(plan_path)
    check("repeat_seed_20261009_plan8000", plan["seed"] == SEED and len(plan["records"]) == 8000)
    check("repeat_plan_independent_seed_domains", all(a[k] != b[k]
        for a, b in zip(plan["records"], explore_plan["records"])
        for k in ("gaussian_seed", "t_seed", "permutation_seed", "epsilon_seed")))
    for arm, model in authorization["exploration_checkpoints"].items():
        side_path = Path(model["path"]).with_suffix(".json")
        if not side_path.exists():
            missing.append("Authorized exploration checkpoint sidecar absent: "+str(side_path))
            continue
        side = _json(side_path)
        check("authorized_exploration_model_sidecar", side["checkpoint_sha256"] == model["sha256"]
            and side["metadata"]["arm"] == arm and side["metadata"]["phase"] == "explore"
            and side["metadata"]["source_checkpoint_sha256"] == START_SHA, arm)
        field = "candidate_label" if arm == candidate else "matched_label"
        check("positive_gate_matches_authorized_models", gate.get(field) == Path(model["path"]).stem, arm)
        hashes[str(side_path)] = file_hash(side_path)
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
        check("completed_teacher_cursor", not completion_issues(marker, ARM, plan_sha))
    candidate_marker_path = out/"training"/f"repeat_{candidate}_complete.json"
    candidate_marker = _json(candidate_marker_path) if candidate_marker_path.exists() else None
    candidate_complete = bool(candidate_marker and candidate_marker.get("status") == "PASS" and candidate_marker.get("steps") == 1000)
    if not candidate_complete:
        missing.append("Matched repeat candidate must also have its full 1000-step PASS marker")
    else:
        check("completed_repeat_candidate_cursor", not completion_issues(candidate_marker, candidate, plan_sha))
    baseline_path = out/"training"/f"repeat_{candidate}_metrics.jsonl"
    baseline = _jsonl(baseline_path, live, snapshots) if rows and baseline_path.exists() else []
    check("repeat_candidate_steps_unique_contiguous", [r["pilot_step"] for r in baseline] == list(range(1, len(baseline)+1)))
    if candidate_complete:
        check("repeat_candidate_exact_1000_rows", len(baseline) == 1000)
    baseline = {r["pilot_step"]: r for r in baseline}
    pool = _pool(out, check) if rows else None
    csv_rows = []
    aux_contexts = {}
    paired_steps = 0
    for r in rows:
        step = r["pilot_step"]
        check("actual_teacher_arm_phase", r["arm"] == ARM and r["phase"] == PHASE, step)
        check("teacher_actual_total_cursor", r["conditional_total_step"] == 5000+step and r["batch_index"] == step, step)
        check("teacher_constant_weight_branch_no_cap_telemetry", "loss_balance" not in r, step)
        actual_batch_hash = hashlib.sha256("".join(v["shared_input_sha256"] for v in r["sample_diagnostics"]).encode()).hexdigest()
        check("teacher_batch_hash_reconstructed", actual_batch_hash == r["shared_batch_sha256"], step)
        for slot, sample in enumerate(r["sample_diagnostics"]):
            expected_record = plan["records"][8*(step-1)+slot]
            check("repeat_primary_matches_frozen_plan", all(sample[k] == expected_record[k]
                for k in ("sample_index", "task_id", "uid", "N", "K", "alpha")), [step, slot])
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
        if b is None:
            if "Candidate common-step metrics not yet complete" not in missing:
                missing.append("Candidate common-step metrics not yet complete")
        else:
            paired_steps += 1
            check("main_stream_matches_authorized_repeat_candidate", b["arm"] == candidate and b["phase"] == PHASE
                and all(r[k] == b[k] for k in ("shared_batch_sha256", "coupled_batch_sha256", "free_coordinate_denominator"))
                and [s["shared_input_sha256"] for s in r["sample_diagnostics"]] == [s["shared_input_sha256"] for s in b["sample_diagnostics"]], step)
        expected, task_id = _expected_aux(step, ARM, SEED, pool)
        aux_contexts[step] = (task_id, expected["request_id"])
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
                micro = ctx.get("microbatch")
                counts[step][(event, kind, micro)] += 1
                if isinstance(micro, int):
                    batch = rows[step-1]["sample_diagnostics"]
                    check("teacher_ledger_primary_input_identity", 0 <= micro < 8
                        and ctx.get("sample_index") == batch[micro]["sample_index"]
                        and ctx.get("shared_input_sha256") == batch[micro]["shared_input_sha256"], [step, micro, event])
                elif micro == "aux":
                    check("teacher_ledger_aux_task_identity", ctx.get("request_id") in aux_contexts[step], [step,event])
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
            check("teacher_full_AdamW_restored", data.get("optimizer_restored") is True
                and data["metadata"].get("optimizer_reset") is False and data["metadata"].get("full_recovery_state") is True, str(path))
            if data["metadata"].get("feedback_step") == 0:
                initial_loads.append(data)
                check("teacher_initial_checkpoint_is_START", data["hashes"]["checkpoint_sha256"] == START_SHA
                    and data["metadata"]["conditional_total_step"] == 5000, str(path))
            else:
                check("teacher_resume_is_repeat_teacher_from_START", data["metadata"]["phase"] == PHASE
                    and data["metadata"]["arm"] == ARM and data["metadata"]["source_checkpoint_sha256"] == START_SHA
                    and data["metadata"]["conditional_total_step"] == 5000+data["metadata"]["feedback_step"], str(path))
        check("teacher_initial_load_receipt_present", bool(initial_loads))
        source_manifests = sorted((out/"code_snapshots").glob(f"{PHASE}_{ARM}_*train_*/manifest.json"))
        check("teacher_executed_code_snapshots_present", bool(source_manifests))
        for path in source_manifests:
            manifest = json.loads(path.read_text())
            check("teacher_snapshot_has_exact_training_source", sum(Path(name).name == "training.py" for name in manifest) == 1, str(path))
            hashes[str(path)] = file_hash(path)
            for name, expected_sha in manifest.items():
                saved = path.parent/Path(name).name
                check("teacher_executed_source_snapshot_hash", saved.exists() and file_hash(saved) == expected_sha, str(saved))
                if Path(name).name == "training.py":
                    check("repeat_teacher_exact_frozen_H3_trainer_identity", expected_sha == expected_trainer_sha, str(saved))
                    old_source = out/"engineering/dormant_H3_support/training.py.before"
                    check("original_teacher_reference_source_SHA", file_hash(old_source) == FIXED_TEACHER_TRAINING_SHA256)
                    same = teacher_branch_equivalent(old_source.read_text(), saved.read_text())
                    check("new_teacher_update_is_original_constant_point_one_AST", same["update_equal"], str(saved))
                    check("new_teacher_gradient_helper_unchanged_AST", same["gradient_helper_equal"], str(saved))
                    check("new_teacher_HYBRID_stream_alias_identity_AST", same["stream_call_equal"], str(saved))
                    hashes[str(old_source)] = file_hash(old_source)

    endpoint = None
    if complete:
        side_path = out/"checkpoints/repeat_REPLAY_FM_CONTROL_1000.json"
        if not side_path.exists():
            missing.append("Endpoint checkpoint sidecar absent")
        else:
            side = json.loads(side_path.read_text())
            endpoint_path = Path(side["path"])
            check("repeat_teacher_endpoint_exact_path", endpoint_path.resolve() == (out/"checkpoints/repeat_REPLAY_FM_CONTROL_1000.pt").resolve())
            hashes[str(side_path)] = file_hash(side_path)
            if not endpoint_path.exists():
                missing.append("Endpoint checkpoint bytes absent")
            else:
                actual_sha = file_hash(endpoint_path)
                check("teacher_endpoint_file_sha256", actual_sha == side["checkpoint_sha256"])
                meta = side["metadata"]
                check("teacher_endpoint_full_state_identity", meta["source_checkpoint_sha256"] == START_SHA
                    and meta["arm"] == ARM and meta["phase"] == PHASE and meta["feedback_step"] == 1000
                    and meta["conditional_total_step"] == 6000 and meta["full_recovery_state"] and not meta["optimizer_reset"])
                check("teacher_endpoint_stream_hash_matches_complete_marker", meta["stream_state_sha256"] == json_hash(marker["stream_state"]))
                check("teacher_endpoint_protocol_pool_bound", meta["protocol_sha256"] == json_hash(_json(out/"protocol.json"))
                    and meta["feedback_pool_sha256"] == file_hash(out/"feedback/pool_v1/manifest.json"))
                endpoint = dict(path=str(endpoint_path), checkpoint_sha256=actual_sha,
                    model_state_sha256=meta["model_state_sha256"], optimizer_state_sha256=meta["optimizer_state_sha256"],
                    rng_sha256=meta["rng_sha256"], stream_state_sha256=meta["stream_state_sha256"],
                    tensor_hash_scope="Stored verified sidecar metadata; this CPU verifier hashes checkpoint bytes without loading model/optimizer tensors.")
    if candidate_complete:
        candidate_loads = sorted((out/"audit").glob(f"repeat_{candidate}_*train_load.json"))
        initial_candidate_loads = []
        for path in candidate_loads:
            data = _json(path)
            hashes[str(path)] = file_hash(path)
            meta = data["metadata"]
            check("repeat_candidate_AdamW_restored", data.get("optimizer_restored") is True
                and meta.get("full_recovery_state") is True and meta.get("optimizer_reset") is False, str(path))
            if meta.get("feedback_step") == 0:
                initial_candidate_loads.append(path)
                check("repeat_candidate_initial_START", data["hashes"]["checkpoint_sha256"] == START_SHA
                    and meta["conditional_total_step"] == 5000, str(path))
        check("repeat_candidate_initial_load_present", bool(initial_candidate_loads))
    candidate_endpoint = None
    if candidate_complete:
        candidate_side_path = out/"checkpoints"/f"repeat_{candidate}_1000.json"
        if not candidate_side_path.exists():
            missing.append("Repeat candidate endpoint sidecar missing")
        else:
            side = _json(candidate_side_path)
            path = Path(side["path"])
            if not path.exists():
                missing.append("Repeat candidate checkpoint bytes missing")
            else:
                digest = file_hash(path)
                meta = side["metadata"]
                check("repeat_candidate_endpoint_file_and_role", digest == side["checkpoint_sha256"]
                    and path.resolve() == (out/"checkpoints"/f"repeat_{candidate}_1000.pt").resolve()
                    and meta["phase"] == PHASE and meta["arm"] == candidate and meta["feedback_step"] == 1000
                    and meta["conditional_total_step"] == 6000 and meta["source_checkpoint_sha256"] == START_SHA)
                check("repeat_candidate_recipe_and_stream_bound", meta["recipe_addendum_file_sha256"] == file_hash(recipe_path)
                    and meta["stream_state_sha256"] == json_hash(candidate_marker["stream_state"]))
                candidate_endpoint = dict(path=str(path), checkpoint_sha256=digest,
                    model_state_sha256=meta["model_state_sha256"], optimizer_state_sha256=meta["optimizer_state_sha256"])
                hashes[str(candidate_side_path)] = file_hash(candidate_side_path)
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
        Path(__file__).with_name("cap_control_equivalence.py"), out/"streams/repeat/train_plan.json",
        out/"streams/explore/train_plan.json", out/"feedback/pool_v1/manifest.json"]:
        if path.exists():
            hashes[str(path)] = file_hash(path)
    _csv(destination/"per_step.csv", csv_rows, fields=list(csv_rows[0]) if csv_rows else ["step", "status", "FM_hex", "aux_MSE_hex", "lambda_eff_hex", "margin", "strict_numerical_margin"])
    receipt = dict(status=status, live=live, phase=PHASE, training_seed=SEED, teacher_arm=ARM,
        matched_candidate_arm=candidate, repeat_authorization_sha256=file_hash(auth_path),
        H3_recipe_sha256=file_hash(recipe_path), frozen_teacher_trainer_sha256=expected_trainer_sha,
        completed_candidate_steps=len(baseline), candidate_marker_complete=candidate_complete,
        matched_primary_input_steps=paired_steps,
        completed_teacher_steps=len(rows), required_steps=1000,
        actual_marker_complete=complete, cap_active_steps=active, boundary_inconclusive_steps=boundary,
        safe_inactive_steps=sum(r["status"] == "INACTIVE_WITH_STRICT_MARGIN" for r in csv_rows),
        minimum_positive_margin=min((r["margin"] for r in csv_rows if r.get("margin", 0) > 0), default=None),
        minimum_relative_margin=min((r["relative_margin"] for r in csv_rows if "relative_margin" in r), default=None),
        checks=sum(check.counts.values()), checks_by_name=dict(check.counts), errors=check.errors, missing=missing,
        snapshots=snapshots, source_and_evidence_hashes=hashes, endpoint=endpoint, candidate_endpoint=candidate_endpoint,
        counterfactual_rule="lambda_eff=min(0.1, Python_float_sequential_FM_sum/max(Python_float_FP32_aux_MSE,1e-30)); coefficient detached",
        margin_rule="Exact rational margin FM-binary64(0.1)*max(aux,binary64(1e-30)) > 0 and lower quotient after downward-FM/upward-denominator/downward-division one-ULP envelope > binary64(0.1).",
        implication="PASS establishes zero activation on the entire actual teacher path. Reusing its weights for a future capped teacher is conditional on preserving all execution premises below; no new teacher training is claimed.",
        execution_premises=["Identical START model, full AdamW and RNG state", "Identical eight-primary and teacher-extra inputs, targets, queues, times and denominators",
            "Identical Python sequential primary loss sum and scalar cap arithmetic", "Inactive branch retains exact original loss=.1*raw expression",
            "No gradient through coefficient; same backward accumulation, clipping, optimizer, precision, deterministic runtime and ordering",
            "New rule consumes no RNG and changes no mutable training state", "This receipt independently verifies repeat seed20261009; exploration proof remains separate"],
        zero_case_policy="No activation waiver for zero loss: the denominator floor still applies, and a rounded-zero MSE does not establish a zero gradient.",
        budget_policy="Any active step or inconclusive evidence prevents this reuse claim. A new matched control cannot be silently added beyond the registered recipe allowance.",
        candidate_registration="NONE; requires a preexisting positive repeat authorization", model_calls=0, backward_calls=0, optimizer_updates=0,
        self_tests=test_receipt, output_directory=str(destination))
    atomic_json(destination/"receipt.json", receipt)
    return receipt


def metadata_tests(out=OUT):
    out=Path(out).resolve()
    folder=out/"tests/repeat_cap_equivalence"
    folder.mkdir(parents=True,exist_ok=True)
    recipe=_json(out/"decisions/H3_recipe.json")
    auth=dict(status="FROZEN",START_sha256=START_SHA,candidate_arm=CANDIDATE,matched_arm=ARM,
        arms=[CANDIDATE,ARM],exploration_checkpoints={CANDIDATE:dict(path="candidate",sha256="candidatehash"),ARM:dict(path="teacher",sha256="teacherhash")},
        DEV_gate_path="fixture_gate",DEV_gate_sha256="fixture_sha")
    tests=[]
    assert not authorization_issues(auth,recipe)
    tests.append(dict(name="valid_metadata_schema",passed=True,synthetic=True))
    mutations=[("empty_arms",lambda a:a.update(arms=[])),("duplicate_arms",lambda a:a.update(arms=[ARM,ARM])),
        ("empty_models",lambda a:a.update(exploration_checkpoints={})),
        ("wrong_candidate",lambda a:a.update(candidate_arm="TRAJECTORY_REPLAY")),
        ("wrong_matched",lambda a:a.update(matched_arm=CANDIDATE)),
        ("wrong_START",lambda a:a.update(START_sha256="wrong")),
        ("not_frozen",lambda a:a.update(status="DRAFT")),
        ("missing_gate_hash",lambda a:a.pop("DEV_gate_sha256"))]
    for name,mutate in mutations:
        value=copy.deepcopy(auth);mutate(value)
        issues=authorization_issues(value,recipe)
        assert issues,name
        tests.append(dict(name=name,passed=True,rejected_reasons=issues,synthetic=True))
    marker=dict(status="PASS",steps=1000,arm=ARM,phase=PHASE,seed=SEED,
        stream_state=dict(seed=SEED,batch_index=1000,sample_index=8000,arm=ARM,plan_sha256="planhash",lambda_value=.25))
    assert not completion_issues(marker,ARM,"planhash")
    tests.append(dict(name="valid_complete_marker",passed=True,synthetic=True))
    for name,key,value in [("500_not_complete","steps",500),("explore_not_repeat","phase","explore"),("wrong_seed","seed",20261008)]:
        m=copy.deepcopy(marker);m[key]=value
        assert completion_issues(m,ARM,"planhash")
        tests.append(dict(name=name,passed=True,synthetic=True))
    m=copy.deepcopy(marker);m['stream_state']['sample_index']=7992
    assert completion_issues(m,ARM,"planhash")
    tests.append(dict(name="wrong_stream_cursor",passed=True,synthetic=True))
    original=out/"engineering/dormant_H3_support/training.py.before"
    current=Path(__file__).with_name("training.py")
    assert file_hash(original)==FIXED_TEACHER_TRAINING_SHA256
    assert file_hash(current)==recipe["sources"]["training.py"]
    comparison=teacher_branch_equivalent(original.read_text(),current.read_text())
    assert all(comparison.values()),comparison
    tests.append(dict(name="actual_frozen_teacher_branch_AST_equivalent",passed=True,comparison=comparison))
    changed=current.read_text().replace("loss=.1*raw", "loss=.2*raw")
    assert not teacher_branch_equivalent(original.read_text(),changed)["update_equal"]
    tests.append(dict(name="modified_teacher_weight_rejected",passed=True,synthetic=True))
    changed=current.read_text().replace("sample=aux_sample(step,aux_arm", "sample=aux_sample(step,'TRAJECTORY_REPLAY'")
    assert not teacher_branch_equivalent(original.read_text(),changed)["update_equal"]
    tests.append(dict(name="wrong_teacher_replay_source_rejected",passed=True,synthetic=True))
    scalar=self_test()
    receipt=dict(status="PASS",test_count=len(tests)+len(scalar["cases"]),metadata_tests=tests,scalar_tests=scalar,
        real_repeat_read=False,repeat_authorization_written=False,model_calls=0,backward_calls=0,optimizer_updates=0,
        source_sha256=file_hash(__file__),frozen_H3_trainer_sha256=file_hash(current),
        original_teacher_trainer_sha256=file_hash(original),output_directory=str(folder))
    atomic_json(folder/"metadata_tests.json",receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = metadata_tests(args.out) if args.self_test else run(args.out, live=args.live)
    print(json.dumps({k:result[k] for k in ("status","phase","completed_teacher_steps","safe_inactive_steps","cap_active_steps","missing","output_directory","test_count") if k in result},indent=2))
    raise SystemExit(1 if result["status"].startswith("FAIL") else 0)
