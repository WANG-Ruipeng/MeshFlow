"""CPU-only review of saved training telemetry; never runs a model or OT.

Run with --live while workers are active. Only completed metric rows are
audited against the call ledger; later, in-flight calls are explicitly outside
that snapshot. CSV blanks represent unobserved values, never estimated zeros.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import OUT, START_SHA, atomic_json, array_hash


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _jsonl(path, live, snapshots):
    data = Path(path).read_bytes()
    lines = data.splitlines(keepends=True)
    rows, skipped = [], 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if live and index == len(lines)-1 and not line.endswith(b"\n"):
                skipped = len(line)
            else:
                raise
    snapshots.append(dict(path=str(path), bytes_read=len(data),
        bytes_sha256=_sha(data), complete_rows=len(rows), ignored_partial_tail_bytes=skipped))
    return rows


def _csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    os.replace(temporary, path)


def _stats(values):
    a = np.asarray([x for x in values if x is not None and math.isfinite(float(x))], np.float64)
    if not len(a):
        return dict(n=0, mean=None, median=None, min=None, max=None, p95=None)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
        min=float(a.min()), max=float(a.max()), p95=float(np.quantile(a, .95)))


def _ratio(a, b):
    return float(a/b) if b and math.isfinite(float(a)) and math.isfinite(float(b)) else None


class Checks:
    def __init__(self):
        self.counts = Counter()
        self.errors = []

    def __call__(self, name, passed, context=None):
        self.counts[name] += 1
        if not bool(passed):
            self.errors.append(dict(check=name, context=context))


def _pool(out, check):
    """Read and hash saved supervision; no matching and no random draws."""
    folder = out/"feedback/pool_v1"
    data = (folder/"manifest.json").read_bytes()
    manifest = json.loads(data)
    records = {r["request_id"]: r for r in manifest["records"]}
    payloads = {}
    check("pool_16_real_training_rollouts", len(records) == 16 and manifest["real_feedback_rollouts"] == 16)
    check("pool_GT_only_supervision_no_DEV_CONFIRM", manifest["GT_used_only_for_supervision"] and not manifest["DEV_and_CONFIRM_scores_read"])
    fields = {"S": "SeamCov", "I": "ConflictCov_C_upper", "E": "ExposedRetain_global_lower"}
    for rid, record in records.items():
        raw = (folder/record["payload"]).read_bytes()
        check("frozen_aux_payload_file_hash", _sha(raw) == record["payload_sha256"], rid)
        with np.load(folder/record["payload"], allow_pickle=False) as loaded:
            p = {k: loaded[k] for k in loaded.files}
        payloads[rid] = p
        for key, expected in record["array_sha256"].items():
            check("frozen_aux_array_hash", array_hash(p[key]) == expected, [rid, key])
        check("aux_GT_origin_preparation_audit", record["original_GT_preserved"] and record["C_all51_bitwise"], rid)
    for queue, field in fields.items():
        for uid in manifest["sorted_parents"]:
            candidates = [r for r in records.values() if r["uid"] == uid]
            check("queue_two_noise_candidates", len(candidates) == 2, [queue, uid])
            candidates.sort(key=lambda r: ((-1 if queue == "I" else 1)*r["full_feedback_score"][field], r["candidate"]))
            check("queue_independent_conservative_metric_selection", manifest["queues"][queue][uid]["selected_request_id"] == candidates[0]["request_id"], [queue, uid])
    return manifest, records, payloads, _sha(data)


def _expected_aux(step, arm, seed, pool):
    manifest, records, payloads, pool_sha = pool
    queue = ("S", "I", "E")[(step-1) % 3]
    uid = manifest["sorted_parents"][((step-1)//3) % 8]
    selection = manifest["queues"][queue][uid]
    rid = selection["selected_request_id"]
    record, p = records[rid], payloads[rid]
    t = np.asarray((.6, .8)[((step-1)//24) % 2], np.float32)
    state = (30, 40)[((step-1)//24) % 2]
    n, k = record["N"], record["K"]
    target = np.zeros((n, 9), np.float32)
    target[:k] = p["C"].reshape(k, 9)
    if arm == "TRAJECTORY_REPLAY":
        kind = "trajectory"
        target[k:] = p["trajectory_target"].reshape(n-k, 9)
        xt = p["xt"+str(state)].copy()
        u = np.zeros_like(xt)
        u[k:] = (target[k:]-xt[k:])/(np.float32(1)-t)
    else:
        kind = "teacher"
        target[k:] = p["teacher_target"].reshape(n-k, 9)
        x0 = np.zeros_like(target)
        x0[:k] = target[:k]
        x0[k:] = p["teacher_x0_"+str(seed)].reshape(n-k, 9)
        xt = (np.float32(1)-t)*x0+t*target
        xt[:k] = target[:k]
        u = target-x0
    expected = dict(step=step, arm=arm, training_seed=seed, queue=queue, uid=uid,
        request_id=rid, candidate=record["candidate"], t=float(t), state_index=state,
        pool_manifest_sha256=pool_sha, payload_sha256=record["payload_sha256"],
        selection=selection, correspondence=kind, target_sha256=array_hash(p[kind+"_target"]),
        xt_sha256=array_hash(xt), u_sha256=array_hash(u), C_sha256=array_hash(p["C"]),
        epsilon=record["epsilons"][str(seed)] if kind == "teacher" else None,
        denominator=9*(n-k), lambda_aux=.1, requires_exact_xt_override=True)
    return expected, record["task_id"]


def _plot(destination, steps, gradients):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"FM_CONTROL": "#555555", "REPLAY_FM_CONTROL": "#287BB5", "TRAJECTORY_REPLAY": "#E79535"}
    groups = defaultdict(list)
    for row in steps:
        groups[(row["phase"], row["arm"])].append(row)
    for (phase, arm), rows in groups.items():
        color = colors.get(arm)
        style = "-" if phase == "explore" else "--"
        label = phase+" / "+arm
        x = np.asarray([r["step"] for r in rows])
        def rolling(key):
            a = np.asarray([r[key] if r[key] is not None else np.nan for r in rows], float)
            return np.asarray([np.nanmean(a[max(0, i-24):i+1]) for i in range(len(a))])
        for ax, key in ((axes[0, 0], "FM"), (axes[1, 1], "clip_active"), (axes[2, 0], "input_plus_update_seconds")):
            ax.plot(x, rolling(key), style, color=color, label=label, linewidth=1.2)
        if arm != "FM_CONTROL":
            axes[0, 1].plot(x, rolling("weighted_aux_over_FM"), style, color=color, label=label)
            gr = [r for r in gradients if (r["phase"], r["arm"]) == (phase, arm)]
            axes[1, 0].plot([r["step"] for r in gr], [r["weighted_aux_over_FM_gradient_norm"] for r in gr], "o", color=color, markersize=3, label=label)
            axes[2, 1].plot([r["step"] for r in gr], [r["cosine"] for r in gr], "o", color=color, markersize=3, label=label)
    titles = ["Primary FM: trailing up-to-25-step mean", "Weighted auxiliary / FM: trailing mean",
        "Actual weighted auxiliary / FM gradient norm", "Fraction with pre-clip norm > 1: trailing mean",
        "Input + update seconds: trailing mean", "Actual FM / auxiliary gradient cosine"]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title)
        ax.set_xlabel("Completed optimizer step")
        ax.grid(alpha=.2)
    axes[2, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].set_ylim(-.02, 1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def run(out=OUT, *, live=False):
    started = time.time()
    out = Path(out)
    destination = out/"training_diagnostics"/("live" if live else "final")
    destination.mkdir(parents=True, exist_ok=True)
    check, snapshots = Checks(), []
    metrics, completion = {}, {}
    # Read completion markers before metrics to avoid a live 999/1000 race.
    for path in sorted((out/"training").glob("*_complete.json")):
        completion[path.name.removesuffix("_complete.json")] = _json(path)
    for path in sorted((out/"training").glob("*_metrics.jsonl")):
        rows = _jsonl(path, live, snapshots)
        if rows:
            metrics[(rows[0]["phase"], rows[0]["arm"])] = rows
    plans = {phase: _json(out/"streams"/phase/"train_plan.json") for phase in ("explore", "repeat")}
    for phase, plan in plans.items():
        check("frozen_primary_plan_8000", len(plan["records"]) == 8000 and plan["batch_size"] == 8, phase)
        check("frozen_primary_plan_32_parents", len(plan["object_counts"]) == 32 and set(plan["object_counts"].values()) == {250}, phase)
    seed_fields = ("gaussian_seed", "t_seed", "permutation_seed", "epsilon_seed")
    independence = {key: sum(a[key] != b[key] for a, b in zip(plans["explore"]["records"], plans["repeat"]["records"])) for key in seed_fields}
    check("explore_repeat_stateless_seed_independence", all(v == 8000 for v in independence.values()), independence)
    pool = _pool(out, check) if any(arm != "FM_CONTROL" for _, arm in metrics) else None
    steps, samples, gradients, aux_rows, arm_summaries = [], [], [], [], []
    row_index, aux_tasks = {}, {}
    for (phase, arm), rows in metrics.items():
        label = phase+"_"+arm
        marked = completion.get(label)
        check("metric_step_unique_contiguous", [r["pilot_step"] for r in rows] == list(range(1, len(rows)+1)), label)
        # A 500-step marker is a valid intermediate controller checkpoint.
        completed = bool(marked and marked.get("status") == "PASS" and marked.get("steps") == 1000)
        if marked:
            check("complete_marker_legal_stage", marked.get("status") == "PASS" and marked.get("steps") in (500, 1000), label)
            check("complete_marker_rows_present", len(rows) >= marked["steps"], label)
        if completed:
            check("completed_arm_exactly_1000", len(rows) == 1000 and marked["stream_state"]["batch_index"] == 1000 and marked["stream_state"]["sample_index"] == 8000, label)
        previous_ot = Counter()
        arm_steps, arm_gradients = [], []
        for r in rows:
            step = int(r["pilot_step"])
            scope = [phase, arm, step]
            row_index[(phase, arm, step)] = r
            batch = r["sample_diagnostics"]
            check("eight_different_primary_parents", len(batch) == 8 and len({s["uid"] for s in batch}) == 8, scope)
            check("four_unit_four_uniform_alpha", sum(s["alpha"] == 1 for s in batch) == 4 and all(.9 <= s["alpha"] <= 1.1 for s in batch), scope)
            denominator = sum(9*(s["N"]-s["K"]) for s in batch)
            check("unchanged_primary_global_free_denominator", denominator == r["free_coordinate_denominator"] == r["effective_free_coordinates"], scope)
            check("primary_FM_sum_of_contributions", math.isclose(sum(s["FM_contribution"] for s in batch), r["FM"], rel_tol=1e-12, abs_tol=1e-12), scope)
            check("primary_batch_hash_reconstruction", _sha("".join(s["shared_input_sha256"] for s in batch).encode()) == r["shared_batch_sha256"], scope)
            check("optimizer_and_batch_cursor", r["conditional_total_step"] == 5000+step and r["batch_index"] == step, scope)
            check("finite_loss_and_gradient", all(math.isfinite(r[k]) for k in ("FM", "aux_unweighted", "aux_weighted", "loss", "FM_gradient_norm", "grad_norm_before_clip")), scope)
            for slot, s in enumerate(batch):
                record = plans[phase]["records"][(step-1)*8+slot]
                fields = ("sample_index", "task_id", "uid", "N", "K", "alpha")
                check("primary_sample_matches_frozen_plan", all(s[k] == record[k] for k in fields), scope+[slot])
                check("sample_free_coordinate_count", s["free_coordinates"] == 9*(s["N"]-s["K"]), scope+[slot])
                check("sample_t_valid", 0 <= s["t"] <= 1, scope+[slot])
                samples.append(dict(phase=phase, arm=arm, step=step, slot=slot, **s,
                    primary_global_denominator=denominator,
                    individual_free_MSE=s["FM_contribution"]*denominator/s["free_coordinates"],
                    t_bin=min(4, int(s["t"]*5))))
            counts = r["actual_OT_counts"]
            increments = {k: counts[k]-previous_ot[k] for k in counts}
            check("OT_cumulative_counts_nonnegative", all(v >= 0 for v in increments.values()), scope)
            statuses = Counter(s["ot_cache_status"] for s in batch)
            check("OT_per_step_counts_match_sample_cache", increments.get("OT_cache_hits", 0) == statuses["HIT"] and increments.get("actual_OT_attempts", 0) == increments.get("actual_OT_returns", 0) == statuses["MISS"], scope)
            previous_ot = Counter(counts)
            audit = r.get("aux_audit", {})
            if arm != "FM_CONTROL":
                expected, task = _expected_aux(step, arm, 20261008 if phase == "explore" else 20261009, pool)
                aux_tasks[(phase, arm, step)] = (task, expected["request_id"])
                mismatches = [k for k, value in expected.items() if audit.get(k) != value]
                check("actual_aux_matches_saved_GT_pool_and_queue", not mismatches, dict(scope=scope, mismatched_fields=mismatches))
                check("actual_aux_t_and_free_denominator", r["aux_t"] == expected["t"] and r["aux_free_denominator"] == expected["denominator"], scope)
                actual_weighted = float(np.float32(.1)*np.float32(r["aux_unweighted"]))
                check("actual_aux_fixed_point_one_weight", actual_weighted == r["aux_weighted"], scope)
                aux_rows.append(dict(phase=phase, **audit, audit_status="PASS" if not mismatches else "FAIL"))
                hook_due = step == 1 or step % 25 == 0
                check("gradient_hook_registered_step_scope", ("weighted_aux_gradient_norm" in r) == hook_due and ("FM_aux_gradient_cosine" in r) == hook_due, scope)
                if hook_due and "weighted_aux_gradient_norm" in r:
                    fmnorm, auxnorm = r["FM_gradient_norm"], r["weighted_aux_gradient_norm"]
                    cosine = r["FM_aux_gradient_cosine"] if fmnorm > 0 and auxnorm > 0 else None
                    check("actual_weighted_aux_gradient_positive_finite", math.isfinite(auxnorm) and auxnorm > 0, scope)
                    check("gradient_hook_measurement_provenance", r.get("gradient_measurement") == "Actual extra backward leaf hooks before accumulation; no extra forward/backward or gradient modification", scope)
                    check("cosine_finite_valid_range", cosine is None or (math.isfinite(cosine) and abs(cosine) <= 1.00001), scope)
                    closure = r["grad_norm_before_clip"]**2-fmnorm**2-auxnorm**2-2*fmnorm*auxnorm*cosine if cosine is not None else None
                    g = dict(phase=phase, arm=arm, step=step, queue=audit.get("queue"), aux_t=r["aux_t"],
                        FM_gradient_norm=fmnorm, actual_weighted_aux_gradient_norm=auxnorm,
                        weighted_aux_over_FM_gradient_norm=_ratio(auxnorm, fmnorm), cosine=cosine,
                        cosine_status="OBSERVED" if cosine is not None else "UNDEFINED_ZERO_NORM",
                        total_gradient_norm=r["grad_norm_before_clip"],
                        norm_squared_closure_residual_descriptive=closure,
                        measurement=r["gradient_measurement"])
                    gradients.append(g)
                    arm_gradients.append(g)
            else:
                check("baseline_has_no_aux", not audit and r["aux_weighted"] == r["aux_unweighted"] == 0 and "weighted_aux_gradient_norm" not in r, scope)
            check("total_loss_exact_components", r["loss"] == r["FM"]+r["aux_weighted"], scope)
            row = dict(phase=phase, arm=arm, step=step, conditional_total_step=r["conditional_total_step"],
                FM=r["FM"], aux_unweighted=r["aux_unweighted"], aux_weighted=r["aux_weighted"], loss=r["loss"],
                effective_aux_weight=_ratio(r["aux_weighted"], r["aux_unweighted"]),
                weighted_aux_over_FM=_ratio(r["aux_weighted"], r["FM"]),
                FM_gradient_norm=r["FM_gradient_norm"], weighted_aux_gradient_norm=r.get("weighted_aux_gradient_norm"),
                FM_aux_gradient_cosine=r.get("FM_aux_gradient_cosine"), grad_norm_before_clip=r["grad_norm_before_clip"],
                clip_active=int(r["grad_norm_before_clip"] > 1.),
                estimated_clip_coefficient=min(1., 1./(r["grad_norm_before_clip"]+1e-6)),
                free_coordinate_denominator=denominator, aux_free_denominator=r.get("aux_free_denominator"),
                aux_t=r.get("aux_t"), queue=audit.get("queue"), aux_uid=audit.get("uid"),
                aux_request_id=audit.get("request_id"), aux_candidate=audit.get("candidate"),
                input_seconds=r["OT_and_input_seconds"], forward_seconds=r["forward_seconds"],
                backward_seconds=r["backward_seconds"], optimizer_seconds=r["optimizer_seconds"],
                update_seconds=r["update_total_seconds"],
                input_plus_update_seconds=r["OT_and_input_seconds"]+r["update_total_seconds"],
                OT_hits=increments.get("OT_cache_hits", 0), OT_attempts=increments.get("actual_OT_attempts", 0),
                OT_returns=increments.get("actual_OT_returns", 0), shared_batch_sha256=r["shared_batch_sha256"],
                coupled_batch_sha256=r["coupled_batch_sha256"], time_unix=r["time_unix"])
            steps.append(row)
            arm_steps.append(row)
        arm_summaries.append(dict(phase=phase, arm=arm, steps=len(rows),
            status="COMPLETE_1000" if completed else "LIVE_PARTIAL" if live else "INCOMPLETE",
            last_completion_marker=marked, clipping_fraction=_stats([r["clip_active"] for r in arm_steps])["mean"],
            FM=_stats([r["FM"] for r in arm_steps]), aux_weighted=_stats([r["aux_weighted"] for r in arm_steps]),
            first_25_FM=_stats([r["FM"] for r in arm_steps[:25]]), last_25_FM=_stats([r["FM"] for r in arm_steps[-25:]]),
            weighted_aux_over_FM=_stats([r["weighted_aux_over_FM"] for r in arm_steps]),
            actual_weighted_aux_norm=_stats([r["actual_weighted_aux_gradient_norm"] for r in arm_gradients]),
            FM_aux_cosine=_stats([r["cosine"] for r in arm_gradients]),
            negative_cosine_fraction=_stats([r["cosine"] < 0 for r in arm_gradients if r["cosine"] is not None])["mean"],
            hook_observation_count=len(arm_gradients), actual_OT_counts=rows[-1]["actual_OT_counts"],
            input_seconds=_stats([r["input_seconds"] for r in arm_steps]),
            input_plus_update_seconds=_stats([r["input_plus_update_seconds"] for r in arm_steps]),
            sum_input_plus_update_seconds=sum(r["input_plus_update_seconds"] for r in arm_steps)))

    pair_rows = []
    for phase in plans:
        arms = sorted(a for p, a in metrics if p == phase)
        for a, b in itertools.combinations(arms, 2):
            for step in range(1, min(len(metrics[(phase, a)]), len(metrics[(phase, b)]))+1):
                x, y = row_index[(phase, a, step)], row_index[(phase, b, step)]
                fields = ("shared_batch_sha256", "coupled_batch_sha256", "free_coordinate_denominator")
                equal = all(x[k] == y[k] for k in fields)
                equal_samples = [s["shared_input_sha256"] for s in x["sample_diagnostics"]] == [s["shared_input_sha256"] for s in y["sample_diagnostics"]]
                check("same_phase_all_primary_batch_inputs_identical", equal and equal_samples, [phase, a, b, step])
                pair_rows.append(dict(phase=phase, arm_a=a, arm_b=b, step=step, batch_fields_equal=equal,
                    all_eight_sample_hashes_equal=equal_samples, shared_batch_sha256=x["shared_batch_sha256"]))
    for arm in set(a for _, a in metrics):
        if ("explore", arm) in metrics and ("repeat", arm) in metrics:
            for x, y in zip(metrics[("explore", arm)], metrics[("repeat", arm)]):
                check("actual_repeat_inputs_differ_from_explore", x["shared_batch_sha256"] != y["shared_batch_sha256"], [arm, x["pilot_step"]])

    ledger_counts = defaultdict(Counter)
    begin, terminal = {}, {}
    outside, callfiles = Counter(), sorted((out/"ledger").glob("calls_*.jsonl"))
    for path in callfiles:
        for entry in _jsonl(path, live, snapshots):
            event = entry.get("event")
            if event not in ("CALL_BEGIN", "CALL_RETURN", "CALL_FAIL"):
                continue
            ctx = entry.get("context") or {}
            key = (ctx.get("phase"), ctx.get("arm"), ctx.get("pilot_step"))
            if key not in row_index:
                outside[event+":"+entry.get("kind", "unknown")] += 1
                continue
            cid = entry["call_id"]
            holder = begin if event == "CALL_BEGIN" else terminal
            check("training_call_id_no_duplicate_event", cid not in holder, [str(path), cid, event])
            holder[cid] = entry
            micro = ctx.get("microbatch")
            slot = "aux" if micro == "aux" else "primary" if isinstance(micro, int) else "update"
            ledger_counts[key][(entry["kind"], slot, event)] += 1
            if slot == "primary":
                samples_at_step = row_index[key]["sample_diagnostics"]
                check("ledger_primary_microbatch_identity", 0 <= micro < 8 and ctx.get("sample_index") == samples_at_step[micro]["sample_index"] and ctx.get("shared_input_sha256") == samples_at_step[micro]["shared_input_sha256"], [cid, event])
            elif slot == "aux":
                check("ledger_aux_identity_task_or_full_request", ctx.get("request_id") in aux_tasks.get(key, ()), [cid, event])
    check("recorded_completed_step_calls_no_pending_or_orphans", set(begin) == set(terminal), dict(pending=sorted(set(begin)-set(terminal)), orphans=sorted(set(terminal)-set(begin))))
    for cid, entry in terminal.items():
        check("completed_step_model_and_update_calls_returned", entry["event"] == "CALL_RETURN", cid)
        if cid in begin:
            check("call_return_preserves_kind_and_context", entry["kind"] == begin[cid]["kind"] and entry.get("context") == begin[cid].get("context"), cid)
    ledger_rows = []
    for key, metric in row_index.items():
        phase, arm, step = key
        counter = ledger_counts[key]
        row = dict(phase=phase, arm=arm, step=step)
        expected = {("training_forward", "primary"): 8, ("training_backward", "primary"): 8,
            ("training_forward", "aux"): int(arm != "FM_CONTROL"), ("training_backward", "aux"): int(arm != "FM_CONTROL"),
            ("optimizer_update", "update"): 1}
        for (kind, slot), count in expected.items():
            for event in ("CALL_BEGIN", "CALL_RETURN", "CALL_FAIL"):
                number = counter[(kind, slot, event)]
                row[kind+"_"+slot+"_"+event] = number
                check("step_actual_F_B_update_budget_exact", number == (0 if event == "CALL_FAIL" else count), [phase, arm, step, kind, slot, event])
        check("step_no_unexpected_training_calls", sum(counter.values()) == 2*sum(expected.values()), [phase, arm, step])
        ledger_rows.append(row)

    starts = []
    start_path = out/"decisions/training_starts.jsonl"
    if start_path.exists():
        starts = _jsonl(start_path, live, snapshots)
        for entry in starts:
            check("every_arm_shared_START_origin", entry.get("source_checkpoint_sha256") == START_SHA, [entry.get("phase"), entry.get("arm"), entry.get("start_step")])
    load_audits = []
    for path in sorted((out/"audit").glob("*_train_load.json")):
        item = _json(path)
        meta, hashes = item.get("metadata", {}), item.get("hashes", {})
        initial = meta.get("feedback_step") == 0
        check("training_AdamW_restored_no_reset", item.get("optimizer_restored") is True and meta.get("optimizer_reset") is False, str(path))
        check("training_load_START_or_recorded_resume_origin", hashes.get("checkpoint_sha256") == START_SHA if initial else meta.get("source_checkpoint_sha256") == START_SHA, str(path))
        load_audits.append(dict(path=str(path), initial_from_START=initial,
            checkpoint_sha256=hashes.get("checkpoint_sha256"), model_state_sha256=hashes.get("model_state_sha256"),
            optimizer_state_sha256=meta.get("optimizer_state_sha256"), feedback_step=meta.get("feedback_step"),
            optimizer_restored=item.get("optimizer_restored")))
    for phase, arm in metrics:
        check("observed_arm_has_START_dispatch_receipt", any(x.get("phase") == phase and x.get("arm") == arm and x.get("start_step") == 0 for x in starts), [phase, arm])

    t_primary, t_aux = [], []
    sample_groups, aux_groups = defaultdict(list), defaultdict(list)
    for row in samples:
        sample_groups[(row["phase"], row["arm"], row["t_bin"])].append(row)
    for (phase, arm, bin_index), rows in sorted(sample_groups.items()):
        numerator = sum(r["FM_contribution"]*r["primary_global_denominator"] for r in rows)
        t_primary.append(dict(phase=phase, arm=arm, t_lower=bin_index/5, t_upper=(bin_index+1)/5,
            samples=len(rows), free_coordinates=sum(r["free_coordinates"] for r in rows),
            coordinate_weighted_MSE=numerator/sum(r["free_coordinates"] for r in rows),
            sample_mean_MSE=float(np.mean([r["individual_free_MSE"] for r in rows]))))
    for row in steps:
        if row["aux_t"] is not None:
            aux_groups[(row["phase"], row["arm"], row["aux_t"])].append(row)
    for (phase, arm, t), rows in sorted(aux_groups.items()):
        fm, aux = np.mean([r["FM"] for r in rows]), np.mean([r["aux_weighted"] for r in rows])
        t_aux.append(dict(phase=phase, arm=arm, aux_t=t, steps=len(rows), FM_mean=float(fm),
            weighted_aux_mean=float(aux), weighted_aux_to_FM_ratio_of_means=_ratio(aux, fm),
            weighted_aux_to_FM_mean_of_step_ratios=_stats([r["weighted_aux_over_FM"] for r in rows])["mean"],
            clipping_fraction=float(np.mean([r["clip_active"] for r in rows]))))
    for name, values in (("per_step", steps), ("primary_samples", samples), ("gradient_observations", gradients),
        ("aux_audit", aux_rows), ("primary_time_strata", t_primary), ("aux_time_strata", t_aux),
        ("shared_input_pairs", pair_rows), ("ledger_by_step", ledger_rows)):
        _csv(destination/(name+".csv"), values)
    _plot(destination/"learning_diagnostics.png", steps, gradients)
    incomplete = [r["phase"]+"_"+r["arm"] for r in arm_summaries if r["status"] == "INCOMPLETE"]
    status = "FAIL" if check.errors else "LIVE_PASS" if live else "INCOMPLETE" if incomplete else "PASS"
    source_files = ("training.py", "stream.py", "replay.py", "training_diagnostics.py")
    summary = dict(status=status, live=live, started_unix=started, finished_unix=time.time(),
        CPU_only=True, model_calls=0, backward_calls=0, optimizer_updates=0, OT_matching_calls=0,
        checks=sum(check.counts.values()), checks_by_name=dict(check.counts), errors=check.errors,
        observed_arms=arm_summaries, incomplete_actual_arms=incomplete,
        not_yet_observed_explore_arms=[a for a in ("FM_CONTROL", "REPLAY_FM_CONTROL", "TRAJECTORY_REPLAY") if ("explore", a) not in metrics],
        independent_random_seed_counts=independence, matched_primary_batch_pairs=len(pair_rows),
        audited_actual_call_pairs=len(begin), calls_outside_completed_metric_snapshot=dict(outside),
        source_START_sha256=START_SHA, load_audits=load_audits, snapshots=snapshots,
        primary_time_strata=t_primary, aux_time_strata=t_aux,
        source_hashes={str(Path(__file__).with_name(p)): _sha(Path(__file__).with_name(p).read_bytes()) for p in source_files},
        definitions=dict(primary="FM is the sum of 8 free-coordinate squared errors divided by 9*sum(N-K).",
            auxiliary="Auxiliary has its own 9*(N-K) denominator and fixed weight 0.1.",
            gradient="Actual weighted auxiliary backward leaf hooks before accumulation; never a difference of gradient norms.",
            cosine_scope="Only step 1 and each multiple of 25. Other steps are NA. Zero norm means UNDEFINED.",
            clipping="Fraction of rows with total pre-clip norm > 1; coefficient is reconstructed from PyTorch clip formula, not observed parameter delta.",
            time="Input + update excludes checkpoint I/O, DEV sampling, loading and idle time; concurrent load can affect wall timing.",
            time_strata="Primary bins show actual t and coordinate-weighted MSE. Auxiliary t strata compare weighted aux with that step's primary batch, not a same-example loss.",
            independence="Repeat uses independent registered main-stream/teacher-epsilon seeds; fixed START trajectory feedback pool and auxiliary selection are intentionally shared.",
            ledger="Every completed metric step is checked against actual BEGIN/RETURN/FAIL. Later calls and inference stages are outside this snapshot.",
            provenance="Source hashes describe code on disk at diagnostic time. Actual loaded-worker source provenance is in worker snapshots; no claim that a running worker hot-reloaded later edits.",
            scope="No DEV or CONFIRM outputs or metrics were read. Losses and gradients are learning diagnostics, not generated-control efficacy or a reason to change the frozen recipe."),
        output_directory=str(destination))
    atomic_json(destination/"summary.json", summary)
    lines = ["# Saved training diagnostics", "", "Status: "+status, "",
        "CPU-only review; no model, backward, update, matching, or new generation calls.", "",
        "| Phase / arm | Steps | FM mean | Weighted aux / FM mean ratio | Clip fraction | Hook observations | Input + update median s |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in arm_summaries:
        ratio = _ratio(arm["aux_weighted"]["mean"], arm["FM"]["mean"])
        lines.append(f"| {arm['phase']} / {arm['arm']} | {arm['steps']} | {arm['FM']['mean']:.6g} | {ratio:.6g} | {arm['clipping_fraction']:.4f} | {arm['hook_observation_count']} | {arm['input_plus_update_seconds']['median']:.4f} |")
    lines += ["", f"Checks: {summary['checks']}; failures: {len(check.errors)}; identical primary batch comparisons: {len(pair_rows)}.", "",
        "Auxiliary gradients are actual weighted extra-backward hook measurements. Cosines are observed only at step 1 / every 25 steps; missing observations are not zeros.", "",
        "Trajectory replay uses a fixed START feedback pool; it is not an on-policy rollout of the changing model. Main-stream repeat seeds are independent while this supervision pool is intentionally shared.", "",
        "These values describe the learning process. Different auxiliary state distributions make raw loss magnitudes incomparable as measures of generated-control quality. No DEV effect, recipe change, or training extension is inferred.", "",
        "Files: per_step.csv, primary_samples.csv, primary_time_strata.csv, aux_time_strata.csv, gradient_observations.csv, aux_audit.csv, shared_input_pairs.csv, ledger_by_step.csv, learning_diagnostics.png, summary.json."]
    (destination/"README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    result = run(args.out, live=args.live)
    print(json.dumps({k: result[k] for k in ("status", "checks", "errors", "output_directory")}, indent=2))
    raise SystemExit(1 if result["status"] == "FAIL" else 0)
