"""Read completed collector tables; no models, meshes, scoring or pilot imports.

Every metric keeps its original name and units. Primary aggregation is equal
noise within condition, equal conditions within parent, then equal parents.
Request means/sums and support strata remain separate, explicitly labelled.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics

from .stream_audit import paired_stream, first_identity_micros, registration_identity

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / "experiments/r2_bench_joint_shift_v1"
SCHEMA = "BENCH_JOINT_SHIFT_OFFLINE_ANALYSIS_V1"
CANDIDATE = "JOINT_SHIFT_S1500"
REFERENCES = ("RELATION_V1_S1500", "USER_V2_S1500")
REFERENCE_ROOT = ROOT / "experiments/r2_bench_relation_training_v1"
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")
KEYS = (
    "CF_illegal_affected_free_fraction", "FF_illegal_affected_free_fraction",
    "ConflictCov_C", "ConflictCov_C_cross", "ConflictCov_C_dup",
    "CF_illegal_pair_count", "FF_illegal_pair_count",
    "fixed_parent_FF_illegal_affected_count", "fixed_parent_free_face_denominator",
    "actual_free_faces", "deleted_free_faces", "retained_bulk_faces", "retained_seam_faces",
    "free_total_surface_area", "retained_bulk_area", "retained_seam_area",
    "free_minimum_angle_lt5_fraction", "free_angle_p05_degrees", "free_q_gt10_fraction",
    "free_exact_zero_area_count", "free_near_degenerate_count", "nonmanifold_edges",
    "hinge_diag_isolated_1face_edge_components", "boundary_edges",
    "C_F_valid_edge_count", "C_F_valid_edge_total_length",
    "legal_seam_bulk_edge_count", "legal_seam_bulk_edge_length",
    "free_only_reached_bulk_area_fraction", "bulk_reach_ge_0_30_descriptive",
    "A_VMR__L_tau_0.005", "PMR__L_tau_0.005",
    "illegal_pairs_bulk-bulk", "illegal_pairs_seam-bulk", "illegal_pairs_seam-seam",
)


def resolve(value):
    s = str(value).replace("\\", "/")
    if os.name == "nt" and s.startswith("/mnt/"):
        return Path(s[5].upper() + ":/" + s[7:])
    if os.name != "nt" and len(s) > 2 and s[1] == ":":
        return Path("/mnt/" + s[0].lower() + s[2:])
    return Path(s)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def num(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)


def avg(values):
    v = [float(x) for x in values if num(x)]
    return statistics.mean(v) if v else None


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_bytes(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refusing to overwrite different analysis: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(content)


def write_json(path, data):
    write_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()})
    write_bytes(path, buf.getvalue().encode("utf-8-sig"))


def panel_rows(rows):
    """ALL plus historical split, separately including part breakdowns."""
    parts = sorted({r["part_type"] for r in rows})
    for group in ("ALL", "TRAIN", "DEV"):
        group_rows = [r for r in rows if group == "ALL" or r["split"] == group]
        for part in ("ALL_PARTS", *parts):
            subset = [r for r in group_rows if part == "ALL_PARTS" or r["part_type"] == part]
            if subset:
                yield dict(panel="HISTORICAL_TRAIN" if group == "TRAIN" else group, part_type=part), subset


def statistics_for(rows, metric):
    valid = [r for r in rows if num(r.get(metric))]
    conditions = defaultdict(list)
    for r in valid:
        conditions[(r["parent_id"], r["condition_id"])].append(r[metric])
    parents = defaultdict(list)
    for (pid, _), values in conditions.items():
        parents[pid].append(avg(values))
    pvalues = [avg(v) for v in parents.values()]
    values = [float(r[metric]) for r in valid]
    return dict(
        metric=metric, parent_macro=avg(pvalues), request_mean=avg(values),
        request_sum=sum(values) if values else None,
        request_min=min(values) if values else None, request_max=max(values) if values else None,
        request_median=statistics.median(values) if values else None,
        requested_rows=len(rows), valid_requests=len(values), missing_requests=len(rows)-len(values),
        valid_conditions=len(conditions), valid_parents=len(parents),
        all_conditions=len({r["condition_id"] for r in rows}), all_parents=len({r["parent_id"] for r in rows}),
        support_rule="FINITE_ORIGINAL_VALUES_ONLY_NO_ZERO_FILL",
    )


def strata(rows):
    yield "ALL_RECORDED_SUPPORT", rows
    yield "PASS_ONLY", [r for r in rows if r.get("I_status") == "PASS"]
    yield "LOW_SUPPORT_ONLY", [r for r in rows if r.get("I_status") == "LOW_SUPPORT"]
    yield "OTHER_STATUS_ONLY", [r for r in rows if r.get("I_status") not in {"PASS", "LOW_SUPPORT"}]


def summarize(rows, metrics):
    all_metrics, support, key_wide = [], [], []
    for label in sorted({r["label"] for r in rows}):
        for stage in STAGES:
            selected = [r for r in rows if r["label"] == label and r["stage"] == stage]
            for panel, subset in panel_rows(selected):
                identity = dict(label=label, stage=stage, **panel)
                wide = dict(**identity, requests=len(subset), parents=len({r["parent_id"] for r in subset}),
                            I_status_counts=dict(Counter(r.get("I_status", "MISSING") for r in subset)),
                            seam_status_counts=dict(Counter(r.get("seam_to_bulk_status", "MISSING") for r in subset)),
                            applicable_requests=sum(r.get("seam_applicability") == "APPLICABLE" for r in subset))
                for m in metrics:
                    stat = statistics_for(subset, m)
                    all_metrics.append(dict(**identity, **stat))
                    if m in KEYS:
                        wide[m] = stat["parent_macro"]
                        wide[m + "__valid_requests"] = stat["valid_requests"]
                key_wide.append(wide)
                for support_name, supported in strata(subset):
                    for m in ("ConflictCov_C", "ConflictCov_C_cross", "ConflictCov_C_dup", "ConflictCov_C_upper"):
                        if m in metrics:
                            support.append(dict(**identity, I_support=support_name, **statistics_for(supported, m)))
    return all_metrics, key_wide, support


def pair_rows(rows, metrics, candidate=CANDIDATE, references=REFERENCES):
    by = {(r["label"], r["stage"], r["request_id"]): r for r in rows}
    pairs = []
    for b in rows:
        if b["label"] != candidate:
            continue
        for reference in references:
            a = by[(reference, b["stage"], b["request_id"])]
            for k in ("condition_id", "parent_id", "noise_index", "N", "K", "part_type", "split"):
                if a[k] != b[k]:
                    raise ValueError("Unpaired request metadata: " + k)
            p = {k: b[k] for k in ("stage", "request_id", "condition_id", "parent_id", "noise_index", "N", "K", "split", "part_type")}
            p.update(reference=reference, candidate=candidate,
                     reference_I_status=a.get("I_status", "MISSING"), candidate_I_status=b.get("I_status", "MISSING"),
                     both_I_PASS=a.get("I_status") == b.get("I_status") == "PASS",
                     reference_seam_status=a.get("seam_to_bulk_status", "MISSING"),
                     candidate_seam_status=b.get("seam_to_bulk_status", "MISSING"))
            for m in metrics:
                av, bv = a.get(m), b.get(m)
                p[m + "__reference"], p[m + "__candidate"] = av, bv
                p[m + "__delta"] = float(bv-av) if num(av) and num(bv) else None
            pairs.append(p)
    return pairs


def pair_summaries(pairs, metrics):
    out = []
    for reference in sorted({r["reference"] for r in pairs}):
        for stage in STAGES:
            selected = [r for r in pairs if r["reference"] == reference and r["stage"] == stage]
            for panel, subset in panel_rows(selected):
                for m in metrics:
                    domains = [("ALL_PAIRED_SUPPORT", subset)]
                    if m.startswith("ConflictCov_C"):
                        domains.append(("BOTH_PASS_ONLY", [r for r in subset if r["both_I_PASS"]]))
                    for domain, group in domains:
                        stat = statistics_for(group, m+"__delta")
                        ds = [r[m+"__delta"] for r in group if num(r.get(m+"__delta"))]
                        out.append(dict(reference=reference, candidate=CANDIDATE, stage=stage, **panel,
                                        pair_support=domain, original_metric=m, **stat,
                                        decreased_requests=sum(x < 0 for x in ds),
                                        unchanged_requests=sum(x == 0 for x in ds),
                                        increased_requests=sum(x > 0 for x in ds),
                                        interpretation="CANDIDATE_MINUS_REFERENCE; NO_AUTOMATIC_BETTER_WORSE_LABEL"))
    return out


def parent_tables(rows, pairs, metrics):
    per_parent, paired = [], []
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["label"], r["stage"], r["split"], r["parent_id"])].append(r)
    for (label, stage, split, parent), group in sorted(grouped.items()):
        for m in metrics:
            per_parent.append(dict(label=label, stage=stage, split=split, parent_id=parent,
                                   **statistics_for(group, m)))
    grouped = defaultdict(list)
    for r in pairs:
        grouped[(r["reference"], r["candidate"], r["stage"], r["split"], r["parent_id"])].append(r)
    for (reference, candidate, stage, split, parent), group in sorted(grouped.items()):
        for m in metrics:
            for suffix in ("reference", "candidate", "delta"):
                paired.append(dict(reference=reference, candidate=candidate, stage=stage,
                                   split=split, parent_id=parent, value_type=suffix, original_metric=m,
                                   **statistics_for(group, m+"__"+suffix)))
    return per_parent, paired


def load_inputs(experiment):
    completion_path = experiment / "results/completion.json"
    required = [completion_path, experiment / "results/per_output.csv", experiment / "eval_manifest.json",
                experiment / "data/evaluation_membership.json", experiment / "registration.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return dict(status="WAITING_FOR_COMPLETED_COLLECTOR", missing=missing)
    c = read(completion_path)
    if c.get("status") != "COMPLETE" or c.get("schema") != "BENCH_JOINT_SHIFT_COLLECTION_V1":
        raise ValueError("Collector is not complete or schema is different")
    if not c.get("whole_mesh_boundary_edges_preserved") or c.get("input_C_boundary_field") != "C_boundary_edges":
        raise ValueError("Boundary metric/metadata separation not authenticated")
    hashes = {str(p): sha(p) for p in required}
    output_path = experiment / "results/per_output.csv"
    expected = next((h for p, h in c["files"].items() if resolve(p).resolve() == output_path.resolve()), None)
    if expected != sha(output_path):
        raise ValueError("Collected table hash mismatch")
    if c["eval_manifest_sha256"] != sha(experiment / "eval_manifest.json"):
        raise ValueError("Evaluation manifest hash mismatch")
    if c["registration_sha256"] != sha(experiment / "registration.json"):
        raise ValueError("Training registration hash mismatch")
    metrics = c["metric_fields"]
    if len(metrics) != 177 or len(set(metrics)) != 177:
        raise ValueError("Expected the 177 distinct original numeric fields")
    if "boundary_edges" not in metrics or "C_boundary_edges" in metrics:
        raise ValueError("Wrong metric namespace")
    rows = csv_rows(output_path)
    if len(rows) != 432 or c["rows"] != 432:
        raise ValueError("Expected all 432 original numeric rows")
    qmap = {q["request_id"]: q for q in read(experiment / "eval_manifest.json")["requests"]}
    reference_queries = read(REFERENCE_ROOT / "eval_manifest.json")["requests"]
    if read(experiment / "eval_manifest.json")["requests"] != reference_queries:
        raise ValueError("Historical 48-request panel changed")
    hashes[str(REFERENCE_ROOT / "eval_manifest.json")] = sha(REFERENCE_ROOT / "eval_manifest.json")
    membership = {r["request_id"]: r for r in read(experiment / "data/evaluation_membership.json")}
    expected_keys = {(label, stage, rid) for label in (*REFERENCES, CANDIDATE) for stage in STAGES for rid in qmap}
    seen = set()
    for r in rows:
        identity = (r["label"], r["stage"], r["request_id"])
        if identity in seen:
            raise ValueError("Duplicate original score row")
        seen.add(identity)
        q = qmap[r["request_id"]]
        for field in ("condition_id", "parent_id", "split", "part_type"):
            if r[field] != q[field]:
                raise ValueError("Collector/manifest mismatch: " + field)
        for field in ("N", "K", "noise_index"):
            r[field] = int(r[field])
            if r[field] != int(q[field]):
                raise ValueError("Collector/manifest count mismatch: " + field)
        for m in metrics:
            v = r.get(m)
            if v is None:
                raise ValueError("Missing registered numeric column: " + m)
            r[m] = None if v == "" else float(v)
            if r[m] is not None and not math.isfinite(r[m]):
                raise ValueError("Nonfinite stored numeric value: " + m)
        if r.get("I_status") != r.get("ConflictCov_C_status"):
            raise ValueError("I status aliases disagree")
        for field in ("in_current_training_parent_pool", "exact_current_training_condition"):
            r[field] = membership[r["request_id"]][field]
    if seen != expected_keys:
        raise ValueError("Incomplete registered label/request/stage coverage")
    return dict(status="READY", rows=rows, metric_fields=metrics, source_hashes=hashes)


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def training_windows(logs, window=100):
    out = []
    for offset in range(0, len(logs), window):
        group = logs[offset:offset+window]
        r = dict(first_step=group[0]["local_step"], last_step=group[-1]["local_step"], updates=len(group),
                 objective_mean=avg([x["objective"] for x in group]),
                 update_seconds_mean=avg([x["update_seconds"] for x in group]),
                 gradient_norm_mean=avg([x["generator_gradient_norm"] for x in group]))
        for name, sse, denom in (("all_free", "all_free_velocity_sse", "all_free_scalar_count"),
                                  ("bulk", "bulk_velocity_sse", "bulk_scalar_count"),
                                  ("seam_tip", "seam_tip_velocity_sse", "seam_tip_scalar_count")):
            numerator = sum(x[sse] for x in group)
            denominator = sum(x[denom] for x in group)
            r[name+"_velocity_sse"] = numerator
            r[name+"_scalar_count"] = denominator
            r[name+"_velocity_MSE"] = numerator / denominator if denominator else None
        for name, sse, denom in (("one_step", "endpoint_sse", "all_free_scalar_count"),
                                  ("bulk", "bulk_endpoint_sse", "bulk_scalar_count"),
                                  ("seam_tip", "seam_tip_endpoint_sse", "seam_tip_scalar_count")):
            numerator = sum(x[sse] for x in group)
            denominator = sum(x[denom] for x in group)
            r[name+"_endpoint_sse"] = numerator
            r[name+"_endpoint_scalar_count"] = denominator
            r[name+"_endpoint_MSE"] = numerator / denominator if denominator else None
        out.append(r)
    return out


def training_data(experiment):
    folder = experiment / "training/JOINT_SHIFT_SEAM"
    paths = [folder / n for n in ("status.json", "training_log.jsonl", "exposure.jsonl", "calls.jsonl")]
    paths.extend([experiment / "data/schedule.json", experiment / "data/exposure_plan.json"])
    paths.extend([REFERENCE_ROOT / "data/schedule.json",
                  REFERENCE_ROOT / "training/RELATION_V1_SEAM/exposure.jsonl",
                  experiment / "registration.json", REFERENCE_ROOT / "registration.json"])
    if any(not p.exists() for p in paths):
        raise ValueError("Training ledger not complete")
    status = read(paths[0])
    if status.get("status") != "COMPLETE" or status.get("local_step") != 1500:
        raise ValueError("Final training is not COMPLETE at step1500")
    logs, exposure = jsonl(paths[1]), jsonl(paths[2])
    if [r["local_step"] for r in logs] != list(range(1, 1501)):
        raise ValueError("Training curve has missing/duplicate updates")
    expected_positions = [(step, mi) for step in range(1,1501) for mi in range(8)]
    if len(exposure) != 12000 or [(r["local_step"],r["sample_index"]) for r in exposure] != expected_positions:
        raise ValueError("Exposure ledger is not the single 12000-sample stream")
    schedule = read(paths[4])
    planned = [(entry["step"], sample) for entry in schedule["steps"] for sample in entry["samples"]]
    if len(planned) != len(exposure):
        raise ValueError("Frozen schedule length differs from exposure")
    schedule_comparisons = 0
    for (step, sample), actual in zip(planned, exposure):
        if step != actual["local_step"] or actual["global_step"] != 12000+step:
            raise ValueError("Schedule/update identity mismatch")
        for key,value in sample.items():
            if key not in actual or actual[key] != value:
                raise ValueError("Frozen schedule/exposure mismatch: " + key)
            schedule_comparisons += 1
    calls = jsonl(paths[3])
    begins, ends = {}, {}
    for r in calls:
        if r["event"] not in {"CALL_BEGIN", "CALL_DONE"}:
            raise ValueError("Failure/unrecognized call event requires independent audit")
        key = (r["attempt_id"], r["call_id"])
        target = begins if r["event"] == "CALL_BEGIN" else ends
        if key in target:
            raise ValueError("Duplicate model/update ledger event")
        target[key] = r
    if set(begins) != set(ends):
        raise ValueError("Unclosed training calls")
    count = Counter()
    for key,start in begins.items():
        end = ends[key]
        if start["kind"] != end["kind"] or start["context"] != end["context"]:
            raise ValueError("Call begin/end identity mismatch")
        count[start["kind"]] += 1
    expected_calls = dict(train_forward=12000, train_backward=12000, optimizer_update=1500)
    if dict(count) != expected_calls:
        raise ValueError("Actual call ledger differs from authorized budget")
    for kind,n in expected_calls.items():
        if status["calls"]["successful"].get(kind) != n or status["calls"]["attempted"].get(kind) != n:
            raise ValueError("Status/call ledger disagreement")
    if status["calls"].get("failed") or status["calls"].get("pending_call_ids"):
        raise ValueError("Training has failure or unresolved call")
    windows = training_windows(logs)
    pc = Counter(r["parent_id"] for r in exposure)
    cc = Counter((r["parent_id"], r["condition_id"], r["part_type"]) for r in exposure)
    exposure_plan = read(paths[5])
    if dict(pc) != exposure_plan["parent"] or dict(Counter(r["condition_id"] for r in exposure)) != exposure_plan["condition"]:
        raise ValueError("Actual exposure differs from frozen parent/condition plan")
    reference_exposure = jsonl(paths[7])
    stream = paired_stream(schedule, read(paths[6]), exposure, reference_exposure)
    initialization = registration_identity(read(paths[8]), read(paths[9]))
    first_micros = first_identity_micros(exposure, reference_exposure)
    result = dict(updates=1500, exposures=len(exposure), parents=len(pc), conditions=len(cc),
                  parent_exposure_min=min(pc.values()), parent_exposure_max=max(pc.values()),
                  first100=windows[0], last100=windows[-1],
                  max_reserved_bytes=max(r["peak_reserved_bytes"] for r in logs),
                  memory_scope="Training-log PyTorch allocator reserved maximum; not whole-process VRAM",
                  actual_call_counts=dict(count), matched_call_pairs=len(begins),
                  matched_schedule_fields=schedule_comparisons, actual_exposure_matches_plan=True,
                  paired_stream_audit=stream, initialization_identity=initialization,
                  first_identity_micro_audit=first_micros,
                  science_audit_status=first_micros["status"],
                  metric_note="Velocity MSE=sum original SSE/sum original scalar count; objective is mean per-update objective. No convergence verdict. Relative to RELATION the old input stream is identical, but half the target coordinates change; loss is not a strict same-target paired comparison.")
    return result, windows, [dict(parent_id=p, exposures=n) for p, n in sorted(pc.items())], [
        dict(parent_id=p, condition_id=c, part_type=t, exposures=n) for (p,c,t),n in sorted(cc.items())
    ], {str(p): sha(p) for p in paths}


def run(experiment=DEFAULT, check_inputs=False):
    experiment = resolve(experiment).resolve()
    data = load_inputs(experiment)
    if data["status"] != "READY":
        return data
    if check_inputs:
        return dict(status="READY", rows=len(data["rows"]), numeric_metric_fields=len(data["metric_fields"]), model_calls=0, scoring_calls=0)
    dest = experiment / "analysis_joint_shift_v1"
    done = dest / "completion.json"
    if done.exists():
        old = read(done)
        if old["script_sha256"] != sha(__file__) or old["collector_input_hashes"] != data["source_hashes"]:
            raise ValueError("Completed analysis identity differs")
        for p,h in old["analysis_code_hashes"].items():
            if sha(resolve(p)) != h:
                raise ValueError("Independent analysis source changed: " + p)
        for p,h in old["training_input_hashes"].items():
            if sha(resolve(p)) != h:
                raise ValueError("Completed training source changed: " + p)
        for p,h in old["files"].items():
            if sha(resolve(p)) != h:
                raise ValueError("Completed analysis output changed: " + p)
        return old
    rows, metrics = data["rows"], data["metric_fields"]
    summary, key, support = summarize(rows, metrics)
    pairs = pair_rows(rows, metrics)
    pair_summary = pair_summaries(pairs, metrics)
    parents, paired_parents = parent_tables(rows, pairs, metrics)
    training, windows, parent_exposure, condition_exposure, training_hashes = training_data(experiment)
    tables = dict(all_metrics_summary=summary, key_metrics=key, I_support_summary=support,
                  paired_per_output=pairs, paired_metric_summary=pair_summary,
                  per_parent_all_metrics=parents, paired_per_parent_all_metrics=paired_parents,
                  per_output_with_membership=rows, training_windows=windows,
                  training_parent_exposure=parent_exposure, training_condition_exposure=condition_exposure,
                  first_update_identity_micro_comparison=training["first_identity_micro_audit"]["rows"],
                  translation_axis_statistics=[dict(axis=axis, **stats) for axis,stats in
                      training["paired_stream_audit"]["axis_statistics_shifted_only"].items()])
    for name, content in tables.items():
        write_csv(dest / (name + ".csv"), content)
    notes = dict(schema=SCHEMA, all_original_numeric_fields=metrics, primary_weighting="noise mean -> condition mean -> parent mean",
                 ALL_group="All 12 registered parents, not mean of unequal-sized TRAIN and DEV groups",
                 historical_TRAIN="Original panel label only; current parent membership explicitly retained",
                 units="Original collector units; fractions remain 0..1, not multiplied by100",
                 status="Original I/LOW_SUPPORT and seam NA statuses preserved; missing never zero-filled",
                 I="Original ConflictCov_C; all support and PASS-only/LOW_SUPPORT separately reported",
                 pair_rule="Same registered request+stage; candidate-reference; I BOTH_PASS intersection separately reported",
                 primary_reference="RELATION_V1_S1500; same training stream except joint translation",
                 historical_secondary_reference="USER_V2_S1500; no attribution of all differences to translation",
                 actual_numeric_field_count=177, older_protocol_expected_179="No synthetic columns added; preserve actual collector schema",
                 counts="request_sum is a sum over sampled outputs, not unique geometric defects or independent parents",
                 surface="actual_free_faces, deleted_free_faces, area and absolute FF pairs stay separate; no ratio-of-means FF redefinition",
                 topology="bulk_reach_ge_0_30_descriptive is the existing descriptive field, not semantic or watertight success",
                 visual_review="NOT_INFERRED_FROM_METRICS; actual image review required separately",
                 no_winner_or_convergence_selected=True, model_calls=0, scoring_calls=0)
    write_json(dest / "definitions.json", notes)
    write_json(dest / "training_summary.json", training)
    files = {str(p): sha(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name != "completion.json"}
    result = dict(status="COMPLETE", schema=SCHEMA, completed_utc=datetime.now(timezone.utc).isoformat(),
                  rows=len(rows), original_numeric_fields=len(metrics), paired_rows=len(pairs),
                  collector_input_hashes=data["source_hashes"], training_input_hashes=training_hashes,
                  analysis_code_hashes={str(p):sha(p) for p in (Path(__file__), Path(__file__).with_name("stream_audit.py"))},
                  science_audit_status=training["science_audit_status"],
                  script_sha256=sha(__file__), files=files, model_calls=0, scoring_calls=0,
                  original_scores_modified=False, claims_about_visual_success=False)
    for p,h in {**data["source_hashes"], **training_hashes}.items():
        if sha(resolve(p)) != h:
            raise ValueError("Read-only source changed during aggregation: " + p)
    write_json(done, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=str(DEFAULT))
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    answer = run(args.experiment, args.check_inputs)
    print(json.dumps({k:v for k,v in answer.items() if k not in {"files", "collector_input_hashes", "training_input_hashes"}}, ensure_ascii=False, indent=2))
