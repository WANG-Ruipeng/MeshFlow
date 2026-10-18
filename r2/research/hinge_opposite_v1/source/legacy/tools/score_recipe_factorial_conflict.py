#!/usr/bin/env python3
"""Versioned CPU adapter: unchanged CF/FF/I math, separate A20 diagnostics.

All-phase RAW completion precedes scoring. Original scorer modules and historical
assets are imported/read, never rewritten or monkeypatched. Unknowns and missing
values retain their registered denominators.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "tools"))
import numpy as np
from meshflow_control.evaluation import control, quality
from meshflow_control.evaluation.evaluate import open_reference, score_request, sources
from meshflow_control.evaluation.geometry import scan_geometry, topology
from meshflow_control.evaluation.io import arrays, array_sha, digest, read_json, resolve, save_npz, write_csv, write_json
from meshflow_control.evaluation.shape import shape_quality
from meshflow_control.evaluation.surface import sample_mesh_surface, surface_distance_squared
from evaluate_recipe_factorial import (DEFAULT_OUT, LABELS, L_CANON, META, NOISES, completed_rows,
    phase_registration, phase_root, read_input, root, validate_requests, verify_output)

VERSION = "recipe_factorial_original_conflict_and_quality_v1"


def definition():
    return dict(version=VERSION, original_metric_version=control.VERSION, original_definition=control.frozen_definition(),
                scale=L_CANON, scale_source="canonical training cube diagonal sqrt(3)*2/.3762",
                denominator="all registered C query weights; CF proper/FF illegal currentN-K; equal parent after noise mean",
                primary_domain="RAW C/free; no target or required interface", query_count=2048,
                reference_diagnostics="A20 only, sourceL0, separate tables, never promotion gate",
                topology="RAW soup and exact-coordinate indexing diagnostic; not watertight certification")


def query_registration_path(out, requests_path):
    return root(out) / "evaluation/query_registrations" / (digest(requests_path) + ".json")


def prepare_queries(out, requests_path):
    out = root(out); requests_path = resolve(requests_path); requests = read_json(requests_path)["rows"]
    path = query_registration_path(out, requests_path)
    identity = dict(requests_sha256=digest(requests_path), definition=definition(), original_sources=sources())
    if path.exists():
        old = read_json(path)
        if old["identity"] != identity:
            raise ValueError("Frozen query registration definition changed")
        for r in old["rows"]:
            load_queries(r)
        return str(path)
    # Later confirm phases can bind an already-frozen C query bank. They may
    # never create new scientific query samples after the first rollout.
    previous_queries = {}
    for old_registration in (out / "evaluation/query_registrations").glob("*.json"):
        previous = read_json(old_registration)
        if previous["identity"]["definition"] == definition() and previous["identity"]["original_sources"] == sources():
            for row in previous["rows"]:
                previous_queries[row["condition_id"]] = row
    rollouts_started = (out / "phases").exists() and any((out / "phases").glob("*/outputs/*/*/calls.jsonl"))
    old_path = out.parents[0] / "chair_early_alignment_schedule40_v1/evaluation/conflict_registration.json"
    old_rows = read_json(old_path)["rows"] if old_path.exists() else []
    by_old = {r["condition_id"]: r for r in old_rows}
    rows = []
    for cid in sorted({q["condition_id"] for q in requests}):
        group = [q for q in requests if q["condition_id"] == cid]
        q = group[0]; b = read_input(q)
        if any(r["C_sha256"] != q["C_sha256"] or r["N"] != q["N"] for r in group):
            raise ValueError("Condition changed across noise")
        existing = previous_queries.get(cid)
        prior = by_old.get(cid)
        if existing and existing["C_sha256"] == q["C_sha256"]:
            load_queries(existing)
            rows.append(existing)
            continue
        if rollouts_started:
            raise RuntimeError("New C queries must be frozen before this run's first rollout")
        if prior and prior["C_sha256"] == q["C_sha256"] and float(prior["identity"]["L0"]) == L_CANON:
            saved = load_queries(prior)
            ref = dict(arrays=saved["arrays"], identity=prior["identity"])
            reuse = dict(source_registration=str(old_path), source_registration_sha256=digest(old_path),
                         source_query_path=prior["query_path"], source_query_sha256=prior["query_sha256"], kind="EXACT_OLD_A40_QUERY_ARRAYS")
        else:
            ref = open_reference(dict(C=b["C"], L0=L_CANON), cid)
            reuse = dict(kind="ORIGINAL2048_SAMPLER_FIXED_CONDITION_SEED")
        target = out / "evaluation/query_banks" / cid / "I_reference.npz"
        save_npz(target, ref["arrays"])
        rows.append(dict(condition_id=cid, parent_id=q["parent_id"], ratio=q["ratio"], C_sha256=q["C_sha256"],
                         query_path=str(target), query_sha256=digest(target), query_array_sha256={k: array_sha(v) for k,v in ref["arrays"].items()},
                         identity=ref["identity"], query_count=2048, all_four_noises_share_same_bank=True,
                         weight_denominator=float(ref["arrays"]["C_weights"].sum()), provenance=reuse))
    write_json(path, dict(status="FROZEN", identity=identity, rows=rows, model_calls=0), freeze=True)
    return str(path)


def load_queries(row):
    if digest(row["query_path"]) != row["query_sha256"]:
        raise ValueError("Query file changed")
    a = arrays(row["query_path"])
    if set(a) != set(row["query_array_sha256"]) or any(array_sha(a[k]) != h for k,h in row["query_array_sha256"].items()):
        raise ValueError("Query arrays changed")
    if len(a["C_points"]) != 2048 or len(a["C_weights"]) != 2048:
        raise ValueError("Original2048 query count differs")
    return dict(arrays=a, identity=row["identity"])


def load_registration(out, requests_path):
    path = query_registration_path(out, requests_path); value = read_json(path)
    if value["identity"] != dict(requests_sha256=digest(requests_path), definition=definition(), original_sources=sources()):
        raise ValueError("Query identity differs")
    return value


def auxiliary_reference_score(q, receipt, dest):
    """P1 branch writes all derived files under dest; does not enter open_reference."""
    if q["ratio"] != .2:
        return dict(status="NA_NOT_APPLICABLE", reason="No trustedA40 interface or exposure contract", metrics={}, promotion_gate=False)
    if not q.get("reference_path"):
        return dict(status="BLOCKED", reason="Missing frozenA20 reference", metrics={}, promotion_gate=False)
    request = dict(q, **{k:v for k,v in receipt.items() if k not in q})
    request.pop("historical_start", None)
    request["panel"] = "P1"
    row = score_request(request, dest)
    row.update(promotion_gate=False, diagnostic_domain="A20 source-specific S/E and distances at originalL0", reference_only=True)
    return row


def score_one(args):
    out, phase, receipt, query = args; p = phase_root(out, phase)
    q = receipt; label = q["label"]; dest = p / "evaluation/rows" / label / (q["request_id"] + ".json")
    identity = dict(version=VERSION, request_id=q["request_id"], label=label, input_sha256=q["input_sha256"],
                    raw_sha256=q["raw_sha256"], result_sha256=q["result_sha256"], ply_sha256=q["ply_sha256"],
                    checkpoint_sha256=q["checkpoint_sha256"], query_sha256=query["query_sha256"],
                    sources=sources(), adapter_sha256=digest(__file__), L_canonical=L_CANON)
    if dest.exists():
        old = read_json(dest)
        if old["identity"] != identity:
            raise ValueError("Completed scoring identity changed")
        verify_output(q, receipt, q["checkpoint_sha256"]); load_queries(query)
        for path, h in old.get("files", {}).items():
            if digest(path) != h:
                raise ValueError("Scoring artifact changed")
        return old
    tick = time.perf_counter(); row = {k:q[k] for k in META}
    row.update(status="BLOCKED", identity=identity, metrics={}, metric_status={}, files={}, contracts={}, L0=L_CANON,
               raw_path=q["raw_path"], registered_free_faces=q["N"]-q["K"], registered_C_queries=2048)
    try:
        raw, facts = verify_output(q, receipt, q["checkpoint_sha256"]); b = read_input(q); ref = load_queries(query)
        C, F = b["C"], raw["raw"][q["K"]:]
        m, cf, ff = scan_geometry(C, F, L_CANON)
        trusted = control.trusted_cf(C, F, L_CANON, cf_scan=cf)
        im, iv = control.conflict_metrics(F, dict(C=C, L0=L_CANON), ref, trusted)
        if abs(im["ConflictCov_C"] - im["ConflictCov_C_cross"] - im["ConflictCov_C_dup"] + im["ConflictCov_C_cross_and_dup"]) > 1e-12:
            raise ValueError("OriginalI union identity failed")
        m.update(im); m.update(topology(raw["raw"], q["K"]))
        cq, wq = shape_quality(C, L_CANON), shape_quality(raw["raw"], L_CANON)
        m.update(C_total_surface_area=cq["total_area"], C_near_degenerate_count=cq["near_degenerate_area_count"],
                 C_exact_zero_area_count=cq["exact_zero_area_count"], whole_total_surface_area=wq["total_area"],
                 whole_near_degenerate_count=wq["near_degenerate_area_count"], whole_minimum_angle_lt5_fraction=wq["min_angle_lt5_fraction"])
        qp = sample_mesh_surface(F, 2048, np.random.RandomState(quality.seed("Q_F", q["condition_id"], q["candidate"], 2048)))
        if qp is None:
            raise ValueError("No positive finite free surface for diversity")
        pairpath = p / "evaluation/pairs" / label / (q["request_id"] + ".json")
        vecpath = p / "evaluation/vectors" / label / (q["request_id"] + ".npz")
        write_json(pairpath, dict(CF=cf, FF=ff), freeze=True)
        save_npz(vecpath, dict(iv, Q_F2048=qp))
        row.update(status="PASS", metrics=m, contracts=facts, pair_audit=str(pairpath), quality_vectors_path=str(vecpath),
                   metric_status=dict(ConflictCov_C=im["ConflictCov_C_status"], whole_mesh_Ri="NOT_IMPLEMENTED", watertight_certification="NOT_IMPLEMENTED", human_usability="NOT_RUN"),
                   files={str(pairpath):digest(pairpath), str(vecpath):digest(vecpath)}, full_CF_scans=1, full_FF_scans=1,
                   I_reuses_exact_CF_scan=True, query_sha256=query["query_sha256"], query_weight_denominator=float(ref["arrays"]["C_weights"].sum()))
    except Exception as exc:
        row.update(status="BLOCKED", reason=str(exc), traceback=traceback.format_exc())
    # Auxiliary diagnostic failure never discards a valid primary result.
    try:
        aux = auxiliary_reference_score(q, receipt, p / "evaluation/reference_A20") if row["status"] == "PASS" else dict(status="BLOCKED", metrics={}, reason="Primary contract/scoring blocked", promotion_gate=False)
    except Exception as exc:
        aux = dict(status="BLOCKED", reason=str(exc), traceback=traceback.format_exc(), metrics={}, promotion_gate=False)
    auxpath = p / "evaluation/auxiliary_rows" / label / (q["request_id"] + ".json")
    write_json(auxpath, dict(**{k:q[k] for k in META}, result=aux), freeze=True)
    row.update(auxiliary_path=str(auxpath), auxiliary_status=aux["status"], seconds=time.perf_counter()-tick)
    row["files"][str(auxpath)] = digest(auxpath)
    write_json(dest, row, freeze=True)
    return row


def numeric_keys(rows):
    return sorted({k for r in rows for k,v in r.get("metrics", {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)})


def aggregate(rows, labels, expected_requests):
    """No survivor average substitutes for a full four-noise/equal-parent mean."""
    keys = numeric_keys(rows); by = {(r["label"],r["request_id"]):r for r in rows}; parents = []; summaries = []
    conditions = defaultdict(list)
    for q in expected_requests:
        conditions[q["parent_id"], q["ratio"]].append(q)
    for label in labels:
        for (pid, ratio), qs in sorted(conditions.items()):
            rr = [by.get((label,q["request_id"]), {}) for q in qs]
            row = dict(label=label, parent_id=pid, ratio=ratio, condition_cohort=qs[0]["condition_cohort"], registered_noises=len(qs),
                       scored_noises=sum(r.get("status") == "PASS" for r in rr))
            for key in keys:
                values = [r["metrics"][key] for r in rr if r.get("status") == "PASS" and isinstance(r.get("metrics", {}).get(key), (int,float)) and np.isfinite(r["metrics"][key])]
                row[key+"_available_count"] = len(values)
                row[key] = float(np.mean(values)) if len(values) == len(qs) else None
            parents.append(row)
        for ratio in (.2, .4):
            for cohort in (("ALL",) if ratio == .2 else ("ALL", "ORIGINAL6", "EXTENDED2")):
                pp = [r for r in parents if r["label"] == label and r["ratio"] == ratio and (cohort == "ALL" or r["condition_cohort"] == cohort)]
                if not pp:
                    continue
                summary = dict(label=label, ratio=ratio, condition_cohort=cohort, registered_parents=len(pp),
                               registered_outputs=sum(r["registered_noises"] for r in pp), scored_outputs=sum(r["scored_noises"] for r in pp))
                for key in keys:
                    values = [r[key] for r in pp if r[key] is not None]
                    summary[key+"_complete_parents"] = len(values)
                    summary[key+"_equal_parent_mean"] = float(np.mean(values)) if len(values) == len(pp) else None
                    summary[key+"_available_parent_mean"] = float(np.mean(values)) if values else None
                summaries.append(summary)
    pairs = []; pb = {(r["label"],r["parent_id"],r["ratio"]):r for r in parents}
    for first, second in combinations(labels, 2):
        for pid, ratio in sorted(conditions):
            a, b = pb[first,pid,ratio], pb[second,pid,ratio]
            pairs.append(dict(first=first, second=second, parent_id=pid, ratio=ratio, condition_cohort=a["condition_cohort"],
                              delta_direction="first minus second after four-noise means",
                              **{key+"_delta": a[key]-b[key] if a[key] is not None and b[key] is not None else None for key in keys}))
    factorial = []
    if set(LABELS[2:]) <= set(labels):
        for pid, ratio in sorted(conditions):
            r1,r2,r3,r4 = [pb[label,pid,ratio] for label in LABELS[2:]]
            row = dict(parent_id=pid, ratio=ratio, condition_cohort=r1["condition_cohort"],
                       interpretation="Strategy contrasts; HYBRID exposure differs")
            for key in keys:
                if any(r[key] is None for r in (r1,r2,r3,r4)):
                    row.update({key+"_"+name:None for name in ("MIX_at_H_ALL", "MIX_at_H_LATE", "LATE_at_40", "LATE_at_MIX", "interaction")})
                else:
                    row.update({key+"_MIX_at_H_ALL":r2[key]-r1[key], key+"_MIX_at_H_LATE":r4[key]-r3[key],
                                key+"_LATE_at_40":r3[key]-r1[key], key+"_LATE_at_MIX":r4[key]-r2[key],
                                key+"_interaction":r4[key]-r3[key]-r2[key]+r1[key]})
            factorial.append(row)
    return parents, summaries, pairs, factorial


def diversity(rows, labels, requests):
    by = {(r["label"],r["request_id"]):r for r in rows}; groups = defaultdict(list); pairs = []; results = []
    for q in requests:
        groups[q["condition_id"]].append(q)
    for label in labels:
        for cid, qs in sorted(groups.items()):
            values = []
            for qa,qb in combinations(sorted(qs,key=lambda q:q["candidate"]),2):
                a,b = by.get((label,qa["request_id"])), by.get((label,qb["request_id"])); value = None
                if a and b and a["status"] == b["status"] == "PASS":
                    pa,pb = arrays(a["quality_vectors_path"])["Q_F2048"], arrays(b["quality_vectors_path"])["Q_F2048"]
                    ra,rb = arrays(a["raw_path"]), arrays(b["raw_path"])
                    fa = ra.get("raw",ra.get("output")).reshape(-1,3,3)[a["K"]:]
                    fb = rb.get("raw",rb.get("output")).reshape(-1,3,3)[b["K"]:]
                    value = float((surface_distance_squared(pa,fb).mean()+surface_distance_squared(pb,fa).mean())/(2*L_CANON**2)); values.append(value)
                pairs.append(dict(label=label, condition_id=cid, parent_id=qa["parent_id"], ratio=qa["ratio"], candidate_a=qa["candidate"], candidate_b=qb["candidate"],
                                  D_free_pair=value, status="PASS" if value is not None else "BLOCKED_MISSING_MEMBER"))
            q = qs[0]
            results.append(dict(label=label, condition_id=cid, parent_id=q["parent_id"], ratio=q["ratio"], condition_cohort=q["condition_cohort"],
                                expected_pairs=6, available_pairs=len(values), D_free_mean=float(np.mean(values)) if len(values)==6 else None,
                                direction="Variation only; not larger-is-better", not_original_TMD=True))
    return pairs, results


def score(out, phase="main", workers=2):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("CPU workers must be1..4 with2 threads each")
    p, registration, requests = phase_registration(out, phase)
    actual = completed_rows(out, phase)  # Whole-phase barrier before any numerical score.
    query_registration = load_registration(out, registration["requests_path"])
    queries = {r["condition_id"]:r for r in query_registration["rows"]}
    jobs = [(str(root(out)),phase,r,queries[r["condition_id"]]) for r in actual]
    tick = time.perf_counter()
    if workers == 1:
        rows = [score_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(score_one,jobs))
    dest = p / "evaluation"
    write_json(dest / "per_output.json", dict(version=VERSION, rows=rows))
    write_csv(dest / "per_output.csv", [dict(**{k:r.get(k) for k in (*META,"status","reason","auxiliary_status")}, **r["metrics"], **r["contracts"]) for r in rows])
    parents,summaries,pairs,factorial = aggregate(rows,registration["labels"],requests)
    for name,table in (("per_parent",parents),("summary",summaries),("paired_parent_deltas",pairs),("factorial_parent_contrasts",factorial)):
        write_csv(dest / (name+".csv"),table)
    noise_pairs = []
    by = {(r["label"],r["request_id"]):r for r in rows}; keys = numeric_keys(rows)
    for a,b in combinations(registration["labels"],2):
        for q in requests:
            ra,rb = by[a,q["request_id"]],by[b,q["request_id"]]
            noise_pairs.append(dict(first=a, second=b, **{k:q[k] for k in ("request_id","parent_id","candidate","ratio","condition_cohort")},
                                    **{k+"_delta":ra["metrics"][k]-rb["metrics"][k] if ra["status"]==rb["status"]=="PASS" and isinstance(ra["metrics"].get(k),(int,float)) and isinstance(rb["metrics"].get(k),(int,float)) else None for k in keys}))
    write_csv(dest / "paired_noise_deltas.csv",noise_pairs)
    dp,ds = diversity(rows,registration["labels"],requests)
    write_csv(dest / "diversity_pairs.csv",dp); write_csv(dest / "diversity_per_condition.csv",ds)
    aux = [read_json(r["auxiliary_path"]) for r in rows]
    write_json(dest / "reference_A20_per_output.json",dict(rows=aux, promotion_gate=False))
    write_csv(dest / "reference_A20_per_output.csv",[dict(**{k:r[k] for k in META}, status=r["result"]["status"], reason=r["result"].get("reason"), **r["result"].get("metrics",{})) for r in aux])
    aux_rows = [dict(**{k:r[k] for k in META}, status=r["result"]["status"], metrics=r["result"].get("metrics",{})) for r in aux if r["ratio"] == .2]
    aux_tables = aggregate(aux_rows,registration["labels"],[q for q in requests if q["ratio"] == .2])
    for name,table in zip(("per_parent","summary","paired_parent_deltas","factorial_parent_contrasts"),aux_tables):
        write_csv(dest / ("reference_A20_"+name+".csv"),table)
    result = dict(status="COMPLETE" if all(r["status"]=="PASS" for r in rows) else "PARTIAL_BLOCKED", registered=len(actual),
                  scored=sum(r["status"]=="PASS" for r in rows), workers=workers, cpu_threads_per_worker=2,
                  wall_seconds=time.perf_counter()-tick, model_forwards=0, auxiliary_is_promotion_gate=False,
                  auxiliary_blocked=sum(r["result"]["status"]=="BLOCKED" for r in aux))
    write_json(dest / "scoring_receipt.json",result); return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("prepare","score"),required=True); p.add_argument("--out",default=DEFAULT_OUT)
    p.add_argument("--phase",default="main"); p.add_argument("--requests"); p.add_argument("--workers",type=int,default=2)
    a = p.parse_args(argv)
    result = dict(status="FROZEN",path=prepare_queries(a.out,a.requests or root(a.out)/"evaluation/requests.json")) if a.stage=="prepare" else score(a.out,a.phase,a.workers)
    print(json.dumps(result),flush=True)
    return 2 if result.get("status")=="PARTIAL_BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
