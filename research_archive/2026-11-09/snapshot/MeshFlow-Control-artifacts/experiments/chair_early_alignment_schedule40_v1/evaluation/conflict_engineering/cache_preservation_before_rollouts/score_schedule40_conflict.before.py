#!/usr/bin/env python3
"""Isolated original raw-I supplement for the fixed schedule40 panel.

CPU geometry only. Freezes the original 2048 C area queries before any rollout.
Reuses verified main CF pair caches; no GT/reference completion, Gamma, exposure,
new mesh sampling or model forward is involved. Existing tables stay unchanged.
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
from meshflow_control.evaluation import control
from meshflow_control.evaluation.evaluate import open_reference
from meshflow_control.evaluation.io import arrays, array_sha, digest, read_json, resolve, save_npz, write_csv, write_json

VERSION = "schedule40_original_conflict_supplement_v1"
ROOT_NAME = "chair_early_alignment_schedule40_v1"
LABELS = ("START5000", "HIST_FM6000", "HIST_FREE_ALIGN6000", "FM_ONLY", "EARLY", "LATE")
L_CANON = float(np.sqrt(3.)*2./.3762)
PRIMARY = ("ConflictCov_C", "ConflictCov_C_upper", "ConflictCov_C_cross",
           "ConflictCov_C_dup", "ConflictCov_C_cross_and_dup",
           "ConflictCov_C_uncertain_fraction")
LEGAL_INPUT_KEYS = {"C", "N", "K", "z"}
METRIC_STATUSES = ("PASS", "PARTIAL", "LOW_SUPPORT")


def output_root(path):
    path = resolve(path).resolve()
    if path.name != ROOT_NAME or path.parent.name != "experiments":
        raise ValueError("Wrong fixed experiment directory")
    return path


def source_identity():
    return dict(adapter_sha256=digest(__file__), original_scorer_sources=control.source_identity(),
                original_main_adapter_sha256=digest(REPO/"tools/evaluate_schedule40.py"))


def definition():
    original = control.frozen_definition()
    return dict(adapter_version=VERSION, original_metric_version=control.VERSION,
        original_ConflictCov_C_definition=original["ConflictCov_C"],
        applied_domain="RAW C versus generated free F; raw I only; no reference subtraction",
        original_sampling=dict(count=2048, C_seed=original["C_seed"], method=original["C_sampling"]),
        denominator="All 2048 C area-sample weights; equal1/2048 weights from original sampler",
        scale=dict(value=L_CANON, source="Public canonical training cube diagonal sqrt(3)*2/.3762",
                   per_output_bbox_normalization=False),
        thresholds=dict(rho_cross=control.RHO_CROSS, rho_dup=control.RHO_DUP,
                        C_interior_border=control.R_BORDER, normal_degrees=control.NORMAL_ANGLE),
        union_identity="ConflictCov_C = cross + dup - cross_and_dup",
        interpretation="Lower is less measured conflict. Near-parallel duplicate support may flag legitimate thin walls; not a semantic validity or attachment score.",
        legacy_reference_note="Original frozen definition also documents GT-relative I; that branch is NOT_RUN here.",
        GT_required=False, Gamma_required=False, E_required=False,
        new_metric_definitions=False, changes_to_original_scorers=False,
        model_forwards=0, generated_meshes=0)


def read_input(request):
    path = resolve(request["input_path"])
    if digest(path) != request["input_sha256"]: raise ValueError("Frozen input file changed")
    bank = arrays(path)
    if set(bank) != LEGAL_INPUT_KEYS: raise ValueError("Input bank must contain only C/N/K/z, no GT or Gamma")
    C = bank["C"]
    if C.dtype != np.float32 or C.shape != (request["K"],3,3) or not np.isfinite(C).all():
        raise ValueError("Expected unchanged finite FP32 C triangles")
    if int(bank["N"]) != request["N"] or int(bank["K"]) != request["K"]:
        raise ValueError("Input N/K differ")
    if array_sha(C) != request["C_sha256"]: raise ValueError("Actual C identity changed")
    return bank


def freeze_query_bank(request, folder):
    """Only the inherited input-only open_reference path, with original seed."""
    bank = read_input(request)
    ref = open_reference(dict(C=bank["C"], L0=L_CANON), request["condition_id"])
    folder = resolve(folder)
    path = folder/"I_reference.npz"
    save_npz(path, ref["arrays"])
    row = dict(request_id=request["request_id"], condition_id=request["condition_id"],
        parent_id=request["parent_id"], N=request["N"], K=request["K"],
        input_path=request["input_path"], input_sha256=request["input_sha256"],
        C_sha256=request["C_sha256"], identity=ref["identity"],
        query_path=str(path), query_sha256=digest(path),
        query_array_sha256={k:array_sha(v) for k,v in ref["arrays"].items()},
        query_count=len(ref["arrays"]["C_points"]),
        weight_denominator=float(ref["arrays"]["C_weights"].sum()),
        C_interior_samples=int((ref["arrays"]["C_border_distance"] >= control.R_BORDER*L_CANON).sum()),
        condition_construction=request["condition_construction"])
    write_json(folder/"I_reference.json", row, freeze=True)
    return row


def load_query_bank(row):
    if digest(row["query_path"]) != row["query_sha256"]: raise ValueError("Frozen I query file changed")
    saved = arrays(row["query_path"])
    if set(saved) != set(row["query_array_sha256"]): raise ValueError("Query bank fields changed")
    if any(array_sha(saved[k]) != h for k,h in row["query_array_sha256"].items()):
        raise ValueError("Frozen I query array changed")
    if len(saved["C_points"]) != 2048 or len(saved["C_weights"]) != 2048:
        raise ValueError("Original2048 query count differs")
    return dict(arrays=saved, identity=row["identity"])


def prepare(out):
    out = output_root(out); path = out/"evaluation/conflict_registration.json"
    request_path = out/"evaluation/requests.json"
    request_manifest = read_json(request_path)
    requests = request_manifest["rows"]
    if len(requests) != 8 or len({r["condition_id"] for r in requests}) != 8:
        raise ValueError("Expected eight fixed unique conditions")
    if tuple(request_manifest["labels"]) != LABELS or any(r["K"] != round(.4*r["N"]) for r in requests):
        raise ValueError("Expected same six labels and fixed40% face counts")
    scientific = dict(version=VERSION, requests_sha256=digest(request_path),
                      labels=list(LABELS), expected_outputs=48, definition=definition(), sources=source_identity())
    if path.exists():
        saved = read_json(path)
        for key,value in scientific.items():
            if saved[key] != value: raise ValueError("Frozen conflict registration changed: "+key)
        for row in saved["rows"]:
            read_input(row)
            load_query_bank(row)
        return dict(status="VERIFIED_EXISTING", registration_path=str(path),
                    sha256=digest(path), conditions=8, model_forwards=0)
    # Refuse retroactive registration once this experiment has any output.
    output_folder = out/"outputs"
    if output_folder.exists() and any(p.is_file() for p in output_folder.rglob("*")):
        raise RuntimeError("Cannot preregister original-I queries after rollout artifacts exist")
    rows = [freeze_query_bank(r, out/"evaluation/conflict_queries"/r["condition_id"]) for r in requests]
    registration = dict(scientific, status="FROZEN_BEFORE_ROLLOUTS", created_unix=time.time(),
        rollout_files_at_registration=0, rows=rows, model_forwards=0, new_rendering=0)
    write_json(path, registration, freeze=True)
    return dict(status=registration["status"], registration_path=str(path),
                sha256=digest(path), conditions=8, total_C_queries=8*2048,
                model_forwards=0, generated_meshes=0)


def load_registration(out):
    out = output_root(out); path = out/"evaluation/conflict_registration.json"
    registered = read_json(path)
    if registered["sources"] != source_identity() or registered["definition"] != definition():
        raise ValueError("Frozen conflict source/definition changed")
    if registered["requests_sha256"] != digest(out/"evaluation/requests.json"):
        raise ValueError("Evaluation request manifest changed")
    if registered["rollout_files_at_registration"] != 0 or registered["status"] != "FROZEN_BEFORE_ROLLOUTS":
        raise ValueError("Conflict query registration was not before rollouts")
    return registered


def verify_main_cf(out, request, label, raw_hash, registration):
    """Bind main cached classifier output to actual RAW, input, source and L."""
    path = out/"evaluation/rows"/label/(request["request_id"]+".json")
    if not path.exists(): raise FileNotFoundError("MAIN_CF_CACHE_NOT_YET_AVAILABLE")
    main = read_json(path)
    if main["status"] != "PASS": raise ValueError("MAIN_SCORING_NOT_PASS: "+str(main.get("reason")))
    identity = main["identity"]
    if identity["raw_sha256"] != raw_hash or identity["input_sha256"] != request["input_sha256"]:
        raise ValueError("Main CF cache refers to different raw/input")
    if identity["scorer_sources"] != registration["sources"]["original_scorer_sources"]:
        raise ValueError("Main CF scorer source hashes differ")
    if identity["adapter_sha256"] != registration["sources"]["original_main_adapter_sha256"]:
        raise ValueError("Main CF adapter source hash differs")
    if float(identity["L_canonical"]) != L_CANON: raise ValueError("Main CF normalization scale differs")
    if not main.get("contracts") or not all(value is True for value in main["contracts"].values()):
        raise ValueError("Main RAW contract not completely verified")
    pair_path = resolve(main["pair_audit"])
    if digest(pair_path) != main["pair_sha256"]: raise ValueError("Main pair cache changed")
    pair = read_json(pair_path)["CF"]
    if pair.get("tested_pairs") != request["K"]*(request["N"]-request["K"]) or pair.get("affected_subset") is not None or not pair.get("independent_verification"):
        raise ValueError("Main CF cache is incomplete or lacks independent verification")
    return pair, dict(main_row_path=str(path), main_row_sha256=digest(path),
        pair_path=str(pair_path), pair_sha256=main["pair_sha256"],
        full_CF_rescans=0, reuse="EXACT_MAIN_INDEPENDENT_CF_CACHE")


def score_one(args):
    out, request, label, stage_ready = args
    out = output_root(out)
    dest = out/"evaluation/conflict_rows"/label/(request["request_id"]+".json")
    row = {k:request[k] for k in ("request_id","condition_id","parent_id","N","K","condition_construction")}
    row.update(label=label, status="BLOCKED", metric_status="NA_BLOCKED", metrics={},
               metric_version=control.VERSION, adapter_version=VERSION,
               registered_C_queries=2048, denominator="all C query weights", GT_used=False)
    if not stage_ready:
        row["reason"] = "ALL48_RAW_REQUIRED_BEFORE_CONFLICT_SCORING"
        write_json(dest,row); return row
    started = time.perf_counter()
    try:
        registration = load_registration(out)
        registered = next(r for r in registration["rows"] if r["request_id"]==request["request_id"])
        bank = read_input(request); ref = load_query_bank(registered)
        receipt_path = out/"outputs"/label/request["request_id"]/"receipt.json"
        if not receipt_path.exists(): raise FileNotFoundError("MISSING_REGISTERED_RAW")
        receipt = read_json(receipt_path)
        if receipt["status"] != "COMPLETE" or receipt["input_sha256"] != request["input_sha256"]:
            raise ValueError("RAW receipt is incomplete or refers to another input")
        raw_hash = digest(receipt["raw_path"])
        if raw_hash != receipt["raw_sha256"]: raise ValueError("RAW changed")
        identity = dict(version=VERSION, registration_sha256=digest(out/"evaluation/conflict_registration.json"),
            request_id=request["request_id"], label=label, raw_sha256=raw_hash,
            input_sha256=request["input_sha256"], query_sha256=registered["query_sha256"],
            sources=registration["sources"])
        cf_scan, cache = verify_main_cf(out, request, label, raw_hash, registration)
        identity.update(main_row_sha256=cache["main_row_sha256"], CF_pair_sha256=cache["pair_sha256"])
        if dest.exists():
            previous = read_json(dest)
            if previous.get("status") == "SCORED":
                if previous["identity"] != identity: raise ValueError("Completed conflict score identity changed")
                if digest(previous["vectors_path"]) != previous["vectors_sha256"]:
                    raise ValueError("Completed conflict vectors changed")
                return previous
        raw = arrays(receipt["raw_path"])
        mesh = raw["raw"].reshape(-1,3,3)
        if mesh.dtype != np.float32 or mesh.shape != (request["N"],3,3) or not np.isfinite(mesh).all():
            raise ValueError("Invalid complete RAW mesh")
        if array_sha(mesh[:request["K"]]) != array_sha(bank["C"]): raise ValueError("RAW C changed")
        F = mesh[request["K"]:]
        missing_witnesses = sum(
            r["classification"]==control.PROPER
            and not r.get("detector",{}).get("intersection_points")
            and not r.get("detector",{}).get("independent",{}).get("intersection_points")
            for r in cf_scan["contacts"])
        cf = control.trusted_cf(bank["C"], F, L_CANON, cf_scan=cf_scan)
        metrics, vectors = control.conflict_metrics(F, dict(C=bank["C"],L0=L_CANON), ref, cf)
        union_value = metrics["ConflictCov_C_cross"]+metrics["ConflictCov_C_dup"]-metrics["ConflictCov_C_cross_and_dup"]
        if abs(metrics["ConflictCov_C"]-union_value) > 1e-12:
            raise ValueError("Original I union identity failed")
        if metrics["ConflictCov_C_status"] not in METRIC_STATUSES:
            raise ValueError("Unexpected original I status")
        vector_path = out/"evaluation/conflict_vectors"/label/(request["request_id"]+".npz")
        save_npz(vector_path, vectors)
        row.update(status="SCORED", metric_status=metrics["ConflictCov_C_status"],
            metrics=metrics, identity=identity, cache_audit=cache,
            proper_witness_reextractions_using_existing_classifier=missing_witnesses,
            full_CF_rescans=0, weight_denominator=float(ref["arrays"]["C_weights"].sum()),
            vectors_path=str(vector_path), vectors_sha256=digest(vector_path),
            vector_array_sha256={k:array_sha(v) for k,v in vectors.items()},
            query_bank_path=registered["query_path"], query_bank_sha256=registered["query_sha256"],
            raw_path=receipt["raw_path"], raw_sha256=raw_hash,
            union_identity_absolute_error=abs(metrics["ConflictCov_C"]-union_value),
            model_forwards=0, source_GT_subtraction="NOT_RUN", SeamCov="NA_NOT_REQUIRED",
            ExposedRetain="NA_NOT_REQUIRED",
            limitation="Duplicate band can count legitimate thin walls; no semantic interpretation")
    except Exception as exc:
        row.update(status="BLOCKED",metric_status="NA_BLOCKED",reason=str(exc),traceback=traceback.format_exc())
    row["seconds"] = time.perf_counter()-started
    write_json(dest,row); return row


def aggregate(rows):
    summaries = []
    for label in LABELS:
        group = [r for r in rows if r["label"]==label]
        scored = [r for r in group if r["status"]=="SCORED"]
        summary = dict(label=label, registered=8, scored=len(scored),
            blocked=8-len(scored), PASS=sum(r["metric_status"]=="PASS" for r in scored),
            PARTIAL=sum(r["metric_status"]=="PARTIAL" for r in scored),
            LOW_SUPPORT=sum(r["metric_status"]=="LOW_SUPPORT" for r in scored))
        for key in PRIMARY:
            values = [r["metrics"][key] for r in scored]
            summary[key+"_available_count"] = len(values)
            summary[key+"_available_mean"] = float(np.mean(values)) if values else None
            summary[key+"_all8_mean"] = float(np.mean(values)) if len(values)==8 else None
        summaries.append(summary)
    pairs = []
    parents = sorted({r["parent_id"] for r in rows})
    by = {(r["parent_id"],r["label"]):r for r in rows}
    for first,second in combinations(LABELS,2):
        for parent in parents:
            a,b = by[parent,first],by[parent,second]
            row = dict(parent_id=parent,first=first,second=second,
                delta_direction="first minus second; lower I is lower measured conflict",
                first_status=a["status"],second_status=b["status"],
                first_metric_status=a["metric_status"],second_metric_status=b["metric_status"])
            if a["status"]=="SCORED" and b["status"]=="SCORED":
                row.update(status="PAIRED",**{key+"_delta":a["metrics"][key]-b["metrics"][key] for key in PRIMARY})
            else: row.update(status="BLOCKED_PAIR",reason="Missing/not-scored member; pair retained in denominator")
            pairs.append(row)
    return summaries,pairs


def score(out,workers=2):
    if type(workers) is not int or not 1<=workers<=4: raise ValueError("CPU workers must be1..4")
    out=output_root(out); registration=load_registration(out)
    requests=read_json(out/"evaluation/requests.json")["rows"]
    ready=all((out/"outputs"/label/r["request_id"]/"receipt.json").exists() for label in LABELS for r in requests)
    jobs=[(str(out),r,label,ready) for r in requests for label in LABELS]
    started=time.perf_counter()
    if workers==1: results=[score_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool: results=list(pool.map(score_one,jobs))
    if len(results)!=48: raise ValueError("Must preserve all48 registered outputs")
    summaries,pairs=aggregate(results)
    write_json(out/"evaluation/per_output_conflict.json",dict(version=VERSION,rows=results))
    write_csv(out/"evaluation/per_output_conflict.csv",
        [dict(**{k:r.get(k) for k in ("label","parent_id","condition_id","request_id","N","K",
              "condition_construction","status","metric_status","reason","weight_denominator",
              "query_bank_sha256","vectors_sha256")},**r["metrics"]) for r in results])
    write_csv(out/"evaluation/conflict_summary.csv",summaries)
    write_csv(out/"evaluation/conflict_paired_parent_deltas.csv",pairs)
    receipt=dict(status="COMPLETE" if all(r["status"]=="SCORED" for r in results) else "PARTIAL_BLOCKED",
        version=VERSION,registered=48,scored=sum(r["status"]=="SCORED" for r in results),
        blocked=sum(r["status"]!="SCORED" for r in results),
        metric_status_counts={status:sum(r["metric_status"]==status for r in results)
                              for status in (*METRIC_STATUSES,"NA_BLOCKED")},
        parent_pairs_registered=120,parent_pairs_scored=sum(r["status"]=="PAIRED" for r in pairs),
        registration_sha256=digest(out/"evaluation/conflict_registration.json"),
        workers=workers,seconds=time.perf_counter()-started,model_forwards=0,
        full_CF_rescans=0,main_per_output_table_modified=False,
        GT_metrics="NOT_RUN",SeamCov="NA_NOT_REQUIRED",ExposedRetain="NA_NOT_REQUIRED")
    write_json(out/"evaluation/conflict_scoring_receipt.json",receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("prepare","score"),required=True)
    parser.add_argument("--out",default="/mnt/e/MeshFlow-Control-artifacts/experiments/"+ROOT_NAME)
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args(argv)
    result=prepare(args.out) if args.stage=="prepare" else score(args.out,args.workers)
    print(json.dumps(result),flush=True)
    return 0 if result["status"]!="PARTIAL_BLOCKED" else 2


if __name__=="__main__": raise SystemExit(main())
