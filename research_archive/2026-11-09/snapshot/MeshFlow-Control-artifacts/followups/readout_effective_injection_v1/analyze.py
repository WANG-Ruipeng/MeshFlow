#!/usr/bin/env python3
"""Read-only CPU analysis of fixed-state ON/OFF readout probes.

This program never imports torch, invokes a model, samples a surface, changes a
plan, or writes to cells/. Only analysis/ is written. Prediction comparisons are
free-only and OFF next_state is an isolated counterfactual Euler step, not an
OFF rollout. Every expected cell/layer/pair keeps its fixed denominator.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VERSION = "readout_effective_injection_cpu_v1"
LAYERS = (3, 6, 9, 12)
QUANTILES = (0, 5, 25, 50, 75, 95, 100)
DEFAULT_ROOT = "E:/MeshFlow-Control-artifacts/followups/readout_effective_injection_v1"
META_FIELDS = {
    "cell_id", "label", "request_id", "panel", "N", "K", "step", "t", "mode",
    "layer", "status", "error", "expected_layer_rows", "available_layer_rows",
    "expected_pair_rows", "available_pair_rows",
}
DEFINITIONS = {
    "scope": "Descriptive activation and free velocity diagnostics; no quality/ranking inference.",
    "residual": "Saved FP32 pre-cast readout residual r.",
    "cast": "Actual captured cast residual, represented as FP32 values.",
    "effective": "after - before, evaluated in float64 from captured FP32 values.",
    "cast_error": "cast - residual.",
    "addition_error": "(after - before) - cast.",
    "effective_over_residual_l2": "||after-before||_2 / ||residual||_2; may exceed 1.",
    "effective_residual_cosine": "<effective,residual> / (||effective||_2 ||residual||_2).",
    "residual_nonzero_swallowed_fraction":
        "count(residual!=0 and effective==0) / count(residual!=0).",
    "cast_zero_fraction_nonzero_residual":
        "count(residual!=0 and cast==0) / count(residual!=0).",
    "changed_element_fraction": "count(after!=before) / number of captured feature elements.",
    "corner": "One row of [3*(N-K),768]; quantiles use per-row Euclidean norms.",
    "corner_effective_over_residual":
        "Per-corner ||effective|| / ||residual||, excluding zero-denominator corners; counts retained.",
    "relative_l2_on": "||ON-OFF||_2 / ||ON||_2 (ON is the explicit reference).",
    "symmetric_relative_l2": "2*||ON-OFF||_2 / (||ON||_2+||OFF||_2).",
    "changed_coordinate_fraction": "Fraction of free scalar coordinates with ON!=OFF.",
    "changed_corner_fraction": "Fraction of free corners with any of 3 coordinates changed.",
    "changed_face_fraction": "Fraction of free faces with any of 9 coordinates changed.",
    "next_state": "Same input state; ON saved next-state replay vs one isolated OFF Euler step. Not an OFF trajectory.",
    "euler": "FP32 integration: next_free = state_free + velocity_free * (1/50).",
    "zero_denominators": "NA/null/empty CSV, including 0/0; never imputed as zero or one.",
    "aggregation":
        "Cells equally weighted within label/layer/step groups. Means require all expected finite values; "
        "observed means retain explicit incomplete denominators. Min/max and NA counts are retained.",
    "pointwise_extrema": "max_abs is the maximum absolute scalar difference; rms uses all free/captured scalars.",
    "float_arithmetic": "Diagnostics/norms/quantiles float64; Euler consistency reproduces float32 multiply then add.",
    "quantiles": "numpy.quantile linear interpolation, percentages 0,5,25,50,75,95,100.",
    "data_identity": "SHA256 of exact input JSON/NPZ bytes; no checkpoint load or RNG execution.",
}

def path(value):
    s = str(value).replace("\\", "/")
    if os.name != "nt" and len(s) >= 3 and s[1:3] == ":/":
        s = "/mnt/" + s[0].lower() + s[2:]
    return Path(s)

def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def json_read(p):
    with Path(p).open(encoding="utf-8-sig") as f:
        return json.load(f)

def json_write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")

def numeric(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(float(v))

def ratio(a, b):
    return float(a / b) if b > 0 else None

def norm(a):
    flat = np.asarray(a, dtype=np.float64).ravel()
    return float(np.sqrt(np.sum(flat * flat, dtype=np.float64)))

def finite_array(a, shape, name):
    if a.dtype != np.float32 or tuple(a.shape) != tuple(shape):
        raise ValueError(f"{name}: expected FP32 {shape}, got {a.dtype} {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: nonfinite values")

def bitwise(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()

def distribution(prefix, values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    row = {prefix + "_valid_count": int(len(a))}
    for q in QUANTILES:
        row[f"{prefix}_q{q:02d}"] = float(np.quantile(a, q / 100.0)) if len(a) else None
    row[prefix + "_mean"] = float(a.mean()) if len(a) else None
    return row

def layer_numbers(before, residual, cast, after):
    b, r, c, a = [np.asarray(x, dtype=np.float64) for x in (before, residual, cast, after)]
    e = a - b
    ce, ae = c - r, e - c
    arrays = {"before": b, "residual": r, "cast": c, "after": a,
              "effective": e, "cast_error": ce, "addition_error": ae}
    row = {"element_count": int(r.size), "corner_count": int(r.shape[0])}
    for name, v in arrays.items():
        row[name + "_l2"] = norm(v)
        row[name + "_rms"] = float(np.sqrt(np.mean(v * v)))
        row[name + "_max_abs"] = float(np.max(np.abs(v)))
    rn, cn, en = row["residual_l2"], row["cast_l2"], row["effective_l2"]
    row.update(
        cast_error_relative_l2=ratio(row["cast_error_l2"], rn),
        addition_error_relative_l2_cast=ratio(row["addition_error_l2"], cn),
        effective_over_residual_l2=ratio(en, rn),
        cast_over_residual_l2=ratio(cn, rn),
        effective_over_cast_l2=ratio(en, cn),
        effective_residual_cosine=ratio(float(np.sum(e * r)), en * rn),
        cast_residual_cosine=ratio(float(np.sum(c * r)), cn * rn),
    )
    nonzero = r != 0
    nz = int(np.count_nonzero(nonzero))
    swallowed = int(np.count_nonzero(nonzero & (e == 0)))
    cast_zero = int(np.count_nonzero(nonzero & (c == 0)))
    row.update(
        residual_nonzero_elements=nz,
        residual_nonzero_swallowed_elements=swallowed,
        residual_nonzero_swallowed_fraction=ratio(swallowed, nz),
        cast_zero_elements_nonzero_residual=cast_zero,
        cast_zero_fraction_nonzero_residual=ratio(cast_zero, nz),
        changed_element_count=int(np.count_nonzero(a != b)),
        changed_element_fraction=float(np.mean(a != b)),
    )
    corner_norms = {}
    for name in ("residual", "cast", "effective", "cast_error", "addition_error"):
        v = np.sqrt(np.sum(arrays[name] * arrays[name], axis=1))
        corner_norms[name] = v
        row.update(distribution("corner_" + name + "_l2", v))
    valid = corner_norms["residual"] > 0
    ratios = corner_norms["effective"][valid] / corner_norms["residual"][valid]
    row.update(distribution("corner_effective_over_residual", ratios))
    row["corner_ratio_zero_denominator_count"] = int(np.count_nonzero(~valid))
    row["corner_nonzero_residual_swallowed_fraction"] = ratio(
        int(np.count_nonzero(valid & (corner_norms["effective"] == 0))), int(np.count_nonzero(valid)))
    row["changed_corner_fraction"] = float(np.mean(np.any(e != 0, axis=1)))
    return row

def pair_numbers(on, off, prefix):
    a, b = np.asarray(on, dtype=np.float64), np.asarray(off, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 9 or len(a) == 0:
        raise ValueError(f"{prefix}: expected nonempty free [F,9]")
    d = a - b
    na, nb, nd = norm(a), norm(b), norm(d)
    different = a != b
    return {
        prefix + "on_l2": na, prefix + "off_l2": nb,
        prefix + "delta_l2": nd,
        prefix + "delta_rms": float(np.sqrt(np.mean(d * d))),
        prefix + "delta_max_abs": float(np.max(np.abs(d))),
        prefix + "relative_l2_on": ratio(nd, na),
        prefix + "symmetric_relative_l2": ratio(2 * nd, na + nb),
        prefix + "changed_coordinates": int(np.count_nonzero(different)),
        prefix + "changed_coordinate_fraction": float(np.mean(different)),
        prefix + "changed_corner_fraction": float(np.mean(np.any(different.reshape(-1, 3), axis=1))),
        prefix + "changed_face_fraction": float(np.mean(np.any(different, axis=1))),
    }

def fixed_plan(plan):
    labels = plan.get("labels", plan.get("checkpoint_labels"))
    requests = plan.get("requests", plan.get("selected_requests"))
    steps = plan.get("steps")
    if not isinstance(labels, list) or len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError("plan must contain exactly two unique labels")
    if isinstance(requests, dict):
        requests = list(requests.values())
    if not isinstance(requests, list) or len(requests) != 4:
        raise ValueError("plan must contain exactly four requests")
    normalized = []
    for q in requests:
        q = {"request_id": q} if isinstance(q, str) else dict(q)
        if not q.get("request_id"):
            raise ValueError("request_id absent from plan")
        normalized.append(q)
    if len({q["request_id"] for q in normalized}) != 4:
        raise ValueError("duplicate request IDs")
    if steps != [0, 25, 45]:
        raise ValueError("fixed plan requires steps [0,25,45]")
    return labels, normalized, steps

def expected_cell(label, q, step, mode):
    return dict(cell_id=f"{label}__{q['request_id']}__s{step:02d}__{mode}",
                label=label, request_id=q["request_id"], panel=q.get("panel"),
                N=q.get("N"), K=q.get("K"), step=step, t=step / 50.0, mode=mode)

def flatten_checks(v):
    if isinstance(v, bool):
        return [v]
    if isinstance(v, dict):
        return [b for x in v.values() for b in flatten_checks(x)]
    if isinstance(v, list):
        return [b for x in v for b in flatten_checks(x)]
    return []

def resolve_array(root, cell_json, value):
    p = path(value)
    if p.is_absolute():
        return p
    candidates = [cell_json.parent / p, root / p]
    return next((x for x in candidates if x.is_file()), candidates[0])

def load_cell(root, expected):
    """Returns audit row, optional prediction arrays, 4 ON layer rows, identities."""
    cid, mode = expected["cell_id"], expected["mode"]
    jp = root / "cells" / (cid + ".json")
    audit = dict(expected, status="MISSING", error=None)
    layers = [dict(expected, layer=layer, status="MISSING", error=None) for layer in LAYERS] if mode == "on" else []
    identities = {"cell_id": cid, "json_path": str(jp)}
    prediction = None
    if not jp.is_file():
        audit["error"] = "Cell JSON missing"
        for x in layers: x["error"] = audit["error"]
        return audit, prediction, layers, identities
    try:
        d = json_read(jp)
        identities["json_sha256"] = digest(jp)
        if d.get("status") != "COMPLETE":
            audit.update(status="INCOMPLETE", error=f"Cell status {d.get('status')!r}")
            for x in layers: x.update(status="INCOMPLETE", error=audit["error"])
            return audit, None, layers, identities
        for key in ("cell_id", "label", "request_id", "step", "mode"):
            if d.get(key) != expected[key]:
                raise ValueError(f"Identity mismatch: {key}")
        for key in ("panel", "N", "K"):
            if expected.get(key) is not None and d.get(key) != expected[key]:
                raise ValueError(f"Identity mismatch: {key}")
            audit[key] = d[key]
        N, K = int(d["N"]), int(d["K"])
        if not 0 <= K < N or not math.isclose(float(d["t"]), expected["t"], abs_tol=1e-8):
            raise ValueError("N/K/t invalid")
        if mode == "on" and d.get("trajectory_replay_exact") is not True:
            raise ValueError("ON saved-trajectory replay not explicitly True")
        checks = d.get("layer_checks")
        audit["layer_checks_json"] = json.dumps(checks, ensure_ascii=False, sort_keys=True)
        checks_values = flatten_checks(checks)
        audit["layer_checks_boolean_count"] = len(checks_values)
        if mode == "on" and any(v is False for v in checks_values):
            raise ValueError("An ON layer check is False")
        ap = resolve_array(root, jp, d["array_path"])
        actual_hash = digest(ap)
        if actual_hash != d["array_sha256"]:
            raise ValueError("Array NPZ SHA256 mismatch")
        identities.update(array_path=str(ap), array_sha256=actual_hash)
        with np.load(ap, allow_pickle=False) as z:
            vel, state, nxt = [np.asarray(z[k]) for k in ("velocity", "state", "next_state")]
            finite_array(vel, (1, N, 9), "velocity")
            finite_array(state, (1, N, 9), "state")
            finite_array(nxt, (1, N, 9), "next_state")
            if not bitwise(state[:, :K], nxt[:, :K]):
                raise ValueError("Known coordinates changed in isolated next_state")
            # This exactly retains the original multiplication expression and FP32 order.
            calculated = state.copy()
            calculated[:, K:] = state[:, K:] + vel[:, K:] * (1 / 50)
            if not bitwise(calculated, nxt):
                raise ValueError("next_state does not match original FP32 Euler expression")
            audit.update(status="COMPLETE", error=None, trajectory_replay_exact=d.get("trajectory_replay_exact"),
                         known_next_bitwise=True, original_euler_expression_bitwise=True)
            prediction = {"velocity": vel.copy(), "state": state.copy(), "next_state": nxt.copy(), "N": N, "K": K}
            for row in layers:
                row.update(N=N, K=K, panel=d["panel"])
                try:
                    layer = row["layer"]
                    arrays = [np.asarray(z[f"l{layer}_{name}"]) for name in ("before", "residual", "cast", "after")]
                    for name, a in zip(("before", "residual", "cast", "after"), arrays):
                        finite_array(a, (3 * (N-K), 768), f"l{layer}_{name}")
                    row.update(layer_numbers(*arrays), status="COMPLETE", error=None)
                except Exception as e:
                    row.update(status="FAILED", error=f"{type(e).__name__}: {e}")
    except Exception as e:
        audit.update(status="FAILED", error=f"{type(e).__name__}: {e}")
        prediction = None
        for x in layers:
            if x["status"] != "COMPLETE":
                x.update(status="FAILED", error=audit["error"])
    return audit, prediction, layers, identities

def write_csv(p, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def aggregated(rows, group_fields):
    numeric_fields = sorted({k for r in rows for k,v in r.items() if k not in META_FIELDS and numeric(v)})
    groups = defaultdict(list)
    for r in rows: groups[tuple(r.get(k) for k in group_fields)].append(r)
    out = []
    for group, rr in sorted(groups.items(), key=lambda x: tuple(str(y) for y in x[0])):
        expected, available = len(rr), sum(r["status"] == "COMPLETE" for r in rr)
        for metric in numeric_fields:
            values = [float(r[metric]) for r in rr if r["status"] == "COMPLETE" and numeric(r.get(metric))]
            complete = len(values) == expected
            out.append(dict(zip(group_fields, group), metric=metric, expected_cells=expected,
                available_cells=available, finite_metric_cells=len(values), NA_cells=expected-len(values),
                mean=statistics_mean(values) if complete else None,
                observed_mean=statistics_mean(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None,
                status="COMPLETE" if complete else ("NA" if not values and available==expected else "INCOMPLETE"),
                aggregation="cell_equal; no quality interpretation"))
    return out

def statistics_mean(values):
    return float(math.fsum(values) / len(values))

def run(root):
    root = path(root).resolve()
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    plan_path = root / "plan.json"
    plan = json_read(plan_path)
    labels, requests, steps = fixed_plan(plan)
    expected = [expected_cell(l,q,s,m) for l,q,s,m in itertools.product(labels,requests,steps,("on","off"))]
    audits, layers, identities, pred = [], [], [], {}
    for e in expected:
        audit, data, lr, identity = load_cell(root,e)
        audits.append(audit); layers.extend(lr); identities.append(identity)
        if data is not None: pred[e["cell_id"]] = data
    pairs = []
    for label,q,step in itertools.product(labels,requests,steps):
        on_id = expected_cell(label,q,step,"on")["cell_id"]
        off_id = expected_cell(label,q,step,"off")["cell_id"]
        row = dict(label=label,request_id=q["request_id"],panel=q.get("panel"),
                   N=q.get("N"),K=q.get("K"),step=step,t=step/50.0,
                   status="INCOMPLETE",error=None,comparison="isolated_step_not_OFF_trajectory")
        if on_id not in pred or off_id not in pred:
            row["error"] = "One or both expected ON/OFF cells are missing or invalid"
        else:
            a,b = pred[on_id],pred[off_id]
            try:
                if a["N"] != b["N"] or a["K"] != b["K"] or not bitwise(a["state"],b["state"]):
                    raise ValueError("ON/OFF do not share the exact same state and N/K")
                K = a["K"]
                row.update(N=a["N"],K=K,state_pair_bitwise=True,
                           **pair_numbers(a["velocity"][0,K:],b["velocity"][0,K:],"velocity_"),
                           **pair_numbers(a["next_state"][0,K:],b["next_state"][0,K:],"next_state_"),
                           status="COMPLETE",error=None)
            except Exception as e:
                row.update(status="FAILED",error=f"{type(e).__name__}: {e}")
        pairs.append(row)
    assert len(audits)==48 and len(layers)==96 and len(pairs)==24
    files = {
        "cell_audit.csv": audits, "layer_metrics.csv": layers, "prediction_pairs.csv": pairs,
        "layer_label_layer_step.csv": aggregated(layers,("label","layer","step")),
        "layer_label_layer.csv": aggregated(layers,("label","layer")),
        "layer_label_step.csv": aggregated(layers,("label","step")),
        "prediction_label_step.csv": aggregated(pairs,("label","step")),
        "prediction_label.csv": aggregated(pairs,("label",)),
    }
    for name, rows in files.items(): write_csv(out/name,rows)
    expected_names = {e["cell_id"]+".json" for e in expected}
    extra = sorted(p.name for p in (root/"cells").glob("*.json") if p.name not in expected_names)
    good_cells=sum(r["status"]=="COMPLETE" for r in audits)
    good_layers=sum(r["status"]=="COMPLETE" for r in layers)
    good_pairs=sum(r["status"]=="COMPLETE" for r in pairs)
    errors=[{k:r.get(k) for k in ("cell_id","label","request_id","step","layer","status","error")}
            for r in audits+layers+pairs if r["status"]!="COMPLETE"]
    status="COMPLETE" if (good_cells,good_layers,good_pairs)==(48,96,24) and not extra else "INCOMPLETE"
    summary=dict(schema=VERSION,status=status,created_utc=datetime.now(timezone.utc).isoformat(),
        plan_path=str(plan_path),plan_sha256=digest(plan_path),script_sha256=digest(Path(__file__)),
        expected_cells=48,complete_cells=good_cells,expected_on_layers=96,complete_on_layers=good_layers,
        expected_prediction_pairs=24,complete_prediction_pairs=good_pairs,
        on_replay_true=sum(r.get("trajectory_replay_exact") is True for r in audits if r["mode"]=="on"),
        expected_on_replays=24,unexpected_cell_json=extra,
        definitions=DEFINITIONS,source_identities=identities,errors=errors,
        zero_denominator_behavior="NA preserved; aggregate mean NA unless fixed expected denominator fully finite",
        cpu_threads=2,model_calls=0,training_calls=0,new_rollouts=0,
        wall_seconds=time.perf_counter()-tick,
        output_hashes={name:digest(out/name) for name in files},
        quality_judgment="NOT_ASSESSED",
        limitation="4 identity-selected requests and 3 fixed times; no distributional or full-OFF-rollout claim.")
    json_write(out/"summary.json",summary)
    json_write(out/"metric_definitions.json",DEFINITIONS)
    print(json.dumps({k:summary[k] for k in ("status","expected_cells","complete_cells","complete_on_layers","complete_prediction_pairs","on_replay_true","wall_seconds")}))
    return 0 if status=="COMPLETE" else 2

def self_test(root):
    """Small fixed arithmetic checks only; no model, no scientific probe cells."""
    b=np.full((3,768),1.0,np.float32);r=np.full_like(b,0.001);c=r.copy();a=b.copy()
    x=layer_numbers(b,r,c,a)
    assert x["residual_nonzero_swallowed_fraction"]==1 and x["effective_over_residual_l2"]==0
    assert x["cast_zero_fraction_nonzero_residual"]==0
    z=np.zeros_like(b);y=layer_numbers(z,z,z,z)
    assert y["effective_over_residual_l2"] is None and y["effective_residual_cosine"] is None
    assert y["residual_nonzero_swallowed_fraction"] is None
    a=b+np.float32(.002); x=layer_numbers(b,r,c,a)
    assert x["effective_over_residual_l2"]>1
    p=pair_numbers(np.zeros((1,9),np.float32),np.zeros((1,9),np.float32),"v_")
    assert p["v_relative_l2_on"] is None and p["v_symmetric_relative_l2"] is None
    q=pair_numbers(np.ones((1,9),np.float32),np.zeros((1,9),np.float32),"v_")
    assert q["v_relative_l2_on"]==1 and q["v_symmetric_relative_l2"]==2
    assert q["v_changed_corner_fraction"]==1 and q["v_changed_face_fraction"]==1
    rr=[{"label":"a","status":"COMPLETE","x":1.0},{"label":"a","status":"MISSING"}]
    ag=next(x for x in aggregated(rr,("label",)) if x["metric"]=="x")
    assert ag["mean"] is None and ag["expected_cells"]==2 and ag["observed_mean"]==1
    out=path(root)/"analysis";out.mkdir(parents=True,exist_ok=True)
    receipt=dict(status="PASS",checks=["swallowed_nonzero","zero_denominator_NA","ratio_may_exceed_one",
        "symmetric_ratio","coordinate_corner_face_denominators","fixed_missing_denominator"],
        model_calls=0,script_sha256=digest(Path(__file__)),created_utc=datetime.now(timezone.utc).isoformat())
    json_write(out/"cpu_arithmetic_test.json",receipt)
    print(json.dumps(receipt))
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default=DEFAULT_ROOT)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    return self_test(args.root) if args.self_test else run(args.root)

if __name__=="__main__":
    sys.exit(main())
