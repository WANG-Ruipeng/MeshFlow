"""Frozen-logit occupancy audit. CPU/NumPy only, no head/model forward.

Analytic class-weight removal is a single registered policy, not a threshold
search. Existing weights, logits, labels, results and scoring code are read-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "experiments/r2_seam_slot_c200_v1"
OUT = ROOT / "experiments/r2_seam_geometry_guard_v1/occupancy"
POLICIES = ("ORIGINAL_ZERO", "ANALYTIC_DEWEIGHT", "ALL_OCCUPIED_REFERENCE")


def resolve(value):
    value = str(value).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", value):
        value = "/mnt/" + value[0].lower() + value[2:]
    elif os.name == "nt" and re.match(r"^/mnt/[a-z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csvwrite(p, rows):
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def calibrated_occupied(logits, edge_mask=None, *, negative_weight, positive_weight):
    """Reusable C-only logits -> decision; does not accept source Y/labels.

    Returns active, corrected_logit. It is only analytic loss reweighting removal,
    NOT a guarantee of empirical probability calibration or valid completions.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError("Finite one-dimensional cached/head logits required")
    if not (math.isfinite(negative_weight) and math.isfinite(positive_weight)
            and negative_weight > 0 and positive_weight > 0):
        raise ValueError("Positive finite training class weights required")
    mask = np.ones(z.shape, bool) if edge_mask is None else np.asarray(edge_mask)
    if mask.shape != z.shape or mask.dtype != np.bool_:
        raise ValueError("Matching boolean edge mask required")
    corrected = z - math.log(positive_weight / negative_weight)
    return (corrected >= 0) & mask, corrected


def summary(y, pred):
    tp, tn = int((y & pred).sum()), int((~y & ~pred).sum())
    p, n = int(y.sum()), int((~y).sum())
    return dict(edge_count=len(y), positive_labels=p, Null_labels=n,
                TP=tp, TN=tn, FP=int((~y & pred).sum()), FN=int((y & ~pred).sum()),
                accuracy=(tp + tn) / len(y), positive_recall=tp / p if p else None,
                Null_recall=tn / n if n else None,
                exact_layout=int(np.array_equal(y, pred)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out != OUT.resolve():
        raise ValueError("This frozen audit owns only its independent occupancy directory")
    if (out / "completion.json").exists():
        previous = read(out / "completion.json")
        for p, expected in previous["source_hashes"].items():
            assert sha(resolve(p)) == expected, p
        print(json.dumps({"status": "ALREADY_COMPLETE_VERIFIED", "summary": previous["summary"]}))
        return
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    reg = read(OLD / "registration.json")
    w0, w1 = map(float, reg["head_class_weights"])
    assert (w0, w1) == (8.25, 66 / 124)
    threshold = math.log(w1 / w0)
    # Freeze before any labels/logits are loaded in this executable.
    spec = dict(schema="seam_occupancy_analytic_deweight_v1", status="FROZEN",
                utc=datetime.now(timezone.utc).isoformat(), policies=POLICIES,
                negative_weight=w0, positive_weight=w1,
                original_logit_threshold=0.0, analytic_raw_logit_threshold=threshold,
                analytic_corrected_logit_threshold=0.0,
                formula="corrected_logit = raw_logit - log(w_positive/w_negative)",
                source="Frozen historical registration class weights, not fitted evaluation thresholds",
                no_scan=True, no_model_forward=True, no_head_training=True,
                interpretation="Weighted-BCE Bayes-optimum identity; empirical calibration not proven")
    write(out / "calibration_spec.json", spec)
    sources = [OLD / "registration.json", OLD / "PROTOCOL.md", OLD / "eval_manifest.json",
               OLD / "evaluation/occupancy_edges.csv", OLD / "evaluation/occupancy_summary.json",
               OLD / "inference/SEAM_SLOT_C200_occupancy.pt",
               OLD / "inference/SEAM_SLOT_C200_occupancy.json",
               ROOT / "r2/prototypes/seam_slot_c200_v1/occupancy.py",
               ROOT / "r2/prototypes/seam_slot_c200_v1/inference.py",
               ROOT / "r2/prototypes/seam_slot_c200_v1/DESIGN.md",
               ROOT / "r2/pilots/seam_slot_c200_v1/train.py",
               ROOT / "r2/pilots/seam_slot_c200_v1/trainer_support.py",
               Path(__file__).resolve()]
    manifest = read(OLD / "eval_manifest.json")
    with (OLD / "evaluation/occupancy_edges.csv").open(encoding="utf-8-sig", newline="") as f:
        old_rows = list(csv.DictReader(f))
    csv_by_c = defaultdict(list)
    for row in old_rows:
        csv_by_c[row["condition_id"]].append(row)
    edge_rows, condition_rows, layouts = [], [], {}
    all_y, all_preds = [], {p: [] for p in POLICIES}
    checked_raws = 0
    for c in manifest["conditions"]:
        cid = c["condition_id"]
        label_path = resolve(c["labels_npz"])
        sources.append(label_path)
        with np.load(label_path, allow_pickle=False) as z:
            y = z["occupied_labels"].astype(bool)
        rows = sorted(csv_by_c[cid], key=lambda r: int(r["edge_id"]))
        logits = np.asarray([float(r["logit"]) for r in rows])
        old_pred = np.asarray([r["predicted_occupied"].lower() == "true" for r in rows])
        assert len(y) == len(rows) and [int(r["edge_id"]) for r in rows] == list(range(len(y)))
        assert np.array_equal(y, [r["source_occupied"].lower() == "true" for r in rows])
        assert np.array_equal(old_pred, logits >= 0)
        requests = [q for q in manifest["requests"] if q["condition_id"] == cid]
        assert len(requests) == 4
        for request in requests:
            raw_path = OLD / "evaluation/SEAM_SLOT_C200" / request["request_id"] / "raw/raw.npz"
            sources.append(raw_path)
            with np.load(raw_path, allow_pickle=False) as z:
                assert np.array_equal(z["occupancy_logits"].astype(np.float64), logits)
                assert np.array_equal(z["occupied"].astype(bool), old_pred)
            checked_raws += 1
        cal, corrected = calibrated_occupied(logits, negative_weight=w0, positive_weight=w1)
        preds = dict(zip(POLICIES, (old_pred, cal, np.ones_like(y))))
        layouts[cid] = dict(condition_id=cid, input_C_only=c["inference_npz"],
                            raw_logits=logits.tolist(), corrected_logits=corrected.tolist(),
                            occupancy={name: p.tolist() for name, p in preds.items()},
                            source_labels_intentionally_omitted=True,
                            no_geometry_or_FM_replay_performed=True)
        for name, pred in preds.items():
            condition_rows.append(dict(condition_id=cid, parent_id=c["parent_id"], policy=name,
                                       **summary(y, pred), independent_C=1, noise_duplicates_excluded=3))
            all_preds[name].append(pred)
        all_y.append(y)
        for i in range(len(y)):
            edge_rows.append(dict(condition_id=cid, edge_id=i, source_occupied=bool(y[i]),
                                  raw_logit=float(logits[i]), original_occupied=bool(old_pred[i]),
                                  corrected_logit=float(corrected[i]), analytic_occupied=bool(cal[i]),
                                  analytic_probability=float(1 / (1 + math.exp(-corrected[i]))),
                                  all_occupied_reference=True,
                                  changed_decision=bool(old_pred[i] != cal[i])))
    full_y = np.concatenate(all_y)
    summaries = []
    for name in POLICIES:
        row = summary(full_y, np.concatenate(all_preds[name]))
        row["exact_layout"] = sum(r["exact_layout"] for r in condition_rows if r["policy"] == name)
        row.update(policy=name, independent_C=4)
        summaries.append(row)
    assert len(full_y) == 66 and int(full_y.sum()) == 62 and checked_raws == 16
    old_summary = read(OLD / "evaluation/occupancy_summary.json")["rows"][0]
    for key in ("TP", "TN", "FP", "FN", "exact_layout", "accuracy", "positive_recall", "Null_recall"):
        assert summaries[0][key] == old_summary[key], key
    # Simple analytic identity check, not a model probe or trained calibration.
    probe_p = np.asarray([.01, .1, .5, .9, .99])
    q_star = w1 * probe_p / (w1 * probe_p + w0 * (1 - probe_p))
    z_star = np.log(q_star / (1 - q_star))
    _, recovered = calibrated_occupied(z_star, negative_weight=w0, positive_weight=w1)
    recovered_p = 1 / (1 + np.exp(-recovered))
    assert np.allclose(probe_p, recovered_p, atol=1e-14, rtol=0)
    mask = np.asarray([True, False, True])
    got, _ = calibrated_occupied(np.asarray([0., 9., -10.]), mask,
                                negative_weight=w0, positive_weight=w1)
    assert got.tolist() == [True, False, False]
    csvwrite(out / "occupancy_edges.csv", edge_rows)
    csvwrite(out / "occupancy_per_condition.csv", condition_rows)
    csvwrite(out / "occupancy_summary.csv", summaries)
    write(out / "calibrated_layouts.json", layouts)
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sources}
    write(out / "SOURCE_HASHES.json", hashes)
    result = dict(status="COMPLETE", schema="seam_occupancy_analytic_deweight_audit_v1",
                  utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started,
                  summary=summaries, source_hashes=hashes,
                  checks=dict(cached_16_raw_logits_match_csv=True, source_labels_match_npz=True,
                              original_confusion_matches=True, class_weight_inverse_identity=True,
                              padding_mask_inactive=True),
                  calls=dict(head_forward=0, head_backward=0, head_optimizer=0,
                             generator_forward=0, D0_forward=0, GPU_calls=0),
                  cpu_threads=1,
                  interpretation=dict(implementation_bug="NOT_ESTABLISHED: code matches fixed weighted-BCE protocol",
                                      analytic_calibration="Equivalent to all-occupied on these four C; Null discrimination lost",
                                      FF_I_causal_claim="UNTESTED: threshold/layout change needs paired rollout, not an algebraic assertion",
                                      labels="One source completion; disagreement does not itself prove invalid alternative completion",
                                      prototype_doc_discrepancy="DESIGN.md proposed per-condition mean; final PROTOCOL/train use global 132-edge mean"))
    write(out / "completion.json", result)
    report = f'''# Frozen occupancy audit\n\nCompleted without head/generator/D0 forward, optimizer update or GPU compute.\n\nThe executed loss is the all-valid-edge mean over each 132-edge update:\n\n`L = mean[-w1*y*log(sigmoid(z)) - w0*(1-y)*log(1-sigmoid(z))]`, with `w0={w0}` and `w1={w1}`. Each microbatch edge mean is multiplied by its edge count /132. The implementation agrees with the frozen final protocol. Prototype DESIGN.md's older per-condition-average description is not the executed reduction.\n\nAt the population optimum for original probability p, weighted sigmoid q satisfies `q=w1*p/(w1*p+w0*(1-p))`. Thus the analytically deweighted logit is `z-log(w1/w0)` and p>=0.5 corresponds to raw z>={threshold:.12f}. This identity does not certify empirical calibration after 200 updates. The old z>=0 is a legitimate cost-sensitive rule, corresponding at the ideal optimum to original p>={w0/(w0+w1):.12f}; it is not an unweighted 50% rule.\n\n| Policy | TP | TN | FP | FN | Positive recall | Null recall | Exact C layouts |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'''
    for r in summaries:
        report += f'| {r["policy"]} | {r["TP"]} | {r["TN"]} | {r["FP"]} | {r["FN"]} | {r["positive_recall"]:.6f} | {r["Null_recall"]:.6f} | {r["exact_layout"]}/4 |\n'
    report += '''\nThe fixed analytic threshold activates all 66 edges: positive recall rises from36/62 to62/62, but all4 source Null edges also activate. It exactly equals the all-occupied reference on this panel. It does not demonstrate a better source-layout classifier, eliminate mismatch, or establish reduced FF/I. A rollout would test a real inference-layout intervention, with the original C-only inputs and no source-label oracle. Four C, not16 independent noise trials, are the occupancy denominator.\n\nAll16 saved RAW prediction arrays exactly agree with the per-edge table, and all source labels agree with saved NPZ labels. Existing checkpoint bytes are hashed but never loaded. The executable exposes `calibrated_occupied(logits, edge_mask, negative_weight=..., positive_weight=...)`, which accepts no Y/labels. Original model/head/threshold/defaults remain unchanged.\n'''
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "summary": summaries, "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
