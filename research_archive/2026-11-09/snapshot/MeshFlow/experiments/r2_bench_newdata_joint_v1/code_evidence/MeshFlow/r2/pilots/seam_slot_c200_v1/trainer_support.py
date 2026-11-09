"""Read-only input validation and detached training diagnostics for seam C200."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import os
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARMS = ("FULL4_C200", "SEAM_SLOT_C200")
BASE_STEP = 12000
UPDATES = 200
L = float(np.sqrt(3.) * 2 / .3762)


def resolve(path, relative_root=ROOT):
    text = str(path).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", text):
        text = "/mnt/" + text[0].lower() + text[2:]
    elif os.name == "nt" and re.match(r"^/mnt/[a-z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    p = Path(text)
    return p if p.is_absolute() else Path(relative_root) / p


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_sample(record, root, arm, step, micro, digest):
    record = {"path": record} if isinstance(record, str) else dict(record)
    p = resolve(record["path"], root)
    if record.get("sha256") and digest(p) != record["sha256"]:
        raise ValueError("Frozen sample bytes changed: " + str(p))
    with np.load(p, allow_pickle=False) as z:
        s = {k: z[k].copy() for k in z.files}
    s["path"] = str(p)
    for key in ("uid", "parent_id", "alpha", "sample_index"):
        if key in record:
            s[key] = record[key]
    s["uid"] = str(np.asarray(s.get("uid", s.get("parent_id", ""))).item())
    if not s["uid"]:
        raise ValueError("Source UID required for exposure accounting, not model input")
    s["alpha"] = float(np.asarray(s["alpha"]).item())
    s["sample_index"] = int(np.asarray(s.get("sample_index", (step - 1) * 8 + micro)).item())
    s["N"], s["K"] = int(s["N"]), int(s["K"])
    n, k = s["N"], s["K"]
    if str(s["arm"]) != arm or not 128 <= n <= 256 or not 2 <= k < n:
        raise ValueError("Registered arm/N/K differs")
    for key in ("x0", "x1", "xt", "u"):
        a = s[key]
        if a.shape != (n, 3, 3) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError("Finite FP32 sample coordinate contract differs: " + key)
    for key in ("known_mask", "valid_mask"):
        if s[key].shape != (n,) or s[key].dtype != np.bool_:
            raise ValueError("Face-mask contract differs")
    if not s["valid_mask"].all() or int(s["known_mask"].sum()) != k:
        raise ValueError("N must count all real source faces")
    fixed = s["anchor_ids"] >= 0
    if (s["anchor_ids"].shape != (n, 3) or not fixed[s["known_mask"]].all()
            or not np.array_equal(~fixed, s["free_corner_mask"])):
        raise ValueError("Corner identity/known/free partition differs")
    if not np.array_equal(s["xt"][fixed], s["C_vertices"][s["anchor_ids"][fixed]]):
        raise ValueError("C/anchor coordinates not exact")
    if np.any(s["u"][fixed] != 0):
        raise ValueError("Known target velocity must be exactly zero")
    expected = 9 * (n - k) - 6 * int(np.asarray(s["occupied_for_FM"]).sum())
    if expected != int(s["free_coordinate_count"]) or expected != 3 * int((~fixed).sum()):
        raise ValueError("Corner FM denominator differs")
    labels = s["occupied_labels"]
    if (labels.ndim != 1 or labels.dtype != np.bool_
            or s["edge_features"].shape != (len(labels), 24)
            or s["edge_features"].dtype != np.float32
            or not np.isfinite(s["edge_features"]).all()):
        raise ValueError("Occupancy training feature/label contract differs")
    if arm == "FULL4_C200" and bool(s["occupied_for_FM"].any()):
        raise ValueError("Ordinary control cannot use active seam slots")
    if arm == "SEAM_SLOT_C200" and not np.array_equal(s["occupied_for_FM"], labels):
        raise ValueError("Supervised training layout must use source occupancy labels")
    third = s["seam_third_corner_mask"]
    if (third.shape != (n, 3) or third.dtype != np.bool_
            or (third.sum(1) > 1).any() or third[s["known_mask"]].any()
            or int(third.sum()) != int(labels.sum()) or (third & fixed).any()):
        raise ValueError("Transported source seam-third diagnostic mask differs")
    winding = s["source_winding_permutation"]
    if winding.shape != (n, 3) or not np.array_equal(np.sort(winding, axis=1), np.tile(np.arange(3), (n, 1))):
        raise ValueError("Source-winding diagnostic permutation differs")
    if not 0 <= float(s["t"]) <= 1 or not .9 <= s["alpha"] <= 1.1:
        raise ValueError("Training t/width outside registered domain")
    return s


def validate_batch(samples):
    if len(samples) != 8 or sorted(Counter(s["uid"] for s in samples).values()) != [2, 2, 2, 2]:
        raise ValueError("Exactly four parents, twice each, required")
    for uid in {s["uid"] for s in samples}:
        if sum(s["alpha"] == 1. for s in samples if s["uid"] == uid) != 1:
            raise ValueError("Each parent needs one clean and one width exposure")
    edges = sum(len(s["occupied_labels"]) for s in samples)
    positives = sum(int(s["occupied_labels"].sum()) for s in samples)
    if (edges, positives) != (132, 124):
        raise ValueError("Frozen four-C edge class counts differ")
    return sum(int(s["free_coordinate_count"]) for s in samples), edges


def diagnostic(sample, predicted_numpy, bins):
    """Only an already-computed train forward; no model call or extra loss."""
    prediction = np.asarray(predicted_numpy, np.float32).reshape(sample["N"], 3, 3)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite normal-forward diagnostic")
    t = float(sample["t"])
    squared = (prediction.astype(np.float64) - sample["u"].astype(np.float64)) ** 2
    endpoint = sample["xt"] + np.float32(1 - t) * prediction
    fixed = sample["anchor_ids"] >= 0
    endpoint[fixed] = sample["xt"][fixed]
    endpoint_squared = (endpoint.astype(np.float64) - sample["x1"].astype(np.float64)) ** 2
    free = sample["free_corner_mask"]
    third = sample["seam_third_corner_mask"] & free
    interface_faces = sample["seam_third_corner_mask"].any(1)
    other = free & interface_faces[:, None] & ~third
    bulk = free & ~interface_faces[:, None]
    bi = min(len(bins) - 2, int(np.searchsorted(bins, t, side="right") - 1))
    row = dict(sample_index=sample["sample_index"], uid=sample["uid"], t=t, alpha=sample["alpha"],
               N=sample["N"], K=sample["K"], time_bin=bi,
               time_bin_lower=float(bins[bi]), time_bin_upper=float(bins[bi + 1]),
               endpoint_interpretation="Training interpolation fit, not rollout")
    for name, mask in (("source_interface_third", third), ("source_interface_other_free", other), ("bulk", bulk)):
        count = 3 * int(mask.sum())
        sse = float(squared[mask].sum())
        es = float(endpoint_squared[mask].sum())
        row.update({name + "_scalar_count": count, name + "_velocity_sse": sse,
                    name + "_velocity_mse": sse / count if count else None,
                    name + "_endpoint_sse": es, name + "_endpoint_mse": es / count if count else None,
                    name + "_endpoint_scaled_velocity_sse": (1 - t) ** 2 * sse,
                    name + "_status": "PASS" if count else "NA_NO_COORDINATES"})
    if sum(row[name + "_scalar_count"] for name in ("source_interface_third", "source_interface_other_free", "bulk")) != int(sample["free_coordinate_count"]):
        raise ValueError("Diagnostics do not partition actual FM degrees of freedom")
    row["all_free_velocity_sse"] = float(squared[free].sum())
    perm = sample["source_winding_permutation"][interface_faces]
    pred_tri = np.take_along_axis(endpoint[interface_faces], perm[..., None], axis=1).astype(np.float64)
    truth_tri = np.take_along_axis(sample["x1"][interface_faces], perm[..., None], axis=1).astype(np.float64)
    pn = np.cross(pred_tri[:, 1] - pred_tri[:, 0], pred_tri[:, 2] - pred_tri[:, 0])
    tn = np.cross(truth_tri[:, 1] - truth_tri[:, 0], truth_tri[:, 2] - truth_tri[:, 0])
    pa, ta = np.linalg.norm(pn, axis=1) / 2, np.linalg.norm(tn, axis=1) / 2
    legal = (pa > 1e-12 * L * L) & (ta > 1e-12 * L * L)
    cosines = (pn[legal] * tn[legal]).sum(1) / (4 * pa[legal] * ta[legal])
    row.update(source_interface_face_count=int(interface_faces.sum()),
               Yhat_interface_area_sum=float(pa.sum()), source_interface_area_sum=float(ta.sum()),
               Yhat_degenerate_interface_faces=int((pa <= 1e-12 * L * L).sum()),
               source_degenerate_interface_faces=int((ta <= 1e-12 * L * L).sum()),
               normal_comparable_faces=int(legal.sum()),
               source_winding_normal_cosine_mean=float(cosines.mean()) if len(cosines) else None,
               source_winding_normal_cosine_min=float(cosines.min()) if len(cosines) else None,
               source_winding_normal_opposed_faces=int((cosines < 0).sum()),
               source_winding_status="RESTORED_BEFORE_COMPARE")
    return row
