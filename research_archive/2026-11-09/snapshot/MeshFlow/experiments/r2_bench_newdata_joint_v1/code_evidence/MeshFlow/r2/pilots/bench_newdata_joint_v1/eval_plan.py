"""Frozen 14-condition / 28-request Bench evaluation; no model import.

The selection uses input part/extent metadata only.  The ten new evaluation
conditions are eight real old-pool training C plus new-parent R1/R2; R2 is a
same-parent C holdout, not an unseen-parent test. Four old DEV C retain their
original noise, queries, layout, baseline outputs and display frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from r2.pilots.bench_parts_convergence_v1 import panel as P

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_newdata_joint_v1"
VERSION = "BENCH_NEWDATA_JOINT_FIXED28_V1"
BASELINE_LABEL = "RELATION_V1_S1500"
LABEL = "NEWDATA_JOINT_S1500"
LABELS = (BASELINE_LABEL, LABEL)
ALL_LABELS = LABELS
BASELINE_ROOT = ROOT / "experiments/r2_bench_relation_training_v1"
OLD = ROOT / "experiments/r2_bench_parts_convergence_v1"
POOL = ROOT / "experiments/r2_bench_user_relation_rebuild_v1/proposed_merged_manifest.json"
THREE = ROOT / "experiments/r2_bench_three_family_v1/tasks/final_manifest.json"
FOLLOWUP = ROOT / "experiments/r2_bench_null_layout_followup_v1/SUPERVISION_CANDIDATES.json"
OLD_GALLERY = OLD / "gallery/NATIVE_S21000__SEAM_S21000_full/definition.json"
L = float(np.sqrt(3) * 2 / .3762)

TRAIN_IDS = (
    "35948e4d06a9b07e8ba8ff316aec8d3d__BACK_SURFACE_1__V2",
    "35948e4d06a9b07e8ba8ff316aec8d3d__SEAT_SURFACE_1__V2",
    "4cd23aae912b7de019fb4103277a6b93__LEG_SUPPORT_1__V2E",
    "4cd23aae912b7de019fb4103277a6b93__SEAT_SURFACE_1__V2E",
    "5f0e28d1324397d9c0b027ae5b223d6__LEG_SUPPORT_1__V2E",
    "5f0e28d1324397d9c0b027ae5b223d6__SEAT_SURFACE_1__V2E",
    "943dde2754ddc7822e8ff3556a90169__TRUE_CURVED_SEAT_UPPER_SURFACE__REL_V1",
    "943dde2754ddc7822e8ff3556a90169__COMPLETE_INNER_CURVED_BACK__REL_V1",
)
DEV_IDS = (
    "2d1e5eeb255e79dcab53bf7fa22f3636__BACK_SURFACE_1",
    "459f90aa2162b1f1d46c340938e2ff1c__SEAT_SURFACE_1",
    "9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1",
    "fd355d06238853c7f0d9805ad6c2059c__COMBO_SEAT_SUPPORT",
)
NEW_PARENT = "27eb79ddf26fd7ac7898580cb9d16864"
NEW_IDS = tuple(NEW_PARENT + "__" + recipe + "__THREE_FAMILY_V1" for recipe in ("R1", "R2"))

read, resolve, sha = P.read, P.resolve, P.sha


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def verify(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError("Frozen evaluation asset changed: " + str(path))


def requests(label=None):
    if label is not None and label not in LABELS:
        raise ValueError("Exactly the registered baseline and new endpoint are allowed")
    manifest = read(OUT / "eval_manifest.json")
    if manifest["schema"] != VERSION or manifest["status"] != "FROZEN":
        raise ValueError("Unexpected evaluation manifest identity")
    rows = manifest["requests"]
    if len(rows) != 28 or len({q["request_id"] for q in rows}) != 28:
        raise ValueError("Exactly 28 unique requests required")
    if {q["condition_id"] for q in rows} != set((*NEW_IDS, *TRAIN_IDS, *DEV_IDS)):
        raise ValueError("Input-only selection changed")
    for cid in (*NEW_IDS, *TRAIN_IDS, *DEV_IDS):
        if {q["noise_index"] for q in rows if q["condition_id"] == cid} != {0, 1}:
            raise ValueError("Exactly two frozen noises per C required")
    return rows


def generated_requests(label):
    """20 new baseline requests or all 28 new endpoint requests."""
    return [q for q in requests(label) if label != BASELINE_LABEL or not q["baseline_reuse"]]


def source_assets(q):
    for key in ("inference_npz", "metadata_npz", "source_npz", "noise_npz"):
        if sha(q[key]) != q[key + "_sha256"]:
            raise ValueError("Input identity changed: " + key)
    a = arrays(q["inference_npz"])
    if set(a) != {"C_vertices", "C_faces_original", "C_faces_model", "N", "K"}:
        raise ValueError("Inference bundle contains non-C fields")
    core = {key: a[key] for key in ("C_vertices", "C_faces_original", "C_faces_model")}
    core["total_N"] = int(a["N"])
    if (core["total_N"], len(a["C_faces_model"])) != (q["N"], q["K"]):
        raise ValueError("Frozen N/K changed")
    return core, arrays(q["source_npz"])


def root_for(label, q=None):
    if label not in LABELS:
        raise ValueError("Unknown label")
    return BASELINE_ROOT if label == BASELINE_LABEL and q is not None and q["baseline_reuse"] else OUT


def validate_assets():
    """Read-only validation, with no RNG draw, geometry scoring or model call."""
    from meshflow_control.source_winding import validate_source
    manifest = read(OUT / "eval_manifest.json")
    verify(manifest["protected"])
    rows = []
    for q in requests():
        core, source = source_assets(q)
        canonical = core["C_vertices"][core["C_faces_model"]]
        validate_source(source, canonical)
        noise = arrays(q["noise_npz"])["z"]
        if noise.shape != (q["N"] - q["K"], 3, 3) or noise.dtype != np.float32 or not np.isfinite(noise).all():
            raise ValueError("Invalid frozen Gaussian")
        descriptor = q["query_descriptor"]
        if sha(descriptor["path"]) != descriptor["sha256"]:
            raise ValueError("Frozen query changed")
        bank = arrays(descriptor["path"])
        if len(bank["C_points"]) != 2048 or not np.isclose(bank["C_weights"].sum(), 1):
            raise ValueError("Wrong C query domain")
        if int(descriptor["source_boundary_audit"]["boundary_edge_count"]) != q["boundary_edges"]:
            raise ValueError("Source C boundary identity changed")
        rows.append(dict(request_id=q["request_id"], eval_group=q["eval_group"],
                         baseline_reuse=q["baseline_reuse"], source_identity=True,
                         noise_identity=True, query_identity=True))
    return dict(status="PASS", requests=28, conditions=14, checks=rows,
                model_calls=0, geometry_scores=0, new_noise_arrays=0, new_query_arrays=0)


def prepare():
    from .eval_assets import prepare as prepare_assets
    return prepare_assets()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"), nargs="?", default="verify")
    args = parser.parse_args()
    result = prepare() if args.action == "prepare" else validate_assets()
    print({key: value for key, value in result.items() if key not in ("checks", "files", "protected")}, flush=True)
