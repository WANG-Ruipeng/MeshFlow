"""CPU-only TRAIN adapter for one boundary-preserving C diagonal exchange.

This module prepares data and provenance only.  It imports no model or Torch,
does not draw random arrays, and never runs OT, training, generation or scoring.
Use ``load_train_case(row)`` before ``apply_flip_to_case(case)``.  The latter
returns ``(copied_case, audit)``; a geometric BLOCKED case is left unchanged.

known_ids/free_ids continue to identify slots in the derived full target.
Changed C triangles are NOT the original source faces: their source_face_ids
are -1 and their two-face provenance is recorded separately.  Compatibility
fields named source_faces/source_corner_permutations refer to the derived
target version after the exchange; original arrays remain in the source NPZ.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
from .geometry import canonical_faces, flip_one

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_c_edgeflip_v1"
MANIFEST = ROOT / "experiments/r2_bench_high_joint_coverage_v1/data/manifest.json"
VERSION = "bench_C_edgeflip_TRAIN_data_v1"
_SPLIT_KEY = "edgeflip_input_split"
_EDGE_LABEL_KEYS = (
    "occupied_labels", "edge_owner_face", "edge_source_triple",
    "source_alias_labels", "source_supported_labels", "source_third_in_C",
)


def resolve(path):
    value = str(path).replace("\\", "/")
    for prefix in ("/mnt/e/MeshFlow/", "E:/MeshFlow/"):
        if value.startswith(prefix):
            return ROOT / value[len(prefix):]
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _require(test, message):
    if not bool(test):
        raise ValueError(message)


def _boundary(faces):
    book = defaultdict(list)
    for fi, face in enumerate(faces):
        _require(len(set(map(int, face))) == 3, "Repeated C vertex index")
        for corner in range(3):
            u, v = int(face[corner]), int(face[(corner + 1) % 3])
            book[tuple(sorted((u, v)))].append((fi, u, v))
    _require(all(len(rows) <= 2 for rows in book.values()), "Nonmanifold C edge")
    _require(all(len(rows) != 2 or rows[0][1:] == rows[1][1:][::-1]
                 for rows in book.values()), "Inconsistent C winding")
    return np.asarray([rows[0] for _, rows in sorted(book.items()) if len(rows) == 1], np.int64).reshape(-1, 3)


def _edge_features(vertices, faces, n, edges):
    """Literal CPU metadata formula used by the inherited Native runtime."""
    v, f = np.asarray(vertices, np.float32), np.asarray(faces, np.int64)
    if not len(edges):
        return np.empty((0, 24), np.float32)
    scale = float(np.sqrt(3) * 2 / .3762)
    center, span = (v.min(0) + v.max(0)) * .5, np.ptp(v, axis=0)
    degree = np.bincount(edges[:, 1:].reshape(-1), minlength=len(v))
    values = []
    for fi, u, w in edges:
        a, b = v[u], v[w]
        delta = b - a
        normal = np.cross(v[f[fi, 1]] - v[f[fi, 0]], v[f[fi, 2]] - v[f[fi, 0]])
        norm = np.linalg.norm(normal)
        _require(norm >= 1e-12, "Degenerate C edge feature")
        values.append(np.r_[(a+b)*.5/scale, delta/scale, np.abs(delta)/scale,
                            normal/norm, span/scale, center/scale,
                            np.linalg.norm(delta)/scale, len(f)/256., n/256.,
                            degree[u]/8., degree[w]/8., 1.])
    result = np.asarray(values, np.float32)
    _require(result.shape == (len(edges), 24), "Edge metadata dimension differs")
    return result


def verify_case_arrays(case):
    """Verify the exact full-target/C/free/vertex/corner roundtrip without OT."""
    n, k = int(case["N"]), int(case["K"])
    known, free = np.asarray(case["known_ids"]), np.asarray(case["free_ids"])
    full, pre = np.asarray(case["full_target"]), np.asarray(case["pre_ot_full"])
    _require(128 <= n <= 799 and 2 <= k < n, "Native N/K domain differs")
    _require(full.dtype == np.float32 and full.shape == (n, 3, 3), "FP32 target shape differs")
    _require(np.isfinite(full).all(), "Nonfinite full target")
    _require(known.dtype == np.int64 and free.dtype == np.int64, "Partition dtype differs")
    _require(len(known) == k and np.array_equal(np.sort(np.r_[known, free]), np.arange(n)), "Target partition differs")
    _require(_same(pre * np.float32(2), full), "Pre-OT half-scale differs")
    for key, expected in (("C", full[known]), ("Y", full[free]),
                          ("C_pre_ot", pre[known]), ("Y_pre_ot", pre[free])):
        _require(_same(case[key], expected), "Target partition array differs: " + key)
    v = case["C_vertices"]
    orig, model = case["C_faces_original"], case["C_faces_model"]
    p = case["C_source_corner_permutations"]
    _require(_same(v[model], full[known]), "C model vertices differ from known target")
    _require(np.array_equal(np.take_along_axis(orig, p, axis=1), model), "C source corner roundtrip differs")
    _require(_same(case["model_vertices"][case["C_source_vertex_ids"]], v), "C/global vertex mapping differs")
    _require(np.array_equal(case["C_source_vertex_ids"][orig], case["C_source_faces_original"]), "C original/global faces differ")
    _require(np.array_equal(case["C_source_vertex_ids"][model], case["source_face_vertex_ids"][known]), "C model/global faces differ")
    _require(_same(case["model_vertices"][case["source_face_vertex_ids"]], full), "Full source vertex roundtrip differs")
    _require(np.array_equal(np.take_along_axis(case["source_faces"], case["source_corner_permutations"], axis=1),
                           case["source_face_vertex_ids"]), "Full source corner roundtrip differs")
    _require(np.array_equal(case["source_faces"][known], case["C_source_faces_original"]), "Known source faces differ")
    edges = _boundary(orig)
    _require(np.array_equal(edges, case["edge_records"]), "Boundary records differ")
    for key in _EDGE_LABEL_KEYS:
        if key in case:
            _require(len(case[key]) == len(edges), "Boundary metadata length differs: " + key)
    return dict(N=n, K=k, free_faces=n-k, boundary_edges=len(edges),
                full_source_roundtrip=True, C_source_roundtrip=True, half_scale_exact=True)


def load_train_case(row, manifest=MANIFEST):
    """Authoritative TRAIN membership gate; several HIGH rows have no group."""
    registered = json.loads(resolve(manifest).read_text("utf-8-sig"))
    train = {item["condition_id"]: item for item in registered["train"]}
    item = train.get(row.get("condition_id"))
    _require(item is not None, "Condition is not in the registered TRAIN pool")
    _require(all(row.get(key) == item.get(key) for key in ("parent_id", "case_sha256")),
             "TRAIN condition source identity differs")
    _require(resolve(row["case_path"]) == resolve(item["case_path"]), "TRAIN case path differs")
    _require(row.get("group") in (None, "TRAIN"), "Evaluation group cannot be augmented")
    held = {item["parent_id"] for key in ("dev", "high_holdout") for item in registered.get(key, [])}
    _require(row["parent_id"] not in held, "Held-out parent cannot be augmented")
    path = resolve(row["case_path"])
    _require(sha(path) == row["case_sha256"], "Frozen TRAIN case hash differs")
    with np.load(path, allow_pickle=False) as archive:
        case = {key: archive[key].copy() for key in archive.files}
    _require(_SPLIT_KEY not in case, "Source NPZ unexpectedly already carries edgeflip metadata")
    case[_SPLIT_KEY] = np.asarray("TRAIN")
    verify_case_arrays(case)
    return case


def apply_flip_to_case(case):
    """Return a fresh TRAIN case and audit, keeping the entire free target exact.

    No eligible C-only edge is a recorded BLOCKED result, not an error or a
    request to try a different target.  Callers must freeze the resulting
    eligibility list and any original/flipped exposure policy before training.
    """
    _require(str(np.asarray(case.get(_SPLIT_KEY, "")).item()) == "TRAIN",
             "Use load_train_case on a registered TRAIN row before augmentation")
    _require("edgeflip_adapter_version" not in case, "Do not apply augmentation twice")
    original_checks = verify_case_arrays(case)
    before = {key: np.asarray(value).tobytes() for key, value in case.items()}
    copied = {key: value.copy() for key, value in case.items()}
    new_faces, geometry = flip_one(case["C_vertices"], case["C_faces_original"])
    audit = dict(schema=VERSION, status=geometry["status"], geometry=geometry,
                 original_checks=original_checks, training_split="TRAIN",
                 model_calls=0, random_draws=0, OT_calls=0)
    if geometry["status"] != "PASS":
        _require(all(_same(copied[key], value) for key, value in case.items()), "Blocked case changed")
        audit.update(reason=geometry["reason"], changed_target_slots=[], free_target_bitwise_same=True,
                     input_arrays_not_mutated=True, all_arrays_unchanged=True)
        return copied, audit

    selected = np.asarray(geometry["selected_face_rows"], np.int64)
    known, free = case["known_ids"], case["free_ids"]
    changed_slots = known[selected]
    new_model, permutations = canonical_faces(case["C_vertices"], new_faces)
    # Keep all untouched C rows byte-exact, including pre-existing corner order.
    model, corner = case["C_faces_model"].copy(), case["C_source_corner_permutations"].copy()
    model[selected], corner[selected] = new_model[selected], permutations[selected]
    copied["C_faces_original"] = new_faces
    copied["C_faces_model"], copied["C_source_corner_permutations"] = model, corner
    copied["C_source_faces_original"] = case["C_source_vertex_ids"][new_faces]
    copied["C"] = case["C_vertices"][model]
    copied["C_pre_ot"] = copied["C"] / np.float32(2)
    copied["full_target"][known] = copied["C"]
    copied["pre_ot_full"][known] = copied["C_pre_ot"]
    copied["source_faces"][known] = copied["C_source_faces_original"]
    copied["source_face_vertex_ids"][known] = case["C_source_vertex_ids"][model]
    copied["source_corner_permutations"][known] = corner

    old_edges, new_edges = case["edge_records"], _boundary(new_faces)
    # Sorted undirected-edge order and directed physical edges are unchanged.
    _require(np.array_equal(old_edges[:, 1:], new_edges[:, 1:]), "Physical boundary/order changed")
    copied["edge_records"] = new_edges
    if "edge_features" in copied:
        copied["edge_features"] = _edge_features(copied["C_vertices"], new_faces, int(copied["N"]), new_edges)

    copied["edgeflip_adapter_version"] = np.asarray(VERSION)
    copied["edgeflip_original_target_version_id"] = case["target_version_id"].copy()
    copied["target_version_id"] = np.asarray(VERSION)
    copied["edgeflip_original_source_face_ids"] = case["source_face_ids"].copy()
    copied["edgeflip_original_known_faces"] = case["C_faces_original"].copy()
    copied["edgeflip_changed_known_rows"] = selected.copy()
    copied["edgeflip_changed_target_slots"] = changed_slots.copy()
    copied["edgeflip_derived_face_mask"] = np.zeros(int(case["N"]), bool)
    copied["edgeflip_derived_face_mask"][changed_slots] = True
    provenance = np.full((int(case["K"]), 2), -1, np.int64)
    provenance[:, 0] = case["source_face_ids"]
    provenance[selected] = case["source_face_ids"][selected]
    copied["edgeflip_parent_source_face_ids"] = provenance
    copied["source_face_ids"][selected] = -1

    for key in ("known_ids", "free_ids", "C_vertices", "C_source_vertex_ids", "N", "K",
                "Y", "Y_pre_ot", "free_source_face_ids", "source_vertices", "normalized_vertices",
                "model_vertices", "transform_json") + _EDGE_LABEL_KEYS:
        if key in case:
            _require(_same(copied[key], case[key]), "Protected array changed: " + key)
    for key in ("full_target", "pre_ot_full", "source_faces", "source_face_vertex_ids", "source_corner_permutations"):
        _require(_same(copied[key][free], case[key][free]), "Free target/source array changed: " + key)
    unchanged_known = np.setdiff1d(np.arange(len(known)), selected)
    for key in ("C", "C_pre_ot", "C_faces_original", "C_faces_model", "C_source_corner_permutations"):
        _require(_same(copied[key][unchanged_known], case[key][unchanged_known]), "Untouched C row changed: " + key)
    _require(all(np.asarray(case[key]).tobytes() == value for key, value in before.items()), "Input case was mutated")
    final_checks = verify_case_arrays(copied)
    audit.update(final_checks=final_checks, changed_known_rows=selected.tolist(),
                 changed_target_slots=changed_slots.tolist(), free_target_bitwise_same=True,
                 free_source_mapping_bitwise_same=True, known_ids_unchanged=True, free_ids_unchanged=True,
                 N_K_free_budget_unchanged=True, directed_boundary_and_order_exact=True,
                 boundary_owner_metadata_unchanged=True, input_arrays_not_mutated=True,
                 changed_faces_have_original_source_identity=False,
                 new_target_version=VERSION,
                 identity_note="known_ids/free_ids are derived target slots; -1 marks new C triangles; original source-pair provenance is separate")
    return copied, audit


def prepare_feasibility(manifest=MANIFEST, output=OUT / "training_preparation/feasibility.json",
                        resource_report=OUT / "resources/before_preparation.json"):
    """Inspect the existing TRAIN pool once, writing a small feasibility list."""
    started = time.perf_counter()
    manifest, output, resource_report = map(resolve, (manifest, output, resource_report))
    _require(not output.exists(), "Feasibility output already exists; review existing result")
    resources = json.loads(resource_report.read_text("utf-8-sig"))
    _require(resources.get("ram", {}).get("available", 0) >= (1 << 30), "Insufficient recorded CPU RAM")
    m = json.loads(manifest.read_text("utf-8-sig"))
    rows = m["train"]
    _require(len(rows) == 30 and len({r["condition_id"] for r in rows}) == 30,
             "Current EXPOSURE TRAIN pool identity/count differs")
    held = {r["parent_id"] for key in ("dev", "high_holdout") for r in m.get(key, [])}
    _require(not held & {r["parent_id"] for r in rows}, "Held-out parent in TRAIN pool")
    protected = {str(manifest): sha(manifest), str(resource_report): sha(resource_report)}
    results = []
    for row in rows:
        case = load_train_case(row, manifest)
        adapted, audit = apply_flip_to_case(case)
        _require(_same(adapted["full_target"][adapted["free_ids"]], case["full_target"][case["free_ids"]]),
                 "Free target differs in feasibility inventory")
        protected[str(resolve(row["case_path"]))] = row["case_sha256"]
        results.append(dict(condition_id=row["condition_id"], parent_id=row["parent_id"],
                            replay_group=row["replay_group"], group="TRAIN", N=int(row["N"]), K=int(row["K"]),
                            source_case_path=str(resolve(row["case_path"])), source_case_sha256=row["case_sha256"],
                            **audit))
    _require(all(sha(path) == expected for path, expected in protected.items()), "Protected input changed")
    groups = {}
    for group in sorted({r["replay_group"] for r in results}):
        subset = [r for r in results if r["replay_group"] == group]
        groups[group] = dict(conditions=len(subset), eligible=sum(r["status"] == "PASS" for r in subset))
    result = dict(schema=VERSION, status="PREPARED_NOT_TRAINED", manifest_path=str(manifest),
                  manifest_sha256=sha(manifest), conditions=len(results),
                  eligible=sum(r["status"] == "PASS" for r in results),
                  blocked=sum(r["status"] != "PASS" for r in results), group_counts=groups,
                  blocked_reasons=dict(Counter(r["geometry"]["reason"] for r in results if r["status"] != "PASS")),
                  rows=results, protected_inputs=protected, source_code_sha256=sha(__file__),
                  geometry_source_sha256=sha(Path(__file__).with_name("geometry.py")),
                  resource_report=str(resource_report), resource_report_sha256=sha(resource_report),
                  execution=dict(workers=1, numeric_threads=1, CPU_only=True,
                                 reason="30 small existing TRAIN cases; no OT or GPU work", wall_seconds=time.perf_counter()-started),
                  model_forwards=0, training_updates=0, OT_calls=0, new_random_arrays=0,
                  scoring_calls=0, generated_outputs=0, saved_augmented_NPZ_files=0,
                  protected_inputs_unchanged=True,
                  training_authorized=False,
                  note="Feasibility only. A matched ordinary/augmentation schedule and training budget must be frozen separately.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-feasibility", action="store_true", required=True)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--output", default=str(OUT / "training_preparation/feasibility.json"))
    parser.add_argument("--resource-report", default=str(OUT / "resources/before_preparation.json"))
    args = parser.parse_args()
    prepared = prepare_feasibility(args.manifest, args.output, args.resource_report)
    print(json.dumps({key: prepared[key] for key in ("status", "conditions", "eligible", "blocked", "group_counts", "training_updates")}))
