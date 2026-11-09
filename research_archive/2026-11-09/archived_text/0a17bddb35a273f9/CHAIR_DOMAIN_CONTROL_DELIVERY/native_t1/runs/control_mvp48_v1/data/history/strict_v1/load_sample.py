"""Frozen Control MVP48 data API; NumPy and the standard library only.

Consumers must wait for the experiment's DATA_READY.json.  The loader itself
also works before that marker so Session A can audit the prospective delivery.
``load_sample`` returns supervision and scoring data.  Only ``load_condition``
is the model-input boundary; its coordinates are already in model scale.

Rows of full_target and pre_ot_full are raw source face IDs.  Their corners use
the official canonical order.  Do not center, normalize, or multiply C by two.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np


__all__ = ["load_sample", "load_condition", "load_plan", "gaussian",
           "training_time", "permutations"]

_CYCLIC = np.asarray([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int64)
_INTERFACE_FIELDS = (
    "edge_source_vertex_ids", "known_source_face_ids", "free_source_face_ids",
    "known_corner_ids", "free_corner_ids",
)
_ARRAY_FIELDS = (
    "source_vertices", "source_faces", "full_target", "pre_ot_full",
    "source_face_vertex_ids", "canonical_to_source_face",
    "source_corner_permutations", "normalized_vertices", "model_vertices",
)


def _root(data_root):
    return Path(__file__).resolve().parent if data_root is None else Path(data_root)


def _json(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its permitted range")
    return result


def _seed(seed):
    return _integer(seed, "seed", maximum=2**32 - 1)


def _bits_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and np.array_equal(np.ascontiguousarray(left).view(np.uint8),
                               np.ascontiguousarray(right).view(np.uint8)))


def _array(value, name, shape, dtype=None):
    result = np.asarray(value)
    if result.shape != shape or (dtype is not None and result.dtype != np.dtype(dtype)):
        raise ValueError(f"{name}: expected shape {shape}, dtype {dtype}; "
                         f"got {result.shape}, {result.dtype}")
    if np.issubdtype(result.dtype, np.number) and not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _json_ids(value, name, shape):
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must contain integer IDs")
    return _array(result.astype(np.int64), name, shape, np.int64)


def _patch(data_root, uid, patch_id):
    document = _json(data_root / "patches.json")
    if not isinstance(document, dict) or not isinstance(document.get("records"), list):
        raise ValueError("patches.json must contain a records list")
    matches = [record for record in document["records"]
               if record.get("uid") == uid and record.get("patch_id") == patch_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one patch record for {uid}/{patch_id}, found {len(matches)}")
    return matches[0]


def _validate_interface(raw_interface, source_faces, corner_vertex_ids, known, free):
    if not isinstance(raw_interface, dict):
        raise ValueError("interface must be a dictionary")
    edge_values = raw_interface.get("edge_source_vertex_ids")
    if not isinstance(edge_values, list) or not edge_values:
        raise ValueError("A patch must have at least one genuine known/free interface edge")
    count = len(edge_values)
    interface = {}
    for name in _INTERFACE_FIELDS:
        shape = (count,) if name.endswith("source_face_ids") else (count, 2)
        interface[name] = _json_ids(raw_interface[name], f"interface.{name}", shape)

    incidences = defaultdict(list)
    for face_id, face in enumerate(source_faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            incidences[tuple(sorted((int(face[a]), int(face[b]))))].append(face_id)
    known_set, free_set = set(known.tolist()), set(free.tolist())
    expected = {}
    adjacency = defaultdict(set)
    for edge, faces in incidences.items():
        inside = [face for face in faces if face in known_set]
        outside = [face for face in faces if face in free_set]
        for face in inside:
            adjacency[face].update(other for other in inside if other != face)
        if inside and outside:
            if len(faces) != 2 or len(inside) != 1 or len(outside) != 1:
                raise ValueError(f"Nonmanifold crossing source edge {edge}")
            expected[edge] = (inside[0], outside[0])
    reached, pending = set(), [int(known[0])]
    while pending:
        face = pending.pop()
        if face not in reached:
            reached.add(face)
            pending.extend(adjacency[face] - reached)
    if reached != known_set:
        raise ValueError("Known patch is not connected through complete source edges")
    edges = interface["edge_source_vertex_ids"]
    edge_keys = [tuple(map(int, edge)) for edge in edges]
    if any(a >= b for a, b in edge_keys) or len(set(edge_keys)) != count:
        raise ValueError("Interface edges must be unique ascending source vertex pairs")
    if set(edge_keys) != set(expected):
        raise ValueError("Interface labels do not enumerate every source known/free crossing")
    for row, edge in enumerate(edge_keys):
        known_face = int(interface["known_source_face_ids"][row])
        free_face = int(interface["free_source_face_ids"][row])
        if (known_face, free_face) != expected[edge]:
            raise ValueError("Interface labels do not match original source face incidence")
        for role, face in (("known", known_face), ("free", free_face)):
            corners = interface[f"{role}_corner_ids"][row]
            if np.any((corners < 0) | (corners > 2)) or corners[0] == corners[1]:
                raise ValueError("Interface endpoint corner IDs must be distinct in [0, 2]")
            if not np.array_equal(corner_vertex_ids[face, corners], edges[row]):
                raise ValueError("Interface corner order does not correspond to ascending source endpoints")
    return interface


def load_sample(uid, patch_id, data_root=None):
    """Load and validate one patch plus its GT, source topology, and transform.

    This result contains supervision-only information and must not be passed as
    a whole to a model.  Source IDs address the original NPZ face rows; known
    IDs retain BFS order and free IDs are their ascending complement.
    """
    if not isinstance(uid, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", uid) is None:
        raise ValueError("uid must be a safe source identifier")
    if not isinstance(patch_id, str) or re.fullmatch(r"p[0-3]", patch_id) is None:
        raise ValueError("patch_id must be p0, p1, p2, or p3")
    root = _root(data_root)
    patch = _patch(root, uid, patch_id)
    n = _integer(patch["N"], "N", minimum=128, maximum=256)
    k = _integer(patch["K"], "K", minimum=1, maximum=n - 1)
    if k != int(np.floor(0.2 * n + 0.5)):
        raise ValueError("K does not follow the registered floor(0.2*N+0.5) rule")
    known = _json_ids(patch["source_face_ids"], "source_face_ids", (k,))
    free = _json_ids(patch["free_source_ids"], "free_source_ids", (n - k,))
    if np.any((known < 0) | (known >= n)) or len(np.unique(known)) != k:
        raise ValueError("Known source IDs must be unique valid source face IDs")
    expected_free = np.setdiff1d(np.arange(n, dtype=np.int64), known, assume_unique=True)
    if not np.array_equal(free, expected_free):
        raise ValueError("Free source IDs must be the complete ascending known complement")
    constraints_key = patch["constraints_array_key"]
    if constraints_key != f"{patch_id}_constraints":
        raise ValueError("Patch constraints array key is inconsistent with patch_id")

    with np.load(root / "meshes" / f"{uid}.npz", allow_pickle=False) as saved:
        if saved["uid"].shape != () or str(saved["uid"].item()) != uid:
            raise ValueError("NPZ source UID differs from the requested patch")
        arrays = {key: saved[key].copy() for key in _ARRAY_FIELDS}
        constraints = saved[constraints_key].copy()
        transform_value = saved["transform_json"]
        if transform_value.shape != ():
            raise ValueError("transform_json must be a scalar JSON string")
        transform = json.loads(str(transform_value.item()))
        bbox_value = saved["bbox_diagonal"]
        if bbox_value.shape != ():
            raise ValueError("bbox_diagonal must be scalar")
        bbox = float(bbox_value.item())

    vertices = arrays["source_vertices"]
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("source_vertices must have shape [V, 3]")
    _array(vertices, "source_vertices", vertices.shape)
    if not np.issubdtype(vertices.dtype, np.floating) or np.ptp(vertices, axis=0).max() <= 0:
        raise ValueError("Raw source coordinates require finite nonzero scale")
    faces = _array(arrays["source_faces"], "source_faces", (n, 3))
    if not np.issubdtype(faces.dtype, np.integer) or np.any((faces < 0) | (faces >= len(vertices))):
        raise ValueError("source_faces must contain valid source vertex indices")
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != n:
        raise ValueError("Duplicate unordered source faces are not permitted")
    target = _array(arrays["full_target"], "full_target", (n, 9), np.float32)
    pre_ot = _array(arrays["pre_ot_full"], "pre_ot_full", (n, 3, 3), np.float32)
    constraints = _array(constraints, "constraints", (k, 3, 3), np.float32)
    if not _bits_equal(constraints, target[known].reshape(k, 3, 3)):
        raise ValueError("C differs bitwise from the selected complete-object target faces")
    if not _bits_equal((pre_ot * np.float32(2)).reshape(n, 9), target):
        raise ValueError("Official captured pre-OT x2/model coordinate contract differs")

    corner_ids = _array(arrays["source_face_vertex_ids"], "source_face_vertex_ids", (n, 3), np.int64)
    corner_permutations = _array(arrays["source_corner_permutations"],
                                 "source_corner_permutations", (n, 3), np.int64)
    if not np.array_equal(np.sort(corner_permutations, axis=1), np.tile(np.arange(3), (n, 1))):
        raise ValueError("Each source corner mapping must be a permutation of [0, 1, 2]")
    if not np.array_equal(corner_ids, np.take_along_axis(faces, corner_permutations, axis=1)):
        raise ValueError("Canonical corner-to-source vertex mapping differs from raw faces")
    face_order = _array(arrays["canonical_to_source_face"], "canonical_to_source_face", (n,), np.int64)
    if not np.array_equal(np.sort(face_order), np.arange(n)):
        raise ValueError("Canonical-to-source face mapping must be a bijection")
    _array(arrays["normalized_vertices"], "normalized_vertices", vertices.shape)
    model_vertices = _array(arrays["model_vertices"], "model_vertices", vertices.shape, np.float32)
    if not _bits_equal(model_vertices[corner_ids], target.reshape(n, 3, 3)):
        raise ValueError("Model vertices do not reproduce complete-object target bits")
    if not isinstance(transform, dict) or not transform:
        raise ValueError("Recorded complete-object transform must be a nonempty JSON object")
    expected_bbox = float(np.linalg.norm(np.ptp(target.reshape(-1, 3), axis=0)))
    if not np.isfinite(bbox) or bbox <= 0 or not np.isclose(bbox, expected_bbox, rtol=1e-6, atol=0):
        raise ValueError("Recorded bbox diagonal differs from full model-coordinate target")
    interface = _validate_interface(patch["interface"], faces, corner_ids, known, free)
    if not isinstance(patch["patch_metrics"], dict):
        raise ValueError("patch_metrics must be a dictionary")
    return dict(uid=uid, patch_id=patch_id, N=n, K=k, constraints=constraints,
                free_target=target[free].copy(), pre_ot_free=pre_ot[free].copy(),
                source_face_ids=known.copy(), free_source_ids=free.copy(),
                bbox_diagonal=bbox, transform=transform, interface=interface,
                patch_metrics=patch["patch_metrics"], **arrays)


def load_condition(uid, patch_id, data_root=None):
    """Return only C, N, known_mask, valid_mask; faces are known first.

    C has shape [K,3,3].  Masks have length N.  The caller provides the free
    state and time separately, and constructs C-only geometry from C.
    """
    sample = load_sample(uid, patch_id, data_root)
    n, k = sample["N"], sample["K"]
    return {"C": sample["constraints"].copy(), "N": n,
            "known_mask": np.arange(n) < k, "valid_mask": np.ones(n, dtype=bool)}


def load_plan(kind, data_root=None):
    """Read the registered train, validation, or test identity plan as JSON."""
    if kind not in ("train", "validation", "test"):
        raise ValueError("kind must be train, validation, or test")
    document = _json(_root(data_root) / f"{kind}_plan.json")
    if not isinstance(document, dict) or not isinstance(document.get("records"), list):
        raise ValueError("A task plan must contain a records list")
    return document


def gaussian(seed, free_count, fp32=False):
    """Local MT19937 standard normals [F,3,3], float64 unless explicitly cast.

    Training OT consumes the float64 array.  Sampling may request FP32.  No
    coordinate correspondence is implied between objects with different N.
    """
    free_count = _integer(free_count, "free_count")
    result = np.random.RandomState(_seed(seed)).randn(free_count, 3, 3)
    return result.astype(np.float32) if fp32 else result


def training_time(t_seed):
    """One independent local MT19937 normal, sigmoid, then scalar FP32."""
    normal = np.random.RandomState(_seed(t_seed)).randn()
    return np.float32(1.0 / (1.0 + np.exp(-normal)))


def permutations(permutation_seed, N, K):
    """Return free order, pre-face-order cyclic corners, then full face order.

    Apply the returned corner rows to known-first/free-second rows before the
    full face permutation.  Source IDs, masks, and all supervision mappings
    must follow the same permutations.  The three draws share only this call's
    local RandomState, in the registered order.
    """
    n = _integer(N, "N", minimum=1)
    k = _integer(K, "K", maximum=n)
    rng = np.random.RandomState(_seed(permutation_seed))
    free_order = rng.permutation(n - k)
    corners = _CYCLIC[rng.randint(0, 3, size=n)]
    face_order = rng.permutation(n)
    return free_order, corners, face_order
