"""Freeze assets and GT-only old/new K12 contexts before any new model call.

This module never constructs a model, calls OT, or reads generated geometry or
scores to choose contexts. Historical input archives are inspected only for
trusted source-index subsets of the two exact registered parent targets.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
INPUTS = REPO / "output/t1_publish_20260930/chair_n112.npz"
INPUT_SHA = "7059671d55f2f3be8777d13e02a62255d8a0c0c7c1c1c5b4f3b24357091ba6fc"
OLD_BANK = REPO / "native_t1/runs/geometry_context_revised_20260930_153000/registration/evaluation_inputs.npz"
OLD_BANK_SHA = "2a27cf5d576272947b6944f71b7f17ab32ca1489d43d267204303c114a82248e"
PRIOR = REPO / "native_t1/runs/l3_l5_interface_pilot_20261001_104600"
MIGRATION = REPO / "native_t1/maintenance/jedge5_cleanup_20261001_115000/checkpoint_migration.json"
EXPECTED_STATES = {
    "L0": "f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b",
    "L5": "3c596db36fd7d5836427906fedecd7485b7be3e62cc6f605dbcb25b62c0c6641",
    "JEdge5": "b9c1df5326ad4815ec3c32ed20e08dc34ff6248ff05be2c412e051fcb728aae3",
}
PARENT_UIDS = ("2af09bd8df40506c9e646678ef50aa3d", "5b0e833cf2ea465e42bd82e95587d8af")
SEED_START = 10201
SEED_STRIDE = 100
SCHEMA = "edge_factorial_registration_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError("Asset hash differs: " + str(path))
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.dtype, value.shape)).encode() + value.tobytes()).hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    return record(path)


def _bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _historical_paths(out):
    roots = [REPO / "native_t1/runs", REPO / "experiments/mf_h1_local0/outputs", REPO / "output"]
    paths = []
    for root in roots:
        for base, dirs, files in os.walk(root):
            base = Path(base)
            dirs[:] = sorted(d for d in dirs if d not in ("source", "source_snapshot", "__pycache__")
                             and not (base / d).resolve().is_relative_to(out))
            for name in sorted(files):
                p = base / name
                if not p.resolve().is_relative_to(out):
                    paths.append(p)
    return roots, paths


def audit_seeds(paths, roots, out):
    """One scan of semantic seed fields and raw filenames; no score selection."""
    field = re.compile(r'"([^"\n]*(?:seed|seeds))"\s*:\s*(\[[^\]]*\]|-?\d+)', re.I)
    used = defaultdict(list)
    json_documents = 0
    for p in paths:
        for match in re.finditer(r"(?:seed|noise)[_-]?(\d+)", p.name, re.I):
            used[int(match.group(1))].append(dict(path=str(p), kind="filename"))
        if p.suffix.lower() not in (".json", ".jsonl"):
            continue
        json_documents += 1
        text = p.read_text(encoding="utf-8-sig")
        for match in field.finditer(text):
            try:
                value = json.loads(match.group(2))
            except json.JSONDecodeError:
                continue
            for value in value if isinstance(value, list) else [value]:
                if type(value) is int:
                    used[value].append(dict(path=str(p), kind="semantic_seed_field", field=match.group(1)))
    attempts = []
    for offset in range(10000):
        seeds = list(range(SEED_START + offset * SEED_STRIDE, SEED_START + offset * SEED_STRIDE + 4))
        hits = [dict(seed=s, **row) for s in seeds for row in used.get(s, ())]
        attempts.append(dict(seeds=seeds, hits=hits, previously_recorded=bool(hits)))
        if not hits:
            return dict(status="PASS", seeds=seeds, fixed_fallback_rule="10201+100*j through 10204+100*j; smallest j>=0 with no prior recorded use",
                        attempts=attempts, roots=[str(p) for p in roots], excluded_current_run=str(out),
                        files_examined=len(paths), json_documents_examined=json_documents,
                        scope="Historical semantic seed/seeds JSON fields and seed/noise filenames; source code constants and numerical metric digits are not usage evidence.",
                        before_new_generation=True)
    raise RuntimeError("No unused deterministic seed interval")


def audit_history(paths, arrays):
    """Inspect every discoverable input-named NPZ in the research roots.

    Parent identity, archived target order, C coordinates and indices must all
    agree before a historical 12-face set is trusted. Unknown layouts remain
    explicitly outside the coverage claim; no nearest-face inference is used.
    """
    candidates = sorted({p for p in paths if p.suffix.lower() == ".npz" and
                         ("input" in p.name.lower() or p.name == "chair_n112.npz") and
                         "noise" not in p.name.lower()} | {INPUTS, OLD_BANK})
    records = []
    historical = {0: {}, 1: {}}
    for p in candidates:
        item = dict(**record(p), verified_K12_records=[], status="NO_VERIFIED_K12_RECORD", unknown_K12_keys=[])
        with np.load(p, allow_pickle=False) as f:
            keys = list(f.files)
            for key in keys:
                match = re.fullmatch(r"parent([01])_(?:K12|patch\d+)_source_face_ids", key)
                if match is None:
                    continue
                ids = np.asarray(f[key])
                if ids.shape != (12,):
                    continue
                parent = int(match.group(1)); prefix = key[:-len("source_face_ids")]
                target_key = f"parent{parent}_target"
                if target_key not in f or prefix + "constraints" not in f:
                    item["unknown_K12_keys"].append(dict(key=key, reason="missing target/C identity evidence")); continue
                target, C = f[target_key], f[prefix + "constraints"]
                if not _bits(target, arrays[target_key]):
                    item["unknown_K12_keys"].append(dict(key=key, reason="different full parent coordinates/order")); continue
                if ids.dtype.kind not in "iu" or len(set(ids.tolist())) != 12 or np.any((ids < 0) | (ids >= 112)):
                    raise ValueError("Invalid historical source IDs: " + str(p) + ":" + key)
                if not _bits(target[ids].reshape(12, 3, 3), C):
                    raise ValueError("Historical C/source identity differs: " + str(p) + ":" + key)
                canon = sorted(map(int, ids)); raw = sorted(map(int, arrays[f"parent{parent}_canonical_to_raw_face"][ids]))
                row = dict(parent_index=parent, key=key, canonical_face_ids=canon, raw_source_face_ids=raw,
                           face_set_sha256=array_hash(np.asarray(raw, dtype=np.int64)), source_mapping_verified=True)
                item["verified_K12_records"].append(row)
                value = historical[parent].setdefault(tuple(canon), dict(**row, occurrences=[]))
                value["occurrences"].append(dict(path=str(p), sha256=item["sha256"], key=key))
        if item["verified_K12_records"]:
            item["status"] = "PASS_VERIFIED_SOURCE_K12"
        records.append(item)
    for parent in (0, 1):
        required = {tuple(sorted(map(int, arrays[f"parent{parent}_patch{j}_source_face_ids"]))) for j in (0, 1, 2)}
        if not required <= set(historical[parent]):
            raise ValueError("Old registered three contexts absent from history audit")
    return historical, dict(status="PASS_WITH_EXPLICIT_SCOPE", archives=records,
        archive_count=len(records), unique_K12_face_sets_by_parent={str(p): list(v.values()) for p, v in historical.items()},
        scope="All discoverable input-named NPZ registrations in native_t1/runs, experiments/mf_h1_local0/outputs and output, plus pinned training archive and old evaluation bank. Only exact full-target/source/C identities are accepted. Snapshot duplicates and generated outputs are not selection inputs.",
        limitation="Does not certify absent/deleted/unrecorded contexts, unrelated worktrees, or contexts without trusted source-index registration. New means different registered face set within this audited history, not unseen parent geometry.",
        selected_using_generated_outputs=False)


def _areas(target):
    f = target.reshape(-1, 3, 3).astype(np.float64)
    return np.linalg.norm(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]), axis=1) * .5


def _adjacency(faces):
    edge_faces = defaultdict(list)
    for i, triangle in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge_faces[tuple(sorted((int(triangle[a]), int(triangle[b]))))].append(i)
    adjacency = [set() for _ in faces]
    for incident in edge_faces.values():
        for i in incident:
            adjacency[i].update(j for j in incident if j != i)
    return adjacency, edge_faces


def _bfs(seed, adjacency, raw_ids, valid):
    queue = deque([seed]); seen = {seed}; order = []
    while queue and len(order) < 12:
        face = queue.popleft()
        if not valid[face]:
            continue
        order.append(face)
        for neighbor in sorted(adjacency[face], key=lambda i: int(raw_ids[i])):
            if neighbor not in seen and valid[neighbor]:
                seen.add(neighbor); queue.append(neighbor)
    return order


def _select(parent, arrays, historical):
    pp = f"parent{parent}_"; F = arrays[pp + "source_face_vertex_ids"]
    raw = arrays[pp + "canonical_to_raw_face"]; area = _areas(arrays[pp + "target"])
    if raw.dtype.kind not in "iu" or sorted(raw.tolist()) != list(range(112)):
        raise ValueError("Raw source face mapping is not the registered 112-face bijection")
    valid = np.isfinite(area) & (area > 0)
    adjacency, edge_faces = _adjacency(F)
    train = set(map(int, arrays[pp + "patch0_source_face_ids"]))
    selected = []; attempts = []
    for seed in np.argsort(raw, kind="stable"):
        ids = _bfs(int(seed), adjacency, raw, valid); chosen = set(ids)
        row = dict(raw_seed_face_id=int(raw[seed]), canonical_seed_face_id=int(seed),
                   canonical_face_ids=ids, raw_source_face_ids=raw[ids].tolist(), faces=len(ids),
                   different_from_training=len(chosen - train))
        if len(ids) != 12: reason = "LESS_THAN_12_NONDEGENERATE_CONNECTED_FACES"
        elif tuple(sorted(ids)) in historical: reason = "ALREADY_REGISTERED_K12_FACE_SET"
        elif len(chosen - train) < 6: reason = "FEWER_THAN_6_FACES_OUTSIDE_TRAIN_C12"
        elif any(len(chosen - set(prior)) < 6 for prior in selected): reason = "FEWER_THAN_6_FACES_DIFFERENT_FROM_OTHER_NEW_PATCH"
        else:
            reason = "ACCEPT"; selected.append(ids)
        row["decision"] = reason; attempts.append(row)
        if len(selected) == 2:
            break
    return selected, dict(parent_index=parent, requested=2, selected=len(selected),
                         status="PASS" if len(selected) == 2 else "INSUFFICIENT_CANDIDATES",
                         nondegenerate_rule="Finite FP64 area>0; no change to original coordinates",
                         minimum_positive_area=float(area[valid].min()), attempts=attempts,
                         raw_face_ID_bijection=sorted(raw.tolist()) == list(range(112)),
                         selection_used_generated_outputs=False)


def _case_metadata(parent, patch, arrays, history, old_row=None):
    prefix = f"parent{parent}_patch{patch}_"; pp = f"parent{parent}_"
    ids = arrays[prefix + "source_face_ids"]; faces = arrays[pp + "source_face_vertex_ids"]
    raw = arrays[pp + "canonical_to_raw_face"]; vertices = arrays[pp + "source_vertices"]
    area = _areas(arrays[pp + "target"]); adjacency, global_edges = _adjacency(faces)
    selected = set(map(int, ids)); train = set(map(int, arrays[pp + "patch0_source_face_ids"]))
    incidence = Counter(edge for i in selected for edge in
                        (tuple(sorted((int(faces[i, a]), int(faces[i, b])))) for a, b in ((0, 1), (1, 2), (2, 0))))
    boundary = [edge for edge, count in incidence.items() if count == 1]
    interface = [edge for edge in boundary if len(global_edges[edge]) == 2 and sum(i in selected for i in global_edges[edge]) == 1]
    seen = set(); components = 0
    for seed in selected:
        if seed in seen: continue
        components += 1; queue = [seed]; seen.add(seed)
        while queue:
            i = queue.pop()
            for j in adjacency[i] & selected:
                if j not in seen: seen.add(j); queue.append(j)
    overlaps = [dict(face_set_sha256=x["face_set_sha256"], overlap_faces=len(selected & set(key)),
                     overlap_fraction=len(selected & set(key)) / 12, occurrences=x["occurrences"])
                for key, x in history.items()]
    row = dict(parent_index=parent, parent_uid=PARENT_UIDS[parent], patch_index=patch,
               unit_id=f"P{parent}_patch{patch}", case_id=f"P{parent}_patch{patch}_K12",
               parent=parent, patch=patch, patch_group="old" if patch < 3 else "new", K=12, N=112,
               canonical_face_ids=ids.tolist(), source_face_ids=ids.tolist(), raw_source_face_ids=raw[ids].tolist(),
               source_corner_order="Unchanged archive corner order; identity [0,1,2] per face",
               face_set_sha256=array_hash(np.sort(raw[ids]).astype(np.int64)),
               constraints_sha256=array_hash(arrays[prefix + "constraints"]),
               free_target_sha256=array_hash(arrays[prefix + "free_target"]),
               bbox_diagonal=float(np.linalg.norm(np.ptp(arrays[pp + "target"].reshape(-1, 3), axis=0))),
               patch_area=float(area[ids].sum()), full_area=float(area.sum()), patch_area_fraction=float(area[ids].sum() / area.sum()),
               source_connected_components=components, source_boundary_edge_count=len(boundary),
               source_interface_edge_count=len(interface),
               source_interface_length=float(sum(np.linalg.norm(vertices[a].astype(np.float64) - vertices[b].astype(np.float64)) for a, b in interface)),
               source_boundary_length=float(sum(np.linalg.norm(vertices[a].astype(np.float64) - vertices[b].astype(np.float64)) for a, b in boundary)),
               original_overlap_faces=len(selected & train), different_faces_from_training=len(selected - train),
               max_historical_overlap_fraction=max(x["overlap_fraction"] for x in overlaps), historical_overlaps=overlaps,
               free_face_count=100, coordinates_modified=False, no_new_parent_generalization_claim=True)
    if old_row is not None: row["prior_registration"] = old_row
    if patch >= 3 and components != 1: raise ValueError("New BFS patch is not connected")
    return row


def load_cases(registration):
    """Return {(parent,patch): native_t1.data.PatchCase}; no OT/model call."""
    from native_t1.data import PatchCase
    info = registration["evaluation_inputs"]
    if sha(info["path"]) != info["sha256"]: raise ValueError("Frozen evaluation inputs changed")
    cases = {}
    with np.load(info["path"], allow_pickle=False) as f:
        embedded = json.loads(str(f["registration_json"].item()))
        for row in embedded["cases"]:
            p, j = row["parent_index"], row["patch_index"]; pre = f"parent{p}_patch{j}_"
            cases[p, j] = PatchCase(p, 12, f[pre + "constraints"], f[f"parent{p}_target"],
                                    f[pre + "free_target"], f[pre + "source_face_ids"],
                                    f[pre + "free_source_ids"], row["bbox_diagonal"], f[pre + "pre_ot_free"])
    return cases


def register(out):
    """Write one new registration directory and return its small JSON record."""
    tick = time.perf_counter(); out = Path(out).resolve(); folder = out / "registration"
    if folder.exists(): raise FileExistsError(folder)
    if out.parent != REPO / "native_t1/runs" or not out.is_dir(): raise ValueError("Root must create the new run directory first")
    folder.mkdir()
    # No mutation of any imported model/data state and no checkpoint torch.load.
    input_record = record(INPUTS, INPUT_SHA); old_bank_record = record(OLD_BANK, OLD_BANK_SHA)
    migration = json.loads(MIGRATION.read_text()); endpoints = {}
    for item in migration["checkpoint_exports"]:
        name = item["arm"]
        if name not in EXPECTED_STATES: continue
        path = REPO / item["destination"]
        if item["state_sha256"] != EXPECTED_STATES[name] or not item["every_tensor_bitwise_equal"] or not item["input_rng_identical"] or item["strict_load"]["status"] != "PASS":
            raise ValueError("Compact migration identity differs: " + name)
        if path.stat().st_size != item["bytes"]: raise ValueError("Compact byte count differs")
        endpoints[name] = dict(path=str(path), bytes=item["bytes"], sha256=item["sha256"], state_sha256=item["state_sha256"],
                               cumulative_updates=3000, identity_basis="Verified lossless compact migration; root performs current strict checkpoint load/hash before GPU", current_file_rehash="ROOT_OWNED")
    if set(endpoints) != set(EXPECTED_STATES): raise ValueError("Missing retained endpoint")
    prior = json.loads((PRIOR / "results.json").read_text())
    if prior["calibration"]["lambda_edge"] != 4.071385484299878 or prior["calibration"]["lambda5"] != .3266104383520167:
        raise ValueError("Frozen coefficient differs")
    with np.load(OLD_BANK, allow_pickle=False) as f:
        old_registration = json.loads(str(f["registration_json"].item()))
        arrays = {k: f[k].copy() for k in f.files if k != "registration_json"}
    with np.load(INPUTS, allow_pickle=False) as f:
        pre_ot = {p: f[f"parent{p}_pre_ot"].copy() for p in (0, 1)}
        for p in (0, 1):
            for suffix in ("target", "source_vertices", "source_face_vertex_ids", "canonical_to_raw_face"):
                if not _bits(arrays[f"parent{p}_{suffix}"], f[f"parent{p}_{suffix}"]): raise ValueError("Training/evaluation parent source identity differs")
            if not _bits((pre_ot[p] * 2).astype(np.float32).reshape(112, 9), arrays[f"parent{p}_target"]): raise ValueError("Original pre-OT x2 scaling differs")
    old_array_hashes = {key: array_hash(value) for key, value in arrays.items()}
    roots, paths = _historical_paths(out)
    seed_audit = audit_seeds(paths, roots, out)
    historical, history_audit = audit_history(paths, arrays)
    parent_selection = []; added = []
    for parent in (0, 1):
        selected, selection = _select(parent, arrays, historical[parent]); parent_selection.append(selection)
        for index, ids in enumerate(selected, 3):
            pp = f"parent{parent}_"; prefix = pp + f"patch{index}_"; ids = np.asarray(ids, dtype=np.int64)
            free_ids = np.asarray([i for i in range(112) if i not in ids], dtype=np.int64)
            target = arrays[pp + "target"]
            arrays[prefix + "constraints"] = target[ids].reshape(12, 3, 3).copy()
            arrays[prefix + "source_face_ids"] = ids
            arrays[prefix + "free_source_ids"] = free_ids
            arrays[prefix + "free_target"] = target[free_ids].copy()
            arrays[prefix + "pre_ot_free"] = pre_ot[parent][free_ids].copy()
            added.append((parent, index))
    from ..evaluate import sample_mesh_surface
    from ..guided_native_geometry_audit import extract_boundary_segments, _sample_boundary
    for parent, patch in added:
        prefix = f"parent{parent}_patch{patch}_"; C = arrays[prefix + "constraints"]
        free = arrays[prefix + "free_target"].reshape(100, 3, 3).astype(np.float64)
        rng = np.random.RandomState(9911); sample_mesh_surface(free, 512, rng)
        arrays[prefix + "pairwise_target_points"] = sample_mesh_surface(free, 512, rng)
        arrays[prefix + "coverage_points"] = sample_mesh_surface(free, 512, np.random.RandomState(9973 + parent))
        segments, _ = extract_boundary_segments(C); points, ids, weights, _ = _sample_boundary(segments, 9919, 512)
        for name, value in (("boundary_segments", segments), ("boundary_points", points), ("boundary_point_segment_ids", ids), ("boundary_weights", weights)):
            arrays[prefix + name] = value
    cases = []
    for parent in (0, 1):
        for patch in range(5):
            if f"parent{parent}_patch{patch}_constraints" not in arrays: continue
            old_row = next((x for x in old_registration["cases"] if x["parent_index"] == parent and x["patch_index"] == patch), None)
            row = _case_metadata(parent, patch, arrays, historical[parent], old_row)
            arrays[f"parent{parent}_patch{patch}_source_corner_ids"] = arrays[f"parent{parent}_source_face_vertex_ids"][row["canonical_face_ids"]].copy()
            arrays[f"parent{parent}_patch{patch}_corner_permutations"] = np.tile(np.arange(3, dtype=np.int64), (12, 1))
            cases.append(row)
    if not all(array_hash(arrays[k]) == digest for k, digest in old_array_hashes.items()): raise ValueError("An old evaluation array changed")
    for parent in (0, 1):
        if not _bits(arrays[f"parent{parent}_source_vertices"][arrays[f"parent{parent}_source_face_vertex_ids"]].reshape(112, 9), arrays[f"parent{parent}_target"]): raise ValueError("Source IDs do not reconstruct target")
    embedded = dict(schema=SCHEMA, status="PASS", cases=cases,
        arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=array_hash(v)) for k, v in arrays.items()},
        generation_seeds=seed_audit["seeds"], old_patches=[0, 1, 2], new_patches=[3, 4],
        source_identity="Canonical face slots plus trusted canonical_to_raw_face bijection; BFS sorted by raw original source face IDs",
        parents=parent_selection, history=history_audit, seed_audit=seed_audit,
        formal_rollout_count=4 * len(cases) * 4, replay_cap=2, model_count=4,
        selection_used_generated_outputs=False, model_forwards=0, optimizer_updates=0, OT_calls=0,
        old_arrays_bitwise_preserved=len(old_array_hashes),
        reference_policy="Unchanged: pairwise second 512 draw at seed9911; coverage512 seed9973+parent; Gamma arc-length seed9919/512; fixed .0005/.001/.002/.005/.01 bbox coverage thresholds")
    npz = folder / "evaluation_inputs.npz"
    with npz.open("xb") as f: np.savez_compressed(f, **arrays, registration_json=np.asarray(json.dumps(embedded, sort_keys=True, allow_nan=False)))
    evaluation_record = record(npz)
    agents = []; searched = set()
    for path in (Path(__file__).parent, folder):
        for parent in (path, *path.parents):
            searched.add(str(parent / "AGENTS.md"))
            if (parent / "AGENTS.md").is_file(): agents.append(dict(path=str(parent / "AGENTS.md"), text=(parent / "AGENTS.md").read_text()))
            if parent == REPO.parent: break
    result = dict(schema=SCHEMA, status="FROZEN", assets=dict(inputs=input_record, old_evaluation_inputs=old_bank_record,
        evaluation_inputs=evaluation_record, endpoints=endpoints, checkpoint_migration=record(MIGRATION), prior_results=record(PRIOR / "results.json")),
        evaluation_inputs=evaluation_record, seeds=seed_audit["seeds"], units=cases,
        seed_audit=seed_audit, history_audit=history_audit, parent_selection=parent_selection,
        actual_units=len(cases), expected_formal_rollouts=16 * len(cases), old_units=6, new_units=len(added),
        old_arrays_bitwise_preserved=len(old_array_hashes),
        AGENTS=dict(searched=sorted(searched), found=agents),
        git=dict(HEAD=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                 status=subprocess.check_output(["git", "-c", "core.autocrlf=true", "status", "--porcelain"], cwd=REPO, text=True)),
        counts=dict(model_calls=0, checkpoint_loads=0, OT_calls=0, autograd_calls=0, optimizer_updates=0,
                    new_patch_BFS_parents=2, new_patch_boundary_reference_calls=len(added), new_patch_surface_sampling_calls=3 * len(added)),
        seconds=time.perf_counter() - tick)
    # This construction validates every own remainder, C and pre-OT identity.
    loaded = load_cases(result)
    if len(loaded) != len(cases): raise ValueError("Frozen bank roundtrip lost a unit")
    result["roundtrip_cases"] = len(loaded)
    result["registration_file"] = dict(path=str(folder / "registration.json"))
    _write(folder / "registration.json", result)
    return result
