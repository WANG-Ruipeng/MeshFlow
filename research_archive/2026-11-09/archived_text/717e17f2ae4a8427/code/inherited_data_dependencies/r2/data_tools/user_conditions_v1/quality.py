"""CPU-only source mesh diagnostics, with hash-checked historical scan reuse.

No model, OT, repair, mesh generation or automatic parent rejection. Run under
WSL Python. ``partition_condition(uid, source_face_ids, quality_root)`` is a
read-only API over the resulting parent files; source IDs are retained rows.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
import types

import numpy as np

VERSION = "chair_r2_user_conditions_quality_v1"
OLD_ROOT = Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_user_condition_data_audit_v1/geometry")
L = math.sqrt(3) * 2 / .3762
FROZEN_HASHES = {
    "pair_audit.py": "2e1e1b5520d808d6f11e3660373118d309dfadf18dd2e3cf90f15a7856ae3d8d",
    "intersections.py": "23a2e8c8deefcceb51ace60ad8d8a58d352d7c38bdcba2e8f85d6ceabd447d86",
    "seams.py": "9ce6927710d80eedec19aa10e2100ade582cf00e6942e9de4ef987618a005e4b",
}
PROPER = "proper_transverse_penetration"
COPLANAR = "coplanar_positive_area_overlap"
CONTACTS = frozenset(("shared_edge_contact", "shared_vertex_contact",
    "coplanar_boundary_or_tolerance_contact", "noncoplanar_edge_or_tangent_contact",
    "point_or_tangent_contact"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    """Immutable content write: identical repeat is harmless; change is refused."""
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise FileExistsError("Existing quality artifact differs: " + str(path))
        return
    with path.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def resources():
    import psutil
    gpu = {}
    for name, query in (("gpu", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu"),
                        ("gpu_processes", "--query-compute-apps=pid,process_name,used_memory")):
        proc = subprocess.run(["nvidia-smi", query, "--format=csv"], text=True, capture_output=True, check=False)
        gpu[name] = dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    return dict(time_unix=time.time(), pid=os.getpid(), cpu_cores=os.cpu_count(),
        cpu_affinity_count=len(os.sched_getaffinity(0)), load_average=list(os.getloadavg()), cpu_utilization_percent=psutil.cpu_percent(interval=.2),
        available_ram_bytes=psutil.virtual_memory().available,
        process_rss_bytes=psutil.Process().memory_info().rss,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        workers=1, numeric_threads=2, cuda_calls=0, **gpu)


def frozen_detector(old_root=OLD_ROOT):
    root = Path(old_root) / "source_snapshot/src/meshflow_control/evaluation"
    paths = {name: root / name for name in FROZEN_HASHES}
    for name, expected in FROZEN_HASHES.items():
        if sha(paths[name]) != expected:
            raise ValueError("Frozen geometry source differs: " + str(paths[name]))
    package = "_r2_quality_frozen_geometry_v1"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(root)]
        sys.modules[package] = module
    # Isolated namespace; never resolves through the installed model package.
    import importlib
    pair = importlib.import_module(package + ".pair_audit")
    detector = importlib.import_module(package + ".intersections")
    return pair.pair_scan, detector.DETECTOR_POLICY, {str(p): sha(p) for p in paths.values()}


def category(label):
    if label == PROPER:
        return "proper"
    if label == COPLANAR:
        return "coplanar_positive_overlap"
    if label == "disjoint":
        return "disjoint"
    if label in CONTACTS:
        return "legal_contact"
    return "UNKNOWN"


def load_mesh(row):
    path = Path(row["npz"])
    if sha(path) != row["sha256"]:
        raise ValueError("Parent NPZ hash differs: " + row["uid"])
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    tri = arrays["full_target"].reshape(-1, 3, 3)
    if len(tri) != row["N"] or tri.dtype != np.float32:
        raise ValueError("Expected registered FP32 full_target and N")
    original = arrays.get("retained_to_original_face")
    if original is not None and (original.shape != (len(tri),) or len(set(original.tolist())) != len(tri)):
        raise ValueError("Invalid retained-to-original face mapping")
    face_vertex_ids = arrays.get("source_face_vertex_ids")
    mapping = "NA_NOT_PROVIDED"
    if face_vertex_ids is not None:
        reconstructed = arrays["model_vertices"][face_vertex_ids]
        if reconstructed.dtype != tri.dtype or reconstructed.tobytes() != tri.tobytes():
            raise ValueError("Source vertex-ID coordinate mapping differs")
        mapping = "BITWISE_VERIFIED"
    if "pre_ot_full" in arrays and (arrays["pre_ot_full"] * np.float32(2)).tobytes() != tri.tobytes():
        raise ValueError("Saved model scale differs")
    return tri, original, face_vertex_ids, mapping


def basic_geometry(tri, original):
    t = tri.astype(np.float64)
    finite = np.isfinite(t).all(axis=(1, 2))
    area = .5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    groups = defaultdict(list)
    for i, face in enumerate(t):
        if finite[i]:
            groups[tuple(sorted(tuple(p) for p in face))].append(i)
    duplicate_groups = [g for g in groups.values() if len(g) > 1]
    per_face = []
    for i, face in enumerate(t):
        angle = None
        if finite[i] and area[i] > 0:
            angles = []
            for j in range(3):
                a, b = face[(j + 1) % 3] - face[j], face[(j + 2) % 3] - face[j]
                angles.append(float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))))
            angle = min(angles)
        per_face.append(dict(source_face_id=i,
            original_face_id=None if original is None else int(original[i]), finite=bool(finite[i]),
            area=float(area[i]) if finite[i] else None,
            exact_zero_area=bool(finite[i] and area[i] == 0), minimum_angle_degrees=angle))
    return dict(nonfinite_source_face_ids=np.flatnonzero(~finite).tolist(),
        exact_zero_area_source_face_ids=np.flatnonzero(finite & (area == 0)).tolist(),
        exact_duplicate_groups=duplicate_groups, per_face=per_face,
        angle_policy="Descriptive only; no minimum-angle cleaning gate introduced")


def summarize_pairs(rows, expected_pairs):
    counts = Counter(category(r["classification"]) for r in rows)
    labels = Counter(r["classification"] for r in rows)
    omitted = expected_pairs - len(rows)
    if omitted < 0:
        raise ValueError("More pair rows than expected")
    counts["disjoint"] += omitted
    labels["disjoint"] += omitted
    for name in ("proper", "coplanar_positive_overlap", "legal_contact", "UNKNOWN", "disjoint"):
        counts.setdefault(name, 0)
    problems = [r for r in rows if category(r["classification"]) in ("proper", "coplanar_positive_overlap", "UNKNOWN")]
    return dict(expected_pairs=expected_pairs, category_counts=dict(counts),
        detector_classification_counts=dict(labels),
        confirmed_illegal_pair_count=counts["proper"] + counts["coplanar_positive_overlap"],
        unknown_pair_count=counts["UNKNOWN"],
        confirmed_illegal_source_face_ids=sorted({i for r in problems if category(r["classification"]) != "UNKNOWN" for i in r["pair"]}),
        unknown_source_face_ids=sorted({i for r in problems if category(r["classification"]) == "UNKNOWN" for i in r["pair"]}),
        problem_pairs=problems,
        status="PARTIAL_UNKNOWN" if counts["UNKNOWN"] else "COMPLETE_STATIC_DIAGNOSTIC")


def partition_condition(uid, source_face_ids, quality_root):
    """Return CC/CF/FF counts and problematic original/source IDs, no new tests."""
    parent = read(Path(quality_root) / "parents" / (uid + ".json"))
    ids = list(source_face_ids)
    n = parent["N"]
    if any(type(i) is not int or not 0 <= i < n for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("C source face IDs must be unique integer retained rows")
    if not 0 < len(ids) < n:
        raise ValueError("C must be nonempty and leave at least one free face")
    known = set(ids)
    groups = {k: [] for k in ("CC", "CF", "FF")}
    for row in parent["pair_contacts"]:
        i, j = row["pair"]
        group = "CC" if i in known and j in known else "FF" if i not in known and j not in known else "CF"
        groups[group].append(row)
    k, f = len(ids), n - len(ids)
    expected = dict(CC=k * (k - 1) // 2, CF=k * f, FF=f * (f - 1) // 2)
    duplicate_c = [sorted(known.intersection(g)) for g in parent["basic_geometry"]["exact_duplicate_groups"] if len(known.intersection(g)) > 1]
    return dict(schema=VERSION + "_condition", uid=uid, N=n, K=k,
        source_face_ids=ids, parent_quality_sha256=sha(Path(quality_root) / "parents" / (uid + ".json")),
        partitions={g: summarize_pairs(rows, expected[g]) for g, rows in groups.items()},
        C_nonfinite_source_face_ids=sorted(known.intersection(parent["basic_geometry"]["nonfinite_source_face_ids"])),
        C_zero_area_source_face_ids=sorted(known.intersection(parent["basic_geometry"]["exact_zero_area_source_face_ids"])),
        C_duplicate_groups=duplicate_c,
        parent_confirmed_illegal_pair_count=parent["whole_parent"]["confirmed_illegal_pair_count"],
        parent_unknown_pair_count=parent["whole_parent"]["unknown_pair_count"],
        automatic_rejection=False, face_id_policy=parent["face_id_policy"],
        contact_policy="Legal contact is the frozen geometric detector class; does not certify shared original vertex-ID topology")


def run(inventory, out, old_root=OLD_ROOT):
    inventory, out, old_root = Path(inventory), Path(out), Path(old_root)
    if out.exists() and (out / "summary.json").exists():
        raise FileExistsError("Completed quality run already exists")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = read(inventory)
    parents = data if isinstance(data, list) else data.get("rows", data.get("parents"))
    if not isinstance(parents, list) or len(parents) != 34 or len({p["uid"] for p in parents}) != 34:
        raise ValueError("Registered preparation requires 34 distinct parents")
    scan_fn, policy, source_hashes = frozen_detector(old_root)
    manifest_path = old_root / "pair_cache_manifest.json"
    cache = {r["uid"]: r for r in read(manifest_path)["parents"]}
    protected = dict(source_hashes)
    protected[str(inventory)] = sha(inventory)
    protected[str(manifest_path)] = sha(manifest_path)
    before = resources()
    write_new(out / "resources_before.json", before)
    rows = []
    ledger = dict(new_unique_pair_tests=0, reused_unique_pair_tests=0, model_calls=0,
        training_updates=0, OT_calls=0, generated_meshes=0, source_modifications=0,
        workers=1, numeric_threads=2)
    for parent in sorted(parents, key=lambda p: p["uid"]):
        tick = time.perf_counter()
        uid, n = parent["uid"], parent["N"]
        tri, original, face_vertex_ids, mapping = load_mesh(parent)
        protected[parent["npz"]] = parent["sha256"]
        basic = basic_geometry(tri, original)
        if basic["nonfinite_source_face_ids"]:
            raise ValueError("Nonfinite source parent; preserve as blocked rather than call detector")
        old = cache.get(uid)
        if old is not None:
            path = old_root / old["relative_path"]
            if old["input_npz_sha256"] != parent["sha256"] or old["N"] != n or old["L"] != L or sha(path) != old["sha256"]:
                raise ValueError("Historical scan reuse identity differs: " + uid)
            scan = read(path)
            protected[str(path)] = old["sha256"]
            origin = dict(mode="HASH_VERIFIED_HISTORICAL_REUSE", path=str(path), sha256=old["sha256"])
            ledger["reused_unique_pair_tests"] += scan["tested_pairs"]
        else:
            calls = {}
            scan = scan_fn(tri, tri, L, self_pairs=True, independent=True, ledger=calls)
            scan.update(uid=uid, input_npz_sha256=parent["sha256"], L=L, ledger=calls)
            for contact in scan["contacts"]:
                contact["original_face_ids"] = None if original is None else [int(original[i]) for i in contact["pair"]]
            path = out / "new_pair_scans" / (uid + ".json")
            write_new(path, scan)
            origin = dict(mode="NEW_CPU_DUAL_DETECTOR_SCAN", path=str(path), sha256=sha(path))
            ledger["new_unique_pair_tests"] += scan["tested_pairs"]
        if scan["uid"] != uid or scan["input_npz_sha256"] != parent["sha256"] or scan["L"] != L:
            raise ValueError("Scan parent metadata differs")
        expected = n * (n - 1) // 2
        if scan["tested_pairs"] != expected or sum(scan["classification_counts"].values()) != expected or not scan["independent_verification"]:
            raise ValueError("Incomplete dual-detector scan")
        contacts, seen = [], set()
        for contact in scan["contacts"]:
            pair = contact["pair"]
            if len(pair) != 2 or not 0 <= pair[0] < pair[1] < n or tuple(pair) in seen:
                raise ValueError("Invalid or duplicate source pair")
            seen.add(tuple(pair))
            raw_ids = None if original is None else [int(original[i]) for i in pair]
            if contact.get("original_face_ids") != raw_ids:
                raise ValueError("Original face mapping differs from frozen scan")
            item = dict(pair=pair, original_face_ids=raw_ids, classification=contact["classification"],
                category=category(contact["classification"]))
            if item["category"] in ("proper", "coplanar_positive_overlap", "UNKNOWN"):
                item["detector"] = contact["detector"]
            contacts.append(item)
        whole = summarize_pairs(contacts, expected)
        value = dict(schema=VERSION + "_parent", uid=uid, split=parent["split"], N=n,
            input_npz=parent["npz"], input_npz_sha256=parent["sha256"],
            full_target_sha256=hashlib.sha256(tri.tobytes()).hexdigest(), L=L,
            face_id_policy="source IDs are unchanged retained full_target rows; original IDs map to saved raw source rows or NA",
            source_vertex_mapping=mapping, original_face_ids=None if original is None else original.tolist(),
            scan_origin=origin, basic_geometry=basic, pair_contacts=contacts, whole_parent=whole,
            automatic_rejection=False)
        destination = out / "parents" / (uid + ".json")
        write_new(destination, value)
        report = dict(uid=uid, split=parent["split"], N=n, origin=origin["mode"],
            confirmed_illegal_pair_count=whole["confirmed_illegal_pair_count"],
            unknown_pair_count=whole["unknown_pair_count"], category_counts=whole["category_counts"],
            quality_path=str(destination), quality_sha256=sha(destination), wall_seconds=time.perf_counter() - tick)
        rows.append(report)
        print(json.dumps(report), flush=True)
    for path, expected_hash in protected.items():
        if sha(path) != expected_hash:
            raise ValueError("Protected input/source changed: " + path)
    summary = dict(schema=VERSION, rows=rows, parents=len(rows),
        parents_with_confirmed_illegal_pairs=sum(r["confirmed_illegal_pair_count"] > 0 for r in rows),
        parents_with_unknown_pairs=sum(r["unknown_pair_count"] > 0 for r in rows),
        confirmed_illegal_pair_count=sum(r["confirmed_illegal_pair_count"] for r in rows),
        unknown_pair_count=sum(r["unknown_pair_count"] for r in rows),
        reused_parents=sum(r["origin"] == "HASH_VERIFIED_HISTORICAL_REUSE" for r in rows),
        newly_scanned_parents=sum(r["origin"] == "NEW_CPU_DUAL_DETECTOR_SCAN" for r in rows),
        fixed_scale=L, scale_policy="Frozen canonical training cube diagonal sqrt(3)*2/.3762",
        detector_policy=policy, source_hashes=source_hashes, protected_hashes=protected,
        implementation_sha256=sha(__file__), ledger=ledger, wall_seconds=time.perf_counter() - started,
        automatic_parent_or_condition_rejection=False,
        limitations=["Static source geometry, not generated output quality or semantics",
            "UNKNOWN is not clean or confirmed illegal", "Legal geometric contact does not certify shared vertex IDs",
            "Condition partition counts repeat source pairs and are not independent defect totals"])
    write_new(out / "resources_after.json", resources())
    write_new(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--old-root", default=str(OLD_ROOT))
    parser.add_argument("--conditions")
    args = parser.parse_args()
    if args.conditions:
        data = read(args.conditions)
        rows = data if isinstance(data, list) else data.get("rows", data.get("conditions"))
        results = []
        for row in rows:
            item = partition_condition(row["uid"], row["source_face_ids"], args.out)
            item["condition_id"] = row["condition_id"]
            write_new(Path(args.out) / "conditions" / (row["condition_id"] + ".json"), item)
            results.append(dict(condition_id=row["condition_id"], uid=row["uid"],
                counts={p: {k: item["partitions"][p][k] for k in ("confirmed_illegal_pair_count", "unknown_pair_count")} for p in ("CC", "CF", "FF")}))
        write_new(Path(args.out) / "condition_summary.json", dict(schema=VERSION, rows=results,
            conditions_sha256=sha(args.conditions), additional_pair_tests=0))
    elif args.inventory:
        result = run(args.inventory, args.out, args.old_root)
        print(json.dumps({k: result[k] for k in ("parents", "reused_parents", "newly_scanned_parents", "confirmed_illegal_pair_count", "unknown_pair_count", "wall_seconds")}), flush=True)
    else:
        parser.error("--inventory or --conditions required")


if __name__ == "__main__":
    main()
