"""Independent CPU-only verifier for reviewed source triangle exports.

Never imports the constructor, loads a model, computes OT, or repairs a mesh.
Quality partitioning reads the already frozen source-pair classifications.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from quality import partition_condition


ACCEPTED = {"ACCEPTED_AI_PROXY", "ACCEPTED_WITH_RISK_AI_PROXY"}
TIERS = ("small", "medium", "full")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def same_bytes(actual, expected):
    return (actual.dtype == expected.dtype and actual.shape == expected.shape
            and actual.tobytes(order="C") == expected.tobytes(order="C"))


def inventory_array_sha(array):
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def connected_components(source_faces, ids):
    """Exact original vertex-ID edge adjacency, with no coordinate welding."""
    selected = set(map(int, ids))
    links = {face: set() for face in selected}
    edge_faces = defaultdict(list)
    for face in selected:
        vertices = list(map(int, source_faces[face]))
        for i in range(3):
            edge_faces[tuple(sorted((vertices[i], vertices[(i + 1) % 3])))].append(face)
    for group in edge_faces.values():
        for face in group:
            links[face].update(set(group) - {face})
    result = []
    while selected:
        pending = [min(selected)]
        component = set()
        while pending:
            face = pending.pop()
            if face in component:
                continue
            component.add(face)
            pending.extend(links[face] - component)
        selected -= component
        result.append(sorted(component))
    return result


def triangle_areas(triangles):
    points = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    return np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) * .5


def final_status(value):
    return (value in ACCEPTED or value == "NA" or value.startswith("NA_")
            or value == "REJECTED" or value.startswith("REJECTED_"))


class Verifier:
    def __init__(self):
        self.failures = []
        self.check_count = 0
        self.hashes = {}

    def check(self, truth, code, subject, details=None):
        self.check_count += 1
        if not truth:
            row = dict(code=code, subject=subject)
            if details is not None:
                row["details"] = details
            self.failures.append(row)
        return bool(truth)

    def hash(self, path, expected, subject):
        path = str(path)
        try:
            actual = self.hashes.get(path)
            if actual is None:
                actual = sha(path)
                self.hashes[path] = actual
            return self.check(actual == expected, "SHA256_UNCHANGED", subject,
                              dict(path=path, expected=expected, actual=actual))
        except Exception as exc:
            return self.check(False, "HASH_READ_EXCEPTION", subject, str(exc))


def quality_summary(rows, parents):
    result = {}
    for split in ("train", "dev", "all"):
        selected = [r for r in rows if split == "all" or r["split"] == split]
        unique = sorted({r["uid"] for r in selected})
        pp = [parents[u] for u in unique]
        partitions = {}
        for name in ("CC", "CF", "FF"):
            parts = [r["partitions"][name] for r in selected]
            partitions[name] = dict(
                confirmed_pair_occurrences=sum(p["confirmed_illegal_pair_count"] for p in parts),
                unknown_pair_occurrences=sum(p["unknown_pair_count"] for p in parts),
                conditions_with_confirmed_pairs=sum(p["confirmed_illegal_pair_count"] > 0 for p in parts),
                conditions_with_unknown_pairs=sum(p["unknown_pair_count"] > 0 for p in parts),
                expected_pair_occurrences=sum(p["expected_pairs"] for p in parts))
        result[split] = dict(condition_count=len(selected), unique_parent_count=len(unique),
            unique_parent_uids=unique,
            unique_parent_confirmed_pairs=sum(p["confirmed_illegal_pair_count"] for p in pp),
            unique_parent_unknown_pairs=sum(p["unknown_pair_count"] for p in pp),
            unique_parents_with_confirmed_pairs=sum(p["confirmed_illegal_pair_count"] > 0 for p in pp),
            unique_parents_with_unknown_pairs=sum(p["unknown_pair_count"] > 0 for p in pp),
            partitions=partitions,
            conditions_without_confirmed_or_unknown_pairs_in_any_partition=sum(
                all(r["partitions"][n]["confirmed_illegal_pair_count"] == 0 and
                    r["partitions"][n]["unknown_pair_count"] == 0 for n in ("CC", "CF", "FF")) for r in selected))
    return result


def verify(out):
    started = time.perf_counter()
    out = Path(out)
    verifier = Verifier()
    inventory = read(out / "parent_inventory.json")
    regions = read(out / "regions.json")
    conditions = read(out / "conditions.json")
    quality = read(out / "quality/summary.json")
    parent_rows = inventory["rows"]
    parent_by_uid = {r["uid"]: r for r in parent_rows}
    region_rows = regions["rows"]
    region_by_id = {r["region_id"]: r for r in region_rows}
    quality_by_uid = {r["uid"]: r for r in quality["rows"]}
    manifest_rows = conditions["rows"]
    arrays = {}
    condition_quality = []
    condition_results = []
    region_results = []
    before_hashes = {str(out / name): sha(out / name) for name in
                     ("parent_inventory.json", "regions.json", "conditions.json", "quality/summary.json")}
    verifier.check(len(parent_rows) == len(parent_by_uid) == 34, "34_UNIQUE_SOURCE_PARENTS", "inventory")
    verifier.check(Counter(r["split"] for r in parent_rows) == {"train": 26, "dev": 8},
                   "SOURCE_SPLIT_COUNTS", "inventory")
    for key in ("uid", "group_id", "sha256", "full_target_sha256"):
        train = {r[key] for r in parent_rows if r["split"] == "train"}
        dev = {r[key] for r in parent_rows if r["split"] == "dev"}
        verifier.check(not train.intersection(dev), "TRAIN_DEV_SEPARATION", key,
                       sorted(train.intersection(dev)))
    for row in parent_rows:
        uid = row["uid"]
        verifier.hash(row["npz"], row["sha256"], uid)
        with np.load(row["npz"], allow_pickle=False) as data:
            arrays[uid] = {key: data[key].copy() for key in data.files}
        source = arrays[uid]
        target = source["full_target"]
        verifier.check(target.shape == (row["N"], 9) and target.dtype == np.float32,
                       "PARENT_FP32_N", uid)
        verifier.check(inventory_array_sha(target) == row["full_target_sha256"],
                       "PARENT_ARRAY_INVENTORY_HASH", uid)
        verifier.check(same_bytes(source["model_vertices"][source["source_face_vertex_ids"]],
                                  target.reshape(-1, 3, 3)), "PARENT_VERTEX_MAPPING", uid)
        verifier.check(same_bytes(source["pre_ot_full"] * np.float32(2), target.reshape(-1, 3, 3)),
                       "PARENT_PRE_OT_SCALE", uid)
        qrow = quality_by_uid[uid]
        verifier.hash(qrow["quality_path"], qrow["quality_sha256"], "quality/" + uid)
        qparent = read(qrow["quality_path"])
        verifier.check(hashlib.sha256(target.tobytes()).hexdigest() == qparent["full_target_sha256"],
                       "QUALITY_RAW_BYTE_HASH", uid)
        verifier.hash(qparent["scan_origin"]["path"], qparent["scan_origin"]["sha256"], "scan/" + uid)
    for path, expected in quality["protected_hashes"].items():
        verifier.hash(path, expected, "quality_protected")
    verifier.hash(Path(__file__).with_name("quality.py"), quality["implementation_sha256"], "quality_implementation")
    old = read(out / "old_regression_selection.json")
    skipped_weight_hashes = []
    for path, expected in old["source_files"].items():
        if Path(path).suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}:
            skipped_weight_hashes.append(path)
            continue
        verifier.hash(path, expected, "old_regression_protected")
    verifier.check(len({r["condition_id"] for r in old["rows"]}) == 8 and len(old["rows"]) == 16,
                   "OLD_REGRESSION_8_CONDITIONS_16_REGISTERED_REQUESTS", "old_regression")
    expected_train = {(r["uid"], kind) for r in parent_rows if r["split"] == "train" for kind in ("seat", "back")}
    actual_train = [(r["uid"], r["region_type"]) for r in region_rows if parent_by_uid[r["uid"]]["split"] == "train"]
    verifier.check(len(actual_train) == 52 and set(actual_train) == expected_train,
                   "52_TRAIN_REGION_RECORDS", "regions")
    verifier.check(len(region_rows) == len(region_by_id), "REGION_ID_UNIQUE", "regions")
    for row in region_rows:
        rid, uid = row["region_id"], row["uid"]
        status = row["status"]
        verifier.check(final_status(status), "REGION_FINAL_STATUS", rid, status)
        if status not in ACCEPTED:
            continue
        ids = row["source_face_ids"]
        n = parent_by_uid[uid]["N"]
        good_ids = bool(ids) and len(set(ids)) == len(ids) and all(isinstance(i, int) and 0 <= i < n for i in ids)
        verifier.check(good_ids, "REGION_SOURCE_IDS", rid)
        components = connected_components(arrays[uid]["source_faces"], ids) if good_ids else []
        verifier.check(len(components) == 1, "REGION_EDGE_CONNECTED", rid, components)
        region_results.append(dict(region_id=rid, uid=uid, split=parent_by_uid[uid]["split"],
                                   status=status, source_face_count=len(ids), components=components))
    verifier.hash(out / "regions.json", conditions["source_region_manifest_sha256"], "conditions_region_binding")
    ids_seen = set()
    face_sets_by_parent = defaultdict(set)
    by_region = defaultdict(list)
    for row in manifest_rows:
        cid, uid = row["condition_id"], row["uid"]
        failures_before = len(verifier.failures)
        try:
            verifier.check(cid not in ids_seen, "CONDITION_ID_UNIQUE", cid)
            ids_seen.add(cid)
            parent = parent_by_uid[uid]
            region = region_by_id[row["region_id"]]
            verifier.check(region["status"] in ACCEPTED, "EXPORTED_REGION_ACCEPTED", cid)
            verifier.check(row["split"] == parent["split"] and region["uid"] == uid,
                           "CONDITION_PARENT_SPLIT", cid)
            verifier.check(row["region_status"] == region["status"] and row["region_type"] == region["region_type"],
                           "CONDITION_REGION_BINDING", cid)
            verifier.check(set(row["region_source_face_ids"]) == set(region["source_face_ids"]),
                           "CONDITION_REGION_FACE_BINDING", cid)
            verifier.check(row["parent_npz"] == parent["npz"] and row["parent_sha256"] == parent["sha256"],
                           "CONDITION_PARENT_FILE_BINDING", cid)
            verifier.hash(row["npz"], row["sha256"], cid)
            with np.load(row["npz"], allow_pickle=False) as data:
                exported = {key: data[key].copy() for key in data.files}
            source = arrays[uid]
            n = len(source["full_target"])
            selected = np.asarray(row["source_face_ids"], dtype=np.int64)
            free = np.asarray(row["free_source_ids"], dtype=np.int64)
            k = len(selected)
            valid_ids = (len(set(selected.tolist())) == k and k > 0 and
                         bool(np.all((selected >= 0) & (selected < n))))
            verifier.check(valid_ids, "CONDITION_SOURCE_IDS", cid)
            if not valid_ids:
                raise ValueError("Invalid source IDs prevent dependent checks")
            expected_free = np.asarray([i for i in range(n) if i not in set(selected.tolist())], dtype=np.int64)
            verifier.check(same_bytes(free, expected_free), "DISJOINT_COMPLETE_COMPLEMENT", cid)
            verifier.check(set(selected.tolist()) <= set(region["source_face_ids"]), "C_SUBSET_R", cid)
            verifier.check(row["N"] == n and row["K"] == k and int(exported["N"]) == n and int(exported["K"]) == k,
                           "N_K_CONTRACT", cid)
            expected_arrays = {
                "C": source["full_target"][selected].reshape(k, 3, 3),
                "Y": source["full_target"][expected_free].reshape(n - k, 3, 3),
                "source_face_ids": selected, "free_source_ids": expected_free,
                "original_source_face_ids": source["retained_to_original_face"][selected],
                "free_original_source_face_ids": source["retained_to_original_face"][expected_free],
                "C_source_vertex_ids": source["source_face_vertex_ids"][selected],
                "Y_source_vertex_ids": source["source_face_vertex_ids"][expected_free],
                "C_source_corner_permutations": source["source_corner_permutations"][selected],
                "Y_source_corner_permutations": source["source_corner_permutations"][expected_free],
                "C_pre_ot": source["pre_ot_full"][selected],
                "Y_pre_ot": source["pre_ot_full"][expected_free],
            }
            for name, expected in expected_arrays.items():
                verifier.check(name in exported and same_bytes(exported[name], expected),
                               "EXACT_SOURCE_ARRAY_" + name, cid)
            reconstructed = np.empty_like(source["full_target"].reshape(n, 3, 3))
            reconstructed[selected] = exported["C"]
            reconstructed[expected_free] = exported["Y"]
            verifier.check(same_bytes(reconstructed, source["full_target"].reshape(n, 3, 3)),
                           "C_Y_REASSEMBLE_M_BITWISE", cid)
            verifier.check(np.isfinite(exported["C"]).all(), "C_FINITE", cid)
            areas = triangle_areas(exported["C"])
            verifier.check(bool(np.all(areas > 0)), "C_NO_EXACT_DEGENERATE_FACE", cid,
                           selected[areas <= 0].tolist())
            points = exported["C"]
            tri_keys = [tuple(sorted(tuple(float(v) for v in corner) for corner in face)) for face in points]
            verifier.check(len(set(tri_keys)) == k, "C_NO_EXACT_DUPLICATE_GEOMETRY", cid)
            comps = connected_components(source["source_faces"], selected)
            verifier.check(len(comps) == 1, "C_EDGE_CONNECTED", cid, comps)
            key = tuple(sorted(selected.tolist()))
            verifier.check(key not in face_sets_by_parent[uid], "C_DEDUP_PER_PARENT", cid)
            face_sets_by_parent[uid].add(key)
            all_areas = triangle_areas(source["full_target"])
            c_area = float(areas.sum())
            r_area = float(all_areas[region["source_face_ids"]].sum())
            m_area = float(all_areas.sum())
            expected_numbers = dict(C_area=c_area, R_area=r_area, M_area=m_area,
                                    face_fraction=k / n, region_area_fraction=c_area / r_area,
                                    parent_area_fraction=c_area / m_area)
            for name, expected in expected_numbers.items():
                verifier.check(np.isclose(row[name], expected, rtol=1e-12, atol=1e-12),
                               "AREA_OR_FRACTION_" + name, cid)
            requested = row["requested_tiers"]
            verifier.check(bool(requested) and len(requested) == len(set(requested)) and set(requested) <= set(TIERS),
                           "VALID_REQUESTED_TIERS", cid)
            by_region[row["region_id"]].append(row)
            q = partition_condition(uid, selected.tolist(), out / "quality")
            verifier.check(q["C_nonfinite_source_face_ids"] == [] and q["C_zero_area_source_face_ids"] == [],
                           "C_QUALITY_FINITE_NONDEGENERATE", cid)
            for name, denominator in (("CC", k * (k - 1) // 2), ("CF", k * (n - k)), ("FF", (n - k) * (n - k - 1) // 2)):
                verifier.check(q["partitions"][name]["expected_pairs"] == denominator,
                               "PARTITION_DENOMINATOR_" + name, cid)
            verifier.check(sum(q["partitions"][name]["confirmed_illegal_pair_count"] for name in ("CC", "CF", "FF")) ==
                           quality_by_uid[uid]["confirmed_illegal_pair_count"], "QUALITY_CONFIRMED_PARTITION_COMPLETENESS", cid)
            verifier.check(sum(q["partitions"][name]["unknown_pair_count"] for name in ("CC", "CF", "FF")) ==
                           quality_by_uid[uid]["unknown_pair_count"], "QUALITY_UNKNOWN_PARTITION_COMPLETENESS", cid)
            condition_quality.append(dict(condition_id=cid, uid=uid, split=row["split"], region_id=row["region_id"],
                N=n, K=k, parent_quality_sha256=q["parent_quality_sha256"], partitions=q["partitions"]))
            condition_results.append(dict(condition_id=cid, status="PASS" if len(verifier.failures) == failures_before else "FAIL",
                K=k, N=n, minimum_C_area=float(areas.min()), source_edge_component_count=len(comps)))
        except Exception as exc:
            verifier.check(False, "CONDITION_CHECK_EXCEPTION", cid, repr(exc))
            condition_results.append(dict(condition_id=cid, status="FAIL"))
    for rid, region in region_by_id.items():
        rows = by_region[rid]
        if region["status"] not in ACCEPTED:
            verifier.check(not rows, "UNACCEPTED_REGION_HAS_NO_EXPORT", rid)
            continue
        split = parent_by_uid[region["uid"]]["split"]
        tier_rows = defaultdict(list)
        for row in rows:
            for tier in row["requested_tiers"]:
                tier_rows[tier].append(row)
        if split == "train":
            verifier.check(set(tier_rows) == set(TIERS) and all(len(v) == 1 for v in tier_rows.values()),
                           "TRAIN_ALL_THREE_TIERS_EXACTLY_ONCE_AFTER_DEDUP", rid)
            if set(tier_rows) == set(TIERS):
                sets = [set(tier_rows[t][0]["source_face_ids"]) for t in TIERS]
                verifier.check(sets[0] <= sets[1] <= sets[2] == set(region["source_face_ids"]),
                               "TIERS_NESTED_FULL_IS_R", rid)
        else:
            verifier.check(len(rows) == 1 and region["dev_requested_tier"] in rows[0]["requested_tiers"],
                           "DEV_SINGLE_REGISTERED_TIER", rid)
        if "full" in tier_rows:
            verifier.check(set(tier_rows["full"][0]["source_face_ids"]) == set(region["source_face_ids"]),
                           "FULL_TIER_EQUALS_R", rid)
    counts = Counter(r["split"] for r in manifest_rows)
    verifier.check(counts["train"] <= 156 and counts["dev"] <= 8,
                   "AUTHORIZED_CONDITION_CAPS", "conditions", dict(counts))
    verifier.check(all(n <= 1 for n in Counter(r["uid"] for r in manifest_rows if r["split"] == "dev").values()),
                   "DEV_MAX_ONE_C_PER_PARENT", "conditions")
    verifier.check(conditions["train_condition_count"] == counts["train"] and conditions["dev_condition_count"] == counts["dev"],
                   "MANIFEST_COUNT_ACCURACY", "conditions")
    for name in ("created_noise_arrays", "OT_calls", "model_calls", "training_updates"):
        verifier.check(conditions.get(name) == 0, "ZERO_STAGE_EXECUTION_LEDGER", name)
    for path, expected in before_hashes.items():
        # Deliberately bypass the earlier read cache to catch a concurrent write.
        verifier.check(sha(path) == expected, "MANIFEST_UNCHANGED_DURING_VERIFICATION", path)
    accepted_summary = quality_summary(condition_quality, quality_by_uid)
    result = dict(schema="chair_r2_user_conditions_independent_export_validation_v1",
        status="PASS" if not verifier.failures else "FAIL", check_count=verifier.check_count,
        failure_count=len(verifier.failures), failures=verifier.failures,
        implementation_sha256=sha(__file__), input_manifests=before_hashes,
        checked_protected_file_count=len(verifier.hashes), checked_protected_file_hashes=verifier.hashes,
        skipped_checkpoint_file_rehashes=skipped_weight_hashes,
        counts=dict(source_parents=len(parent_rows), train_region_records=len(actual_train),
                    accepted_regions=len(region_results), train_conditions=counts["train"], dev_conditions=counts["dev"],
                    old_regression_conditions=8, old_regression_registered_requests=16),
        condition_results=condition_results, accepted_region_results=region_results,
        accepted_condition_source_geometry=accepted_summary,
        source_geometry_all_34=dict(parents=quality["parents"], confirmed_illegal_pair_count=quality["confirmed_illegal_pair_count"],
            unknown_pair_count=quality["unknown_pair_count"], parents_with_confirmed_illegal_pairs=quality["parents_with_confirmed_illegal_pairs"],
            parents_with_unknown_pairs=quality["parents_with_unknown_pairs"]),
        per_condition_source_geometry=condition_quality,
        geometry_interpretation=[
            "Export integrity PASS does not assert that source geometry is clean or that semantic AI_PROXY review is human approval.",
            "CC, CF and FF refer to frozen source M, not generated output; UNKNOWN remains unresolved.",
            "Condition aggregate pair occurrences repeat source pairs; unique parent totals count each parent once.",
            "No automatic parent rejection, repair, new intersection scoring or source mutation was performed.",
            "Degenerate C gate is exact zero area in float64 from stored FP32 coordinates; no new near-degenerate threshold was introduced.",
            "Inventory full_target hashes include dtype and shape; quality full_target hashes cover raw contiguous FP32 bytes."],
        execution=dict(workers=1, numeric_threads=2, cpu_cores=os.cpu_count(), cpu_affinity_count=len(os.sched_getaffinity(0)),
            load_average=list(os.getloadavg()), peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            wall_seconds=time.perf_counter() - started, model_calls=0, OT_calls=0, training_updates=0, CUDA_calls=0,
            generated_meshes=0, source_mutations=0))
    output = out / "validation_export.json"
    temporary = out / "validation_export.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(dict(status=result["status"], check_count=result["check_count"], failure_count=result["failure_count"],
        failures=result["failures"], counts=result["counts"], accepted_condition_source_geometry=accepted_summary,
        wall_seconds=result["execution"]["wall_seconds"], output=str(output)), ensure_ascii=False))
    return not verifier.failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    raise SystemExit(0 if verify(options.out) else 1)
