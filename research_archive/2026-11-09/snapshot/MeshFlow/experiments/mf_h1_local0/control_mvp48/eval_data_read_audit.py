"""Session C read-only data binding and loader compatibility audit; no OT/model."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections import defaultdict

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from .eval_common import RUN, ROOT, REGISTRATION, read_json, sha256, snapshot, write_json


def main():
    started = time.perf_counter()
    resources = snapshot()
    ready_path = RUN / "DATA_READY.json"
    ready = read_json(ready_path)
    ready_hash = sha256(ready_path)
    assert ready["status"] == "READY" and ready["data_revision"] == "repair_v2"
    checked_files = {}
    for path, spec in ready["files"].items():
        observed = sha256(RUN / path)
        assert observed == spec["sha256"], path
        assert (RUN / path).stat().st_size == spec["bytes"], path
        checked_files[path] = observed
    checked_code = {}
    for path, expected in ready["code"].items():
        observed = sha256(ROOT / path)
        assert observed == expected, path
        checked_code[path] = observed
    for field in ("contract", "manifest", "split", "independent_audit"):
        item = ready[field]
        assert sha256(RUN / item["path"]) == item["sha256"], field
    for item in ready["task_plans"].values():
        assert sha256(RUN / item["path"]) == item["sha256"]

    assert read_json(RUN / ready["independent_audit"]["path"])["status"] == "PASS"

    splits = read_json(RUN / "data/splits.json")
    manifest = read_json(RUN / "data/manifest.json")
    patches = read_json(RUN / "data/patches.json")["records"]
    objects = {row["uid"]: row for row in manifest["objects"]}
    assert len(objects) == 48
    split_names = ("train", "validation", "test")
    for name, count in zip(split_names, (32, 8, 8)):
        assert len(splits[name]) == count and len(set(splits[name])) == count
        assert set(splits[name]) <= set(objects)
        expected_official = "test" if name == "test" else "train"
        assert all(splits["official_splits"][uid] == expected_official for uid in splits[name])
    assert len(set(uid for name in split_names for uid in splits[name])) == 48
    assert len(set(splits["group_ids"][uid] for uid in objects)) == 48
    assert splits["historical_exclusion_applied"] is True
    patch_counts = {}
    for name, per_parent in zip(split_names, (4, 2, 2)):
        rows = [row for row in patches if row["uid"] in splits[name]]
        assert len(rows) == len(splits[name]) * per_parent
        for uid in splits[name]:
            own = [row for row in rows if row["uid"] == uid]
            assert {row["patch_id"] for row in own} == {f"p{i}" for i in range(per_parent)}
            assert all(row["N"] == objects[uid]["N"] and row["K"] == objects[uid]["K"] for row in own)
        patch_counts[name] = len(rows)

    api_spec = importlib.util.spec_from_file_location("eval_read_frozen_data", RUN / "data/load_sample.py")
    api = importlib.util.module_from_spec(api_spec)
    api_spec.loader.exec_module(api)
    test_plan = api.load_plan("test")
    val_plan = api.load_plan("validation")
    assert len(test_plan["records"]) == 32 and len(val_plan["records"]) == 8
    assert test_plan["total_generations"] == 96
    assert val_plan["total_teacher_checks"] == 24
    assert {r["uid"] for r in test_plan["records"]} == set(splits["test"])
    assert {r["uid"] for r in val_plan["records"]} == set(splits["validation"])
    per_test = defaultdict(list)
    for row in test_plan["records"]:
        per_test[row["uid"]].append(row)
    for uid, rows in per_test.items():
        assert {(r["patch_id"], r["noise_index"]) for r in rows} == {("p0",0),("p0",1),("p1",0),("p1",1)}
        for noise in (0, 1):
            assert len({r["gaussian_seed"] for r in rows if r["noise_index"] == noise}) == 1
    assert all(row["patch_id"] == "p0" and row["noise_index"] == 0 and row["t"] == .5 for row in val_plan["records"])

    checked_conditions = []
    loaded = {}
    # Only C's 16 test conditions and eight registered teacher conditions.
    for row in test_plan["records"] + val_plan["records"]:
        key = (row["uid"], row["patch_id"])
        if key in loaded:
            continue
        sample = api.load_sample(*key)
        condition = api.load_condition(*key)
        loaded[key] = sample
        n, k = sample["N"], sample["K"]
        assert n == row["N"] and k == row["K"]
        assert set(condition) == {"C", "N", "known_mask", "valid_mask"}
        assert condition["C"].tobytes() == sample["constraints"].tobytes()
        assert sample["constraints"].tobytes() == sample["full_target"][sample["source_face_ids"]].reshape(k,3,3).tobytes()
        assert sample["free_target"].tobytes() == sample["full_target"][sample["free_source_ids"]].tobytes()
        assert np.array_equal(sample["free_source_ids"], np.setdiff1d(np.arange(n),sample["source_face_ids"]))
        assert np.array_equal(condition["known_mask"], np.arange(n) < k) and condition["valid_mask"].all()
        checked_conditions.append({"uid":key[0],"patch_id":key[1],"N":n,"original_N":sample["original_N"],"K":k,"free_count":n-k,"Gamma_edge_count":len(sample["interface"]["edge_source_vertex_ids"]),"own_complement_exact":True,"C_target_bitwise_equal":True,"loader_full_validation":"PASS"})
    assert len(checked_conditions) == 24
    summaries = {}
    for name in split_names:
        rows = [objects[uid] for uid in splits[name]]
        summaries[name] = {"parents":len(rows),"unique_groups":len({splits["group_ids"][r["uid"]] for r in rows}),"N_min":min(r["N"] for r in rows),"N_max":max(r["N"] for r in rows),"N_median":float(np.median([r["N"] for r in rows])),"K_min":min(r["K"] for r in rows),"K_max":max(r["K"] for r in rows)}
    same_n = defaultdict(list)
    for uid in splits["test"]:
        same_n[objects[uid]["N"]].append(uid)
    eligible_n = {str(n):uids for n,uids in same_n.items() if len(uids)>1}
    note = REGISTRATION / "data_to_C.md"
    result = {
        "schema":"control_mvp48_eval_data_read_audit_v1","status":"PASS","owner":"Session C",
        "data_revision":"repair_v2","data_ready_sha256":ready_hash,"contract":ready["contract"],
        "files_checked":checked_files,"code_checked":checked_code,
        "data_to_C":{"exists":note.is_file(),"sha256":sha256(note) if note.is_file() else None},
        "scope":"Binding hashes, split/group identities, frozen plans and 24 evaluation loader conditions; not a new 3135-object grouping audit, model call, teacher OT, geometry score, or source repair.",
        "split_summary":summaries,"patch_counts":patch_counts,"checked_conditions":checked_conditions,
        "cross_split_parent_overlap":0,"cross_split_registered_group_overlap":0,
        "test_case_count":32,"formal_requests":96,"teacher_cases":8,"teacher_forwards":24,
        "same_N_distinct_test_groups":eligible_n,
        "control_status":"NOT_RUN" if not eligible_n else "REQUIRES_FROZEN_SAME_NOISE_GRID",
        "compatibility":{"source_face_ids":"Dense retained NPZ target row indices; unchanged API","original_source_face_ids":"Added raw source provenance; never model inputs","N":"Retained face count; do not pad away different N","target_scale":"Already official pre_OT * 2 float32; do not rescale","C_free":"Each patch uses its own full_target complement","old_DATA_BLOCKED":"Historical strict_v1 superseded by current READY"},
        "limitations":["Minimally cleaned official-derived data, not untouched meshes.","40/48 parents repaired; 109 exact zero-area faces and 571 duplicate copies removed; positive-area geometric union preserved by A protocol.","Full self-intersection and watertight certification not run by A.","Registered groups disjoint; rotation/reflection/semantic-family near-duplicate completeness not asserted.","Validation is conditional-finetuning unseen only; exact official checkpoint pretraining exposure not independently certified."],
        "resource_snapshot":resources,"numerical_threads":1,"cpu_workers":1,"GPU_workers":0,
        "OT_calls":0,"model_forwards":0,"formal_scores":0,"torch_imported":"torch" in sys.modules,
        "seconds":time.perf_counter()-started,
    }
    assert sha256(ready_path) == ready_hash
    assert result["torch_imported"] is False
    write_json(REGISTRATION / "eval_data_read_audit.json", result)
    print(json.dumps({k:result[k] for k in ("status","data_ready_sha256","split_summary","same_N_distinct_test_groups","control_status","data_to_C","seconds")},indent=2))


if __name__ == "__main__":
    main()
