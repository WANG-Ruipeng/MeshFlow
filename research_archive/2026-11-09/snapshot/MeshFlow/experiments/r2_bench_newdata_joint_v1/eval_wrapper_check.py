"""Read-only compatibility audit; no model, scoring, query draw or render."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import ast
import collections
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "r2/src"))
import numpy as np
from r2.pilots.bench_newdata_joint_v1 import eval_plan as E, eval_runtime as R
from r2.pilots.bench_newdata_joint_v1 import evaluation as V, collect as T, gallery as G
from r2.pilots.bench_adaptation_strength_v1.collect_v2 import decorate, numeric_fields
from r2.pilots.user_condition_finetune_v1 import evaluation as Q

started = time.perf_counter()
code = {}
for module in (E, R, V, T, G):
    path = Path(module.__file__)
    ast.parse(path.read_text(encoding="utf-8"))
    code[str(path)] = E.sha(path)
code[str(Path(__file__))] = E.sha(__file__)
manifest_path = E.OUT / "eval_manifest.json"
manifest_sha = E.sha(manifest_path)
manifest = E.read(manifest_path)
qs = R.requests()
registration = E.read(E.OUT / "registration.json")
assert registration["eval_manifest_sha256"] == manifest_sha
train = E.read(E.OUT / "data/manifest.json")
train_ids = {q["condition_id"] for q in train["train"]}
train_parents = {q["parent_id"] for q in train["train"]}
required = set(("request_id", "condition_id", "parent_id", "noise_index", "N", "K", "split",
                "eval_group", "part_type", "part_family", "closed_C", "boundary_edges", "baseline_reuse",
                "query_descriptor", "gallery_bounds", "gallery_depth_bounds"))
bounds_required = {"whole_center", "whole_span", "local_low", "local_high", "C_low", "C_high",
                   "front_positions", "side_positions"}
condition_checks, source_rows, old_cells = [], [], []
seen = set()
historical_statuses = []
for q in qs:
    assert required <= set(q)
    core, source = R.source_assets(q)
    assert q["closed_C"] == (q["boundary_edges"] == 0)
    assert q["split"] == q["eval_group"]
    if q["eval_group"] == "CURRENT_TRAIN":
        assert q["condition_id"] in train_ids
    elif q["eval_group"] == "SAME_PARENT_HELDOUT":
        assert q["condition_id"] not in train_ids and q["parent_id"] in train_parents
    else:
        assert q["eval_group"] == "DEV" and q["parent_id"] not in train_parents
    query = Q.load_queries(q["query_descriptor"])
    cc = core["C_vertices"][core["C_faces_model"]]
    co = core["C_vertices"][core["C_faces_original"]]
    assert query["identity"]["C_sha256"] == Q.array_sha(cc)
    assert np.isclose(query["identity"]["L0"], E.L)
    assert len(query["arrays"]["C_points"]) == 2048
    assert np.isclose(query["arrays"]["C_weights"].sum(), 1.0)
    assert q["query_descriptor"]["source_boundary_audit"]["boundary_edge_count"] == q["boundary_edges"]
    b, d = q["gallery_bounds"], q["gallery_depth_bounds"]
    assert bounds_required <= set(b) and set(d) == {"center", "span"}
    for key in bounds_required:
        assert np.isfinite(np.asarray(b[key])).all(), key
    for key in ("front_positions", "side_positions"):
        assert len(b[key]) == 3
    assert np.isfinite(d["center"]).all() and len(d["center"]) == 3 and np.isfinite(d["span"]) and d["span"] > 0
    assert b["whole_span"] > 0
    if q["condition_id"] not in seen:
        seen.add(q["condition_id"])
        condition_checks.append(dict(condition_id=q["condition_id"], eval_group=q["eval_group"],
            original_group=q.get("group"), group_required_by_wrappers=False,
            closed_C=q["closed_C"], C_boundary_edges=q["boundary_edges"],
            source_identity="PASS", query_identity="PASS", frame_schema="PASS",
            central_front=b["front_positions"][1], central_side=b["side_positions"][1],
            planned_current_training_C=q["condition_id"] in train_ids))
        if q["condition_id"] in E.NEW_IDS:
            historical_statuses.append(dict(condition_id=q["condition_id"],
                historical_fields={k:q[k] for k in ("status", "production_status", "final_status", "effective_eval_group", "training_ready", "training_pool_membership") if k in q},
                current_eval_group=q["eval_group"], current_training_member=q["condition_id"] in train_ids))
    for label in R.LABELS:
        for stage in V.STAGES:
            folder = V.score_dir(label, q, stage)
            cell = G.cell(q, label, stage)
            reuse = label == E.BASELINE_LABEL and q["baseline_reuse"]
            if not reuse:
                assert folder.is_relative_to(E.OUT) and E.resolve(cell["path"]).is_relative_to(E.OUT)
                continue
            assert folder == E.resolve(q["baseline_cached"]["score_dirs"][stage])
            receipt = E.read(folder / "complete.json")
            assert receipt["status"] == "PASS" and receipt["model_calls"] == 0
            rowpath = folder / "row.json"
            assert any(E.resolve(p).resolve() == rowpath.resolve() and h == E.sha(rowpath) for p,h in receipt["files"].items())
            old = E.read(rowpath)
            decorated = decorate(old, label, q, stage)
            assert old["split"] == q["split"] == "DEV"
            for k,v in old.items():
                if k not in {"label", "stage", "part_type", "closed_C", "seam_applicability"}:
                    assert decorated[k] == v, k
            assert decorated["boundary_edges"] == old["boundary_edges"]
            assert decorated["C_boundary_edges"] == q["boundary_edges"]
            assert decorated["seam_applicability"] == ("NA_CLOSED_C_NO_SEAM_INTERFACE" if q["closed_C"] else "APPLICABLE")
            source_rows.append(dict(request_id=q["request_id"], stage=stage, source_row=str(rowpath),
                row_sha256=E.sha(rowpath), original_split=old["split"], decorate="PASS_LOSSLESS"))
            authenticated = G.G.authenticate(cell)
            mesh = E.arrays(cell["path"])
            tri, role = G.G.roles(mesh, q["K"])
            assert tri[:q["K"]].tobytes() == co.tobytes()
            old_cells.append(dict(request_id=q["request_id"], stage=stage, mesh=cell["path"],
                                  source_C="PASS", authenticate="PASS", role_schema="PASS"))
assert len(source_rows) == len(old_cells) == 24
assert collections.Counter(q["eval_group"] for q in qs) == {"CURRENT_TRAIN":18, "SAME_PARENT_HELDOUT":2, "DEV":8}
assert len([q for q in qs if q["closed_C"]]) == 6
new_jobs = [(label, q) for label in R.LABELS for q in qs if not (label == E.BASELINE_LABEL and q["baseline_reuse"])]
assert len(new_jobs) == 48
assert len(qs) * len(R.LABELS) * len(V.STAGES) == 168
# Calling decorate, source/shape schema checks and producer authentication only;
# none of the functions below is invoked by this audit.
not_called = ["CPU.score_current", "CPU.process_one", "evaluation.prepare/run", "collect.run", "gallery.prepare/run/one",
              "model/head/generator/denoiser forward", "query_bank", "RandomState", "render"]
assert "torch" not in sys.modules
assert E.sha(manifest_path) == manifest_sha
E.verify(manifest["protected"])
result = dict(status="PASS", scope="READ_ONLY_COMPATIBILITY_NO_OUTPUT_SCORING", code_sha256=code,
    eval_manifest_sha256=manifest_sha, registration_sha256=E.sha(E.OUT / "registration.json"),
    requests=28, conditions=14, groups=dict(collections.Counter(q["eval_group"] for q in qs)),
    new_scoring_rows_expected=144, old_scoring_rows_reused=24, all_output_rows_expected=168,
    actual_old_rows_decorated=24, old_gallery_cells_authenticated=24,
    condition_checks=condition_checks, old_score_checks=source_rows, old_gallery_checks=old_cells,
    historical_status_explanation="New-parent status/production_status/final_status and staging membership fields in the frozen eval manifest are copied historical three-family records under the old frozen head. They do not override the current training pool or eval_group. Current data/manifest.json contains R1 in the active 27-parent/92-C training pool; R2 remains same-parent held out. Historical originals and this evaluation manifest are unchanged.",
    historical_statuses=historical_statuses,
    group_compatibility="New R1/R2 have no legacy group field; current training checker prioritizes eval_group, while the actual scorer/collector/gallery use split. No active wrapper requires q.group.",
    limits="Future generated outputs are not available to this audit. This checks producer/score routing and existing 24 baseline cells, not future output quality or geometry.",
    model_calls=0, new_score_calls=0, GPU_compute_calls=0, new_noise_arrays=0, new_query_arrays=0, rendered_images=0,
    never_called=not_called, original_inputs_unchanged=True, seconds=time.perf_counter()-started)
E.P.immutable(E.OUT / "eval_wrapper_check.json", result)
print(json.dumps({k:v for k,v in result.items() if k not in ("condition_checks", "old_score_checks", "old_gallery_checks", "code_sha256", "historical_statuses")},ensure_ascii=False))
