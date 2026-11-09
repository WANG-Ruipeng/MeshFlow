"""CPU-only one-time evaluation inputs, queries and display-frame preparation.

This entrypoint creates exactly 20 Gaussian arrays and 10 C-query banks, never
calls a head/generator/denoiser and never computes output scores. Source Y is
used only to freeze a display frame, and is not placed in inference bundles.
"""
from __future__ import annotations

import collections
import hashlib
import sys
import time
import numpy as np
from . import eval_plan as E


def seed(cid, noise_index):
    namespace = E.VERSION + "|EVAL|" + cid + "|" + str(noise_index)
    return int.from_bytes(hashlib.sha256(namespace.encode("utf-8")).digest()[:4], "little")


def selection():
    pool = E.read(E.POOL)
    byid = {row["condition_id"]: row for row in pool["train"]}
    if len(byid) != 91 or len({row["parent_id"] for row in byid.values()}) != 26:
        raise ValueError("Expected retained 26-parent/91-C training pool")
    old = E.read(E.BASELINE_ROOT / "eval_manifest.json")
    dev = {q["condition_id"]: q for q in old["requests"] if q["noise_index"] == 0}
    three = {row["condition_id"]: row for row in E.read(E.THREE)["rows"]}
    follow = E.read(E.FOLLOWUP)
    if E.NEW_IDS[0] not in {r["condition_id"] for r in follow["rows"]}:
        raise ValueError("New source-supervision candidate not registered")
    chosen = []
    for cid in (*E.NEW_IDS, *E.TRAIN_IDS, *E.DEV_IDS):
        row = dict(three[cid] if cid in E.NEW_IDS else byid[cid] if cid in E.TRAIN_IDS else dev[cid])
        group = "SAME_PARENT_HELDOUT" if cid == E.NEW_IDS[1] else "DEV" if cid in E.DEV_IDS else "CURRENT_TRAIN"
        if cid in E.NEW_IDS:
            row.update(part_type="SEAT_BACK_SHELL" if cid == E.NEW_IDS[0] else "ARM_LEG_FRAME",
                       extent_description=row["proposed_region"], boundary_edges=int(row["C_open_edges"]),
                       official_split="UNKNOWN_NO_OFFICIAL_PRETRAIN_NOVELTY_CLAIM")
        row.update(eval_group=group, split=group, baseline_reuse=cid in E.DEV_IDS,
                   planned_current_training_C=group == "CURRENT_TRAIN",
                   source_manifest=str(E.THREE if cid in E.NEW_IDS else E.POOL if cid in E.TRAIN_IDS else E.BASELINE_ROOT / "eval_manifest.json"))
        chosen.append(row)
    plan = dict(schema=E.VERSION, status="FROZEN_BEFORE_ARRAY_CREATION",
        labels=list(E.LABELS), source_selection="INPUT_METADATA_ONLY_NO_GENERATED_OUTPUT_OR_METRIC_SELECTION",
        selection_rationale={
            "new_parent": "R1 exact trained C; R2 disjoint same-parent C held out from adaptation, not unseen-parent generalization",
            "old_train": "Four actual pool parents, two whole coherent C each: flat seat/back, seat/closed lower box, flat seat/curved U support, curved seat/inner back",
            "old_dev": "Four old-panel parents covering back, seat, closed support and seat-support assembly; part_type metadata overrides misleading old filenames",
            "limits": "Small fixed development panel; source GT is one completion, no unique-parent reconstruction success; no claimed official-pretraining novelty"},
        conditions=[{k: r[k] for k in ("condition_id", "parent_id", "N", "K", "boundary_edges", "part_type", "extent_description", "eval_group", "source_manifest", "baseline_reuse")} for r in chosen],
        new_request_seeds=[dict(condition_id=r["condition_id"], noise_index=j, seed=seed(r["condition_id"], j))
                           for r in chosen if not r["baseline_reuse"] for j in (0, 1)],
        requests_per_model=28, models=2, new_noise_arrays=20, new_C_query_banks=10,
        old_DEV_requests_reused=8, new_RAW=48, new_D0=48, generator_forwards=2400,
        head_forwards_upper_bound=dict(baseline_new_C=8, new_endpoint=11),
        creation_does_not_authorize_model_calls=True)
    return chosen, plan, old


def protect(files, path, expected=None):
    path = E.resolve(path)
    digest = E.sha(path)
    if expected is not None and digest != expected:
        raise ValueError("Source hash differs: " + str(path))
    key = str(path)
    if key in files and files[key] != digest:
        raise ValueError("Asset changed during preparation")
    files[key] = digest
    return digest


def new_bundle(row, protected):
    from meshflow_control.source_winding import capture_source, validate_source
    from r2.pilots.user_condition_finetune_v1 import evaluation as old
    from r2.tools.avmr_pmr_v1 import run as A
    from r2.tools.cref_layer_audit_v1 import render as H
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    cid = row["condition_id"]
    for key in ("input", "paired", "case"):
        protect(protected, row[key + "_path"], row[key + "_sha256"])
    inp, pair = E.arrays(row["input_path"]), E.arrays(row["paired_path"])
    core = {key: inp[key] for key in ("C_vertices", "C_faces_original", "C_faces_model", "N", "K")}
    meta = {key: pair[key] for key in ("C_source_vertex_ids", "known_ids")}
    original, canonical = core["C_faces_original"], core["C_faces_model"]
    cc, co = core["C_vertices"][canonical], core["C_vertices"][original]
    if cc.dtype != np.float32 or not np.array_equal(cc, pair["C"]) or cc.shape != (row["K"], 3, 3):
        raise ValueError("Input and supervised C differ")
    if (int(core["N"]), int(core["K"])) != (row["N"], row["K"]):
        raise ValueError("Selected input count differs")
    permutation = np.asarray([[np.flatnonzero(original[i] == vertex).item() for vertex in face]
                              for i, face in enumerate(canonical)], np.int64)
    source = capture_source(co, source_vertex_ids=meta["C_source_vertex_ids"][original],
                            corner_permutation=permutation, source_face_ids=meta["known_ids"])
    validate_source(source, cc)
    dest = E.OUT / "evaluation_inputs/conditions" / cid
    paths = dict(inference_npz=dest / "condition.npz", metadata_npz=dest / "metadata.npz", source_npz=dest / "source_condition.npz")
    for name, values in (("inference_npz", core), ("metadata_npz", meta), ("source_npz", source)):
        E.P.array_file(paths[name], values)
    _, _, topology = A.topology(co, meta["C_source_vertex_ids"][original])
    if int(topology["boundary_edge_count"]) != int(row["boundary_edges"]):
        raise ValueError("C source boundary count changed")
    bank, query_seed = old.query_bank(cc, cid)
    qp = E.OUT / "evaluation/queries" / (cid + ".npz")
    E.P.array_file(qp, bank)
    descriptor = dict(condition_id=cid, parent_id=row["parent_id"], path=str(qp), sha256=protect(protected, qp),
        array_sha256={k: old.array_sha(v) for k, v in bank.items()},
        identity=dict(condition_id=cid, C_seed=int(query_seed), C_sha256=old.array_sha(cc), L0=E.L),
        query_count=2048, weight_denominator=float(bank["C_weights"].sum()), source_boundary_audit=topology)
    # Source-derived framing is frozen before any generation. This is display
    # metadata only, not a unique-style scoring target or model input.
    g = dict(C=co, F=pair["Y"], C_original=co, tri=np.concatenate((co, pair["Y"])))
    bounds = H.bounds_for_condition([g], None, E.L)
    points = g["tri"].reshape(-1, 3)
    center = (points.min(0) + points.max(0)) / 2
    span = max(np.ptp((points - center) @ G.D.basis(az, 23), axis=0)[:2].max() for az in (-60, 120)) * 1.25
    depth = dict(center=center.tolist(), span=float(span))
    result = {**row, **{k: str(p) for k, p in paths.items()}, "query_descriptor": descriptor,
              "gallery_bounds": bounds, "gallery_depth_bounds": depth,
              "gallery_frame_source": "FROZEN_SOURCE_COMPLETION_DISPLAY_ONLY",
              "source_labels_available_to_models": False, "closed_C": row["boundary_edges"] == 0,
              "part_family": row["part_type"] if row["part_type"] in E.P.FAMILIES else "COMBINATION"}
    for key, path in paths.items():
        result[key + "_sha256"] = protect(protected, path)
    result.update(metadata_sha256=result["metadata_npz_sha256"], source_sha256=result["source_npz_sha256"])
    return result


def reused_baseline(q, protected):
    rid = q["request_id"]
    base = E.BASELINE_ROOT / "evaluation" / E.BASELINE_LABEL / rid
    cpu = E.BASELINE_ROOT / "cpu" / E.BASELINE_LABEL / rid
    for kind in ("raw", "post"):
        receipt_path = base / kind / "receipt.json"
        receipt = E.read(receipt_path)
        if receipt["label"] != E.BASELINE_LABEL or receipt["request_id"] != rid or receipt["status"] != "COMPLETE":
            raise ValueError("Reused baseline producer identity differs")
        protect(protected, receipt_path)
        for p, h in receipt["files"].items():
            protect(protected, p, h)
    complete = E.read(cpu / "complete.json")
    if complete["status"] != "COMPLETE" or complete["request_id"] != rid or complete["label"] != E.BASELINE_LABEL:
        raise ValueError("Reused CPU baseline identity differs")
    protect(protected, cpu / "complete.json")
    for p, h in complete["files"].items():
        protect(protected, p, h)
    score_dirs = dict(RAW=E.BASELINE_ROOT / "evaluation/scores" / E.BASELINE_LABEL / rid / "RAW",
                      BASE_POST=cpu / "BASE_POST/scores", CLEAN_FINAL_POST=cpu / "CLEAN_FINAL_POST/scores")
    for p in score_dirs["RAW"].iterdir():
        if p.is_file():
            protect(protected, p)
    return dict(root=str(E.BASELINE_ROOT), label=E.BASELINE_LABEL,
        raw_npz=str(base / "raw/raw.npz"), post_npz=str(base / "post/post.npz"),
        final_npz=str(cpu / "CLEAN_FINAL_POST/post.npz"),
        score_dirs={k: str(p) for k, p in score_dirs.items()},
        raw_receipt=str(base / "raw/receipt.json"), post_receipt=str(base / "post/receipt.json"),
        cpu_receipt=str(cpu / "complete.json"), new_model_calls=0, new_scores=0)


def prepare():
    done = E.OUT / "eval_assets_complete.json"
    if done.exists():
        receipt = E.read(done)
        E.verify(receipt["files"])
        E.validate_assets()
        return receipt
    if (E.OUT / "eval_selection.json").exists():
        raise RuntimeError("Partial preparation retained; inspect before any explicit engineering resume")
    from r2.pilots.seam_abc_v1 import eval_common as C
    started = time.perf_counter()
    E.P.immutable(E.OUT / "eval_resources_before.json", C.resources())
    chosen, plan, old = selection()
    E.P.immutable(E.OUT / "eval_selection.json", plan)
    protected = {}
    for path in (E.POOL, E.THREE, E.FOLLOWUP, E.BASELINE_ROOT / "eval_manifest.json", E.OLD_GALLERY,
                 E.OUT / "eval_selection.json", __file__, E.__file__):
        protect(protected, path)
    from r2.pilots.user_condition_finetune_v1 import evaluation as original
    from r2.tools.avmr_pmr_v1 import run as boundary
    from r2.tools.cref_layer_audit_v1 import render as section
    for path in (original.__file__, boundary.__file__, section.__file__):
        protect(protected, path)
    for path, digest in original.scorer_sources().items():
        protect(protected, path, digest)
    historic = E.read(E.OLD_GALLERY)
    lookup = {(q["condition_id"], q["noise_index"]): q for q in old["requests"]}
    rows, conditions = [], []
    for row in chosen:
        cid = row["condition_id"]
        if not row["baseline_reuse"]:
            base = new_bundle(row, protected)
            conditions.append({k: base[k] for k in ("condition_id", "parent_id", "N", "K", "boundary_edges", "closed_C", "part_type", "eval_group", "gallery_bounds", "gallery_depth_bounds")})
        for ni in (0, 1):
            if row["baseline_reuse"]:
                q = dict(lookup[cid, ni])
                q.update(eval_group="DEV", baseline_reuse=True, planned_current_training_C=False,
                         gallery_bounds=historic["bounds"][cid], gallery_depth_bounds=historic["depth_bounds"][cid],
                         gallery_frame_source="EXACT_INHERITED_OLD48_DISPLAY_FRAME")
                for key in ("inference_npz", "metadata_npz", "source_npz", "noise_npz"):
                    protect(protected, q[key], q[key + "_sha256"])
                protect(protected, q["query_descriptor"]["path"], q["query_descriptor"]["sha256"])
                protect(protected, q["old_layout_npz"], q["old_layout_sha256"])
                protect(protected, q["old_layout_receipt"])
                q["baseline_cached"] = reused_baseline(q, protected)
                if ni == 0:
                    conditions.append({k: q[k] for k in ("condition_id", "parent_id", "N", "K", "boundary_edges", "closed_C", "part_type", "eval_group", "gallery_bounds", "gallery_depth_bounds")})
            else:
                rid = cid + "__n" + str(ni)
                frozen_seed = next(x["seed"] for x in plan["new_request_seeds"] if x["condition_id"] == cid and x["noise_index"] == ni)
                noise = np.random.RandomState(frozen_seed).standard_normal((row["N"] - row["K"], 3, 3)).astype(np.float32)
                path = E.OUT / "evaluation_inputs/requests" / rid / "noise.npz"
                E.P.array_file(path, dict(z=noise, seed=np.asarray(frozen_seed, np.int64)))
                digest = protect(protected, path)
                q = dict(base, request_id=rid, noise_index=ni, seed=frozen_seed, noise_npz=str(path),
                         noise_sha256=digest, noise_npz_sha256=digest)
            rows.append(q)
    if len(rows) != 28 or len(conditions) != 14:
        raise ValueError("Panel cardinality changed")
    manifest = dict(schema=E.VERSION, status="FROZEN", requests=rows, conditions=conditions,
        labels=list(E.LABELS), stages=["RAW", "BASE_POST", "CLEAN_FINAL_POST"],
        source_selection=plan["source_selection"], selection_rationale=plan["selection_rationale"],
        protected=protected, requests_per_model=28, new_noise_arrays=20, new_query_arrays=10,
        reused_noise_arrays=8, reused_query_arrays=4, new_RAW=48, new_D0=48,
        generator_forwards=2400, new_scoring_rows=144, reused_scoring_rows=24,
        C_query_count=2048, L=E.L,
        group_counts=dict(collections.Counter(q["eval_group"] for q in rows)),
        inference_policy="C/N-only per-model learned head; no Y/source owner/alias correction at deployment",
        all_RAW_before_D0=True, no_unique_GT_style_success=True,
        metric_contract="Unchanged original scoring chain; S/E NA without registered requirement; closed C seam fields NA; open predicted all-null graph stays zero, not NA",
        gallery_policy="Old DEV frames exact inherited; new frames from source before generation; azimuth -60/120 elevation23, central front/side section index1; frozen across models/stages",
        source_GT_use="Display framing and supervision provenance only, never unique completion success",
        model_calls=0, head_calls=0, scoring_calls=0)
    E.P.immutable(E.OUT / "eval_manifest.json", manifest)
    checks = E.validate_assets()
    E.P.immutable(E.OUT / "eval_cpu_checks.json", checks)
    if "torch" in sys.modules:
        raise RuntimeError("CPU evaluation preparation imported neural framework")
    E.verify(protected)
    output_files = {str(p): E.sha(p) for p in (E.OUT / "eval_selection.json", E.OUT / "eval_manifest.json", E.OUT / "eval_cpu_checks.json", E.OUT / "eval_resources_before.json")}
    receipt = dict(status="COMPLETE_CPU_PREPARATION", schema=E.VERSION, conditions=14, requests_per_model=28,
        new_noise_arrays=20, new_C_query_banks=10, reused_DEV_requests=8,
        protected_files=len(protected), source_asset_hashes_verified=True, actual_model_calls=0,
        actual_scoring_calls=0, actual_GPU_calls=0, seconds=time.perf_counter()-started, files=output_files)
    E.P.immutable(done, receipt)
    return receipt
