"""Evaluate every frozen endpoint-training arm through unchanged CPU deployment."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from ..exporter import RadialSupport
from ..legacy.networks import DomainVelocity
from ..io_utils import array_hash, file_hash, save_npz, write_json
from ..layout_student.deployment import deploy, named_quality, qualification
from ..layout_student.integrity import mesh_identity, STORED_HASH_SCHEMA
from ..layout_student.train import model_sha256
from ..layout_student.evaluate import metric_cache
from ..layout_value.geometry import audit_mesh

MODEL_NAMES = ("fixed", "teacher_reference", "v1_10", "v1_11",
               "c0_10", "c0_11", "c1_10", "c1_11", "c2_10", "c2_11")
POLICIES = {name: (("offline_teacher",) if name == "teacher_reference"
                  else ("raw",) if name == "fixed"
                  else ("raw", "quality_fallback")) for name in MODEL_NAMES}
SEED_BASES = (392401, 492401)
POINTS = 8192


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def expected_slots(uids):
    return {(uid, model, policy) for uid in uids for model in MODEL_NAMES
            for policy in POLICIES[model]}


def aggregate(run):
    run = Path(run)
    protocol = read_json(run / "RUN_PROTOCOL.json")
    expected_attempts = {(uid, model) for uid in protocol["uids"] for model in MODEL_NAMES}
    attempts = [read_json(path) for path in sorted((run / "attempts").glob("*/*/result.json"))]
    ids = [(a["uid"], a["model"]) for a in attempts]
    require(len(ids) == len(set(ids)) and set(ids) <= expected_attempts, "Duplicate or unregistered attempt")
    rows = []
    errors = []
    for attempt in attempts:
        if attempt["status"] == "success":
            selected = attempt["rows"]
            require({r["policy"] for r in selected} == set(POLICIES[attempt["model"]]),
                    "Successful attempt lacks a predeclared policy")
            require(len(selected) == len(POLICIES[attempt["model"]]), "Duplicate policy in attempt")
            require(all(r["uid"] == attempt["uid"] and r["model"] == attempt["model"]
                        and r["status"] == "success" for r in selected), "Attempt row identity mismatch")
            rows.extend(selected)
        else:
            require(attempt.get("rows", []) == [], "Failed attempt contains successful result rows")
            errors.append(attempt)
    keys = [(r["uid"], r["model"], r["policy"]) for r in rows]
    require(len(keys) == len(set(keys)), "Duplicate result row")
    expected = expected_slots(protocol["uids"])
    failed_slots = [(a["uid"], a["model"], policy) for a in errors for policy in POLICIES[a["model"]]]
    value = {
        "rows": rows, "attempt_count": len(attempts), "errors": errors,
        "expected_attempt_count": len(expected_attempts), "expected_row_count": len(expected),
        "failed_row_slots": failed_slots,
        "unattempted": sorted(expected_attempts - set(ids)),
        "missing_success_rows": sorted(expected - set(keys)),
    }
    write_json(run / "results.json", value, replace=(run / "results.json").exists())
    return value


def load_network(run, name, protocol):
    run = Path(run)
    require(name in MODEL_NAMES, "Unregistered model")
    if name in ("fixed", "teacher_reference"):
        return None, {"kind": name, "checkpoint": None, "deployment_weight": False}
    arm, seed_text = name.rsplit("_", 1)
    seed = int(seed_text)
    if arm == "v1":
        path = run / protocol["paths"]["initial_checkpoints"][str(seed)]
        expected_hash = protocol["hashes"]["initial_checkpoint_sha256"][str(seed)]
        require(file_hash(path) == expected_hash, "Frozen v1 initialization file changed")
    else:
        path = run / "training" / arm / f"seed_{seed}" / "final.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(checkpoint["is_predeclared_final"] is True, "Only predeclared final checkpoints are deployable")
    require(checkpoint["config"]["seed"] == seed, "Checkpoint seed differs from requested model")
    if arm == "v1":
        require(checkpoint["update"] == 3000, "Unexpected v1 initialization budget")
    else:
        require(checkpoint["protocol_sha256"] == file_hash(run / "RUN_PROTOCOL.json"),
                "Training and evaluation protocols differ")
        require(checkpoint["update"] == protocol["training"]["updates"] == 1000,
                "Checkpoint is not the fixed final update")
        require(checkpoint["config"]["arm"] == arm, "Checkpoint arm differs from requested model")
    observed_hash = model_sha256(checkpoint["model"])
    require(observed_hash == checkpoint["model_sha256"], "Model tensor hash mismatch")
    model = DomainVelocity().cpu()
    model.load_state_dict(checkpoint["model"], strict=True)
    require(sum(p.numel() for p in model.parameters()) == 29907, "Original architecture changed")
    model.eval().requires_grad_(False)
    return model, {
        "kind": name, "checkpoint": str(path), "file_sha256": file_hash(path),
        "model_sha256": observed_hash, "training_seed": seed,
        "updates": int(checkpoint["update"]), "parameter_count": 29907,
        "architecture": "unchanged legacy.networks.DomainVelocity",
        "weight_selection": "fixed final only; no metric-based selection",
        "initialization_baseline": arm == "v1",
    }


def save_mesh_with_identity(path, mesh, template_faces):
    save_npz(path, **mesh)
    identity = mesh_identity(path, template_faces)
    identity["stored_array_hash_schema"] = STORED_HASH_SCHEMA
    with np.load(path, allow_pickle=False) as stored:
        for key in mesh:
            require(np.array_equal(mesh[key], stored[key]), "Saved NPZ differs from produced mesh")
    return identity


def evaluate_run(run, models=None, selected_uids=None):
    run = Path(run).resolve()
    protocol = read_json(run / "RUN_PROTOCOL.json")
    uids = sorted(protocol["uids"])
    require(len(uids) == len(set(uids)) == 16, "Expected exactly 16 registered parents")
    selected = uids if selected_uids is None else list(selected_uids)
    models = list(MODEL_NAMES if models is None else models)
    require(len(selected) == len(set(selected)) and set(selected) <= set(uids), "Unregistered/duplicate selected UID")
    require(len(models) == len(set(models)) and set(models) <= set(MODEL_NAMES), "Unregistered/duplicate model")
    manifest_path = run / "DATA_MANIFEST.json"
    require(file_hash(manifest_path) == protocol["hashes"]["data_manifest_sha256"], "Data manifest changed")
    manifest = read_json(manifest_path)["files"]
    evaluation = protocol["evaluation"]
    require(tuple(evaluation["models"]) == MODEL_NAMES, "Evaluation model contract changed")
    require((evaluation["seed_base"], evaluation["second_seed_base"]) == SEED_BASES,
            "Evaluation seeds differ from frozen protocol")
    require(evaluation["points_each_direction"] == POINTS and evaluation["hot_repeats"] == 3,
            "Evaluation sampling/timing contract changed")
    for relative, expected_hash in protocol["fixed_source_hashes"].items():
        require(file_hash(Path(__file__).parents[1] / relative) == expected_hash,
                "Frozen deployment source changed: " + relative)
    support = RadialSupport()
    module_root = Path(__file__).parents[1]
    sources = {str(path.relative_to(module_root)): file_hash(path) for path in module_root.rglob("*.py")}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    write_json(run / "eval_invocations" / f"{stamp}.json", {
        "models": models, "uids": selected, "source_hashes": sources,
        "protocol_sha256": file_hash(run / "RUN_PROTOCOL.json"),
        "evaluation_seed_bases": list(SEED_BASES), "points_each_direction": POINTS,
        "deployment": "unchanged layout_student.deployment.deploy CPU32",
    })
    for name in models:
        started = time.perf_counter()
        load_error = None
        identity = {"kind": name}
        try:
            flow, identity = load_network(run, name, protocol)
        except Exception as exc:
            flow = None
            load_error = {"error": repr(exc), "traceback": traceback.format_exc(), "stage": "checkpoint_load"}
        load_seconds = time.perf_counter() - started
        for uid in selected:
            dest = run / "attempts" / uid / name
            if (dest / "result.json").exists():
                previous = read_json(dest / "result.json")
                require(previous["uid"] == uid and previous["model"] == name, "Existing attempt identity mismatch")
                emit({"event": "skip_recorded_attempt", "uid": uid, "model": name, "status": previous["status"]})
                continue
            if dest.exists():
                write_json(dest / "result.json", {"uid": uid, "model": name, "status": "error",
                    "error": "Interrupted attempt directory retained; no automatic retry", "rows": [],
                    "stage": "previous_interruption"})
                aggregate(run)
                continue
            dest.mkdir(parents=True)
            write_json(dest / "START.json", {"uid": uid, "model": name, "identity": identity})
            emit({"event": "start", "uid": uid, "model": name})
            try:
                if load_error is not None:
                    write_json(dest / "result.json", {"uid": uid, "model": name, "status": "error",
                               "rows": [], **load_error})
                    aggregate(run)
                    emit({"event": "error", "uid": uid, "model": name, **load_error})
                    continue
                relative_data = f"data/{uid}.npz"
                data_path = run / relative_data
                require(file_hash(data_path) == manifest[relative_data], "Prepared object changed")
                with np.load(data_path, allow_pickle=False) as z:
                    y, base, faces = z["logr"], z["base_u"], z["faces"]
                    reference_v, reference_f = z["ref_vertices"], z["ref_faces"]
                    proxy = z["proxy_points"]
                    teacher_u = z["teacher_u"] if name == "teacher_reference" else None
                split = "train" if uid in protocol["train_uids"] else "internal_check"
                seeds = [seed + uids.index(uid) for seed in SEED_BASES]
                if name == "teacher_reference":
                    fixed = deploy(None, y, support, base, faces, fixed_only=True)
                    radii = support.query(y, teacher_u)
                    mesh = {"vertices": teacher_u * radii[:, None], "faces": faces, "u": teacher_u, "r": radii}
                    quality = named_quality(mesh["vertices"], faces)
                    audit = audit_mesh(mesh["vertices"], faces, exact=True)
                    result = {"raw_mesh": mesh, "raw_quality": quality, "raw_audit": audit,
                              "floor": fixed["floor"],
                              "timing": {"offline_teacher_reference": True, "generation_seconds": None}}
                    hot = []
                else:
                    result = deploy(flow, y, support, base, faces, fixed_only=name == "fixed")
                    hot = []
                    for _ in range(3):
                        repeat = deploy(flow, y, support, base, faces, fixed_only=name == "fixed")
                        for prefix in ("raw", "delivered"):
                            for key in ("vertices", "faces", "u", "r"):
                                require(np.array_equal(repeat[prefix + "_mesh"][key],
                                                       result[prefix + "_mesh"][key]), "Repeated deployment changed")
                        require(repeat["delivery"] == result["delivery"], "Repeated fallback decision changed")
                        hot.append(repeat["timing"])
                    if name != "fixed":
                        require(result["timing"]["proxy_points_hash"] == array_hash(proxy),
                                "Deployment proxy differs from original input contract")
                result_rows = []
                for policy in POLICIES[name]:
                    prefix = "delivered" if policy == "quality_fallback" else "raw"
                    mesh = result[prefix + "_mesh"]
                    quality, audit = result[prefix + "_quality"], result[prefix + "_audit"]
                    path = dest / f"{policy}.npz"
                    hashes = save_mesh_with_identity(path, mesh, faces)
                    metrics = metric_cache(run, uid, mesh["vertices"], faces, reference_v, reference_f, seeds, POINTS)
                    require(all(item["metrics"]["seed"] == seed and
                                item["metrics"]["points_per_direction"] == POINTS
                                for item, seed in zip(metrics, seeds)), "Cached metric identity mismatch")
                    timing = result["timing"]
                    if policy == "offline_teacher":
                        seconds = hot_seconds = None
                    else:
                        key = ("quality_fallback_total_seconds" if policy == "quality_fallback"
                               else "raw_with_audit_seconds")
                        seconds = timing[key]
                        hot_seconds = float(np.median([record[key] for record in hot]))
                    result_rows.append({
                        "uid": uid, "split": split, "model": name, "policy": policy, "status": "success",
                        "offline_teacher": policy == "offline_teacher",
                        "mesh_file": str(path.relative_to(run)), **hashes,
                        "mainfield_hash": array_hash(y), "data_file_sha256": file_hash(data_path),
                        "identity": identity, "quality": quality, "audit": audit, "quality_floor": result["floor"],
                        **qualification(quality, audit, result["floor"]),
                        "metrics": [item["metrics"] for item in metrics],
                        "metric_cache_evaluation_seconds": [item["evaluation_seconds"] for item in metrics],
                        "fallback_to_fixed": bool(policy == "quality_fallback" and result["delivery"]["fallback_to_fixed"]),
                        "fallback_reasons": result.get("delivery", {}).get("fallback_reasons", [])
                            if policy == "quality_fallback" else [],
                        "generation_seconds": timing["generation_seconds"],
                        "delivery_with_audit_seconds": seconds,
                        "hot_delivery_with_audit_seconds_median_3": hot_seconds,
                        "model_load_seconds": load_seconds, "timing": timing, "hot_timing_repeats": hot,
                        "cost_scope": "raw: export+self audit; fallback policy includes Fixed floor setup+audit+raw+audit+selection; no AE/FM or RMS cost",
                        "independent_evaluation_not_used_for_deployment": True,
                    })
                write_json(dest / "result.json", {"uid": uid, "model": name, "status": "success", "rows": result_rows})
                aggregate(run)
                emit({"event": "complete", "uid": uid, "model": name,
                      "raw_qualified": result_rows[0]["qualified"],
                      "raw_rms": result_rows[0]["metrics"][0]["rms_bbox_percent"]})
            except Exception as exc:
                error = {"uid": uid, "model": name, "status": "error", "error": repr(exc),
                         "traceback": traceback.format_exc(), "rows": []}
                write_json(dest / "result.json", error)
                aggregate(run)
                emit(error)
    return aggregate(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--uids", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    evaluate_run(args.run, args.models, args.uids)


if __name__ == "__main__":
    main()


