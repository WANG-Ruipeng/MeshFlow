"""Long-budget fixed-checkpoint, training-parent-only diagnosis; never selection.

Teacher-conditioned velocity probes are OFFLINE and separate from original CPU
free deployment. No internal-check object file is opened by this module.
"""
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
from ..io_utils import array_hash, file_hash, save_npz, write_json
from ..legacy.networks import DomainVelocity, point_features
from ..layout_student.deployment import deploy
from ..layout_student.integrity import mesh_identity, STORED_HASH_SCHEMA, safe_path
from ..layout_student.train import model_sha256
from ..layout_fitdiag.diagnose import velocity_probe, endpoint_probe

JOBS = ("joint_10", "joint_11")
UPDATES = (20000, 50000, 100000, 240000)
CHECKPOINTS = ("initial",) + tuple(map(str, UPDATES))
SCOPE = "LONG_BUDGET_TRAIN_ONLY_FIXED_CHECKPOINT_DIAGNOSIS_NOT_MODEL_SELECTION"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def describe(values):
    values = np.asarray([v for v in values if v is not None], np.float64)
    return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None}


def expected_slots(protocol):
    """Every observation has a fixed denominator, including duplicate initial/20000."""
    train = set(protocol["train_uids"])
    check = set(protocol["internal_check_uids"])
    require(not train & check, "Train/internal-check overlap")
    require(set(protocol["jobs"]) == set(JOBS), "Unexpected predeclared jobs")
    require(tuple(protocol["training"]["checkpoint_updates"]) == UPDATES,
            "Fixed observation updates changed")
    slots = set()
    for name, job in protocol["jobs"].items():
        uids = job["diagnostic_uids"]
        require(uids and len(uids) == len(set(uids)), "Empty or duplicate diagnostic UIDs")
        require(set(uids) <= train and not set(uids) & check,
                "Internal-check/unregistered parent forbidden in intermediate diagnosis")
        require(uids == job["active_training_uids"], "Diagnostic UIDs differ from active training set")
        slots.update((name, ckpt, uid) for ckpt in CHECKPOINTS for uid in uids)
    return slots


def checkpoint_path(run, job, checkpoint):
    require(job in JOBS and checkpoint in CHECKPOINTS, "Unregistered job/checkpoint")
    name = ("initial.pt" if checkpoint == "initial" else "final.pt" if checkpoint == "240000"
            else f"step_{int(checkpoint):05d}.pt")
    return safe_path(run, f"training/{job}/{name}")


def load_network(run, job_name, checkpoint, protocol):
    run = Path(run)
    job = protocol["jobs"][job_name]
    path = checkpoint_path(run, job_name, checkpoint)
    file_sha = file_hash(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model_sha = model_sha256(state["model"])
    require(model_sha == state["model_sha256"], "Checkpoint model tensor hash mismatch")
    imported = checkpoint == "initial" or (checkpoint == "20000" and job["start_update"] == 20000)
    if imported:
        require(file_sha == job["initial_checkpoint_sha256"], "Imported snapshot file changed")
        require(model_sha == job["initial_model_sha256"], "Imported model differs from fixed initializer")
        require(state["config"]["seed"] == job["seed"], "Imported checkpoint seed differs")
        require(state["update"] == 20000,
                "Imported checkpoint source update differs")
    else:
        require(state["protocol_sha256"] == file_hash(run / "RUN_PROTOCOL.json"),
                "Checkpoint has another run protocol")
        require(state["update"] == int(checkpoint), "Checkpoint observation update differs")
        cfg = state["config"]
        require(cfg["job"] == job_name and cfg["kind"] == job["kind"]
                and cfg["seed"] == job["seed"] and cfg["arm"] == "c0",
                "Checkpoint job/kind/seed/arm differs")
        require(cfg["active_training_uids"] == job["active_training_uids"],
                "Checkpoint active training parents differ")
        require(state["is_predeclared_final"] is (checkpoint == "240000"),
                "Final checkpoint marker differs from fixed update")
    require(file_hash(path) == file_sha, "Checkpoint changed while reading")
    model = DomainVelocity().cpu().eval()
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False)
    require(sum(p.numel() for p in model.parameters()) == 29907, "Architecture changed")
    cfg = state["config"]
    exposure_uids = cfg.get("active_training_uids", cfg.get("train_uids", []))
    exposures = state.get("sampler", {}).get("exposures", [])
    parent_exposures = dict(zip(exposure_uids, exposures)) if len(exposure_uids) == len(exposures) else None
    import_record = path.parent / "IMPORT_RECORD.json"
    identity = {
        "checkpoint_file": str(path.relative_to(run)), "checkpoint_file_sha256": file_sha,
        "model_sha256": model_sha, "source_checkpoint_update": state["update"],
        "job_update": job["start_update"] if checkpoint == "initial" else int(checkpoint),
        "imported_snapshot": imported, "parameter_count": 29907,
        "source_protocol_sha256": state["protocol_sha256"],
        "checkpoint_parent_exposures_including_source_lineage": parent_exposures,
        "elapsed_training_seconds_including_source_lineage": state.get("elapsed_training_seconds"),
        "import_record_sha256": file_hash(import_record) if import_record.exists() else None,
        "per_parent_exposure_match_reference": "single20k (seed10 only) at 80000 presentations; global240k final only",
        "equal_optimizer_steps_or_compute_claim": False,
        "weight_selection": "every predeclared observation; no ranking or best checkpoint",
        "deployment_device": "cpu", "network_dtype": "float32",
    }
    return model, identity


def load_train_archive(run, protocol):
    """Read one explicitly train-only archive, never data/<check UID>.npz."""
    path = safe_path(run, protocol["paths"]["training_labels"])
    expected = protocol["hashes"]["training_labels_sha256"]
    require(file_hash(path) == expected, "Training label archive changed")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.array(archive[key]) for key in (
            "train_uids", "directions0", "directions1", "proxy_points", "point_features",
            "logr", "faces", "quality_floors_deg")}
    uids = data["train_uids"].tolist()
    require(uids == protocol["train_uids"] and len(uids) == len(set(uids)) == 12,
            "Training archive UID order/count differs")
    require(not set(uids) & set(protocol["internal_check_uids"]), "Internal-check label leakage")
    require(data["directions0"].shape == data["directions1"].shape == (12, 258, 3), "Direction shape changed")
    require(data["proxy_points"].shape == (12, 256, 3), "Proxy256 shape changed")
    require(data["faces"].dtype == np.int32 and data["faces"].shape == (512, 3), "Logical face contract changed")
    for key, value in data.items():
        if key != "train_uids":
            require(np.isfinite(value).all(), "Nonfinite training array: " + key)
    require(np.array_equal(point_features(data["proxy_points"]), data["point_features"]),
            "Cached features differ from original proxy input")
    scale = float(protocol["endpoint_scale_squared"])
    require(np.isfinite(scale) and scale > 0, "Global training scale must be finite and positive")
    require(np.isclose(float(np.mean((data["directions1"] - data["directions0"]) ** 2)),
                       scale, rtol=1e-12, atol=0), "Frozen global12 displacement scale differs")
    require(file_hash(path) == expected, "Training archive changed during read")
    return data


def save_raw(path, mesh, faces):
    save_npz(path, **mesh)
    identity = mesh_identity(path, faces)
    with np.load(path, allow_pickle=False) as stored:
        require(set(stored.files) == set(mesh) and all(np.array_equal(stored[k], v) for k, v in mesh.items()),
                "Saved mesh differs from generated arrays")
    return {**identity, "stored_array_hash_schema": STORED_HASH_SCHEMA}


def aggregate(run):
    run = Path(run)
    protocol = read_json(run / "RUN_PROTOCOL.json")
    expected = expected_slots(protocol)
    protocol_sha = file_hash(run / "RUN_PROTOCOL.json")
    attempts = [read_json(path) for path in sorted((run / "diagnostics/mechanism").glob("*/*/*/result.json"))]
    key = lambda item: (item["job"], item["checkpoint"], item["uid"])
    keys = [key(a) for a in attempts]
    require(len(keys) == len(set(keys)) and set(keys) <= expected, "Duplicate/unregistered diagnostic attempt")
    require(all(a["protocol_sha256"] == protocol_sha for a in attempts), "Stale attempt protocol")
    rows, errors = [], []
    for attempt in attempts:
        if attempt["status"] == "success":
            row = attempt["row"]
            require(key(row) == key(attempt) and row["split"] == "train", "Attempt row identity/split differs")
            rows.append(row)
        else:
            require(attempt["status"] == "error" and not attempt.get("row"), "Invalid diagnostic failure record")
            errors.append(attempt)
    groups = []
    for job in JOBS:
        for checkpoint in CHECKPOINTS:
            selected = [r for r in rows if r["job"] == job and r["checkpoint"] == checkpoint]
            planned = len(protocol["jobs"][job]["diagnostic_uids"])
            group = {"job": job, "checkpoint": checkpoint, "expected_parents": planned,
                     "successful_parents": len(selected), "missing_or_failed_parents": planned - len(selected),
                     "qualified_count": sum(r["qualified"] for r in selected),
                     "equal_optimizer_steps_or_compute_claim": False}
            for section, fields in (("velocity", ("mse", "zero_velocity_mse", "mse_over_zero_velocity")),
                ("endpoint", ("endpoint_direction_mse_normalized", "zero_motion_direction_mse_normalized",
                 "endpoint_mse_over_zero_motion", "movement_energy_over_teacher", "teacher_projection_coefficient",
                 "displacement_cosine_to_teacher", "displacement_edge_roughness", "teacher_displacement_edge_roughness",
                 "edge_roughness_over_teacher", "direction_rmse_to_teacher_deg"))):
                group[section] = {f: describe([r[section][f] for r in selected]) for f in fields}
            groups.append(group)
    value = {"scope": SCOPE, "protocol_sha256": protocol_sha, "checkpoint_selection": False,
             "internal_check_objects_opened": 0, "surface_rms_evaluated": False,
             "expected_attempt_count": len(expected), "attempt_count": len(attempts),
             "rows": rows, "errors": errors, "unattempted": sorted(expected - set(keys)),
             "missing_success_slots": sorted(expected - {key(r) for r in rows}), "groups": groups,
             "pooled_across_checkpoints_or_jobs": False,
             "warning": "Joint240k final matches single20k at 80000 presentations per parent, but not Adam steps, mixed-gradient history or compute; single references have seed10 only."}
    target = run / "TRAIN_DIAGNOSTICS.json"
    write_json(target, value, replace=target.exists())
    return value


def diagnose_run(run, selected_job=None, selected_checkpoint=None):
    run = Path(run).resolve()
    protocol = read_json(run / "RUN_PROTOCOL.json")
    expected_slots(protocol)
    protocol_sha = file_hash(run / "RUN_PROTOCOL.json")
    manifest = run / "DATA_MANIFEST.json"
    require(file_hash(manifest) == protocol["hashes"]["data_manifest_sha256"], "Manifest changed")
    module_root = Path(__file__).parents[1]
    for relative, expected in protocol.get("fixed_source_hashes", {}).items():
        require(file_hash(module_root / relative) == expected, "Frozen source changed: " + relative)
    jobs = list(JOBS) if selected_job is None else [selected_job]
    checkpoints = list(CHECKPOINTS) if selected_checkpoint is None else [str(selected_checkpoint)]
    require(set(jobs) <= set(JOBS) and set(checkpoints) <= set(CHECKPOINTS), "Unregistered diagnosis selection")
    data = load_train_archive(run, protocol)
    support = RadialSupport()
    source_sha = file_hash(Path(__file__))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    write_json(run / "diagnostics/invocations" / f"{stamp}.json", {
        "scope": SCOPE, "jobs": jobs, "checkpoints": checkpoints, "protocol_sha256": protocol_sha,
        "diagnose_source_sha256": source_sha, "training_labels_sha256": protocol["hashes"]["training_labels_sha256"],
        "data_source": "train-only archive; no per-object data files opened", "internal_check_objects_opened": 0,
        "teacher_labels_used_for_deployment_network_inputs": False,
        "teacher_conditioned_bridge_probe_is_offline_only": True})
    for job in jobs:
        for checkpoint in checkpoints:
            load_error = None
            try:
                model, identity = load_network(run, job, checkpoint, protocol)
            except Exception as exc:
                model, identity = None, None
                load_error = {"error": repr(exc), "traceback": traceback.format_exc(), "stage": "checkpoint_load"}
            for uid in protocol["jobs"][job]["diagnostic_uids"]:
                dest = run / "diagnostics/mechanism" / job / checkpoint / uid
                attempt = {"job": job, "checkpoint": checkpoint, "uid": uid, "protocol_sha256": protocol_sha,
                           "diagnose_source_sha256": source_sha}
                if (dest / "result.json").exists():
                    old = read_json(dest / "result.json")
                    require(all(old[k] == v for k, v in attempt.items()), "Existing diagnostic identity/source changed")
                    if old["status"] == "success":
                        require(identity is not None and old["row"]["identity"] == identity,
                                "Recorded checkpoint identity changed")
                    continue
                if dest.exists():
                    write_json(dest / "result.json", {**attempt, "status": "error", "row": None,
                               "stage": "previous_interruption", "error": "Partial attempt retained; no automatic retry"})
                    aggregate(run)
                    continue
                dest.mkdir(parents=True)
                write_json(dest / "START.json", {**attempt, "identity": identity})
                try:
                    if load_error is not None:
                        write_json(dest / "result.json", {**attempt, "status": "error", "row": None, **load_error})
                        aggregate(run)
                        continue
                    started = time.perf_counter()
                    index = data["train_uids"].tolist().index(uid)
                    base, teacher = data["directions0"][index], data["directions1"][index]
                    faces, logr = data["faces"], data["logr"][index]
                    velocity = velocity_probe(model, base, teacher, data["point_features"][index])
                    deployment = deploy(model, logr, support, base, faces)
                    require(deployment["timing"]["proxy_points_hash"] == array_hash(data["proxy_points"][index]),
                            "Free deployment proxy differs from frozen input")
                    floor = deployment["floor"]
                    observed_floor = [floor["min_triangle_angle_deg"], floor["triangle_min_angle_p05_deg"]]
                    require(np.allclose(observed_floor, data["quality_floors_deg"][index], rtol=0, atol=1e-12),
                            "Actual Fixed quality floor differs from frozen archive")
                    raw = deployment["raw_mesh"]
                    require(np.array_equal(raw["r"], support.query(logr, raw["u"]))
                            and np.array_equal(raw["vertices"], raw["u"] * raw["r"][:, None]),
                            "Raw mesh radius is stale")
                    hashes = save_raw(dest / "raw.npz", raw, faces)
                    endpoint = endpoint_probe(raw["u"], base, teacher, faces, protocol["endpoint_scale_squared"])
                    row = {"job": job, "checkpoint": checkpoint, "uid": uid, "split": "train", "identity": identity,
                           "velocity": velocity, "endpoint": endpoint, "quality": deployment["raw_quality"],
                           "quality_floor": floor, "audit": deployment["raw_audit"], **deployment["raw_qualification"],
                           "unchanged_quality_gate_would_fallback": deployment["delivery"]["fallback_to_fixed"],
                           "deployment_policy_reported": "raw only; fallback is a separately reported gate decision",
                           "mesh_file": str((dest / "raw.npz").relative_to(run)), **hashes,
                           "mainfield_hash": array_hash(logr), "teacher_label_hash": array_hash(base, teacher),
                           "training_labels_sha256": protocol["hashes"]["training_labels_sha256"],
                           "alpha_values": raw["alpha"].tolist(), "shrunk_steps": deployment["timing"]["shrunk_steps"],
                           "minimum_alpha": deployment["timing"]["min_alpha"], "deployment_timing": deployment["timing"],
                           "total_diagnostic_seconds": time.perf_counter() - started,
                           "exact_endpoint_audit_only_not_continuous_path_claim": True,
                           "teacher_or_complete_field_added_to_deployment_network_inputs": False}
                    write_json(dest / "result.json", {**attempt, "status": "success", "row": row})
                    print(json.dumps({"job": job, "checkpoint": checkpoint, "uid": uid,
                                      "status": "success", "qualified": row["qualified"]}), flush=True)
                except Exception as exc:
                    write_json(dest / "result.json", {**attempt, "status": "error", "row": None,
                               "error": repr(exc), "traceback": traceback.format_exc()})
                aggregate(run)
    return aggregate(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--job", choices=JOBS)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    result = aggregate(args.run) if args.aggregate_only else diagnose_run(args.run, args.job, args.checkpoint)
    print(json.dumps({"scope": result["scope"], "attempt_count": result["attempt_count"],
                      "expected_attempt_count": result["expected_attempt_count"], "errors": len(result["errors"])}))


if __name__ == "__main__":
    main()
