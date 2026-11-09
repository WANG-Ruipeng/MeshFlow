"""Offline final-model fit diagnosis; no training, tuning, or checkpoint selection.

The velocity probe evaluates on teacher-conditioned bridge states and is NOT a
deployment test. Deployment results remain the separately saved raw rollouts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..io_utils import file_hash, write_json
from .train import discrete_bridge
from .evaluate import load_network


def describe(values):
    values = np.asarray(values, np.float64)
    return {"count": int(len(values)),
            "minimum": float(values.min()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None}


def angle_rms_deg(a, b):
    angle = 2 * np.arcsin(np.clip(np.linalg.norm(a - b, axis=-1) / 2, 0, 1))
    return float(np.rad2deg(np.sqrt(np.mean(angle * angle))))


@torch.no_grad()
def run_diagnostic(run):
    run = Path(run).resolve()
    destination = run / "FIT_DIAGNOSTIC.json"
    if destination.exists():
        raise FileExistsError(destination)
    protocol = json.loads((run / "RUN_PROTOCOL.json").read_text())
    ledger = json.loads((run / "results.json").read_text())
    lookup = {(r["uid"], r["model"], r["policy"]): r for r in ledger["rows"]}
    manifest = json.loads((run / "DATA_MANIFEST.json").read_text())["files"]
    torch.set_num_threads(4)
    models, identities = {}, {}
    for name in ("original_a", "student_10", "student_11"):
        models[name], identities[name] = load_network(run, name, protocol)
    velocity_rows, endpoint_rows = [], []
    for uid in protocol["uids"]:
        rel = "data/" + uid + ".npz"
        if file_hash(run / rel) != manifest[rel]:
            raise ValueError("Prepared data hash changed")
        with np.load(run / rel, allow_pickle=False) as data:
            base, teacher = data["base_u"], data["teacher_u"]
            features = data["point_features"]
        split = "train" if uid in protocol["train_uids"] else "internal_check"
        base_tensor = torch.from_numpy(base)[None].expand(32, -1, -1)
        target_tensor = torch.from_numpy(teacher)[None].expand(32, -1, -1)
        positions, velocity, _, t = discrete_bridge(base_tensor, target_tensor, torch.arange(32))
        positions, velocity, t = positions.float(), velocity.float(), t.float()
        proxy_features = torch.from_numpy(features)[None].expand(32, -1, -1)
        zero_mse = float(velocity.square().double().mean())
        velocity_rows.append({"uid": uid, "split": split, "model": "zero_velocity",
                              "mse": zero_mse, "mse_over_zero": 1.0 if zero_mse > 0 else None,
                              "stationary_teacher": bool(np.array_equal(base, teacher))})
        for name, model in models.items():
            squared_sum = 0.0
            element_count = 0
            for begin in range(0, 32, 4):
                prediction = model(positions[begin:begin + 4], t[begin:begin + 4],
                                   proxy_features[begin:begin + 4])
                squared = (prediction - velocity[begin:begin + 4]).square().double()
                squared_sum += float(squared.sum())
                element_count += squared.numel()
            mse = squared_sum / element_count
            velocity_rows.append({"uid": uid, "split": split, "model": name, "mse": mse,
                                  "mse_over_zero": mse / zero_mse if zero_mse > 0 else None,
                                  "stationary_teacher": bool(np.array_equal(base, teacher))})
        fixed_error = angle_rms_deg(base, teacher)
        for name in ("fixed", "original_a", "student_10", "student_11"):
            row = lookup[(uid, name, "raw")]
            with np.load(run / row["mesh_file"], allow_pickle=False) as mesh:
                actual = mesh["u"]
            error = angle_rms_deg(actual, teacher)
            endpoint_rows.append({
                "uid": uid, "split": split, "model": name,
                "direction_rmse_to_teacher_deg": error,
                "fixed_zero_displacement_rmse_to_teacher_deg": fixed_error,
                "closer_to_teacher_than_fixed": bool(error + 1e-12 < fixed_error),
                "qualified": row["qualified"],
                "min_angle_signed_margin_deg":
                    row["quality"]["min_triangle_angle_deg"] -
                    row["quality_floor"]["min_triangle_angle_deg"],
                "p05_angle_signed_margin_deg":
                    row["quality"]["triangle_min_angle_p05_deg"] -
                    row["quality_floor"]["triangle_min_angle_p05_deg"],
            })
    summaries = []
    for split in ("all", "train", "internal_check"):
        for model in ("zero_velocity", "original_a", "student_10", "student_11"):
            selected = [r for r in velocity_rows if r["model"] == model
                        and (split == "all" or r["split"] == split)]
            values = [r["mse"] for r in selected]
            zero = [r["mse"] for r in velocity_rows if r["model"] == "zero_velocity"
                    and (split == "all" or r["split"] == split)]
            summaries.append({
                "split": split, "model": model, "objects": len(selected),
                "same_grid_mse": float(np.mean(values)),
                "same_grid_zero_velocity_mse": float(np.mean(zero)),
                "same_grid_mse_over_zero": float(np.mean(values) / np.mean(zero)),
                "per_object_mse": describe(values),
                "objects_with_lower_mse_than_zero": sum(
                    r["mse"] < next(z["mse"] for z in velocity_rows
                                     if z["uid"] == r["uid"] and z["model"] == "zero_velocity")
                    for r in selected),
                "stationary_teacher_objects": sum(r["stationary_teacher"] for r in selected),
            })
    endpoint_summaries = []
    for split in ("all", "train", "internal_check"):
        for model in ("fixed", "original_a", "student_10", "student_11"):
            selected = [r for r in endpoint_rows if r["model"] == model
                        and (split == "all" or r["split"] == split)]
            item = {
                "split": split, "model": model, "objects": len(selected),
                "direction_rmse_to_teacher_deg":
                    describe([r["direction_rmse_to_teacher_deg"] for r in selected]),
                "closer_to_teacher_than_fixed_count":
                    sum(r["closer_to_teacher_than_fixed"] for r in selected),
                "qualified_count": sum(r["qualified"] for r in selected),
            }
            for key in ("min_angle_signed_margin_deg", "p05_angle_signed_margin_deg"):
                margins = np.asarray([r[key] for r in selected])
                failed = margins < -1e-6
                item[key] = describe(margins)
                item[key.replace("signed_margin", "shortfall_on_failed")] = describe(-margins[failed])
            endpoint_summaries.append(item)
    report = {
        "scope": "OFFLINE_TEACHER_CONDITIONED_NOT_DEPLOYMENT",
        "changes_to_training_or_checkpoints": False,
        "checkpoint_selection": "previously frozen final.pt only; diagnostic never selects weights",
        "velocity_probe": {
            "states": "exact per-object teacher bridge at all k=0,...,31, normalized chord directions",
            "conditions": "original proxy256 features; teacher-conditioned query direction/time",
            "targets": "same discrete bridge targets as training, double computation then float32",
            "mse": "mean over all 32 steps, 258 directions, and 3 coordinates; object counts equal",
            "comparison": "zero vector field, original A, final seed10 and seed11 on identical grid",
            "device": "cpu", "network_dtype": "float32",
            "warning": "Teacher-conditioned states differ from free rollout; MSE is not deployment success.",
        },
        "endpoint_probe": "saved raw deployment directions versus chosen teachers, with Fixed zero-displacement comparator",
        "quality_margin_definition": "actual raw triangle angle statistic minus per-object Fixed statistic, in degrees",
        "shortfall_definition": "positive Fixed-minus-raw deficit, only objects failing the unchanged 1e-6 degree tolerance",
        "identities": identities,
        "protocol_sha256": file_hash(run / "RUN_PROTOCOL.json"),
        "results_sha256": file_hash(run / "results.json"),
        "source_sha256": {"fit_diagnostic.py": file_hash(Path(__file__)),
                          "train.py": file_hash(Path(__file__).with_name("train.py"))},
        "velocity_summary": summaries, "endpoint_summary": endpoint_summaries,
        "velocity_rows": velocity_rows, "endpoint_rows": endpoint_rows,
    }
    write_json(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    report = run_diagnostic(args.run)
    print(json.dumps({
        "scope": report["scope"],
        "velocity": [x for x in report["velocity_summary"] if x["split"] == "all"],
        "endpoint": [x for x in report["endpoint_summary"] if x["split"] == "all"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()


