"""Offline endpoint-informed oracle audit; never a student or deployment result.

Both oracles contain each object's target directions. They only diagnose
numerical compatibility with the unchanged 32-step CPU guard and final query.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from ..exporter import RadialSupport
from ..io_utils import array_hash, file_hash, write_json
from ..legacy.deform import deform
from .deployment import named_quality, qualification
from ..layout_value.geometry import audit_mesh
from .train import discrete_bridge


class EndpointOracle(torch.nn.Module):
    """Offline-only teacher direction lookup, forbidden as a deployed model."""
    def __init__(self, base, target, mode):
        super().__init__()
        self.base = torch.as_tensor(base, dtype=torch.float64)[None]
        self.target = torch.as_tensor(target, dtype=torch.float64)[None]
        self.mode = mode

    def context(self, proxy_features):
        return None

    def forward(self, current, t, points, context=None):
        if self.mode == "continuous_bridge":
            delta = self.target - self.base
            chord = self.base + t.to(torch.float64).reshape(-1, 1, 1) * delta
            norm = torch.linalg.vector_norm(chord, dim=-1, keepdim=True)
            current64 = current.to(torch.float64)
            velocity = delta / norm
            velocity -= (velocity * current64).sum(-1, keepdim=True) * current64
        elif self.mode == "discrete_bridge":
            k = torch.round(t.reshape(-1) * 32).to(torch.long)
            _, velocity, _, _ = discrete_bridge(self.base, self.target, k, steps=32)
        else:
            raise ValueError(self.mode)
        return velocity.to(torch.float32)


def run_diagnostic(run):
    run = Path(run).resolve()
    protocol = json.loads((run / "RUN_PROTOCOL.json").read_text())
    manifest = json.loads((run / "DATA_MANIFEST.json").read_text())["files"]
    if (run / "ORACLE_NUMERICS.json").exists():
        raise FileExistsError(run / "ORACLE_NUMERICS.json")
    support = RadialSupport()
    torch.set_num_threads(4)
    rows = []
    for uid in protocol["uids"]:
        relative = "data/" + uid + ".npz"
        data_path = run / relative
        if file_hash(data_path) != manifest[relative]:
            raise ValueError("Diagnostic input differs from prepared data")
        with np.load(data_path, allow_pickle=False) as data:
            base, target = data["base_u"], data["teacher_u"]
            faces, logr = data["faces"], data["logr"]
            proxy = data["proxy_points"]
        fixed_vertices = base * support.query(logr, base)[:, None]
        floor = named_quality(fixed_vertices, faces)
        target_vertices = target * support.query(logr, target)[:, None]
        target_quality = named_quality(target_vertices, faces)
        for mode in ("continuous_bridge", "discrete_bridge"):
            started = time.perf_counter()
            oracle = EndpointOracle(base, target, mode)
            directions, path, alpha, _ = deform(oracle, proxy, base, faces, True, 32)
            radii = support.query(logr, directions)
            vertices = directions * radii[:, None]
            quality = named_quality(vertices, faces)
            audit = audit_mesh(vertices, faces, exact=True)
            decision = qualification(quality, audit, floor)
            diff = directions - target
            angular = 2 * np.arcsin(np.clip(np.linalg.norm(diff, axis=1) / 2, 0, 1))
            rows.append({
                "uid": uid, "mode": mode,
                "scope": "OFFLINE_ONLY_TEACHER_ENDPOINT_ORACLE",
                "forbidden_at_deployment": True,
                "split": "train" if uid in protocol["train_uids"] else "internal_check",
                "stationary_teacher": bool(np.array_equal(base, target)),
                "teacher_direction_hash": array_hash(target),
                "output_direction_hash": array_hash(directions),
                "max_direction_chord_error": float(np.linalg.norm(diff, axis=1).max()),
                "max_direction_angular_error_rad": float(angular.max()),
                "mean_direction_angular_error_rad": float(angular.mean()),
                "min_alpha": float(alpha.min()),
                "shrunk_steps": int((alpha < 1).sum()),
                "min_angle_margin_deg": quality["min_triangle_angle_deg"] - floor["min_triangle_angle_deg"],
                "p05_angle_margin_deg": quality["triangle_min_angle_p05_deg"] - floor["triangle_min_angle_p05_deg"],
                "teacher_min_angle_margin_deg": target_quality["min_triangle_angle_deg"] - floor["min_triangle_angle_deg"],
                "teacher_p05_angle_margin_deg": target_quality["triangle_min_angle_p05_deg"] - floor["triangle_min_angle_p05_deg"],
                "quality": quality, "audit": audit, **decision,
                "seconds_including_audit": time.perf_counter() - started,
            })
    modes = {}
    for mode in ("continuous_bridge", "discrete_bridge"):
        selected = [row for row in rows if row["mode"] == mode]
        modes[mode] = {
            "attempts": len(selected),
            "qualified": sum(row["qualified"] for row in selected),
            "valid": sum(row["audit"]["valid"] for row in selected),
            "max_direction_angular_error_rad": max(row["max_direction_angular_error_rad"] for row in selected),
            "shrunk_steps_total": sum(row["shrunk_steps"] for row in selected),
        }
    source_paths = [Path(__file__), Path(__file__).with_name("train.py"),
                    Path(__file__).with_name("deployment.py"),
                    Path(__file__).parents[1] / "legacy/deform.py",
                    Path(__file__).parents[1] / "legacy/chart.py"]
    report = {
        "scope": "OFFLINE_ONLY_TEACHER_ENDPOINT_ORACLE",
        "forbidden_at_deployment": True,
        "contains_teacher_endpoint": True,
        "uses_original_32_step_cpu_deform_and_final_radius_query": True,
        "bridge_target_computation": "float64; oracle output is float32; unchanged guard uses float64",
        "continuous_mode": "analytic tangent velocity using current direction and teacher chord denominator",
        "discrete_mode": "precomputed consecutive normalized teacher chord velocities from training discrete_bridge",
        "purpose": "Numerical compatibility audit only; no learned improvement, no deployment capability claim.",
        "protocol_sha256": file_hash(run / "RUN_PROTOCOL.json"),
        "data_manifest_sha256": file_hash(run / "DATA_MANIFEST.json"),
        "source_sha256": {str(path.relative_to(Path(__file__).parents[1])): file_hash(path) for path in source_paths},
        "summary": modes, "rows": rows,
    }
    write_json(run / "ORACLE_NUMERICS.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    report = run_diagnostic(args.run)
    print(json.dumps({"scope": report["scope"], "summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

