"""CPU-only fixed-view panels; raw geometry is never repaired or resampled.

Input per parent::
    {"parent_id": ..., "conditions": [
      {"condition_id": ..., "gt_free": array[N-K,3,3], "known": array[K,3,3],
       "outputs": {"T1": {"raw": array[N,3,3], "known_mask": array[N],
                            "noise_id": 0, "status": "PASS"}, ...}}, ...]}

The caller selects noise_id=0 from the frozen manifest and supplies each C's own
GT free complement. Masks must be explicit; no face identity is inferred from GT.
Missing/failed results retain their panel. Shared bounds include every finite raw
coordinate for both conditions, so no finite failure/outlier is silently clipped.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = ("T1", "Geo", "Full")
VIEW = {"elevation": 20.0, "azimuth": -55.0, "coordinate_display_order": [0, 2, 1],
        "projection": "ortho", "bounds": "parent union of both conditions and all finite raw outputs"}
COLORS = {"known": "#F6AA3D", "free": "#719DC7", "gt_free": "#A6B5C3", "altered_known": "#C53232"}


def _triangles(value):
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[-1] == 9:
        array = array.reshape(-1, 3, 3)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(f"Expected explicit unpadded triangles [N,3,3] or [N,9], got {array.shape}")
    return array


def _load_array(value, base_dir):
    """CLI array references use {path, key}; plain arrays remain supported."""
    if isinstance(value, dict) and "path" in value:
        path = Path(value["path"])
        if not path.is_absolute():
            path = base_dir / path
        if path.suffix.lower() == ".npy":
            return np.load(path, allow_pickle=False)
        with np.load(path, allow_pickle=False) as data:
            return np.array(data[value["key"]], copy=True)
    return np.asarray(value)


def resolve_arrays(parents, base_dir):
    resolved = []
    for source in parents:
        parent = dict(source, conditions=[])
        for c in source["conditions"]:
            condition = dict(c, gt_free=_load_array(c["gt_free"], base_dir),
                             known=_load_array(c["known"], base_dir), outputs={})
            for model, entry in c["outputs"].items():
                output = dict(entry)
                for field in ("raw", "known_mask"):
                    if output.get(field) is not None:
                        output[field] = _load_array(output[field], base_dir)
                condition["outputs"][model] = output
            parent["conditions"].append(condition)
        resolved.append(parent)
    return resolved


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_fixed_views(parents, output_dir, *, noise_id=0, require_eight=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if noise_id != 0:
        raise ValueError("The fixed view uses the first registered test noise (noise_id=0)")
    if require_eight and len(parents) != 8:
        raise ValueError("All eight test parents must be included; no sample selection")
    if len({p["parent_id"] for p in parents}) != len(parents):
        raise ValueError("Duplicate parent figure input")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for parent_index, parent in enumerate(sorted(parents, key=lambda p: str(p["parent_id"]))):
        conditions = sorted(parent["conditions"], key=lambda c: str(c["condition_id"]))
        if len(conditions) != 2 or len({c["condition_id"] for c in conditions}) != 2:
            raise ValueError("Every parent requires its two distinct registered C conditions")
        coordinate_sets, prepared = [], []
        for condition in conditions:
            known, gt_free = _triangles(condition["known"]), _triangles(condition["gt_free"])
            if not np.isfinite(known).all() or not np.isfinite(gt_free).all():
                raise ValueError("Frozen GT/C used by figures must be finite")
            coordinate_sets.extend((known.reshape(-1, 3), gt_free.reshape(-1, 3)))
            outputs = {}
            for model in MODELS:
                entry = dict(condition.get("outputs", {}).get(model, {}))
                if entry.get("noise_id", noise_id) != noise_id:
                    raise ValueError(f"Wrong figure noise for {parent['parent_id']} / {model}")
                status = entry.get("status", "NOT_RUN")
                if entry.get("raw") is None:
                    if status == "PASS":
                        raise ValueError("PASS figure output must have its raw endpoint")
                    outputs[model] = dict(status=status, missing=True,
                                          failure_reason=entry.get("failure_reason", "No raw endpoint"))
                    continue
                raw = _triangles(entry["raw"])
                mask = np.asarray(entry.get("known_mask"))
                if mask.shape != (len(raw),) or mask.dtype != np.bool_:
                    raise ValueError("Raw output requires its explicit unpadded boolean known_mask")
                if int(mask.sum()) != len(known):
                    raise ValueError("Raw known mask and condition C count disagree")
                finite_coords = raw.reshape(-1, 3)
                finite_coords = finite_coords[np.isfinite(finite_coords).all(axis=1)]
                if finite_coords.size:
                    coordinate_sets.append(finite_coords)
                outputs[model] = dict(status=status, missing=False, raw=raw, mask=mask,
                                      C_preserved=bool(np.array_equal(raw[mask], known)),
                                      actual_N=len(raw), expected_N=len(known)+len(gt_free),
                                      nonfinite_triangles=int((~np.isfinite(raw).all(axis=(1, 2))).sum()),
                                      failure_reason=entry.get("failure_reason"))
            prepared.append(dict(condition_id=condition["condition_id"], known=known,
                                 gt_free=gt_free, outputs=outputs))
        points = np.concatenate(coordinate_sets, axis=0)[:, VIEW["coordinate_display_order"]]
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))
        radius = max(float(np.ptp(points, axis=0).max()) * 0.54, 1e-9)
        fig = plt.figure(figsize=(20, 8.4), layout="constrained")
        panel_audit = []

        def draw(ax, triangles, color, *, known=False):
            valid = np.isfinite(triangles).all(axis=(1, 2))
            visible = triangles[valid][:, :, VIEW["coordinate_display_order"]]
            if len(visible):
                ax.add_collection3d(Poly3DCollection(
                    visible, facecolors=color, edgecolors="#9C5708" if known else "#3C5266",
                    linewidths=0.6 if known else 0.25, alpha=0.98 if known else 0.9))

        for row_index, condition in enumerate(prepared):
            known, gt_free = condition["known"], condition["gt_free"]
            for col_index, title in enumerate(("GT", "C", *MODELS)):
                ax = fig.add_subplot(2, 5, row_index * 5 + col_index + 1, projection="3d")
                ax.set_proj_type(VIEW["projection"])
                ax.view_init(elev=VIEW["elevation"], azim=VIEW["azimuth"])
                ax.set_xlim(center[0]-radius, center[0]+radius)
                ax.set_ylim(center[1]-radius, center[1]+radius)
                ax.set_zlim(center[2]-radius, center[2]+radius)
                ax.set_box_aspect((1, 1, 1))
                ax.set_axis_off()
                label = f"C {condition['condition_id']} | {title}"
                if title == "GT":
                    draw(ax, gt_free, COLORS["gt_free"])
                    draw(ax, known, COLORS["known"], known=True)
                    label += f" (N={len(known)+len(gt_free)})"
                elif title == "C":
                    draw(ax, known, COLORS["known"], known=True)
                    label += f" (K={len(known)})"
                else:
                    entry = condition["outputs"][title]
                    audit = {k: v for k, v in entry.items() if k not in ("raw", "mask")}
                    panel_audit.append(dict(condition_id=condition["condition_id"], model=title,
                                            noise_id=noise_id, **audit))
                    if entry["missing"]:
                        ax.text2D(0.5, 0.5, f"{entry['status']}\nNO RAW ENDPOINT", transform=ax.transAxes,
                                  ha="center", va="center", color="#A32020", fontsize=12)
                    else:
                        draw(ax, entry["raw"][~entry["mask"]], COLORS["free"])
                        ccolor = COLORS["known"] if entry["C_preserved"] else COLORS["altered_known"]
                        draw(ax, entry["raw"][entry["mask"]], ccolor, known=True)
                        label += f" (N={entry['actual_N']})"
                        warnings = []
                        if entry["status"] != "PASS":
                            warnings.append(entry["status"])
                        if not entry["C_preserved"]:
                            warnings.append("C ALTERED")
                        if entry["nonfinite_triangles"]:
                            warnings.append(f"nonfinite triangles: {entry['nonfinite_triangles']}")
                        if entry["actual_N"] != entry["expected_N"]:
                            warnings.append(f"expected N={entry['expected_N']}")
                        if warnings:
                            ax.text2D(0.5, 0.02, "\n".join(warnings), transform=ax.transAxes,
                                      ha="center", va="bottom", color="#A32020", fontsize=9)
                ax.set_title(label, fontsize=11)
        has_raw=any(not c['outputs'][m]['missing'] for c in prepared for m in MODELS)
        stage='Raw endpoints' if has_raw else 'Input preview; model endpoints NOT_RUN'
        fig.suptitle(f"Parent {parent['parent_id']} | fixed first test noise | shared bounds and view\n"
                     f"{stage}; orange = C, blue = generated free, gray = GT unknown region",
                     fontsize=14)
        fig.legend(handles=[Patch(facecolor=COLORS["known"], label="Copied C"),
                            Patch(facecolor=COLORS["free"], label="Generated free"),
                            Patch(facecolor=COLORS["gt_free"], label="GT unknown region")],
                   loc="outside lower center", ncol=3, frameon=False)
        path = output_dir / f"parent_{parent_index:02d}_noise0.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        records.append(dict(parent_id=parent["parent_id"], noise_id=noise_id,
                            conditions=[c["condition_id"] for c in conditions], path=path.name,
                            sha256=_sha256(path), bounds_center=center.tolist(), bounds_radius=radius,
                            panels=panel_audit))
    panels=[p for r in records for p in r['panels']]
    rendered=sum(not p['missing'] for p in panels)
    pending=sum(p['status']=='NOT_RUN' for p in panels)
    stage=('FIXTURE' if len(records)!=8 else 'PREPARATION_PREVIEW' if pending==48
           else 'COMPLETE' if rendered==48 else 'INCOMPLETE_OUTPUT_VIEWS')
    manifest = dict(status=stage, formal_endpoint_panels=rendered, pending_endpoint_panels=pending, records=records,
                    view=VIEW, colors=COLORS, selected_noise_id=0, expected_parents=8,
                    parent_figures=len(records), all_failed_panels_preserved=True,
                    no_geometry_modification=True, no_resampling=True,
                    nonfinite_rendering="Nonfinite triangles cannot be drawn; their panel and count are retained",
                    model_forwards=0, geometry_score_calls=0)
    (output_dir / "eval_fixed_view_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8-sig"))
    parents = resolve_arrays(source["parents"], args.input.parent)
    result = render_fixed_views(parents, args.out)
    print(json.dumps({"status": result["status"], "parent_figures": result["parent_figures"]}))


if __name__ == "__main__":
    main()
