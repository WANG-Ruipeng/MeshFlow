"""CPU-only fixed first-seed views using frozen target points and GT bounds."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .reporting import MODELS, SOLVERS, collect, read_json, sha256, write_json


def array_digest(value):
    import hashlib
    a = np.asarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def run(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from experiments.mf_h1_local0.jedge5_diagnosis.common import load_cases
    from experiments.mf_h1_local0.evaluate import surface_distance_squared

    begin = time.perf_counter()
    out = Path(out).resolve()
    protocol, rows, _, _, _, _ = collect(out)
    seed = int(protocol["generation_seeds"][0])
    plot_spec = protocol.get("plots", {})
    elev = plot_spec.get("elev", plot_spec.get("elevation", 22))
    azim = plot_spec.get("azim", plot_spec.get("azimuth", -55))
    # Fixed before outputs: original parent GT bounding cube plus 10% each side.
    radius_factor = plot_spec.get("radius_factor", .60)
    vertical_axis = plot_spec.get("vertical_axis", "y")
    color = plot_spec.get("coverage_color", {})
    norm = Normalize(vmin=color.get("vmin", 0), vmax=color.get("vmax", 10), clip=True)
    cmap = color.get("cmap", "viridis")
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    selected = {(int(r["parent"]), int(r["patch"]), r["model"], r["solver"]): r for r in rows if int(r["seed"]) == seed}
    cases = load_cases()
    records, query_records = [], []
    configs = [(m, s) for m in MODELS for s in SOLVERS]
    for parent in (0, 1):
        gt_points = cases[parent, 0]["full"].reshape(-1, 3)
        center = (gt_points.max(0) + gt_points.min(0)) / 2
        radius = float(np.ptp(gt_points, axis=0).max()) * radius_factor
        for patch in range(5):
            started = time.perf_counter()
            case = cases[parent, patch]
            q = np.asarray(case["coverage_points"], dtype=np.float64)
            target = np.asarray(case["free"]).reshape(100, 3, 3)
            condition = np.asarray(case["C"]).reshape(12, 3, 3)
            fig = plt.figure(figsize=(20, 10.8))
            annotations = []
            for col, (model, solver) in enumerate(configs):
                row = selected[parent, patch, model, solver]
                path = out / "raw" / (row["task_id"] + ".npz")
                raw, distances = None, None
                status = row["sampling_status"]
                if path.exists():
                    with np.load(path, allow_pickle=False) as z:
                        candidate = z["raw"].copy().reshape(112, 3, 3)
                        known = z["known"].reshape(-1).astype(bool) if "known" in z.files else np.arange(112) < 12
                        free = z["free"].reshape(-1).astype(bool) if "free" in z.files else ~known
                    if known.sum() != 12 or free.sum() != 100 or not np.array_equal(candidate[known], condition):
                        raise ValueError("Raw known/free identity mismatch before plotting")
                    # Incomplete endpoint is shown as missing, never scored as t=1.
                    completed = row["sampling"].get("status", "") in ("PASS", "COMPLETE", "SUCCESS")
                    if row["scalars54"].get("fixed_R12_coverage_RMS_bbox_pct") is not None:
                        completed = True
                    if completed and np.isfinite(candidate).all():
                        raw = candidate
                        tick = time.perf_counter()
                        d2 = surface_distance_squared(q, raw[free].astype(np.float64))
                        distances = 100 * np.sqrt(d2) / case["bbox"]
                        score = row["scalars54"].get("fixed_R12_coverage_RMS_bbox_pct")
                        reconstructed = float(100 * np.sqrt(d2.mean()) / case["bbox"])
                        if score is not None and not np.isclose(reconstructed, score, atol=1e-10, rtol=1e-10):
                            raise ValueError(f"Frozen coverage plot/score mismatch: {row['task_id']}")
                        query_path = folder / (row["task_id"] + "_coverage_queries.npz")
                        np.savez_compressed(query_path, target_points=q, distance_squared=d2,
                                            distance_bbox_percent=distances)
                        query_records.append(dict(task_id=row["task_id"], raw_sha256=sha256(path),
                            query_points_sha256=array_digest(q), query_count=len(q),
                            fixed_R12_coverage_RMS_bbox_pct=reconstructed,
                            above_color_maximum=int(np.sum(distances > norm.vmax)),
                            out_of_fixed_bbox_vertex_count=int(np.any(np.abs(raw.reshape(-1, 3)-center)>radius, axis=1).sum()),
                            query_seconds=time.perf_counter()-tick,
                            query_file=str(query_path.relative_to(out)), query_file_sha256=sha256(query_path)))
                for view in range(3):
                    ax = fig.add_subplot(3, 6, view*6+col+1, projection="3d")
                    if raw is not None and view < 2:
                        ax.add_collection3d(Poly3DCollection(raw[free], facecolors=(.57, .68, .80, .93), edgecolors=(.15, .24, .32, .4), linewidths=.3))
                        if view == 0:
                            ax.add_collection3d(Poly3DCollection(condition, facecolors=(1., .62, .16, .98), edgecolors=(.55, .25, .03, .8), linewidths=.6))
                    elif view == 2:
                        ax.add_collection3d(Poly3DCollection(target, facecolors=(.6, .6, .6, .06), edgecolors=(.35, .35, .35, .18), linewidths=.3))
                        ax.add_collection3d(Poly3DCollection(condition, facecolors=(1., .62, .16, .15), edgecolors=(.7, .35, .05, .6), linewidths=.5))
                        if distances is not None:
                            ax.scatter(q[:, 0], q[:, 1], q[:, 2], c=distances, cmap=cmap, norm=norm, s=8, depthshade=False, edgecolors="none")
                    if raw is None:
                        ax.text2D(.5, .5, "NO COMPLETE ENDPOINT\n" + str(status), transform=ax.transAxes, ha="center", fontsize=8)
                    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
                    ax.set_box_aspect((1, 1, 1))
                    ax.view_init(elev=elev, azim=azim, vertical_axis=vertical_axis)
                    ax.set_axis_off()
                    if view == 0:
                        rms = row["scalars54"].get("remainder_symmetric_RMS_bbox_pct")
                        label = f"{rms:.4f}%" if rms is not None else "NA"
                        ax.set_title(f"{model} / {solver} / 50 NFE\nremainder RMS {label}", fontsize=9)
                    if col == 0:
                        ax.text2D(-.05, .5, ("Full: orange C, blue free", "Generated free only", "GT unknown-region coverage")[view], rotation=90, transform=ax.transAxes, va="center", fontsize=9)
                if raw is not None:
                    count = query_records[-1]["out_of_fixed_bbox_vertex_count"]
                    above = query_records[-1]["above_color_maximum"]
                    if count or above:
                        annotations.append(f"{model}/{solver}: {count} vertices outside fixed GT cube; {above} points >{norm.vmax:g}%L")
            fig.suptitle(f"P{parent} patch {patch} / registered seed {seed} / fixed GT bounds and camera", fontsize=13)
            fig.legend(handles=[Patch(facecolor="#ffa329", label="Known C"), Patch(facecolor="#91adc9", label="Generated free"), Patch(facecolor="white", edgecolor="gray", label="GT free (heatmap row)")], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(.5, .001))
            if annotations:
                fig.text(.04, .036, " | ".join(annotations), fontsize=7)
            fig.subplots_adjust(left=.03, right=.945, top=.92, bottom=.07, wspace=0, hspace=.025)
            bar = fig.add_axes([.96, .13, .012, .25])
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar, label="GT point distance to free [% bbox]", extend="max")
            path = folder / f"P{parent}_patch{patch}_seed{seed}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            records.append(dict(parent=parent, patch=patch, seed=seed, path=str(path.relative_to(out)),
                                sha256=sha256(path), seconds=time.perf_counter()-started,
                                center=center.tolist(), radius=radius, target_points_sha256=array_digest(q),
                                requested_configurations=configs, annotations=annotations))
            print(f"plot P{parent}/patch{patch} complete", flush=True)
    result = dict(status="PASS", records=records, query_records=query_records,
                  model_forwards=0, distance_query_calls=len(query_records),
                  total_target_points=sum(r["query_count"] for r in query_records),
                  seconds=time.perf_counter()-begin, protocol_sha256=sha256(out/"registration/protocol.json"),
                  camera=dict(elev=elev, azim=azim, vertical_axis=vertical_axis, radius_factor=radius_factor,
                              bounds_source="original full GT only; never adjusted using generated outputs"),
                  colors=dict(minimum=norm.vmin, maximum=norm.vmax, cmap=cmap, units="distance percent bbox", clipped=True),
                  note="All ten first-seed contexts retained, including failures. Fixed GT query points; CPU plotting distances separately accounted. No geometry modifications.")
    write_json(folder / "plot_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
