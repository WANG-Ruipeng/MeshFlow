"""All-attempt comparison figures; failure and qualification remain visible.

Call render_report(run_dir) after results.json has been written. This does not
select checkpoints, filter to qualified samples, or modify any mesh/result.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ARMS = ("fixed", "a", "opt_fixed", "opt_a", "vtk_qem", "cgal_constrained")
LABELS = {"fixed": "Fixed", "a": "Frozen A", "opt_fixed": "Fixed + optimization",
          "opt_a": "A + optimization", "vtk_qem": "VTK QEM", "cgal_constrained": "Constrained CGAL"}
COLORS = dict(zip(ARMS, ("#64748b", "#2563eb", "#16a34a", "#8b5cf6", "#ea580c", "#dc2626")))
MARKERS = dict(zip(ARMS, ("o", "s", "^", "D", "v", "P")))


def _number(value):
    try:
        answer = float(value)
        return answer if np.isfinite(answer) else None
    except (TypeError, ValueError):
        return None


def _rms(row):
    return _number((row.get("metrics") or {}).get("rms_bbox_percent"))


def _angle(row):
    return _number((row.get("quality") or {}).get("angle_q05_deg"))


def _draw_point(ax, x, y, row, *, size=34, alpha=.85):
    arm = row["arm"]
    qualified = row.get("qualified") is True
    ax.scatter([x], [y], s=size, marker=MARKERS[arm],
               facecolors=COLORS[arm] if qualified else "none",
               edgecolors=COLORS[arm], linewidths=1.15, alpha=alpha, zorder=3)


def _legend(ax):
    handles = [Line2D([], [], linestyle="", marker=MARKERS[a], color=COLORS[a],
                      markersize=7, label=LABELS[a]) for a in ARMS]
    handles.append(Line2D([], [], linestyle="", marker="o", markerfacecolor="none",
                          color="black", label="Hollow: not qualified / unavailable qualification"))
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")


def _save(fig, directory, name, footer, *, top=1.0):
    fig.text(.015, .012, footer, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, .07, 1, top))
    paths = []
    for extension in ("png", "pdf"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def _safe_mesh_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("mesh_file escapes the run directory")
    return path


def _load_mesh(path):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return np.asarray(z["vertices"], dtype=float), np.asarray(z["faces"], dtype=int)
    import trimesh
    mesh = trimesh.load(path, force="mesh", process=False)
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int)


def render_report(run_dir, *, overwrite=False, metrics_only=False, contact_sheets_only=False):
    """Render paired per-object and cost/quality figures, plus geometry sheets.

    All readable metric records are plotted even when their validity, quality,
    budget, or status fails. Non-readable/missing records are explicitly counted
    in REPORT.json and the figure footer. No aggregate success-only score exists.
    Returns a JSON-native report listing absolute artifact paths. Explicit
    overwrite=True permits updating figures; metrics_only=True updates only the
    three statistical figures; contact_sheets_only=True updates only mesh sheets.
    Both partial-render modes preserve REPORT.json.
    """
    if metrics_only and contact_sheets_only:
        raise ValueError("Choose at most one partial-render mode")
    root = Path(run_dir).resolve()
    document = json.loads((root / "results.json").read_text(encoding="utf-8"))
    rows = document["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("results.json must contain a nonempty rows list")
    index = {}
    for row in rows:
        if row["arm"] not in ARMS:
            raise ValueError(f"Unknown method: {row['arm']}")
        key = (str(row["uid"]), row["arm"])
        if key in index:
            raise ValueError(f"Duplicate object/method result: {key}")
        index[key] = row
    uids = sorted({uid for uid, _ in index})
    missing = [{"uid": uid, "arm": arm} for uid in uids for arm in ARMS if (uid, arm) not in index]
    unreadable = [{"uid": row["uid"], "arm": row["arm"], "status": row.get("status")}
                  for row in rows if _rms(row) is None]
    counts = {arm: {"expected": len(uids),
                    "present": sum((uid, arm) in index for uid in uids),
                    "metric_readable": sum(_rms(r) is not None for r in rows if r["arm"] == arm),
                    "qualified": sum(r.get("qualified") is True for r in rows if r["arm"] == arm),
                    "status_success": sum(r.get("status") == "success" for r in rows if r["arm"] == arm)}
              for arm in ARMS}
    cgal_rows = [row for row in rows if row["arm"] == "cgal_constrained"]
    cgal_faces = [int(row["actual_faces"]) for row in cgal_rows if row.get("actual_faces") is not None]
    cgal_at_512 = sum(count == 512 for count in cgal_faces)
    cgal_budget_note = (
        f"Constrained CGAL: {min(cgal_faces)}-{max(cgal_faces)} faces, only {cgal_at_512}/{len(uids)} at 512; NOT an equal-budget ranking."
        if cgal_faces else "Constrained CGAL face counts unavailable; not an equal-budget ranking."
    )
    footer = (f"All {len(uids)} objects; no qualified-only filter. "
              f"Unreadable RMS: {len(unreadable)} present attempts; missing records: {len(missing)}. "
              "Hollow markers retain nonqualified outputs. See REPORT.json for attempt counts.\n" + cgal_budget_note)
    directory = root / "figures"
    directory.mkdir(exist_ok=overwrite)
    artifacts = {}
    excluded_ratios = []

    time_unplotted, quality_unplotted = [], []
    if not contact_sheets_only:
        fig, ax = plt.subplots(figsize=(13, 6))
        offsets = np.linspace(-.25, .25, len(ARMS))
        for j, uid in enumerate(uids):
            fixed = _rms(index.get((uid, "fixed"), {}))
            if fixed is None or fixed <= 0:
                excluded_ratios.append({"uid": uid, "reason": "Fixed RMS is missing or nonpositive"})
                continue
            for offset, arm in zip(offsets, ARMS):
                row = index.get((uid, arm))
                if row is not None and _rms(row) is not None:
                    _draw_point(ax, j + offset, _rms(row) / fixed, row)
        ax.axhline(1., color="#334155", linewidth=1, linestyle="--")
        ax.set_xticks(range(len(uids)), [uid[:12] for uid in uids], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Surface RMS / same object's Fixed RMS (lower is better)")
        ax.set_title("Paired object comparison — reference surface and evaluation seed shared")
        ax.grid(axis="y", alpha=.2)
        _legend(ax)
        artifacts["paired_rms"] = _save(fig, directory, "paired_rms", footer)

        fig, ax = plt.subplots(figsize=(10, 7))
        time_unplotted = []
        for row in rows:
            x, y = _number(row.get("algorithm_seconds")), _rms(row)
            if x is not None and x > 0 and y is not None:
                _draw_point(ax, x, y, row)
            else:
                time_unplotted.append({"uid": row["uid"], "arm": row["arm"]})
            trajectory = row.get("trajectory") or []
            samples = [(_number(s.get("elapsed_seconds")), _rms(s)) for s in trajectory]
            samples = [(t, e) for t, e in samples if t is not None and t > 0 and e is not None]
            if len(samples) > 1:
                ax.plot([t for t, _ in samples], [e for _, e in samples], color=COLORS[row["arm"]],
                        alpha=.25, linewidth=1, marker=".", markersize=3)
        ax.set_xscale("log")
        ax.set_xlabel("Measured algorithm time (seconds, log scale)")
        ax.set_ylabel("Same-field symmetric surface RMS (% reference bbox diagonal)")
        ax.set_title("Surface error and online cost — thin lines are individual optimization trajectories")
        ax.grid(alpha=.2)
        _legend(ax)
        artifacts["rms_time"] = _save(fig, directory, "rms_time", footer + " Audit time is separate.")

        fig, ax = plt.subplots(figsize=(10, 7))
        quality_unplotted = []
        for row in rows:
            x, y = _angle(row), _rms(row)
            if x is not None and y is not None:
                _draw_point(ax, x, y, row)
            else:
                quality_unplotted.append({"uid": row["uid"], "arm": row["arm"]})
        ax.set_xlabel("5th percentile of per-triangle minimum 3D angles (degrees, higher is better)")
        ax.set_ylabel("Same-field symmetric surface RMS (% reference bbox diagonal)")
        ax.set_title("Surface fidelity and triangle-quality tail — no aggregate winner score")
        ax.grid(alpha=.2)
        _legend(ax)
        artifacts["rms_quality"] = _save(fig, directory, "rms_quality", footer)

    # The target and six methods share a camera and spatial scale per object.
    # Reference is display-only: six-method metric plots and denominators stay unchanged.
    mesh_errors = []
    display_arms = ("reference",) + ARMS
    for first in ([] if metrics_only else range(0, len(uids), 4)):
        page_uids = uids[first:first + 4]
        fig = plt.figure(figsize=(20, 3.1 * len(page_uids)))
        for local_i, uid in enumerate(page_uids):
            meshes = {}
            try:
                v, f = _load_mesh(_safe_mesh_path(root, Path("inputs") / f"{uid}.npz"))
                if not len(v) or not len(f) or not np.isfinite(v).all():
                    raise ValueError("empty or nonfinite reference geometry")
                meshes["reference"] = (v, f)
            except Exception as exc:
                mesh_errors.append({"uid": uid, "arm": "reference", "error": repr(exc)})
            for arm in ARMS:
                row = index.get((uid, arm), {})
                if row.get("mesh_file"):
                    try:
                        v, f = _load_mesh(_safe_mesh_path(root, row["mesh_file"]))
                        if not len(v) or not len(f) or not np.isfinite(v).all():
                            raise ValueError("empty or nonfinite geometry")
                        meshes[arm] = (v, f)
                    except Exception as exc:
                        mesh_errors.append({"uid": uid, "arm": arm, "error": repr(exc)})
            if meshes:
                all_v = np.vstack([v for v, _ in meshes.values()])
                centre = (all_v.min(0) + all_v.max(0)) * .5
                radius = max(float(np.ptp(all_v, axis=0).max()) * .55, 1e-10)
            for col, arm in enumerate(display_arms):
                ax = fig.add_subplot(len(page_uids), len(display_arms), local_i * len(display_arms) + col + 1, projection="3d")
                row = index.get((uid, arm), {})
                ax.set_axis_off()
                title = ("Reference F4096" if arm == "reference" else LABELS[arm]) if local_i == 0 else ""
                if arm in meshes:
                    v, f = meshes[arm]
                    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f,
                                    color="#b7c4d6" if arm == "reference" else COLORS[arm],
                                    edgecolor="#243040", linewidth=.15,
                                    antialiased=True, alpha=1., shade=True)
                    ax.set_xlim(centre[0] - radius, centre[0] + radius)
                    ax.set_ylim(centre[1] - radius, centre[1] + radius)
                    ax.set_zlim(centre[2] - radius, centre[2] + radius)
                    ax.set_box_aspect((1, 1, 1))
                    ax.view_init(elev=20, azim=-60, vertical_axis="y")
                    ax.set_proj_type("ortho")
                    if arm == "reference":
                        subtitle = f"F={len(f)}; shared target surface"
                    else:
                        rms, q05 = _rms(row), _angle(row)
                        subtitle = f"F={len(f)}; RMS={rms:.3g}%" if rms is not None else f"F={len(f)}; RMS unavailable"
                        if q05 is not None:
                            subtitle += f"; q05={q05:.1f} deg"
                        if row.get("qualified") is not True:
                            reasons = []
                            if row.get("budget_match") is False:
                                reasons.append("face budget")
                            if row.get("quality_pass") is False:
                                reasons.append("angle quality")
                            if (row.get("audit") or {}).get("valid") is False:
                                reasons.append("geometry validity")
                            if row.get("status") != "success":
                                reasons.append("execution status")
                            subtitle += "\nNOT QUALIFIED: " + ", ".join(reasons or ["qualification unavailable"])
                else:
                    subtitle = "NO READABLE MESH\n" + str(row.get("status", "missing"))
                    ax.text2D(.05, .45, subtitle, transform=ax.transAxes, fontsize=8)
                ax.set_title((title + "\n" if title else "") + subtitle, fontsize=8)
                if col == 0:
                    ax.text2D(-.05, .05, uid, transform=ax.transAxes, fontsize=6, rotation=90)
        fig.suptitle("Actual triangle meshes — same view and spatial scale within each object", fontsize=14, y=.985)
        mesh_footer = footer + " Qualification checks face budget, 3D angle quality and geometry validity; it does not mean only self-intersection."
        artifacts[f"meshes_{first // 4 + 1:02d}"] = _save(fig, directory, f"meshes_{first // 4 + 1:02d}", mesh_footer, top=.93)

    report = {"artifacts": artifacts, "object_order": uids, "counts": counts,
              "no_qualification_filter": True, "missing_records": missing,
              "unreadable_rms_records": unreadable, "ratio_exclusions": excluded_ratios,
              "time_unplotted": time_unplotted, "quality_unplotted": quality_unplotted,
              "mesh_read_errors": mesh_errors,
              "interpretation": "closed-set diagnostics; object pairing does not establish unseen generalization"}
    if not (metrics_only or contact_sheets_only):
        (directory / "REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
