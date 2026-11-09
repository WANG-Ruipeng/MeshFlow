"""CPU-only views of exact source-mesh subsets; no model or mesh generation.

The Y column is the source-parent complement M\\C, never a generated target.
All panels on one page use the same parent bounds and orthographic camera.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import textwrap
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SCHEMA = "chair_r2_user_conditions_render_v1"
COLORS = {"M": "#b8bec4", "R": "#51a795", "C": "#2387c9", "Y": "#dc9a60"}
CAMERAS = ((22.0, -60.0), (22.0, 120.0))
PARENT_CAMERAS = CAMERAS + ((65.0, -60.0), (0.0, 30.0))
DPI = 150


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8")


def _path(value):
    raw = str(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", raw):
        return Path("/mnt/" + raw[0].lower() + "/" + raw[3:].replace("\\", "/"))
    return Path(raw)


def _safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(value)):
        raise ValueError("Identifier contains path separators or unsupported characters")
    return str(value)


def _face_ids(values, n, name, allow_empty=False):
    a = np.asarray(values)
    if a.ndim != 1 or (a.size and not np.issubdtype(a.dtype, np.integer)):
        raise ValueError(f"{name} must be a one-dimensional integer source-face-ID list")
    a = a.astype(np.int64)
    if (not allow_empty and not len(a)) or len(set(a.tolist())) != len(a):
        raise ValueError(f"{name} must be nonempty and unique")
    if len(a) and (a.min() < 0 or a.max() >= n):
        raise ValueError(f"{name} has out-of-range retained source-face IDs")
    return np.sort(a)


def load_parent(parent):
    """Read one frozen NPZ, preserving stored FP32 triangles and source IDs."""
    path = _path(parent["npz"])
    actual_sha = sha256(path)
    if actual_sha != parent["sha256"]:
        raise ValueError(f"Parent source hash mismatch: {parent['uid']}")
    with np.load(path, allow_pickle=False) as data:
        raw = data["full_target"]
        if raw.dtype != np.float32 or raw.ndim != 2 or raw.shape[1] != 9:
            raise ValueError("Expected unchanged FP32 full_target[N,9]")
        full = raw.reshape(-1, 3, 3).copy()
        if "source_face_vertex_ids" in data and "model_vertices" in data:
            indexed = data["model_vertices"][data["source_face_vertex_ids"]]
            if not np.array_equal(indexed.view(np.uint32), full.view(np.uint32)):
                raise ValueError("Source corner coordinates differ from full_target")
    if len(full) != int(parent["N"]) or not np.isfinite(full).all():
        raise ValueError("Parent N/finite contract failed")
    lo = full.reshape(-1, 3).min(0).astype(float)
    hi = full.reshape(-1, 3).max(0).astype(float)
    span = float(np.max(hi - lo)) * 1.14
    if span <= 0:
        raise ValueError("Parent has no nonzero spatial extent")
    area = np.linalg.norm(np.cross(full[:, 1].astype(float) - full[:, 0], full[:, 2].astype(float) - full[:, 0]), axis=1) / 2
    return full, {"center_model_xyz": ((lo + hi) / 2).tolist(), "span": span,
                  "parent_bounds_model_xyz": [lo.tolist(), hi.tolist()], "projection": "orthographic",
                  "axis_transform": "model[x,y,z] -> plot[x,z,y] (model Y is vertical)",
                  "same_bounds_all_panels": True, "coordinate_rescaling": False}, area


def _shaded(tri, color):
    tri = np.asarray(tri, dtype=float)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normal, axis=1)
    normal = np.divide(normal, lengths[:, None], out=np.zeros_like(normal), where=lengths[:, None] > 0)
    light = np.array([.3, .7, .6]); light /= np.linalg.norm(light)
    # Absolute dot: original triangle winding is not a global orientation contract.
    strength = .70 + .30 * np.abs(normal @ light)
    return np.clip(np.asarray(to_rgb(color))[None, :] * strength[:, None], 0, 1)


def _draw(ax, triangles, colors, bounds, camera, face_ids=None):
    plot_tri = np.asarray(triangles, dtype=float)[:, :, [0, 2, 1]]
    if len(plot_tri):
        ax.add_collection3d(Poly3DCollection(plot_tri, facecolors=colors, edgecolors="#38424b",
                                            linewidths=.35, alpha=1, zsort="average"))
    center = np.asarray(bounds["center_model_xyz"])[[0, 2, 1]]
    half = bounds["span"] / 2
    ax.set(xlim=(center[0] - half, center[0] + half), ylim=(center[1] - half, center[1] + half),
           zlim=(center[2] - half, center[2] + half))
    ax.set_box_aspect((1, 1, 1)); ax.set_proj_type("ortho")
    ax.view_init(elev=camera[0], azim=camera[1]); ax.set_axis_off()
    if face_ids is not None:
        for face_id, point in zip(face_ids, plot_tri.mean(1)):
            ax.text(*point, str(int(face_id)), fontsize=5.5, color="#121b2a", ha="center", va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .6, "pad": .1})


def _identity(parent, kind, camera, extra):
    return {"schema": SCHEMA, "kind": kind, "uid": str(parent["uid"]), "split": parent.get("split"),
            "parent_npz": str(_path(parent["npz"])), "parent_sha256": parent["sha256"], "N": int(parent["N"]),
            "render_source_sha256": sha256(__file__), "camera": camera, "colors": COLORS,
            "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
            "dpi": DPI, "source_mesh_modified": False, "new_vertices": 0, "new_faces": 0,
            "model_calls": 0, "GPU": 0, "training": 0, "Y_definition": "exact source-parent complement M minus C; NOT generated output",
            **extra}


def _reusable(path, identity):
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.exists() and not sidecar.exists():
        return None
    if not path.exists() or not sidecar.exists():
        raise RuntimeError(f"Incomplete existing render, refusing overwrite: {path}")
    previous = read_json(sidecar)
    if previous.get("identity") != identity or previous.get("image_sha256") != sha256(path):
        raise RuntimeError(f"Existing render source/identity/image hash differs: {path}")
    return previous


def _save(fig, path, identity):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(temp, format="png", facecolor="white", dpi=DPI)
    plt.close(fig)
    if sha256(_path(identity["parent_npz"])) != identity["parent_sha256"]:
        raise RuntimeError("Parent source changed during rendering")
    temp.replace(path)
    result = {"identity": identity, "image": str(path), "image_sha256": sha256(path),
              "status": "COMPLETE", "AI_visual_review": "PENDING"}
    path.with_suffix(path.suffix + ".json").write_bytes(_json_bytes(result))
    return result


def render_condition(parent, region_ids, condition_ids, path, caption):
    """Render 2x5 M/R/C/Y/C+Y; caller assigns semantic regions, this function only displays."""
    path = Path(path); full, bounds, area = load_parent(parent); n = len(full)
    region = _face_ids(region_ids, n, "region_ids")
    condition = _face_ids(condition_ids, n, "condition_ids")
    if not set(condition) <= set(region) or len(condition) >= n:
        raise ValueError("Require nonempty C subset of R subset of M and a nonempty source complement")
    free = np.setdiff1d(np.arange(n), condition)
    identity = _identity(parent, "condition_2views_5columns", {**bounds, "views_elev_azim": CAMERAS},
                         {"region_source_face_ids": region.tolist(), "condition_source_face_ids": condition.tolist(),
                          "source_complement_face_ids": free.tolist(), "caption": str(caption),
                          "region_face_count": len(region), "condition_face_count": len(condition),
                          "condition_face_fraction_parent": len(condition) / n,
                          "condition_area_fraction_parent": float(area[condition].sum() / area.sum()),
                          "condition_area_fraction_region": float(area[condition].sum() / area[region].sum()),
                          "columns": ["M source parent", "R selected source region", "C input", "Y source complement", "C+Y source assembly"]})
    # JSON lists normalize tuples for exact resume identity checks.
    identity = json.loads(json.dumps(identity))
    old = _reusable(path, identity)
    if old: return old
    full_colors = _shaded(full, COLORS["Y"]); full_colors[condition] = _shaded(full[condition], COLORS["C"])
    panels = [(full, _shaded(full, COLORS["M"]), f"M | source parent | N={n}"),
              (full[region], _shaded(full[region], COLORS["R"]), f"R | selected region | {len(region)} faces"),
              (full[condition], _shaded(full[condition], COLORS["C"]), f"C | input | K={len(condition)} ({100*len(condition)/n:.1f}% faces)"),
              (full[free], _shaded(full[free], COLORS["Y"]), f"Y | SOURCE complement | {len(free)} faces"),
              (full, full_colors, "C + Y | source assembly")]
    fig = plt.figure(figsize=(20, 8.6), facecolor="white")
    for row, view in enumerate(CAMERAS):
        for col, (tri, colors, label) in enumerate(panels):
            ax = fig.add_subplot(2, 5, row * 5 + col + 1, projection="3d")
            _draw(ax, tri, colors, bounds, view); ax.set_title(label, fontsize=9)
    title = f"SOURCE DATA ONLY | {parent['uid']} | {caption}"
    fig.suptitle("\n".join(textwrap.wrap(title, 155)), fontsize=11, y=.989)
    fig.text(.5, .018, "BLUE = C | ORANGE = source remainder Y (NOT generated) | Same parent bounds in every panel | Two opposite views", ha="center", fontsize=10)
    fig.subplots_adjust(left=.003, right=.997, top=.89, bottom=.05, wspace=0, hspace=.02)
    return _save(fig, path, identity)


def render_parent_sheet(parent, regions, out, annotate_faces=False):
    """Four-view source sheet, optional face-ID selector, and separate region rows."""
    out = Path(out); uid = _safe_name(parent["uid"]); full, bounds, _ = load_parent(parent)
    normalized = []
    for index, region in enumerate(regions):
        ids = _face_ids(region["source_face_ids"], len(full), "region.source_face_ids", allow_empty=True)
        normalized.append({"region_id": str(region.get("region_id", index)), "label": str(region.get("label", "candidate region")),
                           "source_face_ids": ids.tolist(), "status": str(region.get("status", "CANDIDATE"))})
    pages = []
    for annotations in ([False, True] if annotate_faces else [False]):
        path = out / (uid + ("__face_ids.png" if annotations else "__parent.png"))
        identity = _identity(parent, "parent_face_ids" if annotations else "parent_selection", {**bounds, "views_elev_azim": PARENT_CAMERAS},
                             {"face_annotations": annotations, "face_id_domain": "retained row/source_face_ids"})
        identity = json.loads(json.dumps(identity)); old = _reusable(path, identity)
        if old: pages.append(old); continue
        fig = plt.figure(figsize=(18, 5.4), facecolor="white")
        for col, view in enumerate(PARENT_CAMERAS):
            ax = fig.add_subplot(1, 4, col + 1, projection="3d")
            _draw(ax, full, _shaded(full, COLORS["M"]), bounds, view, np.arange(len(full)) if annotations else None)
            ax.set_title(f"elev {view[0]:g} / azim {view[1]:g}", fontsize=10)
        fig.suptitle(f"SOURCE PARENT | {uid} | {parent.get('split', 'NA')} | N={len(full)}" +
                     (" | ALL FACE IDs (selection aid; labels may overlap/occlude)" if annotations else " | NO GENERATED MESH"), fontsize=11)
        fig.subplots_adjust(left=0, right=1, top=.91, bottom=.01, wspace=0)
        pages.append(_save(fig, path, identity))
    if normalized:
        path = out / (uid + "__regions.png")
        identity = _identity(parent, "region_selection", {**bounds, "views_elev_azim": CAMERAS}, {"regions": normalized})
        identity = json.loads(json.dumps(identity)); old = _reusable(path, identity)
        if old: pages.append(old)
        else:
            fig = plt.figure(figsize=(18, 4.6 * len(normalized) + .7), facecolor="white")
            for row, region in enumerate(normalized):
                ids = np.array(region["source_face_ids"], dtype=int)
                mixed = _shaded(full, COLORS["M"]); mixed[ids] = _shaded(full[ids], COLORS["R"])
                for col, (view, isolated) in enumerate(((CAMERAS[0], False), (CAMERAS[1], False), (CAMERAS[0], True), (CAMERAS[1], True))):
                    ax = fig.add_subplot(len(normalized), 4, row * 4 + col + 1, projection="3d")
                    _draw(ax, full[ids] if isolated else full, _shaded(full[ids], COLORS["R"]) if isolated else mixed, bounds, view)
                    label = f"#{row+1} {region['region_id']} | {region['label']} | {region['status']} | {len(ids)} faces"
                    ax.set_title(label + ("\nregion only" if isolated else "\nregion in source"), fontsize=9)
            fig.suptitle(f"REGION CANDIDATES | {uid} | Caller-supplied labels; visual review required | NO GENERATED MESH", fontsize=11)
            fig.subplots_adjust(left=0, right=1, top=.93, bottom=.01, wspace=0, hspace=.04)
            pages.append(_save(fig, path, identity))
    return {"status": "COMPLETE", "uid": uid, "pages": pages, "model_calls": 0}



def render_candidate_sheet(parent, candidates, path):
    """Display geometric patch partition without assigning semantic labels."""
    path = Path(path); full, bounds, area = load_parent(parent)
    seen = set(); rows = []; colors = _shaded(full, COLORS["M"])
    for index, candidate in enumerate(candidates):
        ids = _face_ids(candidate["source_face_ids"], len(full), "candidate.source_face_ids")
        if seen & set(ids): raise ValueError("Candidate partition contains overlapping source-face IDs")
        seen.update(ids)
        color = matplotlib.colors.hsv_to_rgb([(index * .618033988749895) % 1., .50, .88])
        colors[ids] = _shaded(full[ids], color)
        rows.append({"candidate_id": str(candidate["candidate_id"]), "source_face_ids": ids.tolist(), "K": len(ids),
                     "area": float(area[ids].sum()), "parent_area_fraction": float(area[ids].sum()/area.sum()),
                     "color_rgb": color.tolist(), "area_weighted_center_model_xyz": np.average(full[ids].mean(1), axis=0, weights=area[ids]).tolist()})
    ranked = sorted(rows, key=lambda row: (-row["area"], row["candidate_id"]))
    labeled = {row["candidate_id"] for row in ranked[:20]}
    identity = _identity(parent, "geometric_candidate_partition", {**bounds, "views_elev_azim": PARENT_CAMERAS},
                         {"candidates": rows, "unassigned_source_face_ids": sorted(set(range(len(full))) - seen),
                          "labeled_candidate_ids": sorted(labeled), "label_rule": "Largest20 by actual surface area; all patches rendered",
                          "semantic_labels": "NONE; geometric smooth-patch candidates only"})
    identity = json.loads(json.dumps(identity)); old = _reusable(path, identity)
    if old: return old
    legend_rows = max(1, int(np.ceil(len(rows)/4)))
    legend_height = .21 * legend_rows + .45; height = 5.3 + legend_height
    fig = plt.figure(figsize=(18, height), facecolor="white")
    for column, view in enumerate(PARENT_CAMERAS):
        ax = fig.add_axes((column*.25, legend_height/height, .25, 4.7/height), projection="3d")
        _draw(ax, full, colors, bounds, view)
        for row in rows:
            if row["candidate_id"] not in labeled: continue
            xyz = np.asarray(row["area_weighted_center_model_xyz"])[[0,2,1]]
            ax.text(*xyz, row["candidate_id"], fontsize=7, ha="center", va="center", color="#122031",
                    bbox={"facecolor":"white", "edgecolor":"none", "alpha":.8, "pad":.35})
        ax.set_title(f"elev {view[0]:g} / azim {view[1]:g}", fontsize=10)
    for index, row in enumerate(rows):
        col = index//legend_rows; rr = index % legend_rows
        label = f"{row['candidate_id']}  K={row['K']}  area={row['area']:.4g}  ({100*row['parent_area_fraction']:.1f}%)"
        yy = (legend_height - .28 - .21*rr) / height
        fig.text(.008+col*.25, yy, "■", color=row["color_rgb"], fontsize=11)
        fig.text(.020+col*.25, yy, label, fontsize=8)
    fig.suptitle(f"GEOMETRIC PATCH CANDIDATES | {parent['uid']} | N={len(full)} | {len(rows)} patches | NO SEAT/BACK CLAIM", fontsize=11, y=.989)
    return _save(fig, path, identity)


def render_parent_montage(parents, out):
    """Six parents per page, two views each; per-parent scale explicitly stated."""
    out = Path(out); pages = []
    for offset in range(0, len(parents), 6):
        group = parents[offset:offset+6]; loaded = [(p, *load_parent(p)[:2]) for p in group]
        path = out / f"parents_montage_{offset//6+1:03d}.png"
        identity = {"schema":SCHEMA, "kind":"six_parent_montage", "render_source_sha256":sha256(__file__),
                    "parents":[{"uid":p["uid"],"split":p.get("split"),"npz":str(_path(p["npz"])),"sha256":p["sha256"],"N":p["N"],"camera":bounds} for p,_,bounds in loaded],
                    "views_elev_azim":[list(v) for v in CAMERAS], "dpi":DPI,
                    "scale_contract":"Same bounds for the two views of each parent; each parent fitted separately, NOT a common cross-parent physical scale",
                    "matplotlib_version":matplotlib.__version__,"numpy_version":np.__version__,"model_calls":0,"GPU":0}
        old = _reusable(path,identity)
        if old: pages.append(old);continue
        fig = plt.figure(figsize=(18,8.4),facecolor="white")
        for i,(p,full,bounds) in enumerate(loaded):
            for j,view in enumerate(CAMERAS):
                ax=fig.add_subplot(2,6,(i//3)*6+(i%3)*2+j+1,projection="3d")
                _draw(ax,full,_shaded(full,COLORS["M"]),bounds,view)
                ax.set_title(f"{p['uid'][:12]}… | {p.get('split','NA')} | N={len(full)}\nview {j+1}",fontsize=9)
        fig.suptitle("SOURCE PARENTS ONLY | 6 parents/page | Each parent independently fitted; two views share its bounds",fontsize=11,y=.985)
        fig.subplots_adjust(left=0,right=1,top=.91,bottom=.01,wspace=0,hspace=.04)
        path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(".png.tmp")
        fig.savefig(temp,format="png",facecolor="white",dpi=DPI);plt.close(fig)
        for p,_,_ in loaded:
            if sha256(_path(p["npz"]))!=p["sha256"]:raise RuntimeError("Montage source changed")
        temp.replace(path)
        result={"status":"COMPLETE","identity":identity,"image":str(path),"image_sha256":sha256(path),"AI_visual_review":"PENDING"}
        path.with_suffix(".png.json").write_bytes(_json_bytes(result));pages.append(result)
    return {"status":"COMPLETE","parents":len(parents),"pages":pages,"model_calls":0}


def _rows(value, keys):
    if isinstance(value, list): return value
    for key in keys:
        if isinstance(value.get(key), list): return value[key]
    raise ValueError(f"Expected list or list under one of {keys}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("parents", "conditions", "candidates", "montage", "proposals"), required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--regions", type=Path)
    parser.add_argument("--conditions", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--uid", action="append", default=[])
    parser.add_argument("--annotate-faces", action="store_true")
    args = parser.parse_args()
    parents = {p["uid"]: p for p in _rows(read_json(args.inventory), ("rows", "parents"))}
    selected = set(args.uid) if args.uid else set(parents)
    if not selected <= set(parents): raise ValueError("UID outside registered inventory")
    regions = _rows(read_json(args.regions), ("rows", "regions")) if args.regions else []
    results = []
    if args.stage == "parents":
        for uid in sorted(selected):
            result = render_parent_sheet(parents[uid], [r for r in regions if r["uid"] == uid], args.out, args.annotate_faces)
            results.append(result); print(json.dumps({"uid": uid, "pages": len(result["pages"]), "status": "COMPLETE"}), flush=True)
    elif args.stage in ("candidates", "proposals"):
        if args.candidate_dir is None: parser.error("candidates stage requires --candidate-dir")
        if args.stage == "proposals":
            result = render_parent_montage([parents[uid] for uid in sorted(selected)], args.out)
            results.append(result)
            print(json.dumps({"stage":"montage", "parents":len(selected), "pages":len(result["pages"]), "status":"COMPLETE"}),flush=True)
        for uid in sorted(selected):
            source=args.candidate_dir/(uid+".json"); candidates=read_json(source)["candidates"]
            result=render_candidate_sheet(parents[uid],candidates,args.out/(_safe_name(uid)+"__candidates.png"))
            results.append(result);print(json.dumps({"uid":uid,"candidates":len(candidates),"status":"COMPLETE"}),flush=True)
    elif args.stage == "montage":
        result=render_parent_montage([parents[uid] for uid in sorted(selected)],args.out)
        results.append(result);print(json.dumps({"parents":len(selected),"pages":len(result["pages"]),"status":"COMPLETE"}),flush=True)
    else:
        if args.conditions is None: parser.error("conditions stage requires --conditions")
        conditions = _rows(read_json(args.conditions), ("rows", "conditions"))
        region_map = {(r["uid"], r["region_id"]): r for r in regions}
        for row in conditions:
            if row["uid"] not in selected: continue
            if row.get("status", "PASS") not in ("PASS", "ACCEPTED", "COMPLETE"):
                results.append({"uid": row["uid"], "condition_id": row["condition_id"], "status": "NA_NOT_RENDERED", "reason": row.get("reason")}); continue
            region_ids = row.get("region_source_face_ids")
            if region_ids is None: region_ids = region_map[(row["uid"], row["region_id"])]["source_face_ids"]
            path = args.out / (_safe_name(row["condition_id"]) + ".png")
            result = render_condition(parents[row["uid"]], region_ids, row["source_face_ids"], path, row.get("caption", row["condition_id"]))
            results.append(result); print(json.dumps({"condition_id": row["condition_id"], "status": "COMPLETE"}), flush=True)
    receipt = {"schema": SCHEMA, "status": "COMPLETE", "stage": args.stage, "inventory_sha256": sha256(args.inventory),
               "regions_sha256": sha256(args.regions) if args.regions else None, "conditions_sha256": sha256(args.conditions) if args.conditions else None,
               "render_source_sha256": sha256(__file__), "results": results, "torch_imported": "torch" in sys.modules, "model_calls": 0, "GPU": 0}
    args.out.mkdir(parents=True, exist_ok=True)
    receipt_path = args.out / ("render_" + args.stage + "_receipt.json")
    if receipt_path.exists() and receipt_path.read_bytes() != _json_bytes(receipt):
        raise RuntimeError("Refuse overwrite different batch receipt; use a distinct output directory")
    receipt_path.write_bytes(_json_bytes(receipt))


if __name__ == "__main__":
    main()
