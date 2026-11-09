"""Fixed 48-output CPU RAW/POST gallery; no GT, models, sampling or geometry edits.

The renderer reads post_index.csv after the parent runner has produced final
DENOISE_C_GEO_C meshes. Each request has four model rows and RAW/POST columns,
with identical camera and joint bounds across all eight panels. Matplotlib's
opaque polygon rasterization is a review aid, not a topology or attachment test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SCHEMA = "r2_postprocess_gallery_v1"
LABELS = ("R2_FROZEN", "R2_CONT_OLD", "R2_CONT_USER", "R2_USER_MIX")
CAMERAS = ((22.0, -60.0), (22.0, 120.0))
COLORS = {"C": "#2387c9", "F": "#b8bec4", "edges": "#38424b"}
DEFAULT_ROOT = "/mnt/e/MeshFlow/experiments/r2_postprocess_avmr_pmr_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_value(value):
    value = str(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    return Path(value)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def triangles(array, name):
    a = np.asarray(array)
    if a.ndim == 2 and a.shape[1:] == (9,):
        a = a.reshape(-1, 3, 3)
    if a.ndim != 3 or a.shape[1:] != (3, 3) or a.dtype.kind != "f" or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite float triangle array")
    return a.copy()


def load_mesh(path, k, stage):
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if stage == "RAW":
            if "raw" not in keys:
                raise ValueError(f"{path}: missing RAW array")
            mesh = triangles(data["raw"], "raw")
        elif {"vertices", "faces"} <= keys:
            vertices = np.asarray(data["vertices"])
            faces = np.asarray(data["faces"])
            if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
                raise ValueError(f"{path}: invalid indexed mesh shape")
            if not np.issubdtype(faces.dtype, np.integer) or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices))):
                raise ValueError(f"{path}: invalid vertex indices")
            mesh = triangles(vertices[faces], "indexed POST")
            if "output" in keys and not same_bits(mesh, triangles(data["output"], "output")):
                raise ValueError(f"{path}: indexed POST differs from output triangles")
        elif "output" in keys:
            mesh = triangles(data["output"], "output")
        elif {"C", "free"} <= keys:
            mesh = np.concatenate([triangles(data["C"], "C"), triangles(data["free"], "free")])
        else:
            raise ValueError(f"{path}: unsupported explicit POST schema: {sorted(keys)}")
        if k <= 0 or k > len(mesh):
            raise ValueError(f"{path}: invalid K")
        if "C" in keys and not same_bits(mesh[:k], triangles(data["C"], "C")):
            raise ValueError(f"{path}: C identity mismatch")
        if "known_faces" in keys and int(data["known_faces"]) != k:
            raise ValueError(f"{path}: K differs from stored known_faces")
        if "known_face_indices" in keys and not np.array_equal(np.asarray(data["known_face_indices"]), np.arange(k)):
            raise ValueError(f"{path}: C not in declared prefix order")
    return mesh


def shaded(tri, color):
    tri = np.asarray(tri, dtype=float)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normal, axis=1)
    normal = np.divide(normal, norms[:, None], out=np.zeros_like(normal), where=norms[:, None] > 0)
    light = np.asarray([.3, .7, .6]); light /= np.linalg.norm(light)
    factor = .70 + .30 * np.abs(normal @ light)
    return np.clip(np.asarray(to_rgb(color))[None, :] * factor[:, None], 0, 1)


def draw(ax, mesh, k, bounds, camera):
    color = shaded(mesh, COLORS["F"])
    color[:k] = shaded(mesh[:k], COLORS["C"])
    ax.add_collection3d(Poly3DCollection(mesh[:, :, [0, 2, 1]].astype(float), facecolors=color,
                                        edgecolors=COLORS["edges"], linewidths=.35, alpha=1., zsort="average"))
    center = np.asarray(bounds["center_model_xyz"])[[0, 2, 1]]
    half = bounds["span"] / 2
    ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half), zlim=(center[2]-half, center[2]+half))
    ax.set_proj_type("ortho"); ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=camera[0], azim=camera[1]); ax.set_axis_off()


def render_request(rows, out):
    request_id = rows[0]["request_id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", request_id):
        raise ValueError("Unsafe request identifier")
    loaded = []
    source_hashes = {}
    shared_c = None
    for label in LABELS:
        row = next(row for row in rows if row["label"] == label)
        k = int(row["K"])
        raw_path, post_path = path_value(row["raw_path"]), path_value(row["post_path"])
        for path in (raw_path, post_path):
            source_hashes[str(path)] = sha256(path)
        raw, post = load_mesh(raw_path, k, "RAW"), load_mesh(post_path, k, "POST")
        if not same_bits(raw[:k], post[:k]):
            raise ValueError(f"C changed across stages: {label}/{request_id}")
        if shared_c is None:
            shared_c = raw[:k]
        elif not same_bits(shared_c, raw[:k]):
            raise ValueError(f"C differs across labels: {request_id}")
        loaded.append((row, raw, post, k))
    all_vertices = np.concatenate([mesh.reshape(-1, 3) for _, raw, post, _ in loaded for mesh in (raw, post)]).astype(float)
    lo, hi = all_vertices.min(0), all_vertices.max(0)
    span = float(np.max(hi-lo)) * 1.12
    if span <= 0:
        raise ValueError("All panels have zero spatial extent")
    bounds = {"bounds_model_xyz": [lo.tolist(), hi.tolist()], "center_model_xyz": ((lo+hi)/2).tolist(),
              "span": span, "coordinate_rescaling": False, "same_bounds_all_8_panels": True,
              "axis_transform": "model[x,y,z] -> plot[x,z,y]; model Y vertical", "projection": "orthographic"}
    receipts = []
    for view_id, camera in enumerate(CAMERAS):
        path = out / f"{request_id}__v{view_id}.png"
        identity = {"schema": SCHEMA, "request_id": request_id, "parent_id": rows[0]["parent_id"],
                    "region_type": rows[0]["region_type"], "stage": "DENOISE_C_GEO_C", "labels": list(LABELS),
                    "bounds": bounds, "camera_elev_azim": list(camera), "source_hashes": source_hashes,
                    "source_code_sha256": sha256(__file__), "matplotlib_version": matplotlib.__version__,
                    "numpy_version": np.__version__, "colors": COLORS, "pixels": [1400,1600],
                    "hidden_triangles_removed": False, "GT_read": False, "model_calls": 0, "GPU": 0}
        sidecar = path.with_suffix(".png.json")
        if path.exists() or sidecar.exists():
            if not path.exists() or not sidecar.exists():
                raise RuntimeError(f"Incomplete existing image receipt: {path}")
            existing = json.loads(sidecar.read_text(encoding="utf8"))
            if existing["identity"] != identity or existing["sha256"] != sha256(path):
                raise RuntimeError(f"Existing gallery identity mismatch: {path}")
            receipts.append(existing)
            continue
        fig = plt.figure(figsize=(14,16), dpi=100, facecolor="white")
        for i, (row, raw, post, k) in enumerate(loaded):
            for j, (stage, mesh) in enumerate((("RAW",raw),("DENOISE_C_GEO_C",post))):
                ax = fig.add_subplot(4,2,i*2+j+1,projection="3d")
                draw(ax, mesh, k, bounds, camera)
                ax.set_title(f"{row['label']} | {stage}\nN={len(mesh)} | C={k}, F={len(mesh)-k}", fontsize=10, pad=0)
        noise = request_id.rsplit("__", 1)[-1]
        fig.suptitle(f"RAW vs final DENOISE_C_GEO_C | {rows[0]['parent_id']}\n{rows[0]['region_type']} | {noise} | view {view_id+1}: elev {camera[0]:g}, azim {camera[1]:g}", fontsize=12, y=.985)
        fig.text(.5,.015,"BLUE = unchanged C | GRAY = generated F | All 8 panels: same camera and bounds; no geometry normalization",ha="center",fontsize=9)
        fig.text(.5,.005,"Opaque surfaces and edges; C may be occluded. Pictures do not certify attachment or shared topology.",ha="center",fontsize=8)
        fig.subplots_adjust(left=.02,right=.98,top=.945,bottom=.033,hspace=.03,wspace=0)
        temporary = path.with_suffix(".png.tmp")
        fig.savefig(temporary,format="png",dpi=100,facecolor="white")
        plt.close(fig)
        for source, expected in source_hashes.items():
            if sha256(source) != expected:
                raise RuntimeError(f"Source changed while rendering: {source}")
        temporary.replace(path)
        receipt = {"status":"COMPLETE","identity":identity,"image":str(path),"sha256":sha256(path),"visual_review":"PENDING_AI_PROXY"}
        json_write(sidecar,receipt)
        receipts.append(receipt)
    return receipts


def artifact_link(path, gallery_dir, label):
    path = path_value(path)
    if not path.exists():
        return f"{html.escape(label)}: unavailable"
    href = Path(os.path.relpath(path, gallery_dir)).as_posix()
    return f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>'


def make_html(out, rows, receipts):
    blocks = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>R2 RAW / POST review</title>',
        '<style>body{font:15px Arial,sans-serif;margin:24px;max-width:1450px}img{width:49%;vertical-align:top}table{border-collapse:collapse}td,th{padding:5px 12px;border:1px solid #ddd}section{margin:35px 0}code{background:#eee}a{color:#176299}</style>',
        '<h1>R2 RAW / DENOISE_C_GEO_C</h1>',
        '<p>48 existing RAW outputs; 12 fixed requests; all four labels included. Two opposite views per request. '
        'Blue: unchanged C; gray: generated F. Same camera and joint RAW/POST bounds across each request. '
        'No source-parent GT, geometry rescaling, clipping or hidden-face removal. '
        'These are opaque CPU review renders, not human ratings or proof of attachment.</p>',
        '<p>POST is the C-preserving adaptation of the official denoiser followed by GEO_C; '
        'it must not be read as unmodified official unconditional postprocessing. '
        '<a href="../post_index.csv">Input index</a> | <a href="gallery_manifest.json">Render provenance</a></p>']
    for request_id in sorted({r["request_id"] for r in rows}):
        rr = [r for r in rows if r["request_id"] == request_id]
        blocks.append(f'<section><h2>{html.escape(request_id)}</h2><table><tr><th>Model</th><th>RAW</th><th>Final POST</th></tr>')
        for label in LABELS:
            row = next(r for r in rr if r["label"] == label)
            post = path_value(row["post_path"])
            raw = path_value(row["raw_path"])
            post_ply = row.get("post_ply_path") or row.get("post_ply") or str(post.with_suffix(".ply"))
            if not path_value(post_ply).exists() and (post.parent/"output.ply").exists():
                post_ply = str(post.parent/"output.ply")
            blocks.append(f'<tr><td>{label}</td><td>{artifact_link(raw,out,"NPZ")}</td><td>{artifact_link(post,out,"NPZ")} | {artifact_link(post_ply,out,"PLY")}</td></tr>')
        blocks.append('</table>')
        for view_id in range(2):
            name = f"{request_id}__v{view_id}.png"
            blocks.append(f'<a href="{name}"><img src="{name}" alt="{html.escape(request_id)} view {view_id+1}"></a>')
        blocks.append('</section>')
    blocks.append('</html>')
    (out/"index.html").write_text("\n".join(blocks),encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default=DEFAULT_ROOT)
    parser.add_argument("--check-only",action="store_true",help="Check imports and fixed rendering interface, no data/model reads or rendered meshes")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps({"status":"PASS_INTERFACE_IMPORTS","schema":SCHEMA,"matplotlib":matplotlib.__version__,"numpy":np.__version__,"CPU_threads":1,"model_calls":0,"renders":0}))
        return
    started = time.monotonic()
    root = path_value(args.root)
    index = root/"post_index.csv"
    with index.open(encoding="utf-8-sig",newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"label","request_id","parent_id","region_type","raw_path","post_path","K"}
    if len(rows) != 48 or any(not required <= row.keys() for row in rows):
        raise ValueError("Expected fixed 48-row POST index with explicit source identities")
    request_ids = sorted({r["request_id"] for r in rows})
    if len(request_ids) != 12 or {(r["request_id"],r["label"]) for r in rows} != {(q,l) for q in request_ids for l in LABELS}:
        raise ValueError("Request / label panel differs from fixed 12 x 4 contract")
    out = root/"gallery"; out.mkdir(parents=True,exist_ok=True)
    index_sha = sha256(index)
    receipts = []
    for request_id in request_ids:
        rr = [r for r in rows if r["request_id"] == request_id]
        if len({(r["parent_id"],r["region_type"],int(r["K"])) for r in rr}) != 1:
            raise ValueError("Request C identity inconsistent across models")
        receipts.extend(render_request(rr,out))
        print(json.dumps({"request_id":request_id,"renders_complete":len(receipts)}),flush=True)
    if sha256(index) != index_sha:
        raise RuntimeError("POST index changed during rendering")
    make_html(out,rows,receipts)
    manifest = {"schema":SCHEMA,"status":"COMPLETE","requests":12,"labels":list(LABELS),"input_meshes":48,
                "PNG_count":len(receipts),"stage":"DENOISE_C_GEO_C","index_sha256":index_sha,
                "source_code_sha256":sha256(__file__),"wall_seconds":time.monotonic()-started,"CPU_workers":1,
                "CPU_threads":1,"GPU":0,"model_calls":0,"GT_read":False,"source_geometry_modified":False,
                "visual_review_status":"PENDING_AI_PROXY","receipts":receipts}
    json_write(out/"gallery_manifest.json",manifest)
    print(json.dumps({"status":"COMPLETE","images":len(receipts),"wall_seconds":manifest["wall_seconds"]}),flush=True)


if __name__ == "__main__":
    main()