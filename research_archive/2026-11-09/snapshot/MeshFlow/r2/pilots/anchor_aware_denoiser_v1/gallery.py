"""CPU views of all fixed RAW/D0/continued/anchor requests; no GT or model read.

Three opaque whole-object views plus an opposite wireframe per request. The
retained triangles, camera, material, and plotting bounds are never selected
using scores. Local risk views are extra and never replace whole-object views.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image

try:
    from .common import OUT, OLD, sha, read, checked, atomic, write, csvwrite, textfile
except ImportError:
    from common import OUT, OLD, sha, read, checked, atomic, write, csvwrite, textfile

VERSION = "anchor_aware_denoiser_gallery_v1"
STAGES = ("RAW", "D0_OFFICIAL", "D_CONT_NO_ROLE", "D_ANCHOR")
CAMERAS = ((12., -90.), (12., 0.), (24., -60.), (24., 120.))
VIEW_NAMES = ("Front", "Side", "Oblique", "Opposite oblique: wireframe")
COLORS = {"C": "#2387c9", "F": "#b8bec4", "edges": "#38424b"}
GALLERY = OUT / "gallery"
RISK_PREFIXES = ("33ec", "866e", "534b", "f596")


def equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def triangles(a):
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1:] == (9,):
        a = a.reshape(-1, 3, 3)
    if a.ndim != 3 or a.shape[1:] != (3, 3) or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError("Expected finite FP32 triangle array")
    return a.copy()


def mesh(path, k, raw=False):
    with np.load(path, allow_pickle=False) as z:
        result = triangles(z["raw"] if raw else z["output"])
        if not raw:
            assert equal(triangles(z["vertices"][z["faces"]]), result)
        if "C" in z:
            assert equal(result[:k], triangles(z["C"]))
        assert 0 < k < len(result)
    return result


def shades(tri, color):
    t = tri.astype(np.float64)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    length = np.linalg.norm(n, axis=1)
    n = np.divide(n, length[:, None], out=np.zeros_like(n), where=length[:, None] > 0)
    light = np.array([.3, .7, .6]); light /= np.linalg.norm(light)
    return np.clip(to_rgb(color) * (.70 + .30 * np.abs(n @ light))[:, None], 0, 1)


def draw(ax, tri, k, bounds, camera, wire=False):
    plotted = tri[:, :, [0, 2, 1]].astype(np.float64)
    if wire:
        segments = plotted[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2, 3)
        colors = np.repeat(np.array([to_rgb(COLORS["C"])] * k + [to_rgb(COLORS["edges"])] * (len(tri)-k)), 3, axis=0)
        ax.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=.44, alpha=.85))
    else:
        color = shades(tri, COLORS["F"])
        color[:k] = shades(tri[:k], COLORS["C"])
        ax.add_collection3d(Poly3DCollection(plotted, facecolors=color, edgecolors=COLORS["edges"], linewidths=.30, alpha=1., zsort="average"))
    center = np.asarray(bounds["center_model_xyz"])[[0, 2, 1]]
    half = bounds["span_model_units"] / 2
    ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half), zlim=(center[2]-half, center[2]+half))
    ax.set_proj_type("ortho"); ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=camera[0], azim=camera[1]); ax.set_axis_off()


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("Unsafe filename identifier")
    return value


def render_sheet(group, local=False):
    first = group[0]
    generator, rid, k = first["label"], first["request_id"], int(first["K"])
    safe_id(generator); safe_id(rid)
    rows = {r["denoiser"]: r for r in group}
    assert set(rows) == set(STAGES[1:])
    paths = {"RAW": Path(first["raw_path"])}
    paths.update({s: Path(rows[s]["post_path"]) for s in STAGES[1:]})
    sources = {str(p): sha(p) for p in paths.values()}
    meshes = {s: mesh(p, k, s == "RAW") for s, p in paths.items()}
    assert all(equal(m[:k], meshes["RAW"][:k]) for m in meshes.values())
    vertices = np.concatenate([m.reshape(-1, 3) for m in meshes.values()]).astype(np.float64)
    lo, hi = vertices.min(0), vertices.max(0)
    global_span = float(np.max(hi-lo)) * 1.10
    center = (lo+hi)/2
    if local:
        c = meshes["RAW"][:k].reshape(-1, 3).astype(np.float64)
        center = (c.min(0)+c.max(0))/2
        span = max(float(np.max(c.max(0)-c.min(0))) * 1.35, global_span * .35)
    else:
        span = global_span
    assert span > 0
    bounds = dict(center_model_xyz=center.tolist(), span_model_units=span,
        full_joint_bounds_model_xyz=[lo.tolist(), hi.tolist()],
        same_all_stages=True, local_plot_crop=local, geometry_clipping_or_rescaling=False)
    views = (("C vicinity: oblique", CAMERAS[2]), ("C vicinity: opposite wireframe", CAMERAS[3])) if local else tuple(zip(VIEW_NAMES, CAMERAS))
    name = f"{generator}__{rid}" + ("__local" if local else "__full")
    target = checked(GALLERY / (name + ".png"))
    jpg = checked(GALLERY / "review_jpg" / (name + ".jpg"))
    receipt_path = checked(GALLERY / (name + ".json"))
    pixels = [2000, 980 if local else 1800]
    identity = dict(schema=VERSION, generator=generator, request_id=rid, parent_id=first["parent_id"],
        condition_id=first["condition_id"], region_type=first["region_type"], tier=first["tier"],
        source_hashes=sources, code_sha256=sha(__file__), K=k, stages=list(STAGES),
        N_by_stage={s:len(m) for s,m in meshes.items()}, cameras=[list(v[1]) for v in views],
        view_names=[v[0] for v in views], bounds=bounds, colors=COLORS, pixels=pixels,
        axis_transform="model[x,y,z] -> plot[x,z,y]; model Y vertical", projection="orthographic",
        hidden_triangles_removed=False, GT_read=False, model_calls=0, CUDA_calls=0,
        picture_does_not_certify_attachment=True, C_bitwise_preserved=True,
        matplotlib_version=matplotlib.__version__, numpy_version=np.__version__)
    if receipt_path.exists():
        old = read(receipt_path)
        assert old["identity"] == identity
        assert sha(target) == old["png_sha256"] and sha(jpg) == old["jpg_sha256"]
        return old
    if target.exists() or jpg.exists():
        raise RuntimeError("Incomplete gallery render requires inspection: " + name)
    fig = plt.figure(figsize=(pixels[0]/100, pixels[1]/100), dpi=100, facecolor="white")
    for j, (view_name, camera) in enumerate(views):
        for i, stage in enumerate(STAGES):
            ax = fig.add_subplot(len(views), 4, j*4+i+1, projection="3d")
            draw(ax, meshes[stage], k, bounds, camera, "wireframe" in view_name.lower())
            ax.set_title((f"{stage}\nN={len(meshes[stage])} C={k} F={len(meshes[stage])-k}\n" if j == 0 else "") + view_name,
                fontsize=10, pad=0)
    fig.suptitle(f"{generator} | {first['parent_id']} | {first['region_type']} {first['tier']} | {rid.rsplit('__',1)[-1]}\nSame RAW input; one denoiser call + fixed GEO_C per POST" + (" | EXTRA LOCAL VIEW" if local else ""), fontsize=13, y=.995)
    fig.text(.5, .015, "BLUE = unchanged C | GRAY = free | Same camera / bounds / material across stages | AI_PROXY review; images do not certify connectivity", ha="center", fontsize=10)
    fig.subplots_adjust(left=.003, right=.997, bottom=.028, top=.947 if not local else .90, hspace=.02, wspace=0)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, facecolor="white"); plt.close(fig)
    image_bytes = buf.getvalue()
    atomic(target, lambda f: f.write(image_bytes), True)
    # Compact review package copy; full-resolution PNG stays local.
    compact = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    compact.thumbnail((1500, 1350))
    jb = io.BytesIO(); compact.save(jb, format="JPEG", quality=82, optimize=True, subsampling=0)
    atomic(jpg, lambda f: f.write(jb.getvalue()), True)
    assert all(sha(p) == h for p,h in sources.items())
    receipt = dict(status="COMPLETE", identity=identity, png=str(target), png_sha256=sha(target),
        jpg=str(jpg), jpg_sha256=sha(jpg), visual_review="PENDING_AI_PROXY")
    write(receipt_path, receipt)
    return receipt


def make_html(groups, receipts):
    by_key = defaultdict(list)
    for r in receipts:
        by_key[(r["identity"]["generator"], r["identity"]["request_id"])].append(r)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Anchor-aware denoiser pilot</title>',
        '<style>body{font:15px Arial;margin:24px;max-width:2000px}img{width:100%;height:auto}section{margin:32px 0}a{color:#176299}</style>',
        '<h1>RAW / D0_OFFICIAL / D_CONT_NO_ROLE / D_ANCHOR</h1>',
        '<p>48 fixed requests, all 144 POST outputs. Three solid whole-object views and opposite wireframe per request; additional local views only for the eight predeclared USER_MIX risk requests. No GT or source-target ranking.</p>',
        '<p>Blue C, gray free. Same bounds/camera/material across stages; no hidden-face deletion or coordinate normalization. C preservation and appearance do not establish topological attachment. AI_PROXY is not blind human evaluation.</p>',
        '<p><a href="../REPORT.md">Report</a> | <a href="gallery_manifest.json">Provenance</a> | <a href="AI_PROXY_REVIEW.md">Visual observations</a></p>']
    for key, rows in sorted(groups.items()):
        parts.append(f'<section><h2>{html.escape(key[0])} — {html.escape(key[1])}</h2>')
        for r in by_key[key]:
            name = Path(r["png"]).name
            parts.append(f'<a href="{html.escape(name)}"><img loading="lazy" src="{html.escape(name)}"></a>')
        for row in rows:
            href = os.path.relpath(row["post_ply"], GALLERY).replace(os.sep, "/")
            parts.append(f'<a href="{html.escape(href)}">{html.escape(row["denoiser"])} PLY</a> ')
        parts.append('</section>')
    parts.append('</html>')
    textfile(GALLERY / "index.html", "\n".join(parts))


def manifest_receipt(receipt):
    """Hash the persisted sidecar separately; do not put a self-hash in it."""
    path = Path(receipt["png"]).with_suffix(".json")
    assert read(path) == receipt
    return dict(receipt, receipt_path=str(path), receipt_sha256=sha(path))


def verified_completed_gallery():
    """Strict read-only resume: completed gallery never rewrites timing/index files.

    Added before the first real render to preserve finalization prerequisite
    hashes when run.py --stage all is entered again after task completion.
    """
    path = GALLERY / "gallery_manifest.json"
    if not path.exists():
        return False
    manifest = read(path)
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("Existing gallery manifest is incomplete; inspect without rewriting")
    if manifest.get("schema") != VERSION or manifest.get("code_sha256") != sha(__file__):
        raise RuntimeError("Completed gallery source identity differs")
    if manifest.get("post_index_sha256") != sha(OUT / "evaluation/post_index.csv"):
        raise RuntimeError("Completed gallery POST index identity differs")
    if (manifest.get("input_requests"), manifest.get("POST_outputs"), manifest.get("images"),
            manifest.get("whole_object_sheets"), manifest.get("local_sheets")) != (48, 144, 56, 48, 8):
        raise RuntimeError("Completed gallery count contract differs")
    source_hashes = {}
    pngs, jpegs, sides = set(), set(), set()
    full_keys, local_keys = set(), set()
    receipt_rows = manifest.get("receipts", [])
    if len(receipt_rows) != 56:
        raise RuntimeError("Completed gallery missing image receipts")
    for record in receipt_rows:
        identity = record["identity"]
        if record.get("status") != "COMPLETE" or identity.get("code_sha256") != sha(__file__):
            raise RuntimeError("Completed render receipt/source identity differs")
        png, jpg, side = (checked(Path(record[key])) for key in ("png", "jpg", "receipt_path"))
        if png.parent != GALLERY or jpg.parent != GALLERY / "review_jpg" or side != png.with_suffix(".json"):
            raise RuntimeError("Completed gallery receipt path differs")
        if png in pngs or jpg in jpegs or side in sides:
            raise RuntimeError("Duplicate completed gallery image/receipt")
        pngs.add(png); jpegs.add(jpg); sides.add(side)
        if sha(png) != record["png_sha256"] or sha(jpg) != record["jpg_sha256"] or sha(side) != record["receipt_sha256"]:
            raise RuntimeError("Completed gallery PNG/JPEG/sidecar hash differs")
        side_record = {k: v for k, v in record.items() if k not in ("receipt_path", "receipt_sha256")}
        if read(side) != side_record:
            raise RuntimeError("Completed gallery sidecar content differs from manifest")
        for src, expected in identity["source_hashes"].items():
            if src in source_hashes and source_hashes[src] != expected:
                raise RuntimeError("Conflicting completed render source hashes")
            source_hashes[src] = expected
        key = (identity["generator"], identity["request_id"])
        target_keys = local_keys if identity["bounds"]["local_plot_crop"] else full_keys
        if key in target_keys:
            raise RuntimeError("Duplicate whole/local completed request identity")
        target_keys.add(key)
    oldrows = read(OLD / "registration.json")["records"]
    expected_full = {(r["label"], r["request_id"]) for r in oldrows}
    expected_local = {(r["label"], r["request_id"]) for r in oldrows
        if r["label"] == "R2_USER_MIX" and r["parent_id"].startswith(RISK_PREFIXES)}
    if full_keys != expected_full or local_keys != expected_local:
        raise RuntimeError("Completed gallery request coverage differs")
    for src, expected in source_hashes.items():
        if sha(src) != expected:
            raise RuntimeError("Completed gallery mesh source changed: " + src)
    support = manifest.get("support_file_hashes", {})
    if set(support) != {str(GALLERY / "index.html"), str(GALLERY / "review_pending.csv")}:
        raise RuntimeError("Completed gallery index/pending-template provenance missing")
    for src, expected in support.items():
        if sha(src) != expected:
            raise RuntimeError("Completed gallery support artifact changed: " + src)
    print(json.dumps(dict(status="SKIP_VERIFIED_COMPLETE", images=56, POST_outputs=144,
        PNG_JPEG_sidecars_verified=56, source_meshes_verified=len(source_hashes),
        artifact_writes=0, new_renders=0, model_calls=0)), flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if args.check_only:
        print(json.dumps(dict(status="PASS_IMPORTS_CPU_ONLY", renders=0, model_calls=0, CUDA_calls=0)))
        return
    if verified_completed_gallery():
        return
    start = time.monotonic()
    index = OUT / "evaluation/post_index.csv"
    index_hash = sha(index)
    with index.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 144
    groups = defaultdict(list)
    for row in rows:
        assert sha(row["post_path"]) == row["post_sha256"]
        groups[(row["label"], row["request_id"])].append(row)
    assert len(groups) == 48
    oldreg = read(OLD / "registration.json")
    assert set(groups) == {(r["label"], r["request_id"]) for r in oldreg["records"]}
    receipts = []
    for key, group in sorted(groups.items()):
        receipts.append(manifest_receipt(render_sheet(group)))
        if key[0] == "R2_USER_MIX" and group[0]["parent_id"].startswith(RISK_PREFIXES):
            receipts.append(manifest_receipt(render_sheet(group, local=True)))
        print(json.dumps(dict(generator=key[0], request_id=key[1], completed_images=len(receipts))), flush=True)
    assert len(receipts) == 56
    assert sha(index) == index_hash
    make_html(groups, receipts)
    pending = [dict(denoiser=r["denoiser"], generator=r["label"], request_id=r["request_id"],
        parent_id=r["parent_id"], region_type=r["region_type"], tier=r["tier"],
        review_kind="AI_PROXY", review_status="PENDING", C_use="NOT_REVIEWED", boundary_continuation="NOT_REVIEWED",
        overall_quality="NOT_REVIEWED", thin_structure="NOT_REVIEWED", observations="", blind=False,
        figure=f"{r['label']}__{r['request_id']}__full.png") for r in rows]
    csvwrite(GALLERY / "review_pending.csv", pending)
    write(GALLERY / "gallery_manifest.json", dict(schema=VERSION, status="COMPLETE", input_requests=48,
        POST_outputs=144, stages=list(STAGES), whole_object_sheets=48, local_sheets=8, images=len(receipts),
        code_sha256=sha(__file__), post_index_sha256=index_hash, wall_seconds=time.monotonic()-start,
        CPU_workers=1, numeric_threads=1, GPU_calls=0, model_calls=0, GT_read=False,
        visual_review="PENDING_AI_PROXY", receipts=receipts,
        support_file_hashes={str(p): sha(p) for p in (GALLERY / "index.html", GALLERY / "review_pending.csv")}))
    print(json.dumps(dict(status="COMPLETE", images=len(receipts), POST_outputs=144, wall_seconds=time.monotonic()-start)), flush=True)


if __name__ == "__main__":
    main()
