"""Actual new output comparisons at inherited Bench cameras and cut positions."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse, concurrent.futures, html, io, multiprocessing, resource, time
from pathlib import Path
import numpy as np
from .eval_plan import OUT, OLD, P, ALL_LABELS, requests, read, sha, verify, root_for
from r2.pilots.chair_multipart_seam_v1 import gallery as G
from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
C, D, H = G.C, G.D, G.H


def cell(q, label, stage):
    root = root_for(label)
    if stage == "CLEAN_FINAL_POST":
        folder = root / "cpu" / label / q["request_id"]
        mesh, receipt = folder / stage / "post.npz", folder / "complete.json"
    else:
        kind = "raw" if stage == "RAW" else "post"
        folder = root / "evaluation" / label / q["request_id"] / kind
        mesh, receipt = folder / (kind + ".npz"), folder / "receipt.json"
    return dict(model=label, label=label + " " + stage, stage=stage,
                path=str(mesh), receipt=str(receipt), request_id=q["request_id"])


def prepare():
    definition = OLD / "gallery/NATIVE_S21000__SEAM_S21000_full/definition.json"
    prior = read(definition)
    pages = []
    for q in requests():
        specs = [(s, [(m, s) for m in ALL_LABELS])
                 for s in ("RAW", "BASE_POST", "CLEAN_FINAL_POST")]
        for sheet, cells in specs:
            pages.append(dict(**{k: q[k] for k in ("request_id", "condition_id", "parent_id", "noise_index", "N", "K", "split", "part_type")},
                              sheet=sheet, page_id=sheet + "__" + q["request_id"],
                              cells=[cell(q, m, s) for m, s in cells], inference_npz=q["inference_npz"],
                              bounds=prior["bounds"][q["condition_id"]],
                              depth_bounds=prior["depth_bounds"][q["condition_id"]]))
    protected = [definition, OUT / "eval_manifest.json", OUT / "registration.json",
                 Path(__file__), Path(G.__file__), Path(D.__file__), Path(H.__file__)]
    value = dict(status="FROZEN", pages=pages, requests=48, sheets=len(pages),
                 protected={str(p): sha(p) for p in protected},
                 cameras="Original azimuth -60/120 elevation23, inherited center/span, no per-output fit",
                 sections="Original central front/side finite section position and bounds, identical across stages",
                 role_colors="Blue C, orange SEAM, gray BULK, green added BRIDGE",
                 review="PENDING_ACTUAL_NONBLIND_AI_PROXY", GT_display=False, model_calls=0)
    P.immutable(OUT / "gallery/manifest.json", value)
    return value


def one(job):
    page, dh = job
    out = OUT / "gallery"
    png = out / (page["page_id"] + ".png")
    receipt = png.with_suffix(".json")
    if receipt.exists():
        r = read(receipt)
        if r["definition_sha256"] != dh or sha(png) != r["image_sha256"]:
            raise ValueError("Saved gallery identity changed")
        return r
    tick = time.perf_counter()
    ncols = len(page["cells"])
    plt = H._mpl()
    fig = plt.figure(figsize=(5.3 * ncols, 13.5), dpi=110, facecolor="white")
    grid = fig.add_gridspec(4, ncols, left=.025, right=.99, bottom=.055, top=.91, wspace=.15, hspace=.23)
    a = C.arrays(page["inference_npz"])
    original = a["C_vertices"][a["C_faces_original"]]
    K, bounds, depth = page["K"], page["bounds"], page["depth_bounds"]
    display = []
    for col, cell_info in enumerate(page["cells"]):
        tri, role = G.roles(C.arrays(cell_info["path"]), K)
        g = dict(C=tri[:K], F=tri[K:], C_original=original, tri=tri)
        facts = dict(label=cell_info["label"], role_counts={r: int((role == r).sum()) for r in ("C", "SEAM", "BULK", "BRIDGE")}, views=[])
        for row, az in enumerate((-60, 120)):
            raster, scope = G.role_raster(tri, role, K, D.basis(az, 23), np.array(depth["center"]), depth["span"])
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(raster)
            ax.set_axis_off()
            ax.set_title(cell_info["model"] + "\n" + cell_info["stage"] + " | " + str(az), fontsize=9)
            facts["views"].append(dict(azimuth=az, **scope))
            if scope["projected_vertices_outside_inherited_frame"]:
                ax.text(.01, .01, "Inherited-frame clipping: see receipt", transform=ax.transAxes, fontsize=7, color="#b34b27")
        for row, view in ((2, "front"), (3, "side")):
            ax = fig.add_subplot(grid[row, col])
            H.draw_section(ax, g, np.flatnonzero(role[K:] == "SEAM"), bounds, C.L, view, 1)
            from matplotlib.collections import LineCollection
            axis, project = (2, (0, 1)) if view == "front" else (0, (2, 1))
            pos = bounds["front_positions" if view == "front" else "side_positions"][1]
            ff, ids = H.finite_sections(g["F"], axis, pos, 1e-12 * C.L)
            bm = role[K:][ids] == "BRIDGE"
            if bm.any():
                ax.add_collection(LineCollection(ff[bm][:, :, project], colors="#41a46c", linewidths=1.6, zorder=3))
            ax.set_title(cell_info["model"] + " | " + view + f"={pos:.5f}", fontsize=8)
        display.append(facts)
    fig.suptitle(page["request_id"] + " | " + page["split"] + " " + page["part_type"] + f" | K={K} N={page['N']}\n"
                 + page["sheet"] + " | Blue C / orange SEAM / gray BULK / green BRIDGE\n"
                 "Frozen identical OLD_S15000 layout, noise, camera, scale and section planes", fontsize=12)
    fig.text(.5, .018, "Actual AI_PROXY review recorded separately. Section endpoints alone do not certify cracks or topology. No unique-source-style grading.",
             ha="center", fontsize=9)
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110)
        with png.open("xb") as f:
            f.write(buffer.getvalue())
    finally:
        plt.close(fig)
    r = dict(status="PENDING_ACTUAL_VIEW", page_id=page["page_id"], request_id=page["request_id"],
             condition_id=page["condition_id"], sheet=page["sheet"], part_type=page["part_type"], split=page["split"],
             path=str(png), image_sha256=sha(png), definition_sha256=dh, cells=page["cells"], display=display,
             seconds=time.perf_counter() - tick, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             model_calls=0, scoring_calls=0)
    P.immutable(receipt, r)
    return r


def run(workers=2):
    plan = prepare()
    out = OUT / "gallery"
    sources = {}
    for page in plan["pages"]:
        for cell_info in page["cells"]:
            sources.update(G.authenticate(cell_info))
    definition = dict(manifest_sha256=sha(out / "manifest.json"), sources=sources, protected=plan["protected"])
    P.immutable(out / "definition.json", definition)
    done = out / "completion.json"
    if done.exists():
        r = read(done)
        for im in r["images"]:
            if sha(im["path"]) != im["image_sha256"]:
                raise ValueError("Gallery image changed")
        return r
    tick = time.perf_counter()
    jobs = [(page, sha(out / "definition.json")) for page in plan["pages"]]
    first = one(jobs[0])
    count = worker_count(C.resources(), first, workers)
    P.immutable(out / "dispatch.json", dict(workers=count, resources=C.resources(),
                first_peak_RSS_KiB=first["peak_RSS_KiB"], model_calls=0, scoring_calls=0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn")) as pool:
        images = [first, *list(pool.map(one, jobs[1:]))]
    verify(plan["protected"])
    content = '<meta charset="utf-8"><h1>Bench relation coverage: USER_V2_S1500, A_S1500 and RELATION_V1_S1500</h1><p>Same original cameras and section positions. TRAIN is the historical panel label, not membership of the new 26-parent pool. Actual review is separate.</p>'
    for im in images:
        name = Path(im["path"]).name
        content += "<h2>" + html.escape(im["page_id"]) + '</h2><a href="' + name + '"><img loading="lazy" style="width:100%" src="' + name + '"></a>'
    (out / "index.html").write_text(content, encoding="utf-8")
    r = dict(status="RENDER_COMPLETE_PENDING_ACTUAL_REVIEW", sheets=len(images), images=images,
             workers=count, seconds=time.perf_counter() - tick, model_calls=0, new_scores=0)
    P.immutable(done, r)
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run"))
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    r = prepare() if a.action == "prepare" else run(a.workers)
    print({k: v for k, v in r.items() if k not in ("pages", "images", "protected")}, flush=True)
