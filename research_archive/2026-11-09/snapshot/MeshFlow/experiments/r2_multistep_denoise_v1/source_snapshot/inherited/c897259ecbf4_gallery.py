"""CPU-only fixed RAW/D1/D3/D5 gallery for the same 12 R2_USER_MIX requests.
No source parent GT, hidden face deletion, rescaling, learned model, or new generation.
Shared renderer/validation copied from postprocess_avmr_pmr_v1/gallery.py; original readonly.
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

SCHEMA = "r2_multistep_denoise_gallery_v1"
STAGES = ("RAW", "D1", "D3", "D5")
CAMERAS = ((22.0, -60.0), (22.0, 120.0))
COLORS = {"C": "#2387c9", "F": "#b8bec4", "edges": "#38424b"}
DEFAULT_ROOT = "/mnt/e/MeshFlow/experiments/r2_multistep_denoise_v1"
OLD_ROOT = Path("/mnt/e/MeshFlow/experiments/r2_postprocess_avmr_pmr_v1")


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



def mesh_paths(root, row):
    rid=row["request_id"]
    return {"RAW":path_value(row["raw_path"]),
            "D1":OLD_ROOT/"outputs"/"R2_USER_MIX"/rid/"post.npz",
            "D3":root/"outputs"/"D3"/rid/"post.npz",
            "D5":root/"outputs"/"D5"/rid/"post.npz"}


def render_request(root, row, out):
    rid=row["request_id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+",rid):
        raise ValueError("Unsafe request identifier")
    k=int(row["K"])
    paths=mesh_paths(root,row)
    sources={str(p):sha256(p) for p in paths.values()}
    loaded={s:load_mesh(p,k,"RAW" if s=="RAW" else "POST") for s,p in paths.items()}
    for s,mesh in loaded.items():
        if not same_bits(mesh[:k],loaded["RAW"][:k]):
            raise ValueError(f"C changes in {rid}/{s}")
    vv=np.concatenate([m.reshape(-1,3) for m in loaded.values()]).astype(float)
    lo,hi=vv.min(0),vv.max(0)
    span=float(np.max(hi-lo))*1.12
    if span<=0:
        raise ValueError("Zero spatial extent")
    bounds={"bounds_model_xyz":[lo.tolist(),hi.tolist()],"center_model_xyz":((lo+hi)/2).tolist(),
            "span":span,"coordinate_rescaling":False,"same_bounds_all_4_panels":True,
            "axis_transform":"model[x,y,z] -> plot[x,z,y]; model Y vertical","projection":"orthographic"}
    receipts=[]
    for view_id,camera in enumerate(CAMERAS):
        image_path=out/f"{rid}__v{view_id}.png"
        identity={"schema":SCHEMA,"request_id":rid,"parent_id":row["parent_id"],"region_type":row["region_type"],
                  "label":"R2_USER_MIX","stages":list(STAGES),"bounds":bounds,"camera_elev_azim":list(camera),
                  "source_hashes":sources,"source_code_sha256":sha256(__file__),"matplotlib_version":matplotlib.__version__,
                  "numpy_version":np.__version__,"colors":COLORS,"pixels":[1600,1400],"hidden_triangles_removed":False,
                  "GT_read":False,"model_calls":0,"GPU":0,"C_bitwise_unchanged_all_stages":True,
                  "topology_claim_from_picture":False}
        sidecar=image_path.with_suffix(".png.json")
        if image_path.exists() or sidecar.exists():
            if not image_path.exists() or not sidecar.exists():
                raise RuntimeError(f"Incomplete render: {image_path}")
            receipt=json.loads(sidecar.read_text(encoding="utf8"))
            if receipt["identity"]!=identity or receipt["sha256"]!=sha256(image_path):
                raise RuntimeError(f"Render identity mismatch: {image_path}")
            receipts.append(receipt)
            continue
        fig=plt.figure(figsize=(16,14),dpi=100,facecolor="white")
        for i,s in enumerate(STAGES):
            mesh=loaded[s]
            ax=fig.add_subplot(2,2,i+1,projection="3d")
            draw(ax,mesh,k,bounds,camera)
            label="RAW (no postprocessing)" if s=="RAW" else f"{s[1:]} denoise pass(es) + one GEO_C"
            ax.set_title(f"{s}: {label}\nN={len(mesh)} | C={k}, F={len(mesh)-k}",fontsize=13,pad=0)
        fig.suptitle(f"R2_USER_MIX: fixed RAW / D1 / D3 / D5 | {row['parent_id']}\n{row['region_type']} | {rid.rsplit('__',1)[-1]} | view {view_id+1}: elev {camera[0]:g}, azim {camera[1]:g}",fontsize=14,y=.985)
        fig.text(.5,.022,"BLUE = unchanged C | GRAY = generated F | Identical camera and bounds across all four panels",ha="center",fontsize=11)
        fig.text(.5,.008,"C restored after every denoiser pass. Geometry cleanup only once at each endpoint. Opaque surfaces may occlude C; pictures do not certify attachment.",ha="center",fontsize=9)
        fig.subplots_adjust(left=.005,right=.995,top=.935,bottom=.04,hspace=.02,wspace=0)
        temporary=image_path.with_suffix(".png.tmp")
        fig.savefig(temporary,format="png",dpi=100,facecolor="white")
        plt.close(fig)
        if any(sha256(p)!=h for p,h in sources.items()):
            raise RuntimeError("Source changed during rendering")
        temporary.replace(image_path)
        receipt={"status":"COMPLETE","identity":identity,"image":str(image_path),"sha256":sha256(image_path),"visual_review":"PENDING_AI_PROXY"}
        json_write(sidecar,receipt)
        receipts.append(receipt)
    return receipts


def artifact_link(path,out,label):
    path=path_value(path)
    if not path.exists(): return f"{html.escape(label)}: unavailable"
    return f'<a href="{html.escape(Path(os.path.relpath(path,out)).as_posix(),quote=True)}">{html.escape(label)}</a>'


def make_html(root,rows,out):
    blocks=['<!doctype html><html lang="en"><meta charset="utf-8"><title>R2_USER_MIX multistep denoising</title>',
            '<style>body{font:15px Arial,sans-serif;margin:24px;max-width:1650px}img{width:49%;vertical-align:top}table{border-collapse:collapse}td,th{padding:5px 12px;border:1px solid #ddd}section{margin:35px 0}a{color:#176299}</style>',
            '<h1>R2_USER_MIX: RAW / D1 / D3 / D5</h1>',
            '<p>Same 12 existing RAW meshes, two fixed opposite views each. Blue C, gray free. Common camera and joint bounds per request, no geometry normalization or hidden-face deletion. Each learned pass restores original C. One geometric cleanup at each endpoint; intermediate denoising meshes are not geometrically cleaned.</p>',
            '<p>AI_PROXY visual observations are separate from human ratings and numerical topology audits. Pictures alone do not certify attachment. <a href="gallery_manifest.json">Render provenance</a> | <a href="visual_review.md">Visual review</a> | <a href="../REPORT.md">Experiment report</a></p>']
    for row in rows:
        rid=row['request_id']
        paths=mesh_paths(root,row)
        blocks.append(f'<section><h2>{html.escape(rid)}</h2><table><tr><th>Stage</th><th>Mesh</th></tr>')
        for s,p in paths.items():
            ply=p.with_suffix('.ply')
            blocks.append(f'<tr><td>{s}</td><td>{artifact_link(p,out,"NPZ")} | {artifact_link(ply,out,"PLY")}</td></tr>')
        blocks.append('</table>')
        for v in range(2):
            name=f'{rid}__v{v}.png'
            blocks.append(f'<a href="{name}"><img src="{name}" alt="{html.escape(rid)} view {v+1}"></a>')
        blocks.append('</section>')
    blocks.append('</html>')
    (out/'index.html').write_text('\n'.join(blocks),encoding='utf8')


def cpu_resources():
    out={"logical_CPU":os.cpu_count(),"process_threads":1,"pid":os.getpid(),"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),"load_1_5_15":list(os.getloadavg())}
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name,value=line.split(':',1)
        if name in ('MemTotal','MemAvailable'): mem[name+'_KiB']=int(value.strip().split()[0])
    out.update(mem)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=DEFAULT_ROOT)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        print(json.dumps({"status":"PASS_INTERFACE_IMPORTS","schema":SCHEMA,"CPU_threads":1,"model_calls":0,"renders":0,"matplotlib":matplotlib.__version__}))
        return
    started=time.monotonic()
    root=path_value(args.root).resolve()
    if root!=Path(DEFAULT_ROOT).resolve(): raise ValueError('This fixed pilot only writes its registered experiment directory')
    reg_path=root/'registration.json'
    reg_sha=sha256(reg_path)
    registration=json.loads(reg_path.read_text(encoding='utf8'))
    rows=sorted(registration['records'],key=lambda r:r['request_id'])
    if len(rows)!=12 or len({r['request_id'] for r in rows})!=12 or any(r['label']!='R2_USER_MIX' for r in rows):
        raise ValueError('Expected 12 fixed R2_USER_MIX request rows')
    out=root/'gallery'; out.mkdir(parents=True,exist_ok=True)
    resources=cpu_resources()
    receipts=[]
    for row in rows:
        receipts.extend(render_request(root,row,out))
        print(json.dumps({"request_id":row['request_id'],"renders_complete":len(receipts)}),flush=True)
    if sha256(reg_path)!=reg_sha: raise RuntimeError('Registration changed during render')
    make_html(root,rows,out)
    manifest={"schema":SCHEMA,"status":"COMPLETE","requests":12,"stages":list(STAGES),"input_meshes":48,"PNG_count":len(receipts),
              "registration_sha256":reg_sha,"source_code_sha256":sha256(__file__),"wall_seconds":time.monotonic()-started,"CPU_workers":1,"CPU_threads":1,
              "resources_before":resources,"resources_after":cpu_resources(),"GPU":0,"model_calls":0,"GT_read":False,"source_geometry_modified":False,
              "visual_review_status":"PENDING_AI_PROXY","receipts":receipts}
    json_write(out/'gallery_manifest.json',manifest)
    print(json.dumps({"status":"COMPLETE","images":len(receipts),"wall_seconds":manifest['wall_seconds']}),flush=True)


if __name__=='__main__':
    main()
