"""CPU-only complete technical and blinded galleries for the frozen quick pilot.

No mesh repair, vertex welding, geometry alignment, model import or model call.
The renderer changes only camera/display coordinates (x,z,y for a Y-up view).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "native_t1/runs/chair_control_usability_quick_v1"
PURPOSE = "\u7528\u4e8e\u5f62\u72b6\u6784\u601d\u548c\u540e\u7eed\u9009\u62e9/\u7f16\u8f91\u7684\u4f4e\u591a\u8fb9\u5f62\u5019\u9009\uff0c\u4e0d\u662f\u76f4\u63a5\u7528\u4e8e\u5236\u9020\u6216\u7269\u7406\u4eff\u771f\u7684\u5408\u683c\u7f51\u683c\u3002"
RUBRIC = (
    ("A", "\u4e0e\u7ed9\u5b9aC\u7684\u51e0\u4f55\u76f8\u5bb9\u6027"),
    ("B", "\u662f\u5426\u6784\u6210\u5b8c\u6574\u5408\u7406\u7684Chair\u5f62\u72b6\u5019\u9009"),
    ("C", "\u662f\u5426\u5b58\u5728\u59a8\u788d\u5f53\u524d\u7528\u9014\u7684\u660e\u663e\u7a7f\u63d2\u3001\u8131\u79bb\u3001\u788e\u7247\u6216\u7f3a\u5931"),
)
LEVELS = ("0\uff1a\u4e0d\u53ef\u4f5c\u4e3a\u5f53\u524d\u7528\u9014\u5019\u9009", "1\uff1a\u9700\u8981\u5927\u5e45\u91cd\u5efa",
          "2\uff1a\u53ef\u4ee5\u4f5c\u4e3a\u5019\u9009\u7ee7\u7eed\u5c40\u90e8\u8c03\u6574", "3\uff1a\u53ef\u76f4\u63a5\u4f5c\u4e3a\u5f53\u524d\u7528\u9014\u7684\u5019\u9009\u4fdd\u7559")
VIEWS = ((24,45), (24,135), (24,225), (24,315))
KNOWN_COLOR = "#238da7"
FREE_COLOR = "#e69b45"
RANDOMIZATION_CONTRACT = "CHAIR_CONTROL_USABILITY_QUICK_PILOT_V1_BLIND_ORDER_SHA256_v1"
RENDERER_CONTRACT = "raw_triangles_Yup_orthographic_4azimuths_wire_v1"
FAIL_STATUSES = {"FAILED", "BLOCKED", "NOT_APPLICABLE", "NOT_RUN"}

def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def _write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8"); os.replace(tmp, path)

def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576), b""): h.update(block)
    return h.hexdigest()

def _array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode()+a.tobytes()).hexdigest()

def _path(value, out=OUT):
    p = str(value)
    if os.name != "nt" and len(p)>2 and p[1]==":":
        p = "/mnt/"+p[0].lower()+"/"+p[3:].replace("\\","/")
    q = Path(p)
    if q.is_absolute(): return q
    if (ROOT/q).exists(): return ROOT/q
    return Path(out)/q

def _triangles(value):
    a = np.asarray(value)
    if a.size == 0: return a.reshape(0,3,3)
    if a.size % 9: raise ValueError("Triangle array size is not divisible by nine")
    return a.reshape(-1,3,3)

def _rows(path):
    data = _json(path)
    if isinstance(data, list): return data
    for key in ("rows","requests","records"):
        if isinstance(data.get(key), list): return data[key]
    raise ValueError("Request JSON must contain a row list")

def _in_panel(row, panel):
    return row.get("panel")==panel or panel in row.get("panels", [])

def _condition(row, out):
    path = _path(row["condition_npz"], out)
    with np.load(path, allow_pickle=False) as d:
        C = _triangles(d["C"]).copy()
        Y = _triangles(d["GT_free"]).copy()
        full = _triangles(d["GT_full"]).copy()
    if not all(np.isfinite(a).all() for a in (C,Y,full)):
        raise ValueError("Nonfinite registered condition/GT")
    return dict(C=C, Y=Y, full=full, path=str(path), file_sha256=_sha(path))

def _geometry(row, out, cond):
    result_path = Path(out)/"outputs"/row["request_id"]/"result.json"
    if not result_path.exists():
        return dict(status="NOT_RUN", error="Result record absent", result_path=str(result_path))
    result = _json(result_path)
    status = result.get("status", "UNKNOWN")
    if status in FAIL_STATUSES:
        return dict(status=status, error=result.get("error", result.get("reason", status)),
                    result_path=str(result_path))
    raw_path = result.get("raw_path") or result.get("raw_npz")
    if raw_path is None:
        local = result_path.parent/"raw.npz"
        if local.exists(): raw_path = str(local)
    if raw_path is None:
        return dict(status="BLOCKED", error="No raw path", result_path=str(result_path))
    raw_path = _path(raw_path, out)
    if not raw_path.exists():
        return dict(status="BLOCKED", error="Raw file absent", raw_path=str(raw_path))
    with np.load(raw_path, allow_pickle=False) as d:
        raw = _triangles(d["raw"]).copy()
    if not np.isfinite(raw).all():
        return dict(status="FAILED", error="Nonfinite raw cannot be rendered",
                    raw_path=str(raw_path), result_path=str(result_path))
    known = None
    if row.get("input_npz"):
        ip = _path(row["input_npz"], out)
        if ip.exists():
            with np.load(ip, allow_pickle=False) as d:
                if "known_mask" in d: known = np.asarray(d["known_mask"],bool).reshape(-1)
    if known is None:
        known = np.arange(len(raw)) < len(cond["C"])
    if len(known)!=len(raw) or int(known.sum())!=len(cond["C"]):
        raise ValueError("Registered known-mask shape/count differs")
    got_C = np.ascontiguousarray(raw[known]); ref_C = np.ascontiguousarray(cond["C"])
    c_ok = got_C.dtype==ref_C.dtype and got_C.shape==ref_C.shape and got_C.tobytes()==ref_C.tobytes()
    meta = result.get("metadata", result.get("sampling", {}))
    all_states = None
    for key in ("known_all51_bitwise","C_all51_bitwise","known_all_states_bitwise"):
        if key in meta: all_states = bool(meta[key]); break
    return dict(status=status, raw=raw, known=known, C=got_C, F=raw[~known],
                raw_path=str(raw_path), raw_sha256=_sha(raw_path),
                result_path=str(result_path), C_final_preserved=c_ok,
                C_all_states_reported=all_states, human_eligible=c_ok and all_states is not False)

def _display(a):
    # No normalization or in-place modification; plotting uses vertical source Y.
    return np.asarray(a)[..., [0,2,1]]

def _frame(gt, meshes):
    base = np.concatenate([np.asarray(a).reshape(-1,3) for a in gt if np.asarray(a).size])
    points = [base] + [np.asarray(a).reshape(-1,3) for a in meshes if np.asarray(a).size]
    base_lo,base_hi = base.min(0),base.max(0)
    original_center = (base_lo+base_hi)/2
    original_half = max(float(np.max(base_hi-base_lo))/2, 1e-5)*1.08
    all_points = np.concatenate(points)
    out_of_frame = bool(np.any(np.abs(all_points-original_center)>original_half))
    lo,hi = all_points.min(0),all_points.max(0)
    center=(lo+hi)/2
    half=max(float(np.max(hi-lo))/2,1e-5)*1.08
    return dict(center=center.tolist(),half_extent=half,
                adjusted_for_outlier=out_of_frame,
                original_GT_center=original_center.tolist(),
                original_GT_half_extent=original_half,
                union_min=lo.tolist(),union_max=hi.tolist(),
                policy="All raw outputs and GT of this parent; one orthographic cube for all models/candidates/conditions")

def _face_colors(tris, color):
    from matplotlib.colors import to_rgb
    a = _display(tris).astype(np.float64)
    normal = np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])
    norm = np.linalg.norm(normal,axis=1,keepdims=True)
    normal = np.divide(normal,norm,out=np.zeros_like(normal),where=norm>0)
    light=np.asarray([1.,-1.,2.]); light/=np.linalg.norm(light)
    # Two-sided, fixed world light, per-triangle flat shading only.
    intensity=.48+.52*np.abs(normal@light)
    return np.clip(np.asarray(to_rgb(color))[None,:]*intensity[:,None],0,1)

def _render(path, C, F, frame):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from matplotlib.patches import Patch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(16,3.6),dpi=130,facecolor="#f6f7f8")
    center=_display(np.asarray(frame["center"]));half=frame["half_extent"]
    for index,(elev,azim) in enumerate((*VIEWS,VIEWS[0])):
        ax=fig.add_subplot(1,5,index+1,projection="3d")
        ax.set_proj_type("ortho")
        sets=[(arr,color) for arr,color in ((C,KNOWN_COLOR),(F,FREE_COLOR)) if len(arr)]
        if sets:
            plot=np.concatenate([_display(arr) for arr,_ in sets])
            if index==4:
                edges=plot[:,[[0,1],[1,2],[2,0]],:].reshape(-1,2,3)
                edge_colors=[color for arr,color in sets for _ in range(3*len(arr))]
                ax.add_collection3d(Line3DCollection(edges,colors=edge_colors,linewidths=.5,alpha=.83))
            else:
                colors=np.concatenate([_face_colors(arr,color) for arr,color in sets])
                # One collection allows depth ordering between known and free faces.
                ax.add_collection3d(Poly3DCollection(plot,facecolors=colors,
                    edgecolors=(0.08,0.10,0.13,.18),linewidths=.13,zsort="average"))
        ax.set_xlim(center[0]-half,center[0]+half)
        ax.set_ylim(center[1]-half,center[1]+half)
        ax.set_zlim(center[2]-half,center[2]+half)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=elev,azim=azim)
        ax.set_axis_off()
        ax.set_title("Wireframe" if index==4 else f"Azimuth {azim} deg",fontsize=10,pad=-8)
        ax.set_facecolor("#f6f7f8")
    fig.legend(handles=[Patch(facecolor=KNOWN_COLOR,label="Known C"),
                        Patch(facecolor=FREE_COLOR,label="Free")],
               loc="lower center",ncol=2,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.003,right=.997,bottom=.095,top=.95,wspace=-.08)
    tmp=path.with_suffix(".tmp.png")
    fig.savefig(tmp,dpi=130,facecolor=fig.get_facecolor(),metadata={"Software":RENDERER_CONTRACT})
    plt.close(fig);os.replace(tmp,path)

CSS = """body{font:16px/1.6 system-ui,sans-serif;color:#18212b;background:#eef1f4;margin:0}
main{max-width:1800px;margin:auto;padding:24px}h1{font-size:28px}h2{margin-top:40px}
p.note{background:#fff5d9;padding:14px;border-left:4px solid #c98a18}
.card{background:white;border:1px solid #d6dce3;border-radius:8px;margin:18px 0;padding:16px}
img{width:100%;height:auto;display:block;background:#f6f7f8}
label{font-weight:bold}.legend{color:#45556b;font-size:14px}.missing{padding:50px;background:#fee;color:#8b2525}
nav a{display:inline-block;margin:4px 12px 4px 0}table{border-collapse:collapse}
td,th{border:1px solid #cbd3dd;padding:6px 12px}.small{font-size:13px;color:#526170}a{color:#185d8b}
"""
def _page(path,title,body):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    text+=f"<title>{html.escape(title)}</title><style>{CSS}</style><main><h1>{html.escape(title)}</h1>"
    text+="".join(body)+"</main></html>"
    Path(path).write_text(text,encoding="utf-8")

def _instructions():
    return [f'<p class="note">{PURPOSE}</p>',
            '<p>\u9752\u8272\u4e3a\u7ed9\u5b9a C\uff0c\u6a59\u8272\u4e3a free\u3002\u56db\u4e2a\u56fa\u5b9a\u65b9\u4f4d\u53ca\u4e00\u4e2a\u7ebf\u6846\u89c6\u56fe\uff1b\u539f\u4e09\u89d2\u9762\u672a\u710a\u63a5\u3001\u5e73\u6ed1\u6216\u4fee\u8865\u3002\u8bf7\u67e5\u770b\u6240\u6709\u89c6\u56fe\u3002</p>',
            '<p>\u672c\u8f6e\u5f00\u53d1\u6027\u8bc4\u5206\uff0c\u6bcf\u9879 0\u20133\uff1bC \u9879\u4e5f\u6cbf\u7528\u201c\u5206\u6570\u8d8a\u9ad8\u8d8a\u9002\u5408\u5f53\u524d\u7528\u9014\u201d\u7684\u7edf\u4e00\u7b49\u7ea7\u3002</p>',
            '<ul>'+''.join(f'<li>{key}. {text}</li>' for key,text in RUBRIC)+'</ul>',
            '<p>'+'\uff1b'.join(LEVELS)+'\u3002</p>',
            '<p>\u4ec5\u5f53\u4e09\u4e2a\u5206\u9879\u5747 \u22652 \u4e14 C \u4fdd\u6301\u901a\u8fc7\u65f6\uff0c\u624d\u5b9a\u4e49 usable\u3002\u5f53\u524d\u65e0\u4eba\u8bc4\u5206\uff1ahuman_usable=NA\uff0cuser_study=NOT_RUN\u3002\u4e0d\u5f97\u7528\u6570\u503c\u8ddd\u79bb\u4ee3\u66ff\u4eba\u5de5\u6807\u7b7e\u3002</p>']

def _card(title,image_path,extra=""):
    return f'<section class="card"><label>{html.escape(title)}</label>{extra}<img loading="lazy" src="{html.escape(image_path)}" alt="{html.escape(title)}"></section>'

def _csv(path,rows,fields):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)

def run(out=OUT):
    """Build all registered views; missing assets remain visible, not discarded."""
    started=time.time();out=Path(out).resolve();gallery=out/"gallery"
    gallery.mkdir(parents=True,exist_ok=True)
    req_path=out/"requests.json"
    if not req_path.exists():req_path=out/"registration/requests.json"
    requests=_rows(req_path)
    if len({r["request_id"] for r in requests})!=len(requests):
        raise ValueError("Duplicate request IDs")
    conditions={};geometries={};errors=[]
    for row in requests:
        if row.get("status") in ("NA","NOT_APPLICABLE","BLOCKED"):
            geometries[row["request_id"]]=dict(
                status="NOT_APPLICABLE" if row["status"]=="NA" else row["status"],
                error=row.get("reason","Registered asset or construction unavailable"))
            continue
        try:
            cid=row["condition_id"]
            if cid not in conditions:conditions[cid]=_condition(row,out)
            geometries[row["request_id"]]=_geometry(row,out,conditions[cid])
        except Exception as exc:
            msg=f"{type(exc).__name__}: {exc}"
            geometries[row["request_id"]]=dict(status="RENDER_BLOCKED",error=msg)
            errors.append(dict(request_id=row["request_id"],stage="load",error=msg))
    main=[r for r in requests if _in_panel(r,"main")]
    nested=[r for r in requests if _in_panel(r,"multi")]
    frames={}
    for uid in sorted({r["uid"] for r in requests}):
        rows=[r for r in requests if r["uid"]==uid]
        gt=[conditions[r["condition_id"]]["full"] for r in rows if r["condition_id"] in conditions]
        raw=[geometries[r["request_id"]]["raw"] for r in rows if "raw" in geometries[r["request_id"]]]
        if gt:frames[uid]=_frame(gt,raw)
    all_cids=sorted({r["condition_id"] for r in main})+sorted({r["condition_id"] for r in nested}-{r["condition_id"] for r in main})
    condition_ids={cid:f"G{i+1:02d}" for i,cid in enumerate(all_cids)}
    mapping=[];by_request={}
    for cid in all_cids:
        rows=[r for r in requests if r["condition_id"]==cid]
        rows.sort(key=lambda r:hashlib.sha256((RANDOMIZATION_CONTRACT+"|"+r["request_id"]).encode()).hexdigest())
        for pos,row in enumerate(rows,1):
            anon=f"{condition_ids[cid]}-R{pos:02d}"
            m=dict(blind_candidate_id=anon,anonymous_condition_id=condition_ids[cid],
                   display_position=pos,request_id=row["request_id"],uid=row["uid"],
                   condition_id=cid,context_level=row["context_level"],arm=row["arm"],
                   candidate=row["candidate"],panels=";".join(row.get("panels",[row.get("panel","")])))
            mapping.append(m);by_request[row["request_id"]]=m
    _csv(out/"blind_mapping.csv",mapping,list(mapping[0]) if mapping else [])
    _write_json(out/"blind_mapping.json",dict(randomization_contract=RANDOMIZATION_CONTRACT,rows=mapping,
        note="Author-only identity key. Do not distribute with the blinded viewing bundle."))
    prior_path=gallery/"render_manifest.json"
    prior=_json(prior_path).get("images",{}) if prior_path.exists() else {}
    rendered={};renderer_sha=_sha(__file__)
    def ensure_image(name,C,F,uid,identity):
        path=gallery/"images"/(name+".png")
        signature=hashlib.sha256(json.dumps(dict(renderer=renderer_sha,frame=frames[uid],identity=identity,
            C=_array_sha(C),F=_array_sha(F)),sort_keys=True).encode()).hexdigest()
        if not(path.exists() and prior.get(name,{}).get("signature")==signature):
            _render(path,C,F,frames[uid])
        rendered[name]=dict(path=str(path),signature=signature,sha256=_sha(path),views=5)
        return "../images/"+path.name
    image_by_request={};condition_images={};gt_images={}
    for cid in all_cids:
        rows=[r for r in requests if r["condition_id"]==cid]
        if cid not in conditions:continue
        uid=rows[0]["uid"];c=conditions[cid];anon=condition_ids[cid]
        condition_images[cid]=ensure_image(anon+"_C",c["C"],np.empty((0,3,3)),uid,c["file_sha256"]+":C")
        gt_images[cid]=ensure_image(anon+"_GT",c["C"],c["Y"],uid,c["file_sha256"]+":GT")
    for row in requests:
        rid=row["request_id"];g=geometries[rid]
        if "raw" not in g:continue
        try:
            image_by_request[rid]=ensure_image(by_request[rid]["blind_candidate_id"],g["C"],g["F"],
                row["uid"],g["raw_sha256"])
        except Exception as exc:
            errors.append(dict(request_id=rid,stage="render",error=f"{type(exc).__name__}: {exc}"))
    _write_json(prior_path,dict(contract=RENDERER_CONTRACT,renderer_sha256=renderer_sha,
        views=[dict(elevation=e,azimuth=a) for e,a in VIEWS]+[dict(elevation=VIEWS[0][0],azimuth=VIEWS[0][1],wireframe=True)],
        known_color=KNOWN_COLOR,free_color=FREE_COLOR,coordinate_display="source (x,y,z) -> display (x,z,y), no mesh mutation",
        fixed_lighting="fixed world vector [1,-1,2], two-sided per-triangle flat shade, ambient=.48",
        frames=frames,images=rendered,errors=errors))
    def image_for_page(src,blind):
        if not blind:return src
        source=gallery/"images"/Path(src).name
        target=gallery/"blind/assets"/source.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and not os.path.samefile(source,target):target.unlink()
        if not target.exists():
            try:os.link(source,target)
            except OSError:shutil.copyfile(source,target)
        return "assets/"+source.name
    def build_page(rows,blind,nested_panel=False):
        body=_instructions() if blind else [f'<p class="note">{PURPOSE}</p>',
            '<p>\u516c\u5f00\u6280\u672f\u6807\u7b7e\uff1b\u5305\u62ec\u5f53\u524d C \u4e0e\u5bf9\u5e94 GT \u53c2\u7167\u3002\u6240\u6709\u6307\u6807\u89c1 <a href="../../per_output_metrics.csv">\u9010\u8f93\u51fa CSV</a>\u3002</p>']
        if nested_panel:
            body.append('<p class="note">\u5d4c\u5957\u4e0a\u4e0b\u6587\u8bca\u65ad\uff1a2 \u4e2a\u65e7\u5f00\u53d1\u7236\u5bf9\u8c61\u3001\u6bcf\u5c42\u4e24\u4e2a\u5019\u9009\u3002C1 \u4e0e\u4e3b\u9762\u677f\u590d\u7528\uff1b\u4e0d\u80fd\u89c6\u4e3a\u65b0\u7684\u72ec\u7acb\u5f62\u72b6\u6d4b\u8bd5\u3002\u6761\u4ef6\u589e\u52a0\u53ef\u80fd\u6539\u53d8 free \u9884\u7b97\u3002</p>')
        cids=sorted({r["condition_id"] for r in rows},key=lambda c:condition_ids[c])
        body.append('<nav>'+''.join(f'<a href="#{condition_ids[c]}">{condition_ids[c]}</a>' for c in cids)+'</nav>')
        for cid in cids:
            rr=[r for r in rows if r["condition_id"]==cid]
            rr.sort(key=(lambda r:by_request[r["request_id"]]["display_position"]) if blind else (lambda r:(r["arm"],r["candidate"])))
            uid=rr[0]["uid"];anon=condition_ids[cid]
            title=f"\u533f\u540d\u6761\u4ef6 {anon}" if blind else f'{anon} | UID {uid} | C{rr[0]["context_level"]}'
            body.append(f'<h2 id="{anon}">{html.escape(title)}</h2>')
            if frames.get(uid,{}).get("adjusted_for_outlier"):
                body.append('<p class="note">\u672c\u7236\u5bf9\u8c61\u5b58\u5728\u8d85\u51fa\u539f GT \u753b\u5e45\u7684\u8f93\u51fa\uff1b\u8be5\u7236\u5168\u90e8\u6a21\u578b\u3001\u5019\u9009\u3001\u6761\u4ef6\u4e0e\u53c2\u7167\u7edf\u4e00\u6269\u5927\u753b\u5e45\uff0c\u672a\u5355\u72ec\u7f29\u653e\u6216\u88c1\u53bb\u5f02\u5e38\u90e8\u5206\u3002</p>' if not blind else
                    '<p class="note">\u8be5\u7236\u5168\u90e8\u5019\u9009\u7edf\u4e00\u6269\u5927\u5c55\u793a\u753b\u5e45\u4ee5\u4fdd\u7559\u6240\u6709\u51e0\u4f55\uff0c\u672a\u5355\u72ec\u7f29\u653e\u6216\u88c1\u53bb\u5f02\u5e38\u90e8\u5206\u3002</p>')
            if cid in condition_images:body.append(_card("\u5f53\u524d\u7ed9\u5b9a C",image_for_page(condition_images[cid],blind)))
            if not blind and cid in gt_images:body.append(_card("\u6280\u672f\u53c2\u7167\uff1aC + \u5bf9\u5e94 GT free",gt_images[cid]))
            for row in rr:
                rid=row["request_id"];g=geometries[rid];m=by_request[rid]
                label=m["blind_candidate_id"] if blind else f'{row["arm"]} | candidate {row["candidate"]} | {rid} | {g["status"]}'
                warning=''
                if g.get("human_eligible") is False:
                    warning='<p class="note">C \u4fdd\u6301\u5931\u8d25\uff1a\u8be5\u8f93\u51fa\u4e0d\u80fd\u6807\u4e3a usable\u3002</p>'
                if rid in image_by_request:body.append(_card(label,image_for_page(image_by_request[rid],blind),warning))
                else:
                    detail="\u8f93\u51fa\u7f3a\u5931\u6216\u5931\u8d25\uff1b\u4fdd\u7559\u8be5\u767b\u8bb0\u5019\u9009\u4f4d\u7f6e\u3002" if blind else str(g.get("error","Rendering unavailable"))
                    body.append(f'<section class="card"><label>{html.escape(label)}</label><div class="missing">{html.escape(detail)}</div></section>')
        folder="blind" if blind else "technical"
        target=gallery/folder/("nested.html" if nested_panel else "index.html")
        _page(target,("\u76f2\u8bc4" if blind else "\u6280\u672f")+"\u56fe\u5e93"+("\uff1a\u5d4c\u5957\u4e0a\u4e0b\u6587" if nested_panel else "\uff1a\u56fa\u5b9a\u4e3b\u9762\u677f"),body)
    build_page(main,False);build_page(main,True)
    if nested:build_page(nested,False,True);build_page(nested,True,True)
    fields=["reviewer_id","anonymous_condition_id","blind_candidate_id","display_position",
        "C_final_preserved","A_geometry_compatibility_0_3","B_chair_completeness_0_3",
        "C_obstructions_0_3","human_usable","notes"]
    def ratings(rows):
        return [dict(reviewer_id="",anonymous_condition_id=by_request[r["request_id"]]["anonymous_condition_id"],
            blind_candidate_id=by_request[r["request_id"]]["blind_candidate_id"],
            display_position=by_request[r["request_id"]]["display_position"],
            C_final_preserved=geometries[r["request_id"]].get("C_final_preserved","NA"),
            A_geometry_compatibility_0_3="",B_chair_completeness_0_3="",C_obstructions_0_3="",
            human_usable="NA",notes="") for r in sorted(rows,key=lambda q:by_request[q["request_id"]]["blind_candidate_id"])]
    _csv(out/"human_ratings_template.csv",ratings(main),fields)
    if nested:_csv(out/"human_ratings_nested_template.csv",ratings(nested),fields)
    shutil.copyfile(out/"human_ratings_template.csv",gallery/"blind/human_ratings_template.csv")
    if nested:shutil.copyfile(out/"human_ratings_nested_template.csv",gallery/"blind/human_ratings_nested_template.csv")
    geometry_records=[]
    for row in requests:
        g=geometries[row["request_id"]]
        record={key:value for key,value in g.items() if key not in ("raw","known","C","F")}
        record.update(request_id=row["request_id"],condition_id=row["condition_id"],
                      image=image_by_request.get(row["request_id"]))
        geometry_records.append(record)
    _write_json(gallery/"geometry_manifest.json",dict(rows=geometry_records,
        note="Rendering audit only; no model calls, no numerical quality labels, no human ratings."))
    _page(gallery/"index.html","Chair \u53ef\u7528\u6027\u56fa\u5b9a\u9762\u677f\u56fe\u5e93",[
        f'<p class="note">{PURPOSE}</p>',
        '<p><a href="technical/index.html">\u516c\u5f00\u6280\u672f\u4e3b\u9762\u677f\uff08C \u4e0e GT \u53c2\u7167\uff09</a></p>',
        '<p><a href="blind/index.html">\u76f2\u8bc4\u4e3b\u9762\u677f\uff08\u9690\u85cf\u6a21\u578b\u4e0e GT\uff09</a></p>',
        '<p><a href="technical/nested.html">\u6280\u672f\u5d4c\u5957\u8bca\u65ad</a> \u00b7 <a href="blind/nested.html">\u533f\u540d\u5d4c\u5957\u8bca\u65ad</a></p>' if nested else '',
        '<p>\u76f2\u8bc4\u9875\u9762\u4e0d\u94fe\u63a5\u8eab\u4efd\u6620\u5c04\u6216\u516c\u5f00\u6307\u6807\u3002\u4f5c\u8005\u6620\u5c04\u5355\u72ec\u4fdd\u5b58\u5728\u8fd0\u884c\u6839\u76ee\u5f55 blind_mapping.csv/json\uff1b\u53d1\u9001\u76f2\u8bc4\u6750\u6599\u65f6\u4e0d\u8981\u9644\u5e26\u8be5\u6620\u5c04\u6216\u6280\u672f\u9875\u3002</p>',
        '<p><a href="../human_ratings_template.csv">\u7a7a\u4e3b\u9762\u677f\u4eba\u8bc4\u8868</a>\uff1b\u5f53\u524d human_usable=NA\uff0cuser_study=NOT_RUN\u3002</p>'])
    main_ready=sum(r["request_id"] in image_by_request for r in main)
    nested_ready=sum(r["request_id"] in image_by_request for r in nested)
    summary=dict(status="COMPLETE" if len(main)==64 and main_ready==64 and not errors else "PARTIAL",
        main_requests=len(main),main_rendered=main_ready,nested_requests=len(nested),nested_rendered=nested_ready,
        unique_requests=len(requests),unique_rendered=len(image_by_request),images=len(rendered),
        view_count_per_image=5,frame_adjusted_parents=[u for u,f in frames.items() if f["adjusted_for_outlier"]],
        human_usable="NA",user_study="NOT_RUN",AI_PROXY="NOT_RUN",errors=errors,
        model_calls=0,GPU_calls=0,seconds=time.time()-started,
        paths=dict(index=str(gallery/"index.html"),technical=str(gallery/"technical/index.html"),
                   blind=str(gallery/"blind/index.html"),mapping=str(out/"blind_mapping.csv"),
                   human_template=str(out/"human_ratings_template.csv"),render_manifest=str(prior_path)))
    _write_json(gallery/"summary.json",summary)
    return summary

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args()
    print(json.dumps(run(args.out),ensure_ascii=False,indent=2))
