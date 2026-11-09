"""Complete anonymous RAW galleries for the registered contextual-alignment study."""
from __future__ import annotations
import hashlib,html,time
from pathlib import Path
import numpy as np
import contextual_alignment_evaluation as ce
from meshflow_control.evaluation.io import arrays,read_json,write_json,digest,resolve
ren=ce.importlib.import_module("meshflow_control.evaluation.render")
def mapping():
    order=sorted(ce.LABELS,key=lambda label:hashlib.sha256((ce.NAMESPACE+"/anonymous/"+label).encode()).hexdigest())
    return {label:"Method "+str(i+1) for i,label in enumerate(order)}
def render(out,manifest=None):
    out=ce.root(out);rows=ce.completed_rows(out,manifest);qs=ce.request_rows(out);tick=time.perf_counter()
    expected={(label,q["request_id"]) for q in qs for label in ce.LABELS};by={(r["label"],r["request_id"]):r for r in rows}
    if set(by)!=expected:raise ValueError("All192 registered completed outputs required before final camera freeze")
    for r in rows:
        if r.get("status") in ("FAILED","BLOCKED","RUNNING","PENDING","NOT_RUN") or not resolve(r["raw_path"]).is_file():raise ValueError("Cannot render a missing/failed output as completed")
        if digest(r["raw_path"])!=r["raw_sha256"]:raise ValueError("RAW hash mismatch")
    dest=out/"gallery";aliases=mapping();ordered=sorted(ce.LABELS,key=lambda x:aliases[x])
    identity=dict(outputs={l+"__"+rid:by[l,rid]["raw_sha256"] for l,rid in sorted(by)},adapter_sha256=digest(__file__),renderer_sha256=digest(ren.__file__))
    saved=dest/"gallery_manifest.json"
    if saved.exists():
        old=read_json(saved)
        if old["identity"]!=identity:raise ValueError("Existing gallery identity changed")
        for p in old["pages"]:
            for k in ("blind","reference"):
                if k+"_page" in p and digest(p[k+"_page"])!=p[k+"_sha256"]:raise ValueError("Completed gallery page changed")
        return old
    write_json(dest/"method_mapping.json",dict(status="FROZEN",mapping=aliases,method_order="SHA256 of fixed namespace/anonymous/label; independent of measured results",not_double_blind_human_study=True),freeze=True)
    pages=[];camera={}
    for cid in sorted({q["condition_id"] for q in qs}):
        group=[q for q in qs if q["condition_id"]==cid];q0=group[0];panel=q0["panel"];meta=read_json(q0["reference_path"]);c=arrays(meta["condition_path"]);C=c["C"]
        loaded={(label,q["request_id"]):ren._triangles(by[label,q["request_id"]]) for q in group for label in ce.LABELS}
        for (label,rid),t in loaded.items():
            if t[:q0["K"]].tobytes()!=C.tobytes():raise ValueError("Cannot render changed C without flag")
        allgeom=[C]+list(loaded.values())+([c["GT_full"]] if panel=="P1" else [])
        pp=np.concatenate([x.reshape(-1,3) for x in allgeom]);bounds=(pp.min(0),pp.max(0))
        camera[cid]=dict(bounds=[x.tolist() for x in bounds],views=ren.VIEWS,status="FINAL_COMPLETE",scope="All four labels and four fixed noises plus source reference when applicable; no crop/repair/normalization",
            raw_hashes={label+"__n"+str(q["candidate"]):by[label,q["request_id"]]["raw_sha256"] for q in group for label in ce.LABELS})
        for q in group:
            cols=[("Input C",C,np.empty((0,3,3),np.float32))]
            for label in ordered:
                tri=loaded[label,q["request_id"]];cols.append((aliases[label],tri[:q["K"]],tri[q["K"]:]))
            p=dest/"blind"/(q["request_id"]+".png")
            ren._page(p,cols,bounds,f"{panel} {cid} n{q['candidate']} | source hidden | C blue / free orange")
            page=dict(request_id=q["request_id"],parent_id=q["parent_id"],condition_id=cid,panel=panel,candidate=q["candidate"],cohort=q["cohort"],blind_page=str(p),blind_sha256=digest(p),anonymous_order=[aliases[l] for l in ordered],labels=ordered,view_scope=[v[0] for v in ren.VIEWS],AI_VISUAL_REVIEW="PENDING",human_usability="NOT_RUN")
            if panel=="P1":
                p=dest/"reference"/(q["request_id"]+".png")
                ren._page(p,cols+[("Source reference",C,c["GT_free"])],bounds,f"{panel} {cid} n{q['candidate']} | one source completion, not unique valid target")
                page.update(reference_page=str(p),reference_sha256=digest(p))
            pages.append(page)
    result=dict(status="FINAL_COMPLETE",identity=identity,pages=pages,output_count=192,blind_pages=48,reference_pages=32,AI_VISUAL_REVIEW="PENDING",human_usability="NOT_RUN",mesh_modified=False,workers=1,threads=2,model_calls=0,wall_seconds=time.perf_counter()-tick)
    write_json(dest/"camera_manifest.json",dict(status="FINAL_COMPLETE",conditions=camera),freeze=True);write_json(saved,result,freeze=True)
    content=['<!doctype html><meta charset="utf-8"><title>Contextual alignment complete gallery</title>',
        '<h1>Complete RAW gallery: 192 outputs</h1><p>Anonymous method labels; C blue / free orange. Four views + wire. No postprocessing. AI review requires actual image inspection; human usability NOT_RUN.</p>',
        '<h2>1. Inspect all pages without source reference first</h2>']
    for p in pages:content.append(f'<p><a href="{html.escape(str(Path(p["blind_page"]).relative_to(out)).replace(chr(92),"/"))}">{html.escape(p["request_id"])}</a></p>')
    content.append('<h2>2. P1 source reference pages</h2>')
    for p in pages:
        if "reference_page" in p:content.append(f'<p><a href="{html.escape(str(Path(p["reference_page"]).relative_to(out)).replace(chr(92),"/"))}">{html.escape(p["request_id"])}</a></p>')
    content.append('<h2>3. Unblind only after recording observations</h2><p><a href="gallery/method_mapping.json">Method mapping</a></p>')
    (out/"OPEN_GALLERY.html").write_text("\n".join(content),encoding="utf8")
    return dict(status=result["status"],output_count=192,blind_pages=48,reference_pages=32,wall_seconds=result["wall_seconds"],AI_VISUAL_REVIEW="PENDING")
