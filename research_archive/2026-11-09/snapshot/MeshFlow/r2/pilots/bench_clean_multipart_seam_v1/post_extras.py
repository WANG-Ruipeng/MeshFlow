"""Read-only fixed-BASE denominators, point-jump and final safety accounting.

No detector reruns. Area metrics/I/counts/lengths retain their original domains.
Point jumps are explicitly not legal shared-edge connectivity.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
import argparse,collections,csv,json,sys,time
from . import post_cpu as P
import numpy as np
C=P.C;ROOT=P.ROOT


def rows(path):
    with C.resolve(path).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def bfs(seed,adj):
    found=set(seed);todo=list(found)
    while todo:
        i=todo.pop()
        for j in adj[i]-found:found.add(j);todo.append(j)
    return found


def point_jump(mesh,pairs,edges,q,seam_model):
    m=P.holes.initial(mesh);f=m["faces"];tri=m["output"].astype(float);K=q["K"]
    role=m["face_role"];seams=set(np.flatnonzero(role=="SEAM"));bulk=set(np.flatnonzero(role=="BULK"))
    area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    den=float(area[sorted(bulk)].sum());seed=set();adj=collections.defaultdict(set);badj=collections.defaultdict(set)
    for e in edges:
        if not e["valid_edge"]:continue
        i,j=map(int,e["incident_face_ids"])
        if min(i,j)<K:
            if max(i,j) in seams:seed.add(max(i,j))
        else:
            adj[i].add(j);adj[j].add(i)
            if i in bulk and j in bulk:badj[i].add(j);badj[j].add(i)
    reached=bfs(seed,adj);oldbulk=reached&bulk
    byvertex=collections.defaultdict(set)
    for i in bulk:
        for v in f[i]:byvertex[int(v)].add(i)
    classes={tuple(sorted(int(z)+K for z in r["pair"])):r["classification"] for r in pairs["FF"]["contacts"]}
    corners=m.get("source_corner_ids",m.get("base_source_corner_ids"));cvid=set(f[:K].ravel())
    contacts=[];pointseed=set()
    for sf in sorted(seams&reached):
        where=np.flatnonzero(corners[sf]==2)
        if len(where)!=1:raise ValueError("Seam tip source corner identity differs")
        vid=int(f[sf,where[0]])
        if vid in cvid:continue
        for bf in sorted(byvertex[vid]):
            shared=set(f[sf])&set(f[bf]);classification=classes.get(tuple(sorted((sf,bf))),"UNKNOWN_CACHE")
            accepted=len(shared)==1 and classification=="shared_vertex_contact"
            contacts.append(dict(seam_output_face=int(sf),bulk_output_face=int(bf),tip_vertex=vid,
                pair_classification=classification,shared_vertex_count=len(shared),accepted_point_diagnostic=accepted))
            if accepted:pointseed.add(bf)
    reachedpoints=bfs(pointseed,badj)&bulk;extra=reachedpoints-oldbulk
    applicable=seam_model and int(q["boundary_edges"])>0 and den>0
    result=dict(point_jump_status="PASS" if applicable else "NA_NOT_SEAM" if not seam_model else "NA_CLOSED_OR_EMPTY_BULK",
        point_jump_bulk_area_denominator=den,point_jump_verified_pair_count=len([r for r in contacts if r["accepted_point_diagnostic"]]),
        point_jump_reached_bulk_area=float(area[sorted(reachedpoints)].sum()),
        point_jump_extra_bulk_area=float(area[sorted(extra)].sum()),
        point_jump_reached_bulk_fraction=float(area[sorted(reachedpoints)].sum())/den if applicable else None,
        point_jump_extra_bulk_fraction=float(area[sorted(extra)].sum())/den if applicable else None,
        legal_edge_or_verified_point_bulk_fraction=float(area[sorted(oldbulk|reachedpoints)].sum())/den if applicable else None)
    return result,contacts


def fixed_fractions(mesh,pairs,shape,row,q):
    m=P.holes.initial(mesh);K=q["K"];den=int(m["base_free_face_count"]);whole=K+den
    n=len(m["faces"])-K
    out=dict(fixed_BASE_free_face_denominator=den,fixed_BASE_whole_face_denominator=whole,
             actual_output_free_face_denominator=n)
    for kind,scan in pairs.items():
        for typ,legal in (("proper",{"proper_transverse_penetration"}),("illegal",P.H.B.ILLEGAL)):
            faces=set()
            for r in scan["contacts"]:
                if r["classification"] in legal:
                    faces.update([int(r["pair"][1])] if kind=="CF" else map(int,r["pair"]))
            out["fixed_BASE_"+kind+"_"+typ+"_affected_free_count"]=len(faces)
            out["fixed_BASE_"+kind+"_"+typ+"_affected_free_fraction"]=len(faces)/den if den else None
    for name,key in (("minimum_angle_lt5","minimum_angle_lt5"),("q_gt10","q_actual_gt10"),
                     ("near_degenerate","area_near_degenerate"),("exact_zero_area","area_exact_zero")):
        count=sum(bool(r[key]) for r in shape["free"]["per_face"])
        out["fixed_BASE_free_"+name+"_count"]=count
        out["fixed_BASE_free_"+name+"_fraction"]=count/den if den else None
    for name,key in (("minimum_angle_lt5","minimum_angle_lt5"),("q_gt10","q_actual_gt10")):
        count=sum(bool(r[key]) for r in shape["whole"]["per_face"])
        out["fixed_BASE_whole_"+name+"_count"]=count
        out["fixed_BASE_whole_"+name+"_fraction"]=count/whole if whole else None
    old=row["free_exact_duplicate_excess_fraction"]
    count=float(old)*n if old is not None else None
    if count is not None and abs(count-round(count))>1e-6:
        raise AssertionError("Duplicate excess fraction not actual-face count")
    out["fixed_BASE_free_exact_duplicate_excess_count"]=int(round(count)) if count is not None else None
    out["fixed_BASE_free_exact_duplicate_excess_fraction"]=count/den if count is not None and den else None
    return out


def run(scope):
    dest=ROOT/"post_extras"/scope;done=dest/"completion.json"
    if done.exists():
        d=C.read(done);P.E.verify(d["files"]);return d
    if dest.exists():raise RuntimeError("Partial extras preserved")
    tick=time.perf_counter();C.write(dest/"resources_before.json",C.resources())
    qq=C.read(ROOT/"eval_manifest.json")["requests"];qby={q["request_id"]:q for q in qq}
    labels=list(P.LABELS)
    if scope=="all":
        from .eval_prepare import LABELS
        labels.extend(LABELS)
    topology={r["condition_id"]:r for r in rows(ROOT/"topology_audit/condition_topology_audit.csv")}
    outputs=[];contacts=[];safety=[];protected={};affected_shapes=[]
    for label in labels:
        for q in qq:
            rid=q["request_id"];old=label in P.LABELS
            cell=ROOT/("historical" if old else "cpu")/label/rid
            receipt=cell/"complete.json";P.E.verify(C.read(receipt)["files"])
            protected[str(receipt)]=C.sha(receipt)
            mrows={r["stage"]:r for r in C.read(cell/"metric_rows.json")}
            for stage in ("BASE_POST","CLEAN_FINAL_POST"):
                meshp=(P.SOURCE if old else ROOT)/"evaluation"/label/rid/"post/post.npz" if stage=="BASE_POST" else cell/stage/"post.npz"
                folder=cell/stage/"scores"
                mesh=C.arrays(meshp);pairs=C.read(folder/"pairs.json");shape=C.read(folder/"shape_per_face.json")
                row=mrows[stage]
                edges=[]
                for er in rows(folder/"valid_edges.csv"):
                    edges.append(dict(valid_edge=er["valid_edge"]=="True",incident_face_ids=json.loads(er["incident_face_ids"])))
                fixed=fixed_fractions(mesh,pairs,shape,row,q)
                point,detail=point_jump(mesh,pairs,edges,q,label!="NATIVE_S21000")
                top=topology[q["condition_id"]]
                metadata=dict(label=label,stage=stage,request_id=rid,condition_id=q["condition_id"],parent_id=q["parent_id"],
                    split=q["split"],part_type=q["part_type"],source_edge_components=int(top["source_edge_components"]),
                    topology_stratum=top["topology_stratum"],slit_edge_count=int(top["slit_edge_count"]),closed_C=q["closed_C"])
                outputs.append(dict(metadata,**fixed,**point))
                contacts.extend(dict(metadata,**r) for r in detail)
            final=C.read(cell/"base_final_paired_ledger.json")
            bridge=C.read(cell/"CLEAN_FINAL_POST/scores/bridge_shape.json")
            base_path=(P.SOURCE if old else ROOT)/"evaluation"/label/rid/"post/post.npz"
            original=C.arrays(base_path);alpha_mesh=C.arrays(cell/"ALPHA/post.npz")
            # Retained ALPHA is a finite-location vertex-cluster merge, NOT a
            # subsegment split. It must retain all original face/corner IDs.
            if len(alpha_mesh["faces"])!=len(original["faces"]) or not np.array_equal(
                    alpha_mesh["source_face_ids"],original["source_face_ids"]) or not np.array_equal(
                    alpha_mesh["source_corner_ids"],original["source_corner_ids"]):
                raise AssertionError("ALPHA unexpectedly introduced/deleted/reidentified faces")
            before_shape=P.H.PE.shape(original["output"]);after_shape=P.H.PE.shape(alpha_mesh["output"])
            changed=np.any(original["output"].view(np.uint32)!=alpha_mesh["output"].view(np.uint32),axis=(1,2))
            for fi in np.flatnonzero(changed):
                angle0=float(before_shape["angle"][fi]);angle1=float(after_shape["angle"][fi])
                q0=float(before_shape["q"][fi]);q1=float(after_shape["q"][fi])
                if angle1<5-1e-8 and angle1<angle0-1e-8:
                    raise AssertionError("Moved old face has new/worse small angle")
                if q1>10+1e-8 and q1>q0+1e-8:
                    raise AssertionError("Moved old face has new/worse high q")
                affected_shapes.append(dict(label=label,request_id=rid,post_face=int(fi),
                    original_raw_slot=int(original["source_face_ids"][fi]),new_child_face=False,
                    minimum_angle_before=angle0,minimum_angle_after=angle1,q_before=q0,q_after=q1,
                    retained_old_small_angle=angle1<5,retained_old_high_q=q1>10,
                    shape_guard="OLD_FACE_NO_NEW_OR_WORSE_BAD_SHAPE"))
            for face in bridge["per_face"]:
                if face["minimum_angle_degrees"]<5-1e-8 or face["q_actual"]>10+1e-8:
                    raise AssertionError("New bridge violates original shape guard")
            safety.append(dict(label=label,request_id=rid,**final["summary"],
                ALPHA_new_subfaces=0,ALPHA_source_face_and_corner_IDs_unchanged=True,
                HINGE_ALPHA_moved_old_faces=int(changed.sum()),
                new_bridge_count=len(bridge["per_face"]),
                min_new_bridge_angle=min([r["minimum_angle_degrees"] for r in bridge["per_face"]],default=None),
                max_new_bridge_q=max([r["q_actual"] for r in bridge["per_face"]],default=None),
                **final["full_indexed_guard"]))
    C.csvwrite(dest/"per_output_fixed_BASE_and_pointjump.csv",outputs)
    C.csvwrite(dest/"point_contacts.csv",contacts)
    C.csvwrite(dest/"safety_ledger.csv",safety)
    C.csvwrite(dest/"moved_old_faces_shape_audit.csv",affected_shapes)
    policy=dict(status="FROZEN_DERIVED_DEFINITION",version="fixed_BASE_actual_affected_face_count_v1",
        fields=[k for k in outputs[0] if k.startswith("fixed_BASE")],
        denominator="BASE_POST number of free faces (or wholeBASEforwholefacefractions)",
        numerator="Actual current affected face count, including newBRIDGE; can exceed1 if newfacecountlarge. Not oldparent-union fraction.",
        original179_unchanged=True,other_domains="I retains Cqueryweight; area ratios retain actual named area; counts/lengths unchanged",
        point_jump="Verified exact tip shared_vertex_contact then legal BULK edges; diagnostic only, not certifiedconnection")
    C.write(dest/"metric_policy.json",policy)
    if "torch" in sys.modules:raise AssertionError("CPU extras imported neural framework")
    result=dict(status="COMPLETE",scope=scope,rows=len(outputs),safety_rows=len(safety),seconds=time.perf_counter()-tick,
        model_calls=0,GPU_calls=0,new_scores=0,new_pair_scans=0,protected=protected,code_sha256=C.sha(__file__),
        files={str(p):C.sha(p) for p in dest.rglob("*") if p.is_file()})
    C.write(done,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scope",choices=("historical","all"),default="historical")
    a=p.parse_args();print(run(a.scope))
