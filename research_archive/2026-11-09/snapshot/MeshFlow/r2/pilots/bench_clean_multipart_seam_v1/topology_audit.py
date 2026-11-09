"""Source-ID topology and geometric slit candidates; no model/scoring calls.

Slit is a geometry predicate, not a wrong-interface label. Source Null/alias
means one reference completion only; no unique-target claim is made.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_k]="1"
import argparse, collections, sys, time
from pathlib import Path
import numpy as np
from r2.pilots.seam_abc_v1 import eval_common as C

ROOT=C.ROOT/"experiments/r2_bench_clean_multipart_seam_v1"
SOURCE=C.ROOT/"experiments/r2_bench_parts_convergence_v1"
R=.04197
VERSION="source_ID_components_directed_slit_geometry_candidate_v1"


def components(faces):
    adjacency=collections.defaultdict(set);owners=collections.defaultdict(list)
    for i,f in enumerate(faces):
        for k in range(3):
            owners[tuple(sorted((int(f[k]),int(f[(k+1)%3]))))].append(i)
    for ff in owners.values():
        for i in ff:
            adjacency[i].update(set(ff)-{i})
    labels=np.full(len(faces),-1,int);count=0
    for i in range(len(faces)):
        if labels[i]>=0:continue
        labels[i]=count;todo=[i]
        while todo:
            z=todo.pop()
            for j in adjacency[z]:
                if labels[j]<0:labels[j]=count;todo.append(j)
        count+=1
    return labels,count,owners


def point_segment(p,a,b):
    axis=b-a;den=float(axis@axis)
    t=float((p-a)@axis)/den if den else 0.
    return float(np.linalg.norm(p-(a+np.clip(t,0,1)*axis))),t


def segment_distance(a,b,c,d):
    """Exact finite segment min: endpoint cases plus feasible interior min."""
    u,v,w=b-a,d-c,a-c
    A,B,D=float(u@u),float(u@v),float(v@v)
    E,F=float(u@w),float(v@w);det=A*D-B*B
    vals=[point_segment(a,c,d)[0],point_segment(b,c,d)[0],
          point_segment(c,a,b)[0],point_segment(d,a,b)[0]]
    if det>1e-14*max(1.,A*D):
        s=(B*F-D*E)/det;t=(A*F-B*E)/det
        if 0<=s<=1 and 0<=t<=1:vals.append(float(np.linalg.norm(w+s*u-t*v)))
    return min(vals)


def slit_pairs(vertices,edges,source_ids):
    rows=[];flags=np.zeros(len(edges),bool)
    for i,(fi,u,w) in enumerate(edges):
        a,b=np.asarray(vertices[[u,w]],float);axis=b-a;length=np.linalg.norm(axis)
        for j in range(i+1,len(edges)):
            fj,x,y=edges[j]
            if set(source_ids[[u,w]].tolist())&set(source_ids[[x,y]].tolist()):continue
            c,d=np.asarray(vertices[[x,y]],float);other=d-c;ln=np.linalg.norm(other)
            if length<=0 or ln<=0:continue
            cosine=float(axis@other/(length*ln))
            if cosine>=-np.cos(np.deg2rad(15)):continue
            _,t=point_segment((a+b)/2,c,d);_,s=point_segment((c+d)/2,a,b)
            if not (.1<=t<=.9 or .1<=s<=.9):continue
            distance=segment_distance(a,b,c,d)
            if distance>2*R:continue
            flags[i]=flags[j]=True
            rows.append(dict(edge_i=i,edge_j=j,C_face_i=int(fi),C_face_j=int(fj),
                source_edge_i=source_ids[[u,w]].tolist(),source_edge_j=source_ids[[x,y]].tolist(),
                cosine_directed_axis=cosine,midpoint_i_projection_on_j=t,midpoint_j_projection_on_i=s,
                segment_distance_model=distance,segment_distance_L=distance/C.L,
                geometric_candidate=True,semantic_illegality="UNKNOWN_NOT_IMPLIED"))
    return flags,rows


def run():
    dest=ROOT/"topology_audit";done=dest/"completion.json"
    if done.exists():
        r=C.read(done)
        for p,h in r["files"].items():
            if C.sha(p)!=h:raise ValueError("Topology output changed")
        return r
    if dest.exists():raise RuntimeError("Partial topology attempt retained")
    if not (ROOT/"registration.json").exists():raise RuntimeError("No root registration")
    tick=time.perf_counter();C.write(dest/"resources_before.json",C.resources())
    manifest=SOURCE/"data/manifest.json";m=C.read(manifest)
    qs=C.read(SOURCE/"evaluation/score_registrations/SEAM_S15000.json")["requests"]
    evalc={q["condition_id"] for q in qs}
    if len(m["rows"])!=316 or len(evalc)!=24:raise ValueError("Fixed panel differs")
    protected={str(manifest):C.sha(manifest)};top=[];edges_out=[];pairs_out=[];bycid={}
    for row in m["rows"]:
        cid=row["condition_id"];path=C.resolve(row["case_path"]);protected[str(path)]=C.sha(path)
        a=C.arrays(path);v=a["C_vertices"];f=a["C_faces_original"];ids=a["C_source_vertex_ids"]
        labels,ncomp,inc=components(ids[f])
        e=a["edge_records"];sf=a["source_supported_labels"].astype(bool);own=a["occupied_labels"].astype(bool)
        alias=a["source_alias_labels"].astype(bool)
        if not np.array_equal(sf,own|alias):raise ValueError("Unexpected source role overlap")
        flags,slits=slit_pairs(v,e,ids)
        B=len(e);K=len(f)
        if B!=sum(len(x)==1 for x in inc.values()):raise ValueError("Boundary ID mismatch")
        if int(row["source_edge_components"])!=ncomp:raise ValueError("Existing component audit differs")
        strata=("MULTICOMPONENT_OR_LEG" if ncomp>1 or row["part_type"]=="LEG_SUPPORT"
                else "SINGLE_COHERENT_MAIN")
        info=dict(condition_id=cid,parent_id=row["parent_id"],split=row["group"],part_type=row["part_type"],
            N=int(a["N"]),K=K,source_edge_components=ncomp,closed_C=B==0,boundary_edges=B,
            topology_stratum=strata,part_scope="SOURCE_REVIEWED_LABEL_NO_SEMANTIC_INPUT",
            is_fixed_evaluation=cid in evalc,slit_pair_count=len(slits),slit_edge_count=int(flags.sum()),
            source_owner_count=int(own.sum()),source_alias_count=int(alias.sum()),source_null_count=int((~sf).sum()),
            slit_source_owner_count=int((flags&own).sum()),slit_source_alias_count=int((flags&alias).sum()),
            slit_source_null_count=int((flags&~sf).sum()),
            eligible_training_filter=row["group"]=="TRAIN" and B>0 and bool(own.any()),
            connectivity_definition="C source vertex-ID shared-edge components; no coordinate welding")
        top.append(info);bycid[cid]=dict(row=info,flags=flags,owner=own,alias=alias,supported=sf)
        for i,(face,u,w) in enumerate(e):
            edges_out.append(dict(condition_id=cid,parent_id=row["parent_id"],split=row["group"],
                part_type=row["part_type"],is_fixed_evaluation=cid in evalc,edge_id=i,C_face_id=int(face),
                C_vertex_ids=[int(u),int(w)],source_vertex_ids=ids[[u,w]].tolist(),
                C_component=int(labels[face]),is_slit_candidate=bool(flags[i]),
                source_role="OWNER" if own[i] else "ALIAS" if alias[i] else "NULL",
                source_role_scope="REFERENCE_COMPLETION_DIAGNOSTIC_NOT_UNIQUE_REQUIRED_INTERFACE"))
        pairs_out.extend(dict(condition_id=cid,parent_id=row["parent_id"],split=row["group"],
                              part_type=row["part_type"],is_fixed_evaluation=cid in evalc,**p) for p in slits)
    C.csvwrite(dest/"condition_topology_audit.csv",top)
    C.csvwrite(dest/"internal_slit_edges.csv",edges_out)
    C.csvwrite(dest/"internal_slit_pairs.csv",pairs_out)
    C.csvwrite(dest/"evaluation_condition_topology.csv",[r for r in top if r["is_fixed_evaluation"]])
    slots=[]
    for label in ("SEAM_S15000","SEAM_S21000"):
        for q in qs:
            rid=q["request_id"];cid=q["condition_id"];root=SOURCE/"evaluation"/label/rid
            rp=root/"raw/raw.npz";pp=root/"post/post.npz"
            pairp=SOURCE/"evaluation/scores"/label/rid/"POST/pairs.json"
            scorep=pairp.parent/"score.json"
            for p in (rp,pp,pairp,scorep):protected[str(p)]=C.sha(p)
            raw=C.arrays(rp);post=C.arrays(pp);pairs=C.read(pairp);score=C.read(scorep);K=q["K"]
            info=bycid[cid];roles=info["row"]
            mapping={int(slot):i for i,slot in enumerate(post["source_face_ids"])}
            for ei in range(q["boundary_edges"]):
                on=bool(raw["occupied"][ei]);slotids=np.flatnonzero(raw["raw_slot_edge_id"]==ei)
                slotids=[int(s) for s in slotids if raw["raw_seam_slot_mask"][s]]
                if on and len(slotids)!=1:raise ValueError("Occupied edge lacks unique seam slot")
                slot=slotids[0] if slotids else None
                face=mapping.get(slot) if slot is not None else None
                counts=collections.Counter()
                if face is not None:
                    for kind,scan in pairs.items():
                        for pair in scan["contacts"]:
                            i,j=map(int,pair["pair"]);j+=K
                            if kind=="FF":i+=K
                            if face in (i,j):counts[(kind,pair["classification"])]+=1
                slots.append(dict(label=label,request_id=rid,parent_id=q["parent_id"],condition_id=cid,
                    split=q["split"],part_type=q["part_type"],edge_id=ei,is_slit_candidate=bool(info["flags"][ei]),
                    source_role="OWNER" if info["owner"][ei] else "ALIAS" if info["alias"][ei] else "NULL",
                    old_predicted_active=on,old_logit=float(raw["occupancy_logits"][ei]),raw_slot=slot,
                    retained_post_face=face,deleted_active_slot=on and face is None,
                    CF_illegal_pairs=sum(n for (kind,cl),n in counts.items() if kind=="CF" and cl in {"proper_transverse_penetration","coplanar_positive_area_overlap"}),
                    FF_illegal_pairs=sum(n for (kind,cl),n in counts.items() if kind=="FF" and cl in {"proper_transverse_penetration","coplanar_positive_area_overlap"}),
                    contact_classification_counts={kind+"|"+cl:n for (kind,cl),n in counts.items()},
                    output_I_dup=score["metrics"].get("ConflictCov_C_dup"),
                    I_dup_per_slot=None,I_dup_per_slot_status="NA_CACHE_HAS_QUERY_FLAGS_NOT_FREE_FACE_ATTRIBUTION"))
    C.csvwrite(dest/"old_layout_slit_slot_fates.csv",slots)
    summary=[]
    for split in ("TRAIN","DEV","EVAL"):
        rr=[r for r in top if r["is_fixed_evaluation"]] if split=="EVAL" else [r for r in top if r["split"]==split]
        summary.append(dict(scope=split,conditions=len(rr),parents=len({r["parent_id"] for r in rr}),
            open_conditions=sum(not r["closed_C"] for r in rr),closed_conditions=sum(r["closed_C"] for r in rr),
            multi_component_conditions=sum(r["source_edge_components"]>1 for r in rr),
            open_edges=sum(r["boundary_edges"] for r in rr),slit_edges=sum(r["slit_edge_count"] for r in rr),
            slit_pairs=sum(r["slit_pair_count"] for r in rr),slit_source_null=sum(r["slit_source_null_count"] for r in rr)))
    C.write(dest/"summary.json",summary)
    for p,h in protected.items():
        if C.sha(p)!=h:raise ValueError("Protected input changed")
    if "torch" in sys.modules:raise AssertionError("Topology audit imported neural framework")
    result=dict(status="COMPLETE",version=VERSION,conditions=316,evaluation_conditions=24,
        slit_rule="Source-ID nonshared endpoints; directed axis cos < -cos15; either midpoint projection in[.1,.9]; exact segment gap<=2*.04197; candidate only",
        old_head_training_predictions="NA_NOT_IN_SAVED_CACHE; independent head audit may append separately",
        source_role_scope="Reference completion only, not unique task correctness",
        seconds=time.perf_counter()-tick,model_calls=0,GPU_calls=0,new_scores=0,
        summary=summary,protected=protected,code_sha256=C.sha(__file__),
        files={str(p):C.sha(p) for p in dest.rglob("*") if p.is_file()})
    C.write(done,result);return result


if __name__=="__main__":
    result=run();print(result["summary"],flush=True)
