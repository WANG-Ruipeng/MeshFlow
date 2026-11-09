#!/usr/bin/env python3
"""Read-only source-data geometry audit. No model, OT, repair or mesh output."""
from __future__ import annotations
import os
for _n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_n]="2"
import argparse,csv,hashlib,itertools,json,math,sys,time,resource
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
REPO=Path("/mnt/e/MeshFlow-Control")
sys.path.insert(0,str(REPO/"src"))
from meshflow_control.evaluation.pair_audit import pair_scan,PROPER,COPLANAR
from meshflow_control.evaluation.intersections import DETECTOR_POLICY
from meshflow_control.evaluation.shape import shape_quality
L=math.sqrt(3)*2/.3762
VERSION="chair_user_condition_geometry_audit_v1"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if p.exists():
        if p.read_text()!=data:raise RuntimeError("Immutable audit differs: "+str(p))
        return
    tmp=p.with_suffix(p.suffix+".tmp")
    if tmp.exists():
        if tmp.read_text()!=data:raise RuntimeError("Interrupted temp differs: "+str(tmp))
    else:
        with tmp.open("x",encoding="utf-8") as f:f.write(data);f.flush();os.fsync(f.fileno())
    tmp.rename(p)
def event(out,kind,**kw):
    x={"time_unix":time.time(),"event":kind,**kw}
    with (out/"ledger.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(x)+"\n");f.flush();os.fsync(f.fileno())
    print(json.dumps(x),flush=True)
def quantiles(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1);x=x[np.isfinite(x)]
    return {str(p):float(np.quantile(x,p)) if len(x) else None for p in (0,.05,.25,.5,.75,.95,1)}
def edge_graph(faces,vertex_coordinates):
    incidence=defaultdict(set)
    for i,f in enumerate(faces):
        for a,b in ((0,1),(1,2),(2,0)):incidence[tuple(sorted((f[a],f[b])))].add(i)
    incidence={e:sorted(v) for e,v in incidence.items()}
    return {"incidence":incidence,"coordinates":vertex_coordinates}
def scope_topology(graph,ids,areas,original,total_area):
    ids=set(map(int,ids));inc={e:[i for i in v if i in ids] for e,v in graph["incidence"].items()}
    inc={e:v for e,v in inc.items() if v};adj={i:set() for i in ids}
    for v in inc.values():
        for i in v:adj[i].update(set(v)-{i})
    todo=set(ids);components=[]
    scope_area=float(areas[list(ids)].sum()) if ids else 0.
    while todo:
        start=min(todo);seen=set();stack=[start]
        while stack:
            i=stack.pop()
            if i not in seen:seen.add(i);stack.extend(adj[i]-seen)
        todo-=seen;fs=sorted(seen);a=float(areas[fs].sum())
        components.append({"source_face_ids":fs,"original_face_ids":[int(original[i]) for i in fs],
            "face_count":len(fs),"area":a,"area_fraction_scope":a/scope_area if scope_area else None,
            "area_fraction_parent":a/total_area if total_area else None})
    components.sort(key=lambda r:(-r["area"],r["source_face_ids"]))
    boundary=[e for e,v in inc.items() if len(v)==1];bad=[e for e,v in inc.items() if len(v)>2]
    def edge_record(e):
        xyz=[list(map(float,graph["coordinates"][q])) for q in e]
        return {"vertices":list(e),"coordinates":xyz,"length":float(np.linalg.norm(np.array(xyz[1])-xyz[0])),
            "source_face_ids":inc[e],"original_face_ids":[int(original[i]) for i in inc[e]]}
    br=[edge_record(e) for e in boundary];bd=defaultdict(set)
    for a,b in boundary:
        if a!=b:bd[a].add(b);bd[b].add(a)
    bseen=set();bcomps=[]
    for first in sorted(bd):
        if first in bseen:continue
        seen=set();stack=[first]
        while stack:
            a=stack.pop()
            if a not in seen:seen.add(a);stack.extend(bd[a]-seen)
        bseen|=seen;bcomps.append({"vertex_count":len(seen),"all_degree_two":all(len(bd[a])==2 for a in seen)})
    branches=[{"vertex":a,"coordinate":list(map(float,graph["coordinates"][a])),"boundary_degree":len(v)}
              for a,v in sorted(bd.items()) if len(v)>2]
    return {"component_count":len(components),"components":components,
        "small_component_policy":"No invented size cutoff; all component face counts and areas retained",
        "smallest_component_face_count":min([r["face_count"] for r in components],default=0),
        "smallest_component_area_fraction":min([r["area_fraction_scope"] for r in components if r["area_fraction_scope"] is not None],default=None),
        "single_face_components":sum(r["face_count"]==1 for r in components),
        "boundary_edge_count":len(br),"boundary_length":sum(r["length"] for r in br),
        "boundary_length_over_L":sum(r["length"] for r in br)/L,"boundary_edges":br,
        "boundary_branch_vertices":branches,"boundary_branch_count":len(branches),
        "boundary_endpoint_count":sum(len(v)==1 for v in bd.values()),
        "boundary_component_count":len(bcomps),"closed_degree2_boundary_components":sum(c["all_degree_two"] for c in bcomps),
        "nonmanifold_edge_count":len(bad),"nonmanifold_edges":[edge_record(e) for e in bad],
        "collapsed_edge_count":sum(e[0]==e[1] for e in inc)}
def free_attachment(graph,known,free_topology,areas,original,total_area):
    known=set(map(int,known));interface=[]
    for e,v in graph["incidence"].items():
        c=sorted(known.intersection(v));f=sorted(set(v)-known)
        if c and f:
            xyz=np.asarray([graph["coordinates"][q] for q in e],dtype=np.float64)
            interface.append({"vertices":list(e),"C_source_face_ids":c,"F_source_face_ids":f,
                "length":float(np.linalg.norm(xyz[1]-xyz[0])),"total_retained_incidence":len(v)})
    touched={i for e in interface for i in e["F_source_face_ids"]}
    components=[]
    for comp in free_topology["components"]:
        r=dict(comp);r["shares_complete_edge_with_C"]=bool(touched.intersection(comp["source_face_ids"]));components.append(r)
    detached=[r for r in components if not r["shares_complete_edge_with_C"]]
    free_area=sum(r["area"] for r in components);da=sum(r["area"] for r in detached)
    return {"interface_edges":interface,"interface_edge_count":len(interface),
        "interface_length":sum(r["length"] for r in interface),"F_components":components,
        "F_components_without_complete_edge_to_C":detached,"unattached_F_component_count":len(detached),
        "unattached_F_source_face_ids":sorted(i for r in detached for i in r["source_face_ids"]),
        "unattached_F_area":da,"unattached_F_area_fraction_of_F":da/free_area if free_area else None,
        "unattached_F_area_fraction_of_parent":da/total_area if total_area else None,
        "interpretation":"No complete shared edge path through F to C. May represent legitimate separate parts; not a semantic badness label."}
def basic_geometry(tri,ids,original,total_area):
    ids=list(map(int,ids));t=tri[ids].astype(np.float64);finite=np.isfinite(t).all(axis=(1,2))
    out={"faces":len(ids),"finite_coordinate_values":int(np.isfinite(t).sum()),"coordinate_value_count":int(t.size),
        "nonfinite_source_face_ids":[ids[i] for i in np.flatnonzero(~finite)]}
    if not finite.all():out["status"]="BLOCKED_NONFINITE";return out
    area=.5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    groups=defaultdict(list)
    for i,x in zip(ids,t):groups[tuple(sorted(tuple(float(v) for v in q) for q in x))].append(i)
    duplicates=[v for v in groups.values() if len(v)>1]
    quality=shape_quality(t,L);angles=[]
    for c in range(3):
        u,v=t[:,(c+1)%3]-t[:,c],t[:,(c+2)%3]-t[:,c]
        den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
        cosine=np.divide((u*v).sum(1),den,out=np.ones(len(t)),where=den>0)
        angles.append(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    angles=np.stack(angles,axis=1);valid_angles=area>0
    out.update(status="PASS",area=float(area.sum()),area_fraction_parent=float(area.sum()/total_area) if total_area else None,
        exact_zero_area_source_face_ids=[ids[i] for i in np.flatnonzero(area==0)],
        exact_zero_area_count=int((area==0).sum()),exact_duplicate_groups=duplicates,
        exact_duplicate_extra_faces=sum(len(g)-1 for g in duplicates),
        exact_duplicate_original_groups=[[int(original[i]) for i in g] for g in duplicates],
        near_degenerate_area_count=quality["near_degenerate_area_count"],
        near_degenerate_area_threshold_L_squared=quality["near_degenerate_area_threshold"],
        angle_status="Zero-area face angles excluded from distribution and reported separately",
        all_valid_angles_quantiles_degrees=quantiles(angles[valid_angles]),
        minimum_valid_face_angle_quantiles_degrees=quantiles(angles[valid_angles].min(axis=1)) if valid_angles.any() else quantiles([]),
        minimum_angle_lt5_count=int((angles[valid_angles].min(axis=1)<5).sum()) if valid_angles.any() else 0,
        minimum_angle_lt5_fraction=float((angles[valid_angles].min(axis=1)<5).mean()) if valid_angles.any() else None,
        minimum_angle_threshold_source="Existing evaluation/shape.py and quality.py five-degree diagnostic; not an acceptance gate",
        per_face=[{"source_face_id":i,"original_face_id":int(original[i]),"area":float(area[j]),
                   "angles_degrees":angles[j].tolist() if valid_angles[j] else None,
                   "minimum_angle_degrees":float(angles[j].min()) if valid_angles[j] else None}
                  for j,i in enumerate(ids)])
    return out
def load_parent(data,parent):
    p=data/parent["npz"]
    if sha(p)!=parent["sha256"]:raise ValueError("Parent NPZ hash differs")
    with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    tri=a["full_target"].reshape(-1,3,3);N=len(tri);ids=a["source_face_vertex_ids"]
    if N!=parent["N"] or not np.array_equal(a["model_vertices"][ids],tri):raise ValueError("Source corner mapping")
    if not np.array_equal(a["original_source_faces"][a["retained_to_original_face"]],a["source_faces"]):raise ValueError("Retained original face mapping")
    if not np.array_equal(np.take_along_axis(a["source_faces"],a["source_corner_permutations"],axis=1),ids):raise ValueError("Corner permutation")
    if not np.array_equal(a["pre_ot_full"]*np.float32(2),tri):raise ValueError("Whole-parent model scale")
    source_faces=[tuple(map(int,f)) for f in ids];v={i:tuple(map(float,x)) for i,x in enumerate(a["model_vertices"])}
    coordinate_faces=[tuple(tuple(map(float,q)) for q in face) for face in tri]
    cv={q:q for face in coordinate_faces for q in face}
    graphs={"source_index":edge_graph(source_faces,v),"exact_coordinate":edge_graph(coordinate_faces,cv)}
    area=.5*np.linalg.norm(np.cross(tri[:,1].astype(float)-tri[:,0],tri[:,2].astype(float)-tri[:,0]),axis=1)
    return a,tri,graphs,area
def basic_parent(data,parent,tasks):
    a,tri,graphs,areas=load_parent(data,parent);original=a["retained_to_original_face"];total=float(areas.sum());N=len(tri)
    base={"uid":parent["uid"],"N":N,"L":L,"scale_policy":"Fixed canonical training cube diagonal sqrt(3)*2/.3762, shared by parent/C/F; not output-fitted",
        "input_npz":str(data/parent["npz"]),"input_npz_sha256":parent["sha256"],"source_id_policy":"source_face_id is retained row (source corners may be canonically permuted); original_face_id maps to raw row",
        "geometry":basic_geometry(tri,range(N),original,total),
        "topology":{k:scope_topology(g,range(N),areas,original,total) for k,g in graphs.items()},
        "near_duplicate_status":"NA_NO_REGISTERED_QUERY_BANK","near_duplicate_reason":"Existing I duplicate-band estimator requires area-sampled C query support. No new random queries or detector/semantic thresholds introduced.",
        "repair_metadata":json.loads(str(a["repair_metadata_json"].item()))}
    cond=[]
    for task in tasks:
        known=list(map(int,task["source_face_ids"]));free=list(map(int,task["free_source_ids"]))
        if len(known)!=task["K"] or sorted(known+free)!=list(range(N)) or len(set(known+free))!=N:raise ValueError("Task partition")
        c={"task_id":task["task_id"],"uid":parent["uid"],"N":N,"K":len(known),"ratio":task["ratio"],
           "source_face_ids":known,"free_source_ids":free,"L":L,
           "geometry_C":basic_geometry(tri,known,original,total),"geometry_F":basic_geometry(tri,free,original,total),
           "topology":{},"attachment":{},"near_duplicate_status":base["near_duplicate_status"]}
        for key,g in graphs.items():
            ct=scope_topology(g,known,areas,original,total);ft=scope_topology(g,free,areas,original,total)
            c["topology"][key]={"C":ct,"F":ft}
            c["attachment"][key]=free_attachment(g,known,ft,areas,original,total)
        cond.append(c)
    return base,cond
def partition_contacts(scan,known,N):
    known=set(known);nC=len(known);nF=N-nC
    result={}
    for mode in ("verified","primary","independent"):
        groups={k:{"classification_counts":Counter(),"proper_pairs":[],"coplanar_positive_area_pairs":[],"unresolved_pairs":[]} for k in ("CC","CF","FF")}
        for row in scan["contacts"]:
            i,j=row["pair"];k="CC" if i in known and j in known else "FF" if i not in known and j not in known else "CF"
            label=row["classification"] if mode=="verified" else row["detector"].get(mode,{}).get("classification",row["classification"])
            g=groups[k];g["classification_counts"][label]+=1
            pair=[i,j] if k!="CF" or i in known else [j,i]
            if label==PROPER:g["proper_pairs"].append(pair)
            if label==COPLANAR:g["coplanar_positive_area_pairs"].append(pair)
            if "unresolved" in label.lower() or "degenerate" in label.lower():g["unresolved_pairs"].append(pair)
        for k,g in groups.items():
            expected={"CC":nC*(nC-1)//2,"CF":nC*nF,"FF":nF*(nF-1)//2}[k]
            g["classification_counts"]["disjoint"]+=expected-sum(g["classification_counts"].values())
            g["classification_counts"]=dict(g["classification_counts"]);g["expected_pairs"]=expected
            g["proper_pair_count"]=len(g["proper_pairs"]);g["coplanar_overlap_pair_count"]=len(g["coplanar_positive_area_pairs"])
            g["unresolved_pair_count"]=len(g["unresolved_pairs"])
            g["proper_affected_source_face_ids"]=sorted({i for pair in g["proper_pairs"] for i in pair})
            g["coplanar_affected_source_face_ids"]=sorted({i for pair in g["coplanar_positive_area_pairs"] for i in pair})
            g["status"]="PARTIAL_UNRESOLVED" if g["unresolved_pairs"] else "PASS_STATIC_SCAN"
        result[mode]=groups
    return result
def csv_write(path,rows):
    fields=sorted({k for r in rows for k in r})
    temp=path.with_suffix(".csv.tmp")
    with temp.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,separators=(",",":")) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    if path.exists():
        if path.read_bytes()!=temp.read_bytes():raise ValueError("Immutable CSV differs")
        temp.unlink()
    else:temp.rename(path)
def flat_parent(p):
    r={"uid":p["uid"],"N":p["N"],"geometry_status":p["geometry"]["status"]}
    g=p["geometry"]
    for k in ("area","exact_zero_area_count","exact_duplicate_extra_faces","near_degenerate_area_count","minimum_angle_lt5_count","minimum_angle_lt5_fraction"):r[k]=g[k]
    r["minimum_face_angle_quantiles"]=g["minimum_valid_face_angle_quantiles_degrees"]
    for mode,t in p["topology"].items():
        for k in ("component_count","boundary_edge_count","boundary_length","boundary_branch_count","nonmanifold_edge_count","smallest_component_area_fraction"):r[mode+"_"+k]=t[k]
    return r
def flat_condition(c):
    r={k:c[k] for k in ("task_id","uid","N","K","ratio","source_face_ids","free_source_ids","near_duplicate_status")}
    for scope in ("C","F"):
        g=c["geometry_"+scope]
        for k in ("faces","area","area_fraction_parent","exact_zero_area_count","exact_duplicate_extra_faces","near_degenerate_area_count","minimum_angle_lt5_count","minimum_angle_lt5_fraction"):r[scope+"_"+k]=g[k]
        r[scope+"_minimum_angle_quantiles"]=g["minimum_valid_face_angle_quantiles_degrees"]
    for mode in c["topology"]:
        for scope in ("C","F"):
            t=c["topology"][mode][scope]
            for k in ("component_count","boundary_edge_count","boundary_length","boundary_branch_count","nonmanifold_edge_count","smallest_component_area_fraction"):r[mode+"_"+scope+"_"+k]=t[k]
        for k in ("interface_edge_count","interface_length","unattached_F_component_count","unattached_F_area_fraction_of_F","unattached_F_source_face_ids"):r[mode+"_"+k]=c["attachment"][mode][k]
    return r
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--stage",choices=("basic","intersections","all"),default="all");a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    data=Path("/mnt/e/MeshFlow-Control-artifacts/data")
    plan_path=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1/streams/recipe_plan.json")
    manifest_path=data/"train_manifest.json";manifest=read(manifest_path);uids=set(read(plan_path)["eligible_parents"])
    parents=sorted([r for r in manifest["parents"] if r["uid"] in uids],key=lambda r:r["uid"])
    tasks=sorted([r for r in manifest["tasks"] if r["uid"] in uids and r["ratio"] in (.2,.4)],key=lambda r:r["task_id"])
    assert len(parents)==26 and len(tasks)==411 and Counter(r["ratio"] for r in tasks)=={.2:208,.4:203}
    sources=[Path(__file__).resolve()]+[REPO/"src/meshflow_control/evaluation"/n for n in ("pair_audit.py","intersections.py","seams.py","shape.py")]+[REPO/"src/meshflow_control/__init__.py",REPO/"src/meshflow_control/evaluation/__init__.py"]
    source_hash={str(p):sha(p) for p in sources};inputs={str(p):sha(p) for p in [manifest_path,plan_path]+[data/r["npz"] for r in parents]}
    definition={"schema":VERSION,"implementation_sha256":sha(__file__),"source_hashes":source_hash,"input_hashes":inputs,
        "parents":26,"conditions":411,"ratio_counts":{"0.2":208,"0.4":203},"L":L,
        "connectivity":"Full shared edge: original source vertex IDs versus exact stored model-coordinate tuples; signed zero equal, no tolerance welding",
        "boundary":"Scope-only edges incident on exactly one unique face; branch vertex boundary degree>2; all component IDs/areas retained, no semantic threshold",
        "detector":DETECTOR_POLICY,"detector_reuse":"One full unique parent pair scan independent=True; same stored results partitioned CC/CF/FF for all conditions",
        "near_duplicates":"NA_NO_REGISTERED_QUERY_BANK; no extra random sampling",
        "limits":["Index nonmanifold/coordinate contacts are not semantic defect labels","Multiple components may be legitimate chair parts","Shared-edge graph does not include near-contact, vertex-only contact or finite-area overlap","Scales/thresholds are diagnostic, not cleaning decisions"],
        "resource_contract":{"workers":1,"numeric_threads":2,"GPU":0},"model_calls":0,"OT_calls":0,"new_meshes":0,"source_data_modified":0}
    write(out/"definitions.json",definition)
    for p in sources:
        target=out/"source_snapshot"/(p.name if p==Path(__file__).resolve() else p.relative_to(REPO).as_posix());target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(p.read_bytes())
        if sha(target)!=sha(p):raise ValueError("Source snapshot differs")
    existing_receipt=out/("receipt_"+a.stage+".json")
    if existing_receipt.exists():
        old=read(existing_receipt)
        if old["input_hashes_after"]!=inputs or old["source_hashes_after"]!=source_hash:raise ValueError("Completed receipt identity differs")
        print(json.dumps({"status":"ALREADY_COMPLETE","receipt":str(existing_receipt)}));return
    start=time.monotonic();event(out,"START",stage=a.stage,parents=26,conditions=411,expected_pairs=sum(r["N"]*(r["N"]-1)//2 for r in parents))
    if a.stage in ("basic","all"):
        for parent in parents:
            p=out/"parents_basic"/(parent["uid"]+".json")
            if p.exists() and all((out/"conditions_basic"/(t["task_id"]+".json")).exists() for t in tasks if t["uid"]==parent["uid"]):continue
            tick=time.monotonic();base,conditions=basic_parent(data,parent,[t for t in tasks if t["uid"]==parent["uid"]])
            write(p,base)
            for c in conditions:write(out/"conditions_basic"/(c["task_id"]+".json"),c)
            event(out,"BASIC_PARENT_COMPLETE",uid=parent["uid"],conditions=len(conditions),seconds=time.monotonic()-tick)
        csv_write(out/"parents_basic.csv",[flat_parent(read(out/"parents_basic"/(r["uid"]+".json"))) for r in parents])
        csv_write(out/"conditions_basic.csv",[flat_condition(read(out/"conditions_basic"/(r["task_id"]+".json"))) for r in tasks])
    if a.stage in ("intersections","all"):
        for parent in parents:
            uid=parent["uid"];dest=out/"parent_intersections"/(uid+".json")
            if dest.exists():continue
            tick=time.monotonic();arrays,tri,_,_=load_parent(data,parent);ledger={}
            scan=pair_scan(tri,tri,L,self_pairs=True,independent=True,ledger=ledger)
            for row in scan["contacts"]:row["original_face_ids"]=[int(arrays["retained_to_original_face"][i]) for i in row["pair"]]
            scan.update(uid=uid,input_npz_sha256=parent["sha256"],L=L,ledger=ledger,
                scientific_scope="All unique retained source-parent pairs; no generated mesh",seconds=time.monotonic()-tick)
            write(dest,scan);event(out,"INTERSECTION_PARENT_COMPLETE",uid=uid,seconds=scan["seconds"],counts=scan["classification_counts"],ledger=ledger)
        pr=[];cr=[]
        for parent in parents:
            uid=parent["uid"];p=read(out/"parents_basic"/(uid+".json"));scan=read(out/"parent_intersections"/(uid+".json"))
            p["intersection_scan_path"]=str(out/"parent_intersections"/(uid+".json"));p["intersection_summary"]=partition_contacts(scan,[],parent["N"])
            write(out/"parents"/(uid+".json"),p);row=flat_parent(p)
            for mode,groups in p["intersection_summary"].items():
                for k in ("proper_pair_count","coplanar_overlap_pair_count","unresolved_pair_count"):row[mode+"_"+k]=groups["FF"][k]
            pr.append(row)
            for task in [t for t in tasks if t["uid"]==uid]:
                c=read(out/"conditions_basic"/(task["task_id"]+".json"));c["intersection_partitions"]=partition_contacts(scan,c["source_face_ids"],c["N"])
                c["parent_intersection_scan_path"]=p["intersection_scan_path"];write(out/"conditions"/(c["task_id"]+".json"),c);row=flat_condition(c)
                for mode,groups in c["intersection_partitions"].items():
                    for scope,g in groups.items():
                        for k in ("proper_pair_count","coplanar_overlap_pair_count","unresolved_pair_count"):row[mode+"_"+scope+"_"+k]=g[k]
                        for k in ("proper_pairs","coplanar_positive_area_pairs","unresolved_pairs"):row[mode+"_"+scope+"_"+k]=g[k]
                cr.append(row)
        csv_write(out/"parents.csv",pr);csv_write(out/"conditions.csv",cr)
    after={p:sha(p) for p in inputs};after_sources={p:sha(p) for p in source_hash}
    if after!=inputs or after_sources!=source_hash:raise ValueError("Read-only source/input hashes changed")
    if "torch" in sys.modules:raise ValueError("Torch unexpectedly imported")
    result={"status":"COMPLETE" if a.stage!="basic" else "BASIC_COMPLETE","stage":a.stage,"parents":26,"conditions":411,
        "wall_seconds":time.monotonic()-start,"max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "input_hashes_before":inputs,"input_hashes_after":after,"source_hashes_before":source_hash,"source_hashes_after":after_sources,
        "model_calls":0,"OT_calls":0,"new_meshes":0,"torch_imported":False,"GPU":0}
    write(out/("receipt_"+a.stage+".json"),result);event(out,"COMPLETE",stage=a.stage,wall_seconds=result["wall_seconds"],max_rss_kib=result["max_rss_kib"])
if __name__=="__main__":main()
