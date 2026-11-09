"""Construct reviewed region subsets; no noise, OT, model or mesh repair."""
from pathlib import Path
import collections, heapq, json, hashlib
import numpy as np
from prepare import sha, read, write, event, geometry, DEFAULT_OUT

def area(tri):
    t=np.asarray(tri,dtype=np.float64).reshape(-1,3,3)
    return np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2

def graph(faces, ids):
    chosen=set(map(int,ids));edges=collections.defaultdict(list)
    for i in chosen:
        for j in range(3):
            e=tuple(sorted((int(faces[i,j]),int(faces[i,(j+1)%3]))))
            edges[e].append(i)
    adj={i:set() for i in chosen}
    for v in edges.values():
        for i in v:adj[i].update(set(v)-{i})
    return adj,edges

def growth_order(tri, faces, ids, kind):
    ids=sorted(set(map(int,ids)));t=np.asarray(tri).reshape(-1,3,3).astype(float)
    centers=t.mean(1);points=t[ids].reshape(-1,3)
    lo,hi=points.min(0),points.max(0);anchor=(lo+hi)/2
    if kind=="seat":anchor[[0,2]]=lo[[0,2]]
    elif kind=="back":anchor[0]=lo[0];anchor[1]=hi[1]
    else:raise ValueError("Reviewed semantic region type required")
    adj,edges=graph(faces,ids)
    boundary_faces=set(i for fs in edges.values() if len(fs)==1 for i in fs)
    candidates=sorted(boundary_faces or ids)
    seed=min(candidates,key=lambda i:(float(np.linalg.norm(centers[i]-anchor)),i))
    heap=[(0.,seed)];distance={seed:0.};seen=set();order=[]
    while heap:
        d,i=heapq.heappop(heap)
        if i in seen:continue
        seen.add(i);order.append(i)
        for j in sorted(adj[i]):
            nd=d+float(np.linalg.norm(centers[i]-centers[j]))
            if j not in seen and nd<distance.get(j,float("inf")):
                distance[j]=nd;heapq.heappush(heap,(nd,j))
    if len(order)!=len(ids):raise ValueError("Reviewed region is not full-edge connected")
    return order,dict(seed_source_face_id=seed,anchor=anchor.tolist(),
        growth="Dijkstra centroid distance on source-face complete-edge graph; fixed face-ID ties")

def nested_extents(tri,faces,ids,kind):
    ids=sorted(set(map(int,ids)))
    if not ids:raise ValueError("Empty region")
    a=area(tri);order,audit=growth_order(tri,faces,ids,kind)
    cum=np.cumsum(a[order]);total=float(cum[-1])
    if total<=0:raise ValueError("Zero area region")
    extents=[];seen={}
    for tier,fraction in (("small",.25),("medium",.5),("full",1.)):
        n=len(ids) if tier=="full" else min(range(1,len(ids)+1),key=lambda k:(abs(float(cum[k-1])-fraction*total),k))
        selected=sorted(order[:n]);key=tuple(selected)
        if key in seen:
            extents[seen[key]]["requested_tiers"].append(tier)
            extents[seen[key]]["requested_area_fractions"].append(fraction)
        else:
            seen[key]=len(extents)
            extents.append(dict(source_face_ids=selected,requested_tiers=[tier],
                requested_area_fractions=[fraction],region_area_fraction=float(a[selected].sum()/total)))
    for row in extents:
        row["tier"]="full" if "full" in row["requested_tiers"] else ("medium" if "medium" in row["requested_tiers"] else "small")
    return extents,audit

def partition_arrays(a,ids):
    M=a["full_target"];n=len(M);ids=np.asarray(sorted(set(map(int,ids))),dtype=np.int64)
    if M.dtype!=np.float32 or not 0<len(ids)<n or ids.min()<0 or ids.max()>=n:raise ValueError("Invalid FP32/M/K/source IDs")
    free=np.setdiff1d(np.arange(n,dtype=np.int64),ids)
    C=M[ids].reshape(-1,3,3).copy();Y=M[free].reshape(-1,3,3).copy()
    reconstructed=np.empty_like(M);reconstructed[ids]=C.reshape(-1,9);reconstructed[free]=Y.reshape(-1,9)
    assert reconstructed.tobytes()==M.tobytes()
    assert np.array_equal(np.sort(np.r_[ids,free]),np.arange(n))
    result=dict(C=C,Y=Y,source_face_ids=ids,free_source_ids=free,
        original_source_face_ids=a["retained_to_original_face"][ids],
        free_original_source_face_ids=a["retained_to_original_face"][free],
        C_source_vertex_ids=a["source_face_vertex_ids"][ids],
        Y_source_vertex_ids=a["source_face_vertex_ids"][free],
        C_source_corner_permutations=a["source_corner_permutations"][ids],
        Y_source_corner_permutations=a["source_corner_permutations"][free],
        C_pre_ot=a["pre_ot_full"][ids],Y_pre_ot=a["pre_ot_full"][free],
        N=np.asarray(n,np.int64),K=np.asarray(len(ids),np.int64))
    assert (result["C_pre_ot"]*np.float32(2)).tobytes()==C.tobytes()
    assert (result["Y_pre_ot"]*np.float32(2)).tobytes()==Y.tobytes()
    return result

def build(out=DEFAULT_OUT):
    out=Path(out)
    parents={p["uid"]:p for p in read(out/"parent_inventory.json")["rows"]}
    reviewed=read(out/"regions.json")["rows"];results=[];regions=[]
    for r in reviewed:
        if r["status"] not in ("ACCEPTED_AI_PROXY","ACCEPTED_WITH_RISK_AI_PROXY"):continue
        p=parents[r["uid"]];a=geometry(p);M=a["full_target"];aa=area(M)
        ids=r["source_face_ids"];extents,growth=nested_extents(M,a["source_faces"],ids,r["region_type"])
        regions.append(dict(r,growth=growth,unique_extent_count=len(extents),requested_extent_count=3))
        for e in extents:
            # DEV selects one reviewed requested tier per parent; no hidden extra DEV examples.
            if p["split"]=="dev" and r["dev_requested_tier"] not in e["requested_tiers"]:continue
            cid=r["uid"]+"__"+r["region_type"]+"__"+e["tier"]
            arrays=partition_arrays(a,e["source_face_ids"])
            path=out/"conditions"/(cid+".npz");path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                with np.load(path,allow_pickle=False) as z:
                    assert set(z.files)==set(arrays)
                    assert all(z[k].dtype==v.dtype and z[k].shape==v.shape and z[k].tobytes()==v.tobytes() for k,v in arrays.items())
            else:
                with path.open("xb") as f:np.savez_compressed(f,**arrays)
            row=dict(condition_id=cid,uid=r["uid"],split=p["split"],region_id=r["region_id"],
                     region_type=r["region_type"],region_source_face_ids=ids,region_status=r["status"],
                     source_face_ids=e["source_face_ids"],free_source_ids=arrays["free_source_ids"].tolist(),
                     N=len(M),K=len(e["source_face_ids"]),face_fraction=len(e["source_face_ids"])/len(M),
                     region_area_fraction=e["region_area_fraction"],
                     parent_area_fraction=float(aa[e["source_face_ids"]].sum()/aa.sum()),
                     C_area=float(aa[e["source_face_ids"]].sum()),R_area=float(aa[ids].sum()),M_area=float(aa.sum()),
                     area_definition="sum of triangle areas; not visible area or union area",
                     tier=e["tier"],requested_tiers=e["requested_tiers"],
                     npz=str(path),sha256=sha(path),parent_npz=p["npz"],parent_sha256=p["sha256"],
                     growth=growth,review_kind="AI_PROXY",source_partition_bitwise=True,
                     full_mesh_repaired=False,training_status="NOT_RUN")
            results.append(row)
    train=[r for r in results if r["split"]=="train"];dev=[r for r in results if r["split"]=="dev"]
    assert len(train)<=156 and len(dev)<=8 and len({r["uid"] for r in dev})==len(dev)
    value=dict(schema="r2_reviewed_user_conditions_v1",source_region_manifest_sha256=sha(out/"regions.json"),
               rows=results,train_condition_count=len(train),dev_condition_count=len(dev),
               train_parents=sorted({r["uid"] for r in train}),dev_parents=sorted({r["uid"] for r in dev}),
               created_noise_arrays=0,OT_calls=0,model_calls=0,training_updates=0,
               pending="Data preparation; no training until review of this package")
    write(out/"conditions.json",value);write(out/"region_extent_audit.json",dict(rows=regions))
    event(out,"REVIEWED_CONDITIONS_EXPORTED",train=len(train),dev=len(dev),model_calls=0,OT_calls=0)
    print(json.dumps({"train":len(train),"dev":len(dev),"train_parents":len(value["train_parents"])}),flush=True)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=DEFAULT_OUT);args=p.parse_args();build(args.out)
