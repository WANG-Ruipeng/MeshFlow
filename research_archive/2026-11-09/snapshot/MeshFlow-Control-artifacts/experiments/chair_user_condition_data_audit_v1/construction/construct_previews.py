"""Bounded exact-subset geometry preview v1. No model/scorer/OT import."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np

POLICY = {
 "schema": "chair_condition_preview_v1",
 "types": ["PLANAR_DISK", "BENT_DISK", "TWO_PATCHES"],
 "parent_count": 26, "target_ratio": .4, "rounding": "floor(ratio*N+0.5)",
 "max_starts_per_parent_per_type": 32, "max_two_start_pairs": 16,
 "max_selected_previews": 78,
 "topology": "Each patch: source-index legal edge connected disk with a single simple boundary and manifold vertex links",
 "adjacency": "Retained degree2 and original incidence exactly equals mapped retained pair; no winding requirement",
 "coordinate_connectivity": "Exact numeric FP32 coordinate tuples, signed zero equal, no tolerance welding",
 "compactness_min_4piA_over_P2": .08, "surface_pca_second_to_first_std_min": .15,
 "planar_area95_normal_degrees_max": 15., "planar_max_plane_distance_over_sqrt_area": .03,
 "bent_sector_half_angle_degrees": 15., "bent_sector_center_separation_degrees_min": 25.,
 "bent_disjoint_sector_area_fraction_min": .2,
 "thin_neck_partition_area_fraction_min": .2,
 "thin_neck_shared_edge_over_sqrt_smaller_area_min": .12,
 "two_exact_components": 2, "two_component_area_fraction_min": .2,
 "two_component_face_min": "max(4,ceil(.2*K))",
 "two_min_separation_over_parent_bbox_diagonal": 1e-6,
 "numerical_near_degenerate_area_over_L2": 1e-12,
 "L": float(math.sqrt(3)*2/.3762),
 "growth": "Boundary-disk-preserving geometric priority: normalized perimeter plus seed-centroid squared distance; PLANAR adds fixed seed-plane and unsigned-normal envelope",
 "seed_selection": "Largest-area valid face, then farthest centroid from existing seed set; face-ID tie break; maximum32",
 "two_pair_selection": "Seed i with seed i+16 for i<16, fallback half-list offset for fewer seeds; exactly one ordered split floor(K/2),ceil(K/2)",
 "winner": "First passing candidate in fixed seed order. If none: greatest face count, then fewest failed gates, then earliest attempt; NA candidate is never valid C",
 "scope_warning": "Engineering criteria for clean previews, not universal legality or semantic suitability. Bounded failure is not proof no valid subset exists.",
 "forbidden": ["training", "model_forward", "OT", "new_vertices", "new_full_mesh", "generated_output_selection", "source_data_mutation"],
}

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def jload(path):
    return json.loads(Path(path).read_text(encoding="utf8"))

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if path.exists():
        if path.read_text(encoding="utf8") != data: raise RuntimeError(f"Refuse overwrite: {path}")
        return
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(data,encoding="utf8"); tmp.replace(path)

def edges_of(face):
    a,b,c=map(int,face)
    return [(min(a,b),max(a,b)),(min(b,c),max(b,c)),(min(c,a),max(c,a))]

def edge_map(faces, ids=None):
    out=collections.defaultdict(list)
    for i in range(len(faces)) if ids is None else ids:
        for e in set(edges_of(faces[i])): out[e].append(int(i))
    return dict(out)

def components(nodes, adjacency):
    left=set(map(int,nodes)); groups=[]
    while left:
        seed=min(left); todo=[seed]; left.remove(seed); group=[]
        while todo:
            i=todo.pop(); group.append(i)
            for j in sorted(adjacency.get(i,())):
                if j in left: left.remove(j); todo.append(j)
        groups.append(sorted(group))
    return sorted(groups,key=lambda x:(-len(x),x[0]))

def graph_from_edges(n, emap):
    adj={i:set() for i in range(n)}
    for fs in emap.values():
        for i in fs:
            adj[i].update(j for j in fs if j!=i)
    return adj

def topology(faces):
    faces=np.asarray(faces,dtype=np.int64); n=len(faces)
    em=edge_map(faces); adj=graph_from_edges(n,em)
    used=sorted(set(faces.reshape(-1).tolist())) if n else []
    boundary=[e for e,fs in em.items() if len(fs)==1]
    bad_edges=[list(e) for e,fs in em.items() if len(fs)>2 or e[0]==e[1]]
    bgraph=collections.defaultdict(set)
    for a,b in boundary: bgraph[a].add(b); bgraph[b].add(a)
    bloops=components(bgraph,bgraph)
    bad_links=[]
    for v in used:
        incident=[f for f in faces if v in f]
        link_edges=[tuple(int(x) for x in f if x!=v) for f in incident]
        if any(len(x)!=2 for x in link_edges):
            bad_links.append(v); continue
        links=collections.defaultdict(set)
        duplicate_link=len(set(tuple(sorted(e)) for e in link_edges))!=len(link_edges)
        for a,b in link_edges: links[a].add(b); links[b].add(a)
        degrees=sorted(map(len,links.values()))
        target_path=(sum(d==1 for d in degrees)==2 and all(d in(1,2) for d in degrees))
        target_cycle=bool(degrees) and all(d==2 for d in degrees)
        if duplicate_link or len(components(links,links))!=1 or not (target_path if v in bgraph else target_cycle):
            bad_links.append(v)
    comps=components(range(n),adj)
    euler=len(used)-len(em)+n
    disk=bool(n and len(comps)==1 and not bad_edges and not bad_links and
              euler==1 and len(bloops)==1 and all(len(a)==2 for a in bgraph.values()))
    return {"faces":n,"vertices":len(used),"edges":len(em),"euler":euler,
            "edge_components":len(comps),"boundary_edges":len(boundary),
            "boundary_components":len(bloops),"boundary_branch_vertices":[int(v) for v,a in bgraph.items() if len(a)!=2],
            "nonmanifold_edges":bad_edges,"bad_vertex_links":bad_links,"disk":disk}

def exact_coordinate_faces(tri):
    lookup={}; rows=[]
    for face in tri:
        row=[]
        for p in face:
            key=tuple(map(float,p))
            if key not in lookup: lookup[key]=len(lookup)
            row.append(lookup[key])
        rows.append(row)
    return np.array(rows,dtype=np.int64)

class Parent:
    def __init__(self, arrays, cache=None):
        self.a=arrays
        self.tri=np.asarray(arrays["full_target"],np.float32).reshape(-1,3,3)
        self.faces=np.asarray(arrays["source_face_vertex_ids"],np.int64)
        self.source=np.asarray(arrays.get("source_faces",self.faces),np.int64)
        self.original=np.asarray(arrays.get("original_source_faces",self.source),np.int64)
        self.retained=np.asarray(arrays.get("retained_to_original_face",np.arange(len(self.faces))),np.int64)
        self.vertices=np.asarray(arrays["model_vertices"],np.float64)
        self.n=len(self.faces)
        assert np.array_equal(self.vertices[self.faces].astype(np.float32).view(np.uint32),self.tri.view(np.uint32))
        assert np.array_equal(self.original[self.retained],self.source)
        self.cp=np.asarray(arrays.get("source_corner_permutations",np.tile(np.arange(3),(self.n,1))),np.int64)
        assert np.array_equal(np.take_along_axis(self.source,self.cp,axis=1),self.faces)
        self.centroids=self.tri.astype(np.float64).mean(1)
        self.cross=np.cross(self.tri[:,1].astype(float)-self.tri[:,0],self.tri[:,2].astype(float)-self.tri[:,0])
        double=np.linalg.norm(self.cross,axis=1); self.area=double*.5
        self.normal=self.cross/np.maximum(double[:,None],np.finfo(float).tiny)
        self.diag=float(np.linalg.norm(np.ptp(self.tri.reshape(-1,3),axis=0)))
        self.good=np.isfinite(self.tri).all((1,2)) & (self.area>POLICY["numerical_near_degenerate_area_over_L2"]*POLICY["L"]**2)
        keys=[tuple(sorted(tuple(map(float,p)) for p in f)) for f in self.tri]
        self.duplicate_groups=[v for v in _groups(keys).values() if len(v)>1]
        self.re=edge_map(self.source); self.oe=edge_map(self.original)
        self.legal={i:set() for i in range(self.n)}; self.legal_edge={}
        for e,fs in self.re.items():
            if len(fs)==2 and set(self.oe.get(e,[]))==set(self.retained[fs]):
                i,j=fs; self.legal[i].add(j); self.legal[j].add(i); self.legal_edge[tuple(sorted(fs))]=e
        self.cache=cache or {"contacts":[]}; self.contacts=self.cache.get("contacts",[])
        self.forbidden={i:set() for i in range(self.n)}
        self.touch={i:set() for i in range(self.n)}
        for row in self.contacts:
            i,j=map(int,row["pair"]); self.touch[i].add(j); self.touch[j].add(i)
            if row["classification"] in ("proper_transverse_penetration","coplanar_positive_area_overlap") or not row.get("detector",{}).get("resolved",False):
                self.forbidden[i].add(j); self.forbidden[j].add(i)
        for group in self.duplicate_groups:
            for i in group: self.forbidden[i].update(j for j in group if j!=i)
        self.coordinate_faces=exact_coordinate_faces(self.tri)

    def interface(self,ids):
        chosen=set(ids); trusted=[]; ambiguous=[]; source_boundary=[]
        for e,fs in self.re.items():
            c=[i for i in fs if i in chosen]
            if not c: continue
            free=[i for i in fs if i not in chosen]
            original_fs=self.oe.get(e,[])
            removed=set(original_fs)-set(self.retained[fs])
            if free or removed:
                ok=(len(c)==1 and len(free)==1 and len(fs)==2 and not removed and set(original_fs)==set(self.retained[fs]))
                row={"edge":list(e),"condition_faces":c,"free_faces":free,"removed_original_faces":sorted(removed)}
                (trusted if ok else ambiguous).append(row)
            elif len(fs)==1:
                source_boundary.append(list(e))
        return {"trusted_edges":trusted,"ambiguous_edges":ambiguous,"original_open_boundary_edges":source_boundary}

def _groups(values):
    d=collections.defaultdict(list)
    for i,v in enumerate(values): d[v].append(i)
    return d

def surface_geometry(parent,ids):
    ids=np.array(sorted(ids),dtype=int); tri=parent.tri[ids].astype(float); a=parent.area[ids]
    if not len(ids) or not np.isfinite(a).all() or a.sum()<=0: return {"status":"INVALID_AREA"}
    total=float(a.sum()); center=np.sum(a[:,None]*tri.mean(1),axis=0)/total
    # E[xx^T] for a uniform triangle: (sum vv^T + (sum v)(sum v)^T)/12.
    q=np.einsum("fvi,fvj->fij",tri,tri)
    s=tri.sum(1); q=(q+np.einsum("fi,fj->fij",s,s))/12.
    cov=np.sum(a[:,None,None]*q,axis=0)/total-np.outer(center,center)
    vals,vecs=np.linalg.eigh((cov+cov.T)*.5); vals=np.maximum(vals,0)
    fitnormal=vecs[:,0]
    unsigned_angles=np.degrees(np.arccos(np.clip(np.abs(parent.normal[ids]@fitnormal),0,1)))
    order=np.argsort(unsigned_angles,kind="stable")
    q95=float(unsigned_angles[order[min(np.searchsorted(np.cumsum(a[order]),.95*total),len(ids)-1)]])
    maxdist=float(np.max(np.abs((tri-center)@fitnormal)))
    em=edge_map(parent.faces,ids); boundary=[e for e,fs in em.items() if len(fs)==1]
    perimeter=float(sum(np.linalg.norm(parent.vertices[e[0]]-parent.vertices[e[1]]) for e in boundary))
    compact=float(4*np.pi*total/perimeter**2) if perimeter>0 else 0.
    width=float(np.sqrt(vals[1]/vals[2])) if vals[2]>0 else 0.
    planar=q95<=POLICY["planar_area95_normal_degrees_max"] and maxdist/math.sqrt(total)<=POLICY["planar_max_plane_distance_over_sqrt_area"]
    normals=parent.normal[ids]; dots=np.clip(np.abs(normals@normals.T),0,1)
    theta=np.degrees(np.arccos(dots)); cone=POLICY["bent_sector_half_angle_degrees"]
    bent=None; best=0.
    for i in range(len(ids)):
        if bent: break
        for j in range(i+1,len(ids)):
            if theta[i,j]<POLICY["bent_sector_center_separation_degrees_min"]: continue
            ai=theta[i]; aj=theta[j]
            gi=(ai<=cone)&(ai<=aj); gj=(aj<=cone)&(aj<ai)
            fi=float(a[gi].sum()/total); fj=float(a[gj].sum()/total)
            best=max(best,min(fi,fj))
            if min(fi,fj)>=POLICY["bent_disjoint_sector_area_fraction_min"]:
                bent={"center_source_face_ids":[int(ids[i]),int(ids[j])],"center_separation_degrees":float(theta[i,j]),
                      "disjoint_sector_area_fractions":[fi,fj]};break
    return {"status":"PASS","area":total,"area_fraction_parent":total/float(parent.area.sum()),
            "surface_center":center.tolist(),"surface_covariance_eigenvalues":vals.tolist(),
            "surface_pca_second_to_first_std":width,"boundary_perimeter":perimeter,"compactness_4piA_over_P2":compact,
            "area95_unsigned_normal_degrees":q95,"max_fit_plane_distance_over_sqrt_area":maxdist/math.sqrt(total),
            "PLANAR":bool(planar),"BENT":bool(not planar and bent),"bent_sectors":bent,
            "best_min_normal_sector_area_fraction":best}

def necks(parent,ids):
    # Tarjan dual-graph bridges; all partitions recorded, only substantial narrow bridges reject.
    chosen=set(ids); graph={i:parent.legal[i]&chosen for i in chosen}
    order={}; low={}; bridges=[]; area_sub={}; tick=0
    total=float(parent.area[list(ids)].sum())
    def dfs(v,prev):
        nonlocal tick
        order[v]=low[v]=tick; tick+=1; area_sub[v]=float(parent.area[v])
        for w in sorted(graph[v]):
            if w==prev: continue
            if w not in order:
                dfs(w,v); area_sub[v]+=area_sub[w]; low[v]=min(low[v],low[w])
                if low[w]>order[v]:
                    smaller=min(area_sub[w],total-area_sub[w]); edge=parent.legal_edge[tuple(sorted((v,w)))]
                    length=float(np.linalg.norm(parent.vertices[edge[0]]-parent.vertices[edge[1]]))
                    fraction=smaller/total if total else 0.
                    relative=length/math.sqrt(max(smaller,np.finfo(float).tiny))
                    bridges.append({"face_pair":[v,w],"edge":list(edge),"smaller_area_fraction":fraction,
                        "edge_over_sqrt_smaller_area":relative,
                        "thin_neck":bool(fraction>=POLICY["thin_neck_partition_area_fraction_min"] and relative<POLICY["thin_neck_shared_edge_over_sqrt_smaller_area_min"])})
            else: low[v]=min(low[v],order[w])
    for v in sorted(chosen):
        if v not in order: dfs(v,None)
    return bridges

def patch_report(parent,ids,kind="ANY"):
    ids=sorted(ids); selected=set(ids); reasons=[]
    t=topology(parent.faces[ids]); ct=topology(parent.coordinate_faces[ids])
    g=surface_geometry(parent,ids)
    lawful_components=components(ids,parent.legal)
    if not t["disk"]: reasons.append("not_source_index_disk")
    if not ct["disk"]: reasons.append("not_exact_coordinate_disk")
    if len(lawful_components)!=1: reasons.append("not_connected_by_trusted_legal_edges")
    if not all(parent.good[ids]): reasons.append("nonfinite_or_numerical_degenerate_face")
    bad=[r for r in parent.contacts if set(r["pair"])<=selected and (
        r["classification"] in ("proper_transverse_penetration","coplanar_positive_area_overlap") or not r.get("detector",{}).get("resolved",False))]
    duplicates=[grp for grp in parent.duplicate_groups if len(set(grp)&selected)>1]
    if bad: reasons.append("internal_illegal_or_unresolved_pair")
    if duplicates: reasons.append("exact_duplicate_triangle")
    iface=parent.interface(ids)
    if iface["ambiguous_edges"]: reasons.append("ambiguous_original_common_interface")
    if not iface["trusted_edges"]: reasons.append("no_trusted_common_interface")
    bridges=necks(parent,ids)
    if any(x["thin_neck"] for x in bridges): reasons.append("thin_neck")
    if g["status"]!="PASS": reasons.append("invalid_area")
    else:
        if g["compactness_4piA_over_P2"]<POLICY["compactness_min_4piA_over_P2"]: reasons.append("low_compactness")
        if g["surface_pca_second_to_first_std"]<POLICY["surface_pca_second_to_first_std_min"]: reasons.append("thin_global_strip")
        if kind=="PLANAR_DISK" and not g["PLANAR"]: reasons.append("not_near_planar")
        if kind=="BENT_DISK" and not g["BENT"]: reasons.append("no_substantial_bend")
    winding={"same":0,"opposite":0}
    for e,fs in edge_map(parent.source,ids).items():
        if len(fs)!=2: continue
        directions=[]
        for f in fs:
            face=parent.source[f].tolist()
            directions.append(any((face[k],face[(k+1)%3])==e for k in range(3)))
        winding["same" if directions[0]==directions[1] else "opposite"]+=1
    return {"status":"PASS" if not reasons else "NA","failure_reasons":reasons,
            "source_index_topology":t,"exact_coordinate_topology":ct,"trusted_edge_components":lawful_components,
            "geometry":g,"interface":iface,"dual_bridges":bridges,"source_winding_shared_edges":winding,
            "internal_conflicts":[{"pair":r["pair"],"classification":r["classification"]} for r in bad],
            "exact_duplicate_groups":duplicates}


def seed_list(parent):
    available=[i for i in range(parent.n) if parent.good[i]]
    if not available: return []
    seeds=[min(available,key=lambda i:(-parent.area[i],i))]
    while len(seeds)<min(POLICY["max_starts_per_parent_per_type"],len(available)):
        left=[i for i in available if i not in seeds]
        chosen=min(left,key=lambda i:(-float(np.min(np.sum((parent.centroids[seeds]-parent.centroids[i])**2,axis=1))),i))
        seeds.append(chosen)
    return seeds

def grow(parent,seed,k,kind,excluded=()):
    """Single deterministic no-backtracking disk growth, not FIFO BFS."""
    excluded=set(excluded)
    if seed in excluded or not parent.good[seed]: return []
    selected={seed}; vertices=set(map(int,parent.faces[seed]))
    boundary=set(edges_of(parent.faces[seed])); blocked=collections.Counter()
    allowed=set(i for i in range(parent.n) if parent.good[i])-excluded
    if kind=="PLANAR_DISK":
        upper=float(np.sort(parent.area[list(allowed)])[-k:].sum())
        for i in list(allowed):
            angle=math.degrees(math.acos(float(np.clip(abs(parent.normal[i]@parent.normal[seed]),0,1))))
            dist=float(np.max(np.abs((parent.tri[i]-parent.centroids[seed])@parent.normal[seed])))
            if angle>30. or dist>.03*math.sqrt(upper): allowed.remove(i)
    total=float(parent.area[seed]); perimeter=sum(np.linalg.norm(parent.vertices[a]-parent.vertices[b]) for a,b in boundary)
    while len(selected)<k:
        frontier=set().union(*(parent.legal[i] for i in selected)) & allowed - selected
        candidates=[]
        for i in sorted(frontier):
            if parent.forbidden[i]&selected: continue
            fedges=set(edges_of(parent.faces[i])); shared=boundary&fedges
            if len(shared) not in (1,2): continue
            shared_vertices=set(v for edge in shared for v in edge)
            if set(map(int,parent.faces[i]))&vertices != shared_vertices: continue
            p=float(perimeter+sum(np.linalg.norm(parent.vertices[a]-parent.vertices[b]) for a,b in fedges)
                    -2*sum(np.linalg.norm(parent.vertices[a]-parent.vertices[b]) for a,b in shared))
            area=total+float(parent.area[i])
            d=float(np.sum((parent.centroids[i]-parent.centroids[seed])**2))
            score=p*p/(4*np.pi*area)+.3*d/area
            candidates.append((score,i,p,area))
        if not candidates: break
        _,i,perimeter,total=min(candidates)
        selected.add(i); vertices.update(map(int,parent.faces[i])); boundary.symmetric_difference_update(edges_of(parent.faces[i]))
    return sorted(selected)

def point_segment_squared(points,a,b):
    # Broadcast P[:,None] against segment arrays [M].
    u=b-a; d=np.sum(u*u,axis=-1)
    t=np.sum((points[:,None,:]-a[None,:,:])*u[None,:,:],axis=-1)/np.maximum(d[None,:],np.finfo(float).tiny)
    q=a[None,:,:]+np.clip(t,0,1)[:,:,None]*u[None,:,:]
    return np.sum((points[:,None,:]-q)**2,axis=-1)

def point_tri_squared(points,tri):
    a=tri[:,0]; u=tri[:,1]-a; v=tri[:,2]-a
    normal=np.cross(u,v); n2=np.sum(normal*normal,axis=1)
    off=points[:,None,:]-a[None,:,:]; signed=np.sum(off*normal[None,:,:],axis=-1)
    proj=off-signed[:,:,None]/np.maximum(n2[None,:,None],np.finfo(float).tiny)*normal[None,:,:]
    uu=np.sum(u*u,axis=1);uv=np.sum(u*v,axis=1);vv=np.sum(v*v,axis=1)
    pu=np.sum(proj*u[None,:,:],axis=-1);pv=np.sum(proj*v[None,:,:],axis=-1)
    denom=uu*vv-uv*uv
    s=(pu*vv-pv*uv)/np.maximum(denom,np.finfo(float).tiny)
    t=(pv*uu-pu*uv)/np.maximum(denom,np.finfo(float).tiny)
    inside=(s>=0)&(t>=0)&(s+t<=1)
    result=np.where(inside,signed*signed/np.maximum(n2,np.finfo(float).tiny),np.inf)
    for i in range(3):
        result=np.minimum(result,point_segment_squared(points,tri[:,i],tri[:,(i+1)%3]))
    return result

def minimum_triangle_distance(first,second):
    """Exact finite-triangle distance formula for already verified disjoint sets."""
    first=np.asarray(first,float); second=np.asarray(second,float)
    if not len(first) or not len(second): return None
    best=min(float(point_tri_squared(first.reshape(-1,3),second).min()),
             float(point_tri_squared(second.reshape(-1,3),first).min()))
    a=np.concatenate([first[:,i] for i in range(3)]); b=np.concatenate([first[:,(i+1)%3] for i in range(3)])
    c=np.concatenate([second[:,i] for i in range(3)]); d=np.concatenate([second[:,(i+1)%3] for i in range(3)])
    u=b-a;v=d-c;w=a[:,None,:]-c[None,:,:]
    uu=np.sum(u*u,axis=1)[:,None];vv=np.sum(v*v,axis=1)[None,:]
    uv=u@v.T;uw=np.sum(u[:,None,:]*w,axis=-1);vw=np.sum(v[None,:,:]*w,axis=-1)
    det=uu*vv-uv*uv;valid=det>1e-14*uu*vv
    with np.errstate(divide="ignore",invalid="ignore"):
        s=(uv*vw-vv*uw)/np.where(valid,det,1)
        t=(uu*vw-uv*uw)/np.where(valid,det,1)
    valid &= (s>=0)&(s<=1)&(t>=0)&(t<=1)
    dist=np.sum((w+s[:,:,None]*u[:,None,:]-t[:,:,None]*v[None,:,:])**2,axis=-1)
    if valid.any(): best=min(best,float(dist[valid].min()))
    return float(math.sqrt(max(0,best)))

def two_report(parent,first,second,k):
    first=sorted(first);second=sorted(second);ids=sorted(set(first)|set(second));reasons=[]
    parts=[patch_report(parent,first),patch_report(parent,second)]
    for n,r in enumerate(parts):
        reasons.extend(f"component{n+1}:{x}" for x in r["failure_reasons"])
    if len(ids)!=k: reasons.append("exact_K40_not_reached")
    if set(first)&set(second): reasons.append("overlapping_face_membership")
    comp=components(ids,parent.legal); itopo=topology(parent.faces[ids]); ctopo=topology(parent.coordinate_faces[ids])
    if len(comp)!=2 or itopo["edge_components"]!=2 or ctopo["edge_components"]!=2: reasons.append("not_exactly_two_edge_components")
    shared=set(parent.faces[first].reshape(-1))&set(parent.faces[second].reshape(-1))
    if shared: reasons.append("cross_component_shared_source_vertices")
    contact=[r for r in parent.contacts if (r["pair"][0] in first and r["pair"][1] in second) or
               (r["pair"][1] in first and r["pair"][0] in second)]
    if contact: reasons.append("cross_component_geometric_contact_or_unresolved")
    areas=[float(parent.area[x].sum()) for x in (first,second)]; total=sum(areas)
    fractions=[a/total if total else 0 for a in areas]
    if min(fractions)<POLICY["two_component_area_fraction_min"]: reasons.append("insubstantial_component_area")
    if min(len(first),len(second))<max(4,math.ceil(.2*k)): reasons.append("insubstantial_component_face_count")
    distance=minimum_triangle_distance(parent.tri[first],parent.tri[second]) if first and second and not contact else None
    if distance is None or distance<=POLICY["two_min_separation_over_parent_bbox_diagonal"]*parent.diag: reasons.append("nonpositive_or_numerically_tiny_separation")
    parent_comps=components(range(parent.n),parent.legal); owner={v:i for i,g in enumerate(parent_comps) for v in g}
    same=(bool(first and second) and owner[first[0]]==owner[second[0]])
    return {"status":"PASS" if not reasons else "NA","failure_reasons":reasons,
            "components":parts,"component_face_ids":[first,second],"component_area_fractions":fractions,
            "source_index_topology":itopo,"exact_coordinate_topology":ctopo,
            "cross_contacts":[{"pair":r["pair"],"classification":r["classification"]} for r in contact],
            "minimum_triangle_distance":distance,"minimum_distance_over_parent_diagonal":distance/parent.diag if distance is not None else None,
            "centroid_distance":float(np.linalg.norm(parent.centroids[first].mean(0)-parent.centroids[second].mean(0))) if first and second else None,
            "both_from_same_parent_legal_component":same}

def enumerate_type(parent,kind):
    k=math.floor(.4*parent.n+.5); seeds=seed_list(parent); attempts=[]; best=None
    if kind=="TWO_PATCHES":
        offset=min(16,len(seeds)//2); starts=[(seeds[i],seeds[i+offset]) for i in range(offset)]
    else: starts=[(s,) for s in seeds]
    for ordinal,start in enumerate(starts,1):
        if kind=="TWO_PATCHES":
            first=grow(parent,start[0],k//2,"ANY")
            used_vertices=set(parent.faces[first].reshape(-1))
            excluded=set(first)
            for i in range(parent.n):
                if used_vertices&set(parent.faces[i]): excluded.add(i)
            for i in first: excluded.update(parent.touch[i])
            second=grow(parent,start[1],k-k//2,"ANY",excluded)
            ids=sorted(set(first)|set(second)); report=two_report(parent,first,second,k)
            component_ids=[first,second]
        else:
            ids=grow(parent,start[0],k,kind);report=patch_report(parent,ids,kind)
            if len(ids)!=k:
                report["failure_reasons"].append("exact_K40_not_reached");report["status"]="NA"
            component_ids=[ids]
        attempt={"ordinal":ordinal,"start_source_face_ids":list(start),"achieved_face_count":len(ids),
                 "target_face_count":k,"status":report["status"],"failure_reasons":report["failure_reasons"],
                 "source_face_ids":ids,"component_source_face_ids":component_ids}
        attempts.append(attempt)
        candidate={"attempt":attempt,"report":report}
        if best is None or (len(ids),-len(report["failure_reasons"]))>(best["attempt"]["achieved_face_count"],-len(best["report"]["failure_reasons"])):
            best=candidate
        if report["status"]=="PASS": best=candidate;break
    if best is None:
        best={"attempt":{"ordinal":0,"source_face_ids":[],"component_source_face_ids":[],"achieved_face_count":0,"start_source_face_ids":[]},
              "report":{"status":"NA","failure_reasons":["no_numerically_valid_seed"]}}
    return {"type":kind,"N":parent.n,"K_target":k,"target_ratio":.4,
            "actual_face_ratio":best["attempt"]["achieved_face_count"]/parent.n,
            "status":best["report"]["status"],"selected_or_best_failed":best,
            "attempt_count":len(attempts),"startpoint_budget_used":sum(len(a["start_source_face_ids"]) for a in attempts),
            "attempts":attempts}

def validate_cache(cache,parent_entry):
    n=parent_entry["N"];expected=n*(n-1)//2
    assert cache["uid"]==parent_entry["uid"]
    assert cache["input_npz_sha256"]==parent_entry["sha256"]
    assert cache["expected_full_pairs"]==expected and cache["tested_pairs"]==expected
    assert cache["independent_verification"] is True
    assert sum(cache["classification_counts"].values())==expected
    pairs=[]
    for row in cache["contacts"]:
        i,j=row["pair"]; assert 0<=i<j<n;pairs.append((i,j))
        assert row["classification"]!="disjoint"
        detector=row["detector"]
        if detector.get("resolved"):
            assert detector.get("verified") is True
            assert detector["primary"]["classification"]==detector["independent"]["classification"]==row["classification"]
    assert len(pairs)==len(set(pairs))
    assert len(pairs)==expected-cache["classification_counts"].get("disjoint",0)

def execute(run,data_root):
    out=run/"construction"; t0=time.perf_counter()
    receipt=run/"geometry"/"receipt_intersections.json"
    audit=jload(receipt); assert audit["status"]=="COMPLETE",audit.get("status")
    manifest_path=data_root/"train_manifest.json";manifest=jload(manifest_path)
    assert len(manifest["parents"])==26
    old_bfs=Path("/mnt/e/MeshFlow/experiments/mf_h1_local0/chair_domain_control_v1/prepare.py")
    sources={str(p):sha(p) for p in (Path(__file__),manifest_path,receipt,run/"geometry"/"definitions.json",old_bfs)}
    for p in manifest["parents"]:
        source=data_root/p["npz"];assert sha(source)==p["sha256"];sources[str(source)]=p["sha256"]
        cached=run/"geometry"/"parent_intersections"/(p["uid"]+".json")
        validate_cache(jload(cached),p);sources[str(cached)]=sha(cached)
    registration={"schema":"chair_condition_construction_registration_v1","policy":POLICY,"source_hashes":sources,
                  "input_parent_ids":[p["uid"] for p in manifest["parents"]],"model_calls":0,"OT_calls":0,
                  "generated_outputs_read":0,"new_source_vertices":0,"new_full_meshes":0}
    write_json(out/"registration.json",registration)
    rows=[];results=[]
    for p in manifest["parents"]:
        parent_started=time.perf_counter();source=data_root/p["npz"]
        with np.load(source,allow_pickle=False) as n:
            arrays={k:n[k].copy() for k in n.files}
        cache_path=run/"geometry"/"parent_intersections"/(p["uid"]+".json")
        parent=Parent(arrays,jload(cache_path))
        for kind in POLICY["types"]:
            path=out/"previews"/p["uid"]/(kind+".json")
            if path.exists():
                result=jload(path)
                assert result["source_npz_sha256"]==p["sha256"] and result["registration_sha256"]==sha(out/"registration.json")
            else:
                result=enumerate_type(parent,kind)
                result.update({"uid":p["uid"],"source_npz":str(source),"source_npz_sha256":p["sha256"],
                    "source_pair_cache":str(cache_path),"source_pair_cache_sha256":sha(cache_path),
                    "registration_sha256":sha(out/"registration.json"),"preview_only":True,
                    "semantic_label":None,"no_training_data_replacement":True})
                ids=np.array(result["selected_or_best_failed"]["attempt"]["source_face_ids"],dtype=np.int64)
                result["source_face_ids"]=ids.tolist();result["original_face_ids"]=parent.retained[ids].tolist()
                if result["status"]=="PASS":
                    npz=path.with_suffix(".npz");npz.parent.mkdir(parents=True,exist_ok=True)
                    assert not npz.exists()
                    component_ids=result["selected_or_best_failed"]["attempt"]["component_source_face_ids"]
                    component={face:i for i,group in enumerate(component_ids) for face in group}
                    np.savez_compressed(npz,C=arrays["full_target"][ids],source_face_ids=ids,
                        original_face_ids=parent.retained[ids],source_faces=parent.source[ids],
                        source_face_vertex_ids=parent.faces[ids],source_corner_permutations=parent.cp[ids],
                        component_ids=np.array([component[i] for i in ids],dtype=np.int64))
                    with np.load(npz,allow_pickle=False) as z:
                        assert np.array_equal(z["C"].view(np.uint32),arrays["full_target"][ids].view(np.uint32))
                    result["condition_npz"]=str(npz);result["condition_npz_sha256"]=sha(npz)
                else:
                    result["condition_npz"]=None;result["render_contract"]="NA / BEST FAILED CANDIDATE; NOT ACCEPTED C40"
                write_json(path,result)
            results.append({"uid":p["uid"],"type":kind,"status":result["status"],"path":str(path),"sha256":sha(path),
                            "condition_npz":result.get("condition_npz"),"condition_npz_sha256":result.get("condition_npz_sha256")})
            rows.append({"uid":p["uid"],"type":kind,"status":result["status"],"N":parent.n,"K_target":result["K_target"],
                        "achieved_faces":len(result["source_face_ids"]),"actual_face_ratio":result["actual_face_ratio"],
                        "attempts":result["attempt_count"],"startpoints_used":result["startpoint_budget_used"],
                        "reasons":";".join(result["selected_or_best_failed"]["report"]["failure_reasons"]),
                        "source_face_ids":json.dumps(result["source_face_ids"]),"result_json":str(path)})
        print(json.dumps({"uid":p["uid"],"seconds":time.perf_counter()-parent_started,
                          "results":{r["type"]:r["status"] for r in results[-3:]}}),flush=True)
    assert len(results)==78 and all(r["startpoints_used"]<=32 for r in rows)
    summary=out/"preview_summary.csv"
    if not summary.exists():
        with summary.open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    unchanged=all(sha(Path(p))==v for p,v in sources.items()); assert unchanged
    final={"schema":"chair_condition_construction_receipt_v1","status":"COMPLETE","parent_count":26,"result_rows":78,
       "accepted":sum(r["status"]=="PASS" for r in results),"NA":sum(r["status"]=="NA" for r in results),
       "by_type":{k:dict(collections.Counter(r["status"] for r in results if r["type"]==k)) for k in POLICY["types"]},
       "attempts":sum(r["attempts"] for r in rows),"startpoints":sum(r["startpoints_used"] for r in rows),
       "seconds":time.perf_counter()-t0,"max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       "source_hashes_unchanged":unchanged,"source_hashes":sources,"registration_sha256":sha(out/"registration.json"),
       "summary_csv_sha256":sha(summary),"results":results,"model_calls":0,"OT_calls":0,"new_vertices":0,
       "new_full_meshes":0,"GPU":0,"numeric_threads":2,"geometry_workers":1,"torch_imported":"torch" in sys.modules,
       "limitations":[POLICY["scope_warning"],"Face fraction does not imply surface-area fraction.",
                       "Source-free internal conflicts are reported, not attributed to C or rejected as C-only quality.",
                       "Not semantic seat/back classification; no user study or generated model evaluation."]}
    write_json(out/"receipt.json",final)
    print(json.dumps({k:final[k] for k in ("status","accepted","NA","by_type","attempts","seconds")}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,default=Path("/mnt/e/MeshFlow-Control-artifacts/data"))
    parser.add_argument("--register-policy-only",action="store_true");args=parser.parse_args()
    out=args.run/"construction";out.mkdir(parents=True,exist_ok=True)
    write_json(out/"preview_policy.json",POLICY)
    if args.register_policy_only:
        print(json.dumps({"status":"POLICY_REGISTERED","policy_sha256":sha(out/"preview_policy.json"),"model_calls":0}))
    else: execute(args.run,args.data_root)
if __name__=="__main__":main()
