"""Frozen-cache-only finite prism / seam distance diagnostic. No torch."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import argparse, concurrent.futures, csv, hashlib, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
import psutil

ROOT = Path("/mnt/e/MeshFlow")
OUT = Path(__file__).resolve().parent
VERSION = "finite_parallel_prism_union_and_finite_seam_distance_v1"
COS = math.cos(math.radians(30))

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def csvwrite(p, rows):
    if not rows:
        Path(p).write_text("", encoding="utf-8-sig")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def normgeom(tris):
    cross = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    return lengths/2, cross/np.maximum(lengths[:, None], 1e-300)

def clip(poly, normal, offset, eps=1e-12):
    """Keep normal dot x >= offset, linear half-space, dimension agnostic."""
    if len(poly) < 3:
        return np.empty((0, len(normal)))
    values = poly @ normal - offset
    result = []
    prev, pv = poly[-1], values[-1]
    for cur, cv in zip(poly, values):
        pi, ci = pv >= -eps, cv >= -eps
        if pi != ci:
            denom = pv-cv
            if abs(denom) > 1e-30:
                result.append(prev+(pv/denom)*(cur-prev))
        if ci:
            result.append(cur)
        prev, pv = cur, cv
    if len(result) < 3:
        return np.empty((0, len(normal)))
    p = np.asarray(result)
    keep = np.linalg.norm(p-np.roll(p, 1, axis=0), axis=1) > eps
    p = p[keep]
    return p if len(p) >= 3 else np.empty((0, len(normal)))

def signed_area(poly):
    if len(poly) < 3:
        return 0.
    return float(np.sum(poly[:, 0]*np.roll(poly[:, 1], -1)-poly[:, 1]*np.roll(poly[:, 0], -1))/2)

def subtract_convex(subject, cutter):
    """Disjoint convex pieces of subject minus a CCW convex cutter."""
    current, outside = subject, []
    if signed_area(cutter) < 0:
        cutter = cutter[::-1]
    for a, b in zip(cutter, np.roll(cutter, -1, axis=0)):
        e = b-a
        n = np.array([-e[1], e[0]])
        n /= max(np.linalg.norm(n), 1e-300)
        d = float(n @ a)
        piece = clip(current, -n, -d)
        if abs(signed_area(piece)) > 1e-16:
            outside.append(piece)
        current = clip(current, n, d)
        if len(current) < 3:
            break
    return outside

def union_area(polys):
    """Union of convex polygons, exact linear clipping up to float tolerance."""
    covered, area = [], 0.
    for poly in polys:
        parts = [poly]
        for old in covered:
            next_parts = []
            for part in parts:
                if (part.max(0) < old.min(0)-1e-12).any() or (old.max(0) < part.min(0)-1e-12).any():
                    next_parts.append(part)
                else:
                    next_parts.extend(subtract_convex(part, old))
            parts = next_parts
            if not parts:
                break
        area += sum(abs(signed_area(p)) for p in parts)
        covered.append(poly)
    return float(area)

def prism_features(C, F, L):
    # Normalize first: one fixed coordinate scale for geometry tolerances.
    c, f = np.asarray(C, float)/L, np.asarray(F, float)/L
    ca, cn = normgeom(c)
    fa, fn = normgeom(f)
    height = .06
    valid_c = ca > 1e-14
    candidates = np.abs(fn @ cn.T) > COS
    candidates &= valid_c[None, :]
    matches, centroid_hits, counts = [], [], []
    for i, tri in enumerate(f):
        if fa[i] <= 1e-14:
            matches.append(0.); centroid_hits.append(False); counts.append(0)
            continue
        u = tri[1]-tri[0]
        u /= np.linalg.norm(u)
        v = np.cross(fn[i], u)
        polygons, centroid_hit = [], False
        g = tri.mean(0)
        for j in np.flatnonzero(candidates[i]):
            ct, n = c[j], cn[j]
            h = (tri-ct[0]) @ n
            if h.min() > height+1e-12 or h.max() < -height-1e-12:
                continue
            p = clip(tri, n, float(n@ct[0]-height))
            p = clip(p, -n, float(-n@ct[0]-height))
            centroid_ok = abs(float((g-ct[0])@n)) <= height+1e-12
            for a, b in zip(ct, np.roll(ct, -1, axis=0)):
                inward = np.cross(n, b-a)
                inward /= max(np.linalg.norm(inward), 1e-300)
                d = float(inward@a)
                centroid_ok &= float(inward@g-d) >= -1e-12
                p = clip(p, inward, d)
                if len(p) < 3:
                    break
            centroid_hit |= centroid_ok
            if len(p) >= 3:
                projected = np.column_stack(((p-tri[0])@u, (p-tri[0])@v))
                if abs(signed_area(projected)) > 1e-16:
                    polygons.append(projected)
        a = union_area(polygons)
        if a > fa[i]+1e-9:
            raise AssertionError(f"Union exceeds original bulk area: {a} > {fa[i]}")
        matches.append(min(a, fa[i]))
        centroid_hits.append(centroid_hit)
        counts.append(len(polygons))
    return ca*L*L, fa*L*L, np.asarray(matches)*L*L, np.asarray(centroid_hits), counts

def finite_distance(p, C):
    """Exact point-to-finite-triangle distance, including edges/vertices."""
    a, b, c = C[:, 0], C[:, 1], C[:, 2]
    _, normals = normgeom(C)
    height = np.einsum("ij,ij->i", p-a, normals)
    proj = p-height[:, None]*normals
    inside = np.ones(len(C), bool)
    best = np.full(len(C), np.inf)
    for x, y in ((a,b),(b,c),(c,a)):
        edge = y-x
        inward = np.cross(normals, edge)
        inside &= np.einsum("ij,ij->i", proj-x, inward) >= -1e-11
        t = np.clip(np.einsum("ij,ij->i", p-x, edge)/np.maximum(np.sum(edge*edge,1),1e-300),0,1)
        dist = np.linalg.norm(p-x-t[:, None]*edge, axis=1)
        best = np.minimum(best, dist)
    area, _ = normgeom(C)
    inside &= area > 1e-14
    best[inside] = np.minimum(best[inside], np.abs(height[inside]))
    return float(best.min())

def selftest():
    tests = []
    c = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    f = c+np.array([0,0,.03])
    def rat(cc, ff, L=1):
        ca,fa,a,_,_=prism_features(cc,ff,L)
        return a.sum()/ca.sum()
    def check(name, val, expect, atol=1e-8):
        assert abs(val-expect)<atol, (name,val,expect)
        tests.append(dict(name=name, actual=float(val), expected=float(expect), status="PASS"))
    check("same_footprint_inside",rat(c,f),1.)
    check("outside_height",rat(c,c+[0,0,.061]),0.)
    check("perpendicular",rat(c,np.array([[[0,0,0],[1,0,0],[0,0,1]]],float)),0.)
    check("normal_reversal",rat(c[:, ::-1],f[:, ::-1]),1.)
    check("two_bulk_layers",rat(c,np.concatenate((f,c+[0,0,.04]))),2.)
    # Duplicate C doubles denominator, not union numerator.
    check("duplicate_prism_union",prism_features(np.concatenate((c,c)),f,1)[2].sum(),.5)
    square = np.array([[[0,0,0],[1,0,0],[1,1,0]],[[0,0,0],[1,1,0],[0,1,0]]],float)
    alt = np.array([[[0,0,0],[1,0,0],[0,1,0]],[[1,0,0],[1,1,0],[0,1,0]]],float)
    check("retriangulation",rat(square,alt+[0,0,.03]),1.)
    check("partial_triangle_centroid_outside",rat(c,np.array([[[-1,0,.03],[1,0,.03],[-1,1,.03]]])),.5)
    # Both target and scale transformed, orthogonal rotation + translation.
    R = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    check("rigid_and_scale",rat(3*c@R+[2,4,5],3*f@R+[2,4,5],3),1.)
    check("finite_surface_distance",finite_distance(np.array([2.,0.,0.]),c),1.)
    check("finite_interior_height",finite_distance(np.array([.2,.2,.03]),c),.03)
    # Analytic overlapping squares (union 1.5).
    p = np.array([[0,0],[1,0],[1,1],[0,1]],float)
    check("polygon_union_overlap",union_area([p,p+[.5,0]]),1.5)
    write(OUT/"analytic_checks.json",tests)
    print("ANALYTIC_CHECKS",len(tests),flush=True)

def make_manifest():
    jobs = []
    sources = [
        ("CHAIR_SEAT", ROOT/"experiments/r2_hinge_v2_cpu_v1/protocol.json"),
        ("CHAIR_BACK", ROOT/"experiments/r2_chair_back_transfer_v1/eval_manifest.json"),
        ("BENCH", ROOT/"experiments/r2_bench_clean_multipart_seam_v1/eval_manifest.json")]
    for domain, p in sources:
        for q in read(p)["requests"]:
            rid=q["request_id"]
            if domain=="CHAIR_SEAT":
                base=ROOT/"experiments/r2_seam_quality_wave1_wave2_v1/evaluation/PLAIN_S2000"/rid
                final=ROOT/"experiments/r2_hinge_v2_cpu_v1/cells/ALPHA_HOLE"/rid/"post.npz"
                label="PLAIN_S2000"
            elif domain=="CHAIR_BACK":
                base=ROOT/"experiments/r2_chair_back_transfer_v1/evaluation/S2000"/rid
                final=base/"cpu/FINAL_POST/post.npz"
                label="PLAIN_S2000"
            else:
                base=ROOT/"experiments/r2_bench_clean_multipart_seam_v1/evaluation/NEW_HEAD"/rid
                final=ROOT/"experiments/r2_bench_clean_multipart_seam_v1/cpu/NEW_HEAD"/rid/"CLEAN_FINAL_POST/post.npz"
                label="NEW_HEAD"
            desc=q.get("query_descriptor",{})
            facts=q.get("input_facts",{})
            B=q.get("boundary_edges", facts.get("C_boundary_edges"))
            for stage,path in [("RAW",base/"raw/raw.npz"),("BASE_POST",base/"post/post.npz"),("FINAL_POST",final)]:
                jobs.append(dict(domain=domain,panel="MAIN92",model=label,request_id=rid,
                    condition_id=q["condition_id"],parent_id=q["parent_id"],noise_index=q["noise_index"],
                    split=q.get("split",q.get("group")),part_type=q.get("part_type",domain),
                    K=q["K"],N=q["N"],boundary_edges=B,closed_C=q.get("closed_C", B==0 if B is not None else None),
                    L=desc.get("identity",{}).get("L0",9.208138264587333),
                    stage=stage,path=str(path),raw_path=str(base/"raw/raw.npz"),
                    sha256=sha(path) if path.exists() else None))
    assert len(jobs)==276
    # Exact older model cells pictured by the user, additional cached reads only.
    extra=[("SEAM_S15000","459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n0"),
           ("SEAM_S21000","1dd7409c95d59d0da192483aa282f8e5__SEAT_SURFACE_1__n1")]
    for label,rid in extra:
        template=next(j for j in jobs if j["domain"]=="BENCH" and j["request_id"]==rid)
        base=ROOT/"experiments/r2_bench_parts_convergence_v1/evaluation"/label/rid
        final=ROOT/"experiments/r2_bench_clean_multipart_seam_v1/historical"/label/rid/"CLEAN_FINAL_POST/post.npz"
        for stage,path in [("RAW",base/"raw/raw.npz"),("BASE_POST",base/"post/post.npz"),("FINAL_POST",final)]:
            j=dict(template,panel="USER_PICTURE_APPENDIX",model=label,stage=stage,path=str(path),
                   raw_path=str(base/"raw/raw.npz"),sha256=sha(path) if path.exists() else None)
            jobs.append(j)
    for i,j in enumerate(jobs):
        j["job_id"]=f"{i:03d}"
        j["raw_sha256"]=sha(j["raw_path"]) if Path(j["raw_path"]).exists() else None
    manifest=dict(version=VERSION,main_requests=92,main_cells=276,appendix_cached_cells=6,
                  model_calls=0,GPU_calls=0,training_updates=0,new_noise=0,new_surface_queries=0,
                  input_manifests={str(p):sha(p) for _,p in sources},jobs=jobs,
                  definition=dict(H_over_L=.06,normal_abs_dot_gt=COS,
                    area="per bulk triangle exact finite-prism intersection union, original 3D area",
                    denominator="sum of preserved C triangle areas; layers not quotient-collapsed",
                    seam="original unanchored raw corner traced through source_corner_ids; finite C distance",
                    shared_C="primary includes shared-C bulk; separate excluded-shared-C-vertex sensitivity",
                    bridge="excluded from BULK numerator and denominator",
                    semantics="descriptors only; no illegal/semantic claim; not higher/lower universally better",
                    threshold_tuning=False,GT_style_target=False))
    write(OUT/"manifest.json",manifest)
    return manifest

def process(j):
    start=time.perf_counter()
    output=OUT/"cells"/(j["job_id"]+".json")
    if output.exists():
        old=read(output)
        if old["job"]==j and old["status"]=="COMPLETE":
            return dict(id=j["job_id"],status="REUSED",seconds=0)
        raise RuntimeError("Existing nonmatching/failed cell retained: "+str(output))
    try:
        if not j["sha256"] or not j["raw_sha256"]:
            write(output,dict(job=j,status="BLOCKED_MISSING_INPUT"))
            return dict(id=j["job_id"],status="BLOCKED",seconds=0)
        if sha(j["path"])!=j["sha256"] or sha(j["raw_path"])!=j["raw_sha256"]:
            raise AssertionError("Frozen input hash changed")
        a=arrays(j["path"]); raw=arrays(j["raw_path"])
        tris=np.asarray(a["output"],float)
        K=int(a["known_faces"]); C=tris[:K]
        if K!=j["K"]:
            raise AssertionError("K differs from request")
        if "face_role" in a:
            roles=a["face_role"].astype(str)
        else:
            roles=np.full(len(tris),"BULK",dtype="U8")
            roles[:K]="C"; roles[a["retained_seam_mask"]]="SEAM"
        assert len(roles)==len(tris) and np.all(roles[:K]=="C")
        ids=np.asarray(a.get("parent_source_raw_slot",a.get("raw_slot_ids",a.get("source_face_ids"))))
        assert ids.shape==(len(tris),)
        bulk=np.flatnonzero(roles=="BULK")
        seam=np.flatnonzero(roles=="SEAM")
        ca,fa,matched,centroids,overlap=prism_features(C,tris[bulk],j["L"])
        cvids=set(np.asarray(a["faces"][:K]).ravel().tolist())
        shared=np.array([bool(cvids.intersection(a["faces"][i].tolist())) for i in bulk])
        active=int(np.count_nonzero(raw["raw_seam_slot_mask"]))
        corners=a.get("source_corner_ids",a.get("base_source_corner_ids"))
        if len(seam) and corners is None:
            raise RuntimeError("No audited output-to-raw corner identity")
        seamrows=[]
        for i in seam:
            slot=int(ids[i])
            unknown=np.flatnonzero(raw["raw_anchor_ids"][slot]<0)
            if len(unknown)!=1:
                raise AssertionError("Active seam must have exactly one free corner")
            matches=np.flatnonzero(corners[i]==unknown[0])
            if len(matches)!=1:
                raise AssertionError("Seam tip corner map ambiguous")
            tip=tris[i,int(matches[0])]
            dist=finite_distance(tip/j["L"],C/j["L"])
            area,n=normgeom(tris[i:i+1])
            seamrows.append(dict(output_face=int(i),raw_slot=slot,tip_corner=int(matches[0]),
                                 tip_distance_over_L=dist,seam_area=float(area[0]),
                                 tip_x=float(tip[0]),tip_y=float(tip[1]),tip_z=float(tip[2])))
        distances=np.array([r["tip_distance_over_L"] for r in seamrows])
        status=("NA_CLOSED_C" if j.get("closed_C") else
                "NA_NO_ACTIVE_SEAM" if active==0 else
                "NA_ALL_ACTIVE_SEAM_DELETED" if not len(seam) else "OK")
        areaC=float(ca.sum()); areaF=float(fa.sum()); hit=float(matched.sum())
        stats={k:j[k] for k in ("job_id","domain","panel","model","request_id","condition_id",
                               "parent_id","noise_index","split","part_type","N","K","stage","closed_C")}
        stats.update(status="COMPLETE",C_area=areaC,bulk_area=areaF,matched_parallel_area=hit,
                     Phantom_Layer_Ratio=hit/areaC if areaC else None,
                     bulk_parallel_area_fraction=hit/areaF if areaF else None,
                     no_shared_C_vertex_layer_ratio=float(matched[~shared].sum()/areaC) if areaC else None,
                     centroid_full_face_ratio=float(fa[centroids].sum()/areaC) if areaC else None,
                     C_degenerate_faces=int(np.count_nonzero(ca<=1e-14*j["L"]**2)),
                     bulk_faces=len(bulk),bulk_degenerate_faces=int(np.count_nonzero(fa<=1e-14*j["L"]**2)),
                     matched_bulk_faces=int(np.count_nonzero(matched>1e-12*j["L"]**2)),
                     retained_seam=len(seam),active_raw_seam=active,seam_deleted=active-len(seam),
                     seam_status=status,Seam_Extension_mean_over_L=float(distances.mean()) if len(distances) else None,
                     Seam_Extension_median_over_L=float(np.median(distances)) if len(distances) else None,
                     Seam_Extension_min_over_L=float(distances.min()) if len(distances) else None,
                     Seam_Extension_max_over_L=float(distances.max()) if len(distances) else None,
                     seam_tip_within_005L_fraction=float(np.mean(distances<=.005)) if len(distances) else None,
                     source_C_span_model=np.ptp(C.reshape(-1,3),axis=0).tolist(),
                     source_C_span_over_L=(np.ptp(C.reshape(-1,3),axis=0)/j["L"]).tolist(),
                     source_C_area_centroid=(np.sum(C.mean(1)*ca[:,None],axis=0)/areaC).tolist() if areaC else None)
        faces=[dict(output_face=int(i),raw_slot=int(ids[i]),area=float(ar),
                    prism_union_area=float(ma),centroid_inside=bool(ce),
                    shared_C_vertex=bool(sh),prism_polygons=int(co))
               for i,ar,ma,ce,sh,co in zip(bulk,fa,matched,centroids,shared,overlap)]
        payload=dict(job=j,status="COMPLETE",stats=stats,bulk_faces=faces,seam_tips=seamrows,
                     seconds=time.perf_counter()-start,rss_bytes=psutil.Process().memory_info().rss,
                     model_calls=0,torch_imported="torch" in sys.modules)
        if payload["torch_imported"]:
            raise AssertionError("Neural framework imported")
        write(output,payload)
        return dict(id=j["job_id"],status="COMPLETE",seconds=payload["seconds"],rss_bytes=payload["rss_bytes"])
    except Exception:
        write(output,dict(job=j,status="FAILED",traceback=traceback.format_exc()))
        return dict(id=j["job_id"],status="FAILED",seconds=time.perf_counter()-start)

def aggregate(manifest):
    rows, faces, tips, failures=[],[],[],[]
    for j in manifest["jobs"]:
        p=OUT/"cells"/(j["job_id"]+".json")
        if not p.exists():
            failures.append(dict(job_id=j["job_id"],status="NOT_RUN"));continue
        x=read(p)
        if x["status"]!="COMPLETE":
            failures.append(dict(job_id=j["job_id"],status=x["status"]));continue
        rows.append(x["stats"])
        identity={k:x["stats"][k] for k in ("job_id","domain","panel","model","request_id","stage")}
        faces.extend(dict(identity,**r) for r in x["bulk_faces"])
        tips.extend(dict(identity,**r) for r in x["seam_tips"])
    csvwrite(OUT/"per_output.csv",rows);csvwrite(OUT/"per_bulk_face.csv",faces);csvwrite(OUT/"per_seam_tip.csv",tips)
    write(OUT/"status.json",dict(complete=len(rows),expected=len(manifest["jobs"]),failures=failures))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["prepare","run","aggregate","status"],default="status")
    ap.add_argument("--workers",type=int,default=4)
    args=ap.parse_args()
    if args.stage=="prepare":
        selftest()
        if (OUT/"manifest.json").exists():
            raise RuntimeError("Manifest already frozen")
        m=make_manifest()
        print("FROZEN",len(m["jobs"]),flush=True)
        return
    m=read(OUT/"manifest.json")
    if args.stage=="run":
        start=time.perf_counter()
        jobs=m["jobs"]
        # First authorized cell is measured before additional workers.
        first=process(jobs[0]);print("FIRST",json.dumps(first),flush=True)
        rss=first.get("rss_bytes",64*1024**2)
        mem=psutil.virtual_memory().available
        workers=min(args.workers,4,max(1,psutil.cpu_count()//4),max(1,int((mem-2*1024**3)/(1.2*rss))))
        dispatch=dict(workers=workers,first=first,cpu_percent=psutil.cpu_percent(interval=.3),
                      available_ram=mem,load=os.getloadavg(),threads_per_worker=1,GPU_workers=0)
        write(OUT/"dispatch.json",dispatch)
        results=[first]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(process,jobs[1:]):
                results.append(r)
                if len(results)%20==0 or r["status"] in ("FAILED","BLOCKED"):
                    print("PROGRESS",len(results),"/",len(jobs),json.dumps(r),flush=True)
        rows=aggregate(m)
        write(OUT/"execution.json",dict(seconds=time.perf_counter()-start,dispatch=dispatch,cells=results,
              training_updates=0,model_calls=0,GPU_calls=0,new_noise=0,new_surface_queries=0,
              resources_after=dict(memory_available=psutil.virtual_memory().available,load=os.getloadavg())))
    elif args.stage=="aggregate":
        aggregate(m)
    else:
        print(json.dumps(read(OUT/"status.json") if (OUT/"status.json").exists() else dict(status="FROZEN")))

if __name__=="__main__":
    main()
