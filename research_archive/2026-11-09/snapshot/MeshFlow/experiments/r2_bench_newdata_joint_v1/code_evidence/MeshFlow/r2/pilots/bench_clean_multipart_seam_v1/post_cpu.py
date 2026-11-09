"""Frozen Bench BASE -> HINGE_OPP015 -> ALPHA -> HOLE, CPU only.

Reuses the retained Chair operators verbatim. No D0/anti-collapse guard, model,
noise, query generation, geometric repair outside the registered cascade.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse, collections, concurrent.futures, inspect, json, multiprocessing
from pathlib import Path
import resource, sys, time, traceback, types
import numpy as np
from r2.pilots.seam_abc_v1 import evaluate as E, eval_common as C
from r2.tools.hinge_opposite_v1 import engine as H
from r2.tools.hinge_v2_cpu_v1 import alpha, holes, scoring as M
from r2.pilots.chair_back_transfer_v1.post_cpu import save_npz

ROOT = C.ROOT / "experiments/r2_bench_clean_multipart_seam_v1"
SOURCE = C.ROOT / "experiments/r2_bench_parts_convergence_v1"
LABELS = ("NATIVE_S21000", "SEAM_S15000", "SEAM_S21000")
STAGES = ("HINGE_OPP015", "ALPHA", "CLEAN_FINAL_POST")
VERSION = "bench_clean_original_hinge_opposite_alpha_hole_v1"


def score_current(q, mesh, dest, label, seam_model=True, existing_pairs=None):
    """Original 179-field bridge-aware scorer; only applicability is routed."""
    ns = dict(vars(M))
    for name, obj in vars(M).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(M):
            fn = types.FunctionType(obj.__code__, ns, obj.__name__, obj.__defaults__, obj.__closure__)
            fn.__kwdefaults__ = obj.__kwdefaults__
            ns[name] = fn
    old_reach = ns["role_reach"]
    def scoped_reach(*args, **kwargs):
        internal, stats, rows = old_reach(*args, **kwargs)
        if not seam_model or int(q["boundary_edges"]) == 0:
            status = "NA_NOT_SEAM_REPRESENTATION" if not seam_model else "NA_CLOSED_C_NO_SEAM_INTERFACE"
            internal = dict(status=status, **{k: None for k in E.INTERNAL_FIELDS})
        return internal, stats, rows
    ns["role_reach"] = scoped_reach
    a = dict(mesh)
    if "index_domain" not in a:
        a["index_domain"] = np.asarray("SOURCE_C_SHARED_BASE_GEO")
    if np.any(a.get("face_role", np.asarray([])) == "BRIDGE"):
        existing_pairs = None
    result = ns["score"](q, a, dest, label, existing_pairs=existing_pairs)
    return result


def pair_keys(pairs):
    return {kind: {(tuple(r["pair"]), r["classification"]) for r in scan["contacts"]}
            for kind, scan in pairs.items()}


def ledger(mesh, pairs, K):
    m = holes.initial(mesh)
    ids = m["parent_source_raw_slot"].copy()
    bridge = ids < 0
    ids[bridge] = -1 - np.arange(int(bridge.sum()))
    oldcorners = m.get("source_corner_ids", m.get("base_source_corner_ids"))
    corners = np.vstack((oldcorners, np.tile(np.arange(3), (int(bridge.sum()), 1))))
    return H.Q.connectivity_ledger(m["vertices"], m["faces"], K, pairs, ids, corners)


def stage_invariants(base, current, previous, pairs, before_pairs, K):
    if current["C"].tobytes() != base["C"].tobytes():
        raise AssertionError("Original C coordinate/order changed")
    if not np.array_equal(current["faces"][:K], base["faces"][:K]):
        raise AssertionError("Original C indexed connectivity changed")
    if not np.array_equal(current["vertices"][np.unique(base["faces"][:K])],
                          base["vertices"][np.unique(base["faces"][:K])]):
        raise AssertionError("Original C vertex coordinates changed")
    old = ledger(previous, before_pairs, K)
    new = ledger(current, pairs, K)
    m = holes.initial(previous)
    affected = m["parent_source_raw_slot"][K:]
    paired = H.Q.compare_ledgers(old, new, before_pairs, pairs, affected[affected >= 0].tolist())
    for key in ("CF_lost_legal_edges", "FF_lost_legal_edges", "CF_new_illegal_pairs",
                "FF_new_illegal_pairs", "deleted_source_faces"):
        if paired["summary"][key] != 0:
            raise AssertionError("Frozen guard violated " + key)
    # All newly appearing exact indexed nonmanifold or same-direction edges are
    # forbidden, including incidental edges that are not detector-legal.
    def bad(mesh):
        faces=mesh["faces"];m=holes.initial(mesh)
        sf=m["parent_source_raw_slot"].copy();new=sf<0
        sf[new]=-1-np.arange(int(new.sum()))
        sc=m.get("source_corner_ids",m.get("base_source_corner_ids"))
        sc=np.vstack((sc,np.tile(np.arange(3),(int(new.sum()),1))))
        inc = H.PE.incidence(faces)
        nm, same = set(), set()
        for edge, rows in inc.items():
            identity=H.B.edge_identity(rows,sf,sc)
            if len(rows) > 2:
                nm.add(identity)
            elif len(rows) == 2:
                i, a, b = rows[0]; j, c, d = rows[1]
                if tuple(faces[i, [a,b]]) != tuple(faces[j, [c,d]][::-1]):
                    same.add(identity)
        return nm, same
    old_nm, old_same = bad(previous)
    new_nm, new_same = bad(current)
    if new_nm-old_nm or new_same-old_same:
        raise AssertionError("New nonmanifold/same-winding indexed edge")
    paired["full_indexed_guard"] = dict(new_nonmanifold=0, new_same_winding=0,
        C_coordinate_winding_and_vertex_IDs_preserved=True)
    return paired


def process_one(q, label, base_path, pre_path, raw_path, pairs_path, dest,
                seam_model=True, registration_sha=""):
    """Callable from the new-model evaluator after its single official D0."""
    dest = C.resolve(dest)
    paths = {k: C.resolve(v) for k,v in dict(base=base_path, pre=pre_path, raw=raw_path, pairs=pairs_path).items()}
    identity = dict(version=VERSION, label=label, request_id=q["request_id"],
                    registration_sha256=registration_sha, seam_model=seam_model,
                    sources={str(p): C.sha(p) for p in paths.values()})
    done = dest/"complete.json"
    if done.exists():
        receipt = C.read(done)
        if receipt["identity"] != identity:
            raise ValueError("Completed CPU producer changed")
        E.verify(receipt["files"])
        return receipt
    if dest.exists():
        raise RuntimeError("Partial CPU attempt retained; explicit engineering recovery required: "+str(dest))
    tick = time.perf_counter()
    C.write(dest/"started.json", dict(identity=identity, pid=os.getpid(), utc=C.utc()))
    try:
        base = C.arrays(paths["base"]); pre = C.arrays(paths["pre"])["output"]
        rawdata = C.arrays(paths["raw"]); raw = rawdata.get("raw", rawdata.get("output"))
        pairs = C.read(paths["pairs"])
        K, N = int(q["K"]), int(q["N"])
        if raw.shape != pre.shape or raw.shape != (N,3,3):
            raise ValueError("Frozen RAW/PRE slot domains differ")
        inp = C.arrays(q["inference_npz"])
        if base["C"].tobytes() != inp["C_vertices"][inp["C_faces_original"]].tobytes():
            raise ValueError("Base C is not source-restored original input")
        rows = []
        baseline = score_current(q, base, dest/"BASE_POST/scores", label, seam_model, pairs)
        rows.append(dict(baseline["row"], label=label, stage="BASE_POST", part_type=q["part_type"],
                         part_family=q["part_family"], closed_C=q["closed_C"],
                         seam_applicability=baseline["score"]["seam_to_bulk"]["status"]))
        original_pairs = pairs; current = base; phases = []
        for stage in STAGES:
            stage_start = time.perf_counter(); previous = current; oldpairs = pairs
            if stage == "HINGE_OPP015":
                current, records, summary, _, _, pairs = H.decode(current, pre, raw, pairs, .015, require_opposite_winding=True)
            elif stage == "ALPHA":
                current, records, summary, _, _, pairs = alpha.decode(current, pre, raw, pairs, .015, alphas=True)
            else:
                current, pairs, _, records, summary = holes.decode(current, pairs)
            paired = stage_invariants(base, current, previous, pairs, oldpairs, K)
            p = dest/stage
            save_npz(p/"post.npz", current)
            for name, value in (("pairs",pairs),("summary",summary),("candidate_records",records),("paired_ledger",paired)):
                C.write(p/(name+".json"), value)
            C.csvwrite(p/"candidates.csv", records)
            phases.append(dict(stage=stage, seconds=time.perf_counter()-stage_start, summary=summary))
        final = score_current(q, current, dest/"CLEAN_FINAL_POST/scores", label, seam_model, pairs)
        if pair_keys(final["pairs"]) != pair_keys(pairs):
            raise AssertionError("Independent final full detector disagrees with cascade cache")
        rows.append(dict(final["row"], label=label, stage="CLEAN_FINAL_POST", part_type=q["part_type"],
                         part_family=q["part_family"], closed_C=q["closed_C"],
                         seam_applicability=final["score"]["seam_to_bulk"]["status"]))
        paired_final = stage_invariants(base,current,base,pairs,original_pairs,K)
        C.write(dest/"base_final_paired_ledger.json",paired_final)
        C.write(dest/"metric_rows.json",rows)
        if "torch" in sys.modules:
            raise RuntimeError("Pure CPU cascade imported neural framework")
        result = dict(status="COMPLETE",identity=identity,label=label,request_id=q["request_id"],
                      stages=phases,seconds=time.perf_counter()-tick,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      model_calls=0,D0_calls=0,GPU_calls=0,new_noise=0,new_queries=0,
                      expanded_base_scores=1,final_scores=1,
                      files={str(p):C.sha(p) for p in dest.rglob("*") if p.is_file()})
        C.write(done,result)
        print("CLEAN_POST_COMPLETE",label,q["request_id"],round(result["seconds"],2),flush=True)
        return result
    except BaseException as exc:
        C.write(dest/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc(),model_calls=0))
        raise


def jobs():
    out = []
    for label in LABELS:
        manifest = SOURCE/"evaluation/score_registrations"/(label+".json")
        reg = C.read(manifest)
        if len(reg["requests"]) != 48:
            raise ValueError("Expected original fixed48 panel")
        for q in reg["requests"]:
            base = SOURCE/"evaluation"/label/q["request_id"]
            out.append(dict(q=q,label=label,base_path=str(base/"post/post.npz"),
                pre_path=str(base/"post/pre_geo.npz"),raw_path=str(base/"raw/raw.npz"),
                pairs_path=str(SOURCE/"evaluation/scores"/label/q["request_id"]/"POST/pairs.json"),
                dest=str(ROOT/"historical"/label/q["request_id"]),seam_model=label.startswith("SEAM")))
    return out


def freeze():
    path = ROOT/"historical_registration.json"
    if path.exists():
        r=C.read(path);E.verify(r["protected"]);verify_code(r);return r
    js=jobs();protected={}
    for j in js:
        for key in ("base_path","pre_path","raw_path","pairs_path"):
            protected[j[key]]=C.sha(j[key])
        for key in ("inference_npz","metadata_npz","noise_npz"):
            protected[j["q"][key]]=C.sha(j["q"][key])
        p=j["q"]["query_descriptor"]["path"];protected[p]=C.sha(p)
    code=H.Q.definition_hashes()
    for module in (H,alpha,holes,M):
        code[str(Path(module.__file__))]=C.sha(module.__file__)
    code[str(Path(__file__))]=C.sha(__file__)
    r=dict(status="FROZEN",version=VERSION,jobs=js,protected=protected,code=code,
           cells=144,delta=.015,chain="BASE_POST -> HINGE_OPP015 -> ALPHA -> HOLE",
           D0_guard=False,GEO_guard=False,source_C_hard_preserved=True,
           model_calls=0,D0_calls=0,new_queries=0,new_noise=0)
    C.write(path,r);return r


def verify_code(reg):
    revision=ROOT/"historical_execution_revision_01.json"
    if revision.exists():
        r=C.read(revision)
        if r["registration_sha256"]!=C.sha(ROOT/"historical_registration.json"):
            raise ValueError("Original registration changed")
        E.verify(r["prior_files"]);E.verify(r["code"])
    else:E.verify(reg["code"])


def run(workers=2):
    r=freeze();dest=ROOT/"historical_completion.json"
    if dest.exists():
        d=C.read(dest);E.verify(d["files"]);return d
    if not (ROOT/"registration.json").exists() or not (ROOT/"resources_before_preparation.json").exists():
        raise RuntimeError("Root experiment/resource registration must precede CPU dispatch")
    resource_before=C.resources()
    path=ROOT/("historical_resources_before_"+str(time.time_ns())+".json")
    C.write(path,resource_before)
    identity=C.sha(ROOT/"historical_registration.json")
    js=[dict(j,registration_sha=identity) for j in r["jobs"]]
    tick=time.perf_counter();first=process_one(**js[0]);snap=C.resources()
    from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
    count=worker_count(snap,first,workers)
    C.write(ROOT/("historical_dispatch_"+str(time.time_ns())+".json"),
        dict(workers=count,threads_per_worker=1,resources=snap,observed_first_RSS_KiB=first["peak_RSS_KiB"]))
    receipts=[first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context("spawn")) as pool:
        pending=iter(js[1:]);active={}
        while True:
            while len(active)<count:
                job=next(pending,None)
                if job is None:break
                active[pool.submit(process_one,**job)]=job
            if not active:break
            ready,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in ready:
                receipts.append(f.result());del active[f]
    E.verify(r["protected"]);verify_code(r)
    rows=[row for j in js for row in C.read(C.resolve(j["dest"])/"metric_rows.json")]
    C.csvwrite(ROOT/"historical_per_output.csv",rows)
    files={str(C.resolve(j["dest"])/"complete.json"):C.sha(C.resolve(j["dest"])/"complete.json") for j in js}
    files[str(ROOT/"historical_per_output.csv")]=C.sha(ROOT/"historical_per_output.csv")
    result=dict(status="COMPLETE",requests=144,expanded_baseline_scores=144,final_scores=144,
                workers=count,seconds=time.perf_counter()-tick,files=files,model_calls=0,D0_calls=0)
    C.write(dest,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,choices=(1,2),default=2)
    args=p.parse_args();print(json.dumps(run(args.workers),ensure_ascii=False))
