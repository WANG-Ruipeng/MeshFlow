"""CPU-only A-VMR/PMR diagnostics on the 48 frozen USER RAW outputs.

No model, torch, sampling, postprocessing, S_join or original scorer imports.
Run --stage all; completed per-output receipts are verified and reused.
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

VERSION = "r2_avmr_pmr_all_C_open_boundary_v1"
REPO = Path(__file__).resolve().parents[3]
OLD = REPO / "experiments/r2_user_mix_from_official_v1"
OUT = REPO / "experiments/r2_avmr_pmr_offline_v1"
LABELS = ("R2_FROZEN", "R2_CONT_OLD", "R2_CONT_USER", "R2_USER_MIX")
L = float(np.sqrt(3.) * 2 / .3762)
SETTINGS = [("raw_tau_1e-6", 1e-6, "model_coordinates", 1e-6)]
SETTINGS += [(f"L_tau_{v:g}", v * L, "fixed_L_fraction", v)
             for v in (1e-6, 1e-4, .001, .002, .005, .01, .02)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def bit_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def parent_path(q):
    return REPO / "native_t1/runs/control_mvp48_v1/data/meshes" / (q["parent_id"] + ".npz")


def event(kind, **extra):
    with (OUT / "ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "event": kind, **extra}, ensure_ascii=False) + "\n")


def resources():
    def command(args):
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=10)
            return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        except Exception as e:
            return {"error": str(e)}
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_logical": os.cpu_count(), "loadavg": list(os.getloadavg()),
        "memory": Path("/proc/meminfo").read_text(),
        "cpu_processes": command(["ps", "-eo", "pid,comm,pcpu,pmem", "--sort=-pcpu"]),
        "gpu": command(["nvidia-smi"]),
        "dispatch": {"CPU_workers": 1, "BLAS_threads": 1, "GPU_workers": 0,
                     "reason": "Only 48 small endpoint nearest-vertex queries; process dispatch overhead exceeds work. Independent asset and literature audits ran in parallel."}
    }


def topology(C, source_ids):
    ids = np.asarray(source_ids, dtype=np.int64)
    if ids.shape != C.shape[:2]: raise ValueError("C vertex-ID shape mismatch")
    coords = {}
    edges = collections.Counter()
    for face, face_ids in zip(C, ids):
        if len(set(map(int, face_ids))) != 3: raise ValueError("Repeated source vertex in C face")
        for p, v in zip(face, face_ids):
            v = int(v)
            if v in coords and not bit_equal(coords[v], p):
                raise ValueError("Source vertex has inconsistent coordinates")
            coords[v] = p.copy()
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edges[tuple(sorted((int(face_ids[i]), int(face_ids[j]))))] += 1
    boundary_edges = sorted(e for e, count in edges.items() if count == 1)
    boundary_ids = sorted({v for e in boundary_edges for v in e})
    points = np.asarray([coords[v] for v in boundary_ids], dtype=np.float64).reshape(-1, 3)
    coordinate_edges = collections.Counter()
    for face in C:
        pts = [tuple(map(float, p)) for p in face]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            coordinate_edges[tuple(sorted((pts[i], pts[j])))] += 1
    cb = {p for e, count in coordinate_edges.items() if count == 1 for p in e}
    sb = {tuple(map(float, coords[v])) for v in boundary_ids}
    audit = {
        "boundary_vertex_count": len(boundary_ids), "boundary_edge_count": len(boundary_edges),
        "C_vertex_count": len(coords), "C_nonmanifold_edge_count": sum(n > 2 for n in edges.values()),
        "coordinate_boundary_vertex_count": len(cb),
        "source_and_coordinate_boundary_locations_equal": cb == sb,
        "coincident_distinct_source_vertex_count": len(coords) - len({tuple(map(float, p)) for p in coords.values()}),
        "boundary_length_model": sum(float(np.linalg.norm(coords[a].astype(float) - coords[b])) for a, b in boundary_edges),
    }
    return boundary_ids, points, audit


def nearest(points, free):
    vertices = np.asarray(free, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(points).all() or not np.isfinite(vertices).all():
        raise ValueError("Nonfinite geometry")
    if len(vertices) == 0:
        return np.full(len(points), np.inf), np.full(len(points), -1, dtype=int)
    d2 = np.sum((points[:, None, :] - vertices[None, :, :]) ** 2, axis=-1)
    indices = np.argmin(d2, axis=1)
    return np.sqrt(d2[np.arange(len(points)), indices]), indices


def metric(distances, threshold):
    count = len(distances)
    if count == 0:
        return {"status": "NA_EMPTY_BOUNDARY", "boundary_vertex_count": 0,
                "matched_count": 0, "VMR": None, "perfect": None}
    matched = int(np.count_nonzero(distances < threshold))
    return {"status": "PASS", "boundary_vertex_count": count, "matched_count": matched,
            "VMR": matched / count, "perfect": int(matched == count)}


def self_check():
    cases = []
    C = np.asarray([[[0,0,0],[1,0,0],[0,1,0]],
                    [[0,1,0],[1,0,0],[1,1,0]]], dtype=np.float32)
    ids = np.array([[0,1,2],[2,1,3]])
    bid, pts, audit = topology(C, ids)
    assert len(bid) == 4 and audit["boundary_edge_count"] == 4
    cases.append("two triangles: internal diagonal excluded, four unique boundary vertices")
    exact, _ = nearest(pts, C)
    assert metric(exact, 1e-6)["VMR"] == 1 and metric(exact, 1e-6)["perfect"] == 1
    cases.append("exact duplicated patch: vertex metric passes; no topology guarantee")
    shifted = C.astype(float) + np.array([0,0,0.1])
    ds, _ = nearest(pts, shifted)
    assert metric(ds, .05)["VMR"] == 0
    cases.append("separated patch fails strict distance")
    one, _ = nearest(pts, C[:1])
    assert metric(one, 1e-6)["VMR"] == .75 and metric(one, 1e-6)["perfect"] == 0
    cases.append("partial vertex match passes VMR but fails perfect")
    assert metric(np.array([1e-6]), 1e-6)["matched_count"] == 0
    cases.append("threshold equality rejected: strict less than")
    empty, _ = nearest(pts, np.empty((0,3,3)))
    assert metric(empty, 1e-6)["VMR"] == 0
    assert metric(np.empty(0), 1e-6)["status"] == "NA_EMPTY_BOUNDARY"
    cases.append("empty free gives zero; empty boundary gives NA, never vacuous perfect")
    leaked, _ = nearest(pts, np.concatenate([C, shifted]))
    assert metric(leaked, 1e-6)["VMR"] == 1 and metric(ds, 1e-6)["VMR"] == 0
    cases.append("C union F self-match positive control proves why free-only is required")
    macro = np.mean([1 / 4, 8 / 8])
    assert macro == .625 and macro != 9 / 12
    cases.append("sample macro average differs from pooled vertex fraction")
    return {"status": "PASS", "checks": cases, "synthetic_metric_checks": len(cases),
            "model_calls": 0, "new_generated_model_meshes": 0}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OLD / "evaluation/raw_index.csv"
    reg_path = OLD / "evaluation/registration.json"
    index, oldreg = csv_read(index_path), read(reg_path)
    requests = {q["request_id"]: q for q in oldreg["requests"]}
    if len(index) != 48 or len(requests) != 12: raise ValueError("Unexpected panel size")
    if collections.Counter(r["label"] for r in index) != {x: 12 for x in LABELS}: raise ValueError("Unexpected labels")
    if len({(r["label"], r["request_id"]) for r in index}) != 48: raise ValueError("Duplicate output key")
    protected = {str(p): sha(p) for p in
                 (index_path, reg_path, OLD/"R2_USER_MIX_REPORT.md",
                  OLD/"FINAL_ARTIFACT_HASHES.csv", OLD/"evaluation/per_output.csv")}
    for q in requests.values():
        for field, shafield in (("input_path", "input_sha256"), ("condition_source", "condition_source_sha256")):
            p = q[field]
            actual = sha(p)
            if actual != q[shafield]: raise ValueError("Frozen input hash mismatch: " + p)
            protected[p] = actual
        protected[str(parent_path(q))] = sha(parent_path(q))
    for r in index:
        actual = sha(r["raw_path"])
        if actual != r["raw_sha256"]: raise ValueError("Frozen RAW hash mismatch: " + r["raw_path"])
        protected[r["raw_path"]] = actual
    value = {
        "version": VERSION, "source_sha256": sha(__file__), "protected": protected,
        "source_index": str(index_path), "source_registration": str(reg_path),
        "labels": list(LABELS), "outputs": 48, "conditions": 6, "noises_per_condition": 2,
        "domain": "ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY",
        "required_join_boundary_status": "NA_NOT_REGISTERED",
        "interpretation": "Descriptive boundary-vertex matching only, not task success or actual attachment",
        "boundary": "C-only source vertex-ID undirected edge incidence=1; unique endpoint IDs; source IDs already in C corner order",
        "candidate_vertices": "all raw free face corners only; nearest Euclidean vertex; many-to-one allowed; no C candidates, no surface or edge distance",
        "comparison": "strict_lt", "L_model_coordinates": L,
        "settings": [{"name": n, "threshold_model": t, "scale": s, "tau": v} for n,t,s,v in SETTINGS],
        "threshold_selection": "Frozen before any nearest-distance scoring; inherited fixed L and existing 0.5/1/2 percent tolerances plus explicit strict and finer settings",
        "paper": "https://arxiv.org/html/2605.08744v1#A1",
        "paper_scale_compatibility": "UNKNOWN: no scoring normalization contract/code verified; Appendix C unit-sphere defect detection not evidence for main-table scale",
        "aggregation": "mean per-output VMR and mean per-output perfect flag; 2 noises per parent means all-output mean equals equal-six-parent mean",
        "budget": {"CPU_outputs": 48, "model_loads": 0, "model_forwards": 0,
                   "training_updates": 0, "new_Gaussian": 0, "new_rollouts": 0,
                   "postprocess_jobs": 0, "S_join": 0},
        "dispatch": {"CPU_workers": 1, "threads": 1, "GPU_workers": 0},
    }
    reg = OUT/"registration.json"
    if reg.exists():
        if read(reg) != value: raise ValueError("Frozen offline registration changed; no silent overwrite")
    else:
        write(reg, value)
        write(OUT/"resources_before.json", resources())
        event("REGISTERED", source_outputs=48, model_calls=0)
    snap = OUT/"source_snapshot/run.py"
    snap.parent.mkdir(exist_ok=True)
    if snap.exists() and sha(snap) != sha(__file__): raise ValueError("Source snapshot drift")
    if not snap.exists(): shutil.copyfile(__file__, snap)
    write(OUT/"metric_self_checks.json", self_check())
    return value, index, requests


def run():
    started = time.perf_counter()
    reg, index, requests = prepare()
    regsha = sha(OUT/"registration.json")
    audits, boundary_rows, domains = [], [], {}
    for q in requests.values():
        cid = q["condition_id"]
        if cid in domains: continue
        with np.load(q["condition_source"], allow_pickle=False) as a:
            C, source_ids = a["C"], a["C_source_vertex_ids"]
            with np.load(parent_path(q), allow_pickle=False) as parent:
                if not bit_equal(parent["model_vertices"][source_ids], C):
                    raise ValueError("Source vertex IDs not aligned with C coordinates")
            bid, pts, facts = topology(C, source_ids)
        if facts["C_nonmanifold_edge_count"]: raise ValueError("Ambiguous nonmanifold C boundary")
        domains[cid] = (C, bid, pts)
        audits.append({"condition_id": cid, "parent_id": q["parent_id"], "region_type": q["region_type"],
                       "K": q["K"], "N": q["N"], "required_interface": "NA_NOT_REGISTERED", **facts})
        for vid, p in zip(bid, pts):
            boundary_rows.append({"condition_id": cid, "source_vertex_id": vid, "x": p[0], "y": p[1], "z": p[2]})
    csv_write(OUT/"condition_audit.csv", audits)
    csv_write(OUT/"boundary_vertices.csv", boundary_rows)
    scored, failed = 0, 0
    for row in index:
        q = requests[row["request_id"]]
        dest = OUT/"scores"/row["label"]/(row["request_id"]+".json")
        identity = {"registration_sha256": regsha, "raw_sha256": row["raw_sha256"],
                    "input_sha256": q["input_sha256"], "label": row["label"], "request_id": row["request_id"]}
        if dest.exists():
            receipt = read(dest)
            if receipt["identity"] != identity: raise ValueError("Existing score identity mismatch")
            if receipt["status"] != "PASS": failed += 1
            continue
        try:
            t0 = time.perf_counter()
            C, bid, pts = domains[q["condition_id"]]
            with np.load(q["input_path"], allow_pickle=False) as z:
                if not bit_equal(C, z["C"]): raise ValueError("Condition versus frozen input mismatch")
            with np.load(row["raw_path"], allow_pickle=False) as a:
                raw = a["raw"].reshape(int(q["N"]),3,3)
                free = raw[int(q["K"]):]
                if not bit_equal(raw[:int(q["K"])], C) or not bit_equal(a["C"], C):
                    raise ValueError("C exact preservation mismatch")
                if not bit_equal(a["free"], free): raise ValueError("Stored free versus raw slice mismatch")
                if raw.dtype != np.float32 or not np.isfinite(raw).all(): raise ValueError("Invalid RAW")
            ds, nearest_ids = nearest(pts, free)
            meta = {k:q[k] for k in ("request_id","condition_id","parent_id","region_type","tier","candidate","N","K")}
            meta["label"] = row["label"]
            vertices = []
            for vid,p,d,j in zip(bid,pts,ds,nearest_ids):
                f = free.reshape(-1,3)[j]
                vertices.append({"source_vertex_id": vid, "cx": float(p[0]), "cy": float(p[1]), "cz": float(p[2]),
                    "nearest_free_face_index": int(j)//3, "nearest_free_corner": int(j)%3,
                    "fx": float(f[0]), "fy": float(f[1]), "fz": float(f[2]),
                    "distance_model": float(d), "distance_L_fraction": float(d/L)})
            values = [{"setting":n, "threshold_model":t, "scale":s, "tau":v, **metric(ds,t)}
                      for n,t,s,v in SETTINGS]
            receipt = {"status":"PASS", "identity":identity, "metadata":meta, "metrics":values,
                       "vertices":vertices, "seconds":time.perf_counter()-t0,
                       "nearest_vertex_distance_mean_model":float(np.mean(ds)),
                       "nearest_vertex_distance_max_model":float(np.max(ds)),
                       "candidate_vertices":int(free.size//3),
                       "zero_area_free_faces":int(np.count_nonzero(np.linalg.norm(np.cross(free[:,1].astype(float)-free[:,0],
                                                  free[:,2].astype(float)-free[:,0]),axis=1)==0))}
            scored += 1
        except Exception as e:
            receipt = {"status":"BLOCKED", "identity":identity, "reason":repr(e), "metadata":row, "metrics":[], "vertices":[]}
            failed += 1
        write(dest,receipt)
        event("OUTPUT_SCORED" if receipt["status"]=="PASS" else "OUTPUT_BLOCKED",
              label=row["label"],request_id=row["request_id"],model_calls=0)
    receipts = [read(OUT/"scores"/r["label"]/(r["request_id"]+".json")) for r in index]
    summarize(receipts)
    after = {p:sha(p) for p in reg["protected"]}
    if after != reg["protected"]: raise ValueError("Protected source changed")
    write(OUT/"protected_after.json", {"status":"PASS","files":after})
    write(OUT/"resources_after.json", resources())
    event("COMPLETE", new_scored=scored, blocked=failed, seconds=time.perf_counter()-started, model_calls=0)
    write(OUT/"completion.json", {"version":VERSION,"status":"COMPLETE" if not failed else "PARTIAL",
        "outputs_registered":48,"outputs_pass":sum(r["status"]=="PASS" for r in receipts),
        "outputs_blocked":sum(r["status"]!="PASS" for r in receipts),"new_scored_this_invocation":scored,
        "protected_files_unchanged":len(after),"metric_checks":read(OUT/"metric_self_checks.json")["synthetic_metric_checks"],
        "model_calls":0,"new_rollouts":0,"POST":0,"S_join":0,
        "wall_seconds":time.perf_counter()-started,"numpy":np.__version__,"python":sys.version})
    hashes = [{"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha(p)}
              for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="artifact_hashes.csv"]
    csv_write(OUT/"artifact_hashes.csv", hashes)
    print(json.dumps(read(OUT/"completion.json"),ensure_ascii=False))
    print((OUT/"REPORT.md").read_text(encoding="utf-8"))


def summarize(receipts):
    per_output, vertices = [], []
    for r in receipts:
        if r["status"]!="PASS":
            per_output.append({**r["metadata"],"status":r["status"],"reason":r.get("reason","")})
            continue
        for m in r["metrics"]:
            per_output.append({**r["metadata"], **m,
                "nearest_vertex_distance_mean_model":r["nearest_vertex_distance_mean_model"],
                "nearest_vertex_distance_max_model":r["nearest_vertex_distance_max_model"],
                "zero_area_free_faces":r["zero_area_free_faces"]})
        vertices += [{**r["metadata"],**v} for v in r["vertices"]]
    csv_write(OUT/"per_output.csv",per_output)
    csv_write(OUT/"per_vertex_distances.csv",vertices)
    summary, condition = [], []
    for label in LABELS:
        for name,*_ in SETTINGS:
            for region in ("ALL","seat","back"):
                rows=[r for r in per_output if r.get("setting")==name and r["label"]==label and
                      (region=="ALL" or r["region_type"]==region)]
                valid=[r for r in rows if r["status"]=="PASS"]
                summary.append({"label":label,"setting":name,"region":region,"registered_outputs":12 if region=="ALL" else 4 if region=="seat" else 8,
                    "scored_outputs":len(valid),
                    "A_VMR":float(np.mean([r["VMR"] for r in valid])) if valid else None,
                    "PMR":float(np.mean([r["perfect"] for r in valid])) if valid else None,
                    "perfect_outputs":sum(r["perfect"] for r in valid),
                    "status":"PASS" if len(valid)==(12 if region=="ALL" else 4 if region=="seat" else 8) else "PARTIAL",
                    "mean_nearest_vertex_distance_L":float(np.mean([r["nearest_vertex_distance_mean_model"]/L for r in valid])) if valid else None})
            for cid in sorted({r["condition_id"] for r in rows} |
                              {r["condition_id"] for r in per_output if r.get("setting")==name and r["label"]==label}):
                v=[r for r in per_output if r.get("setting")==name and r["label"]==label and r["condition_id"]==cid and r["status"]=="PASS"]
                if v: condition.append({"label":label,"setting":name,"condition_id":cid,"region_type":v[0]["region_type"],
                                        "noises":len(v),"A_VMR":float(np.mean([x["VMR"] for x in v])),
                                        "PMR":float(np.mean([x["perfect"] for x in v]))})
    csv_write(OUT/"summary.csv",summary)
    csv_write(OUT/"per_condition.csv",condition)
    keyed={(r["label"],r["request_id"],r.get("setting")):r for r in per_output}
    paired=[]
    for ref in LABELS[:-1]:
        for r in per_output:
            if r["label"]!="R2_USER_MIX" or r["status"]!="PASS":continue
            b=keyed.get((ref,r["request_id"],r["setting"]))
            if b and b["status"]=="PASS":
                paired.append({"reference":ref, "request_id":r["request_id"],"condition_id":r["condition_id"],
                    "candidate":r["candidate"],"region_type":r["region_type"],"setting":r["setting"],
                    "delta_VMR":r["VMR"]-b["VMR"],"delta_perfect":r["perfect"]-b["perfect"],
                    "delta_mean_vertex_distance_L":(r["nearest_vertex_distance_mean_model"]-b["nearest_vertex_distance_mean_model"])/L})
    csv_write(OUT/"paired_noise.csv",paired)
    lines=["# A-VMR / PMR: 48 USER RAW offline diagnostic",
        "", "仅CPU；4模型 × 6条件 × 2噪声。无训练、模型加载/前向、新噪声、新生成、POST或S_join。",
        "", "## 评价范围与文献关系",
        "", "- 沿用 [MeshFIM 附录A Eq8–11](https://arxiv.org/html/2605.08744v1#A1) 的顶点匹配公式：L2严格小于阈值；生成候选为所有F顶点；不要求一对一。",
        "- 本地边界域为C自身原始顶点ID拓扑中出现一次的边的端点，按源ID去重；没有声明这些边全都必须接合。USER任务接合成功率仍NA。",
        "- 输入C、来源C及RAW前K面逐字节核验；只查RAW后N-K面的顶点，排除C自匹配。未重新应用source_corner_permutations。",
        f"- 固定尺度 L = sqrt(3)*2/0.3762 = {L:.15g}，沿用历史规范尺度；不按输出bbox缩放。raw_tau_1e-6使用模型单位；L_tau使用距离/L。",
        "- 论文主表评分尺度未能从公开文本/代码确认，因此不是论文数值复现，不与论文表格横向排名。附录C单位球缺陷检测不等于主表归一化合同。",
        "- 全部8档阈值在计算任何输出的最近距离前登记。宽容差曲线为敏感性诊断，不能改称原论文严格匹配。",
        "- A-VMR逐输出宏平均；PMR是全部待评边界顶点匹配的输出比例。每父恰好2噪声，ALL与等父平均一致；仅6个重复使用的开发父。",
        "- 顶点匹配可以被重层、孤立局部面或错误连接满足，不认证共享边、水密、无相交或语义采用；本文不覆盖既有I/CF/FF和定性结论。",
        "", "## 全部阈值结果（A-VMR / PMR，%）", "",
        "| Setting | R2_FROZEN | R2_CONT_OLD | R2_CONT_USER | R2_USER_MIX |",
        "|---|---:|---:|---:|---:|"]
    for name,*_ in SETTINGS:
        cells=[]
        for label in LABELS:
            r=next(r for r in summary if r["label"]==label and r["setting"]==name and r["region"]=="ALL")
            cells.append("NA" if r["A_VMR"] is None else f'{100*r["A_VMR"]:.2f} / {100*r["PMR"]:.2f}')
        lines.append("|"+name+"|"+"|".join(cells)+"|")
    lines+=["","## 分区与配对","","完整座面/靠背分区见 summary.csv；逐条件双噪声均值见 per_condition.csv；MIX减三个参照的同请求差值见 paired_noise.csv。",
             "逐输出8档匹配数/分母见 per_output.csv；每个边界顶点的源ID、最近F角点与原始距离见 per_vertex_distances.csv。",
             "condition_audit.csv核对源ID边界与精确坐标边界的位置集合，并保留非流形、重合顶点和边界长度信息。",
             "", "## 执行与恢复", "", "registration.json先冻结公式、阈值、输入hash与代码hash。scores逐输出保存；重复执行只复用身份一致的完成记录，不重跑模型。缺失/异常标BLOCKED，保留其请求身份。",
             "metric_self_checks.json覆盖精确匹配、分离、部分匹配、严格阈值、空输入、自匹配陷阱及聚合分母。",
             "单CPU worker/单BLAS线程；48个小数组的距离查询无需多进程。资源快照在resources_before/after.json。ledger.jsonl记录实际评分，保护文件hash前后核对。",
             "", "命令：wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/tools/avmr_pmr_v1/run.py --stage all", ""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("prepare","all"),default="all")
    args=parser.parse_args()
    if args.stage=="prepare":
        registration,_,_=prepare()
        print(json.dumps({"status":"PREPARED","version":registration["version"],"protected":len(registration["protected"])}))
    else:
        run()
