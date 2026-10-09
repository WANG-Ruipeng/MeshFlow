"""One-time explicit import of frozen repair_v2 assets; never imports old code."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import os
import shutil
import subprocess
import time
import numpy as np
from .io import digest, array_hash, read_json, freeze_json, save_npz, bits

_SOURCE_CODES = (
    "experiments/mf_h1_local0/chair_hybrid_coupling_v1/stream.py",
    "experiments/mf_h1_local0/chair_hybrid_coupling_v1/coupling.py",
    "experiments/mf_h1_local0/chair_hybrid_coupling_v1/training.py",
    "experiments/mf_h1_local0/chair_domain_control_v1/prepare.py",
    "native_t1/runs/control_mvp48_v1/data/load_sample.py",
    "utils/ot_utils.py",
    "native_t1/training.py",
)

def resources():
    r = {"time_unix": time.time(), "cpu_cores": os.cpu_count(), "cpu_threads_for_prepare": 2,
         "model_calls": 0, "cuda_initialized": False}
    for key, cmd in (
        ("gpu", ["nvidia-smi","--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu","--format=csv"]),
        ("gpu_processes", ["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv"]),
        ("memory", ["free","-b"]),
        ("cpu_processes", ["ps","-eo","pid,pcpu,pmem,args","--sort=-pcpu"]),
    ):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            r[key] = {"exit_code":p.returncode, "stdout":"\n".join(p.stdout.splitlines()[:14]), "stderr":p.stderr}
        except (OSError, subprocess.TimeoutExpired) as e:
            r[key] = {"unavailable":str(e)}
    return r

def _validate_parent(arrays, n):
    target, pre = arrays["full_target"], arrays["pre_ot_full"]
    if target.shape != (n, 9) or target.dtype != np.float32 or pre.dtype != np.float32:
        raise ValueError("Frozen parent coordinates/dtype differ")
    if not np.isfinite(target).all() or not bits(pre * np.float32(2), target.reshape(n,3,3)):
        raise ValueError("Original pre-OT x2 identity differs")
    ids = arrays["source_face_vertex_ids"]
    if not bits(arrays["model_vertices"][ids], target.reshape(n,3,3)):
        raise ValueError("Canonical source corner mapping differs")
    retained, inverse = arrays["retained_to_original_face"], arrays["original_to_retained_face"]
    expected = np.full(len(arrays["original_source_faces"]), -1, np.int64)
    expected[retained] = np.arange(n)
    if not np.array_equal(inverse, expected) or not bits(arrays["original_source_faces"][retained], arrays["source_faces"]):
        raise ValueError("Frozen repair_v2 retained mapping differs")
    cp = arrays["source_corner_permutations"]
    if not np.array_equal(np.sort(cp, axis=1), np.tile(np.arange(3), (n,1))):
        raise ValueError("Invalid canonical source corner permutations")
    if not np.array_equal(np.take_along_axis(arrays["source_faces"],cp,axis=1), ids):
        raise ValueError("Source corner indices do not reconstruct original face")

def _validate_task(row, arrays):
    n, k = row["N"], row["K"]
    known, free = [np.asarray(row[key],np.int64) for key in ("source_face_ids","free_source_ids")]
    if not 128 <= n <= 256 or not 0 < k < n or k != int(np.floor(row["ratio"]*n+.5)):
        raise ValueError("Task N/K/ratio differs")
    if len(known) != k or not np.array_equal(np.sort(np.r_[known,free]),np.arange(n)):
        raise ValueError("Task source-face partition differs")
    if not np.array_equal(free,np.setdiff1d(np.arange(n),known)):
        raise ValueError("Task free IDs are not ascending complement")
    # Validate whole-edge connected C without selecting or repairing new geometry.
    incidence = defaultdict(list)
    for i, face in enumerate(arrays["source_faces"]):
        for a,b in ((0,1),(1,2),(2,0)):
            incidence[tuple(sorted((int(face[a]),int(face[b]))))].append(i)
    adjacency = defaultdict(set); ks=set(known.tolist())
    for faceids in incidence.values():
        inside=ks.intersection(faceids)
        for i in inside: adjacency[i].update(inside-{i})
    todo=[int(known[0])]; reached=set()
    while todo:
        i=todo.pop()
        if i not in reached: reached.add(i); todo.extend(adjacency[i]-reached)
    if reached!=ks: raise ValueError("Frozen C lost complete-edge connectivity")
    return known,free

def import_legacy_data(source_mvp, source_chair, out, *, source_repo=None):
    source_mvp, source_chair, out = map(lambda p:Path(p).resolve(), (source_mvp,source_chair,out))
    repo = Path(source_repo).resolve() if source_repo else source_mvp.parents[2]
    if out == repo or repo in out.parents or out == source_mvp or out == source_chair:
        raise ValueError("New data must be outside the protected source repository")
    out.mkdir(parents=True,exist_ok=True)
    before_path=out/"import_resources_before.json"
    if not before_path.exists(): freeze_json(before_path,resources())
    started=time.perf_counter()
    task_path=source_chair/"task_manifest.json"
    binding_path=source_mvp/"registration/train_data_binding.json"
    ready_path=source_mvp/"DATA_READY.json"
    binding,legacy=read_json(binding_path),read_json(task_path)
    if legacy["data_revision"]!="repair_v2" or digest(ready_path)!=legacy["source_data_ready_sha256"]:
        raise ValueError("Original repaired data identity differs")
    if digest(ready_path)!=binding["data_ready_sha256"]:
        raise ValueError("Binding and data-ready identities disagree")
    parents=sorted(legacy["splits"]["train"])
    if len(parents)!=32 or parents!=sorted(binding["train_object_ids"]):
        raise ValueError("Original training split differs")
    tasks=[r for r in legacy["tasks"] if r["role"]=="train" and r["status"]=="READY"]
    if len(tasks)!=453 or set(r["uid"] for r in tasks)!=set(parents):
        raise ValueError("Expected exact original 453 READY training tasks")
    source_files={str(p):digest(p) for p in (task_path,binding_path,ready_path)}
    parent_rows=[]; arrays_by_uid={}
    for uid in parents:
        src=source_mvp/"data/meshes"/(uid+".npz")
        sha=digest(src)
        if sha!=binding["files"]["data/meshes/"+uid+".npz"]:
            raise ValueError("Frozen parent checksum differs: "+uid)
        source_files[str(src)]=sha
        with np.load(src,allow_pickle=False) as z:
            arrays={key:z[key].copy() for key in z.files}
        n=len(arrays["full_target"]); _validate_parent(arrays,n)
        dest=out/"parents"/(uid+".npz"); dest.parent.mkdir(exist_ok=True)
        if not dest.exists(): shutil.copyfile(src,dest)
        if digest(dest)!=sha: raise ValueError("Imported parent bytes differ")
        arrays_by_uid[uid]=arrays
        parent_rows.append({"uid":uid,"N":n,"npz":str(dest.relative_to(out)),"sha256":sha,
            "full_target_sha256":array_hash(arrays["full_target"]),
            "pre_ot_full_sha256":array_hash(arrays["pre_ot_full"]),
            "coordinate_transform":str(arrays["transform_json"].item())})
    task_rows=[]
    for row in tasks:
        a=arrays_by_uid[row["uid"]]; known,free=_validate_task(row,a)
        C=a["full_target"][known].reshape(-1,3,3).copy()
        dest=out/"tasks"/(row["task_id"]+".npz")
        values={"C":C,"N":np.asarray(row["N"],np.int64),"source_face_ids":known,"free_source_ids":free}
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                if set(z.files)!=set(values) or any(not bits(z[k],v) for k,v in values.items()):
                    raise ValueError("Existing task data differs")
        else: save_npz(dest,**values)
        copy=dict(row)
        copy.update(condition_npz=str(dest.relative_to(out)),condition_sha256=digest(dest),C_sha256=array_hash(C))
        task_rows.append(copy)
    for relative in _SOURCE_CODES:
        p=repo/relative
        if not p.exists(): raise FileNotFoundError("Necessary historical source unavailable: "+str(p))
        source_files[str(p)]=digest(p)
    manifest={"schema":"meshflow_control_training_data_v1","data_revision":"repair_v2",
        "parents":parent_rows,"tasks":task_rows,"splits":{"train":parents},
        "N_range":[128,256],"context_distribution":"uniform available ratio, then uniform original task within ratio",
        "ratio_task_counts":dict(Counter(str(r["ratio"]) for r in tasks)),
        "source_manifest_sha256":digest(task_path),"source_binding_sha256":digest(binding_path),
        "coordinate_contract":"Saved FP32 full_target and original pre_ot_full*2; alpha scales X only, no renormalization",
        "model_input_allowlist":["xt","t","y","valid_mask","known_mask"],
        "supervision_only":["parents","tasks.*.source_face_ids","tasks.*.free_source_ids","tasks.*.interface"]}
    manifest_sha=freeze_json(out/"train_manifest.json",manifest)
    for p,h in source_files.items():
        if digest(p)!=h: raise ValueError("Protected source changed while importing: "+p)
    source_doc={"schema":"meshflow_control_data_source_identity_v1","files":source_files,
        "migration":"Direct copied parent NPZ; original READY C partitions and FP32 coordinates retained; no legacy code execution"}
    freeze_json(out/"source_identity.json",source_doc)
    receipt={"status":"PASS","train_parents":32,"ready_training_tasks":453,
        "ratio_task_counts":manifest["ratio_task_counts"],"manifest_sha256":manifest_sha,
        "source_files_read":len(source_files),"source_bytes":sum(Path(p).stat().st_size for p in source_files),
        "data_bytes":sum(p.stat().st_size for p in out.rglob("*.npz")),
        "model_forward":0,"model_backward":0,"optimizer_updates":0,"OT_calls":0,
        "source_before_after_equal":True,"legacy_runtime_imports":0,"wall_seconds":time.perf_counter()-started}
    from .io import atomic_json
    atomic_json(out/"import_receipt.json",receipt)
    return receipt

def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-mvp",required=True); p.add_argument("--source-chair",required=True)
    p.add_argument("--out",required=True); p.add_argument("--source-repo")
    args=p.parse_args(argv)
    import json
    print(json.dumps(import_legacy_data(args.source_mvp,args.source_chair,args.out,source_repo=args.source_repo),indent=2))

if __name__=="__main__": main()
