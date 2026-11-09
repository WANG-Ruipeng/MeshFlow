"""Bench multipart data adapter, preserving the Chair owner-only SEAM contract.

This module performs CPU data preparation only.  It does not decide a training
checkpoint, update budget, convergence rule, or evaluation budget.  Open-C
samples delegate literally to the existing Chair/Bench packing functions.
Closed C has no seam slots and is the ordinary Native completion special case.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "r2/src"))
import numpy as np
from meshflow_control.seam import slot_data as legacy
from meshflow_control.training.coupling import nested_ot_map, to_noise_slots
from r2.pilots.seam_abc_v1.slot_data_extension import supervised_sample as owner_sample

DEFAULT_SOURCE = ROOT / "experiments/r2_bench_parts_v1/data/manifest.json"
DEFAULT_OUT = ROOT / "experiments/r2_bench_parts_convergence_v1/data"
SCHEMA = "BENCH_MULTIPART_OWNER_ONLY_DATA_V1"
SEED = 20261018
MAX_SCHEDULE_STEPS = 24000


def resolve(path):
    value = str(path)
    if os.name == "nt":
        value = value.replace("/mnt/e/MeshFlow", "E:/MeshFlow")
    else:
        value = value.replace("E:\\MeshFlow", "/mnt/e/MeshFlow").replace("E:/MeshFlow", "/mnt/e/MeshFlow")
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def sha(path):
    result = hashlib.sha256()
    with resolve(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if read(path) != value:
            raise ValueError("Existing immutable JSON differs: " + str(path))
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_case(row_or_path):
    """Load a consolidated training case; never used as model inference input."""
    path = row_or_path["case_path"] if isinstance(row_or_path, dict) else row_or_path
    return arrays(path)


def _save_npz(path, value):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = arrays(path)
        if set(old) != set(value) or any(not np.array_equal(old[key], value[key]) for key in value):
            raise ValueError("Existing immutable case differs: " + str(path))
        return
    with path.open("xb") as stream:
        np.savez_compressed(stream, **value)


def resources():
    import psutil
    def query(args):
        result = subprocess.run(args, capture_output=True, text=True)
        return dict(returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())
    return dict(unix=time.time(), pid=os.getpid(), cpu_logical=psutil.cpu_count(),
                available_RAM_bytes=psutil.virtual_memory().available,
                cpu_utilization_percent=psutil.cpu_percent(interval=.1), workers=1, numeric_threads=1,
                GPU_computation=False,
                gpu=query(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]),
                gpu_processes=query(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"]))


def boundary_from_C(C_vertices, C_faces):
    """Original validation/order, with the empty-boundary case represented [0,3]."""
    v, f = np.asarray(C_vertices), np.asarray(C_faces)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("Invalid C vertices")
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu":
        raise ValueError("Invalid C indexed triangles")
    if not len(f) or f.min() < 0 or f.max() >= len(v):
        raise ValueError("Invalid C IDs")
    book = collections.defaultdict(list)
    for fi, face in enumerate(f):
        if len(set(map(int, face))) != 3:
            raise ValueError("Degenerate C identities")
        for corner in range(3):
            u, w = int(face[corner]), int(face[(corner + 1) % 3])
            book[tuple(sorted((u, w)))].append((fi, u, w))
    if any(len(incidence) > 2 for incidence in book.values()):
        raise ValueError("Nonmanifold C")
    if any(len(rows) == 2 and rows[0][1:] != rows[1][1:][::-1] for rows in book.values()):
        raise ValueError("C winding must be consistent")
    return np.asarray([rows[0] for _, rows in sorted(book.items()) if len(rows) == 1], np.int64).reshape(-1, 3)


def edge_features(C_vertices, C_faces, total_N):
    edges = boundary_from_C(C_vertices, C_faces)
    n, k = int(total_N), len(C_faces)
    if not 128 <= n <= 256 or not 0 < k < n or len(edges) > n - k:
        raise ValueError("N cannot hold the original boundary-slot layout")
    if len(edges):
        return legacy.edge_features(C_vertices, C_faces, total_N)
    return edges, np.empty((0, legacy.FEATURE_DIM), np.float32)


def inference_layout(C_vertices, C_faces_original, C_faces_model, total_N, occupied):
    """C/N-only layout. Empty boundary creates no head call and no seam slots."""
    v = np.asarray(C_vertices, np.float32)
    original, canonical = np.asarray(C_faces_original, np.int64), np.asarray(C_faces_model, np.int64)
    if original.shape != canonical.shape or not np.array_equal(np.sort(original, 1), np.sort(canonical, 1)):
        raise ValueError("Model C changes original faces")
    edges, features = edge_features(v, original, total_N)
    occupancy = np.asarray(occupied)
    if occupancy.dtype != np.bool_ or occupancy.shape != (len(edges),):
        raise ValueError("One explicit boolean per C open edge is required")
    if len(edges):
        return legacy.inference_layout(v, original, canonical, total_N, occupancy)
    n, k = int(total_N), len(original)
    anchors = np.full((n, 3), -1, np.int64)
    anchors[:k] = canonical
    known = np.arange(n) < k
    return dict(anchor_ids=anchors, known_mask=known, valid_mask=np.ones(n, bool),
                free_corner_mask=anchors < 0, occupied=occupancy.copy(), edge_records=edges,
                features=features, free_coordinate_count=9 * (n - k),
                seam_slots=np.empty(0, np.int64), bulk_slots=np.arange(k, n, dtype=np.int64),
                C_vertices=v, C_faces_original=original, C_faces_model=canonical, N=n, K=k)


def _closed_sample(case, raw_noise, epsilon, t, permutation_seed, arm):
    """Identical bulk OT/HYBRID algebra as the historical ordinary packing."""
    full, pre = np.asarray(case["full_target"], np.float32), np.asarray(case["pre_ot_full"], np.float32)
    known_ids, free_ids = np.asarray(case["known_ids"], np.int64), np.asarray(case["free_ids"], np.int64)
    n, k = len(full), len(known_ids)
    if not np.array_equal(pre * np.float32(2), full):
        raise ValueError("Pre-OT half-scale differs")
    z, eps = np.asarray(raw_noise), np.asarray(epsilon)
    if z.shape != eps.shape or z.shape != (n-k, 3, 3) or not np.isfinite(z).all() or not np.isfinite(eps).all():
        raise ValueError("Invalid paired noise arrays")
    if not 0 <= float(t) <= 1:
        raise ValueError("Invalid time")
    labels = np.asarray(case["occupied_labels"], bool)
    if labels.shape != (0,):
        raise ValueError("Closed C requires empty source labels")
    layout = inference_layout(case["C_vertices"], case["C_faces_original"], case["C_faces_model"], n, labels)
    anchor = layout["anchor_ids"]
    target = np.zeros((n, 3, 3), np.float32)
    ids, corner = np.full(n, -1, np.int64), np.full((n, 3), -1, np.int64)
    target[:k], ids[:k], corner[:k] = full[known_ids], known_ids, np.arange(3)
    slots = layout["bulk_slots"]
    rng = np.random.RandomState(int(permutation_seed))
    bulk_faces = free_ids[rng.permutation(len(free_ids))]
    cp0 = np.asarray([[0,1,2], [1,2,0], [2,0,1]], np.int64)[rng.randint(0,3,len(bulk_faces))]
    face_map, corner_map = nested_ot_map(pre[bulk_faces[:,None], cp0], z[slots-k])
    ids[slots] = bulk_faces[np.argsort(face_map)]
    corner[slots] = to_noise_slots(cp0, face_map, corner_map)
    target[slots] = to_noise_slots(full[bulk_faces[:,None],cp0], face_map, corner_map)
    mixed = np.float32(np.sqrt(.75)) * z.astype(np.float32) + np.float32(.5) * eps.astype(np.float32)
    x0 = np.zeros_like(target); x0[k:] = mixed
    fixed = anchor >= 0
    x0[fixed] = layout["C_vertices"][anchor[fixed]]
    if not np.array_equal(target[fixed], x0[fixed]):
        raise ValueError("Known C changed")
    velocity = target - x0
    xt = (np.float32(1)-np.float32(t))*x0 + np.float32(t)*target
    xt[fixed] = x0[fixed]
    order = np.random.RandomState(int(permutation_seed) ^ 0xD00D).permutation(n)
    return dict(x0=x0[order], x1=target[order], xt=xt[order], u=velocity[order],
                anchor_ids=anchor[order], known_mask=layout["known_mask"][order],
                valid_mask=layout["valid_mask"][order], free_corner_mask=layout["free_corner_mask"][order],
                source_face_ids=ids[order], source_corner_permutations=corner[order],
                face_permutation=order, ot_face_map=face_map, ot_corner_map=corner_map,
                t=np.asarray(t,np.float32), C_vertices=layout["C_vertices"],
                free_coordinate_count=np.asarray(layout["free_coordinate_count"]),
                occupied_labels=labels, occupied_for_FM=labels, edge_features=layout["features"],
                N=np.asarray(n), K=np.asarray(k), arm=np.asarray("SEAM_SLOT_C200" if arm=="SEAM" else "FULL4_C200"),
                production_OT_calls=np.asarray(1))


def supervised_sample(case, raw_noise, epsilon, t, permutation_seed, arm):
    if arm not in ("PLAIN", "SEAM"):
        raise ValueError("Expected PLAIN or SEAM")
    if len(case["occupied_labels"]) == 0:
        return _closed_sample(case, raw_noise, epsilon, t, permutation_seed, arm)
    if arm == "SEAM":
        return owner_sample(case, raw_noise, epsilon, t, permutation_seed)
    return legacy.supervised_sample(case, raw_noise, epsilon, t, permutation_seed, "FULL4_C200")


def sample_for_training(case, row, arm):
    """Same alpha/noise/time policy as the previous Bench and Chair trainer."""
    copied = {key: value.copy() for key, value in case.items()}
    alpha = np.float32(row["alpha"])
    for key in ("full_target", "pre_ot_full", "C_vertices"):
        copied[key][..., 0] *= alpha
    n, k = len(copied["full_target"]), len(copied["known_ids"])
    z = np.random.RandomState(row["noise_seed"]).standard_normal((n-k,3,3)).astype(np.float32)
    eps = np.random.RandomState(row["epsilon_seed"]).standard_normal(z.shape).astype(np.float32)
    normal = float(np.random.RandomState(row["time_seed"]).standard_normal())
    t = float(1/(1+np.exp(-normal)))
    return supervised_sample(copied, z, eps, t, row["permutation_seed"], arm)


def _make_schedule(train, steps):
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for index, row in enumerate(train):
        groups[row["parent_id"]][row["part_type"]].append(index)
    parents = sorted(groups)
    if len(parents) != 64 or len(parents) % 4:
        raise ValueError("Frozen source split is not the 64-parent Bench pool")
    parent_rng = np.random.RandomState(SEED)
    aug_rng = np.random.RandomState(SEED ^ 0xABCDEF)
    parts = {p: sorted(groups[p]) for p in parents}
    cursors = collections.Counter(); variants = collections.Counter()
    queue, schedule = [], []
    for step in range(1, steps+1):
        if not queue:
            queue = parent_rng.permutation(len(parents)).tolist()
        chosen, queue = queue[:4], queue[4:]
        rows=[]
        for j, pi in enumerate(chosen):
            parent=parents[pi]
            part=parts[parent][cursors[parent] % len(parts[parent])]; cursors[parent]+=1
            options=groups[parent][part]
            index=options[variants[(parent,part)] % len(options)]; variants[(parent,part)]+=1
            r=train[index]
            for aug in range(2):
                seed=10000000+step*1000+j*10+aug
                rows.append(dict(index=index, parent_id=parent, condition_id=r["condition_id"], part_type=part,
                                 alpha=1. if aug==0 else float(aug_rng.uniform(.9,1.1)),
                                 noise_seed=seed, epsilon_seed=seed+100000000,
                                 time_seed=seed+200000000, permutation_seed=seed+300000000))
        schedule.append(dict(step=step, samples=rows))
    return dict(schema="BENCH_PARENT_PART_VARIANT_STREAM_V1", max_prepared_steps=steps,
                training_budget_authorized_by_this_file=False, seed=SEED, batch=8, distinct_parents_per_update=4,
                policy="parent shuffled cycles; per-parent part sorted round robin; per-part variant sorted round robin; two paired width augmentations",
                record_index_domain="manifest.train", steps=schedule)


def prepare(source=DEFAULT_SOURCE, out=DEFAULT_OUT):
    source,out=resolve(source),resolve(out)
    if (out/"manifest.json").exists():
        old=read(out/"manifest.json")
        if sha(source)!=old["source_manifest_sha256"]:
            raise ValueError("Source manifest changed")
        for row in old["rows"]:
            if sha(row["case_path"])!=row["case_sha256"]:
                raise ValueError("Consolidated case changed")
        if sha(out/"input_schedule.json")!=old["schedule_sha256"]:
            raise ValueError("Frozen schedule changed")
        return old
    out.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    write_new(out/"resources_before.json",resources())
    original=read(source)
    rows=sorted(original["rows"],key=lambda r:(r["group"]!="TRAIN",r["parent_id"],r["part_type"],r["condition_id"]))
    if len(rows)!=316 or sum(r["group"]=="TRAIN" for r in rows)!=277:
        raise ValueError("Expected all 316 frozen multipart conditions")
    sources={}; protected={str(source):sha(source)}; records=[]
    for row in rows:
        for key,hkey in (("source_case_path","source_case_sha256"),("paired_path","paired_sha256"),("supervision_file","supervision_sha256"),("input_path","input_sha256")):
            path=resolve(row[key]);actual=protected.setdefault(str(path),sha(path))
            if actual!=row[hkey]:raise ValueError("Changed protected source: "+str(path))
        sp=str(resolve(row["source_case_path"]))
        if sp not in sources:sources[sp]=arrays(sp)
        full=sources[sp];pair=arrays(row["paired_path"]);sup=arrays(row["supervision_file"])
        for key in ("known_ids","free_ids","C_vertices","C_faces_original","C_faces_model","C_source_vertex_ids","C_source_corner_permutations","N","K"):
            if not np.array_equal(pair[key],sup[key]):raise ValueError("Pair/supervision differ: "+key)
        case={key:full[key].copy() for key in ("full_target","pre_ot_full","source_face_vertex_ids")}
        case.update({key:value.copy() for key,value in pair.items()})
        case.update({key:value.copy() for key,value in sup.items()})
        known,free=case["known_ids"],case["free_ids"]
        if not np.array_equal(np.sort(np.r_[known,free]),np.arange(len(case["full_target"]))):raise ValueError("Source partition differs")
        if not np.array_equal(case["full_target"][known],case["C"]) or not np.array_equal(case["full_target"][free],case["Y"]):raise ValueError("C/Y source identity differs")
        edges,features=edge_features(case["C_vertices"],case["C_faces_original"],int(case["N"]))
        if not np.array_equal(edges,case["edge_records"]):raise ValueError("Source label edge order differs")
        case["edge_features"]=features
        path=out/"cases"/(row["condition_id"]+".npz");_save_npz(path,case)
        records.append(dict(row,case_path=str(path),case_sha256=sha(path),
                            original_candidate_case_path=row["case_path"],original_candidate_case_sha256=row["case_sha256"],
                            adapter_semantics="CLOSED_NATIVE_NO_SEAM" if not len(edges) else "CHAIR_OWNER_ONLY_EXTENSION_UNCHANGED"))
    train=[r for r in records if r["group"]=="TRAIN"];dev=[r for r in records if r["group"]=="DEV"]
    assert not {r["parent_id"] for r in train}&{r["parent_id"] for r in dev}
    schedule=_make_schedule(train,MAX_SCHEDULE_STEPS)
    write_new(out/"input_schedule.json",schedule)
    code_paths=[Path(__file__),ROOT/"r2/pilots/seam_abc_v1/slot_data_extension.py",ROOT/"r2/src/meshflow_control/seam/slot_data.py",ROOT/"r2/src/meshflow_control/seam/model_adapter.py",ROOT/"r2/src/meshflow_control/training/coupling.py"]
    result=dict(schema=SCHEMA,source_manifest=str(source),source_manifest_sha256=sha(source),rows=records,train=train,dev=dev,
                summary=dict(conditions=len(records),train_conditions=len(train),dev_conditions=len(dev),train_parents=64,dev_parents=8,
                             closed_train=sum(r["boundary_edges"]==0 for r in train),closed_dev=sum(r["boundary_edges"]==0 for r in dev)),
                schedule_path=str(out/"input_schedule.json"),schedule_sha256=sha(out/"input_schedule.json"),
                protected_inputs=protected,code_provenance={str(p):sha(p) for p in code_paths},
                scientific_changes="Only multipart C selection and empty-boundary admission; owner-only packing, FM, OT and HYBRID unchanged",
                no_inference_source_labels=True,no_part_name_model_input=True,training_budget_defined=False)
    write_new(out/"manifest.json",result)
    for path,expected in protected.items():
        if sha(path)!=expected:raise ValueError("Protected source changed during preparation")
    write_new(out/"preparation_receipt.json",dict(status="COMPLETE",wall_seconds=time.perf_counter()-started,
                conditions=len(records),training_updates=0,model_forwards=0,GPU_jobs=0,OT_calls=0,
                random_array_generation="schedule scalar seeds/width only; training Gaussian arrays generated by actual requested update",
                protected_input_hashes_unchanged=True,resources_after=resources()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",default=str(DEFAULT_SOURCE));parser.add_argument("--out",default=str(DEFAULT_OUT))
    args=parser.parse_args();result=prepare(args.source,args.out)
    print(json.dumps(result["summary"],ensure_ascii=False,indent=2))


if __name__=="__main__":main()
