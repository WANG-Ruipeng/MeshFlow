"""Frozen original 179-field scoring and identical CPU postprocess cascade."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse, collections, concurrent.futures, csv, multiprocessing, resource, sys, time
from pathlib import Path
import numpy as np
from .eval_runtime import OUT, P, LABELS, requests, read, sha, verify
REFERENCE_LABEL = None
REFERENCE_ROOT = OUT
from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
from r2.pilots.seam_abc_v1 import eval_common as C
from r2.pilots.chair_back_transfer_v1.evaluate import worker_count

STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")


def prepare():
    from r2.pilots.chair_back_transfer_v1.evaluate import source_hashes
    paths = [Path(__file__), Path(__file__).with_name("eval_runtime.py"),
             Path(CPU.__file__), Path(CPU.H.__file__), Path(CPU.alpha.__file__),
             Path(CPU.holes.__file__), Path(CPU.M.__file__)]
    code = {str(p): sha(p) for p in paths}
    code.update(source_hashes())
    value = dict(status="FROZEN", labels=list(LABELS), stages=list(STAGES),
                 outputs={label: len(requests(label)) for label in LABELS},
                 eval_manifest_sha256=sha(OUT / "eval_manifest.json"),
                 registration_sha256=sha(OUT / "registration.json"),
                 source_hashes=code, model_calls=0, new_queries=0,
                 chain="BASE_POST -> HINGE_OPP015 -> ALPHA -> HOLE",
                 metric_contract="Unchanged original fields fields and original fixed queries; Native representation and closed C seam fields NA",
                 source_GT_success=False, visual_review="SEPARATE_ACTUAL_AI_PROXY")
    P.immutable(OUT / "evaluation/score_registration.json", value)
    return value


def producer(label, q, kind):
    root = OUT / "evaluation" / label / q["request_id"] / kind
    r = read(root / "receipt.json")
    verify(r["files"])
    if r["status"] != "COMPLETE" or r["request_id"] != q["request_id"] or r["label"] != label:
        raise ValueError("Wrong producer")
    if r[kind + "_sha256"] != sha(root / (kind + ".npz")):
        raise ValueError("Mesh hash differs")
    if kind == "raw" and r["generator_forwards"] != 50:
        raise ValueError("Wrong RAW call budget")
    if kind == "post" and (r["D0_forwards"] != 1 or r["GEO_calls"] != 1):
        raise ValueError("Wrong D0/GEO call budget")
    return root


def raw_one(label, q, regsha):
    rawdir = producer(label, q, "raw")
    dest = OUT / "evaluation/scores" / label / q["request_id"] / "RAW"
    identity = dict(label=label, request_id=q["request_id"], stage="RAW",
                    registration_sha256=regsha, mesh_sha256=sha(rawdir / "raw.npz"))
    done = dest / "audit_receipt.json"
    if done.exists():
        r = read(done)
        if r["identity"] != identity:
            raise ValueError("RAW score identity changed")
        verify(r["files"])
        return r
    if dest.exists():
        raise RuntimeError("Partial RAW score retained: " + str(dest))
    tick = time.perf_counter()
    mesh = C.arrays(rawdir / "raw.npz")
    mesh.setdefault("index_domain", np.asarray("ACTUAL_RAW_C_SEAM_SHARED_INDEX"))
    CPU.score_current(q, mesh, dest, "RAW", label == LABELS[2])
    r = dict(status="PASS", identity=identity, seconds=time.perf_counter() - tick,
             model_calls=0, new_queries=0,
             peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             files={str(p): sha(p) for p in dest.iterdir() if p.is_file()})
    P.immutable(done, r)
    return r


def post_one(label, q, regsha):
    rawdir = producer(label, q, "raw")
    postdir = producer(label, q, "post")
    cache = OUT / "base_pair_caches" / label / q["request_id"]
    pairpath = cache / "pairs.json"
    done = cache / "complete.json"
    if done.exists():
        r = read(done)
        if r["mesh_sha256"] != sha(postdir / "post.npz"):
            raise ValueError("BASE pair cache changed")
        verify(r["files"])
    else:
        if cache.exists():
            raise RuntimeError("Partial BASE pair scan retained: " + str(cache))
        P.immutable(cache / "started.json", dict(mesh_sha256=sha(postdir / "post.npz"), model_calls=0))
        a = C.arrays(postdir / "post.npz")
        pairs = CPU.H.scan(a["output"], int(q["K"]))
        P.immutable(pairpath, pairs)
        P.immutable(done, dict(status="COMPLETE", mesh_sha256=sha(postdir / "post.npz"),
                    model_calls=0, full_pair_scans=1,
                    files={str(p): sha(p) for p in cache.iterdir() if p.is_file()}))
    return CPU.process_one(q, label, postdir / "post.npz", postdir / "pre_geo.npz",
                rawdir / "raw.npz", pairpath, OUT / "cpu" / label / q["request_id"], label == LABELS[2], regsha)


def one(job):
    label, q, mode, regsha = job
    r = raw_one(label, q, regsha) if mode == "raw" else post_one(label, q, regsha)
    if "torch" in sys.modules:
        raise RuntimeError("CPU scoring imported neural framework")
    print("EVALUATED", mode, label, q["request_id"], flush=True)
    return r


def run(labels, mode, workers=2):
    if not labels or any(label not in LABELS for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Unique registered labels required")
    if mode not in ("raw", "post"):
        raise ValueError("Unknown CPU stage")
    reg = prepare()
    key = mode + "__" + "__".join(labels)
    done = OUT / "evaluation/completions" / (key + ".json")
    if done.exists():
        r = read(done)
        verify(r["files"])
        return r
    jobs = [(label, q, mode, sha(OUT / "evaluation/score_registration.json"))
            for label in labels for q in requests(label)
            if not (label == REFERENCE_LABEL and q['baseline_reuse'])]
    for label, q, _, _ in jobs:
        producer(label, q, "raw")
        if mode == "post":
            producer(label, q, "post")
    tick = time.perf_counter()
    snap = C.resources()
    first = one(jobs[0])
    count = worker_count(C.resources(), first, workers)
    P.immutable(OUT / "evaluation/dispatch" / (key + ".json"),
                dict(workers=count, threads_per_worker=1, resources_before=snap,
                     resources_after_first=C.resources(), first_peak_RSS_KiB=first["peak_RSS_KiB"]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn")) as pool:
        receipts = [first, *list(pool.map(one, jobs[1:]))]
    verify(reg["source_hashes"])
    def completion(label, q):
        if mode == "raw":
            return OUT / "evaluation/scores" / label / q["request_id"] / "RAW/audit_receipt.json"
        return OUT / "cpu" / label / q["request_id"] / "complete.json"
    r = dict(status="COMPLETE", mode=mode, labels=labels, cells=len(receipts), workers=count,
             seconds=time.perf_counter() - tick, model_calls=0, new_queries=0,
             files={str(completion(label, q)): sha(completion(label, q)) for label, q, _, _ in jobs})
    P.immutable(done, r)
    return r


def score_dir(label, q, stage):
    root = REFERENCE_ROOT if label == REFERENCE_LABEL and q['baseline_reuse'] else OUT
    if stage == "RAW":
        return root / "evaluation/scores" / label / q["request_id"] / "RAW"
    return root / "cpu" / label / q["request_id"] / stage / "scores"


def collect():
    """Use the lossless collector; historical A scores are read-only."""
    from .collect import run as collect_results
    return collect_results()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "raw", "post", "collect"))
    p.add_argument("--labels", nargs="+", choices=LABELS, default=list(LABELS))
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    r = prepare() if a.action == "prepare" else collect() if a.action == "collect" else run(a.labels, a.action, a.workers)
    print({k: v for k, v in r.items() if k not in ("files", "source_rows", "source_hashes")}, flush=True)
