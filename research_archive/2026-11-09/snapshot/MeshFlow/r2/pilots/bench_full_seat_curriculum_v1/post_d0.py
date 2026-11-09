"""One official D0 per new RAW, after the complete paired RAW barrier.

All 72 RAW products must be saved before the first D0 call. All remaining geometry is the unchanged official-D0/GEO_C/source
winding chain, without RBF, denoiser iteration or threshold changes.
"""
from __future__ import annotations
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse, json, time, types
from pathlib import Path
from .eval_runtime import (
    OUT, P, LABELS, all_raw_barrier, label_spec, requests, new_requests,
    reuse_spec, copy_reference, sha, verify,
)


def run(label, estimated_peak_mib=3500):
    spec = label_spec(label)
    barrier = all_raw_barrier()
    qs = requests(label)
    for q in qs:
        if reuse_spec(q, label):
            copy_reference(q, "post")
    from r2.pilots.bench_parts_convergence_v1 import postprocess as legacy
    from r2.pilots.seam_slot_c200_v1 import postprocess as geometry
    code = {str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name("eval_runtime.py"), Path(__file__).with_name("panel.py"),
                                   Path(legacy.__file__), Path(geometry.__file__), Path(legacy.OLD.__file__))}
    registration = dict(status="FROZEN", label=label, outputs=24,
        new_D0_budget=len(new_requests(label)), new_GEO_budget=len(new_requests(label)),
        reused_POST=24-len(new_requests(label)), code=code,
        D0_checkpoint=str(legacy.OLD.D0), D0_checkpoint_sha256=legacy.OLD.D0_SHA,
        RAW_barrier={str(p): sha(p) for p in barrier},
        contract="Single official Bench D0, restore C and seam anchors, original GEO_C, source-C winding plus fixed-C free orientation")
    P.immutable(OUT / "evaluation/inference_registrations" / (label + "__D0.json"), registration)
    # The mature implementation keeps exact checkpoint, per-forward ledger,
    # resume and geometry contracts. Only its fixed-size barrier is adapted.
    namespace = dict(vars(legacy))
    def checked_barrier(root, barrier_labels, subset):
        if root.resolve() != OUT.resolve() or tuple(barrier_labels) != LABELS or subset != spec["subset"]:
            raise ValueError("Unexpected D0 root, labels or panel")
        return qs, barrier
    namespace["verify_raw_barrier"] = checked_barrier
    fn = types.FunctionType(legacy.run.__code__, namespace, legacy.run.__name__,
                            legacy.run.__defaults__, legacy.run.__closure__)
    fn.__kwdefaults__ = legacy.run.__kwdefaults__
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    attempt = OUT / "evaluation" / label / ("D0_resource_observer_" + str(time.time_ns()))
    attempt.mkdir(parents=True)
    observer = GPUObserver("full_seat_curriculum_single_D0_requests", interval=1.)
    observer.__enter__()
    try:
        result = fn(OUT, label, list(LABELS), spec["subset"], legacy.OLD.D0, estimated_peak_mib)
        if result["new_D0_forwards"] > len(new_requests(label)):
            raise ValueError("D0 budget exceeded")
        verify(code)
        return result
    finally:
        observer.__exit__(None, None, None)
        P.immutable(attempt / "resource_observation.json", dict(summary=observer.summary(), samples=observer.samples,
            scope="This invocation's actual registered D0 and GEO phase; no extra probe/model calls"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", choices=LABELS, required=True)
    p.add_argument("--estimated-peak-mib", type=float, default=3500)
    a = p.parse_args()
    print(json.dumps(run(a.label, a.estimated_peak_mib)), flush=True)
