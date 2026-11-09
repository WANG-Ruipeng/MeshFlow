"""Exactly one official Bench D0, gated by all 224 completed RAW outputs."""
from __future__ import annotations
import argparse, json, types
from .eval_plan import OUT, LABELS, all_raw_barrier, label_spec, requests


def run(label, estimated_peak_mib=3500):
    spec = label_spec(label)
    barrier = all_raw_barrier()
    qs = requests(label)
    from r2.pilots.bench_parts_convergence_v1 import postprocess as legacy
    # A private globals namespace adapts the heterogeneous 16/48 panel sizes.
    # The original single-D0 implementation and all geometry contracts stay exact.
    namespace = dict(vars(legacy))
    def checked_barrier(root, barrier_labels, subset):
        if root.resolve() != OUT.resolve() or tuple(barrier_labels) != LABELS or subset != spec["subset"]:
            raise ValueError("Unexpected D0 root, labels or panel")
        return qs, barrier
    namespace["verify_raw_barrier"] = checked_barrier
    fn = types.FunctionType(legacy.run.__code__, namespace, legacy.run.__name__,
                            legacy.run.__defaults__, legacy.run.__closure__)
    fn.__kwdefaults__ = legacy.run.__kwdefaults__
    return fn(OUT, label, list(LABELS), spec["subset"], legacy.OLD.D0, estimated_peak_mib)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", choices=LABELS, required=True)
    p.add_argument("--estimated-peak-mib", type=float, default=3500)
    a = p.parse_args()
    print(json.dumps(run(a.label, a.estimated_peak_mib)), flush=True)
