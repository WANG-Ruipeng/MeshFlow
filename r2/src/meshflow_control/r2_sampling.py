"""One explicit R2 sample with its own durable 1-rollout/50-forward ledger."""
import json
import os
from pathlib import Path
import time


class SampleBudget:
    limits = {"rollout_attempt": 1, "sampling_forward": 50}

    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError("Existing sample ledger cannot be reset or silently retried")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        self.counts = {key: 0 for key in self.limits}

    def event(self, event, **fields):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event=event, time_unix=time.time(), **fields),
                                    allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def reserve(self, kind, context=None):
        from .runtime import check_gpu_deadline
        check_gpu_deadline()
        if kind not in self.limits or self.counts[kind] >= self.limits[kind]:
            raise RuntimeError("R2 sample budget exceeded: " + kind)
        self.event("RESERVE", kind=kind, ordinal=self.counts[kind] + 1, context=context or {})
        self.counts[kind] += 1

    def call(self, kind, fn, context=None):
        self.reserve(kind, context)
        try:
            result = fn()
        except BaseException as error:
            self.event("ERROR", kind=kind, context=context or {}, error=repr(error))
            raise
        self.event("RETURN", kind=kind, context=context or {})
        return result


def run(*, checkpoint, condition, condition_key, num_faces, seed, out,
        expected_sha256=None):
    from . import runtime
    from .data.io import atomic_json, digest
    from .mesh_io import load_condition
    from .sampling import sample, validate_C
    from .r2_checkpoints import load_generator
    from .r2_runs import external_path
    out = external_path(out)
    if out.exists():
        raise FileExistsError("Sampling requires a new output directory")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("MT19937 seed must be in [0, 2**32)")
    C = validate_C(load_condition(condition, condition_key), num_faces)
    checkpoint_sha = digest(checkpoint)
    if expected_sha256 is not None and checkpoint_sha != expected_sha256.lower():
        raise ValueError("Checkpoint file SHA256 differs")
    out.mkdir(parents=True, exist_ok=False)
    budget = SampleBudget(out / "audit/calls.jsonl")
    started = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=checkpoint_sha,
                   condition=str(Path(condition).resolve()), condition_sha256=digest(condition),
                   N=num_faces, K=len(C), seed=seed, status="STARTED")
    atomic_json(out / "attempt.json", started)
    try:
        with runtime.gpu_worker(out / "audit/gpu_owner", "r2_sampling"):
            model, identity = load_generator(checkpoint, device="cuda", expected_sha256=checkpoint_sha)
            atomic_json(out / "checkpoint_identity.json", identity)
            _, _, receipt = sample(model, C, num_faces, seed=seed, budget=budget,
                                    out=out, context={"recipe": "R2_MIX_H_ALL"})
        atomic_json(out / "attempt.json", dict(started, status="COMPLETE", counts=budget.counts))
        return receipt
    except BaseException as error:
        atomic_json(out / "failure.json", dict(started, status="FAILED", error=repr(error),
                                               counts=budget.counts))
        raise
