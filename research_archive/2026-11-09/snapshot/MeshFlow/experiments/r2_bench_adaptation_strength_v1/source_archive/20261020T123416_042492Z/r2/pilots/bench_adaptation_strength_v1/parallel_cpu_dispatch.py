"""Add one registered independent CPU label without changing frozen workers."""
from __future__ import annotations

import argparse
import time

from . import workflow as W
from .common import OUT, read, resources, sha, write_new


def run(label: str, workers: int, pool_limit: int):
    if workers not in range(1, 7) or pool_limit not in (2, 3):
        raise ValueError("Bounded CPU dispatch only: <=6 workers and <=3 pools")
    if label not in W.label_specs():
        raise ValueError("Unregistered label")
    jobs = W.records()
    active = [j for _, j in jobs if j["stage"] in W.CPU_STAGES and W.alive(j)]
    if not active or any(j["stage"] not in ("cpu_raw", "cpu_post") for j in active):
        raise RuntimeError("Require an existing independent scoring pool; no collection race")
    if len(active) >= pool_limit:
        raise RuntimeError("Active pool limit reached")
    if W.ownership("cpu_post", label, jobs) or W.complete("cpu_post", label):
        raise RuntimeError("Label already owned or complete")
    if not W.complete("post", label):
        raise RuntimeError("Complete registered D0 label required")
    if (OUT / "cpu" / label).exists() or (OUT / "base_pair_caches" / label).exists():
        raise RuntimeError("Partial unowned CPU output must not be replayed")
    total_workers = sum(j.get("workers_cap", 1) for j in active) + workers
    snapshot = resources()
    if total_workers > 18 or total_workers > snapshot["CPU_cores"] - 4:
        raise RuntimeError("CPU headroom insufficient")
    if snapshot["CPU_percent"] > 70:
        raise RuntimeError("CPU busy; leave existing workers untouched")
    # Conservative process workspace allowance; neither a measured peak nor
    # permission to add GPU work. Current worker RSS is retained in snapshot.
    if snapshot["RAM_available"] < (16 + 2 * workers) * (1 << 30):
        raise RuntimeError("RAM headroom insufficient")
    if any(W.alive(j) for _, j in jobs if j["stage"] == "train"):
        raise RuntimeError("This supplemental dispatch is post-training only")
    stamp = str(time.time_ns())
    plan = dict(
        schema="BOUNDED_CPU_DISPATCH_EXTENSION_V1", utc=W.utc(),
        target_label=label, workers=workers, total_pool_limit=pool_limit,
        total_worker_cap=total_workers, resources=snapshot,
        active_owned_pools=active, source_sha256=sha(__file__),
        reason="Training is complete; independent registered CPU labels can overlap",
        unchanged="Frozen scorer, output identities, queries, geometric definitions and all neural budgets",
        original_controller_left_running=True, original_workers_left_running=True,
        replay=False, new_samples=0, model_calls=0,
        interpretation="Wall time is concurrent, not an isolated performance benchmark",
    )
    path = OUT / "workflow" / ("cpu_parallel_extension_" + stamp + ".json")
    write_new(path, plan)
    # Original controller sees an active CPU pool and will not race a dispatch.
    # Recheck ownership immediately before spawning this distinct label.
    if W.ownership("cpu_post", label, W.records()):
        raise RuntimeError("Ownership changed before dispatch")
    job = W.spawn_cpu("cpu_post", label, workers, snapshot)
    W.event("RESOURCE_MEASURED_CPU_POOL_EXTENSION", plan=str(path),
            target=label, total_worker_cap=total_workers, job=job)
    return dict(status="DISPATCHED", label=label, pid=job["pid"], plan=str(path))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--pool-limit", type=int, default=2)
    a = p.parse_args()
    print(run(a.label, a.workers, a.pool_limit), flush=True)
