"""One authorized 16-update round; no dispatch, automatic retry, or new data."""
from __future__ import annotations

from . import common as W  # Environment and CPU thread policy before torch.
from . import data as D
from pathlib import Path
from collections import Counter
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, str(D.ROOT / "r2/src"))

ROUNDS = 16
UPDATES_PER_ROUND = 16
MICROBATCHES_PER_UPDATE = 8
PASSES = 4
ENDPOINTS = 32


def prepare_stream(manifest, round_index, stream_path):
    """CPU-only verification and caching before any CUDA/model work."""
    payload = D.read(stream_path)
    rows = payload if isinstance(payload, list) else payload["records"]
    if len(rows) != PASSES * ENDPOINTS:
        raise ValueError("Each round must have exactly four passes over 32 endpoints")
    requests = {q["request_id"]: q for q in manifest["training_requests"]
                if q["round_index"] == round_index}
    if len(requests) != ENDPOINTS:
        raise ValueError("Frozen round has other than 32 requests")
    expected = set(requests)
    for i in range(PASSES):
        group = rows[i * ENDPOINTS:(i + 1) * ENDPOINTS]
        if set(r["request_id"] for r in group) != expected:
            raise ValueError("Each frozen pass must contain each endpoint exactly once")
    counts = Counter(r["request_id"] for r in rows)
    if set(counts.values()) != {PASSES}:
        raise ValueError("Unequal endpoint exposure")
    cache, identities = {}, {}
    for row in rows:
        rid = row["request_id"]
        q = requests[rid]
        if (row["N"], row["K"]) != (q["N"], q["K"]):
            raise ValueError("Training stream N/K differs from frozen request")
        if type(row["known_reward"]) is not bool:
            raise ValueError("known_reward must be a JSON boolean")
        if row["known_reward"] and (not isinstance(row["r"], (float, int))
                or not math.isfinite(row["r"]) or not 0 <= row["r"] <= 1):
            raise ValueError("Known reward must be a finite probability")
        if not math.isfinite(row["t"]) or not .001 <= row["t"] <= .999:
            raise ValueError("Frozen time must be uniform then clipped to [.001,.999]")
        if type(row["renoise_seed"]) is not int or not 0 <= row["renoise_seed"] < 2**64:
            raise ValueError("Independent-noise seed must be an unsigned 64-bit integer")
        identity = (str(D.resolve(row["raw_path"])), row["raw_sha256"], row["r"], row["known_reward"])
        if rid in identities and identity != identities[rid]:
            raise ValueError("Endpoint/reward identity changed across passes")
        identities[rid] = identity
        if rid in cache:
            continue
        if D.sha(row["raw_path"]) != row["raw_sha256"]:
            raise ValueError("Training RAW endpoint hash changed")
        values = D.arrays(row["raw_path"])
        key = next((k for k in ("raw", "vertices", "output") if k in values), None)
        if key is None:
            raise ValueError("No registered endpoint coordinate key")
        raw = values[key]
        n, k = q["N"], q["K"]
        if raw.dtype != np.float32 or raw.size != n * 9 or not np.isfinite(raw).all():
            raise ValueError("Training endpoint must be finite FP32 N triangles")
        raw = raw.reshape(n, 9)
        known = np.zeros(n, dtype=np.bool_)
        known[:k] = True
        if "known_mask" in values and not D.same(values["known_mask"].reshape(n), known):
            raise ValueError("RAW known mask differs from frozen first-K contract")
        cond = D.load_request(q)["C"].reshape(k, 9)
        if not D.same(raw[:k], cond):
            raise ValueError("Training RAW does not preserve exact C")
        cache[rid] = dict(endpoint=raw.copy(), known_mask=known, N=n, K=k)
    return rows, cache


class TrainingBudget:
    """Count calls before invoking them; exceptions retain partial consumption."""
    def __init__(self, path, arm, round_index):
        self.handle = Path(path).open("x", encoding="utf-8", buffering=1)
        self.arm, self.round_index = arm, round_index
        self.forwards = 0
        self.max_forwards = 128 * (3 if arm == "NFT" else 2)

    def event(self, name, **fields):
        row = dict(event=name, utc=W.utc(), pid=os.getpid(), arm=self.arm, round_index=self.round_index)
        for key, value in fields.items():
            if key in row and key not in ("utc",) and row[key] != value:raise ValueError("Event identity conflict: " + key)
            row[key] = value
        self.handle.write(json.dumps(D.clean(row), allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def call(self, kind, fn, context):
        if kind != "training_forward":
            raise ValueError("Unexpected training budget kind")
        self.forwards += 1
        if self.forwards > self.max_forwards:
            raise RuntimeError("Frozen training forward limit exceeded")
        self.event("FORWARD_START", count=self.forwards, context=context)
        output = fn()
        self.event("FORWARD_RETURN", count=self.forwards)
        return output

    def close(self):
        self.handle.close()


def materialize(row, cached, device):
    """One independently noised free endpoint; no matching or GT reconstruction."""
    import torch
    raw = cached["endpoint"]
    noise = np.random.default_rng(row["renoise_seed"]).standard_normal(raw.shape).astype(np.float32)
    endpoint = torch.from_numpy(raw[None].copy()).to(device)
    known = torch.from_numpy(cached["known_mask"][None].copy()).to(device)
    reward = float(row["r"]) if row["known_reward"] else 0.5
    return dict(endpoint=endpoint, noise=torch.from_numpy(noise[None]).to(device),
        t=torch.tensor([row["t"]], dtype=torch.float32, device=device),
        known_mask=known, valid_mask=torch.ones_like(known),
        y=torch.tensor([cached["N"]], dtype=torch.long, device=device),
        r=torch.tensor([reward], dtype=torch.float32, device=device),
        known_reward=torch.tensor([row["known_reward"]], dtype=torch.bool, device=device))


def run(round_index, arm, manifest_path=None, recovery=False):
    if not 0 <= round_index < ROUNDS or arm not in ("NFT", "SELF_FM"):
        raise ValueError("Only frozen rounds 0..15 and NFT/SELF_FM arms are allowed")
    manifest_path = D.resolve(manifest_path or D.OUT / "inputs/manifest.json")
    manifest = D.read(manifest_path)
    if manifest.get("status") != "FROZEN":
        raise ValueError("Inputs must be frozen before training")
    stage = D.OUT / f"training/round_{round_index:02d}"
    stream_path = stage / "training_stream.json"
    rows, cache = prepare_stream(manifest, round_index, stream_path)
    stream_hash, manifest_hash = D.sha(stream_path), D.sha(manifest_path)
    checkpoint = D.OUT / f"checkpoints/{arm}/round{round_index:02d}.pt"
    if checkpoint.exists():
        raise RuntimeError("Completed checkpoint must be reused, never retrained")
    previous = None if round_index == 0 else D.OUT / f"checkpoints/{arm}/round{round_index - 1:02d}.pt"
    previous_receipt = None if previous is None else previous.with_suffix(".json")
    if previous is not None and (not previous.exists() or not previous_receipt.exists()):
        raise FileNotFoundError("Missing prior complete checkpoint/receipt")
    identity = dict(stage="training", arm=arm, round_index=round_index,
                    manifest_sha256=manifest_hash, training_stream_sha256=stream_hash,
                    start_step=round_index * UPDATES_PER_ROUND,
                    stop_step=(round_index + 1) * UPDATES_PER_ROUND)
    replay_rows = []
    if recovery:
        if round_index != 0 or arm != 'SELF_FM':raise ValueError('Only the registered interrupted SELF round0 recovery is authorized')
        auth=D.read(D.OUT/'engineering_recovery_round00/authorization.json')
        if auth['stream_sha256']!=D.sha(stream_path):raise ValueError('Recovery stream changed')
        old_events=stage/'train_SELF_FM/events.jsonl'
        if D.sha(old_events)!=auth['old_events_sha256']:raise ValueError('Interrupted ledger changed')
        replay_rows=[json.loads(line) for line in old_events.read_text().splitlines() if json.loads(line)['event']=='UPDATE_COMPLETE']
        if len(replay_rows)!=14:raise ValueError('Recovery complete-update evidence differs')
    worker_dir = stage / ("train_" + arm + ('_recovery01' if recovery else ''))
    with W.Phase(worker_dir, identity):
        import torch
        torch.set_num_threads(2)
        from .model import load_policy
        from .learning import make_optimizer, load_checkpoint, save_checkpoint, step_microbatches
        binding = manifest["source_binding"]
        policy = load_policy(D.resolve(binding["path"]), expected_sha256=binding["sha256"],
                             device="cuda", init_seed=20261015)
        optimizer = make_optimizer(policy)
        if previous is not None:
            receipt = D.read(previous_receipt)
            prior = load_checkpoint(previous, policy, optimizer, expected_sha256=receipt["sha256"])
            if (prior["step"], prior["round_index"], prior["arm"]) != (round_index * UPDATES_PER_ROUND, round_index - 1, arm):
                raise ValueError("Prior adapter/optimizer clock or arm differs")
        policy.update_old(decay=0.0)
        budget = TrainingBudget(worker_dir / "events.jsonl", arm, round_index)
        completed = 0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        try:
            for local in range(UPDATES_PER_ROUND):
                global_step = round_index * UPDATES_PER_ROUND + local + 1
                group = rows[local * MICROBATCHES_PER_UPDATE:(local + 1) * MICROBATCHES_PER_UPDATE]
                context = dict(update=global_step, local_update=local + 1,
                               request_ids=[r["request_id"] for r in group],
                               stream_row_start=local * MICROBATCHES_PER_UPDATE)
                budget.event("UPDATE_START", **context)
                batches = [materialize(row, cache[row["request_id"]], "cuda") for row in group]
                step_tick = time.perf_counter()
                stats = step_microbatches(policy, optimizer, batches, arm=arm,
                                          budget=budget, context=context)
                torch.cuda.synchronize()
                completed += 1
                stats.update(update=global_step, seconds=time.perf_counter() - step_tick,
                             cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                budget.event("UPDATE_COMPLETE", **stats)
                if recovery and local < len(replay_rows):
                    prior_stats=replay_rows[local]
                    equal=(stats['rows']==prior_stats['rows'] and stats['gradient_norm_before_clip']==prior_stats['gradient_norm_before_clip'])
                    budget.event('RECOVERY_NUMERIC_REPLAY_CHECK',update=global_step,exact_logged_loss_and_grad_equal=equal)
                    if not equal:raise RuntimeError('Recovery numeric log mismatch; stop, no further replay')
                print(f"TRAIN {arm} round={round_index:02d} update={global_step:03d} "
                      f"grad={stats['gradient_norm_before_clip']:.6g} seconds={stats['seconds']:.3f} "
                      f"reserved={stats['cuda_peak_reserved_bytes']}", flush=True)
                del batches
            if budget.forwards != budget.max_forwards or completed != UPDATES_PER_ROUND:
                raise RuntimeError("Completed round does not match registered call/update budget")
            metrics = dict(status="COMPLETE", **identity, completed_updates=completed,
                microbatches=128, actual_model_forwards=budget.forwards,
                training_seconds=time.perf_counter() - tick,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_scope="reset after policy/optimizer load, before all 16 complete forward/backward/optimizer updates; resident allocation included",
                process_peak_scope="separate Phase nvidia-smi polls; PyTorch counters do not include full process CUDA context",
                loss_label="NFT_STYLE" if arm == "NFT" else "SELF_REFLOW",
                self_control_off_policy=(arm == "SELF_FM"), old_update="exact_round_start_snapshot",
                endpoint_noise_coupling="independent_Gaussian_no_OT", reference="velocity_MSE_not_exact_KL")
            receipt = save_checkpoint(checkpoint, policy, optimizer,
                step=(round_index + 1) * UPDATES_PER_ROUND, round_index=round_index, arm=arm,
                metadata=dict(identity=identity, metrics=metrics))
            D.write(checkpoint.with_suffix(".json"), receipt)
            D.write(worker_dir / "training_metrics.json", metrics)
            budget.event("CHECKPOINT_COMPLETE", **receipt)
        except BaseException as exc:
            budget.event("TRAINING_FAILURE", error=repr(exc), completed_updates=completed,
                         consumed_model_forward_attempts=budget.forwards, retry_allowed=False)
            raise
        finally:
            budget.close()
    return dict(status="COMPLETE", checkpoint=str(checkpoint), **identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--arm", choices=["NFT", "SELF_FM"], required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--recovery-round0-self",action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.round, args.arm, args.manifest, args.recovery_round0_self), sort_keys=True))


if __name__ == "__main__":
    main()
