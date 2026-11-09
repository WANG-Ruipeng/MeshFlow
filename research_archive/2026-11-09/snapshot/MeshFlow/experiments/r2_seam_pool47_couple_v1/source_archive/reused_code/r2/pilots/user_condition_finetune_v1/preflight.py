"""Eight fixed cached samples: 8 forward, 8 zero-update backward calls.

Invocation is owned by the controller. Persistent 16F/8B attempt limits include
failures; this entrypoint never retries, changes samples, or calls optimizer.step.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import traceback
try:
    from . import trainer as train
except ImportError:
    import trainer as train


def run_preflight(*, out, registration, attempt_id, data_root=None,
                  base_checkpoint=train.BASE_CHECKPOINT, device="cuda"):
    if not str(device).startswith("cuda"):
        raise ValueError("Real preflight requires controller-owned CUDA")
    train.safe_attempt_id(attempt_id)
    out = Path(out).resolve()
    registered = train.check_registration(registration, out)
    if data_root is not None and Path(data_root).resolve() != Path(registered["data_root"]).resolve():
        raise ValueError("Prepared dataset root differs")
    try:
        from . import data as pilot_data
    except ImportError:
        import data as pilot_data
    folder = out / "preflight"
    with train.exclusive_writer(folder):
        if (folder / "completed.json").exists():
            raise FileExistsError("Successful preflight already exists; no extra calls authorized")
        ledger = train.CallLedger(folder / "calls.jsonl", train.PREFLIGHT_LIMITS, "PREFLIGHT", attempt_id)
        if ledger.pending:
            raise RuntimeError("Unclosed preflight calls require explicit accounting recovery")
        samples = pilot_data.preflight_samples(out)
        denominator = train.validate_samples(samples, effective_batch=False)
        prior_rows = {}
        for receipt_path in sorted((folder / "attempts").glob("*/samples.jsonl")):
            for line in receipt_path.read_text().splitlines():
                prior = json.loads(line)
                index = prior["preflight_index"]
                if index in prior_rows or not 0 <= index < 8:
                    raise ValueError("Duplicate/unregistered successful preflight sample")
                sample = samples[index]
                if prior.get("input_bytehash") != sample["input_bytehash"] or prior.get("label_bytehash") != sample["label_bytehash"]:
                    raise ValueError("Preflight input changed since prior attempt")
                prior_rows[index] = prior
        remaining = [i for i in range(8) if i not in prior_rows]
        if not remaining:
            raise RuntimeError("All eight backward calls were consumed without final PASS; explicit failure resolution required")
        # An explicit subsequent invocation may finish remaining fixed samples;
        # successful backward calls are never repeated or erased from the limit.
        if ledger.attempted["preflight_forward"] + len(remaining) > 16 or ledger.attempted["preflight_backward"] + len(remaining) > 8:
            raise RuntimeError("Insufficient remaining budget for unfinished fixed preflight samples")
        attempt = folder / "attempts" / attempt_id
        attempt.mkdir(parents=True, exist_ok=False)
        state = dict(status="LOADING", attempt_id=attempt_id, fixed_sample_count=8,
            optimizer_updates=0, source_registration_sha256=train.digest(registration),
            pilot_code_manifest_sha256=train.digest(out / "code_manifest.json"),
            preflight_peak_scope="GPU peaks reset after full model/AdamW load; eight separate finite-gradient forward/backward checks, zero optimizer steps",
            sample_ids=[dict(arm=s["arm"], sample_index=int(s["sample_index"]), task_id=s["task_id"],
                N=int(s["y"]), K=int(s["K"]), region_type=s.get("region_type"), input_bytehash=s["input_bytehash"],
                label_bytehash=s["label_bytehash"], ot_cache_key=s["ot_cache_key"]) for s in samples])
        train.atomic_json(attempt / "status.json", state)
        started = time.perf_counter()
        observer = train.GPUObserver("preflight_8F8B_zero_optimizer")
        try:
            train.torch.set_num_threads(2)
            train.runtime.configure_stable_runtime()
            model, optimizer, lineage = train.load_base(base_checkpoint, device)
            state["lineage"] = lineage
            before = dict(generator=train.state_hash(model.state_dict()),
                          optimizer=train.baseline.tree_hash(optimizer.state_dict()))
            rng_before = train._cpu_copy(train.rng_state())
            rng_before_sha = train.baseline.tree_hash(rng_before)
            optimizer.zero_grad(set_to_none=True)
            train.torch.cuda.reset_peak_memory_stats(device)
            rows = list(prior_rows.values())
            with observer:
                for index in remaining:
                    sample = samples[index]
                    optimizer.zero_grad(set_to_none=True)
                    batch = train.collate([sample], device)
                    known = batch["known_mask"]
                    input_before = batch["xt"].detach().clone()
                    context = dict(preflight_index=index, arm=sample["arm"],
                        sample_index=int(sample["sample_index"]), N=int(sample["y"]), K=int(sample["K"]))
                    with train.native_execution(model, coordinates=(batch["xt"], batch["t"]), autocast=True):
                        velocity, forward_seconds = ledger.call("preflight_forward", lambda: model(*train.model_inputs(batch)).float(), context)
                        if not bool(train.torch.isfinite(velocity).all()):
                            raise FloatingPointError("Nonfinite preflight velocity")
                        fm = train.masked_fm_loss(velocity, batch["u"], batch["valid_mask"], known, denominator=denominator)
                        if not bool(train.torch.isfinite(fm)):
                            raise FloatingPointError("Nonfinite preflight FM")
                        _, backward_seconds = ledger.call("preflight_backward", fm.backward, context)
                    if not train.torch.equal(batch["xt"], input_before):
                        raise ValueError("Model mutated the training input")
                    # Compare raw bytes to include signed zero, not torch.equal only.
                    actual = batch["xt"][known].detach().cpu().numpy()
                    expected = sample["context"][sample["known_mask"]]
                    if actual.tobytes() != expected.tobytes():
                        raise ValueError("Known C bit pattern changed")
                    parameters = list(model.parameters())
                    if any(p.grad is None or not bool(train.torch.isfinite(p.grad).all()) for p in parameters):
                        raise FloatingPointError("Missing or nonfinite preflight gradient")
                    gradient_norm = float(train.torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
                    row = dict(**context, input_bytehash=sample["input_bytehash"], label_bytehash=sample["label_bytehash"],
                        finite_gradient_norm_before_clipping=gradient_norm, FM_contribution=float(fm.detach()),
                        batch_free_denominator=denominator, known_C_bitwise=True,
                        free_count=int((batch["valid_mask"] & ~known).sum()),
                        forward_seconds=forward_seconds, backward_seconds=backward_seconds,
                        velocity_finite=True)
                    rows.append(row)
                    train.append_event(attempt / "samples.jsonl", dict(event="PREFLIGHT_SAMPLE_DONE", **row))
                    del velocity, fm, batch, input_before
                peak_allocated = train.torch.cuda.max_memory_allocated(device)
                peak_reserved = train.torch.cuda.max_memory_reserved(device)
            after = dict(generator=train.state_hash(model.state_dict()),
                         optimizer=train.baseline.tree_hash(optimizer.state_dict()))
            if before != after:
                raise ValueError("Zero-update preflight changed parameters or optimizer state")
            train.validate_optimizer(model, optimizer, 6000, lineage["optimizer_hyperparameters"])
            # Observation cannot advance the continuation's inherited RNG.
            train.restore_rng(rng_before)
            if train.baseline.tree_hash(train.rng_state()) != rng_before_sha:
                raise ValueError("Preflight RNG restoration differs")
            optimizer.zero_grad(set_to_none=True)
            state.update(status="PASS", before=before, after=after, hashes_unchanged=True,
                inherited_rng_sha256=rng_before_sha, rng_restored=True, results=sorted(rows,key=lambda r:r["preflight_index"]),
                completed_samples_reused=len(prior_rows), successful_fixed_samples=len(rows),
                cuda_peak_allocated_bytes=peak_allocated, cuda_peak_reserved_bytes=peak_reserved,
                scope_note="Backward peak does not substitute for measured AdamW-update process peak")
        except BaseException as exc:
            state.update(status="FAILED", error=repr(exc), traceback=traceback.format_exc(), no_automatic_retry=True)
            raise
        finally:
            state.update(calls=ledger.snapshot(), wall_seconds=time.perf_counter()-started,
                resource_observation=observer.summary(), optimizer_updates=0)
            train.atomic_json(attempt / "resources.json", dict(summary=observer.summary(), samples=observer.samples))
            train.atomic_json(attempt / "status.json", state)
            train.atomic_json(folder / "status.json", state)
            if state["status"] == "PASS":
                train.atomic_json(folder / "completed.json", state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, default=train.BASE_CHECKPOINT)
    result = run_preflight(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
