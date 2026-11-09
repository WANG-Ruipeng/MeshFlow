"""Independent log-only paired-stream checks. No pilot, torch or scorer imports."""
from __future__ import annotations

from collections import Counter
import hashlib
import math

TRANSLATION_KEYS = frozenset(("translation_seed", "translation_xyz"))
TRANSLATION_DOMAIN = "R2_BENCH_JOINT_SHIFT_V1_TRANSLATION_ONLY"
L = math.sqrt(3.) * 2. / .3762
AXIS_BOUND = .04 * L
FIRST_MICRO_METRICS = (
    "loss_contribution", "all_free_velocity_sse", "all_free_scalar_count",
    "bulk_velocity_sse", "bulk_scalar_count", "seam_tip_velocity_sse",
    "seam_tip_scalar_count", "endpoint_sse", "bulk_endpoint_sse",
    "seam_tip_endpoint_sse",
)
SAME_RUNTIME_INPUTS = ("N", "K", "t", "local_step", "global_step", "microbatch",
                       "all_free_scalar_count", "bulk_scalar_count", "seam_tip_scalar_count")


def require_equal(a, b, message):
    if a != b:
        raise ValueError(message)


def paired_stream(schedule, reference_schedule, exposure, reference_exposure,
                  expected_updates=1500):
    """Compare every old scheduled field, then actual sampled identity and bounds.

    Actual SSE, timestamps, PID and mechanism prose are outputs, not input-stream
    fields. SSE/loss equality is tested only for first-update identity micros.
    """
    new_steps, old_steps = schedule["steps"], reference_schedule["steps"]
    require_equal(len(new_steps), expected_updates, "New schedule length differs")
    require_equal(len(old_steps), expected_updates, "Reference schedule length differs")
    require_equal(len(exposure), expected_updates * 8, "New exposure length differs")
    require_equal(len(reference_exposure), expected_updates * 8, "Reference exposure length differs")
    fields_compared = runtime_compared = zero = shifted = 0
    vectors, by_augmentation = [], Counter()
    observed_old_fields = set()
    for index, (new, old) in enumerate(zip(new_steps, old_steps)):
        step = index + 1
        require_equal(new["step"], step, "New schedule step order differs")
        require_equal({k:v for k,v in new.items() if k != "samples"},
                      {k:v for k,v in old.items() if k != "samples"}, "Old step fields differ")
        require_equal(len(new["samples"]), 8, "New batch differs")
        require_equal(len(old["samples"]), 8, "Reference batch differs")
        for micro, (sample, ref) in enumerate(zip(new["samples"], old["samples"])):
            position = index * 8 + micro
            actual, before = exposure[position], reference_exposure[position]
            require_equal(set(sample), set(ref) | TRANSLATION_KEYS, "Unexpected scheduled keys")
            require_equal({k:v for k,v in sample.items() if k not in TRANSLATION_KEYS}, ref,
                          "Old scheduled input fields changed")
            require_equal((actual["local_step"], actual["sample_index"]), (step, micro),
                          "Actual exposure order differs")
            require_equal((before["local_step"], before["sample_index"]), (step, micro),
                          "Reference actual exposure order differs")
            for key, value in ref.items():
                require_equal(actual.get(key), value, "New actual input mismatch: " + key)
                require_equal(before.get(key), value, "Reference actual input mismatch: " + key)
                fields_compared += 1
                observed_old_fields.add(key)
            for key in SAME_RUNTIME_INPUTS:
                require_equal(actual[key], before[key], "Actual runtime input mismatch: " + key)
                runtime_compared += 1
            for key in TRANSLATION_KEYS:
                require_equal(actual.get(key), sample[key], "Actual translation mismatch: " + key)
            token = f'{TRANSLATION_DOMAIN}|{step}|{int(sample["sample_index"])}|{sample["condition_id"]}'
            seed = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "little")
            require_equal(sample["translation_seed"], seed, "Translation RNG identity differs")
            vector = sample["translation_xyz"]
            if len(vector) != 3 or not all(math.isfinite(x) and abs(x) <= AXIS_BOUND for x in vector):
                raise ValueError("Joint translation outside frozen per-axis +/-0.04L bound")
            kind = sample["augmentation"]
            by_augmentation[kind] += 1
            if kind == "CLEAN":
                require_equal(vector, [0., 0., 0.], "Identity half has nonzero translation")
                require_equal(sample["scale_xyz"], [1., 1., 1.], "Identity scale changed")
                if micro >= 4:
                    raise ValueError("Identity micro is not in first half of update")
                zero += 1
            elif kind == "ANISOTROPIC":
                if micro < 4 or vector == [0., 0., 0.]:
                    raise ValueError("Translated half identity/order differs")
                vectors.append(vector)
                shifted += 1
            else:
                raise ValueError("Unknown original augmentation")
    require_equal((zero, shifted), (expected_updates*4, expected_updates*4),
                  "Expected equal zero/translated halves")
    norms = [math.sqrt(sum(x*x for x in v)) for v in vectors]
    axes = {}
    for i, axis in enumerate(("x", "y", "z")):
        values = [v[i] for v in vectors]
        axes[axis] = dict(min=min(values), max=max(values), mean=sum(values)/len(values),
                          rms=math.sqrt(sum(x*x for x in values)/len(values)),
                          negative=sum(x < 0 for x in values), positive=sum(x > 0 for x in values))
    return dict(status="PASS", updates=expected_updates, exposures=len(exposure),
                old_schedule_fields=sorted(observed_old_fields), added_keys=sorted(TRANSLATION_KEYS),
                old_input_field_comparisons=fields_compared, runtime_input_comparisons=runtime_compared,
                runtime_input_fields=list(SAME_RUNTIME_INPUTS), zero_translation_samples=zero,
                shifted_samples=shifted, by_augmentation=dict(by_augmentation),
                L=L, axis_bound_model=AXIS_BOUND, axis_bound_L=.04,
                vector_bound_model=math.sqrt(3.)*AXIS_BOUND, axis_statistics_shifted_only=axes,
                vector_norm_statistics_shifted_only=dict(min=min(norms), max=max(norms),
                    mean=sum(norms)/len(norms), max_fraction_L=max(norms)/L),
                reproducibility="Seed identity and actual vector match frozen schedule; vector bounds checked. No RNG or model rerun.",
                comparison_scope="All old scheduled input fields plus N/K/t/cursor/scalar counts; loss/SSE after optimizer updates are not expected equal.")


def first_identity_micros(exposure, reference_exposure):
    """Exact logged scalar comparison before the first optimizer update."""
    rows = []
    for micro in range(4):
        new = next(r for r in exposure if r["local_step"] == 1 and r["microbatch"] == micro)
        old = next(r for r in reference_exposure if r["local_step"] == 1 and r["microbatch"] == micro)
        require_equal(new["condition_id"], old["condition_id"], "First micro condition differs")
        require_equal(new["augmentation"], "CLEAN", "First four micros must be identity")
        require_equal(new["translation_xyz"], [0.,0.,0.], "First micro is translated")
        for metric in FIRST_MICRO_METRICS:
            a, b = old[metric], new[metric]
            if not all(isinstance(x, (int,float)) and math.isfinite(x) for x in (a,b)):
                raise ValueError("Missing/nonfinite first identity diagnostic")
            rows.append(dict(local_step=1, microbatch=micro, condition_id=new["condition_id"],
                             metric=metric, reference=a, candidate=b, delta=b-a, exact_equal=a == b))
    mismatches = [r for r in rows if not r["exact_equal"]]
    return dict(status="PASS" if not mismatches else "FAIL", micros=4,
                comparisons=len(rows), exact_equal=len(rows)-len(mismatches), mismatches=mismatches,
                metric_fields=list(FIRST_MICRO_METRICS), rows=rows,
                model_calls=0, extra_probe_calls=0, automatic_retry=False,
                boundary="Only first-update identity micros before any update; not full trajectory repeatability.")


def registration_identity(new, old):
    keys = ("updates", "batch", "microbatch", "hybrid_lambda", "source_checkpoint_sha256",
            "source_generator_sha256", "source_optimizer_sha256", "source_rng_sha256",
            "source_global_step", "final_global_step", "training_seed", "model_config")
    for key in keys:
        if key not in new or key not in old:
            raise ValueError("Missing registration identity: " + key)
        require_equal(new[key], old[key], "Registration initialization/config changed: " + key)
    return dict(status="PASS", matching_fields=list(keys),
                scope="Recorded frozen initialization identities/config; no weight load or forward")
