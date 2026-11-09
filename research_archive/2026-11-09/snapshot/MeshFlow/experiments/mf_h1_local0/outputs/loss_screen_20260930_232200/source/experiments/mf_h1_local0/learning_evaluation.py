"""Bounded, paired evaluation of the new Q learnability checkpoints.

No model is loaded and no optimizer is created here. The caller owns the one
official backbone and serializes evaluation with training. Import the shared
runtime policy before importing torch in every executable entrypoint.
"""
from __future__ import annotations

from .stable_runtime import stable_execution, assert_stable_runtime

import hashlib
import itertools
import json
import math
import traceback
from pathlib import Path

import numpy as np
import torch

from .evaluate import guidance_metrics, raw_quality, completion_rms
from .official_wrapper import state_sha256
from .prompt_dataset import collate_samples, collate_prompts, PROMPT_TYPES
from .sample import euler_sample

ARMS = ("Q_correct", "Q_independent")
CHECKPOINT_STEPS = (0, 200, 500, 1000)
EVAL_SEEDS = (6101, 6102)
TEACHER_TIMES = (.05, .2, .5, .8, .95)
MODES = ("off", "correct", "cyclic_mismatch")


def array_hash(value):
    """Hash numerical content, dtype and shape, independently of its location."""
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, str(value.dtype))).encode()
                          + value.tobytes()).hexdigest()


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_safe_json(value), indent=2, allow_nan=False), encoding="utf-8")


def _count_delta(before, after):
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in set(before) | set(after)}


def _mean(rows, fn):
    values = [fn(row) for row in rows]
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def _rng_snapshot():
    # Evaluation has supplied noise and local scoring RNGs, so it must not
    # advance the training RNG. Compare states instead of silently restoring it.
    import random
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=[state.clone() for state in torch.cuda.get_rng_state_all()])


def _rng_equal(a, b):
    return (a["python"] == b["python"]
            and a["numpy"][0] == b["numpy"][0]
            and np.array_equal(a["numpy"][1], b["numpy"][1])
            and a["numpy"][2:] == b["numpy"][2:]
            and torch.equal(a["torch"], b["torch"])
            and len(a["cuda"]) == len(b["cuda"])
            and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"])))


class EvaluationRunner:
    """One runner per experiment, sharing two Base draws and exact step-0 results.

    ``evaluate_checkpoint(wrapper, arm, step)`` is the training callback.
    ``counters`` is optional; only names prefixed ``evaluation_`` are updated.
    The caller's global official-backbone hook remains the authoritative total.
    Reports also contain actual wrapper deltas, including any failed attempt.
    """
    def __init__(self, base, bank, output, counters=None):
        self.base, self.bank = base, bank
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.results = self.output / "results"
        self.raw = self.output / "raw_evaluation"
        self.results.mkdir(exist_ok=True)
        self.raw.mkdir(exist_ok=True)
        self.counters = counters if counters is not None else {}
        self.parent_uids = list(bank["parent_uids"])
        if len(self.parent_uids) != 4 or len(set(self.parent_uids)) != 4:
            raise ValueError("Exactly four distinct training parents are required")
        if len(bank["teacher_samples"]) != 4:
            raise ValueError("Exactly four fixed real teacher target/noise pairs are required")
        if [sample["parent_uid"] for sample in bank["teacher_samples"]] != self.parent_uids:
            raise ValueError("Fixed teacher targets must follow the four training-parent order")
        if set(bank["eval_noises"]) != set(EVAL_SEEDS):
            raise ValueError("Evaluation uses exactly independent seeds 6101 and 6102")
        self.cases = {}
        for parent, uid in enumerate(self.parent_uids):
            for kind in PROMPT_TYPES:
                record = bank["records_by_key"][(parent, kind)]
                if record["parent_uid"] != uid:
                    raise ValueError("Prompt record parent order mismatch")
                with np.load(record["path"], allow_pickle=False) as stored:
                    checks = stored["check_points"].copy()
                    target = stored["target"].copy()
                prompt = bank["prompts"][(uid, kind)]
                if not np.array_equal(target, np.asarray(bank["targets"][uid])):
                    raise ValueError("Prompt evaluation target differs from the frozen parent target")
                self.cases[(parent, kind)] = dict(record=record, prompt=prompt,
                                                 check_points=checks, target=target)
        for noise in bank["eval_noises"].values():
            if noise.shape != (1, 512, 9) or noise.dtype != torch.float32 or noise.device.type != "cpu":
                raise ValueError("Evaluation noise must be fixed CPU FP32 [1,512,9]")
        self.bank_hash = self._bank_hash()
        self.base_report = None
        self.base_arrays = {}
        self.zero_report = None

    def _bank_hash(self):
        records = dict(parents=self.parent_uids,
            noise={str(seed): array_hash(self.bank["eval_noises"][seed]) for seed in EVAL_SEEDS},
            prompts={f"{parent}:{kind}": {key: array_hash(value) for key, value in case["prompt"].items()}
                     for (parent, kind), case in self.cases.items()},
            checks={f"{parent}:{kind}": array_hash(case["check_points"])
                    for (parent, kind), case in self.cases.items()},
            targets={uid: array_hash(self.bank["targets"][uid]) for uid in self.parent_uids},
            teacher=[dict(parent_uid=item["parent_uid"],
                coords=array_hash(item["coords"]), noise=array_hash(item["noise"]),
                num_faces=int(item["num_faces"])) for item in self.bank["teacher_samples"]])
        return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()

    def _increment(self, report, key, value=1):
        report["actual_counts"][key] = report["actual_counts"].get(key, 0) + int(value)

    def _sample(self, wrapper, seed, prompt, report):
        assert_stable_runtime(wrapper.backbone, wrapper.adapter)
        device = next(wrapper.backbone.parameters()).device
        noise = self.bank["eval_noises"][seed].to(device)
        y = torch.tensor([512], device=device, dtype=torch.long)
        mask = torch.ones((1, 512), device=device, dtype=torch.bool)
        assert_stable_runtime(wrapper.backbone, wrapper.adapter, coordinates=noise)
        before_hash = array_hash(noise)
        prompts = None if prompt is None else collate_prompts([prompt], device)
        self._increment(report, "sampler_calls")
        generated, stats = euler_sample(wrapper, noise, y, mask, prompts=prompts, precision="bf16")
        if stats["status"] != "PASS" or not torch.isfinite(generated).all():
            raise RuntimeError("Nonfinite complete generation; no repairs or resampling are allowed")
        if array_hash(noise) != before_hash:
            raise RuntimeError("Sampling modified its independent initial noise")
        if stats["call_delta"]["backbone_forward"] != 50:
            raise RuntimeError("The sampler did not execute the specified 50 official forwards")
        expected_encodes = int(prompts is not None)
        if stats["call_delta"]["condition_encoding"] != expected_encodes:
            raise RuntimeError("Sampling condition encoding count changed")
        if stats["call_delta"]["cache_hits"]:
            raise RuntimeError("Q evaluation must not enable geometry caching")
        self._increment(report, "complete_rollouts")
        self._increment(report, "base_rollouts" if prompts is None else "adapted_rollouts")
        return generated[0].detach().float().cpu().numpy(), stats

    def _base(self, wrapper, report, base_hash):
        path = self.results / "evaluation_base.json"
        raw_path = self.raw / "evaluation_base.npz"
        if self.base_report is None and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved.get("status") != "PASS" or saved["bank_hash"] != self.bank_hash
                    or saved["base_state_hash"] != base_hash):
                raise RuntimeError("Existing Base cache is incomplete or belongs to different assets")
            with np.load(raw_path, allow_pickle=False) as stored:
                self.base_arrays = {key: stored[key].copy() for key in stored.files}
            if {key: array_hash(value) for key, value in self.base_arrays.items()} != saved["array_hashes"]:
                raise RuntimeError("Persisted shared Base output hashes differ")
            self.base_report = saved
        if self.base_report is not None:
            if self.base_report["base_state_hash"] != base_hash or self.base_report["bank_hash"] != self.bank_hash:
                raise RuntimeError("Shared Base results no longer match the frozen base/input state")
            self._increment(report, "base_rollouts_reused", 2)
            return
        if raw_path.exists():
            raise FileExistsError("Incomplete prior Base artifact is preserved: " + str(raw_path))
        before = dict(wrapper.counts)
        base = dict(status="RUNNING", bank_hash=self.bank_hash, base_state_hash=base_hash,
                    independent_base_draws=2, noise_seeds=list(EVAL_SEEDS), rows=[], actual_rollouts=[],
                    interpretation="Two unconditional generated meshes are shared across parents, types, arms and checkpoints; repeated scores are not independent generated samples.")
        for seed in EVAL_SEEDS:
            tokens, stats = self._sample(wrapper, seed, None, report)
            self.base_arrays[f"seed{seed}"] = tokens
            base["actual_rollouts"].append(dict(seed=seed, stats=stats))
            quality = raw_quality(tokens)
            self._increment(report, "quality_scores")
            for parent, kind in itertools.product(range(4), PROMPT_TYPES):
                case = self.cases[(parent, kind)]
                guide = guidance_metrics(tokens, case["check_points"], case["record"]["bbox_diagonal"])
                completion = completion_rms(tokens, case["target"], case["record"]["bbox_diagonal"])
                self._increment(report, "guide_scores")
                self._increment(report, "completion_scores")
                base["rows"].append(dict(parent_index=parent, parent_uid=self.parent_uids[parent],
                    kind=kind, noise_seed=seed, guide=guide, quality=quality,
                    completion_symmetric_rms_bbox_pct=completion, output_reference=f"seed{seed}"))
        base.update(status="PASS", wrapper_count_delta=_count_delta(before, wrapper.counts),
                    raw_path=str(raw_path), array_hashes={key: array_hash(value) for key, value in self.base_arrays.items()})
        np.savez_compressed(raw_path, **self.base_arrays)
        _write_json(path, base)
        self.base_report = base

    def _teacher(self, wrapper, report):
        from .learning_residuals import ResidualObserver

        device = next(wrapper.backbone.parameters()).device
        samples = self.bank["teacher_samples"]
        teacher = dict(time_points=list(TEACHER_TIMES), effective_samples=4, microbatch=2,
            parent_uids=self.parent_uids, prompt_kind="mixed_2edges_1face",
            mismatch_rule="Teacher target j gets the same-type prompt of parent (j+1) modulo 4; target, OT-paired noise, mask and time are unchanged.",
            target_noise_source="Fixed real teacher pairs in learning_data bank; distinct from independent evaluation noise.",
            time_distribution_changed_for_training=False, rows=[], microbatches=[], residuals={})
        report["teacher"] = teacher
        for time in TEACHER_TIMES:
            sums = dict(denominator=0, d2=0., two_ed=0., correct_minus_off=0.,
                        correct_minus_off_identity=0., correct_minus_off_mse_subtraction=0.,
                        mse={mode: 0. for mode in MODES}, samples=[])
            for start in (0, 2):
                selected = []
                for j in range(start, start + 2):
                    item = dict(samples[j])
                    item["t"] = time
                    item["prompt"] = self.bank["prompts"][(self.parent_uids[j], "mixed_2edges_1face")]
                    selected.append(item)
                batch = collate_samples(selected, device)
                x0, x1, mask, t = batch["noise"], batch["tokens"], batch["masks"], batch["t"]
                if x0.dtype != torch.float32 or x1.dtype != torch.float32:
                    raise RuntimeError("Teacher integration state must remain FP32")
                xt = t[:, None, None] * x1 + (1 - t[:, None, None]) * x0
                target = x1 - x0
                mismatch = collate_prompts([self.bank["prompts"][(self.parent_uids[(j + 1) % 4], "mixed_2edges_1face")]
                                           for j in range(start, start + 2)], device)
                inputs = dict(xt=array_hash(xt), velocity_target=array_hash(target), mask=array_hash(mask),
                              target=array_hash(x1), noise=array_hash(x0), t=array_hash(t), N=array_hash(batch["num_faces"]))
                predictions = {}
                observer = ResidualObserver(wrapper, retain_effective=True) if time == .5 else None
                try:
                    if observer is not None:
                        observer.__enter__()
                    for mode in MODES:
                        assert_stable_runtime(wrapper.backbone, wrapper.adapter, coordinates=xt)
                        prompt = None if mode == "off" else batch["prompts"] if mode == "correct" else mismatch
                        self._increment(report, "teacher_forwards")
                        kwargs = dict(y=batch["num_faces"], mask=mask, prompts=prompt,
                                      control_scale=0. if mode == "off" else 1.)
                        if observer is not None and mode != "off":
                            with observer.capture(mode, mask):
                                prediction = wrapper(xt, t, **kwargs)
                        else:
                            prediction = wrapper(xt, t, **kwargs)
                        if not torch.isfinite(prediction).all():
                            raise RuntimeError("Nonfinite fixed teacher velocity")
                        predictions[mode] = prediction.detach().double()
                    if observer is not None:
                        teacher["residuals"][f"micro{start//2}"] = dict(
                            at_time=.5, parent_indices=[start, start+1], observations=observer.report(),
                            correct_vs_mismatch=observer.compare_effective("correct", "cyclic_mismatch"))
                finally:
                    if observer is not None:
                        observer.__exit__(None, None, None)
                target64 = target.double()
                errors = {mode: target64 - pred for mode, pred in predictions.items()}
                e = errors["off"]
                d = predictions["correct"] - predictions["off"]
                for j in range(2):
                    valid = mask[j]
                    denominator = int(valid.sum().item()) * 9
                    loss_sums = {mode: float(value[j][valid].square().sum().item()) for mode, value in errors.items()}
                    d2 = float(d[j][valid].square().sum().item())
                    two_ed = float((2 * e[j][valid] * d[j][valid]).sum().item())
                    # Both expressions are evaluated in FP64 from the same FP32
                    # model outputs, reducing cancellation in the identity test.
                    loss_delta = float((errors["correct"][j][valid].square() - e[j][valid].square()).sum().item())
                    identity = d2 - two_ed
                    sums["denominator"] += denominator
                    sums["d2"] += d2
                    sums["two_ed"] += two_ed
                    sums["correct_minus_off"] += loss_delta
                    sums["correct_minus_off_identity"] += identity
                    sums["correct_minus_off_mse_subtraction"] += loss_sums["correct"] - loss_sums["off"]
                    for mode in MODES:
                        sums["mse"][mode] += loss_sums[mode]
                    sums["samples"].append(dict(parent_index=start+j, parent_uid=self.parent_uids[start+j],
                        true_faces=int(batch["num_faces"][j]), valid_coordinate_denominator=denominator,
                        mse={mode: value/denominator for mode, value in loss_sums.items()},
                        mean_d2=d2/denominator, two_mean_ed=two_ed/denominator,
                        Lcorrect_minus_Loff=loss_delta/denominator,
                        algebra_absolute_error=abs(loss_delta-identity)/denominator))
                if inputs != dict(xt=array_hash(xt), velocity_target=array_hash(target), mask=array_hash(mask),
                                   target=array_hash(x1), noise=array_hash(x0), t=array_hash(t), N=array_hash(batch["num_faces"])):
                    raise RuntimeError("Teacher input tensors were modified in place")
                teacher["microbatches"].append(dict(t=time, parent_indices=[start,start+1], input_hashes=inputs))
            denom = sums["denominator"]
            row = dict(t=time, valid_coordinate_denominator=denom,
                       mse={mode: value/denom for mode, value in sums["mse"].items()},
                       mean_d2=sums["d2"]/denom, two_mean_ed=sums["two_ed"]/denom,
                       Lcorrect_minus_Loff=sums["correct_minus_off"]/denom,
                       d2_minus_two_ed=sums["correct_minus_off_identity"]/denom,
                       Lcorrect_minus_Loff_from_separate_sums=sums["correct_minus_off_mse_subtraction"]/denom,
                       algebra_absolute_error=abs(sums["correct_minus_off"]-sums["correct_minus_off_identity"])/denom,
                       per_parent=sums["samples"])
            if row["algebra_absolute_error"] > 1e-10:
                raise RuntimeError("The fixed-mask teacher d2 - 2ed loss identity failed")
            teacher["rows"].append(row)
        teacher["identity_normalization"] = "All terms sum over the identical valid true-N coordinates and divide by their full shared count; no time reweighting."

    def _score(self, outputs, report):
        cache, quality, own_rows, cross_rows = {}, {}, [], []

        def metric(mesh_parent, target_parent, kind, seed):
            key = mesh_parent, target_parent, kind, seed
            if key not in cache:
                case = self.cases[(target_parent, kind)]
                tokens = outputs[(mesh_parent, kind, seed)]
                cache[key] = dict(guide=guidance_metrics(tokens, case["check_points"], case["record"]["bbox_diagonal"]),
                    completion=completion_rms(tokens, case["target"], case["record"]["bbox_diagonal"]))
                self._increment(report, "guide_scores")
                self._increment(report, "completion_scores")
            return cache[key]

        for key, tokens in outputs.items():
            quality[key] = raw_quality(tokens)
            self._increment(report, "quality_scores")
        for parent, kind, seed in itertools.product(range(4), PROMPT_TYPES, EVAL_SEEDS):
            value = metric(parent, parent, kind, seed)
            own_rows.append(dict(parent_index=parent, parent_uid=self.parent_uids[parent], kind=kind,
                noise_seed=seed, guide=value["guide"], quality=quality[(parent, kind, seed)],
                completion_symmetric_rms_bbox_pct=value["completion"]))
        for kind, seed, a in itertools.product(PROMPT_TYPES, EVAL_SEEDS, range(4)):
            b = (a + 1) % 4
            aa, ab = metric(a, a, kind, seed), metric(b, a, kind, seed)
            bb, ba = metric(b, b, kind, seed), metric(a, b, kind, seed)
            guide = [row["guide"]["guide_rms_bbox_pct"] for row in (aa, ab, bb, ba)]
            completion = [row["completion"] for row in (aa, ab, bb, ba)]
            ma, mb = outputs[(a, kind, seed)], outputs[(b, kind, seed)]
            difference = ma.astype(np.float64) - mb.astype(np.float64)
            cross_rows.append(dict(parent_a=a, parent_b=b, parent_uid_a=self.parent_uids[a],
                parent_uid_b=self.parent_uids[b], kind=kind, noise_seed=seed,
                guide=dict(zip(("E_Ca_Ma", "E_Ca_Mb", "E_Cb_Mb", "E_Cb_Ma"), guide)),
                guide_a_binding_improvement=guide[1]-guide[0],
                guide_b_binding_improvement=guide[3]-guide[2],
                guide_mean_binding_improvement=((guide[1]-guide[0])+(guide[3]-guide[2]))/2,
                full_surface_completion=dict(zip(("E_Ta_Ma", "E_Ta_Mb", "E_Tb_Mb", "E_Tb_Ma"), completion)),
                completion_mean_binding_improvement=None if any(v is None for v in completion) else
                    ((completion[1]-completion[0])+(completion[3]-completion[2]))/2,
                same_noise_verified=True, noise_sha256=array_hash(self.bank["eval_noises"][seed]),
                output_change=dict(bitwise_equal=bool(np.array_equal(ma, mb)),
                    rms=float(np.sqrt(np.mean(difference*difference))), max_abs=float(np.max(np.abs(difference))))))
        summary = {}
        for kind in (*PROMPT_TYPES, "all"):
            own = [r for r in own_rows if kind == "all" or r["kind"] == kind]
            base = [r for r in self.base_report["rows"] if kind == "all" or r["kind"] == kind]
            cross = [r for r in cross_rows if kind == "all" or r["kind"] == kind]
            summary[kind] = dict(conditional_scores=len(own), independent_noise_seeds=2, distinct_parents=4,
                own_guide_mean=_mean(own, lambda r:r["guide"]["guide_rms_bbox_pct"]),
                base_guide_mean=_mean(base, lambda r:r["guide"]["guide_rms_bbox_pct"]),
                own_completion_mean=_mean(own, lambda r:r["completion_symmetric_rms_bbox_pct"]),
                base_completion_mean=_mean(base, lambda r:r["completion_symmetric_rms_bbox_pct"]),
                guide_mean_binding_improvement=_mean(cross, lambda r:r["guide_mean_binding_improvement"]),
                completion_mean_binding_improvement=_mean(cross, lambda r:r["completion_mean_binding_improvement"]),
                both_guide_bindings_improve_pairs=sum(r["guide_a_binding_improvement"] > 0 and r["guide_b_binding_improvement"] > 0 for r in cross),
                cyclic_pairs=len(cross), narrow_face_fraction_mean=_mean(own, lambda r:r["quality"]["min_angle_lt5_fraction"]),
                degenerate_fraction_mean=_mean(own, lambda r:r["quality"]["degenerate_fraction"]))
        report["scores"] = dict(own_condition_rows=own_rows, same_noise_cross_four_grid=cross_rows, summary=summary,
            shared_base_reference=str(self.results/"evaluation_base.json"),
            topology=dict(self_intersection="NOT_RUN", watertight="NOT_RUN", manifold="NOT_RUN"),
            interpretation="Training-parent learnability only. Paired scores reuse the same two independent noise draws; parent/type/cyclic rows are not independent experiments. Coordinate changes alone do not establish useful control.")

    def evaluate_checkpoint(self, wrapper, arm, step):
        if arm not in ARMS or step not in CHECKPOINT_STEPS:
            raise ValueError("Only the two authorized Q arms and four fixed checkpoints are evaluated")
        if wrapper.backbone is not self.base or wrapper.adapter.arm != "Q_center":
            raise ValueError("Evaluation must use the caller's sole official backbone and Q adapter")
        if any(parameter.requires_grad for parameter in self.base.parameters()):
            raise RuntimeError("The official backbone must remain frozen")
        if self._bank_hash() != self.bank_hash:
            raise RuntimeError("Frozen evaluation bank changed")
        name = f"evaluation_{arm}_step{step:04d}"
        destination = self.results/(name+".json")
        raw_path = self.raw/(name+".npz")
        if destination.exists() or raw_path.exists():
            raise FileExistsError("Preserve prior evaluation evidence: " + str(destination))
        adapter_hash, base_hash = state_sha256(wrapper.adapter), state_sha256(self.base)
        if step == 0 and self.zero_report is None:
            candidates = [self.results/f"evaluation_{other}_step0000.json" for other in ARMS if other != arm]
            for candidate in candidates:
                if candidate.exists():
                    self.zero_report = json.loads(candidate.read_text(encoding="utf-8"))
        if step == 0 and self.zero_report is not None:
            source = self.zero_report
            if (source.get("status") != "PASS" or source["adapter_state_hash"] != adapter_hash
                    or source["base_state_hash"] != base_hash or source["bank_hash"] != self.bank_hash):
                raise RuntimeError("Step-0 reuse requires complete state and input hash equality")
            with np.load(source["raw_path"], allow_pickle=False) as arrays:
                hashes = {key: array_hash(arrays[key]) for key in arrays.files}
            if hashes != source["array_hashes"]:
                raise RuntimeError("Step-0 raw outputs changed before cross-arm reuse")
            reused = dict(source, arm=arm, step=0, result_path=str(destination),
                status="PASS_REUSED_IDENTICAL_INITIAL_STATE",
                reused_from=source["result_path"], actual_counts=dict(official_forwards=0,
                    condition_encodes=0, complete_rollouts=0, teacher_forwards=0,
                    optimizer_updates=0, backbone_updates=0, phase3_updates=0),
                wrapper_count_delta={}, reuse=dict(adapted_rollouts=24, teacher_forwards=30,
                                                  note="Exact same Q initialization, base and all evaluation inputs; no independent sample added."))
            _write_json(destination, reused)
            self.counters["evaluation_checkpoint_reuses"] = self.counters.get("evaluation_checkpoint_reuses", 0) + 1
            return reused
        before = dict(wrapper.counts)
        modes_before = wrapper.training, wrapper.adapter.training, self.base.training
        rng_before = _rng_snapshot()
        arrays = {}
        report = dict(status="RUNNING", arm=arm, step=step, bank_hash=self.bank_hash,
            adapter_state_hash=adapter_hash, base_state_hash=base_hash,
            result_path=str(destination), raw_path=str(raw_path), actual_rollouts=[],
            policy=dict(training_updates_here=0, geometric_cache=False, raw_face_slots=512,
                euler_updates=50, eval_noise_seeds=list(EVAL_SEEDS), face_cfg=1., geometry_cfg=1.,
                parameter_and_integrator_dtype="float32", precision="bf16",
                true_N_training_and_teacher="Teacher and training retain original real N and masks; only complete-generation face condition is 512.",
                target_access="Targets and independent primitive checks are used only by teacher or CPU scoring; the free sampler receives independent noise, 512, masks and prompts."),
            actual_counts=dict(official_forwards=0, condition_encodes=0, complete_rollouts=0,
                base_rollouts=0, adapted_rollouts=0, teacher_forwards=0, sampler_calls=0,
                quality_scores=0, guide_scores=0, completion_scores=0,
                optimizer_updates=0, backbone_updates=0, phase3_updates=0))
        try:
            wrapper.eval()
            with torch.inference_mode(), stable_execution(self.base, wrapper.adapter) as facts:
                report["stable_runtime"] = facts
                self._base(wrapper, report, base_hash)
                report["base_reference"] = dict(path=str(self.results/"evaluation_base.json"),
                    total_independent_draws=2, reused_score_rows=24,
                    actual_base_rollouts_this_call=report["actual_counts"]["base_rollouts"])
                self._teacher(wrapper, report)
                outputs = {}
                for parent, kind, seed in itertools.product(range(4), PROMPT_TYPES, EVAL_SEEDS):
                    case = self.cases[(parent, kind)]
                    tokens, stats = self._sample(wrapper, seed, case["prompt"], report)
                    key = f"parent{parent}_{kind}_seed{seed}"
                    outputs[(parent, kind, seed)] = tokens
                    arrays[key] = tokens
                    report["actual_rollouts"].append(dict(parent_index=parent,
                        parent_uid=self.parent_uids[parent], kind=kind, noise_seed=seed,
                        initial_noise_sha256=array_hash(self.bank["eval_noises"][seed]),
                        prompt_hashes={k:array_hash(v) for k,v in case["prompt"].items()},
                        array_key=key, stats=stats))
                    print(json.dumps(dict(stage="learning_evaluation", arm=arm, step=step,
                        complete=len(outputs), planned=24)), flush=True)
                assert_stable_runtime(self.base, wrapper.adapter)
            np.savez_compressed(raw_path, **arrays)
            report["array_hashes"] = {key: array_hash(value) for key, value in arrays.items()}
            self._score(outputs, report)
            report["adapter_state_hash_after"] = state_sha256(wrapper.adapter)
            report["base_state_hash_after"] = state_sha256(self.base)
            report["rng_unchanged"] = _rng_equal(rng_before, _rng_snapshot())
            report["bank_unchanged"] = self._bank_hash() == self.bank_hash
            if (report["adapter_state_hash_after"] != adapter_hash or report["base_state_hash_after"] != base_hash
                    or not report["rng_unchanged"] or not report["bank_unchanged"]):
                raise RuntimeError("Evaluation mutated model, input-bank or training RNG state")
            expected = 30 + 50 * (24 + report["actual_counts"]["base_rollouts"])
            if wrapper.counts["backbone_forward"] - before["backbone_forward"] != expected:
                raise RuntimeError("Checkpoint official forward count differs from the bounded evaluation plan")
            if wrapper.counts["condition_encoding"] - before["condition_encoding"] != 44:
                raise RuntimeError("Checkpoint condition encode count differs from 20 teacher plus 24 sampler encodes")
            report["status"] = "PASS"
        except BaseException as error:
            report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
            if self.base_arrays and not (self.raw/"evaluation_base.npz").exists():
                np.savez_compressed(self.raw/(name+"_partial_base.npz"), **self.base_arrays)
            if arrays and not raw_path.exists():
                np.savez_compressed(self.raw/(name+"_partial.npz"), **arrays)
            raise
        finally:
            wrapper.train(modes_before[0])
            wrapper.adapter.train(modes_before[1])
            self.base.train(modes_before[2])
            report["wrapper_count_delta"] = _count_delta(before, wrapper.counts)
            report["actual_counts"]["official_forwards"] = report["wrapper_count_delta"]["backbone_forward"]
            report["actual_counts"]["condition_encodes"] = report["wrapper_count_delta"]["condition_encoding"]
            for key, value in report["actual_counts"].items():
                counter_key = "evaluation_" + key
                self.counters[counter_key] = self.counters.get(counter_key, 0) + value
            _write_json(destination, report)
        if step == 0:
            self.zero_report = report
        return report


def evaluate_checkpoint(wrapper, arm, step, bank, output, counters=None):
    """Convenience callback; prefer one persistent EvaluationRunner in the driver.

    Base and initial-state reuse can also be recovered from verified new-run
    files, so a newly constructed callback does not rerun these cached outputs.
    """
    return EvaluationRunner(wrapper.backbone, bank, output, counters).evaluate_checkpoint(wrapper, arm, step)
