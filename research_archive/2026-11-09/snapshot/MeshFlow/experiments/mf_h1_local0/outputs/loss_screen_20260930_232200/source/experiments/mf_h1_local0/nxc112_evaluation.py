"""Bounded N x C and 112-face evaluation; no training or official-model loading.

Public entrypoints return JSON-safe evidence and save only raw NPZ arrays.
``progress_cb(event_dict)`` receives small events, never the full report.
The caller owns the sole official backbone and its authoritative global counts.
Failures carry ``error.evaluation_report`` for the caller's central results file.
"""
from __future__ import annotations

from .stable_runtime import stable_execution, assert_stable_runtime

from contextlib import contextmanager
import hashlib
import itertools
from pathlib import Path
import random
import traceback

import numpy as np
import torch

from .adapters import ControlAdapter
from .evaluate import guidance_metrics, completion_rms, raw_quality
from .learning_residuals import ResidualObserver
from .official_wrapper import OfficialWrapper, state_sha256
from .prompt_dataset import collate_prompts, collate_samples
from .sample import euler_sample

KIND = "mixed_2edges_1face"
N_VALUES = (217, 348, 750, 768, 512)
A_SEEDS = (6101, 6102)
B_SEEDS = (7101, 7102, 7103, 7104)
B_STEPS = (0, 100, 250, 500)
TEACHER_TIMES = (.05, .2, .5, .8, .95)
EXPECTED_BASE = "b32197096ecf8d662bfee18bbc4abf16cc4b4690cd28e5c47d14850729d2c222"
OLD_Q = {
    "Q_correct": dict(file="034325073569edf82149cd11b91895e8c95135d9bb2603b2c8244a513ec04a8a",
                      state="844f03824032a47a7a9a542613b4420e9fa6bb5154e37aea7b890c6e6e7bfbe6"),
    "Q_independent": dict(file="87b2bc41e69341aeaa58b29d64794634c6983af71c56304ffc97ea9aa7a295eb",
                          state="418a1a4e8063355d8e721b3cf9cccb74d1428941145281f01dd42bdc8bf30a17"),
}


def array_hash(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        dtype, shape = str(value.dtype), tuple(value.shape)
        raw = value.view(torch.uint8).numpy().tobytes()
    else:
        value = np.ascontiguousarray(value)
        dtype, shape, raw = str(value.dtype), tuple(value.shape), value.tobytes()
    return hashlib.sha256(str((dtype, shape)).encode() + raw).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_snapshot():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=[state.clone() for state in torch.cuda.get_rng_state_all()])


def _rng_equal(a, b):
    return (a["python"] == b["python"] and a["numpy"][0] == b["numpy"][0]
            and np.array_equal(a["numpy"][1], b["numpy"][1])
            and a["numpy"][2:] == b["numpy"][2:]
            and torch.equal(a["torch"], b["torch"])
            and len(a["cuda"]) == len(b["cuda"])
            and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"])))


def _fingerprint(bank):
    digest = hashlib.sha256()
    for uid in bank["parent_uids"]:
        digest.update(str(uid).encode())
        for name, value in sorted(bank["prompts"][(uid, KIND)].items()):
            digest.update(name.encode()); digest.update(array_hash(value).encode())
        digest.update(array_hash(bank["targets"][uid]).encode())
        digest.update(array_hash(bank["checks"][(uid, KIND)]).encode())
    for key, noise in sorted(bank["eval_noises"].items()):
        digest.update(str(key).encode()); digest.update(array_hash(noise).encode())
    for sample in bank.get("teacher_samples", []):
        for key in ("coords", "noise", "num_faces", "len", "t"):
            digest.update(key.encode()); digest.update(array_hash(np.asarray(sample[key])).encode())
    return digest.hexdigest()


def _cases(bank, count):
    uids = list(bank["parent_uids"])
    if len(uids) != count or len(set(uids)) != count:
        raise ValueError("Unexpected fixed parent identities")
    result = []
    for index, uid in enumerate(uids):
        record = bank["records_by_key"][(index, KIND)]
        if record["parent_uid"] != uid:
            raise ValueError("Prompt record identity differs from fixed parent order")
        prompt = bank["prompts"][(uid, KIND)]
        checks = np.asarray(bank["checks"][(uid, KIND)])
        target = np.asarray(bank["targets"][uid])
        if np.asarray(prompt["points"]).shape != (32, 3) or checks.shape != (3, 128, 3):
            raise ValueError("Mixed prompts require exactly 32 inputs and 3 x 128 independent checks")
        diagonal = float(record["bbox_diagonal"])
        if not np.isfinite(diagonal) or diagonal <= 0 or not np.isfinite(target).all():
            raise ValueError("Invalid target geometry or bbox")
        result.append(dict(parent_index=index, parent_uid=uid, prompt=prompt,
                           checks=checks, target=target, bbox_diagonal=diagonal))
    return result


def _injection_counts(base):
    # Root-level hooks used for global accounting are deliberately not included.
    return [sum(getattr(hook, "__name__", "") == "residual_hook"
                and getattr(hook, "__module__", "") == OfficialWrapper.__module__
                for hook in layer._forward_hooks.values()) for layer in base.layers]


def _assert_injections(base, expected):
    counts = _injection_counts(base)
    wanted = [int(expected and number in (3, 6, 9, 12)) for number in range(1, 13)]
    if counts != wanted:
        raise RuntimeError(f"Duplicate, stale, or missing wrapper insertion hooks: {counts}")


def _assert_base(base):
    if base.training or any(p.requires_grad or p.grad is not None for p in base.parameters()):
        raise RuntimeError("Official backbone must be frozen, eval, and gradient-free")
    value = state_sha256(base)
    if value != EXPECTED_BASE:
        raise RuntimeError("Official EMA state identity changed")
    return value


@contextmanager
def _count_base(base, counts):
    def before(module, args):
        counts["official_forward_attempts"] += 1
    def after(module, args, output):
        counts["official_forward_returns"] += 1
    handles = [base.register_forward_pre_hook(before), base.register_forward_hook(after)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _counts():
    return dict(official_forward_attempts=0, official_forward_returns=0,
                sampler_calls=0, complete_rollouts=0, sampling_official_forwards=0,
                adapted_rollouts=0, base_rollouts=0, condition_encodes=0,
                teacher_forward_attempts=0, teacher_forward_returns=0,
                teacher_examples=0, diagnostic_forward_attempts=0,
                guide_scores=0, surface_scores=0, quality_scores=0,
                reused_complete_rollouts=0, optimizer_updates=0, backbone_updates=0, phase3_updates=0)


def _emit(callback, report, event, **extra):
    if callback is not None:
        callback(dict(stage=report["stage"], event=event, step=report.get("step"),
                      counts=dict(report["actual_counts"]), **extra))


def _save_raw(path, arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve existing raw evidence: " + str(path))
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return dict(path=str(path.resolve()), sha256=_file_hash(path),
                arrays={key: dict(sha256=array_hash(value), shape=list(value.shape), dtype=str(value.dtype))
                        for key, value in arrays.items()})


def _difference(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    delta = a.double() - b.double()
    return dict(bitwise_equal=a.dtype == b.dtype and torch.equal(a, b),
                max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()))


def _sample(model, base, adapter, noise, n, prompt, report):
    counts = report["actual_counts"]
    device = next(base.parameters()).device
    if not torch.is_tensor(noise):
        noise = torch.from_numpy(np.asarray(noise))
    if noise.shape != (1, n, 9) or noise.dtype != torch.float32 or noise.device.type != "cpu":
        raise ValueError("Complete sampling requires its fixed CPU FP32 [1,N,9] Gaussian")
    source_hash = array_hash(noise)
    z = noise.to(device).clone()
    z_before = array_hash(z)
    y = torch.tensor([n], device=device, dtype=torch.long)
    mask = torch.ones((1, n), device=device, dtype=torch.bool)
    prompts = None if prompt is None else collate_prompts([prompt], device)
    assert_stable_runtime(base, adapter, coordinates=z)
    before = counts["official_forward_attempts"]
    wrapper_before = dict(model.counts) if isinstance(model, OfficialWrapper) else None
    counts["sampler_calls"] += 1
    try:
        value, stats = euler_sample(model, z, y, mask, prompts=prompts, precision="bf16")
        # A fully executed, nonfinite trajectory is still an actual completed call.
        counts["complete_rollouts"] += 1
        counts["base_rollouts" if prompt is None else "adapted_rollouts"] += 1
    finally:
        counts["sampling_official_forwards"] += counts["official_forward_attempts"]-before
        if wrapper_before is not None:
            counts["condition_encodes"] += model.counts["condition_encoding"]-wrapper_before["condition_encoding"]
    if counts["official_forward_attempts"]-before != 50 or stats["nfe"] != 50:
        raise RuntimeError("A trajectory did not execute exactly 50 official forwards")
    if stats["status"] != "PASS" or not torch.isfinite(value).all():
        raise RuntimeError("Nonfinite complete generation; no repair or resampling allowed")
    if array_hash(z) != z_before or array_hash(noise) != source_hash:
        raise RuntimeError("Sampling mutated its fixed initial noise")
    if wrapper_before is not None:
        delta = stats["call_delta"]
        if (delta["backbone_forward"] != 50 or delta["wrapper_forward"] != 50
                or delta["condition_encoding"] != int(prompt is not None)
                or delta["adapter_layer"] != 200*int(prompt is not None) or delta["cache_hits"]):
            raise RuntimeError("Wrapper sampling counts or cache policy changed")
    return value[0].detach().float().cpu().numpy(), dict(stats=stats,
        initial_noise_sha256=source_hash, actual_shape=[1,n,9], y=n,
        valid_mask_elements=n, mask_shape=[1,n], input_noise_unchanged=True)


def _single(model, base, adapter, noise, n, prompt, report, scale=1.):
    z = noise.to(next(base.parameters()).device).clone()
    before = array_hash(z)
    assert_stable_runtime(base, adapter, coordinates=z)
    t = torch.full((1,), .5, device=z.device, dtype=torch.float32)
    y = torch.tensor([n], device=z.device, dtype=torch.long)
    mask = torch.ones((1,n), device=z.device, dtype=torch.bool)
    wrapper_before = dict(model.counts) if isinstance(model, OfficialWrapper) else None
    report["actual_counts"]["diagnostic_forward_attempts"] += 1
    try:
        kwargs = dict(y=y, mask=mask)
        if isinstance(model, OfficialWrapper):
            kwargs.update(prompts=None if prompt is None else collate_prompts([prompt],z.device),
                          control_scale=scale, inference_geometry_cache=False)
        value = model(z,t,**kwargs)
    finally:
        if wrapper_before is not None:
            report["actual_counts"]["condition_encodes"] += model.counts["condition_encoding"]-wrapper_before["condition_encoding"]
    if array_hash(z) != before or not torch.isfinite(value).all():
        raise RuntimeError("Single-step diagnostic mutated input or produced nonfinite velocity")
    return value.detach().cpu()


def _score(tokens, cases, report):
    counts = report["actual_counts"]
    counts["quality_scores"] += 1
    quality = raw_quality(tokens)
    guide, guide_common, surface, surface_common, details = [], [], [], [], []
    for case in cases:
        diagonal = case["bbox_diagonal"]
        counts["guide_scores"] += 1
        g = guidance_metrics(tokens, case["checks"], diagonal)
        counts["surface_scores"] += 1
        s = completion_rms(tokens, case["target"], diagonal)
        guide.append(g["guide_rms_bbox_pct"])
        guide_common.append(g["guide_rms_bbox_pct"]*diagonal/100.)
        surface.append(s)
        surface_common.append(None if s is None else s*diagonal/100.)
        details.append(g["per_primitive"])
    nearest = None if any(value is None or not np.isfinite(value) for value in surface_common) else int(np.argmin(surface_common))
    return dict(quality=quality, guide_bbox_pct=guide, guide_common_units=guide_common,
                guide_per_primitive=details, surface_bbox_pct=surface,
                surface_common_units=surface_common, nearest_target_common_units=nearest)


def _load_q(path, arm, device):
    path = Path(path)
    digest = _file_hash(path)
    if digest != OLD_Q[arm]["file"]:
        raise RuntimeError(f"Wrong historical {arm} step1000 checkpoint bytes")
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (saved.get("arm") != arm or saved.get("step") != 1000
            or saved.get("completed_updates") != 1000 or saved.get("base_state_sha256") != EXPECTED_BASE):
        raise RuntimeError("Only the completed four-object Q step1000 state is authorized")
    with torch.device("meta"):
        adapter = ControlAdapter("Q_center")
    loaded = adapter.load_state_dict(saved["adapter"], strict=True, assign=True)
    adapter = adapter.to(device).requires_grad_(False).eval()
    state = state_sha256(adapter)
    if state != OLD_Q[arm]["state"]:
        raise RuntimeError("Historical Q adapter state identity changed")
    return adapter, dict(path=str(path.resolve()), file_sha256=digest, state_sha256=state,
                         checkpoint_step=1000, strict=True, missing_keys=loaded.missing_keys,
                         unexpected_keys=loaded.unexpected_keys,
                         parameters=sum(p.numel() for p in adapter.parameters()))


def _stage_a_matrices(rows):
    names = ("guide_bbox_pct", "guide_common_units", "surface_bbox_pct", "surface_common_units")
    result = {name: {} for name in names}
    for model in ("Base", *OLD_Q):
        for name in names:
            result[name][model] = {}
        for seed in A_SEEDS:
            for name in names:
                result[name][model][str(seed)] = {}
            for n in N_VALUES:
                selected = [r for r in rows if r["model"] == model and r["noise_seed"] == seed and r["N"] == n]
                selected.sort(key=lambda row: -1 if row["input_condition"] is None else row["input_condition"])
                for name in names:
                    result[name][model][str(seed)][str(n)] = [r[name] for r in selected]
    return dict(axes="metric[model][noise][N][input_condition][scoring_condition_or_target]",
                base_input_axis="one unconditional row, not four independent Base generations",
                scoring_bbox_denominators="Each column uses the scoring condition/target parent's fixed bbox; identical denominator for same-row comparisons E(Ci,.)",
                **result)


def _bindings(rows, condition_count, n_values, seeds, models):
    result = []
    for model, n, seed in itertools.product(models, n_values, seeds):
        indexed = {r["input_condition"]: r for r in rows
                   if r["model"] == model and r["N"] == n and r["noise_seed"] == seed}
        for a, b in itertools.combinations(range(condition_count), 2):
            ma, mb = indexed[a], indexed[b]
            four = [ma["guide_bbox_pct"][a], mb["guide_bbox_pct"][a],
                    mb["guide_bbox_pct"][b], ma["guide_bbox_pct"][b]]
            da, db = four[1]-four[0], four[3]-four[2]
            result.append(dict(model=model,N=n,noise_seed=seed,condition_a=a,condition_b=b,
                E_Ca_Ma=four[0],E_Ca_Mb=four[1],E_Cb_Mb=four[2],E_Cb_Ma=four[3],
                a_binding_improvement=da,b_binding_improvement=db,
                mean_binding_improvement=(da+db)/2,both_improve=da>0 and db>0))
    return result


def _stage_a_by_n(rows, bindings, cases):
    """Arithmetic on saved scores only; no added model or geometry calls."""
    def mean(values):
        values=list(values)
        return None if not values or any(v is None or not np.isfinite(v) for v in values) else float(np.mean(values))
    result={}
    for model in ("Base",*OLD_Q):
        result[model]={}
        for n in N_VALUES:
            selected=[r for r in rows if r["model"]==model and r["N"]==n]
            per_condition=[]
            for index,case in enumerate(cases):
                current=[r for r in selected if model=="Base" or r["input_condition"]==index]
                per_condition.append(dict(input_condition=index,parent_uid=case["parent_uid"],
                    source_target_true_N=int(case["target"].reshape(-1,9).shape[0]),
                    guide_bbox_pct=mean(r["guide_bbox_pct"][index] for r in current),
                    guide_common_units=mean(r["guide_common_units"][index] for r in current),
                    surface_bbox_pct=mean(r["surface_bbox_pct"][index] for r in current),
                    surface_common_units=mean(r["surface_common_units"][index] for r in current)))
            fixed=[r for r in bindings if r["model"]==model and r["N"]==n]
            result[model][str(n)]=dict(actual_rollouts=len(selected),per_condition=per_condition,
                own_guide_bbox_pct_mean=mean(r["guide_bbox_pct"] for r in per_condition),
                own_surface_bbox_pct_mean=mean(r["surface_bbox_pct"] for r in per_condition),
                own_surface_common_units_mean=mean(r["surface_common_units"] for r in per_condition),
                fixed_N_mean_binding_improvement=mean(r["mean_binding_improvement"] for r in fixed),
                fixed_N_both_binding_improve_pairs=sum(r["both_improve"] for r in fixed),
                fixed_N_binding_pairs=len(fixed),
                nearest_target_common_unit_counts=[sum(r["nearest_target_common_units"]==i for r in selected) for i in range(4)],
                nearest_target_invalid=sum(r["nearest_target_common_units"] is None for r in selected),
                input_condition_matches_nearest_target=None if model=="Base" else
                    sum(r["nearest_target_common_units"]==r["input_condition"] for r in selected),
                narrow_face_fraction_mean=mean(r["quality"]["min_angle_lt5_fraction"] for r in selected),
                degenerate_fraction_mean=mean(r["quality"]["degenerate_fraction"] for r in selected),
                interpretation="Within-N descriptive scores only; shared noises/targets are not independent replications")
    return result


def run_stage_a(base, bank_a, checkpoint_paths, output_dir, progress_cb=None):
    """New 90 trajectories plus seven single-step off/repeat diagnostics."""
    output_dir = Path(output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    if set(checkpoint_paths) != set(OLD_Q):
        raise ValueError("Stage A needs exactly Q_correct and Q_independent step1000 paths")
    if set(bank_a["eval_noises"]) != set(itertools.product(N_VALUES,A_SEEDS)):
        raise ValueError("Stage A requires exactly five N values and two fixed noises each")
    _assert_injections(base,False)
    cases = _cases(bank_a,4)
    identity = _fingerprint(bank_a); before_base = _assert_base(base); rng_before = _rng_snapshot()
    report = dict(stage="A",status="RUNNING",actual_counts=_counts(),parent_uids=bank_a["parent_uids"],
        actual_N_values=list(N_VALUES),noise_seeds=list(A_SEEDS),bank_fingerprint=identity,
        base_state_sha256_before=before_base,models={},outputs=[],raw_outputs=[],diagnostics={},
        historical_complete_rollouts_reused=0,
        score_units="bbox percentages use each scoring target's diagonal; common units undo exactly that normalization",
        surface_metric="Existing independent symmetric surface RMS, 512 points each surface, local seed9911",
        limitations=["Changing N changes both face conditioning and actual sequence length; their effects are not separately identified.",
                     "No coordinate-difference shape metric across N; noises across N are not a shared shape latent.",
                     "Association with N does not identify a unique y-embedding mechanism."])
    current_arrays = {}; current_model = "Base"; wrapper = None
    try:
        with _count_base(base,report["actual_counts"]), torch.inference_mode():
            with stable_execution(base) as facts:
                report["actual_runtime"] = facts
                for n, seed in itertools.product(N_VALUES,A_SEEDS):
                    noise=bank_a["eval_noises"][(n,seed)]
                    tokens,stats=_sample(base,base,None,noise,n,None,report)
                    key=f"N{n}_seed{seed}";current_arrays[key]=tokens
                    report["outputs"].append(dict(model="Base",N=n,noise_seed=seed,input_condition=None,
                        raw_key=key,sampling=stats,**_score(tokens,cases,report)))
                    _emit(progress_cb,report,"trajectory",model="Base",N=n,noise_seed=seed)
                base_reference=_single(base,base,None,bank_a["eval_noises"][(512,6101)],512,None,report)
            report["raw_outputs"].append(dict(model="Base",**_save_raw(output_dir/"stage_a_Base.npz",current_arrays)))
            current_arrays={}
            for arm in OLD_Q:
                current_model=arm
                adapter,audit=_load_q(checkpoint_paths[arm],arm,next(base.parameters()).device)
                report["models"][arm]=audit
                _assert_injections(base,False)
                wrapper=OfficialWrapper(base,adapter).eval();_assert_injections(base,True)
                try:
                    with stable_execution(base,adapter):
                        reference_noise=bank_a["eval_noises"][(512,6101)]
                        first=_single(wrapper,base,adapter,reference_noise,512,cases[0]["prompt"],report)
                        second=_single(wrapper,base,adapter,reference_noise,512,cases[0]["prompt"],report)
                        disabled=_single(wrapper,base,adapter,reference_noise,512,cases[0]["prompt"],report,scale=0.)
                        checks=dict(reencoded_repeat=_difference(first,second),control_off_vs_base=_difference(disabled,base_reference))
                        report["diagnostics"][arm]=checks
                        if not all(value["bitwise_equal"] for value in checks.values()):
                            raise RuntimeError("Stage A strict repeat/control-off check failed")
                        for n,seed,parent in itertools.product(N_VALUES,A_SEEDS,range(4)):
                            tokens,stats=_sample(wrapper,base,adapter,bank_a["eval_noises"][(n,seed)],n,cases[parent]["prompt"],report)
                            key=f"N{n}_C{parent}_seed{seed}";current_arrays[key]=tokens
                            report["outputs"].append(dict(model=arm,N=n,noise_seed=seed,input_condition=parent,
                                raw_key=key,sampling=stats,**_score(tokens,cases,report)))
                            _emit(progress_cb,report,"trajectory",model=arm,N=n,noise_seed=seed,input_condition=parent)
                    if state_sha256(adapter)!=audit["state_sha256"] or _file_hash(checkpoint_paths[arm])!=audit["file_sha256"]:
                        raise RuntimeError("Old Q weights changed during Stage A")
                    audit["unchanged"]=True
                finally:
                    wrapper.close();wrapper=None;_assert_injections(base,False)
                report["raw_outputs"].append(dict(model=arm,**_save_raw(output_dir/f"stage_a_{arm}.npz",current_arrays)))
                current_arrays={}
            report["matrices"]=_stage_a_matrices(report["outputs"])
            report["fixed_N_condition_bindings"]=_bindings(report["outputs"],4,N_VALUES,A_SEEDS,tuple(OLD_Q))
            report["by_N"]=_stage_a_by_n(report["outputs"],report["fixed_N_condition_bindings"],cases)
            report["expected_counts"]=dict(official_forward_attempts=4507,official_forward_returns=4507,
                sampler_calls=90,complete_rollouts=90,sampling_official_forwards=4500,base_rollouts=10,
                adapted_rollouts=80,condition_encodes=84,diagnostic_forward_attempts=7,
                guide_scores=360,surface_scores=360,quality_scores=90,optimizer_updates=0)
            if any(report["actual_counts"][k]!=v for k,v in report["expected_counts"].items()):
                raise RuntimeError("Stage A actual counts differ from its fixed plan")
            report["base_state_sha256_after"]=_assert_base(base)
            report["bank_unchanged"]=_fingerprint(bank_a)==identity
            report["rng_unchanged"]=_rng_equal(rng_before,_rng_snapshot())
            if not report["bank_unchanged"] or not report["rng_unchanged"]:
                raise RuntimeError("Stage A mutated its fixed inputs or RNG")
            report["status"]="PASS"
    except BaseException as error:
        report.update(status="FAIL",error=repr(error),traceback=traceback.format_exc())
        if current_arrays:
            report["raw_outputs"].append(dict(model=current_model,partial=True,
                **_save_raw(output_dir/f"stage_a_{current_model}_partial.npz",current_arrays)))
        error.evaluation_report=report
        _emit(progress_cb,report,"failed",error=repr(error))
        raise
    finally:
        if wrapper is not None:wrapper.close()
    _emit(progress_cb,report,"complete")
    return report


def _teacher_b(wrapper,bank,cases,report):
    samples=bank["teacher_samples"]
    if len(samples)!=2 or [s["parent_uid"] for s in samples]!=list(bank["parent_uids"]):
        raise ValueError("Stage B teacher requires two fixed, ordered real OT pairs")
    device=next(wrapper.backbone.parameters()).device
    result=dict(times=list(TEACHER_TIMES),microbatch=2,parents=bank["parent_uids"],rows=[],
        conditions=["off","correct","swapped"],source="Independent fixed real OT target/noise pairs; never free-generation noise")
    for time in TEACHER_TIMES:
        selected=[dict(sample,t=time,prompt=cases[index]["prompt"]) for index,sample in enumerate(samples)]
        batch=collate_samples(selected,device)
        x1,x0,t,mask=batch["tokens"],batch["noise"],batch["t"],batch["masks"]
        if (x1.shape!=(2,112,9) or x0.shape!=(2,112,9) or mask.shape!=(2,112)
                or not mask.all() or not (batch["num_faces"]==112).all()):
            raise RuntimeError("Stage B teacher identity channel has unequal N/masks")
        xt=(1-t[:,None,None])*x0+t[:,None,None]*x1; target=x1-x0
        before={k:array_hash(v) for k,v in dict(xt=xt,target=target,t=t,mask=mask,x0=x0,x1=x1).items()}
        swapped=collate_prompts([cases[1]["prompt"],cases[0]["prompt"]],device)
        predictions={};observer=ResidualObserver(wrapper) if time==.5 else None
        try:
            if observer is not None:observer.__enter__()
            for mode in ("off","correct","swapped"):
                assert_stable_runtime(wrapper.backbone,wrapper.adapter,coordinates=xt)
                prompts=None if mode=="off" else batch["prompts"] if mode=="correct" else swapped
                kwargs=dict(y=batch["num_faces"],mask=mask,prompts=prompts,
                            control_scale=0. if mode=="off" else 1.,inference_geometry_cache=False)
                report["actual_counts"]["teacher_forward_attempts"]+=1
                prior=wrapper.counts["condition_encoding"]
                try:
                    if observer is not None and mode!="off":
                        with observer.capture(mode,mask):value=wrapper(xt,t,**kwargs)
                    else:value=wrapper(xt,t,**kwargs)
                    report["actual_counts"]["teacher_forward_returns"]+=1
                    report["actual_counts"]["teacher_examples"]+=2
                finally:
                    report["actual_counts"]["condition_encodes"]+=wrapper.counts["condition_encoding"]-prior
                if not torch.isfinite(value).all():raise RuntimeError("Nonfinite fixed teacher output")
                predictions[mode]=value.detach().double()
            if observer is not None:
                result["residuals_t05"]=dict(observations=observer.report(),
                    correct_vs_swapped=observer.compare_effective("correct","swapped"))
                if not all(row["next_input_equals_h_after"] for capture in observer.report()["captures"] for row in capture["layers"]):
                    raise RuntimeError("Stage B observed residual boundary differs from next input")
        finally:
            if observer is not None:observer.close()
        errors={mode:target.double()-pred for mode,pred in predictions.items()}
        e=errors["off"];d=predictions["correct"]-predictions["off"]
        per_parent=[]
        for parent in range(2):
            valid=mask[parent];denom=int(valid.sum())*9
            loss={mode:float(error[parent][valid].square().sum())/denom for mode,error in errors.items()}
            d2=float(d[parent][valid].square().sum())/denom
            two_ed=float((2*e[parent][valid]*d[parent][valid]).sum())/denom
            delta=float((errors["correct"][parent][valid].square()-e[parent][valid].square()).sum())/denom
            per_parent.append(dict(parent_index=parent,valid_coordinates=denom,mse=loss,
                mean_d2=d2,two_mean_ed=two_ed,Lcorrect_minus_Loff=delta,
                algebra_absolute_error=abs(delta-(d2-two_ed))))
        row=dict(t=time,valid_coordinates=2016,per_parent=per_parent,
            mse={mode:sum(p["mse"][mode] for p in per_parent)/2 for mode in errors},
            mean_d2=sum(p["mean_d2"] for p in per_parent)/2,
            two_mean_ed=sum(p["two_mean_ed"] for p in per_parent)/2,
            Lcorrect_minus_Loff=sum(p["Lcorrect_minus_Loff"] for p in per_parent)/2,
            algebra_max_absolute_error=max(p["algebra_absolute_error"] for p in per_parent),input_hashes=before)
        if row["algebra_max_absolute_error"]>1e-10:raise RuntimeError("Teacher loss identity failed")
        if before!={k:array_hash(v) for k,v in dict(xt=xt,target=target,t=t,mask=mask,x0=x0,x1=x1).items()}:
            raise RuntimeError("Teacher inputs were mutated")
        result["rows"].append(row)
    return result


def _load_base_results(saved,bank_hash,base_hash):
    if (saved.get("status")!="PASS" or saved["bank_fingerprint"]!=bank_hash
            or saved["base_state_sha256"]!=base_hash):
        raise RuntimeError("Shared Stage B Base results do not match fixed assets")
    raw=saved["raw"]
    if _file_hash(raw["path"])!=raw["sha256"]:raise RuntimeError("Shared Base raw file changed")
    with np.load(raw["path"],allow_pickle=False) as stored:
        arrays={key:stored[key].copy() for key in stored.files}
    if {key:array_hash(value) for key,value in arrays.items()}!={key:value["sha256"] for key,value in raw["arrays"].items()}:
        raise RuntimeError("Shared Base output arrays changed")
    if set(arrays)!={f"seed{s}" for s in B_SEEDS}:raise RuntimeError("Shared Base needs exactly four noises")
    return arrays


def evaluate_stage_b(base,adapter,bank_b,step,output_dir,base_results=None,progress_cb=None,wrapper=None):
    """8 new trajectories and 15 batch-two teachers per fixed checkpoint.

    The first call additionally generates four shared Base draws. Step 500 has
    two repeat single forwards. Pass the returned ``base_results`` to later
    calls. Pass a live wrapper to retain its training counters; no second
    insertion hooks are registered, and its modes are restored in finally.
    """
    if step not in B_STEPS or adapter.arm!="S32_static":
        raise ValueError("Stage B evaluates only S32_static at steps 0/100/250/500")
    if bank_b is None:raise ValueError("Stage B geometry/prompt gate did not provide a valid bank")
    cases=_cases(bank_b,2)
    if any(case["target"].reshape(-1,9).shape[0]!=112 for case in cases):
        raise ValueError("Both Stage B targets must contain exactly 112 real triangles")
    if set(bank_b["eval_noises"])!=set(B_SEEDS):raise ValueError("Stage B requires fixed seeds7101-7104")
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    raw_path=output_dir/f"stage_b_step{step:04d}.npz"
    if raw_path.exists():raise FileExistsError(raw_path)
    bank_hash=_fingerprint(bank_b);base_hash=_assert_base(base);adapter_hash=state_sha256(adapter)
    rng_before=_rng_snapshot();own_wrapper=wrapper is None
    if own_wrapper:
        _assert_injections(base,False);wrapper=OfficialWrapper(base,adapter)
    elif wrapper.backbone is not base or wrapper.adapter is not adapter or wrapper._context is not None:
        raise ValueError("Provided wrapper must own these exact inactive base/adapter objects")
    _assert_injections(base,True)
    modes=wrapper.training,adapter.training,base.training
    prior_counts=dict(wrapper.counts)
    report=dict(stage="B_evaluation",step=step,status="RUNNING",actual_counts=_counts(),
        bank_fingerprint=bank_hash,base_state_sha256_before=base_hash,adapter_state_sha256_before=adapter_hash,
        parent_uids=bank_b["parent_uids"],N=112,noise_seeds=list(B_SEEDS),outputs=[],
        limits="Two real parents/four noises, no significance or generalization claim; endpoint500 fixed in advance.")
    arrays={};base_arrays={};first_base=base_results is None
    try:
        wrapper.eval()
        with _count_base(base,report["actual_counts"]),torch.inference_mode(),stable_execution(base,adapter) as facts:
            report["actual_runtime"]=facts
            if first_base:
                base_rows=[]
                for seed in B_SEEDS:
                    tokens,stats=_sample(wrapper,base,adapter,bank_b["eval_noises"][seed],112,None,report)
                    base_arrays[f"seed{seed}"]=tokens
                    base_rows.append(dict(model="Base",N=112,noise_seed=seed,input_condition=None,
                        sampling=stats,raw_key=f"seed{seed}",**_score(tokens,cases,report)))
                    _emit(progress_cb,report,"trajectory",model="Base",noise_seed=seed)
                base_results=dict(status="PASS",bank_fingerprint=bank_hash,base_state_sha256=base_hash,
                    rows=base_rows,raw=_save_raw(output_dir/"stage_b_Base.npz",base_arrays),independent_draws=4)
            else:
                base_arrays=_load_base_results(base_results,bank_hash,base_hash)
                report["actual_counts"]["reused_complete_rollouts"]=4
            report["base_results"]=base_results
            report["teacher"]=_teacher_b(wrapper,bank_b,cases,report)
            for seed,parent in itertools.product(B_SEEDS,range(2)):
                tokens,stats=_sample(wrapper,base,adapter,bank_b["eval_noises"][seed],112,cases[parent]["prompt"],report)
                key=f"C{parent}_seed{seed}";arrays[key]=tokens
                row=dict(model="S32_static",N=112,noise_seed=seed,input_condition=parent,
                         sampling=stats,raw_key=key,**_score(tokens,cases,report))
                if step==0:
                    reference=base_arrays[f"seed{seed}"]
                    row["zero_exit_vs_base"]=dict(bitwise_equal=tokens.dtype==reference.dtype and np.array_equal(tokens,reference),
                        max_abs=float(np.max(np.abs(tokens.astype(np.float64)-reference.astype(np.float64)))))
                    if not row["zero_exit_vs_base"]["bitwise_equal"]:
                        raise RuntimeError("Fresh S32 step0 generation does not recover Base")
                report["outputs"].append(row)
                _emit(progress_cb,report,"trajectory",model="S32_static",noise_seed=seed,input_condition=parent)
            if step==500:
                first=_single(wrapper,base,adapter,bank_b["eval_noises"][7101],112,cases[0]["prompt"],report)
                second=_single(wrapper,base,adapter,bank_b["eval_noises"][7101],112,cases[0]["prompt"],report)
                report["reencoded_single_step_repeat"]=_difference(first,second)
                if not report["reencoded_single_step_repeat"]["bitwise_equal"]:
                    raise RuntimeError("Learned S32 repeatability failed")
            report["bindings"]=_bindings(report["outputs"],2,(112,),B_SEEDS,("S32_static",))
            report["raw"]=_save_raw(raw_path,arrays)
            report["adapter_state_sha256_after"]=state_sha256(adapter)
            report["base_state_sha256_after"]=_assert_base(base)
            report["bank_unchanged"]=_fingerprint(bank_b)==bank_hash
            report["rng_unchanged"]=_rng_equal(rng_before,_rng_snapshot())
            if (report["adapter_state_sha256_after"]!=adapter_hash or not report["bank_unchanged"] or not report["rng_unchanged"]):
                raise RuntimeError("Stage B evaluation mutated weights, bank, or RNG")
            repeats=2*int(step==500);rollouts=8+4*int(first_base)
            report["expected_counts"]=dict(official_forward_attempts=50*rollouts+15+repeats,
                official_forward_returns=50*rollouts+15+repeats,sampler_calls=rollouts,complete_rollouts=rollouts,
                sampling_official_forwards=50*rollouts,base_rollouts=4*int(first_base),adapted_rollouts=8,
                condition_encodes=18+repeats,teacher_forward_attempts=15,teacher_forward_returns=15,
                teacher_examples=30,diagnostic_forward_attempts=repeats,guide_scores=2*rollouts,
                surface_scores=2*rollouts,quality_scores=rollouts,optimizer_updates=0)
            if any(report["actual_counts"][k]!=v for k,v in report["expected_counts"].items()):
                raise RuntimeError("Stage B checkpoint counts differ from its bounded plan")
            report["status"]="PASS"
    except BaseException as error:
        report.update(status="FAIL",error=repr(error),traceback=traceback.format_exc())
        if arrays and not raw_path.exists():report["partial_raw"]=_save_raw(output_dir/f"stage_b_step{step:04d}_partial.npz",arrays)
        error.evaluation_report=report
        _emit(progress_cb,report,"failed",error=repr(error))
        raise
    finally:
        report["wrapper_count_delta"]={key:wrapper.counts[key]-prior_counts.get(key,0) for key in wrapper.counts}
        wrapper.train(modes[0]);adapter.train(modes[1]);base.train(modes[2])
        if own_wrapper:wrapper.close();_assert_injections(base,False)
        else:_assert_injections(base,True)
    _emit(progress_cb,report,"complete")
    return report
