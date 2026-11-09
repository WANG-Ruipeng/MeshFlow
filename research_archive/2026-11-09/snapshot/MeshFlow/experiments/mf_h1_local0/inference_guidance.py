"""Base-referenced final-velocity guidance; the adapter's internal scale is one.

This module never loads weights or creates an optimizer. Its CUDA entrypoints
require the existing stable runtime. The caller owns the sole active wrapper.
"""
from __future__ import annotations

from .stable_runtime import stable_execution, assert_stable_runtime

from dataclasses import fields, is_dataclass
import hashlib
import time
import torch

SETTINGS = ("A0", "A1", "A2", "A3", "A4", "A5")
EXPECTED_NFE = dict(A0=50, A1=100, A2=100, A3=100, A4=59, A5=100, OFF=50)


def alpha_table(setting):
    """Exactly the preregistered integer-clock schedules; OFF is a test branch."""
    if setting == "A4":
        return [1. if k <= 40 else (50-k)/10 for k in range(50)]
    constants = dict(A0=1., A1=1.5, A2=2., A3=2.5, A5=.91, OFF=0.)
    if setting not in constants:
        raise ValueError("Unregistered final-velocity setting: " + str(setting))
    return [constants[setting]] * 50


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode()
                          + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _tree_hash(value):
    digest = hashlib.sha256()
    def visit(current):
        if torch.is_tensor(current):
            digest.update(_tensor_hash(current).encode())
        elif is_dataclass(current):
            for field in fields(current):
                digest.update(field.name.encode()); visit(getattr(current, field.name))
        elif isinstance(current, dict):
            for key, item in sorted(current.items()):
                digest.update(str(key).encode()); visit(item)
        elif isinstance(current, (tuple, list)):
            for item in current: visit(item)
        elif current is None:
            digest.update(b"None")
        else:
            raise TypeError("Unsupported inference input container: " + type(current).__name__)
    visit(value)
    return digest.hexdigest()


def combine_velocity(v_cond, v_off, alpha):
    """Pure FP32 arithmetic, with exact identity branches (also CPU-testable)."""
    if alpha == 1.:
        if v_cond is None: raise ValueError("alpha=1 requires conditional velocity")
        return v_cond.float()
    if alpha == 0.:
        if v_off is None: raise ValueError("alpha=0 requires off velocity")
        return v_off.float()
    if v_cond is None or v_off is None or v_cond.shape != v_off.shape:
        raise ValueError("Both same-shape velocities are required for extrapolation")
    return v_off.float() + float(alpha) * (v_cond.float() - v_off.float())


def guided_velocity(wrapper, xt, t, y, mask, encoded, alpha, audit=None):
    """Evaluate both branches at this exact Xt, never on another trajectory.

    Off means absent conditioning. Every invocation explicitly keeps the
    wrapper's internal control_scale=1, including the off invocation.
    """
    if xt.dtype != torch.float32 or t.dtype != torch.float32:
        raise ValueError("Velocity input and integer-clock time must be FP32")
    if wrapper._context is not None:
        raise RuntimeError("Stale wrapper context before velocity evaluation")
    if alpha != 0. and encoded is None:
        raise ValueError("Conditional velocity requires this trajectory's encoding")
    audit = {} if audit is None else audit
    def call(conditional):
        key = "conditional_forwards" if conditional else "off_forwards"
        audit[key] = audit.get(key, 0) + 1
        return wrapper(xt, t, y=y, mask=mask, prompts=None,
                       encoded=encoded if conditional else None,
                       control_scale=1., inference_geometry_cache=False)
    if alpha == 1.:
        return combine_velocity(call(True), None, alpha)
    if alpha == 0.:
        return combine_velocity(None, call(False), alpha)
    # One explicit object is passed to both calls. Check its value as well as
    # identity, so an in-place mutation cannot silently invalidate the pair.
    before = xt.clone()
    v_cond = call(True)
    if not torch.equal(xt, before): raise RuntimeError("Conditional call mutated Xt")
    v_off = call(False)
    if not torch.equal(xt, before): raise RuntimeError("Off call mutated the shared Xt")
    audit["same_xt_pair_checks"] = audit.get("same_xt_pair_checks", 0) + 1
    return combine_velocity(v_cond, v_off, alpha)


@torch.no_grad()
def guided_euler_sample(wrapper, z, prompts, setting="A0"):
    """50 FP32 Euler updates; one encoding, actual NFE, no hidden model call."""
    alphas = alpha_table(setting)
    if z.shape != (1,112,9) or z.dtype != torch.float32 or z.device.type != "cuda":
        raise ValueError("This pilot requires CUDA FP32 [1,112,9] initial noise")
    if any(module.training for module in (wrapper, wrapper.backbone, wrapper.adapter)):
        raise RuntimeError("Guided sampling requires eval mode for wrapper, base and adapter")
    if wrapper._context is not None:
        raise RuntimeError("Stale wrapper context before sampling")
    if (prompts is None) != all(alpha == 0. for alpha in alphas):
        raise ValueError("Only the OFF diagnostic omits prompts")
    before_counts = dict(wrapper.counts)
    before_z, before_prompt = _tensor_hash(z), _tree_hash(prompts)
    stats = dict(status="RUNNING", setting=setting, nfe=0, euler_updates=0,
                 conditional_forwards=0, off_forwards=0, same_xt_pair_checks=0,
                 condition_encodes=0, alpha=alphas, alpha_mean=sum(alphas)/50,
                 last_evaluation_time=.98, last_alpha=alphas[-1],
                 internal_control_scale=1., face_cfg=1., dt=1/50,
                 grid=[k/50 for k in range(51)], precision="bf16",
                 velocity_combination_dtype="float32", integrator_dtype="float32",
                 expected_nfe=EXPECTED_NFE[setting])
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        with stable_execution(wrapper.backbone, wrapper.adapter):
            assert_stable_runtime(wrapper.backbone, wrapper.adapter, coordinates=z)
            state = z.clone()
            y = torch.tensor([112], device=z.device, dtype=torch.long)
            mask = torch.ones((1,112), device=z.device, dtype=torch.bool)
            encoded = wrapper.encode(prompts) if prompts is not None else None
            encoded_before = _tree_hash(encoded)
            for k, alpha in enumerate(alphas):
                t = torch.full((1,), k/50, device=z.device, dtype=torch.float32)
                velocity = guided_velocity(wrapper,state,t,y,mask,encoded,alpha,stats)
                state = state + velocity.float() * (1/50)
                stats["euler_updates"] += 1
            stats["encoded_unchanged"] = _tree_hash(encoded) == encoded_before
            stats["input_unchanged"] = _tensor_hash(z) == before_z and _tree_hash(prompts) == before_prompt
            if not stats["encoded_unchanged"] or not stats["input_unchanged"]:
                raise RuntimeError("Trajectory mutated its fixed inputs or encoding")
            stats["status"] = "PASS" if bool(torch.isfinite(state).all()) else "FAIL_NONFINITE"
        return state, stats
    except BaseException as error:
        stats.update(status="FAIL",error=repr(error))
        error.guidance_stats = stats
        raise
    finally:
        torch.cuda.synchronize()
        stats["seconds"] = time.perf_counter()-start
        stats["peak_cuda_allocated"] = torch.cuda.max_memory_allocated()
        stats["call_delta"] = {k:wrapper.counts[k]-before_counts.get(k,0) for k in wrapper.counts}
        stats["nfe"] = stats["call_delta"]["backbone_forward"]
        stats["condition_encodes"] = stats["call_delta"]["condition_encoding"]
        if stats["status"] == "PASS":
            if (stats["nfe"] != EXPECTED_NFE[setting] or stats["euler_updates"] != 50
                    or stats["condition_encodes"] != int(prompts is not None)
                    or stats["call_delta"]["adapter_layer"] != 4*stats["conditional_forwards"]
                    or stats["call_delta"].get("cache_hits",0)):
                error=RuntimeError("Guidance execution counts differ from registered schedule")
                stats["status"]="FAIL";error.guidance_stats=stats
                raise error


def combination_precheck(base, post, bank, historical_base, output_dir=None):
    """58 official calls, zero updates: eight velocities and one OFF rollout."""
    from pathlib import Path
    import numpy as np
    from .nxc112_evaluation import (_cases, _counts, _count_base, _fingerprint,
        _assert_base, _load_base_results, _difference, _rng_snapshot, _rng_equal, _save_raw)
    from .official_wrapper import state_sha256
    from .prompt_dataset import collate_prompts
    counts = _counts()
    report = dict(stage="A_combination_precheck",status="RUNNING",actual_counts=counts,
                  optimizer_updates=0,formal_development_rollouts=0,diagnostic_rollouts=0)
    if post.backbone is not base or post._context is not None:
        raise ValueError("Precheck requires the inactive Post wrapper on this base")
    cases = _cases(bank,2);bank_hash = _fingerprint(bank)
    before_base = _assert_base(base);before_adapter = state_sha256(post.adapter)
    archived = _load_base_results(historical_base,bank_hash,before_base)
    rng_before = _rng_snapshot(); modes = post.training,post.adapter.training,base.training
    prior_counts = dict(post.counts)
    try:
        post.eval()
        with _count_base(base,counts),torch.inference_mode(),stable_execution(base,post.adapter):
            device = next(base.parameters()).device
            x = bank["eval_noises"][7101].to(device).clone()
            t = torch.full((1,),.5,device=device,dtype=torch.float32)
            y = torch.tensor([112],device=device,dtype=torch.long)
            mask = torch.ones((1,112),device=device,dtype=torch.bool)
            prompts = collate_prompts([cases[0]["prompt"]],device)
            encoded = post.encode(prompts)
            before = _tree_hash(dict(x=x,t=t,y=y,mask=mask,prompts=prompts,encoded=encoded))
            observed = []
            def observe(module,args,kwargs):
                actual_x,actual_t=args[:2]
                actual_y=args[2] if len(args)>2 else kwargs["y"]
                observed.append(dict(x_ptr=actual_x.data_ptr(),x=_tensor_hash(actual_x),
                                     t=_tensor_hash(actual_t),y=_tensor_hash(actual_y),
                                     mask=_tensor_hash(kwargs["mask"])))
            handle = base.register_forward_pre_hook(observe,with_kwargs=True)
            audit={}
            try:
                cond = post(x,t,y,mask=mask,encoded=encoded,control_scale=1.)
                a1 = guided_velocity(post,x,t,y,mask,encoded,1.,audit)
                off = base(x,t,y,mask=mask)
                a0 = guided_velocity(post,x,t,y,mask,encoded,0.,audit)
                mixed = guided_velocity(post,x,t,y,mask,encoded,1.5,audit)
                repeat = guided_velocity(post,x,t,y,mask,encoded,1.5,audit)
            finally:
                handle.remove()
                counts["diagnostic_forward_attempts"] = counts["official_forward_attempts"]
            report["alpha1_vs_direct_cond"] = _difference(a1,cond.float())
            report["alpha0_vs_direct_base"] = _difference(a0,off.float())
            report["fp32_combination"] = _difference(mixed,off.float()+1.5*(cond.float()-off.float()))
            report["repeat"] = _difference(mixed,repeat)
            report["same_xt_actual_base_inputs"] = dict(calls=len(observed),
                identical=len(observed)==8 and all(item==observed[0] for item in observed),
                input=observed[0] if observed else None)
            report["single_input_and_encoded_unchanged"] = before == _tree_hash(
                dict(x=x,t=t,y=y,mask=mask,prompts=prompts,encoded=encoded))
            for key in ("alpha1_vs_direct_cond","alpha0_vs_direct_base","fp32_combination","repeat"):
                if not report[key]["bitwise_equal"]: raise RuntimeError("Combination gate failed: "+key)
            if not report["same_xt_actual_base_inputs"]["identical"] or not report["single_input_and_encoded_unchanged"]:
                raise RuntimeError("Branches did not preserve the identical observed inputs")
            counts["sampler_calls"] += 1
            sampling_before = counts["official_forward_attempts"]
            try:
                sampled,stats = guided_euler_sample(post,x,None,"OFF")
                counts["complete_rollouts"] += 1; counts["base_rollouts"] += 1
                report["diagnostic_rollouts"] += 1
            finally:
                counts["sampling_official_forwards"] += counts["official_forward_attempts"]-sampling_before
            result = sampled[0].detach().cpu().numpy()
            report["alpha0_full_vs_historical_base"] = dict(
                bitwise_equal=result.dtype==archived["seed7101"].dtype and np.array_equal(result,archived["seed7101"]),
                max_abs=float(np.max(np.abs(result.astype(np.float64)-archived["seed7101"].astype(np.float64)))),
                sampling=stats,history_raw_sha256=historical_base["raw"]["sha256"])
            if output_dir is not None:
                path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
                report["raw"]=_save_raw(path/"combination_precheck_alpha0.npz",{"seed7101":result})
            if not report["alpha0_full_vs_historical_base"]["bitwise_equal"]:
                raise RuntimeError("Full alpha0 generation does not recover archived Base")
            report["schedules"]={key:dict(alpha=alpha_table(key),
                mean=sum(alpha_table(key))/50,last_alpha=alpha_table(key)[-1],
                nfe=sum(1 if v in (0.,1.) else 2 for v in alpha_table(key))) for key in SETTINGS}
            if (abs(report["schedules"]["A4"]["mean"]-.91)>1e-15
                    or report["schedules"]["A4"]["last_alpha"]!=.1
                    or any(report["schedules"][s]["nfe"]!=EXPECTED_NFE[s] for s in SETTINGS)):
                raise RuntimeError("Integer-clock schedule contract failed")
        report["base_unchanged"] = _assert_base(base)==before_base
        report["adapter_unchanged"] = state_sha256(post.adapter)==before_adapter
        report["bank_unchanged"] = _fingerprint(bank)==bank_hash
        report["rng_unchanged"] = _rng_equal(rng_before,_rng_snapshot())
        counts["condition_encodes"] = post.counts["condition_encoding"]-prior_counts["condition_encoding"]
        report["expected_counts"]=dict(official_forward_attempts=58,official_forward_returns=58,
            diagnostic_forward_attempts=8,sampling_official_forwards=50,complete_rollouts=1,
            condition_encodes=1,optimizer_updates=0)
        if (any(counts[k]!=v for k,v in report["expected_counts"].items())
                or not all(report[k] for k in ("base_unchanged","adapter_unchanged","bank_unchanged","rng_unchanged"))):
            raise RuntimeError("Combination gate counters or asset invariance failed")
        report["status"]="PASS"
    except BaseException as error:
        report.update(status="FAIL",error=repr(error))
        if hasattr(error,"guidance_stats"):report["failed_guidance_stats"]=error.guidance_stats
        error.evaluation_report=report
        raise
    finally:
        counts["condition_encodes"] = post.counts["condition_encoding"]-prior_counts["condition_encoding"]
        report["wrapper_count_delta"]={k:post.counts[k]-prior_counts.get(k,0) for k in post.counts}
        post.train(modes[0]);post.adapter.train(modes[1]);base.train(modes[2])
    return report
