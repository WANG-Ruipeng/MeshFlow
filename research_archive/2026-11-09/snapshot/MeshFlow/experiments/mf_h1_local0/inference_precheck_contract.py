"""Pre-FFN gates with Base reference observations classified by the contract.

This independent copy preserves the original precheck and its failed run.
All Pre/off/zero/repeat/gradient/lifecycle gates and tolerances are unchanged.
Only the three Base reference roundoff measurements are observations: their
original same-tolerance PASS/FAIL remains visible, while nonfinite Base values
still stop the check. The root owns scheduling, counts, and any rerun record.
"""
from __future__ import annotations

from .stable_runtime import stable_execution, assert_stable_runtime

import copy
import random
import traceback
from pathlib import Path
import numpy as np
import torch

from .official_wrapper import OfficialWrapper, state_sha256
from .prompt_dataset import collate_samples, file_sha256
from .nxc112_precheck import run_cpu_checks, _difference, _gradient
from .inference_pre_wrapper import PreFFNWrapper, PreResidualObserver, active_pre_owner, LAYERS

SOURCE_STEP0 = Path(__file__).resolve().parent / "outputs/nxC_112_sandbox_20260928_173427/stage_b/checkpoints/adapter_step0000.pt"


def _hook_identity(base):
    return dict(base_pre=list(base._forward_pre_hooks),base_post=list(base._forward_hooks),
        blocks=[dict(pre=list(layer._forward_pre_hooks),post=list(layer._forward_hooks),
            norm2=list(layer.norm2._forward_pre_hooks),
            method_instance="_forward_v3" in layer.__dict__,
            method_function=id(getattr(layer._forward_v3,"__func__",None))) for layer in base.layers])


def _rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state().clone(),
                cuda=[s.clone() for s in torch.cuda.get_rng_state_all()])


def _same_rng(a,b):
    return (a["python"]==b["python"] and a["numpy"][0]==b["numpy"][0]
        and np.array_equal(a["numpy"][1],b["numpy"][1]) and a["numpy"][2:]==b["numpy"][2:]
        and torch.equal(a["cpu"],b["cpu"]) and len(a["cuda"])==len(b["cuda"])
        and all(torch.equal(x,y) for x,y in zip(a["cuda"],b["cuda"])))


def run_prechecks(base,adapter,bank,counters=None):
    """Preconditions: no live wrapper; strict old Post step0 adapter already loaded.

    Returns a JSON-safe dict. Uses full real112 teacher states and no optimizer.
    Optional counters receive only ``pre_injection_precheck_`` names.
    """
    report=dict(status="RUNNING",tests=[],baseline_observations=[],wrapper_runs=[],runtime_entries=[],
        actual_counts=dict(official_forward_attempts=0,official_forward_returns=0,
            backward_attempts=0,backward_completed=0,wrapper_forward=0,condition_encodes=0,
            adapter_layers=0,pre_site_visits=0,complete_rollouts=0,optimizer_updates=0,
            backbone_updates=0,phase3_updates=0),
        numerical_policy=dict(zero_off_repeat_atol=0.,zero_off_repeat_rtol=0.,
            bf16_permutation_atol=.015625,bf16_permutation_rtol=.03125,
            cpu_fp32_formula_atol=1e-5,cpu_fp32_formula_rtol=1e-5,
            mask_scope="112 slots retained; only separate interface probes mark the last two faces invalid."),
        parameter_change="None: same strict old Post step0 S32 tensors; only query/insertion location changes.")
    current=None;handles=[];initial_hash=None;base_hash=None
    old_mode=adapter.training;old_calls=adapter.call_count,adapter.encoder_call_count
    hook_before=_hook_identity(base);rng_before=_rng()

    def add(name,value):
        row=dict(name=name,**value);report["tests"].append(row)
        if row["status"]!="PASS":raise RuntimeError("Pre-FFN gate failed: "+name)
        return row

    def observe_base(name,value):
        # Preserve the original comparison result; only finiteness is a gate.
        row=dict(name=name,**value,same_atol=value["atol"],same_rtol=value["rtol"],
            within_tolerance=value["status"],classification="BASE_REFERENCE_OBSERVATION",
            finite_gate_status="PASS" if value["finite"] else "FAIL")
        report["baseline_observations"].append(row)
        if not value["finite"]:raise RuntimeError("Nonfinite Base reference: "+name)
        return row

    def attach(a,label):
        nonlocal current
        if current is not None:raise RuntimeError("A precheck controller remains live")
        current=PreFFNWrapper(base,a).eval();current._precheck_label=label
        return current

    def close():
        nonlocal current
        if current is not None:
            report["wrapper_runs"].append(dict(label=current._precheck_label,counts=dict(current.counts)))
            current.close();current=None

    def restored(wrapper):
        wrapper._assert_original_methods()
        return wrapper._context is None and not wrapper._in_forward and not wrapper._saved_methods

    def backward(wrapper,x,t,y,mask,prompts,target,label,nonzero):
        observed={};probe_handles=[];a=wrapper.adapter;a.zero_grad(set_to_none=True)
        def capture(index):
            def hook(module,args,value):
                value.retain_grad();observed[index]=value
            return hook
        try:
            for index,layer in enumerate(a.layers):probe_handles.append(layer.register_forward_hook(capture(index)))
            wrapper.train()
            with torch.enable_grad(),stable_execution(base,a):
                report["runtime_entries"].append(dict(label=label,facts=assert_stable_runtime(base,a,coordinates=x)))
                prediction=wrapper(x,t,y,mask=mask,prompts=prompts,control_scale=1.)
                loss=(prediction.float()-target).square()[mask].mean()
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite zero-update backward probe")
                report["actual_counts"]["backward_attempts"]+=1
                loss.backward();torch.cuda.synchronize(x.device)
                report["actual_counts"]["backward_completed"]+=1
            grads={name:_gradient(p.grad) for name,p in a.named_parameters()}
            required=[f"layers.{i}.output_projection.weight" for i in range(4)]
            if nonzero:
                required += ["condition_encoder.0.weight","condition_encoder.2.weight"]
                required += [f"layers.{i}.{p}.weight" for i in range(4)
                    for p in ("query_projection","key_projection","value_projection")]
            residual_grads={f"pre_block{n}":_gradient(observed[i].grad) for i,n in enumerate(LAYERS)}
            ok=lambda row:row["finite"] and row["max_abs"] is not None and row["max_abs"]>0
            add(label,dict(status="PASS" if all(ok(grads[n]) for n in required)
                and all(ok(v) for v in residual_grads.values()) and all(p.grad is None for p in base.parameters()) else "FAIL",
                loss=float(loss.detach()),required_gradients={n:grads[n] for n in required},
                injection_output_gradients=residual_grads,
                earliest_gradient_path="Block3 Pre residual through its FFN and subsequent frozen blocks4-12",
                zero_exit_upstream_zero_is_expected=not nonzero,optimizer_updates=0))
        finally:
            for handle in probe_handles:handle.remove()
            a.zero_grad(set_to_none=True);wrapper.eval()

    try:
        if base.training or any(p.requires_grad or p.grad is not None for p in base.parameters()):
            raise ValueError("Official backbone must remain frozen/eval/gradient-free")
        if active_pre_owner(base) is not None or any(layer._forward_hooks for layer in base.layers):
            raise ValueError("Close all old insertion wrappers before Pre checks")
        if adapter.arm!="S32_static" or sum(p.numel() for p in adapter.parameters())!=950144:
            raise ValueError("Exact existing S32 architecture required")
        if any(p.grad is not None or p.dtype!=torch.float32 for p in adapter.parameters()):
            raise ValueError("Formal step0 must have FP32 parameters and no previous gradients")
        saved=torch.load(SOURCE_STEP0,map_location="cpu",weights_only=True,mmap=True)
        source=saved["adapter"];state=adapter.state_dict()
        equal=(saved["step"]==0 and saved["arm"]=="S32_static" and state.keys()==source.keys()
               and all(v.shape==source[k].shape and v.dtype==source[k].dtype and torch.equal(v.detach().cpu(),source[k]) for k,v in state.items()))
        add("strict_old_post_step0_tensor_identity",dict(status="PASS" if equal else "FAIL",
            path=str(SOURCE_STEP0),sha256=file_sha256(SOURCE_STEP0),tensors=len(source),parameters=950144))
        del saved,source,state
        if any(torch.count_nonzero(p) for name,p in adapter.named_parameters() if ".output_projection." in name):
            raise ValueError("The supplied old step0 is not zero-exit")
        report["cpu_fp32_checks"]=run_cpu_checks(adapter,bank)
        if report["cpu_fp32_checks"]["status"]!="PASS":raise RuntimeError("Existing static attention CPU formula gate failed")
        initial_hash=state_sha256(adapter);base_hash=state_sha256(base)
        report.update(formal_adapter_sha256_before=initial_hash,base_sha256_before=base_hash)
        device=next(base.parameters()).device
        if device.type!="cuda":raise ValueError("Actual Pre checks require the audited stable CUDA BF16 path")
        samples=[dict(item,t=.5,prompt=bank["prompts"][(item["parent_uid"],"mixed_2edges_1face")]) for item in bank["teacher_samples"]]
        batch=collate_samples(samples,device)
        y,mask,t,prompts=batch["num_faces"],batch["masks"],batch["t"],batch["prompts"]
        x=(1-t[:,None,None])*batch["noise"]+t[:,None,None]*batch["tokens"]
        target=batch["tokens"]-batch["noise"]
        if x.shape!=(2,112,9) or not mask.all() or not (y==112).all():raise ValueError("Both real teacher inputs must be full112 faces")
        x_before=x.detach().clone();prompt_before={k:v.detach().clone() for k,v in prompts.items()}
        report["input"]=dict(shape=list(x.shape),valid_faces=mask.sum(1).cpu().tolist(),y=y.cpu().tolist(),t=.5)
        handles.append(base.register_forward_pre_hook(lambda m,a:report["actual_counts"].__setitem__("official_forward_attempts",report["actual_counts"]["official_forward_attempts"]+1)))
        handles.append(base.register_forward_hook(lambda m,a,o:report["actual_counts"].__setitem__("official_forward_returns",report["actual_counts"]["official_forward_returns"]+1)))
        with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
            with torch.no_grad(),stable_execution(base,adapter):
                report["runtime_entries"].append(dict(label="zero_exit",facts=assert_stable_runtime(base,adapter,coordinates=x)))
                reference=base(x,t,y,mask=mask)
                w=attach(adapter,"formal_old_step0")
                off=w(x,t,y,mask=mask,prompts=None,encoded=None,control_scale=1.)
                with PreResidualObserver(w) as observer:
                    with observer.capture("zero",mask):zero=w(x,t,y,mask=mask,prompts=prompts)
                    report["zero_position_observer"]=observer.report()
                repeat=w(x,t,y,mask=mask,prompts=prompts)
                add("off_exact_original_base",_difference(reference,off))
                add("zero_exact_original_base",_difference(reference,zero))
                add("zero_same_input_repeat",_difference(zero,repeat))
                add("zero_four_true_pre_sites",dict(status="PASS" if len(report["zero_position_observer"]["captures"][0]["layers"])==4 and restored(w) else "FAIL"))
            backward(w,x,t,y,mask,prompts,target,"zero_exit_backward_no_update",False)
            close()
            test=copy.deepcopy(adapter);test.zero_grad(set_to_none=True)
            generator=torch.Generator(device="cpu").manual_seed(2026092818)
            with torch.no_grad():
                for layer in test.layers:
                    layer.output_projection.weight.copy_(torch.randn(layer.output_projection.weight.shape,generator=generator).to(device)*.001)
                    layer.output_projection.bias.fill_(.001)
            report["test_nonzero_adapter_sha256"]=state_sha256(test)
            w=attach(test,"disposable_nonzero")
            with torch.no_grad(),stable_execution(base,test):
                report["runtime_entries"].append(dict(label="nonzero",facts=assert_stable_runtime(base,test,coordinates=x)))
                with PreResidualObserver(w) as observer:
                    before=w.counts["adapter_layer"]
                    with observer.capture("correct",mask):value=w(x,t,y,mask=mask,prompts=prompts)
                    change=_difference(reference,value)
                    add("nonvacuous_nonzero_residual",dict(status="PASS" if change["finite"] and change["max_abs"]>0 else "FAIL",difference=change))
                    add("exactly_four_pre_injections",dict(status="PASS" if w.counts["adapter_layer"]-before==4 else "FAIL",actual=w.counts["adapter_layer"]-before))
                    swapped={k:v.flip(0) for k,v in prompts.items()}
                    with observer.capture("swapped",mask):other=w(x,t,y,mask=mask,prompts=swapped)
                    report["nonzero_position_observer"]=observer.report()
                    report["effective_correct_swapped"]=observer.compare_effective("correct","swapped")
                conditional=_difference(value,other)
                add("changed_condition_reaches_output",dict(status="PASS" if conditional["finite"] and conditional["max_abs"]>0 else "FAIL",difference=conditional))
                with PreResidualObserver(w,retain_effective=False) as observer:
                    with observer.capture("with_observer",mask):observed=w(x,t,y,mask=mask,prompts=prompts)
                unobserved=w(x,t,y,mask=mask,prompts=prompts)
                add("observer_no_change_and_stable_repeat",_difference(observed,unobserved))
                add("nonzero_original_repeat",_difference(value,unobserved))
                point_order=torch.arange(31,-1,-1,device=device);reordered=dict(prompts)
                for key in ("points","point_primitive","point_mask"):reordered[key]=prompts[key][:,point_order]
                add("point_permutation",_difference(value,w(x,t,y,mask=mask,prompts=reordered),.015625,.03125))
                faces=torch.arange(111,-1,-1,device=device)
                add("face_permutation",_difference(value[:,faces],w(x[:,faces].contiguous(),t,y,mask=mask[:,faces],prompts=prompts),.015625,.03125))
                corner_order=torch.tensor([[2,0,1],[1,0,2]],device=device)[torch.arange(112,device=device)%2]
                gather=corner_order[None,:,:,None].expand(2,-1,-1,3)
                corner_x=x.reshape(2,112,3,3).gather(2,gather).reshape_as(x)
                expected=value.reshape(2,112,3,3).gather(2,gather).reshape_as(value)
                add("corner_permutation",_difference(expected,w(corner_x,t,y,mask=mask,prompts=prompts),.015625,.03125))
                partial_mask=mask.clone();partial_mask[:,-2:]=False
                poisoned=x.clone();poisoned[:,-2:]=7.25
                masked=w(x,t,y,mask=partial_mask,prompts=prompts)
                masked_poison=w(poisoned,t,y,mask=partial_mask,prompts=prompts)
                add("masked_coordinates_do_not_leak",_difference(masked[partial_mask],masked_poison[partial_mask],.015625,.03125))
                off=w(x,t,y,mask=mask,prompts=None,encoded=None,control_scale=1.)
                add("nonzero_off_exact_base",_difference(reference,off))
                def fail_at_norm2(module,args):raise RuntimeError("EXPECTED_PRECHECK_EXCEPTION_AT_NORM2")
                failure=base.layers[5].norm2.register_forward_pre_hook(fail_at_norm2)
                caught=False
                try:
                    try:w(x,t,y,mask=mask,prompts=prompts)
                    except RuntimeError as error:
                        if str(error)!="EXPECTED_PRECHECK_EXCEPTION_AT_NORM2":raise
                        caught=True
                finally:failure.remove()
                add("expected_exception_restores_methods_and_context",dict(status="PASS" if caught and restored(w) else "FAIL",expected_exception=True,actual_backbone_attempt_counted=True))
                add("next_forward_recovers_exactly",_difference(value,w(x,t,y,mask=mask,prompts=prompts)))
            backward(w,x,t,y,mask,prompts,target,"nonzero_backward_through_frozen_layers",True)
            # These rejected ownership attempts execute no official forward.
            rejected_pre=False
            try:
                unexpected=PreFFNWrapper(base,test)
                unexpected.close()
            except RuntimeError:rejected_pre=True
            add("second_pre_owner_rejected",dict(status="PASS" if rejected_pre else "FAIL"))
            post=OfficialWrapper(base,test).eval()
            rejected_stack=False
            try:
                try:w(x,t,y,mask=mask,prompts=prompts)
                except RuntimeError as error:
                    rejected_stack="Post and Pre" in str(error)
            finally:post.close()
            add("post_pre_execution_overlap_rejected",dict(status="PASS" if rejected_stack and restored(w) else "FAIL"))
            close()
            post=OfficialWrapper(base,test).eval();rejected_constructor=False
            try:
                try:
                    unexpected=PreFFNWrapper(base,test);unexpected.close()
                except RuntimeError:rejected_constructor=True
            finally:post.close()
            add("pre_construction_while_post_live_rejected",dict(status="PASS" if rejected_constructor else "FAIL"))
            with torch.no_grad(),stable_execution(base,adapter):
                add("original_base_after_close",_difference(reference,base(x,t,y,mask=mask)))
                base_faces=base(x[:,faces].contiguous(),t,y,mask=mask[:,faces])
                observe_base("base_own_face_permutation_roundoff",_difference(reference[:,faces],base_faces,.015625,.03125))
                base_corners=base(corner_x,t,y,mask=mask)
                base_expected=reference.reshape(2,112,3,3).gather(2,gather).reshape_as(reference)
                observe_base("base_own_corner_permutation_roundoff",_difference(base_expected,base_corners,.015625,.03125))
                base_mask=base(x,t,y,mask=partial_mask);base_poison=base(poisoned,t,y,mask=partial_mask)
                observe_base("base_own_mask_roundoff",_difference(base_mask[partial_mask],base_poison[partial_mask],.015625,.03125))
            add("input_tensors_unchanged",dict(status="PASS" if torch.equal(x,x_before) and all(torch.equal(v,prompt_before[k]) for k,v in prompts.items()) else "FAIL"))
        report["status"]="PASS"
    except BaseException as error:
        report.update(status="FAIL",error=repr(error),traceback=traceback.format_exc())
    finally:
        close()
        for handle in handles:handle.remove()
        adapter.zero_grad(set_to_none=True);adapter.train(old_mode)
        adapter.call_count,adapter.encoder_call_count=old_calls;base.eval()
        if initial_hash is not None:
            report["formal_adapter_sha256_after"]=state_sha256(adapter)
            report["formal_initialization_unchanged"]=report["formal_adapter_sha256_after"]==initial_hash
            if not report["formal_initialization_unchanged"]:report["status"]="FAIL"
        if base_hash is not None:
            report["base_sha256_after"]=state_sha256(base)
            report["base_unchanged"]=report["base_sha256_after"]==base_hash
            report["backbone_gradients_absent"]=all(p.grad is None for p in base.parameters())
            if not report["base_unchanged"] or not report["backbone_gradients_absent"]:report["status"]="FAIL"
        report["hooks_and_methods_restored"]=_hook_identity(base)==hook_before and active_pre_owner(base) is None
        report["rng_unchanged"]=_same_rng(rng_before,_rng())
        if not report["hooks_and_methods_restored"] or not report["rng_unchanged"]:report["status"]="FAIL"
        for item in report["wrapper_runs"]:
            for key,source in (("wrapper_forward","wrapper_forward"),("condition_encodes","condition_encoding"),
                               ("adapter_layers","adapter_layer"),("pre_site_visits","pre_site_visits")):
                report["actual_counts"][key]+=item["counts"][source]
        if counters is not None:
            for key,value in report["actual_counts"].items():
                target="pre_injection_precheck_"+key;counters[target]=counters.get(target,0)+value
    return report
