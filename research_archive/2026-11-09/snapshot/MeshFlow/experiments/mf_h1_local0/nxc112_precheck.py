"""Zero-update S32 checks on the real frozen official backbone.

The driver owns model loading and the global forward ledger. This function
returns evidence in memory and never writes another report or checkpoint.
Nonzero test projections belong to a separate disposable adapter.
"""
from __future__ import annotations

from .stable_runtime import stable_execution, assert_stable_runtime

import copy
import math
import torch.nn.functional as F
import traceback
import torch

from .official_wrapper import OfficialWrapper, state_sha256
from .prompt_dataset import collate_samples
from .s32_adapter import S32StaticAdapter


def _difference(expected, actual, atol=0., rtol=0.):
    a, b = expected.detach().double(), actual.detach().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = a-b
    return dict(status="PASS" if finite and torch.allclose(a, b, atol=atol, rtol=rtol) else "FAIL",
                bitwise_equal=bool(torch.equal(expected, actual)), finite=finite,
                max_abs=float(delta.abs().max()) if finite else None,
                rms=float(delta.square().mean().sqrt()) if finite else None,
                relative_rms=float(delta.square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-12)) if finite else None,
                max_relative=float((delta.abs()/a.abs().clamp_min(1e-12)).max()) if finite else None,
                atol=atol, rtol=rtol)


def _gradient(value):
    if value is None:
        return dict(present=False, finite=False, l2=None, max_abs=None)
    value = value.detach().float()
    finite = bool(torch.isfinite(value).all())
    return dict(present=True, finite=finite,
                l2=float(value.double().square().sum().sqrt()) if finite else None,
                max_abs=float(value.abs().max()) if finite else None)


def _hooks(base):
    """Root's global counting hooks are allowed and preserved by identity."""
    return dict(base_pre=list(base._forward_pre_hooks), base_post=list(base._forward_hooks),
        layer_pre=[list(layer._forward_pre_hooks) for layer in base.layers],
        layer_post=[list(layer._forward_hooks) for layer in base.layers])


def run_cpu_checks(adapter, bank):
    """Independent FP32 static-attention formula and symmetry checks, no backbone."""
    from .prompt_dataset import collate_prompts
    clone = copy.deepcopy(adapter).cpu().float().eval()
    clone.call_count = clone.encoder_call_count = 0
    generator = torch.Generator(device='cpu').manual_seed(2026112)
    prompts = collate_prompts([bank['prompts'][(uid,'mixed_2edges_1face')]
                               for uid in bank['parent_uids']], 'cpu')
    if prompts['points'].shape != (2,32,3):
        raise ValueError('CPU formula check requires the two frozen32-point prompts')
    with torch.no_grad():
        for layer in clone.layers:
            layer.output_projection.weight.copy_(torch.randn(layer.output_projection.weight.shape,generator=generator)*.001)
            layer.output_projection.bias.fill_(.001)
        h = torch.randn((2,336,768),generator=generator)
        xt = torch.from_numpy(__import__('numpy').stack([item['coords'] for item in bank['teacher_samples']])).float().reshape(2,336,3)
        t = torch.full((2,),.5)
        encoded = clone.encode(prompts)
        p, groups = prompts['points'],prompts['point_primitive']
        types = prompts['arity'].gather(1,groups)
        bands = torch.arange(6,dtype=torch.float32)
        phase = p.unsqueeze(-1)*(torch.pi*2.**bands)
        features = torch.cat((p,phase.sin().flatten(-2),phase.cos().flatten(-2),
                              torch.stack((types == 2,types == 3),-1).float()),-1)
        e = F.linear(F.silu(F.linear(features,clone.condition_encoder[0].weight,
            clone.condition_encoder[0].bias)),clone.condition_encoder[2].weight,clone.condition_encoder[2].bias)
        count = torch.stack([(groups == j).sum(1) for j in range(3)],1)
        log_prior = -count.gather(1,groups).float().log()
        rows = [dict(name='independent_point_feature_encoder',**_difference(e,encoded.tokens,1e-5,1e-5)),
                dict(name='independent_group_log_prior',**_difference(log_prior,encoded.log_weights))]
        point_order = torch.arange(31,-1,-1)
        reordered = dict(prompts)
        for key in ('points','point_primitive','point_mask'):
            reordered[key] = prompts[key][:,point_order]
        encoded_permuted = clone.encode(reordered)
        primitive_order = torch.tensor([2,0,1])
        regrouped = dict(prompts)
        for key in ('vertices','arity','primitive_mask'):
            regrouped[key] = prompts[key][:,primitive_order]
        regrouped['point_primitive'] = torch.argsort(primitive_order)[groups]
        encoded_regrouped = clone.encode(regrouped)
        rows.extend([dict(name='group_relabel_token_identity',**_difference(encoded.tokens,encoded_regrouped.tokens)),
                     dict(name='group_relabel_prior_identity',**_difference(encoded.log_weights,encoded_regrouped.log_weights))])
        encoded_swapped = clone.encode({key:value.flip(0) for key,value in prompts.items()})
        query_order = torch.arange(335,-1,-1)
        for index,layer in enumerate(clone.layers):
            q = F.linear(F.layer_norm(h,(768,),layer.query_norm.weight,layer.query_norm.bias,layer.query_norm.eps),
                         layer.query_projection.weight,layer.query_projection.bias)
            k = F.linear(e,layer.key_projection.weight,layer.key_projection.bias)
            v = F.linear(e,layer.value_projection.weight,layer.value_projection.bias)
            q = q.reshape(2,336,4,32).transpose(1,2)
            k = k.reshape(2,32,4,32).transpose(1,2)
            v = v.reshape(2,32,4,32).transpose(1,2)
            alpha = (torch.matmul(q,k.transpose(-1,-2))/math.sqrt(32)+log_prior[:,None,None,:]).softmax(-1)
            message = torch.matmul(alpha,v).transpose(1,2).reshape(2,336,128)
            manual = F.linear(message,layer.output_projection.weight,layer.output_projection.bias)
            actual = clone.forward_layer(index,h,xt,t,encoded)
            rows.append(dict(name=f'layer{index}_independent_standard_attention_formula',**_difference(manual,actual,1e-5,1e-5)))
            reordered = clone.forward_layer(index,h,xt,t,encoded_permuted)
            rows.append(dict(name=f'layer{index}_point_permutation',**_difference(actual,reordered,1e-5,1e-5)))
            changed_query = clone.forward_layer(index,h[:,query_order],xt[:,query_order],t,encoded)
            rows.append(dict(name=f'layer{index}_query_permutation',**_difference(actual[:,query_order],changed_query,1e-5,1e-5)))
            changed_geometry = clone.forward_layer(index,h,xt+7.25,t+.125,encoded)
            rows.append(dict(name=f'layer{index}_no_direct_xt_or_time_dependency',**_difference(actual,changed_geometry)))
            different_condition = clone.forward_layer(index,h,xt,t,encoded_swapped)
            delta = _difference(actual,different_condition)
            rows.append(dict(name=f'layer{index}_changed_condition_nonzero',status='PASS' if delta['finite'] and delta['max_abs']>0 else 'FAIL',
                             max_abs=delta['max_abs'],rms=delta['rms']))
    return dict(status='PASS' if all(row['status']=='PASS' for row in rows) else 'FAIL',tests=rows,
                precision='float32',device='cpu',official_forwards=0,optimizer_updates=0,
                condition_encoder_calls=clone.encoder_call_count,adapter_layer_calls=clone.call_count,
                input_note='Real frozen prompts and real112 Xt geometry, fixed local-generator unit-test hidden features; no model or training surrogate.')

def run_prechecks(base, adapter, bank, counters=None):
    """Return PASS/FAIL plus exact actual counts; never call optimizer.step.

    Preconditions: one frozen official FP32 eval backbone with no injection
    wrapper attached; an untouched, seed10, zero-exit S32StaticAdapter; the
    verified two-real-112-parent bank. The caller must check ``status == PASS``.
    Optional counters receive only ``s32_precheck_*`` entries, so the driver's
    actual official-forward hook is never double counted.
    """
    report = dict(status="RUNNING", tests=[], wrapper_runs=[], runtime_entries=[],
        actual_counts=dict(official_forward_attempts=0, official_forward_returns=0,
            wrapper_forward=0, condition_encodes=0, adapter_layers=0,
            backward_attempts=0, backward_completed=0, optimizer_updates=0,
            backbone_updates=0, complete_rollouts=0, phase3_updates=0, direct_adapter_encodes=0),
        numerical_policy=dict(identity_atol=0., identity_rtol=0.,
            nonzero_permutation_atol=0.015625, nonzero_permutation_rtol=0.03125,
            cpu_fp32_formula_and_permutation_atol=1e-5, cpu_fp32_formula_and_permutation_rtol=1e-5,
            note="Symmetry tolerances declared before execution. BF16 raw differences are retained; no failed threshold is relaxed."),
        scope="Full real 112-face teacher interface states; no optimizer, sampler or altered formal training initialization.")
    active_wrapper = None
    helper_handles = []
    before_hooks = _hooks(base)
    initial_adapter_mode = adapter.training
    initial_calls = getattr(adapter, "call_count", 0), getattr(adapter, "encoder_call_count", 0)
    initial_base_hash = initial_adapter_hash = None

    def add(name, value):
        row = dict(name=name, **value)
        report["tests"].append(row)
        if row["status"] != "PASS":
            raise RuntimeError("S32 precheck failed: " + name)
        return row

    def attach(current, label):
        nonlocal active_wrapper
        if active_wrapper is not None:
            raise RuntimeError("Precheck attempted duplicate wrapper injection")
        if any(layer._forward_hooks for layer in base.layers):
            raise RuntimeError("A previous wrapper or observation hook remains on a backbone layer")
        active_wrapper = OfficialWrapper(base, current).eval()
        after = [len(layer._forward_hooks) for layer in base.layers]
        if after != [int(index in (2,5,8,11)) for index in range(12)]:
            raise RuntimeError("Expected exactly one residual hook at each complete block3/6/9/12")
        active_wrapper._precheck_label = label
        return active_wrapper

    def close():
        nonlocal active_wrapper
        if active_wrapper is not None:
            report["wrapper_runs"].append(dict(label=active_wrapper._precheck_label,
                                                counts=dict(active_wrapper.counts)))
            active_wrapper.close()
            active_wrapper = None

    def gradient_probe(wrapper, x, t, y, mask, prompts, velocity, label, nonzero):
        current = wrapper.adapter
        current.zero_grad(set_to_none=True)
        residuals, encoder = {}, {}
        handles = []

        def capture_residual(index):
            def hook(module, args, result):
                result.retain_grad()
                residuals[index] = result
            return hook

        def capture_encoder(module, args, result):
            encoder.update(calls=encoder.get("calls", 0)+1, shape=list(result.shape),
                           dtype=str(result.dtype), requires_grad=result.requires_grad)

        try:
            for index, layer in enumerate(current.layers):
                handles.append(layer.register_forward_hook(capture_residual(index)))
            handles.append(current.condition_encoder.register_forward_hook(capture_encoder))
            wrapper.train()
            with torch.enable_grad(), stable_execution(base, current):
                report["runtime_entries"].append(dict(label=label,
                    facts=assert_stable_runtime(base, current, coordinates=x)))
                prediction = wrapper(x, t, y, mask=mask, prompts=prompts)
                loss = (prediction.float()-velocity).square()[mask].mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite zero-update S32 gradient-probe loss")
                report["actual_counts"]["backward_attempts"] += 1
                loss.backward()
                if x.device.type == "cuda":
                    torch.cuda.synchronize(x.device)
                report["actual_counts"]["backward_completed"] += 1
            gradients = {name:_gradient(parameter.grad) for name,parameter in current.named_parameters()}
            residual_grads = {f"after_block{number}":_gradient(residuals[index].grad)
                              for index,number in enumerate((3,6,9,12))}
            required = [f"layers.{index}.output_projection.weight" for index in range(4)]
            if nonzero:
                required += ["condition_encoder.0.weight", "condition_encoder.2.weight"]
                required += [f"layers.{index}.{projection}.weight" for index in range(4)
                             for projection in ("query_projection", "key_projection", "value_projection")]
            active = lambda item: item["finite"] and item["max_abs"] is not None and item["max_abs"] > 0
            passed = (all(active(gradients[name]) for name in required)
                      and all(active(value) for value in residual_grads.values())
                      and encoder.get("calls") == 1 and encoder.get("requires_grad") is True
                      and all(parameter.grad is None for parameter in base.parameters()))
            upstream = [value["max_abs"] for name,value in gradients.items()
                        if "output_projection" not in name and value["max_abs"] is not None]
            add(label, dict(status="PASS" if passed else "FAIL", loss=float(loss.detach()),
                required_parameter_gradients={name:gradients[name] for name in required},
                first_injection_through_frozen_blocks4_to12=residual_grads["after_block3"],
                all_injection_output_gradients=residual_grads,
                encoder_actual_call=encoder, backbone_gradients_absent=all(p.grad is None for p in base.parameters()),
                upstream_max_gradient=max(upstream) if upstream else None,
                zero_exit_first_upstream_gradient_can_be_zero=not nonzero, optimizer_updates=0))
        finally:
            for handle in handles:
                handle.remove()
            current.zero_grad(set_to_none=True)
            wrapper.eval()

    try:
        if not isinstance(adapter, S32StaticAdapter) or adapter.arm != "S32_static":
            raise ValueError("S32 precheck requires the new explicit static32 adapter")
        if any(p.requires_grad or p.grad is not None for p in base.parameters()) or base.training:
            raise ValueError("The official backbone must be frozen, eval, and free of parameter gradients")
        if any(layer._forward_hooks or layer._forward_pre_hooks for layer in base.layers):
            raise ValueError("Close all old wrappers and diagnostic hooks before S32 precheck")
        if any(p.grad is not None for p in adapter.parameters()):
            raise ValueError("Formal initialization must not carry previous gradients")
        if any(p.dtype != torch.float32 for p in adapter.parameters()):
            raise ValueError("Formal adapter parameters must remain FP32")
        for layer in adapter.layers:
            if (hasattr(layer, "dynamic_mlp") or layer.arm != "S_static"
                    or torch.count_nonzero(layer.output_projection.weight)
                    or torch.count_nonzero(layer.output_projection.bias)):
                raise ValueError("Require the exact static architecture with zero exits")
        report["architecture"] = adapter.architecture_summary()
        add("parameter_inventory", dict(status="PASS" if report["architecture"]["total_parameters"] == 950144 else "FAIL",
                                        **report["architecture"]))
        report["cpu_fp32_checks"] = run_cpu_checks(adapter, bank)
        if report["cpu_fp32_checks"]["status"] != "PASS":
            raise RuntimeError("CPU FP32 formula/permutation audit failed")
        initial_base_hash, initial_adapter_hash = state_sha256(base), state_sha256(adapter)
        report.update(base_sha256_before=initial_base_hash, formal_adapter_sha256_before=initial_adapter_hash)
        device = next(base.parameters()).device
        if device.type != "cuda":
            raise ValueError("This official precheck is for the audited CUDA BF16 path")
        samples = [dict(item, t=.5) for item in bank["teacher_samples"]]
        if len(samples) != 2 or len(set(item["parent_uid"] for item in samples)) != 2:
            raise ValueError("Use exactly two distinct true112 teacher parents")
        for item in samples:
            if int(item["num_faces"]) != 112 or tuple(item["coords"].shape) != (112,9):
                raise ValueError("No truncation, replication or padding may create the 112-face state")
            item["prompt"] = bank["prompts"][(item["parent_uid"], "mixed_2edges_1face")]
        batch = collate_samples(samples, device)
        mask, y, t, prompts = batch["masks"], batch["num_faces"], batch["t"], batch["prompts"]
        x = (1-t[:,None,None])*batch["noise"] + t[:,None,None]*batch["tokens"]
        velocity = batch["tokens"]-batch["noise"]
        if x.shape != (2,112,9) or mask.shape != (2,112) or not mask.all() or not (y == 112).all():
            raise ValueError("True112 identity-channel input shape/mask check failed")
        report["input"] = dict(parent_uids=[item["parent_uid"] for item in samples], shape=list(x.shape),
                               y=y.cpu().tolist(), valid_faces=mask.sum(1).cpu().tolist(), teacher_time=.5)
        helper_handles.append(base.register_forward_pre_hook(lambda module,args:
            report["actual_counts"].__setitem__("official_forward_attempts",report["actual_counts"]["official_forward_attempts"]+1)))
        helper_handles.append(base.register_forward_hook(lambda module,args,result:
            report["actual_counts"].__setitem__("official_forward_returns",report["actual_counts"]["official_forward_returns"]+1)))
        # The formal adapter is not seeded or reinitialized here. Test-only
        # randomness is isolated even if a future helper starts using RNG.
        with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
            with torch.no_grad(), stable_execution(base, adapter):
                report["runtime_entries"].append(dict(label="zero_exit_and_off",
                    facts=assert_stable_runtime(base, adapter, coordinates=x)))
                reference = base(x, t, y, mask=mask)
                wrapper = attach(adapter, "formal_zero_exit")
                off = wrapper(x, t, y, mask=mask, prompts=prompts, control_scale=0.)
                zero = wrapper(x, t, y, mask=mask, prompts=prompts)
                repeat = wrapper(x, t, y, mask=mask, prompts=prompts)
                add("control_off_exact_base", _difference(reference, off))
                add("zero_exit_exact_base", _difference(reference, zero))
                add("stable_same_input_repeat", _difference(zero, repeat))
                report["actual_counts"]["direct_adapter_encodes"] += 1
                encoded = adapter.encode(prompts)
                group = prompts["point_primitive"]
                prior_sums = []
                for j in range(3):
                    prior_sums.append((encoded.log_weights.exp()*(group == j)).sum(1))
                prior_sums = torch.stack(prior_sums,1)
                add("32_tokens_equal_primitive_prior", dict(
                    status="PASS" if encoded.tokens.shape == (2,32,128) and torch.allclose(prior_sums,torch.ones_like(prior_sums),atol=1e-6,rtol=0) else "FAIL",
                    token_shape=list(encoded.tokens.shape), prior_sums=prior_sums.cpu().tolist(),
                    group_counts=[(group == j).sum(1).cpu().tolist() for j in range(3)],
                    direct_adapter_encodes_outside_wrapper=1))
            gradient_probe(wrapper,x,t,y,mask,prompts,velocity,"zero_exit_backward_no_update",False)
            close()
            test_adapter = copy.deepcopy(adapter)
            test_adapter.zero_grad(set_to_none=True)
            test_generator = torch.Generator(device="cpu").manual_seed(2026112)
            with torch.no_grad():
                for layer in test_adapter.layers:
                    layer.output_projection.weight.copy_(torch.randn(layer.output_projection.weight.shape,
                        generator=test_generator,dtype=torch.float32).to(device)*.001)
                    layer.output_projection.bias.fill_(.001)
            report["disposable_nonzero_adapter_sha256"] = state_sha256(test_adapter)
            wrapper = attach(test_adapter,"disposable_nonzero_exit")
            with torch.no_grad(), stable_execution(base, test_adapter):
                report["runtime_entries"].append(dict(label="nonzero_permutations",
                    facts=assert_stable_runtime(base, test_adapter, coordinates=x)))
                value = wrapper(x,t,y,mask=mask,prompts=prompts)
                repeat = wrapper(x,t,y,mask=mask,prompts=prompts)
                change = _difference(reference,value)
                add("nonvacuous_nonzero_test_residual", dict(status="PASS" if change["finite"] and change["max_abs"] > 0 else "FAIL",
                    max_output_change=change["max_abs"], rms_output_change=change["rms"]))
                add("nonzero_stable_same_input_repeat", _difference(value,repeat))
                swapped = {key:field.flip(0) for key,field in prompts.items()}
                swapped_value = wrapper(x,t,y,mask=mask,prompts=swapped)
                conditional_change = _difference(value,swapped_value)
                add("nonzero_changed_condition_participates", dict(status="PASS" if conditional_change["finite"] and conditional_change["max_abs"] > 0 else "FAIL",
                    max_output_change=conditional_change["max_abs"], rms_output_change=conditional_change["rms"]))
                point_order = torch.arange(31,-1,-1,device=device)
                reordered = dict(prompts)
                for key in ("points","point_primitive","point_mask"):
                    reordered[key] = prompts[key][:,point_order]
                changed = wrapper(x,t,y,mask=mask,prompts=reordered)
                add("nonzero_point_token_permutation",_difference(value,changed,0.015625,0.03125))
                primitive_order = torch.tensor([2,0,1],device=device)
                inverse = torch.argsort(primitive_order)
                regrouped = dict(prompts)
                for key in ("vertices","arity","primitive_mask"):
                    regrouped[key] = prompts[key][:,primitive_order]
                regrouped["point_primitive"] = inverse[prompts["point_primitive"]]
                changed = wrapper(x,t,y,mask=mask,prompts=regrouped)
                add("nonzero_group_relabel_without_identity_embedding",_difference(value,changed))
                faces = torch.arange(111,-1,-1,device=device)
                changed = wrapper(x[:,faces].contiguous(),t,y,mask=mask[:,faces],prompts=prompts)
                add("nonzero_face_permutation",_difference(value[:,faces],changed,0.015625,0.03125))
                corners = torch.tensor([[2,0,1],[1,0,2]],device=device)[torch.arange(112,device=device)%2]
                gather = corners[None,:,:,None].expand(2,-1,-1,3)
                changed_x = x.reshape(2,112,3,3).gather(2,gather).reshape_as(x)
                expected = value.reshape(2,112,3,3).gather(2,gather).reshape_as(value)
                changed = wrapper(changed_x,t,y,mask=mask,prompts=prompts)
                add("nonzero_within_face_corner_permutation",_difference(expected,changed,0.015625,0.03125))
                off = wrapper(x,t,y,mask=mask,prompts=prompts,control_scale=0.)
                add("nonzero_control_off_exact_base",_difference(reference,off))
            gradient_probe(wrapper,x,t,y,mask,prompts,velocity,"nonzero_backward_through_frozen_backbone",True)
            close()
            with torch.no_grad(), stable_execution(base, adapter):
                cleaned = base(x,t,y,mask=mask)
                add("base_restored_after_all_wrapper_hooks_removed",_difference(reference,cleaned))
                base_corner = base(changed_x,t,y,mask=mask)
                base_corner_expected = reference.reshape(2,112,3,3).gather(2,gather).reshape_as(reference)
                add("base_own_corner_permutation_roundoff",_difference(base_corner_expected,base_corner,0.015625,0.03125))
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL",error=repr(error),traceback=traceback.format_exc())
    finally:
        close()
        for handle in helper_handles:
            handle.remove()
        adapter.zero_grad(set_to_none=True)
        adapter.train(initial_adapter_mode)
        adapter.call_count, adapter.encoder_call_count = initial_calls
        base.eval()
        if initial_adapter_hash is not None:
            report["formal_adapter_sha256_after"] = state_sha256(adapter)
            report["formal_initialization_unchanged"] = report["formal_adapter_sha256_after"] == initial_adapter_hash
            if not report["formal_initialization_unchanged"]:
                report["status"] = "FAIL"
        if initial_base_hash is not None:
            report["base_sha256_after"] = state_sha256(base)
            report["base_unchanged"] = report["base_sha256_after"] == initial_base_hash
            report["backbone_gradients_absent"] = all(parameter.grad is None for parameter in base.parameters())
            if not report["base_unchanged"] or not report["backbone_gradients_absent"]:
                report["status"] = "FAIL"
        report["hooks_restored_exactly"] = _hooks(base) == before_hooks
        if not report["hooks_restored_exactly"]:
            report["status"] = "FAIL"
        for record in report["wrapper_runs"]:
            counts = record["counts"]
            report["actual_counts"]["wrapper_forward"] += counts["wrapper_forward"]
            report["actual_counts"]["condition_encodes"] += counts["condition_encoding"]
            report["actual_counts"]["adapter_layers"] += counts["adapter_layer"]
        report["actual_counts"]["total_condition_encoder_calls"] = (
            report["actual_counts"]["condition_encodes"] + report["actual_counts"]["direct_adapter_encodes"])
        if counters is not None:
            for key,value in report["actual_counts"].items():
                name = "s32_precheck_"+key
                counters[name] = counters.get(name,0)+value
    return report
