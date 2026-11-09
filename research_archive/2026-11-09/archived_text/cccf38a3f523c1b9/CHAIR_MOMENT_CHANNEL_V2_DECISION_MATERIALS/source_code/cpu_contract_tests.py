"""V2 stream, denominator, checkpoint and recovery contracts; no real model.

One actual CPU eight-sample preprocessing batch is compared between both arms.
The only backward differentiates a scalar masked-loss unit test, never a model.
No optimizer is instantiated or stepped. No CUDA, generation, or training.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from native_t1.training import masked_fm_loss
from native_t1.portable_checkpoint import config_sha256
from native_t1.checkpoint import DEFAULT_CONFIG
from experiments.mf_h1_local0.control_mvp48.train_io import tree_hash
from experiments.mf_h1_local0.chair_domain_control_v1.checkpoint import tensor_state_hash
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.checkpoint import validate as validate_hybrid
from . import checkpoint
from .common import SOURCE, CHAIR, HYBRID, OUT, START, START_SHA, atomic_json, read_json, json_hash, digest
from .stream import MomentStream, make_plan, prepare_training_plan, domain_seed
from .training import batch_contract, validate_recovery, validate_training_protocol, LOG_COLUMNS


def run(out=OUT):
    out = Path(out)/"cpu_contracts"
    out.mkdir(parents=True, exist_ok=True)
    attempt = out/"attempt.json"
    if attempt.exists():
        raise RuntimeError("Preserve prior CPU attempt before explicitly authorizing a revised check")
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA must remain uninitialized in CPU contracts")
    started = time.perf_counter()
    result = dict(status="RUNNING", model_forwards=0, model_backwards=0,
        optimizer_updates=0, scalar_loss_backwards=0, CPU_OT_attempts=0,
        CPU_OT_returns=0, CPU_OT_cache_hits=0, checks=[],
        source_sha256={str(Path(__file__).with_name(name)):digest(Path(__file__).with_name(name))
            for name in ("checkpoint.py", "stream.py", "training.py", "cpu_contract_tests.py")})
    atomic_json(attempt, result)

    def check(name, value, detail=None):
        result["checks"].append(dict(name=name, status="PASS" if value else "FAIL", detail=detail))
        atomic_json(out/"progress.json", result)
        if not value:
            raise AssertionError(name)

    def rejects(name, fn):
        try:
            fn()
        except (AssertionError, ValueError, RuntimeError):
            check(name, True)
        else:
            check(name, False)

    try:
        manifest = read_json(CHAIR/"task_manifest.json")
        fixture = out/"stream_fixture"
        plan_record = prepare_training_plan(SOURCE, CHAIR, fixture)
        plan = read_json(fixture/"train_plan.json")
        check("plan_1000x8_seed_and_cursor", plan["seed"] == 20261006
            and len(plan["records"]) == 8000
            and all((row["moment_step"], row["conditional_total_step"], row["sample_index"])
                == (i//8+1, 5001+i//8, i) for i, row in enumerate(plan["records"])), plan_record)
        check("all_32_parents_250_exposures", len(plan["object_counts"]) == 32
            and set(plan["object_counts"].values()) == {250})
        check("every_batch_exactly_four_unit_alpha", all(sum(row["alpha"] == 1.
            for row in plan["records"][i:i+8]) == 4 for i in range(0, 8000, 8)))
        check("reproduced_schedule", make_plan(manifest) == {k:v for k,v in plan.items()
            if k != "source_task_manifest_sha256"})
        check("separate_rng_domains", len({domain_seed(domain, 0) for domain in
            ("gaussian", "epsilon", "time", "permutation", "parent_schedule", "task_schedule")}) == 6)
        rejects("reject_unregistered_seed", lambda: make_plan(manifest, seed=20261005))
        first = MomentStream(SOURCE, CHAIR, fixture, "H_CONT")
        second = MomentStream(SOURCE, CHAIR, fixture, "MOMENT_XY")
        samples = first.next_effective_batch()
        paired = second.next_effective_batch()
        for counts in (first.counts, second.counts):
            result["CPU_OT_attempts"] += counts["actual_OT_attempts"]
            result["CPU_OT_returns"] += counts["actual_OT_returns"]
            result["CPU_OT_cache_hits"] += counts["OT_cache_hits"]
        check("one_actual_batch_OT_accounting", result["CPU_OT_attempts"] == 8
            and result["CPU_OT_returns"] == 8 and result["CPU_OT_cache_hits"] == 8)
        check("actual_two_arm_input_and_label_hashes", all(a["shared_input_sha256"] == b["shared_input_sha256"]
            and a["coupled_input_sha256"] == b["coupled_input_sha256"] for a,b in zip(samples, paired)))
        check("actual_batch_hashes_match", first.last_audit == second.last_audit or
            all(first.last_audit[k] == second.last_audit[k] for k in
                ("shared_batch_sha256", "coupled_batch_sha256", "effective_free_coordinates", "samples")))
        exposure = []
        for sample, other in zip(samples, paired):
            free = sample["valid_mask"] & ~sample["known_mask"]
            check("actual_tensor_bytes_"+str(sample["sample_index"]), all(
                sample[key].dtype == other[key].dtype and sample[key].shape == other[key].shape
                and sample[key].tobytes() == other[key].tobytes() for key in
                ("xt", "u", "x0", "x1", "context", "t", "known_mask", "valid_mask")))
            check("actual_velocity_and_path_"+str(sample["sample_index"]),
                np.array_equal(sample["u"], sample["x1"]-sample["x0"])
                and np.array_equal(sample["xt"][free], ((np.float32(1)-sample["t"])*sample["x0"]+sample["t"]*sample["x1"])[free]))
            exposure.append(dict(sample_index=sample["sample_index"], uid=sample["object_id"],
                task_id=sample["task_id"], alpha=sample["alpha"], N=int(sample["y"]), K=sample["K"],
                free_coordinates=sample["free_coordinate_count"], shared_input_sha256=sample["shared_input_sha256"],
                coupled_input_sha256=sample["coupled_input_sha256"]))
        with (out/"actual_paired_input_checks.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(exposure[0]))
            writer.writeheader();writer.writerows(exposure)
        check("actual_scalar_denominator", batch_contract(samples) == first.last_audit["effective_free_coordinates"])
        state = first.state_dict()
        restored = MomentStream(SOURCE, CHAIR, fixture, "H_CONT")
        restored.load_state_dict(state)
        check("stream_cursor_restore", restored.state_dict() == state and restored.batch_index == 1)
        wrong = dict(state, sample_index=9)
        rejects("reject_stream_cursor_mismatch", lambda: restored.load_state_dict(wrong))
        wrong = dict(state, arm="MOMENT_XY")
        rejects("reject_stream_arm_mismatch", lambda: restored.load_state_dict(wrong))

        # A varied-length batch distinguishes total-free-coordinate weighting
        # from an incorrect arithmetic average of eight per-sample MSEs.
        velocities, targets, masks = [], [], []
        denominator = 9*sum(range(1,9))
        total_loss = torch.zeros(())
        expected_sum = 0.
        sample_mses = []
        for i in range(8):
            n = i+4
            velocity = torch.arange(n*9, dtype=torch.float32).reshape(1,n,9).div(23).requires_grad_()
            target = torch.full_like(velocity, .1*(i+1))
            valid = torch.zeros((1,n), dtype=torch.bool);valid[:,:i+2] = True
            known = torch.zeros_like(valid);known[:,0] = True
            free = valid & ~known
            total_loss = total_loss + masked_fm_loss(velocity, target, valid, known, denominator=denominator)
            difference = velocity.detach()-target
            expected_sum += float(difference[free].square().sum())
            sample_mses.append(float(difference[free].square().mean()))
            velocities.append(velocity);targets.append(target);masks.append(free)
        result["scalar_loss_backwards"] += 1
        total_loss.backward()
        expected_loss = expected_sum/denominator
        check("batch_scalar_normalization_value", abs(float(total_loss)-expected_loss) < 1e-6,
            dict(actual=float(total_loss), expected=expected_loss, denominator=denominator,
                 wrong_equal_microbatch_mean=sum(sample_mses)/8))
        check("normalization_test_discriminates_wrong_mean", abs(expected_loss-sum(sample_mses)/8) > .01)
        for i,(v,target,free) in enumerate(zip(velocities,targets,masks)):
            expected_grad = 2*(v.detach()-target)/denominator
            expected_grad[~free] = 0
            check("free_only_gradient_"+str(i), torch.allclose(v.grad,expected_grad,atol=1e-8,rtol=0)
                and torch.count_nonzero(v.grad[~free]) == 0)

        actual_hash = digest(START)
        source_manifest = read_json(HYBRID/"checkpoint_manifest.json")
        expected_source = [row for row in source_manifest["endpoints"] if row["metadata"]["arm"] == "OT_HYBRID"]
        check("actual_source_hash_and_manifest", actual_hash == START_SHA
            and len(expected_source) == 1 and expected_source[0]["checkpoint_sha256"] == START_SHA)
        # Read payload tensors without constructing or evaluating a real model.
        payload = torch.load(START, map_location="cpu", weights_only=True, mmap=True)
        metadata = validate_hybrid(payload)
        check("actual_source_identity", metadata["arm"] == "OT_HYBRID" and metadata["pilot_step"] == 1000
            and metadata["conditional_total_step"] == 5000 and metadata["lambda_variance"] == .25)
        check("actual_old_AdamW_step5000", all(int(s["step"]) == 5000 for s in payload["optimizer"]["state"].values()),
            dict(state_count=len(payload["optimizer"]["state"]), optimizer_hash=metadata["optimizer_state_sha256"],
                 model_hash=metadata["model_state_sha256"], rng_hash=metadata["rng_sha256"]))
        del payload

        for arm in ("H_CONT", "MOMENT_XY"):
            mock = synthetic_checkpoint(arm)
            checkpoint.validate(mock)
            check("checkpoint_contract_"+arm, True)
            broken = copy.deepcopy(mock)
            broken["metadata"]["conditional_total_step"] += 1
            rejects("reject_total_step_"+arm, lambda: checkpoint.validate(broken))
            broken = copy.deepcopy(mock)
            old_id = broken["optimizer"]["param_groups"][0]["params"][0]
            broken["optimizer"]["state"][old_id]["step"] = torch.tensor(200.)
            broken["metadata"]["optimizer_state_sha256"] = tree_hash(broken["optimizer"])
            rejects("reject_reset_old_optimizer_"+arm, lambda: checkpoint.validate(broken))
            if arm == "MOMENT_XY":
                broken = copy.deepcopy(mock)
                new_id = broken["optimizer"]["param_groups"][1]["params"][0]
                broken["optimizer"]["state"][new_id]["step"] = torch.tensor(5200.)
                broken["metadata"]["optimizer_state_sha256"] = tree_hash(broken["optimizer"])
                rejects("reject_inherited_new_optimizer_step", lambda: checkpoint.validate(broken))

        valid_events = [dict(event=kind,moment_step=step) for step in (1,2)
            for kind in ("INPUT_BEGIN","INPUT_DONE","UPDATE_BEGIN","UPDATE_DONE")]
        validate_recovery(valid_events, 2)
        check("covered_recovery_accepted", True)
        rejects("uncovered_update_cannot_replay", lambda: validate_recovery(valid_events,1))
        rejects("duplicate_update_rejected", lambda: validate_recovery(valid_events+[valid_events[-1]],2))
        rejects("unfinished_update_rejected", lambda: validate_recovery(valid_events[:-1],2))
        contract = dict(experiment="chair_moment_channel_v2", seed=20261006, updates_per_arm=1000,
            base_conditional_total_step=5000,effective_batch=8,microbatch=1,lambda_variance=.25,
            checkpoint_interval=200,arms=["H_CONT","MOMENT_XY"])
        validate_training_protocol(contract)
        check("V2_contract_accepts_two_arms", True)
        rejects("V2_rejects_surface_arm", lambda: validate_training_protocol(dict(contract,
            arms=["H_CONT","MOMENT_XY","SURFACE_XY"])))
        check("required_light_logs", set(("old_grad_norm_before_clip", "new_grad_norm_before_clip",
            "new_parameter_delta_l2", "new_first_projection_grad_norm", "new_final_projection_grad_norm",
            "unused_parameter_count", "free_coordinate_denominator", "b_RMS_mean", "b_max", "u_RMS_mean", "u_max")) <= set(LOG_COLUMNS))
        check("no_CUDA_initialization", not torch.cuda.is_initialized())
        result.update(status="PASS", elapsed_seconds=time.perf_counter()-started)
        atomic_json(out/"result.json", result)
        return result
    except BaseException as exc:
        result.update(status="BLOCKED",error=repr(exc),traceback=traceback.format_exc(),
                      elapsed_seconds=time.perf_counter()-started)
        atomic_json(out/"result.json", result)
        raise


def synthetic_checkpoint(arm):
    """Small metadata-contract fixture; its AdamW tensors are constructed, never updated."""
    step = 200
    model = {"backbone.weight":torch.ones((2,2),dtype=torch.float32)}
    old_names = ["backbone.weight"]
    new_names = []
    if arm == "MOMENT_XY":
        for layer in (3,6,9,12):
            for suffix,shape in (("fc1.weight",(32,11)),("fc1.bias",(32,)),
                                 ("fc2.weight",(780,32)),("fc2.bias",(780,))):
                name = f"relation_modules.layer_{layer:02d}."+suffix
                model[name] = torch.zeros(shape,dtype=torch.float32)
                new_names.append(name)
    groups = []
    state = {}
    names = [old_names]+([new_names] if new_names else [])
    for gi,group_names in enumerate(names):
        ids = []
        for name in group_names:
            pid = len(state);ids.append(pid)
            state[pid] = dict(step=torch.tensor(float(5200 if gi==0 else 200)),
                exp_avg=torch.zeros_like(model[name]),exp_avg_sq=torch.zeros_like(model[name]))
        groups.append(dict(params=ids,lr=1e-5,betas=(.9,.95),weight_decay=0.))
    optimizer = dict(state=state,param_groups=groups)
    protocol = {"synthetic_metadata_contract_only":True}
    rng = {"synthetic_metadata_contract_only":True}
    stream = dict(batch_index=200,sample_index=1600,seed=20261006,arm=arm)
    metadata = dict(arm=arm,moment_step=step,conditional_total_step=5200,
        source_checkpoint_sha256=START_SHA,context_encoder="geo",loss_recipe="fm",lambda_variance=.25,
        protocol_sha256=json_hash(protocol),config_sha256=config_sha256(DEFAULT_CONFIG),
        model_state_sha256=tensor_state_hash(model),optimizer_state_sha256=tree_hash(optimizer),
        rng_sha256=tree_hash(rng),stream_state_sha256=json_hash(stream),completed_updates=step,
        surface_attention=dict(new_parameter_names=new_names,new_parameter_count=sum(model[n].numel() for n in new_names)))
    return dict(schema=checkpoint.SCHEMA,metadata=metadata,model=model,optimizer=optimizer,
        protocol=protocol,rng=rng,stream_state=stream,update_in_progress=False,optimizer_parameter_names=names)


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args();value=run(args.out)
    print({k:v for k,v in value.items() if k not in ("checks","source_sha256")},flush=True)
