"""Fixed 200-update paired seam-slot training; controller owns GPU dispatch.

Usage: python -B -m r2.pilots.seam_slot_c200_v1.train --arm FULL4_C200
       --root experiments/r2_seam_slot_c200_v1
No GPU work is executed by importing this module. No automatic failed replay.
"""
from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "r2/src"))
import argparse
import copy
import json
import signal
import time
import traceback
from collections import Counter
import numpy as np
from meshflow_control import runtime
import torch
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash, rng_state, restore_rng
from meshflow_control.models.native import build_model, native_execution
from meshflow_control.training import schedule40 as base
from meshflow_control.training.context_checkpoint import _cpu_copy, _check_generator
from meshflow_control.training.losses import masked_fm_loss
from meshflow_control.data.io import atomic_json, append_event, digest
from r2.pilots.user_condition_finetune_v1 import trainer as historical
from r2.pilots.full_seat_learning_curve_v1 import common as old_common
from r2.prototypes.seam_slot_c200_v1.model_adapter import seam_forward, corner_masked_fm_loss
from r2.prototypes.seam_slot_c200_v1.occupancy import OccupancyHead, occupancy_bce_loss
from . import trainer_support as S

EXPERIMENT = "R2_SEAM_SLOT_C200_V1"
SCHEMA = "r2_seam_slot_c200_training_v1"


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def resource():
    result = old_common.resources()
    try:
        result["free_mib"] = float(result["gpu"]["stdout"].splitlines()[0].split(",")[3])
    except (ValueError, IndexError, KeyError):
        result["free_mib"] = None
    return result


def check_registration(root):
    reg = S.read(root / "registration.json")
    if reg.get("status") != "FROZEN" or (reg["updates_per_arm"], reg["batch"], reg["microbatch"], reg["source_global_step"]) != (200, 8, 1, 12000):
        raise ValueError("Only frozen FULL_S6000 -> 200-update batch8/micro1 accepted")
    if reg.get("head_lr", .001) != .001 or reg.get("head_class_weights", [8.25, 66 / 124]) != [8.25, 66 / 124]:
        raise ValueError("Registered independent occupancy optimization differs")
    for p, sha in reg.get("training_code_hashes", {}).items():
        if digest(S.resolve(p)) != sha:
            raise ValueError("Frozen training code changed: " + p)
    execution = S.read(root / "execution_freeze.json")
    if execution.get("registration_sha256") and execution["registration_sha256"] != digest(root / "registration.json"):
        raise ValueError("Execution freeze registration differs")
    for field, relative_root in (("code_hashes", ROOT), ("input_hashes", root)):
        if not isinstance(execution.get(field), dict) or not execution[field]:
            raise ValueError("Execution freeze requires nonempty " + field)
        for p, sha in execution[field].items():
            if digest(S.resolve(p, relative_root)) != sha:
                raise ValueError("Execution-frozen file changed: " + p)
    manifest_path = root / "data_manifest.json"
    if reg.get("data_manifest_sha256") and digest(manifest_path) != reg["data_manifest_sha256"]:
        raise ValueError("Frozen data manifest changed")
    manifest = S.read(manifest_path)
    if [r["step"] for r in manifest["steps"]] != list(range(1, 201)):
        raise ValueError("Need exactly 200 registered batch entries")
    if any(set(r["samples"]) != set(S.ARMS) or any(len(r["samples"][a]) != 8 for a in S.ARMS) for r in manifest["steps"]):
        raise ValueError("Paired eight-sample manifests required")
    return reg, manifest


def require_cursor(ledger, step, arm, preflight_expected):
    expected = dict(train_forward=step * 8, train_backward=step * 8, optimizer_update=step)
    if arm == "SEAM_SLOT_C200":
        expected.update(head_forward=step * 8, head_backward=step * 8, head_optimizer_update=step,
                        preflight_forward=preflight_expected)
    if ledger.pending or ledger.failed:
        raise RuntimeError("Failed/pending calls forbid implicit replay")
    if any(ledger.attempted[k] != count or ledger.successful[k] != count for k, count in expected.items()):
        raise RuntimeError("Checkpoint is behind actual work; implicit rollback forbidden")
    sequence = [r["context"]["global_step"] for r in ledger.rows if r["event"] == "CALL_BEGIN" and r["kind"] == "optimizer_update"]
    if sequence != list(range(S.BASE_STEP + 1, S.BASE_STEP + step + 1)):
        raise RuntimeError("Optimizer update sequence contains replay/gaps")


def validate_optimizers(model, optimizer, head, head_optimizer, step, generator_hyper):
    base.validate_optimizer(model, None, optimizer, "COMMON", S.BASE_STEP + step)
    if historical.optimizer_hyperparameters(optimizer) != generator_hyper:
        raise ValueError("Inherited generator AdamW hyperparameters changed")
    if head is None:
        if head_optimizer is not None:
            raise ValueError("Ordinary control cannot own occupancy optimizer")
        return
    if len(head_optimizer.param_groups) != 1:
        raise ValueError("Independent head needs exactly one optimizer group")
    group = head_optimizer.param_groups[0]
    if (group["lr"], tuple(group["betas"]), group["weight_decay"]) != (.001, (.9, .95), 0.):
        raise ValueError("Head AdamW differs")
    if [id(p) for p in group["params"]] != [id(p) for p in head.parameters()]:
        raise ValueError("Head optimizer ownership differs")
    if {id(p) for p in head.parameters()} & {id(p) for p in model.parameters()}:
        raise ValueError("Head and generator must not share parameters")
    for p in head.parameters():
        entry = head_optimizer.state.get(p, {})
        if step == 0 and not entry:
            continue
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"} or float(entry["step"]) != step:
            raise ValueError("Head optimizer cursor differs")
        if any(entry[k].dtype != torch.float32 or entry[k].shape != p.shape for k in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Head moment dtype/shape differs")


def tensors(sample):
    n = sample["N"]
    cv = lambda a: torch.as_tensor(np.asarray(a), device="cuda")
    return dict(x=cv(sample["xt"].reshape(1, n, 9)), target=cv(sample["u"].reshape(1, n, 9)),
                t=cv(np.asarray([sample["t"]], np.float32)), y=cv(np.asarray([n], np.int64)),
                valid=cv(sample["valid_mask"][None]), C=cv(sample["known_mask"][None]),
                corner=cv((sample["anchor_ids"] >= 0)[None]))


def null_gate(model, sample, ledger, folder):
    b = tensors(sample)
    corner = b["C"][..., None].expand_as(b["corner"])
    with torch.no_grad(), native_execution(model, coordinates=(b["x"], b["t"]), autocast=True):
        ordinary, first = ledger.call("preflight_forward", lambda: model(b["x"], b["t"], b["y"], valid_mask=b["valid"], known_mask=b["C"]), dict(check="ALL_NULL", route="native"))
        delegated, second = ledger.call("preflight_forward", lambda: seam_forward(model, b["x"], b["t"], b["y"], b["valid"], b["C"], corner), dict(check="ALL_NULL", route="seam_adapter"))
    equal = torch.equal(ordinary, delegated)
    result = dict(status="PASS" if equal else "FAIL", actual_forward_calls=2,
                  max_abs=float((ordinary.float() - delegated.float()).abs().max()),
                  bitwise_equal=equal, seconds=first + second, no_optimizer_update=True,
                  scope="Actual loaded Native all-Null path; mixed path uses first formal training backward")
    write(folder / "all_null_gate.json", result)
    if not equal:
        raise RuntimeError("All-Null Native equivalence failed")


def update(model, optimizer, head, head_optimizer, samples, arm, step, ledger, hyper, bins):
    denominator, edge_denominator = S.validate_batch(samples)
    validate_optimizers(model, optimizer, head, head_optimizer, step - 1, hyper)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    if head is not None:
        head_optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    result = dict(FM=0.,occupancy_BCE=0.,free_coordinate_denominator=denominator,
                  occupancy_edge_denominator=edge_denominator if head is not None else None,
                  forward_seconds=0.,backward_seconds=0.,head_forward_seconds=0.,head_backward_seconds=0.)
    diagnostics, head_rows = [], []
    for micro, sample in enumerate(samples):
        b = tensors(sample)
        context = dict(local_step=step,global_step=S.BASE_STEP+step,microbatch=micro,sample_index=sample["sample_index"],uid=sample["uid"],alpha=sample["alpha"])
        with native_execution(model, coordinates=(b["x"], b["t"]), autocast=True):
            fn = (lambda: seam_forward(model,b["x"],b["t"],b["y"],b["valid"],b["C"],b["corner"]).float()) if head is not None else (lambda: model(b["x"],b["t"],b["y"],valid_mask=b["valid"],known_mask=b["C"]).float())
            prediction, elapsed = ledger.call("train_forward", fn, context)
            result["forward_seconds"] += elapsed
            loss = corner_masked_fm_loss(prediction,b["target"],b["valid"],b["corner"],denominator=denominator) if head is not None else masked_fm_loss(prediction,b["target"],b["valid"],b["C"],denominator=denominator)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite FM")
            value = float(loss.detach())
            _, elapsed = ledger.call("train_backward", loss.backward, context)
            result["backward_seconds"] += elapsed
        predicted_cpu = prediction.detach().cpu().numpy()[0].reshape(sample["N"],3,3)
        row = S.diagnostic(sample,predicted_cpu,bins)
        if not np.isclose(row["all_free_velocity_sse"]/denominator,value,rtol=3e-6,atol=1e-7):
            raise ValueError("Detached FM diagnostic partition disagrees")
        row.update(**context,FM_contribution=value,batch_free_coordinate_denominator=denominator,
                   sample_free_coordinate_denominator=int(sample["free_coordinate_count"]))
        diagnostics.append(row)
        result["FM"] += value
        if head is not None:
            features = torch.as_tensor(sample["edge_features"],device="cuda")
            labels = torch.as_tensor(sample["occupied_labels"],device="cuda")
            mask = torch.ones(labels.shape,dtype=torch.bool,device="cuda")
            # Independent head uses the same BF16 forward / FP32 loss convention.
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits, elapsed = ledger.call("head_forward",lambda:head(features,mask),context)
                result["head_forward_seconds"] += elapsed
                bce = occupancy_bce_loss(logits,labels,mask,class_weights=[8.25,66/124]) * len(labels)/edge_denominator
                if not bool(torch.isfinite(bce)):
                    raise FloatingPointError("Nonfinite occupancy BCE")
                _, elapsed = ledger.call("head_backward",bce.backward,context)
                result["head_backward_seconds"] += elapsed
            result["occupancy_BCE"] += float(bce.detach())
            predicted = (logits.detach().float() >= 0).cpu().numpy()
            truth = sample["occupied_labels"]
            head_rows.append(dict(**context,edge_count=len(truth),positive_count=int(truth.sum()),null_count=int((~truth).sum()),
                                  true_positive=int((predicted&truth).sum()),true_negative=int((~predicted&~truth).sum()),
                                  false_positive=int((predicted&~truth).sum()),false_negative=int((~predicted&truth).sum()),
                                  whole_layout_match=bool(np.array_equal(predicted,truth)),
                                  weighted_BCE_contribution=float(bce.detach()),edge_batch_denominator=edge_denominator,
                                  logits=logits.detach().float().cpu().tolist(),labels=truth.tolist()))
            del features,labels,mask,logits,bce
        del prediction,loss,b
    if any(p.grad is None for p in model.parameters()):
        raise RuntimeError("Generator has unused parameters on formal training path")
    result["generator_gradient_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    if head is not None:
        if any(p.grad is None for p in head.parameters()):
            raise RuntimeError("Occupancy head has unused parameters")
        result["head_gradient_norm"] = float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True))
    context = dict(local_step=step,global_step=S.BASE_STEP+step)
    _, result["optimizer_seconds"] = ledger.call("optimizer_update",optimizer.step,context)
    if head is not None:
        _,result["head_optimizer_seconds"] = ledger.call("head_optimizer_update",head_optimizer.step,context)
    if any(not bool(torch.isfinite(p).all()) for p in list(model.parameters())+([] if head is None else list(head.parameters()))):
        raise FloatingPointError("Nonfinite updated parameter")
    validate_optimizers(model,optimizer,head,head_optimizer,step,hyper)
    result.update(update_seconds=time.perf_counter()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  peak_scope="One formal batch: forward, backward, both optimizer steps; not process VRAM")
    return result,diagnostics,head_rows


def save_payload(path,payload,*,replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError("Refuse immutable checkpoint overwrite: "+str(path))
    temp = path.with_name(path.name+".partial")
    with temp.open("xb") as f:
        torch.save(payload,f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp,path)
    receipt = dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size,
                   **{k:payload[k] for k in ("schema","arm","local_step","global_step","generator_state_sha256","optimizer_state_sha256","rng_sha256","head_state_sha256","head_optimizer_state_sha256") if k in payload})
    write(path.with_suffix(".json"),receipt)
    return receipt


def train(arm,root,attempt=None):
    root = S.resolve(root).resolve()
    if not root.is_relative_to((ROOT/"experiments").resolve()) or arm not in S.ARMS:
        raise ValueError("Independent experiment root and registered arm required")
    reg,manifest = check_registration(root)
    reg_sha,manifest_sha = digest(root/"registration.json"),digest(root/"data_manifest.json")
    folder = root/"training"/arm
    folder.mkdir(parents=True,exist_ok=True)
    status_path = folder/"status.json"
    previous = S.read(status_path) if status_path.exists() else None
    if previous and previous["status"] == "COMPLETE":
        for receipt in (previous["checkpoint"],previous["inference_export"]):
            if digest(receipt["path"]) != receipt["sha256"]:
                raise ValueError("Completed output changed")
        return previous
    if previous and previous["status"] not in ("INTERRUPTED_SAFE","SAFE_SEGMENT"):
        raise RuntimeError("Prior attempt failed/ambiguous; no automatic replay")
    step = previous["local_step"] if previous else 0
    attempt = historical.safe_attempt_id(attempt or "attempt_"+str(int(time.time())))
    adir = folder/"attempts"/attempt
    adir.mkdir(parents=True,exist_ok=False)
    initial_resource = resource()
    write(adir/"resources_dispatch.json",initial_resource)
    policy_path = root / "dispatch_policy.json"
    policy = S.read(policy_path) if policy_path.exists() else {}
    minimum_free = float(policy.get("minimum_free_mib_before_load",6963))
    if not np.isfinite(minimum_free) or minimum_free < 2048:
        raise ValueError("Dispatch policy must preserve at least 2048MiB headroom")
    write(adir/"dispatch_policy.json",dict(policy=policy,minimum_free_mib_before_load=minimum_free,
        source_sha256=digest(policy_path) if policy_path.exists() else None))
    if initial_resource["free_mib"] is None or initial_resource["free_mib"] < minimum_free:
        write(adir/"resource_wait.json",dict(status="RESOURCE_WAIT",resource=initial_resource))
        raise RuntimeError("RESOURCE_WAIT: insufficient measured VRAM; no model loaded")
    limits = dict(train_forward=1600,train_backward=1600,optimizer_update=200)
    if arm == "SEAM_SLOT_C200":
        limits.update(head_forward=1600,head_backward=1600,head_optimizer_update=200,preflight_forward=2)
    ledger = historical.CallLedger(folder/"calls.jsonl",limits,arm,attempt)
    require_cursor(ledger,step,arm,2 if previous and arm=="SEAM_SLOT_C200" else 0)
    state = dict(status="LOADING",arm=arm,local_step=step,global_step=S.BASE_STEP+step,attempt=attempt,
                 registration_sha256=reg_sha,data_manifest_sha256=manifest_sha)
    started = time.perf_counter()
    flag = {"stop":False}
    def interrupt(*unused): flag["stop"] = True
    handlers = {s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM)}
    observer = historical.GPUObserver("seam_slot_training_"+arm,interval=2.)
    bins = np.asarray(reg.get("diagnostic_time_bin_edges",[0,.2,.4,.6,.8,1]),np.float64)
    owned = False
    try:
        with historical.exclusive_writer(folder), observer:
            owned = True
            write(status_path,state)
            torch.set_num_threads(2)
            runtime.configure_stable_runtime()
            checkpoint = S.resolve(previous["checkpoint"]["path"]) if previous else S.resolve(reg["source_checkpoint"])
            expected = previous["checkpoint"]["sha256"] if previous else reg["source_checkpoint_sha256"]
            if digest(checkpoint) != expected:
                raise ValueError("Source/resume checkpoint bytes differ")
            payload = torch.load(checkpoint,map_location="cpu",weights_only=True,mmap=True)
            if payload["model_config"] != MODEL_CONFIG or payload.get("update_in_progress") is not False or payload["global_step"] != S.BASE_STEP+step:
                raise ValueError("Full live checkpoint config/completed optimizer cursor differs")
            for key,hash_fn,hash_key in (("generator_state",state_hash,"generator_state_sha256"),("optimizer",base.tree_hash,"optimizer_state_sha256"),("rng",base.tree_hash,"rng_sha256")):
                if hash_fn(payload[key]) != payload[hash_key]:
                    raise ValueError("Checkpoint internal state hash differs: "+key)
            if previous:
                if payload.get("schema") != SCHEMA or payload["arm"] != arm or payload["registration_sha256"] != reg_sha or payload["data_manifest_sha256"] != manifest_sha:
                    raise ValueError("Resume experiment/stream identity differs")
            else:
                if payload.get("arm") != "FULL_SEAT" or payload["local_step"] != 6000 or payload.get("EMA") is not False:
                    raise ValueError("Only live FULL_SEAT step6000 source accepted")
                for field,expected_field in (("generator_state_sha256","source_generator_sha256"),("optimizer_state_sha256","source_optimizer_sha256"),("rng_sha256","source_rng_sha256")):
                    if payload[field] != reg[expected_field]:
                        raise ValueError("Registered source differs: "+field)
            model = build_model(copy.deepcopy(MODEL_CONFIG),readout_mode="none",trainable=True).eval()
            if payload["optimizer_parameter_names"] != base.optimizer_names(model,None):
                raise ValueError("Generator optimizer parameter mapping changed")
            model.load_state_dict(payload["generator_state"],strict=True)
            model.to("cuda")
            optimizer = base.make_optimizer(model,None)
            optimizer.load_state_dict(payload["optimizer"])
            head = OccupancyHead(24,hidden_dim=32,seed=reg.get("head_seed",20261016)).eval().to("cuda") if arm=="SEAM_SLOT_C200" else None
            head_optimizer = torch.optim.AdamW(head.parameters(),lr=.001,betas=(.9,.95),weight_decay=0.) if head is not None else None
            if head is not None and sum(p.numel() for p in head.parameters()) != 833:
                raise ValueError("Occupancy head capacity differs")
            if previous and head is not None:
                head.load_state_dict(payload["head_state"],strict=True)
                head_optimizer.load_state_dict(payload["head_optimizer"])
                if state_hash(head.state_dict()) != payload["head_state_sha256"] or base.tree_hash(head_optimizer.state_dict()) != payload["head_optimizer_state_sha256"]:
                    raise ValueError("Resumed head state differs")
            restore_rng(payload["rng"])
            if state_hash(model.state_dict()) != payload["generator_state_sha256"] or base.tree_hash(optimizer.state_dict()) != payload["optimizer_state_sha256"] or base.tree_hash(rng_state()) != payload["rng_sha256"]:
                raise ValueError("Exact live model/AdamW/RNG restore failed")
            hyper = historical.optimizer_hyperparameters(optimizer)
            validate_optimizers(model,optimizer,head,head_optimizer,step,hyper)
            initialization = dict(generator_state_sha256=payload["generator_state_sha256"],optimizer_state_sha256=payload["optimizer_state_sha256"],rng_sha256=payload["rng_sha256"],
                                  global_step=payload["global_step"],head_fresh=bool(head is not None and not previous),head_parameters=0 if head is None else 833,
                                  head_state_sha256=None if head is None else state_hash(head.state_dict()),generator_hyperparameters=hyper,
                                  separate_head_optimizer=True,EMA=False,exact_generator_optimizer_RNG_restoration=True,
                                  occupancy_BCE_normalization="Weighted sum / all 132 effective-batch edges; not mean of four C means",
                                  occupancy_class_weights=[8.25,66/124] if head is not None else None,
                                  head_precision="BF16 forward, FP32 parameters/loss/AdamW" if head is not None else None)
            write(adir/"initialization.json",initialization)
            del payload
            def load_batch(current,which=arm):
                return [S.load_sample(rec,root,which,current,micro,digest) for micro,rec in enumerate(manifest["steps"][current-1]["samples"][which])]
            if head is not None and not previous:
                null_sample = load_batch(1,"FULL4_C200")[0]
                null_gate(model,null_sample,ledger,folder)
            require_cursor(ledger,step,arm,2 if head is not None else 0)
            state["status"] = "TRAINING"
            write(status_path,state)
            def checkpoint_save(final=False):
                require_cursor(ledger,step,arm,2 if head is not None else 0)
                validate_optimizers(model,optimizer,head,head_optimizer,step,hyper)
                generator = _cpu_copy(model.state_dict())
                _check_generator(generator)
                opt,rng = _cpu_copy(optimizer.state_dict()),_cpu_copy(rng_state())
                hs = None if head is None else _cpu_copy(head.state_dict())
                ho = None if head is None else _cpu_copy(head_optimizer.state_dict())
                data = dict(schema=SCHEMA,experiment=EXPERIMENT,arm=arm,local_step=step,global_step=S.BASE_STEP+step,update_in_progress=False,
                            generator_state=generator,generator_state_sha256=state_hash(generator),optimizer=opt,optimizer_state_sha256=base.tree_hash(opt),
                            optimizer_parameter_names=base.optimizer_names(model,None),rng=rng,rng_sha256=base.tree_hash(rng),model_config=copy.deepcopy(MODEL_CONFIG),
                            head_state=hs,head_state_sha256=None if hs is None else state_hash(hs),head_optimizer=ho,head_optimizer_state_sha256=None if ho is None else base.tree_hash(ho),
                            head_config=None if head is None else dict(features_dim=24,hidden_dim=32,seed=reg.get("head_seed",20261016)),
                            registration_sha256=reg_sha,data_manifest_sha256=manifest_sha,source_checkpoint_sha256=reg["source_checkpoint_sha256"],
                            stream_state=dict(batch_index=step,next_sample_index=step*8),execution=ledger.snapshot(),EMA=False)
                destination = root/"checkpoints"/(f"{arm}_step200.pt" if final else f"{arm}_rolling.pt")
                state["checkpoint"] = save_payload(destination,data,replace=not final)
                append_event(folder/"checkpoint_history.jsonl",state["checkpoint"])
                if final:
                    export = {k:data[k] for k in ("experiment","arm","local_step","global_step","generator_state","generator_state_sha256","model_config","EMA")}
                    export.update(schema="r2_seam_slot_c200_generator_v1",training_checkpoint_sha256=state["checkpoint"]["sha256"])
                    state["inference_export"] = save_payload(root/"inference"/f"{arm}_generator.pt",export)
                    if head is not None:
                        out = {k:data[k] for k in ("experiment","arm","local_step","global_step","head_state","head_state_sha256","head_config")}
                        out.update(schema="r2_seam_slot_c200_occupancy_v1",threshold_logit=0.0,training_checkpoint_sha256=state["checkpoint"]["sha256"])
                        state["occupancy_export"] = save_payload(root/"inference"/f"{arm}_occupancy.pt",out)
                write(status_path,state)
            for current in range(step+1,201):
                if current==step+1 and (current==1 or current%10==1) or flag["stop"]:
                    rr = resource()
                    append_event(folder/"resource_checks.jsonl",rr)
                    if flag["stop"] or rr["free_mib"] is None or rr["free_mib"] < 2048:
                        checkpoint_save()
                        state.update(status="INTERRUPTED_SAFE",reason="signal_or_resource_headroom")
                        break
                tick = time.perf_counter()
                samples = load_batch(current)
                input_seconds = time.perf_counter()-tick
                for micro,sample in enumerate(samples):
                    append_event(folder/"exposure.jsonl",dict(local_step=current,global_step=S.BASE_STEP+current,microbatch=micro,uid=sample["uid"],
                        sample_index=sample["sample_index"],alpha=sample["alpha"],t=float(sample["t"]),N=sample["N"],K=sample["K"],
                        free_coordinate_count=int(sample["free_coordinate_count"]),path=sample["path"]))
                result,diagnostics,head_rows = update(model,optimizer,head,head_optimizer,samples,arm,current,ledger,hyper,bins)
                step = current
                append_event(folder/"training_log.jsonl",dict(local_step=step,global_step=S.BASE_STEP+step,input_seconds=input_seconds,**result))
                for row in diagnostics: append_event(folder/"geometry_diagnostics.jsonl",row)
                for row in head_rows: append_event(folder/"occupancy_training.jsonl",row)
                state.update(local_step=step,global_step=S.BASE_STEP+step,last_FM=result["FM"],last_BCE=result["occupancy_BCE"],
                             peak_reserved_bytes=result["cuda_peak_reserved_bytes"],peak_allocated_bytes=result["cuda_peak_allocated_bytes"],
                             wall_seconds=time.perf_counter()-started)
                if step==1:
                    write(folder/"first_formal_backward.json",dict(status="PASS",arm=arm,no_extra_forward=True,
                        generator_gradient_norm=result["generator_gradient_norm"],head_gradient_norm=result.get("head_gradient_norm"),
                        free_coordinate_denominator=result["free_coordinate_denominator"],head_edge_denominator=result["occupancy_edge_denominator"]))
                if step%50==0:
                    checkpoint_save(final=step==200)
                write(status_path,state)
                if step==1 or step%10==0:
                    print(json.dumps(dict(arm=arm,step=step,FM=result["FM"],BCE=result["occupancy_BCE"],update_seconds=result["update_seconds"],
                                          peak_reserved_mib=result["cuda_peak_reserved_bytes"]/2**20,wall_seconds=state["wall_seconds"])),flush=True)
            else:
                state["status"] = "COMPLETE"
    except BaseException as exc:
        state.update(status="OOM" if isinstance(exc,torch.cuda.OutOfMemoryError) else "FAILED",error=repr(exc),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(adir/"failure.json",state)
        raise
    finally:
        for sig,handler in handlers.items(): signal.signal(sig,handler)
        state.update(local_step=step,global_step=S.BASE_STEP+step,wall_seconds=time.perf_counter()-started,calls=ledger.snapshot(),resources=observer.summary())
        write(adir/"resources.json",dict(samples=observer.samples,summary=observer.summary()))
        if owned:
            write(status_path,state)
        write(adir/"final_status.json",state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",choices=S.ARMS,required=True)
    parser.add_argument("--root",default="experiments/r2_seam_slot_c200_v1")
    parser.add_argument("--attempt")
    args = parser.parse_args()
    print(json.dumps(train(args.arm,args.root,args.attempt),indent=2))


if __name__ == "__main__":
    main()
