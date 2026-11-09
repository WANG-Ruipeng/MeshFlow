"""Budgeted real START/readout preflight. Run only under explicit GPU dispatch.

New script schedule: 43 one-step forwards, 11 zero-update backwards, 4 Euler50
rollouts. The independent official references already consume 4 forwards and
2 Euler50 rollouts. DONE cells are durable and reused only with identical inputs
and scientific source hashes. An incomplete cell never silently reexecutes.
"""
from __future__ import annotations
from meshflow_control import runtime
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback
import numpy as np
import torch
from meshflow_control.accounting import Budget
from meshflow_control.artifacts import atomic_json, file_sha256, array_hash
from meshflow_control.checkpoints import (load_start, state_hash, START_STATE_SHA256,
    rng_state, restore_rng)
from meshflow_control.data.stream import HybridStream
from meshflow_control.models.native import native_execution
from meshflow_control.sampling import sample
from meshflow_control.training.losses import masked_fm_loss

MODES = (("A", "none"), ("B", "mixed"), ("C", "clean"))
SCHEDULE = dict(new_preflight_forward=43, new_preflight_backward=11,
                new_rollout_attempt=4, new_sampling_forward=200,
                previously_official_forward=4, previously_official_rollout=2)


def digest_tree(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            x = v.detach().cpu().contiguous()
            h.update(str((tuple(x.shape), str(x.dtype))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str): h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for q in v: visit(q)
        else: h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def optimizer_hash(model, optimizer):
    names = {id(p): n for n,p in model.named_parameters()}
    groups = []
    for g in optimizer.param_groups:
        groups.append({**{k:v for k,v in g.items() if k != "params"},
                       "params": [names[id(p)] for p in g["params"]]})
    return digest_tree({"groups":groups,
        "state":{names[id(p)]:s for p,s in optimizer.state.items()}})


def scientific_sources():
    import meshflow_control
    root = Path(meshflow_control.__file__).parent
    paths = sorted((root/"models").rglob("*.py"))
    paths += [root/q for q in ("runtime.py", "sampling.py", "checkpoints.py",
        "training/losses.py", "training/coupling.py", "data/stream.py", "data/dataset.py")]
    return {str(p.relative_to(root)):file_sha256(p) for p in paths}


class Store:
    def __init__(self, out, identity, budget, resume=False):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.cells=self.out/"cells";self.cells.mkdir(exist_ok=True)
        self.path=self.out/"preflight.json";self.budget=budget
        if self.path.exists():
            if not resume: raise RuntimeError("Prior preflight exists; use explicit --resume after review")
            self.report=json.loads(self.path.read_text())
            if self.report["identity"] != identity: raise RuntimeError("Preflight input/scientific source identity changed")
            self.report.setdefault("resume_events",[]).append(dict(time_unix=time.time(),prior_status=self.report["status"]))
        else:
            self.report=dict(schema="native_direct_context_readout_real_preflight_v1",
                identity=identity,status="RUNNING",schedule=SCHEDULE,cells={},comparisons={},
                arms={},cost={},errors=[],optimizer_updates=0,initial_budget=budget.snapshot())
        sha=file_sha256(__file__)
        self.report.setdefault("script_versions",[])
        if sha not in self.report["script_versions"]:
            self.report["script_versions"].append(sha)
            (self.out/("preflight_gpu_"+sha[:12]+".py")).write_bytes(Path(__file__).read_bytes())
        self.report["status"]="RUNNING";self.flush()

    def flush(self):
        self.report["budget"]=self.budget.snapshot()
        atomic_json(self.path,self.report)

    def cell(self, key, compute):
        meta=self.cells/(key+".json");arrays=self.cells/(key+".npz")
        started=self.cells/(key+".started.json")
        if meta.exists():
            data=json.loads(meta.read_text())
            if data.get("status")!="DONE" or file_sha256(arrays)!=data["array_file_sha256"]:
                raise RuntimeError("Completed cell artifact identity differs: "+key)
            with np.load(arrays,allow_pickle=False) as f: values={k:f[k].copy() for k in f.files}
            self.report["cells"][key]=data;self.flush()
            return values,data["metadata"]
        if started.exists():
            raise RuntimeError("Incomplete cell requires explicit engineering recovery; no silent model retry: "+key)
        atomic_json(started,dict(cell=key,status="STARTED",time_unix=time.time(),budget=self.budget.snapshot()))
        values,details=compute()
        tmp=arrays.with_suffix(".partial")
        with tmp.open("wb") as f:
            np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno())
        os.replace(tmp,arrays)
        data=dict(status="DONE",cell=key,metadata=details,array_file_sha256=file_sha256(arrays))
        atomic_json(meta,data)
        self.report["cells"][key]=data;self.flush()
        return values,details

    def check(self, key, values, passed=True):
        self.report["comparisons"][key]=dict(values,status="PASS" if passed else "FAIL")
        self.flush()
        if not passed: raise RuntimeError("Implementation gate failed: "+key)


def errors(actual, expected, mask):
    a=np.asarray(actual,np.float64)[mask];b=np.asarray(expected,np.float64)[mask]
    if a.size==0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Comparison needs nonempty finite values")
    delta=a-b; max_abs=float(np.abs(delta).max());rms=float(np.sqrt(np.mean(delta**2)))
    scale_inf=max(float(np.abs(b).max()),1e-30);scale_rms=max(float(np.sqrt(np.mean(b**2))),1e-30)
    return dict(max_abs=max_abs,rms=rms,e_inf=max_abs/scale_inf,e_rms=rms/scale_rms,
        denominator_inf=scale_inf,denominator_rms=scale_rms,
        bitwise=bool(np.ascontiguousarray(actual).tobytes()==np.ascontiguousarray(expected).tobytes()),
        normalization="error / reference maxabs or RMS, floor=1e-30")


def as_inputs(row, *, padded=0, pad_value=0., perturb_free=False, requires_grad=False):
    x=np.asarray(row["x"],np.float32).copy()
    t=np.asarray(row["t"],np.float32).reshape(1).copy()
    known=np.asarray(row["known"],bool).copy();valid=np.asarray(row["valid"],bool).copy()
    if perturb_free:
        x[valid & ~known]=-.37*x[valid & ~known]+np.float32(.11)
        t=(t+np.float32(.23)) % np.float32(1)
    if padded:
        x=np.concatenate((x,np.full((1,padded,9),pad_value,np.float32)),1)
        known=np.concatenate((known,np.zeros((1,padded),bool)),1)
        valid=np.concatenate((valid,np.zeros((1,padded),bool)),1)
    return dict(x=torch.from_numpy(x).cuda().requires_grad_(requires_grad),
        t=torch.from_numpy(t).cuda(),y=torch.from_numpy(np.asarray(row["y"],np.int64).reshape(1)).cuda(),
        valid=torch.from_numpy(valid).cuda(),known=torch.from_numpy(known).cuda())


def permute_input(row, seed):
    rng=np.random.RandomState(seed);n=row["x"].shape[1]
    face=rng.permutation(n);corners=np.asarray([rng.permutation(3) for _ in range(n)])
    tris=row["x"].reshape(n,3,3)
    changed=dict(row,x=tris[np.arange(n)[:,None],corners][face].reshape(1,n,9),
                 valid=row["valid"][:,face],known=row["known"][:,face])
    return changed,face,corners


def unpermute(output, face, corners):
    n=len(face);tri=output.reshape(n,3,3)[np.argsort(face)]
    return tri[np.arange(n)[:,None],np.argsort(corners,axis=1)].reshape(1,n,9)


def memory_snapshot(observer_records):
    if len(observer_records)!=1: raise RuntimeError("Expected one CLEAN H0 memory per forward")
    return {"memory_"+str(i):v.detach().float().cpu().numpy().copy()
            for i,v in enumerate(observer_records[0])}


def forward_cell(model, data, budget, context, *, bf16, backward=None, capture_memory=False):
    model.zero_grad(set_to_none=True);memory=[];need_grad=backward is not None
    def observe(record):
        if record["site"]=="clean_memory":memory.append(record["memory"])
    with torch.set_grad_enabled(need_grad),native_execution(model,coordinates=(data["x"],data["t"]),autocast=bf16):
        output=budget.call("preflight_forward",lambda:model(data["x"],data["t"],data["y"],data["valid"],data["known"],
            observer=observe if capture_memory else None),context=context)
        if output.shape!=data["x"].shape or not bool(torch.isfinite(output).all()):
            raise RuntimeError("Nonfinite/malformed model output")
        expected=torch.bfloat16 if bf16 else torch.float32
        if output.dtype!=expected: raise RuntimeError("Actual output precision differs")
        values={"output":output.detach().float().cpu().numpy().copy()}
        if capture_memory:values.update(memory_snapshot(memory))
        loss=None
        if backward=="padding":loss=output.float()[data["valid"]].square().mean()
        elif backward=="memory":
            if len(memory)!=1: raise RuntimeError("CLEAN memory observer did not execute")
            loss=sum(m.float().square().sum() for m in memory[0])/sum(m.numel() for m in memory[0])
        elif backward is not None:raise ValueError("Unknown backward objective")
        if loss is not None:
            if not bool(torch.isfinite(loss)):raise RuntimeError("Nonfinite preflight loss")
            budget.call("preflight_backward",loss.backward,context=context)
            if data["x"].grad is None:raise RuntimeError("Expected input gradient")
            values["input_gradient"]=data["x"].grad.detach().cpu().numpy().copy()
            if not np.isfinite(values["input_gradient"]).all():raise RuntimeError("Nonfinite input gradient")
    details=dict(context=context,precision="bf16" if bf16 else "fp32",output_dtype=str(output.dtype),
        actual_forwards=1,actual_backwards=int(loss is not None),loss=None if loss is None else float(loss.detach()))
    model.zero_grad(set_to_none=True)
    return values,details


def cost_cell(model, training_sample, denominator, budget, context):
    s=training_sample
    data=as_inputs(dict(x=s["xt"][None],t=s["t"].reshape(1),y=s["y"].reshape(1),
        valid=s["valid_mask"][None],known=s["known_mask"][None]))
    target=torch.from_numpy(s["u"][None].copy()).cuda()
    model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start_reserved=torch.cuda.memory_reserved();start_free,total=torch.cuda.mem_get_info()
    free_samples=[int(start_free)];times={}
    with native_execution(model,coordinates=(data["x"],data["t"])):
        def forward():
            torch.cuda.synchronize();times["whole_start"]=time.perf_counter()
            tick=time.perf_counter()
            result=model(data["x"],data["t"],data["y"],data["valid"],data["known"])
            torch.cuda.synchronize();times["forward_seconds"]=time.perf_counter()-tick
            free_samples.append(int(torch.cuda.mem_get_info()[0]));return result
        output=budget.call("preflight_forward",forward,context=context)
        if output.dtype!=torch.bfloat16 or output.shape!=data["x"].shape or not bool(torch.isfinite(output).all()):
            raise RuntimeError("Invalid BF16 cost output")
        loss=masked_fm_loss(output,target,data["valid"],data["known"],denominator=denominator)
        if not bool(torch.isfinite(loss)):raise RuntimeError("Nonfinite training micro loss")
        def backward():
            torch.cuda.synchronize();tick=time.perf_counter();loss.backward();torch.cuda.synchronize()
            times["backward_seconds"]=time.perf_counter()-tick
            times["forward_backward_seconds"]=time.perf_counter()-times["whole_start"]
            free_samples.append(int(torch.cuda.mem_get_info()[0]))
        budget.call("preflight_backward",backward,context=context)
    # Capture cost/peaks before diagnostic norm work; no optimizer update occurs.
    allocated=int(torch.cuda.max_memory_allocated());reserved=int(torch.cuda.max_memory_reserved())
    conservative_free=min(min(free_samples),int(start_free)-max(0,reserved-int(start_reserved)))
    needed=max(2*1024**3,.15*int(total))
    new_norms={};total_grad_sq=0.;missing=[]
    for name,p in model.named_parameters():
        if p.grad is None:missing.append(name);continue
        if not bool(torch.isfinite(p.grad).all()):raise RuntimeError("Nonfinite parameter gradient: "+name)
        norm=float(torch.linalg.vector_norm(p.grad.float()));total_grad_sq+=norm*norm
        if name.startswith("readouts."):new_norms[name]=norm
    values={"output":output.detach().float().cpu().numpy().copy()}
    metadata=dict(context=context,shared_input_sha256=s["shared_input_sha256"],
        input_bytehash=s["input_bytehash"],label_bytehash=s["label_bytehash"],
        denominator=denominator,free_coordinates=s["free_coordinate_count"],
        loss=float(loss.detach()),actual_forwards=1,actual_backwards=1,optimizer_updates=0,
        forward_seconds=times["forward_seconds"],backward_seconds=times["backward_seconds"],
        forward_backward_seconds=times["forward_backward_seconds"],
        timing_scope="synchronized actual forward/backward; whole includes loss and intervening ledger writes; excludes gradient summaries",
        prior_same_model_zero_forwards=4,extra_warmup_calls=0,
        peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,
        peak_scope="reset before this micro F+B; includes resident model and complete original/new AdamW",
        driver_free_samples_bytes=free_samples,total_device_bytes=int(total),
        conservative_free_bytes=conservative_free,required_headroom_bytes=needed,
        headroom_pass=conservative_free>=needed,
        headroom_scope="min synchronized device-free samples and base free minus torch peak-reserved increment; device includes other processes",
        own_process_peak_bytes=None,own_process_peak_status="NOT_MEASURED",
        new_gradient_norms=new_norms,all_gradient_norm=total_grad_sq**.5,missing_gradient_names=missing)
    model.zero_grad(set_to_none=True)
    return values,metadata


def run_arm(arm, mode, refs, samples, denominator, args, store, budget):
    model=optimizer=None;before=None;wo={};rng=None
    try:
        model,optimizer,audit=load_start(args.start,mode=mode,device="cuda",trainable=True)
        old={n:p for n,p in model.state_dict().items() if not n.startswith("readouts.")}
        original_sha=state_hash(old)
        if original_sha!=START_STATE_SHA256:raise RuntimeError("Loaded original tensors differ")
        rng=rng_state()
        before=dict(old_model_sha256=original_sha,optimizer_sha256=optimizer_hash(model,optimizer),
            readout_sha256=state_hash(model.readouts.state_dict()),rng_sha256=digest_tree(rng))
        store.report["arms"][arm]=dict(status="RUNNING",mode=mode,load_audit=audit,before=before)
        store.flush()
        # Actual loaded optimizer owns each original/new parameter exactly once.
        named=dict(model.named_parameters());owned=[id(p) for g in optimizer.param_groups for p in g["params"]]
        store.check(arm+"_optimizer_ownership",dict(parameters=len(named),optimizer_entries=len(optimizer.state),
            original_steps=sorted({float(optimizer.state[p]["step"]) for n,p in named.items() if not n.startswith("readouts.")}),
            new_steps=sorted({float(optimizer.state[p]["step"]) for n,p in named.items() if n.startswith("readouts.")})),
            len(owned)==len(set(owned))==len(named) and set(owned)=={id(p) for p in named.values()})
        for i,row in enumerate(refs):
            for precision in ("fp32","bf16"):
                key=f"{arm}_zero_ref{i}_{precision}"
                values,_=store.cell(key,lambda row=row,key=key,precision=precision:
                    forward_cell(model,as_inputs(row),budget,dict(arm=arm,cell=key),bf16=precision=="bf16"))
                cmp=errors(values["output"],row[precision+"_output"],row["valid"])
                store.check(key,cmp,cmp["bitwise"])
        costs=[]
        for index,s in enumerate(samples[:2]):
            key=f"{arm}_cost_micro{index}"
            _,metadata=store.cell(key,lambda s=s,key=key:cost_cell(model,s,denominator,budget,dict(arm=arm,cell=key)))
            costs.append(metadata)
            expected=0 if mode=="none" else 16
            store.check(key+"_gradients_and_headroom",dict(new_gradient_keys=len(metadata["new_gradient_norms"]),
                missing_gradient_names=metadata["missing_gradient_names"],
                conservative_free_bytes=metadata["conservative_free_bytes"],
                required_headroom_bytes=metadata["required_headroom_bytes"]),
                len(metadata["new_gradient_norms"])==expected and not metadata["missing_gradient_names"] and metadata["headroom_pass"])
        store.report["cost"][arm]=dict(rows=costs,
            median_forward_seconds=statistics.median(v["forward_seconds"] for v in costs),
            median_forward_backward_seconds=statistics.median(v["forward_backward_seconds"] for v in costs))
        store.flush()
        if arm!="A":
            base=store.report["cost"]["A"]
            ratio=store.report["cost"][arm]["median_forward_backward_seconds"]/base["median_forward_backward_seconds"]
            paired=[costs[i]["forward_backward_seconds"]/base["rows"][i]["forward_backward_seconds"] for i in range(2)]
            identity=[costs[i]["shared_input_sha256"]==base["rows"][i]["shared_input_sha256"] for i in range(2)]
            store.check(arm+"_cost",dict(median_forward_backward_ratio_to_A=ratio,
                per_micro_forward_backward_ratio_to_A=paired,paired_inputs_equal=identity,
                criterion="median whole F+B <= 3x A; two matched micro rows also retained"),ratio<=3 and all(identity))
        if mode!="none":
            # Replays run at exact zero initialization before any temporary perturbation.
            model.zero_grad(set_to_none=True);model.set_trainable(False)
            for i,row in enumerate(refs):
                key=f"{arm}_zero_euler_ref{i}"
                def rollout(row=row,key=key):
                    raw,path,audit_sample=sample(model,row["C"],int(row["y"][0]),z=row["z"],budget=budget,
                        out=store.out/"replays"/key,context=dict(arm=arm,cell=key,purpose="zero_initialization"))
                    return dict(raw=raw,trajectory=path),audit_sample
                values,details=store.cell(key,rollout)
                raw_equal=values["raw"].tobytes()==row["raw"].tobytes()
                path_equal=values["trajectory"].tobytes()==row["trajectory"].tobytes()
                store.check(key,dict(raw_bitwise=raw_equal,all51_bitwise=path_equal,
                    known_all51_bitwise=details["known_all51_bitwise"],nfe=details["nfe"]),raw_equal and path_equal)
            model.set_trainable(True)
            gen=torch.Generator(device="cpu").manual_seed(2026100701)
            with torch.no_grad():
                for layer,module in model.readouts.items():
                    wo[layer]=module.W_O.weight.detach().cpu().clone()
                    module.W_O.weight.copy_(torch.randn(module.W_O.weight.shape,generator=gen,dtype=torch.float32)*.01)
            store.report["arms"][arm]["temporary_WO"]=dict(seed=2026100701,std=.01,
                hash=state_hash({layer:m.W_O.weight for layer,m in model.readouts.items()}),
                isolated_copy=True,saved_as_training_start=False)
            store.flush()
            clean_original_memory=None
            for i,row in enumerate(refs):
                changed,face,corners=permute_input(row,2026100750+i)
                for precision in ("fp32","bf16"):
                    key=f"{arm}_nonzero_ref{i}_{precision}_original"
                    capture=mode=="clean" and i==0 and precision=="fp32"
                    original,_=store.cell(key,lambda row=row,key=key,precision=precision,capture=capture:
                        forward_cell(model,as_inputs(row),budget,dict(arm=arm,cell=key),bf16=precision=="bf16",capture_memory=capture))
                    if capture:clean_original_memory={k:v for k,v in original.items() if k.startswith("memory_")}
                    key=f"{arm}_nonzero_ref{i}_{precision}_permuted"
                    permuted,_=store.cell(key,lambda changed=changed,key=key,precision=precision:
                        forward_cell(model,as_inputs(changed),budget,dict(arm=arm,cell=key),bf16=precision=="bf16"))
                    pulled=unpermute(permuted["output"],face,corners)
                    valid_error=errors(pulled,original["output"],row["valid"])
                    free_error=errors(pulled,original["output"],row["valid"] & ~row["known"])
                    passed=precision=="bf16" or (valid_error["e_inf"]<=5e-5 and valid_error["e_rms"]<=1e-5
                        and free_error["e_inf"]<=5e-5 and free_error["e_rms"]<=1e-5)
                    store.check(f"{arm}_permutation_ref{i}_{precision}",dict(valid=valid_error,free=free_error,
                        face_permutation=face.tolist(),corner_permutations=corners.tolist(),
                        criterion="FP32 5e-5/1e-5; BF16 descriptive only",descriptive_only=precision=="bf16"),passed)
                key=f"{arm}_padding_ref{i}_zero"
                pad0,_=store.cell(key,lambda row=row,key=key:
                    forward_cell(model,as_inputs(row,padded=4),budget,dict(arm=arm,cell=key),bf16=False))
                key=f"{arm}_padding_ref{i}_changed"
                pad1,_=store.cell(key,lambda row=row,key=key:
                    forward_cell(model,as_inputs(row,padded=4,pad_value=17.25,requires_grad=True),budget,
                        dict(arm=arm,cell=key),bf16=False,backward="padding"))
                n=row["x"].shape[1];valid_equal=pad0["output"][:,:n][row["valid"]].tobytes()==pad1["output"][:,:n][row["valid"]].tobytes()
                padding_grad_max=float(np.abs(pad1["input_gradient"][:,n:]).max())
                store.check(f"{arm}_padding_ref{i}",dict(valid_output_bitwise=valid_equal,
                    padding_gradient_max_abs=padding_grad_max,padding_faces=4,padding_value=17.25,
                    same_shape=True,padding_length_invariance="CPU readout tested; whole-backbone length differences not used as exemption"),
                    valid_equal and padding_grad_max==0)
            if mode=="clean":
                row=refs[0];key="C_memory_changed_free_and_time"
                values,_=store.cell(key,lambda:forward_cell(model,
                    as_inputs(row,perturb_free=True,requires_grad=True),budget,dict(arm=arm,cell=key),
                    bf16=False,backward="memory",capture_memory=True))
                same=all(values[k].tobytes()==v.tobytes() for k,v in clean_original_memory.items())
                free_gradient=float(np.abs(values["input_gradient"][row["valid"] & ~row["known"]]).max())
                known_gradient=float(np.linalg.norm(values["input_gradient"][row["known"]].astype(np.float64)))
                store.check("C_clean_memory_independence",dict(memory_bitwise=same,
                    free_coordinate_gradient_max_abs=free_gradient,known_coordinate_gradient_norm=known_gradient,
                    source="H0 gathered after coordinate+role+Geo; before block/time readout; no detach"),
                    same and free_gradient==0 and known_gradient>0)
        store.report["arms"][arm]["status"]="CHECKS_COMPLETE"
    finally:
        if model is not None:
            with torch.no_grad():
                for layer,value in wo.items():model.readouts[layer].W_O.weight.copy_(value)
            model.zero_grad(set_to_none=True);model.set_trainable(True)
            if before is not None:
                after=dict(old_model_sha256=state_hash({n:p for n,p in model.state_dict().items() if not n.startswith("readouts.")}),
                    optimizer_sha256=optimizer_hash(model,optimizer),readout_sha256=state_hash(model.readouts.state_dict()),
                    rng_sha256=digest_tree(rng_state()))
                restored={k:before[k]==after[k] for k in before}
                store.report["arms"][arm].update(after=after,restored=restored)
                if not all(restored.values()):
                    store.report["arms"][arm]["status"]="RESTORE_FAILED"
                    store.flush()
                    if rng is not None:restore_rng(rng)
                    raise RuntimeError("Model/AdamW/RNG changed during zero-update preflight: "+arm)
                if store.report["arms"][arm]["status"]=="CHECKS_COMPLETE":store.report["arms"][arm]["status"]="PASS"
                store.flush()
            del model,optimizer
            gc.collect();torch.cuda.empty_cache()


def load_references(path):
    path=Path(path);official=json.loads((path/"official_reference.json").read_text())
    if official.get("status")!="PASS":raise RuntimeError("Official references did not pass")
    refs=[];hashes={"official_reference.json":file_sha256(path/"official_reference.json")}
    required={"x","t","y","valid","known","C","z","bf16_output","fp32_output","raw","trajectory"}
    for i in range(2):
        p=path/f"ref_{i}.npz"
        with np.load(p,allow_pickle=False) as z:row={k:z[k].copy() for k in z.files}
        if set(row)!=required:raise ValueError("Official reference key schema differs")
        n=int(row["y"][0]);k=len(row["C"])
        if row["x"].shape!=(1,n,9) or row["trajectory"].shape!=(51,n,9) or row["raw"].shape!=(n,9):
            raise ValueError("Reference shapes differ")
        for name in ("x","t","C","z","bf16_output","fp32_output","raw","trajectory"):
            if row[name].dtype!=np.float32 or not np.isfinite(row[name]).all():raise ValueError("Invalid reference precision/finiteness: "+name)
        if row["valid"].dtype!=bool or row["known"].dtype!=bool or not row["valid"].all() or not row["known"][0,:k].all() or row["known"][0,k:].any():
            raise ValueError("References must use C-first fully valid original sampling slots")
        if row["raw"].tobytes()!=row["trajectory"][-1].tobytes() or any(v[:k].tobytes()!=row["C"].tobytes() for v in row["trajectory"]):
            raise ValueError("Reference C/raw identity failed")
        refs.append(row);hashes[p.name]=file_sha256(p)
    return refs,hashes


def run(args):
    torch.set_num_threads(2)
    refs,ref_hashes=load_references(args.references)
    stream=HybridStream(args.training_data,args.stream_root,"A")
    samples=stream.next_effective_batch();denominator=sum(s["free_coordinate_count"] for s in samples)
    identity=dict(reference_hashes=ref_hashes,source_start_sha256=file_sha256(args.start),
        data_manifest_sha256=stream.data.manifest_sha256,plan_sha256=stream.plan_sha256,
        shared_first_batch_sha256=stream.last_audit["shared_batch_sha256"],
        scientific_sources=scientific_sources(),precision_criteria=dict(e_inf=5e-5,e_rms=1e-5),
        permutation_seed=2026100750,temporary_WO_seed=2026100701,temporary_WO_std=.01)
    budget=Budget(args.ledger);store=Store(args.out,identity,budget,resume=args.resume)
    store.report["training_fixture"]=dict(arm_local_cursor=1,formal_training_cursor_advanced=False,
        free_coordinate_denominator=denominator,stream_state=stream.state_dict(),
        selected_sample_indices=[s["sample_index"] for s in samples[:2]],
        shared_first_batch_sha256=stream.last_audit["shared_batch_sha256"],OT_counts=stream.counts)
    store.flush()
    try:
        with runtime.gpu_worker(args.ledger,"direct_readout_preflight") as resources:
            store.report["dispatch"]=resources;store.flush()
            for arm,mode in MODES:
                if store.report["arms"].get(arm,{}).get("status")=="PASS":continue
                run_arm(arm,mode,refs,samples,denominator,args,store,budget)
                gc.collect();torch.cuda.empty_cache()
            b=store.report["arms"]["B"]["before"]["readout_sha256"]
            c=store.report["arms"]["C"]["before"]["readout_sha256"]
            store.check("B_C_initialization_identity",dict(B=b,C=c),b==c)
        store.report["status"]="PASS";store.report["completed_arms"]=[x for x,_ in MODES]
        store.report["actual_saved_cells"]=len(store.report["cells"])
        store.report["unexecuted_claims"]=["No training optimizer step in this preflight", "No quality inference from branch norms or attention"]
        store.flush()
    except BaseException as exc:
        store.report["status"]="FAILED"
        store.report["errors"].append(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),time_unix=time.time()))
        store.flush();raise
    return store.report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("start","references","out","ledger","training-data","stream-root"):
        p.add_argument("--"+flag,required=True)
    p.add_argument("--resume",action="store_true",help="Reuse completed immutable cells; refuse partial-cell retries")
    args=p.parse_args(argv)
    result=run(args)
    print(json.dumps(dict(status=result["status"],out=str(Path(args.out).resolve()),budget=result["budget"]),indent=2),flush=True)


if __name__=="__main__":main()
