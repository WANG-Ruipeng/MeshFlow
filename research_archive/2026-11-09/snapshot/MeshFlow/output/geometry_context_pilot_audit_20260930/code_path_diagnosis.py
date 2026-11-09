"""User-requested bounded read-only BF16 code diagnosis; never trains or samples."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from native_t1 import runtime as stable_runtime
import ast, hashlib, importlib.util, json, subprocess, time
import numpy as np
import torch
from native_t1.artifacts import atomic_json, array_hash, file_sha256, state_sha256
from native_t1.baseline import load_baseline
from native_t1.data import load_archive, model_inputs
from native_t1.geometry_gates import fixed_batch
from native_t1.model import native_execution

RUN=ROOT/"native_t1/runs/geometry_gate_code_audit_20260930_150400"
PREVIOUS=ROOT/"native_t1/runs/geometry_context_20260930_142000"
BASE="2502d5f89c54706cc4700d3cdea4a12fd6792529"
FORWARD_CAP=9
ATOL=.015625
RTOL=.03125

def exact(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint8),b.view(np.uint8))

def compare(a,b):
    d=a.astype(np.float32)-b.astype(np.float32)
    allowed=np.float32(ATOL)+np.float32(RTOL)*np.abs(b.astype(np.float32))
    return dict(finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),bitwise=exact(a,b),
        max_abs=float(np.max(np.abs(d))),RMS=float(np.sqrt(np.mean(d.astype(np.float64)**2))),
        violating_elements=int(np.count_nonzero(np.abs(d)>allowed)),elements=a.size,
        atol=ATOL,rtol=RTOL,allclose=bool(np.all(np.abs(d)<=allowed)))

def transform_torch(value,order,corners):
    b,n,d=value.shape
    return value[:,order].reshape(b,n,3,d//3).gather(2,corners[None,:,:,None].expand(b,n,3,d//3)).reshape_as(value)

def transform_numpy(value,order,corners):
    b,n,_=value.shape
    if n==336:
        reshaped=value.reshape(b,112,3,-1)
    elif n==112:
        reshaped=value.reshape(b,112,3,-1)
    else:raise ValueError(value.shape)
    selected=reshaped[:,order]
    return np.take_along_axis(selected,corners[None,:,:,None],axis=2).reshape(value.shape)

def permute_batch(batch,order,corners):
    result=dict(batch)
    for key in ("x0","x1","u","xt","context"):
        result[key]=transform_torch(batch[key],order,corners)
    for key in ("known_mask","valid_mask","loss_mask"):
        result[key]=batch[key][:,order]
    return result

class Trace:
    def __init__(self):
        self.states={};self.records=[]
    def __call__(self,event):
        site=event["site"]
        label=site+"_"+str(event["layer"]) if site=="block" else site
        hidden=event.get("hidden")
        if hidden is not None: self.states[label]=hidden.detach().float().cpu().numpy().copy()
        row=dict(site=label,dtype=str(hidden.dtype) if hidden is not None else None,shape=list(hidden.shape) if hidden is not None else None)
        if "selected_mods" in event:
            row["time_role_selection_exact"]=all(
                torch.equal(m[~event["known_mask"]],f[:,None,:].expand_as(m)[~event["known_mask"]])
                and torch.equal(m[event["known_mask"]],k[:,None,:].expand_as(m)[event["known_mask"]])
                for m,f,k in zip(event["selected_mods"],event["free_mods"],event["known_mods"]))
        row["known_time_exact_one"]=bool((event["face_time"][event["known_mask"]]==1).all())
        self.records.append(row)

def run():
    if RUN.exists():raise FileExistsError("No overwrite or retry: "+str(RUN))
    RUN.mkdir(parents=True)
    old_gate=json.loads((PREVIOUS/"gates.json").read_text())
    previous_registration=json.loads((PREVIOUS/"registration.json").read_text())
    original_bytes=subprocess.check_output(["git","show",BASE+":native_t1/model.py"],cwd=ROOT)
    snapshot=Path(__file__).with_name("base_model_2502_snapshot.py")
    if snapshot.exists():raise FileExistsError(snapshot)
    snapshot.write_bytes(original_bytes)
    source_names=["native_t1/model.py","native_t1/geometry_gates.py","native_t1/data.py","native_t1/runtime.py",
                  "native_t1/baseline.py","models/equidit.py","models/attention.py","models/utils.py"]
    registration=dict(schema="native_t1_code_diagnosis_v1",reason="User requests investigation of code after numerical gate failure",
        no_training=True,no_sampling=True,model_forward_cap=FORWARD_CAP,backward_cap=0,optimizer_updates=0,
        precision="Original BF16 autocast; FP32 parameters/coordinates; no precision intervention",
        thresholds=dict(atol=ATOL,rtol=RTOL),source_commit=BASE,source_pins={n:file_sha256(ROOT/n) for n in source_names},
        original_source_sha256=hashlib.sha256(original_bytes).hexdigest(),script_sha256=file_sha256(__file__),
        checkpoint_sha256=old_gate["initialization"]["checkpoint_sha256"],previous_gate_sha256=file_sha256(PREVIOUS/"gates.json"),
        inputs_sha256=previous_registration["inputs_sha256"],
        tests=["current_reference","current_reference_observed","current_face_only","current_corners_only",
               "current_combined","current_historical_reverse","current_historical_cycle",
               "base_commit_reference","base_commit_combined"],
        interpretation="Measurements only; do not overwrite/relax the failed experimental gate; no automatic experiment restart")
    atomic_json(RUN/"registration.json",registration)
    counts=dict(model_forward_attempts=0,model_forward_returns=0,model_backward_calls=0,optimizer_updates=0,
                complete_rollouts=0,OT_calls=0,standalone_model_submodule_calls=0)
    result=dict(status="RUNNING",counts=counts,comparisons={},calls={},traces={},registration=registration)
    velocities={};trace_arrays={}
    started=time.perf_counter()
    model=None;old_model=None
    def persist():
        result["seconds"]=time.perf_counter()-started
        atomic_json(RUN/"results.json",result)
    persist()
    try:
        archive=load_archive(previous_registration["inputs_path"],expected_sha256=registration["inputs_sha256"])
        cpu=fixed_batch(archive,device="cpu")
        # Rebuilding the no-OT gate does not consume the registered training stream.
        cpu_hashes={key:array_hash(value) for key,value in cpu.items() if torch.is_tensor(value)}
        result["fixed_batch_hashes_match_failed_gate"]=cpu_hashes==old_gate["fixed_batch"]["hashes"]
        if not result["fixed_batch_hashes_match_failed_gate"]:raise RuntimeError("Diagnostic input changed")
        order_cpu=torch.tensor(old_gate["fixed_batch"]["face_permutation"],dtype=torch.long)
        corner_cpu=torch.tensor(old_gate["fixed_batch"]["corner_permutations"],dtype=torch.long)
        identity_cpu=torch.arange(112)
        no_corners=torch.arange(3).expand(112,3)
        intrinsic_corners=corner_cpu[torch.argsort(order_cpu)]
        transformations={
          "reference":(identity_cpu,no_corners),
          "face_only":(order_cpu,no_corners),
          "corners_only":(identity_cpu,intrinsic_corners),
          "combined":(order_cpu,corner_cpu),
          "historical_reverse":(torch.arange(111,-1,-1),no_corners),
          "historical_cycle":(identity_cpu,torch.tensor([2,0,1]).expand(112,3))}
        tr_cp=permute_batch(cpu,*transformations["corners_only"])
        tr_composed=permute_batch(tr_cp,*transformations["face_only"])
        direct=permute_batch(cpu,*transformations["combined"])
        result["combined_equals_face_after_intrinsic_corner"]=all(
            torch.equal(tr_composed[k],direct[k]) for k in ("xt","x0","x1","u","known_mask","context","loss_mask","valid_mask"))
        if not result["combined_equals_face_after_intrinsic_corner"]:raise RuntimeError("Bad split transform")
        registration["transforms"]={name:dict(faces=o.tolist(),corners=c.tolist()) for name,(o,c) in transformations.items()}
        atomic_json(RUN/"registration.json",registration)
        np.savez_compressed(RUN/"fixed_inputs.npz",**{k:v.numpy() for k,v in cpu.items() if torch.is_tensor(v)})
        model,identity=load_baseline()
        result["initialization"]=identity
        if identity["checkpoint_sha256"]!=registration["checkpoint_sha256"]:raise RuntimeError("Wrong baseline")
        result["state_before"]=state_sha256(model)
        if model.context_encoder is not None:raise RuntimeError("Geometry encoder unexpectedly attached")
        spec=importlib.util.spec_from_file_location("native_t1._diagnostic_base2502",snapshot)
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        old_model=original.NativeInpaintingModel(model.backbone,trainable=False)
        with torch.no_grad():old_model.role_embedding.copy_(model.role_embedding)
        result["base_wrapper_state_sha256"]=state_sha256(old_model)
        if result["base_wrapper_state_sha256"]!=result["state_before"]:raise RuntimeError("Original wrapper state differs")
        result["same_backbone_object"]=old_model.backbone is model.backbone
        batch={key:value.to("cuda") if torch.is_tensor(value) else value for key,value in cpu.items()}
        torch.cuda.reset_peak_memory_stats()
        previous=np.load(PREVIOUS/"gate_single_step_proof.npz",allow_pickle=False)
        baseline_trace=None
        def call(label,selected_model,context,transform,observe):
            nonlocal baseline_trace
            if counts["model_forward_attempts"]>=FORWARD_CAP:raise RuntimeError("Forward budget exhausted")
            order,corners=transformations[transform]
            perm=permute_batch(batch,order.to("cuda"),corners.to("cuda"))
            trace=Trace() if observe else None
            counts["model_forward_attempts"]+=1
            call_start=time.perf_counter()
            with torch.no_grad(),context(selected_model,coordinates=(perm["xt"],perm["t"]),autocast=True) as runtime:
                raw=selected_model(*model_inputs(perm),observer=trace)
                counts["model_forward_returns"]+=1
                if not bool(torch.isfinite(raw).all()):raise RuntimeError("Nonfinite diagnostic output")
                velocity=raw.float().cpu().numpy().copy()
            torch.cuda.synchronize()
            velocities[label]=velocity
            result["calls"][label]=dict(seconds=time.perf_counter()-call_start,runtime=runtime,
                hash=array_hash(velocity),precision=str(raw.dtype),transform=transform,observer=observe)
            if "current_reference" in velocities:
                expected=transform_numpy(velocities["current_reference"],order.numpy(),corners.numpy())
                result["comparisons"][label]=compare(velocity,expected)
            if trace is not None:
                if transform=="reference":baseline_trace=trace.states
                records=[]
                for site,value in trace.states.items():
                    expected=transform_numpy(baseline_trace[site],order.numpy(),corners.numpy())
                    records.append(dict(site=site,**compare(value,expected)))
                    if transform in ("reference","combined"):
                        trace_arrays[label+"__"+site]=value
                result["traces"][label]=dict(events=trace.records,permutation_comparisons=records,
                    first_nonbitwise_site=next((r["site"] for r in records if not r["bitwise"]),None))
            persist()
            print(label+" complete",flush=True)
            return velocity
        call("current_reference",model,native_execution,"reference",False)
        result["matches_saved_reference_bitwise"]=exact(velocities["current_reference"],previous["T0_reference"])
        if not result["matches_saved_reference_bitwise"]:raise RuntimeError("Fresh reference does not reproduce saved result")
        call("current_reference_observed",model,native_execution,"reference",True)
        if not exact(velocities["current_reference"],velocities["current_reference_observed"]):
            raise RuntimeError("Observation changed actual output")
        for name in ("face_only","corners_only","combined","historical_reverse","historical_cycle"):
            call("current_"+name,model,native_execution,name,True)
        result["matches_saved_combined_bitwise"]=exact(velocities["current_combined"],previous["T0_permuted"])
        if not result["matches_saved_combined_bitwise"]:raise RuntimeError("Fresh combined result differs from saved result")
        call("base_commit_reference",old_model,original.native_execution,"reference",False)
        call("base_commit_combined",old_model,original.native_execution,"combined",False)
        result["current_vs_base_commit"]=dict(
            reference=compare(velocities["current_reference"],velocities["base_commit_reference"]),
            combined=compare(velocities["current_combined"],velocities["base_commit_combined"]))
        result["state_after"]=state_sha256(model)
        result["base_wrapper_state_after"]=state_sha256(old_model)
        result["parameters_unchanged"]=result["state_after"]==result["state_before"]==result["base_wrapper_state_after"]
        if not result["parameters_unchanged"]:raise RuntimeError("Read-only diagnostic changed parameters")
        result["previous_gate_unchanged"]=file_sha256(PREVIOUS/"gates.json")==registration["previous_gate_sha256"]
        result["source_unchanged"]=all(file_sha256(ROOT/name)==sha for name,sha in registration["source_pins"].items())
        result["status"]="DIAGNOSIS_COMPLETED_EXPERIMENT_REMAINS_STOPPED"
        persist()
    except BaseException as error:
        result.update(status="DIAGNOSTIC_STOPPED",error=repr(error))
        persist();raise
    finally:
        if velocities:
            np.savez_compressed(RUN/"velocities.npz",**velocities)
            result["velocity_proof"]=dict(path=str(RUN/"velocities.npz"),sha256=file_sha256(RUN/"velocities.npz"))
        if trace_arrays:
            np.savez_compressed(RUN/"hidden_trace.npz",**trace_arrays)
            result["trace_proof"]=dict(path=str(RUN/"hidden_trace.npz"),sha256=file_sha256(RUN/"hidden_trace.npz"))
        result["peak_cuda_allocated"]=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
        persist()
        del model,old_model
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
    print(json.dumps(dict(status=result["status"],counts=counts,current_vs_base=result.get("current_vs_base_commit")),ensure_ascii=False))

if __name__=="__main__":run()
