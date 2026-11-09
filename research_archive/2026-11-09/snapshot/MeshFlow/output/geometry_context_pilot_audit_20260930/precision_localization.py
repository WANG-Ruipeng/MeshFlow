"""Four FP32 diagnostic forwards and two original BF16 first-block replays."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from native_t1 import runtime
import json,time
import numpy as np
import torch
from native_t1.artifacts import atomic_json,file_sha256,state_sha256
from native_t1.baseline import load_baseline
from native_t1.model import native_execution,native_block_forward
from native_t1.data import model_inputs
OLD=ROOT/"native_t1/runs/geometry_gate_code_audit_20260930_150400"
OUT=ROOT/"native_t1/runs/geometry_precision_audit_20260930_152000"
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint8),b.view(np.uint8))
def comparison(a,b):
    d=a.astype(np.float64)-b.astype(np.float64)
    return dict(finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),bitwise=exact(a,b),
        max_abs=float(np.abs(d).max()),RMS=float(np.sqrt(np.mean(d*d))),
        BF16_historical_tolerance_allclose=bool(np.all(np.abs(d)<=.015625+.03125*np.abs(b))),
        FP32_diagnostic_allclose_1e5=bool(np.all(np.abs(d)<=1e-5+1e-5*np.abs(b))),
        FP32_tolerance_is_diagnostic_only=True)
def permute_np(a,order,corner,kind="tokens"):
    if kind=="face":return a[:,order]
    if kind=="packed_corner":return np.take_along_axis(a[order],corner[:,:,None],axis=1)
    return np.take_along_axis(a.reshape(1,112,3,-1)[:,order],corner[None,:,:,None],axis=2).reshape(a.shape)
def transformed(batch,order,corner):
    r=dict(batch)
    for key in ("x0","x1","u","xt","context"):
        v=batch[key][:,order].reshape(1,112,3,3)
        r[key]=v.gather(2,corner[None,:,:,None].expand(1,112,3,3)).reshape(1,112,9)
    for key in ("known_mask","valid_mask","loss_mask"):r[key]=batch[key][:,order]
    return r
def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    old=json.loads((OLD/"results.json").read_text())
    reg=json.loads((OLD/"registration.json").read_text())
    counts=dict(model_forward_attempts=0,model_forward_returns=0,model_backward_calls=0,optimizer_updates=0,
                complete_rollouts=0,OT_calls=0,standalone_block_calls=0,standalone_condition_embedder_calls=0)
    result=dict(status="RUNNING",counts=counts,FP32={},block_replay={},comparisons={})
    manifest=dict(schema="fixed_precision_localization_v1",model_forward_cap=4,
        standalone_block_cap=2,standalone_condition_embedder_cap=3,training_updates=0,sampling=0,
        precision_policy="FP32 diagnostic forwards only; two layer replays use unchanged original BF16 policy; no production code or precision switches changed",
        previous_results_sha256=file_sha256(OLD/"results.json"),previous_velocities_sha256=file_sha256(OLD/"velocities.npz"),
        script_sha256=file_sha256(__file__),checkpoint_sha256=old["initialization"]["checkpoint_sha256"],
        source_pins={n:file_sha256(ROOT/n) for n in ("native_t1/model.py","native_t1/runtime.py","native_t1/baseline.py","models/equidit.py","models/attention.py")})
    atomic_json(OUT/"registration.json",manifest)
    tick=time.perf_counter();proof={};block_proof={};handles=[];model=None
    def persist():
        result["seconds"]=time.perf_counter()-tick;atomic_json(OUT/"results.json",result)
    persist()
    try:
        model,identity=load_baseline()
        if identity["checkpoint_sha256"]!=manifest["checkpoint_sha256"]:raise RuntimeError("Wrong baseline")
        result["initialization"]=identity;result["state_before"]=state_sha256(model)
        with np.load(OLD/"fixed_inputs.npz",allow_pickle=False) as saved:
            batch={k:torch.from_numpy(saved[k].copy()).to("cuda") for k in saved.files}
        batch["free_coordinates"]=900
        transformation={k:(torch.tensor(v["faces"],device="cuda"),torch.tensor(v["corners"],device="cuda"))
                         for k,v in reg["transforms"].items()}
        dtypes=[]
        def record_dtype(label):
            def hook(module,args,output):
                dtypes.append(dict(site=label,input=str(args[0].dtype),output=str(output.dtype)))
            return hook
        modules={"QKV":model.backbone.layers[0].attn.qkv_proj,
                 "Q_norm":model.backbone.layers[0].attn.q_norm,
                 "K_norm":model.backbone.layers[0].attn.k_norm,
                 "attention_output":model.backbone.layers[0].attn,
                 "first_MLP_output":model.backbone.layers[0].mlp,
                 "final_head":model.backbone.final_layer.linear}
        for key,module in modules.items():handles.append(module.register_forward_hook(record_dtype(key)))
        def observer(event):
            if "hidden" in event:dtypes.append(dict(site=event["site"]+str(event.get("layer","")),output=str(event["hidden"].dtype)))
        torch.cuda.reset_peak_memory_stats()
        for name in ("reference","face_only","corners_only","combined"):
            dtypes.clear()
            order,corners=transformation[name];b=transformed(batch,order,corners)
            counts["model_forward_attempts"]+=1
            with torch.no_grad(),native_execution(model,coordinates=(b["xt"],b["t"]),autocast=False) as facts:
                raw=model(*model_inputs(b),observer=observer)
                counts["model_forward_returns"]+=1
                if not bool(torch.isfinite(raw).all()):raise RuntimeError("Nonfinite FP32 output")
                proof[name]=raw.float().cpu().numpy().copy()
            if any(r["output"]!="torch.float32" or r.get("input","torch.float32")!="torch.float32" for r in dtypes):
                raise RuntimeError("FP32 diagnostic used non-FP32 observed activations")
            expected=permute_np(proof["reference"],order.cpu().numpy(),corners.cpu().numpy())
            result["FP32"][name]=dict(runtime=facts,dtypes=list(dtypes),permutation=comparison(proof[name],expected))
            persist();print("FP32 "+name+" completed",flush=True)
        for handle in handles:handle.remove()
        handles=[]
        with np.load(OLD/"hidden_trace.npz",allow_pickle=False) as saved:
            h0=torch.from_numpy(saved["current_reference_observed__block_1"].copy()).to("cuda",dtype=torch.bfloat16)
            hP=torch.from_numpy(saved["current_combined__block_1"].copy()).to("cuda",dtype=torch.bfloat16)
            expected_block=[saved["current_reference_observed__block_2"].copy(),saved["current_combined__block_2"].copy()]
        order,corners=transformation["combined"]
        bP=transformed(batch,order,corners)
        with torch.no_grad(),native_execution(model,coordinates=(batch["xt"],batch["t"]),autocast=True):
            c_free=model.backbone.t_embedder(batch["t"]);counts["standalone_condition_embedder_calls"]+=1
            c_known=model.backbone.t_embedder(torch.ones_like(batch["t"]));counts["standalone_condition_embedder_calls"]+=1
            count=model.backbone.y_embedder(batch["y"]//model.backbone.face_bin,False);counts["standalone_condition_embedder_calls"]+=1
            c_free,c_known=c_free+count,c_known+count
        current_record={};block=model.backbone.layers[0]
        hooks={"pooled_face_before_norm":(block.norm1,"pre","face"),
               "normalized_face":(block.norm1,"post","face"),
               "QKV_input":(block.attn.qkv_proj,"pre","face"),
               "QKV_output":(block.attn.qkv_proj,"post","face"),
               "Q_normalized":(block.attn.q_norm,"post","face"),
               "K_normalized":(block.attn.k_norm,"post","face"),
               "SDPA_output_before_projection":(block.attn.out_proj,"pre","face"),
               "attention_after_projection":(block.attn.out_proj,"post","face"),
               "hidden_after_attention":(block.norm2,"pre","tokens"),
               "MLP_output":(block.mlp,"post","tokens")}
        def read_tensor(label,value):
            current_record[label]=dict(dtype=str(value.dtype),value=value.detach().float().cpu().numpy().copy())
        for label,(module,when,kind) in hooks.items():
            if when=="pre":
                def pre(module,args,label=label):read_tensor(label,args[0])
                handles.append(module.register_forward_pre_hook(pre))
            else:
                def post(module,args,output,label=label):read_tensor(label,output)
                handles.append(module.register_forward_hook(post))
        records=[]
        for index,(hidden,b) in enumerate(((h0,batch),(hP,bP))):
            current_record.clear()
            with torch.no_grad(),native_execution(model,coordinates=(b["xt"],b["t"]),autocast=True):
                counts["standalone_block_calls"]+=1
                output=native_block_forward(block,hidden,c_free,c_known,b["known_mask"],b["valid_mask"])
            value=output.float().cpu().numpy()
            if not exact(value,expected_block[index]):raise RuntimeError("Partial block replay differs from observed actual full-model trace")
            records.append(dict(current_record))
            block_proof[("reference" if index==0 else "combined")+"__output"]=value
        details=[]
        for label,(_,_,kind) in hooks.items():
            a=records[1][label]["value"];ref=records[0][label]["value"]
            expected=permute_np(ref,order.cpu().numpy(),corners.cpu().numpy(),kind=kind)
            details.append(dict(site=label,dtype=records[1][label]["dtype"],**comparison(a,expected)))
            for index,key in enumerate(("reference","combined")):
                block_proof[key+"__"+label]=records[index][label]["value"]
        result["block_replay"]=dict(original_trace_both_bitwise=True,sites=details,
            first_nonbitwise_site=next((r["site"] for r in details if not r["bitwise"]),None),
            dtype_note="Saved tensors converted exactly to float32 for NPZ; dtype column records actual operation output")
        result["BF16_saved_comparisons"]={k:old["comparisons"]["current_"+k] for k in ("reference","face_only","corners_only","combined")}
        result["state_after"]=state_sha256(model)
        result["parameters_unchanged"]=result["state_after"]==result["state_before"]
        if not result["parameters_unchanged"]:raise RuntimeError("Diagnostic changed parameters")
        result["source_unchanged"]=all(file_sha256(ROOT/n)==sha for n,sha in manifest["source_pins"].items())
        result["status"]="DIAGNOSIS_COMPLETED"
    except BaseException as error:
        result.update(status="STOPPED",error=repr(error));raise
    finally:
        for handle in handles:handle.remove()
        if model is not None:
            result["no_hooks_remaining"]=all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        if proof:
            np.savez_compressed(OUT/"fp32_velocities.npz",**proof)
            result["FP32_proof_sha256"]=file_sha256(OUT/"fp32_velocities.npz")
        if block_proof:
            np.savez_compressed(OUT/"first_block_proof.npz",**block_proof)
            result["block_proof_sha256"]=file_sha256(OUT/"first_block_proof.npz")
        result["peak_cuda_allocated"]=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
        persist()
        del model
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
    print(json.dumps(dict(status=result["status"],counts=counts,first_difference=result["block_replay"].get("first_nonbitwise_site"),
        FP32={k:v["permutation"] for k,v in result["FP32"].items()}),ensure_ascii=False))
if __name__=="__main__":main()
