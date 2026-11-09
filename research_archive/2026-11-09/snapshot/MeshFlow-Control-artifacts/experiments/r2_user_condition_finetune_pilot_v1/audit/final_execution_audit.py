"""Independent final CPU audit. Run after two completed arms and 84 output receipts.
No model construction, CUDA, OT, sampling or scoring. Each of two final full and
two generator files gets one CPU mmap torch.load; base/recovery remain hash-only.
Not-ready exits without creating reports. Existing reports are never overwritten.
"""
from __future__ import annotations
import argparse, collections, csv, gc, hashlib, json, math, os, time, traceback
from pathlib import Path
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="2"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
VERSION="r2_user_finetune_final_execution_audit_v1"
ARMS=("R2_CONT_OLD","R2_CONT_USER")
LABELS=("R2_FROZEN",)+ARMS
DEFAULT="/mnt/e/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1"
REG_SHA="e0ad17e15915efba7dd5e20c668e99b27ed40e7630fcdaf8bd8690325abd5961"
CODE_SHA="3f7561b2884bb77dc44938279561ed953b977bb96703b54bf90b166295ebe7e7"
PROTOCOL_SHA="a120eda2d82b9a2b4350f4c4792d32bf592894808983c442080edf4342a86678"
BASE={"base_checkpoint_sha256":"89f49e014615eb4a41e052ed90719a2580ead46418997c268e35e09ff37a7541",
 "generator_state_sha256":"0e988aec51a7b9026d0a8bcf1312f2b3ff03873d09a3e75552ddab77af92563f",
 "optimizer_state_sha256":"900ceec2b3ef6458b0a17868a58b2bbfb39a9f61a94859acb35da8ac16c4ccd6",
 "rng_sha256":"4edcac4143f8d44c823b7a8b6787e9330cbb5da697bf32e53ab94e087576305a",
 "stream_state_sha256":"00cbc3bf394cf1505d8b5c61d9979c72d2516ba7afcd62ded6359b1eac40d923"}
REPORTS=("execution_audit.json","ledger_summary.json","execution_ledger.csv","execution_endpoints.csv","execution_outputs.csv","execution_protected_files.csv","execution_checks.csv")

def path(p):
    s=str(p)
    return Path("/mnt/"+s[0].lower()+"/"+s[3:].replace("\\","/")) if os.name!="nt" and len(s)>2 and s[1]==":" else Path(s)
def read(p): return json.loads(path(p).read_text(encoding="utf-8-sig"))
def jhash(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def lines(p):
    with path(p).open(encoding="utf-8-sig") as f: return [json.loads(s) for s in f]
def arrays(p):
    with np.load(path(p),allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def same(a,b): return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def ahash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode("ascii")+str(a.shape).encode("ascii")+a.tobytes()).hexdigest()
def require(ok,reason):
    if not bool(ok): raise ValueError(reason)
def state_hash(state,torch):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        v=v.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str((tuple(v.shape),v.dtype)).encode())
        h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def tree_hash(v,torch):
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            x=x.detach().cpu().contiguous()
            h.update(str(("tensor",str(x.dtype),tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b"dict")
            for k in sorted(x,key=lambda z:(type(z).__name__,repr(z))): add(k); add(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str((type(x).__name__,len(x))).encode())
            for y in x: add(y)
        else: h.update(str((type(x).__name__,repr(x))).encode())
    add(v); return h.hexdigest()

class Audit:
    def __init__(self,root):
        self.root=root; self.started=time.time()
        self.checks=[]; self.ledger=[]; self.outputs=[]; self.endpoints=[]; self.files=[]
        self.sha={}; self.stamps={}; self.expected={}; self.loaded=collections.Counter()
        self.call_paths=set(); self.train={}; self.meta={}
        self.reg=read(root/"registration.json"); self.code=read(root/"code_manifest.json")
        self.ev=read(root/"evaluation/registration.json")
    def ck(self,name,ok,detail=None):
        r={"check":name,"status":"PASS" if bool(ok) else "FAIL"}
        if detail is not None: r["detail"]=detail
        self.checks.append(r); return bool(ok)
    def section(self,name,fn):
        t=time.perf_counter()
        try: fn()
        except Exception as e: self.ck(name,False,{"reason":repr(e),"traceback":traceback.format_exc()})
        print(name,round(time.perf_counter()-t,2),"seconds",flush=True)
    def digest(self,p):
        p=path(p).resolve(); key=str(p); st=p.stat(); stamp=(st.st_size,st.st_mtime_ns)
        if key in self.sha:
            require(stamp==self.stamps[key],"Concurrent file mutation: "+key); return self.sha[key]
        h=hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda:f.read(4<<20),b""): h.update(b)
        st=p.stat(); require(stamp==(st.st_size,st.st_mtime_ns),"Mutation during hash: "+key)
        self.sha[key]=h.hexdigest(); self.stamps[key]=stamp; return h.hexdigest()
    def expect(self,p,sha,origin,size=None):
        key=str(path(p).resolve())
        if key in self.expected:
            require(self.expected[key]["expected_sha256"]==sha,"Conflicting frozen hashes: "+key)
            self.expected[key]["origins"].add(origin); return
        self.expected[key]={"path":key,"expected_sha256":sha,"expected_bytes":size,"origins":{origin}}
    def immutable_inputs(self):
        self.expect(self.root/"registration.json",REG_SHA,"pinned_registration")
        self.expect(self.root/"code_manifest.json",CODE_SHA,"pinned_code_manifest")
        for origin,files in (("old_math",self.reg["source_hashes"]),("pilot_code",self.code["source_hashes"]),
                ("protected_inputs",self.reg["protected_inputs"]),("scorers",self.ev["scorer_sources"]),("eval_source_inputs",self.ev["source_inputs"])):
            for p,h in files.items(): self.expect(p,h,origin)
        for p,x in read(self.root/"protected_before.json")["files"].items():
            self.expect(p,x["sha256"],"protected_before",x.get("bytes"))
        self.expect(self.reg["base_checkpoint"],BASE["base_checkpoint_sha256"],"base_main6000")
        self.expect(self.ev["frozen_checkpoint"]["path"],self.ev["frozen_checkpoint"]["sha256"],"FROZEN_generator")
        r2=Path("/mnt/e/MeshFlow/r2")
        for p,key in ((r2/"pilots/user_condition_finetune_v1/evaluation.py","adapter_sha256"),
                      (r2/"data_tools/user_conditions_v1/render.py","render_helper_sha256"),
                      (r2/"src/meshflow_control/sampling.py","sampling_sha256")):
            self.expect(p,self.ev[key],"eval_implementation")
        receipt=read(self.root/"data/precompute_receipt.json"); reg=read(self.root/"data/input_registration.json")
        self.expect(self.root/"data/input_registration.json",receipt["input_registration_sha256"],"prepared_input_registration")
        self.expect(self.root/"data/paired_plan.json",receipt["plan_sha256"],"paired_plan")
        self.ck("data_source_map_hash",jhash(reg["source_files"])==reg["source_sha256"])
        for key in ("source_files","implementation_sources"):
            for p,h in reg[key].items(): self.expect(p,h,"data_"+key)
        for arm in ARMS:
            p=self.root/"data"/arm/"paired_inputs.json"; self.expect(p,receipt["arm_cache_index_sha256"][arm],"cache_index")
            c=read(p)
            self.ck(arm+".cache8000",len(c["records"])==8000 and [x["sample_index"] for x in c["records"]]==list(range(8000)))
            for x in c["records"]: self.expect(x["sample_npz"],x["sample_npz_sha256"],"frozen_training_batch")
        for q in self.ev["requests"]:
            self.expect(q["input_path"],q["input_sha256"],"inference_C_and_z")
            if q.get("condition_source"): self.expect(q["condition_source"],q["condition_source_sha256"],"DEV_condition_source")
        for q in self.ev["queries"]: self.expect(q["path"],q["sha256"],"query_bank")
        self.ck("evaluation_budget84_68_3400_16",self.ev["labels"]==list(LABELS) and len(self.ev["requests"])==28 and len(self.ev["queries"])==14 and
            all(self.ev["budget"][k]==v for k,v in {"logical_outputs":84,"new_rollouts":68,"sampling_forwards":3400,"old_reuses":16}.items()))
    def call_ledger(self,p,phase,entity,expected):
        p=path(p).resolve(); self.call_paths.add(str(p))
        begins={}; done={}; failed={}; pending=set(); errors=[]
        for r in lines(p):
            ident=r.get("call_id"); ev=r.get("event")
            if r.get("arm")!=entity: errors.append("arm differs")
            if ev=="CALL_BEGIN":
                if ident!=len(begins)+1 or ident in begins or pending: errors.append("unordered/duplicate/concurrent begin")
                begins[ident]=r; pending.add(ident)
            elif ev in ("CALL_DONE","CALL_FAILED"):
                if ident not in pending: errors.append("unmatched terminal event")
                else:
                    if any(r.get(k)!=begins[ident].get(k) for k in ("kind","context","arm","attempt_id")): errors.append("terminal identity")
                    pending.remove(ident)
                (done if ev=="CALL_DONE" else failed)[ident]=r
            else: errors.append("unknown event "+str(ev))
        ac,dc,fc=(collections.Counter(r["kind"] for r in x.values()) for x in (begins,done,failed))
        pc=collections.Counter(begins[i]["kind"] for i in pending)
        for k in sorted(set(expected)|set(ac)|set(dc)|set(fc)):
            self.ledger.append(dict(phase=phase,entity=entity,kind=k,expected=expected.get(k,0),
                attempted=ac[k],successful=dc[k],failed=fc[k],pending=pc[k],ledger_path=str(p)))
        self.ck(phase+".call_ledger:"+entity,not errors and not failed and not pending and
            all(ac[k]==dc[k]==expected.get(k,0) for k in set(ac)|set(expected)),dict(errors=errors[:10],failed=len(failed),pending=sorted(pending)))
        return dict(attempted=dict(ac),successful=dict(dc),failed=dict(fc),pending_call_ids=sorted(pending)),list(begins.values())
    def snapshot(self,name,supplied,actual):
        self.ck(name,all(supplied.get(k)==actual[k] for k in ("attempted","successful","failed","pending_call_ids")))
    def training_arm(self,arm):
        folder=self.root/"training"/arm; c=read(folder/"completed.json")
        actual,b=self.call_ledger(folder/"calls.jsonl","training",arm,dict(train_forward=8000,train_backward=8000,optimizer_update=1000))
        self.train[arm]=actual; self.snapshot(arm+".completed_ledger",c["calls"],actual)
        self.ck(arm+".completed1000_global7000",c["status"]=="COMPLETE" and c["arm"]==arm and c["base_global_step"]==6000 and
            c["finetune_step"]==1000 and c["global_step"]==7000 and c["physical_update_attempts"]==c["physical_successful_updates"]==1000)
        expected=[(k,s,6000+s,m,(s-1)*8+m) for s in range(1,1001) for m in range(8) for k in ("train_forward","train_backward")]
        observed=[(r["kind"],r["context"].get("local_step"),r["context"].get("global_step"),r["context"].get("microbatch"),r["context"].get("sample_index"))
            for r in b if r["kind"] in ("train_forward","train_backward")]
        updates=[(r["context"].get("local_step"),r["context"].get("global_step")) for r in b if r["kind"]=="optimizer_update"]
        pattern=[k for _ in range(8) for k in ("train_forward","train_backward")]+["optimizer_update"]
        self.ck(arm+".exact_once_cursors_and_order",observed==expected and updates==[(s,6000+s) for s in range(1,1001)] and [r["kind"] for r in b]==pattern*1000)
        inits=list((folder/"attempts").glob("*/initialization_audit.json"))
        self.ck(arm+".initialization_evidence",bool(inits) and all(read(p)==c["lineage"] for p in inits))
    def preflight(self):
        c=read(self.root/"preflight/completed.json")
        actual,b=self.call_ledger(self.root/"preflight/calls.jsonl","preflight","PREFLIGHT",dict(preflight_forward=8,preflight_backward=8,optimizer_update=0))
        self.snapshot("preflight.receipt_ledger",c["calls"],actual)
        initial=dict(generator=BASE["generator_state_sha256"],optimizer=BASE["optimizer_state_sha256"])
        self.ck("preflight8F8B0step_unchanged_state_and_RNG",c["status"]=="PASS" and c["before"]==c["after"]==initial and
            c["optimizer_updates"]==0 and c["hashes_unchanged"] and c["rng_restored"] and c["inherited_rng_sha256"]==BASE["rng_sha256"] and
            c["successful_fixed_samples"]==c["fixed_sample_count"]==8 and c["source_registration_sha256"]==REG_SHA and c["pilot_code_manifest_sha256"]==CODE_SHA)
        self.ck("preflight.order",[(r["kind"],r["context"]["preflight_index"]) for r in b]==[(k,i) for i in range(8) for k in ("preflight_forward","preflight_backward")])
    def load_once(self,p,sha,torch):
        key=str(path(p).resolve()); require(not self.loaded[key],"Second torch.load prohibited")
        require(self.digest(p)==sha,"Checkpoint file SHA mismatch before load")
        self.loaded[key]+=1
        return torch.load(path(p),map_location="cpu",weights_only=True,mmap=True)
    def endpoint(self,arm):
        import torch
        require(not torch.cuda.is_initialized(),"CUDA unexpectedly initialized")
        base=read(self.root/"audit/base_payload_cpu.json")
        self.ck("base_CPU194_steps6000",base["status"]=="PASS" and base["optimizer_tensor_states"]==194 and base["optimizer_step_values"]==[6000.0] and
            all(base[k]==BASE[k] for k in ("generator_state_sha256","optimizer_state_sha256","rng_sha256","stream_state_sha256")))
        c=read(self.root/"training"/arm/"completed.json"); fr,er=c["checkpoint"],c["inference_export"]
        fp=self.root/"checkpoints"/(arm+"_local1000_global7000.pt"); ep=self.root/"inference"/(arm+"_generator_step7000.pt")
        self.ck(arm+".endpoint_paths_sidecars",path(fr["path"]).resolve()==fp and path(er["path"]).resolve()==ep and
            read(fp.with_suffix(".json"))==fr and read(ep.with_suffix(".json"))==er)
        rp=self.root/"checkpoints"/(arm+"_local0500_global6500.pt"); rr=read(rp.with_suffix(".json"))
        self.expect(rp,rr["sha256"],"recovery500")
        self.ck(arm+".recovery500_identity",rr["arm"]==arm and rr["finetune_step"]==500 and rr["global_step"]==6500)
        self.expect(fp,fr["sha256"],"final_full_checkpoint")
        self.expect(ep,er["sha256"],"final_generator_export")
        p=self.load_once(fp,fr["sha256"],torch)
        try:
            self.ck(arm+".full_schema",p["schema"]=="r2_user_condition_finetune_training_v1" and p["experiment"]==self.reg["experiment"] and
                p["arm"]==arm and p["base_global_step"]==6000 and p["finetune_step"]==1000 and p["global_step"]==7000 and p["update_in_progress"] is False)
            h=dict(generator_state_sha256=state_hash(p["generator_state"],torch),optimizer_state_sha256=tree_hash(p["optimizer"],torch),
                rng_sha256=tree_hash(p["rng"],torch),stream_state_sha256=jhash(p["stream_state"]))
            self.ck(arm+".full_payload_subtree_hashes",all(p[k]==fr[k]==v for k,v in h.items()))
            self.ck(arm+".protocol_identity",p["protocol_sha256"]==jhash(p["protocol"])==PROTOCOL_SHA)
            ids,lin,stream=p["identities"],p["lineage"],p["stream_state"]
            self.ck(arm+".lineage_main6000_inherited",all(lin[k]==v for k,v in BASE.items()) and lin["global_step"]==6000 and
                lin["inherited_rng"] is True and lin["old_stream_is_provenance_only"] is True and lin["new_stream_cursor"]==0 and
                path(lin["base_checkpoint"]).resolve()==path(self.reg["base_checkpoint"]).resolve())
            self.ck(arm+".full_identities",ids==c["identities"] and jhash(lin)==jhash(c["lineage"]) and ids["registration_sha256"]==REG_SHA and
                ids["pilot_code_manifest_sha256"]==CODE_SHA and ids["training_sources"]==self.reg["source_hashes"] and ids["protocol_sha256"]==PROTOCOL_SHA)
            self.ck(arm+".stream_final_cursor",stream["schema"]=="r2_user_finetune_stream_state_v1" and stream["batch_index"]==stream["finetune_step"]==1000 and
                stream["sample_index"]==8000 and stream["global_step"]==7000 and stream["identities"]==ids["data"] and ids["data"]["arm"]==arm)
            for k,file in (("input_registration_sha256",self.root/"data/input_registration.json"),("plan_sha256",self.root/"data/paired_plan.json"),
                ("paired_inputs_sha256",self.root/"data"/arm/"paired_inputs.json")): self.ck(arm+"."+k,ids["data"][k]==self.digest(file))
            self.snapshot(arm+".full_execution_ledger",p["execution_receipt"],self.train[arm])
            names=p["optimizer_parameter_names"]; opt=p["optimizer"]; gen=p["generator_state"]
            groups=opt["param_groups"]; flat_names=[n for g in names for n in g]; flat_ids=[i for g in groups for i in g["params"]]
            mapping=(names==base["optimizer_parameter_names"]==lin["optimizer_parameter_names"] and len(names)==len(groups) and
                len(flat_names)==len(flat_ids)==len(set(flat_ids))==len(opt["state"])==len(gen)==194 and set(flat_ids)==set(opt["state"]))
            self.ck(arm+".AdamW_mapping194",mapping)
            hp=[{k:v for k,v in g.items() if k!="params"} for g in groups]
            self.ck(arm+".original_AdamW_hyperparameters",jhash(hp)==jhash(lin["optimizer_hyperparameters"]) and
                all(g["lr"]==1e-5 and list(g["betas"])==[.9,.95] and g["eps"]==1e-8 and g["weight_decay"]==0 for g in groups))
            steps=[]; ok=mapping
            for name,i in zip(flat_names,flat_ids):
                tensor=gen[name.removeprefix("generator.")]; st=opt["state"][i]; step=st["step"]
                steps.append(float(step.item()) if torch.is_tensor(step) else float(step))
                ok=ok and set(st)=={"step","exp_avg","exp_avg_sq"} and tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all())
                for k in ("exp_avg","exp_avg_sq"):
                    v=st[k]; ok=ok and v.device.type=="cpu" and v.dtype==torch.float32 and v.shape==tensor.shape and bool(torch.isfinite(v).all())
            self.ck(arm+".AdamW_all194_steps7000_finite",ok and len(steps)==194 and set(steps)=={7000.0},dict(states=len(steps),steps=sorted(set(steps))))
            self.ck(arm+".no_teacher_aligner",p.get("teacher_state") is None and p.get("aligner_state") is None)
            keep={k:p[k] for k in ("experiment","arm","base_global_step","finetune_step","global_step","model_config","protocol","protocol_sha256","identities","lineage")}
            self.meta[arm]=dict(full_sha256=fr["sha256"],export_sha256=er["sha256"],**h,**keep)
            self.endpoints.append(dict(arm=arm,kind="full",path=str(fp),file_sha256=fr["sha256"],global_step=7000,finetune_step=1000,optimizer_states=len(steps),optimizer_steps=sorted(set(steps)),**h))
        finally:
            del p
            gen=opt=groups=tensor=st=v=step=None
            gc.collect()
        e=self.load_once(ep,er["sha256"],torch)
        try:
            eh=state_hash(e["generator_state"],torch)
            self.ck(arm+".export_matches_full",e["schema"]=="r2_user_condition_finetune_generator_v1" and all(e[k]==v for k,v in keep.items()) and
                eh==e["generator_state_sha256"]==er["generator_state_sha256"]==h["generator_state_sha256"] and e["training_checkpoint_sha256"]==fr["sha256"])
            self.ck(arm+".export_contract",e["inference_contract"]==dict(readout_mode="none",teacher=False,aligner=False,forward="BF16",parameters="FP32",integration="FP32") and
                not ({"optimizer","rng","teacher_state","aligner_state"}&set(e)))
            self.endpoints.append(dict(arm=arm,kind="generator",path=str(ep),file_sha256=er["sha256"],global_step=7000,finetune_step=1000,generator_state_sha256=eh,training_checkpoint_sha256=fr["sha256"]))
        finally: del e; gc.collect()
    def sampling_ledger(self,p,label,q,sha):
        p=path(p).resolve(); self.call_paths.add(str(p))
        ctx=dict(label=label,request_id=q["request_id"],input_sha256=q["input_sha256"],checkpoint_sha256=sha,phase="pilot")
        counts=collections.Counter(); errors=[]; pending=[]; registered=complete=roll_failed=0
        for r in lines(p):
            ev,k=r.get("event"),r.get("kind")
            if ev in ("CELL_REGISTERED","RESERVE","ROLLOUT_COMPLETE") and any(r.get("context",{}).get(k)!=v for k,v in ctx.items()): errors.append("context mismatch")
            if ev=="CELL_REGISTERED": registered+=1
            elif ev=="RESERVE":
                counts[(k,"attempted")]+=1
                if r.get("ordinal")!=counts[(k,"attempted")]: errors.append("ordinal")
                if k=="sampling_forward":
                    if pending or r["context"].get("euler_step")!=counts[(k,"attempted")]-1: errors.append("Euler order")
                    pending.append(k)
                elif k!="rollout_attempt": errors.append("unknown kind")
            elif ev in ("RETURN","ERROR"):
                if ev=="ERROR" and k=="rollout_attempt": roll_failed+=1
                else:
                    if pending!=[k]: errors.append("unmatched return/error")
                    pending.clear(); counts[(k,"successful" if ev=="RETURN" else "failed")]+=1
            elif ev=="ROLLOUT_COMPLETE":
                complete+=1
                if r.get("nfe")!=50 or r.get("known_all51_bitwise") is not True or pending: errors.append("complete marker")
            else: errors.append("unknown event")
        counts[("rollout_attempt","successful")]=complete; counts[("rollout_attempt","failed")]=roll_failed
        for k,n in (("rollout_attempt",1),("sampling_forward",50)):
            a,d,f=(counts[(k,x)] for x in ("attempted","successful","failed")); remain=a-d-f
            self.ledger.append(dict(phase="sampling",entity=label+"/"+q["request_id"],kind=k,expected=n,attempted=a,successful=d,failed=f,pending=remain,ledger_path=str(p)))
            if (a,d,f,remain)!=(n,n,0,0): errors.append(k+" count")
        self.ck("sampling_ledger:"+label+"/"+q["request_id"],registered==1 and complete==1 and not errors,errors[:10])
    def output(self,label,q,target):
        rid=q["request_id"]; p=self.root/"evaluation/outputs"/label/rid/"receipt.json"; r=read(p)
        reuse=label=="R2_FROZEN" and q["panel"]=="OLDREG"
        row=dict(label=label,request_id=rid,panel=q["panel"],N=q["N"],K=q["K"],generation_status=r["generation_status"])
        start=len(self.checks)
        keys=("request_id","parent_id","condition_id","panel","candidate","region_type","tier","N","K","ratio","input_sha256","C_sha256","Gaussian_sha256")
        self.ck("receipt:"+label+"/"+rid,r["status"]=="COMPLETE" and r["label"]==label and all(r[k]==q[k] for k in keys) and
            r["checkpoint_sha256"]==target["sha256"] and r["new_rollouts"]==(0 if reuse else 1) and
            r["generation_status"]==("VERIFIED_HISTORICAL_RAW_REUSE" if reuse else "NEW_EULER50"))
        for k in ("raw","ply","result"): self.expect(r[k+"_path"],r[k+"_sha256"],"output_artifact")
        self.expect(p,self.digest(p),"output_receipt_snapshot")
        if reuse:
            hist=q["historical_receipt"]
            self.ck("reuse:"+rid,all(r[k+s]==hist[k+s] for k in ("raw","ply","result") for s in ("_path","_sha256")) and hist["checkpoint_sha256"]==target["sha256"] and not p.with_name("calls.jsonl").exists())
            self.ledger.append(dict(phase="historical_reuse",entity=label+"/"+rid,kind="reused_output",expected=1,attempted=0,successful=1,failed=0,pending=0,ledger_path="Historical receipt; no current calls"))
        else: self.sampling_ledger(p.with_name("calls.jsonl"),label,q,target["sha256"])
        b,a=arrays(q["input_path"]),arrays(r["raw_path"]); n,k=q["N"],q["K"]
        self.ck("input_arrays:"+rid,b["C"].shape==(k,3,3) and b["z"].shape==(1,n,9) and b["C"].dtype==b["z"].dtype==np.float32 and
            int(b["N"])==n and int(b["K"])==k and ahash(b["C"])==q["C_sha256"] and ahash(b["z"])==q["Gaussian_sha256"] and
            np.isfinite(b["C"]).all() and np.isfinite(b["z"]).all() and not np.any(b["z"][:,:k]))
        raw=np.asarray(a.get("raw",a.get("output"))).reshape(n,3,3)
        traj=np.asarray(a.get("trajectory",a.get("path"))).reshape(51,n,3,3)
        c=np.asarray(a["C"]).reshape(k,3,3); z=np.asarray(a.get("z",a.get("initial_gaussian"))).reshape(1,n,9)
        facts=dict(C_output_bitwise=same(raw[:k],b["C"]),C_all51_bitwise=all(same(t[:k],b["C"]) for t in traj),
            actual_Gaussian_bitwise=same(z,b["z"]),NPZ_C_bitwise=same(c,b["C"]),trajectory_final_is_RAW=same(traj[-1],raw),
            initial_free_is_Gaussian=same(traj[0,k:].reshape(1,n-k,9),b["z"][:,k:]),N_preserved=len(raw)==n,
            finite=bool(all(np.isfinite(x).all() for x in (raw,traj,c,z))),FP32=raw.dtype==traj.dtype==c.dtype==z.dtype==np.float32)
        row.update(facts); self.ck("RAW_trajectory:"+label+"/"+rid,all(facts.values()),facts)
        self.ck("producer_contract:"+label+"/"+rid,all(r["contracts"].get(k)==v for k,v in facts.items()))
        run=read(r["result_path"]); run=run.get("sampling",run)
        self.ck("Euler50_BF16_FP32:"+label+"/"+rid,run.get("nfe")==50 and run.get("readout_mode")=="none" and all(x in str(run.get("dtype","")).upper() for x in ("BF16","FP32")))
        row["status"]="PASS" if all(x["status"]=="PASS" for x in self.checks[start:]) else "FAIL"; self.outputs.append(row)
    def all_outputs(self):
        expected=set()
        for label in LABELS:
            binding=read(self.root/"evaluation/checkpoints"/(label+".json"))
            target=self.ev["frozen_checkpoint"] if label=="R2_FROZEN" else read(self.root/"training"/label/"completed.json")["inference_export"]
            self.ck("binding:"+label,binding["label"]==label and binding["sha256"]==target["sha256"] and
                path(binding["path"]).resolve()==path(target["path"]).resolve() and binding["evaluation_registration_sha256"]==self.digest(self.root/"evaluation/registration.json"))
            for q in self.ev["requests"]:
                expected.add(str((self.root/"evaluation/outputs"/label/q["request_id"]/"receipt.json").resolve()))
                start=len(self.outputs)
                self.section("output:"+label+"/"+q["request_id"],lambda label=label,q=q,target=target:self.output(label,q,target))
                if len(self.outputs)==start: self.outputs.append(dict(label=label,request_id=q["request_id"],panel=q["panel"],status="FAIL",reason="See execution_checks"))
        actual={str(p.resolve()) for p in (self.root/"evaluation/outputs").rglob("receipt.json")}
        self.ck("exact84_receipts",actual==expected and len(expected)==84,dict(extra=sorted(actual-expected),missing=sorted(expected-actual)))
        counts=collections.Counter(r.get("generation_status") for r in self.outputs)
        self.ck("68new_16reuse",counts=={"NEW_EULER50":68,"VERIFIED_HISTORICAL_RAW_REUSE":16},dict(counts))
    def protected(self):
        for key,r in sorted(self.expected.items()):
            try:
                actual=self.digest(key); size=path(key).stat().st_size
                ok=actual==r["expected_sha256"] and (r["expected_bytes"] is None or size==r["expected_bytes"]); err=""
            except Exception as e: actual=size=None; ok=False; err=repr(e)
            self.files.append({**{k:v for k,v in r.items() if k!="origins"},"origins":"|".join(sorted(r["origins"])),
                "actual_sha256":actual,"bytes":size,"status":"PASS" if ok else "FAIL","error":err})
        bad=[r["path"] for r in self.files if r["status"]!="PASS"]
        self.ck("frozen_files_changes_zero",not bad,dict(unique_files=len(self.files),changed_or_missing=len(bad),paths=bad))
    def final(self):
        found={str(p.resolve()) for p in self.root.rglob("calls.jsonl")}
        self.ck("all_call_ledgers_accounted",found==self.call_paths,dict(unexpected=sorted(found-self.call_paths),missing=sorted(self.call_paths-found)))
        self.totals={}
        for r in self.ledger:
            t=self.totals.setdefault(r["phase"]+"/"+r["kind"],{k:0 for k in ("expected","attempted","successful","failed","pending")})
            for k in t: t[k]+=r[k]
        expected={"training/train_forward":16000,"training/train_backward":16000,"training/optimizer_update":2000,
            "preflight/preflight_forward":8,"preflight/preflight_backward":8,"preflight/optimizer_update":0,"sampling/rollout_attempt":68,"sampling/sampling_forward":3400}
        for key,n in expected.items(): self.ck("total:"+key,self.totals.get(key)==dict(expected=n,attempted=n,successful=n,failed=0,pending=0),self.totals.get(key))
        self.ck("16_reuse_zero_current_calls",self.totals.get("historical_reuse/reused_output")==dict(expected=16,attempted=0,successful=16,failed=0,pending=0))
        self.ck("both_arms_identical_base_lineage",len(self.meta)==2 and self.meta[ARMS[0]]["lineage"]==self.meta[ARMS[1]]["lineage"])
        changed=[]
        for p,stamp in self.stamps.items():
            try:
                st=path(p).stat()
                if stamp!=(st.st_size,st.st_mtime_ns): changed.append(p)
            except OSError: changed.append(p)
        self.ck("hashed_files_stable_during_audit",not changed,changed)
        import torch
        self.ck("four_CPU_mmap_loads_once_no_CUDA",len(self.loaded)==4 and set(self.loaded.values())=={1} and not torch.cuda.is_initialized(),dict(self.loaded))
    def save(self,dest):
        failures=[r for r in self.checks if r["status"]!="PASS"]; status="FAIL" if failures else "PASS"
        limitations=[
            "CPU artifact/ledger audit only: no numerical replay, model construction, CUDA context, model forward/backward, optimizer update, OT, generation or scoring.",
            "Base step6000 tensors are not loaded again: pinned full-file SHA and existing CPU payload audit anchor branch lineage.",
            "Recovery500 files use file SHA and sidecar checks only; two final full and two generator payloads each receive one CPU mmap torch.load.",
            "Persisted ledgers cannot prove absence of undocumented external model calls or files outside the registered experiment.",
            "Bitwise C checks establish coordinates/corner order, not original vertex-ID topology or generated geometry quality.",
            "RAW and trajectory arrays are independently checked. PLY bytes are hash-checked but not independently geometrically parsed.",
            "Full-file SHA covers serialized payload identity; generator/AdamW/RNG/stream hashes are independently recomputed.",
            "Current endpoint RNG hashes need not equal each other after branches process different K-dependent shapes."]
        report=dict(schema=VERSION,status=status,root=str(self.root),created_unix=time.time(),elapsed_seconds=time.time()-self.started,
            audit_script_sha256=self.digest(__file__),resources=dict(processes=1,CPU_threads=2,CUDA_visible_devices="",model_constructions=0,model_calls=0,optimizer_updates=0),
            limitations=limitations,checks=self.checks,failures=failures,ledger_totals=getattr(self,"totals",{}),endpoints=self.endpoints,
            output_count=len(self.outputs),output_contract_pass_count=sum(r["status"]=="PASS" for r in self.outputs),
            protected_file_count=len(self.files),protected_changed_or_missing=sum(r["status"]!="PASS" for r in self.files),
            torch_load_counts=dict(self.loaded),unique_file_digest_count=len(self.sha),
            payload_read_policy="CPU mmap=True, weights_only=True, each final full/export loaded once; unique file SHA pass cached; base/recovery hash-only")
        ledger=dict(schema=VERSION+"_ledger_summary",status=status,totals=getattr(self,"totals",{}),rows=self.ledger,
            reuse_note="16 historical outputs count toward 84 logical outputs; old model calls excluded from current 68 rollouts/3400 sampling forwards.")
        dest.mkdir(parents=True,exist_ok=True)
        for name,value in (("execution_audit.json",report),("ledger_summary.json",ledger)):
            with (dest/name).open("x",encoding="utf8") as f: json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False); f.write("\n")
        for name,rows in (("execution_ledger.csv",self.ledger),("execution_endpoints.csv",self.endpoints),
            ("execution_outputs.csv",self.outputs),("execution_protected_files.csv",self.files),("execution_checks.csv",self.checks)):
            fields=list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
            with (dest/name).open("x",newline="",encoding="utf8") as f:
                w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
                w.writerows({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
        print(json.dumps(dict(status=status,checks=len(self.checks),failures=len(failures),outputs=len(self.outputs),reports=str(dest))),flush=True)
        return 1 if failures else 0

def ready(root):
    missing=[]
    for arm in ARMS:
        p=root/"training"/arm/"completed.json"
        if not p.exists() or read(p).get("status")!="COMPLETE": missing.append(str(p))
    er=root/"evaluation/registration.json"
    if not er.exists(): missing.append(str(er))
    else:
        for label in LABELS:
            for q in read(er)["requests"]:
                p=root/"evaluation/outputs"/label/q["request_id"]/"receipt.json"
                if not p.exists() or read(p).get("status")!="COMPLETE": missing.append(str(p))
    require(not missing,"NOT_READY; no reports written: "+json.dumps(missing))
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--out",default=DEFAULT)
    ap.add_argument("--report-dir",help="Default OUT/audit; must be inside OUT. Existing reports are never overwritten.")
    args=ap.parse_args(); root=path(args.out).resolve(); dest=path(args.report_dir).resolve() if args.report_dir else root/"audit"
    require(root.name=="r2_user_condition_finetune_pilot_v1" and root.parent.name=="experiments","Wrong experiment root")
    require(dest.is_relative_to(root),"Report directory must remain in OUT")
    require(not any((dest/n).exists() for n in REPORTS),"Refuse report overwrite; choose a fresh --report-dir")
    ready(root)
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    require(not torch.cuda.is_initialized(),"CUDA initialized unexpectedly")
    a=Audit(root); a.section("immutable_inputs",a.immutable_inputs)
    for arm in ARMS: a.section("training:"+arm,lambda arm=arm:a.training_arm(arm))
    a.section("preflight",a.preflight)
    for arm in ARMS: a.section("endpoint:"+arm,lambda arm=arm:a.endpoint(arm))
    a.section("all_outputs",a.all_outputs); a.section("protected",a.protected); a.section("final",a.final)
    return a.save(dest)
if __name__=="__main__": raise SystemExit(main())
