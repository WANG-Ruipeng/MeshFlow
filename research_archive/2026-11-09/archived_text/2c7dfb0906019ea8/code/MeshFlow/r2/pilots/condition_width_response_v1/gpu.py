"""Bounded EXP67 inference; unchanged sampler, explicit response probes, one D0."""
from __future__ import annotations
import argparse, datetime, json, os, time, traceback
from contextlib import contextmanager
from pathlib import Path
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key,"2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
from . import common as C
LABEL="EXP67_FROZEN"
STEPS=(5,15,30,45)
def event(name,**fields):
    p=C.OUT/"execution"/"events.jsonl";p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf8") as f:
        f.write(json.dumps(dict(event=name,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields),allow_nan=False)+"\n")
        f.flush();os.fsync(f.fileno())
@contextmanager
def gpu_owner():
    import fcntl
    lock=C.OUT/"execution/gpu_owner.lock";lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("a+") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("Another worker owns this diagnostic GPU stage")
        event("GPU_OWNER_ACQUIRED",pid=os.getpid())
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def reg(): return C.read(C.OUT/"registration.json")
def old(): return C.module("_width_original_evaluation",C.ROOT/"r2/pilots/user_condition_finetune_v1/evaluation.py")
def arrays(p): return C.arrays(p)
def save(p,a): return C.frozen_npz(p,a)
def snap(tag,need_mib=0):
    C.package()
    from meshflow_control.denoise import resources
    a=resources()
    C.write(C.OUT/"resources"/(tag+"_"+str(time.time_ns())+".json"),a)
    if a["gpu"]["returncode"]!=0: raise RuntimeError("Resource observation failed")
    v=a["gpu"]["stdout"].strip().splitlines()[0].split(",")
    total,free=float(v[1]),float(v[3]);headroom=max(2048.,.1*total)
    if free-headroom<need_mib: raise RuntimeError("Insufficient headroom; no additional scientific call")
    return a
def source(q,c):
    if C.sha(q["condition_source"])!=q["condition_source_sha256"]: raise ValueError("Source changed")
    from meshflow_control.source_winding import validate_source
    a=arrays(q["condition_source"]);validate_source(a,c)
    return a
def raw_record(q):
    return C.OUT/"evaluation/raw"/LABEL/q["request_id"]/"receipt.json"
def read_raw(q,binding):
    r=C.read(raw_record(q));old().verify_output(q,r,binding["sha256"])
    from .verify_saved import verify_raw
    verify_raw(q,r,binding)
    return r
def collect_raw():
    r=reg();rows=[read_raw(q,r["checkpoint"]) for q in r["requests"]]
    if len(rows)!=48:raise ValueError("Expected all 48 RAW")
    C.write(C.OUT/"evaluation/raw_index.json",dict(status="COMPLETE",rows=rows),replace=True)
    return rows
def own_code_identity():
    paths=[Path(__file__),C.ROOT/"r2/src/meshflow_control/sampling.py",C.ROOT/"r2/src/meshflow_control/models/native.py",
       C.ROOT/"r2/src/meshflow_control/models/geometry.py",C.ROOT/"r2/src/meshflow_control/runtime.py",
       C.ROOT/"r2/src/meshflow_control/denoise_model.py",C.ROOT/"r2/src/meshflow_control/geo_c.py",
       C.ROOT/"r2/src/meshflow_control/source_winding.py",C.ROOT/"r2/src/meshflow_control/winding_solver.py"]
    paths += [C.CODE/"common.py",C.CODE/"prepare.py",C.CODE/"verify_saved.py",
        C.ROOT/"r2/pilots/user_condition_finetune_v1/evaluation.py",
        C.ROOT/"r2/pilots/user_data_scaling_32_67_v1/training.py",
        C.ROOT/"r2/pilots/user_data_scaling_32_67_v1/training_data.py"]
    paths += list((C.ROOT/"r2/src/meshflow_control").rglob("*.py"))
    return {str(p):C.sha(p) for p in sorted(set(paths))}
def freeze_code():
    C.write(C.OUT/"execution/gpu_code_freeze.json",dict(registration_sha256=C.sha(C.OUT/"registration.json"),files=own_code_identity()))
def check_code():
    a=C.read(C.OUT/"execution/gpu_code_freeze.json")
    if a["registration_sha256"]!=C.sha(C.OUT/"registration.json") or a["files"]!=own_code_identity():
        raise ValueError("GPU implementation changed after freeze")
def sample_and_probe():
    check_code();r=reg();C.package()
    from meshflow_control import runtime
    import torch
    from meshflow_control.sampling import sample
    runtime.configure_stable_runtime();torch.set_num_threads(2)
    pending=[q for q in r["requests"] if not raw_record(q).exists()]
    for q in r["requests"]:
        if raw_record(q).exists(): read_raw(q,r["checkpoint"])
        elif raw_record(q).parent.exists():raise RuntimeError("Incomplete rollout; no silent replay")
    probe_pending=[]
    for q in r["requests"]:
        if q["variant"]=="base":
            for step in STEPS:
                dest=C.OUT/"probes"/q["base_request_id"]/("step_"+str(step).zfill(2))
                if (dest/"receipt.json").exists():
                    from .verify_saved import verify_probe
                    verify_probe(q,step,dest,r["checkpoint"])
                else:
                    if dest.exists():raise RuntimeError("Incomplete probe; no silent replay")
                    probe_pending.append((q,step))
    if not pending and not probe_pending:
        collect_raw();return dict(status="COMPLETE",new_calls=0)
    snap("generator_before_load",2600.)
    loader=C.module("_width_original_generator_loader",C.ROOT/"r2/pilots/user_data_scaling_32_67_v1/training.py")
    model,identity=loader.load_generator(C.resolve(r["checkpoint"]["path"]),device="cuda",expected_sha256=r["checkpoint"]["sha256"])
    if identity["arm"]!="USER_EXPANDED_67" or identity["global_step"]!=6000:raise ValueError("Wrong endpoint")
    C.write(C.OUT/"execution/generator_identity.json",identity)
    snap("generator_resident")
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    tick=time.perf_counter();nroll=0;nprobe=0
    class Budget(old().CellBudget):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);self.velocities=[]
        def call(self,kind,fn,context=None):
            v=super().call(kind,fn,context)
            if kind=="sampling_forward":self.velocities.append(v.detach().float().cpu().numpy()[0])
            return v
    try:
        for q in pending:
            b=old().read_input(q);source(q,b["C"])
            dest=raw_record(q).parent
            ctx=dict(label=LABEL,request_id=q["request_id"],input_sha256=q["input_sha256"],checkpoint_sha256=r["checkpoint"]["sha256"])
            budget=Budget(dest/"calls.jsonl",ctx)
            _,path,audit=sample(model,b["C"],int(q["N"]),seed=q["noise_seed"],z=b["z"],budget=budget,out=dest,context=ctx)
            path=path.reshape(51,q["N"],9);K=q["K"]
            velocities=np.stack(budget.velocities)
            save(dest/"velocities.npz",dict(velocity=velocities,source_dtype=np.array("torch.bfloat16")))
            if not all(C.same(path[s+1,K:],path[s,K:]+v[K:]*np.float32(1/50)) for s,v in enumerate(velocities)):
                raise ValueError("Euler recurrence mismatch")
            rec=dict(q,**ctx,status="COMPLETE",new_rollouts=1,generation_status="NEW_EULER50",
                velocities_path=str(dest/"velocities.npz"),velocities_sha256=C.sha(dest/"velocities.npz"))
            for field,name in (("raw","raw.npz"),("ply","raw.ply"),("result","run.json")):
                rec[field+"_path"]=str(dest/name);rec[field+"_sha256"]=C.sha(dest/name)
            _,rec["contracts"]=old().verify_output(q,rec,r["checkpoint"]["sha256"])
            C.write(dest/"receipt.json",rec);nroll+=1
            if nroll==1 or nroll%8==0:snap("generator_after_rollout_"+str(nroll))
            C.write(C.OUT/"execution/progress.json",dict(stage="RAW",completed=sum(raw_record(v).exists() for v in r["requests"]),
                registered=48,seconds=time.perf_counter()-tick,last_sampling=audit),replace=True)
            print(json.dumps(dict(stage="RAW",completed=nroll,request_id=q["request_id"],seconds=audit["seconds"])),flush=True)
        collect_raw()
        by={(q["base_request_id"],q["variant"]):q for q in r["requests"]}
        from meshflow_control.models.native import native_execution
        for base,step in probe_pending:
            brec=read_raw(base,r["checkpoint"])
            trajectory=arrays(brec["raw_path"])["trajectory"].reshape(51,base["N"],9)
            state=trajectory[step].copy();K=base["K"];N=base["N"]
            dest=C.OUT/"probes"/base["base_request_id"]/("step_"+str(step).zfill(2))
            dest.mkdir(parents=True,exist_ok=False)
            paired={};records={}
            for variant in ("m095","p105"):
                q=by[base["base_request_id"],variant];b=old().read_input(q)
                x=state.copy();x[:K]=b["C"].reshape(K,9)
                if not C.same(x[K:],state[K:]):raise ValueError("Probe free state changed")
                tx=torch.from_numpy(x[None]).to("cuda");tt=torch.tensor([step/50],dtype=torch.float32,device="cuda")
                known=torch.zeros((1,N),dtype=torch.bool,device="cuda");known[:,:K]=True
                valid=torch.ones_like(known);y=torch.tensor([N],dtype=torch.long,device="cuda")
                observed={}
                def observe(info):
                    if info["site"]=="embedding":
                        h=info["hidden"].detach()
                        if h.dtype!=torch.bfloat16:raise ValueError("Actual block input is not BF16")
                        bits=h.view(torch.uint16).cpu().numpy()[0]
                        observed["known_hidden_bf16_bits"]=bits[:3*K].copy()
                        observed["free_hidden_sha256"]=C.array_sha(bits[3*K:])
                        observed["face_time"]=info["face_time"].detach().cpu().numpy().copy()
                call=dest/(variant+"_call.json")
                C.write(call,dict(status="RESERVED",variant=variant,request_id=q["request_id"],step=step,
                    time_fp32=float(np.float32(step/50)),time_fp32_bits=int(np.float32(step/50).view(np.uint32)),
                    input_sha256=C.array_sha(x),free_state_sha256=C.array_sha(x[K:]),
                    C_sha256=q["C_sha256"],checkpoint_sha256=r["checkpoint"]["sha256"]))
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                with torch.no_grad(),native_execution(model,coordinates=(tx,tt)):
                    v=model(tx,tt,y,valid,known,observer=observe)
                torch.cuda.synchronize()
                if v.dtype!=torch.bfloat16 or not torch.isfinite(v).all():raise ValueError("Invalid probe velocity")
                va=v.detach().float().cpu().numpy()[0]
                artifact=dict(velocity=va,input_state=x,known_hidden_bf16_bits=observed["known_hidden_bf16_bits"],
                    free_state=state[K:],time=tt.cpu().numpy(),known_mask=known.cpu().numpy(),valid_mask=valid.cpu().numpy(),
                    N=y.cpu().numpy(),face_time=observed["face_time"])
                save(dest/(variant+".npz"),artifact)
                records[variant]=dict(status="RETURNED",artifact=str(dest/(variant+".npz")),
                    artifact_sha256=C.sha(dest/(variant+".npz")),velocity_sha256=C.array_sha(va),
                    free_hidden_sha256=observed["free_hidden_sha256"],actual_hidden_dtype="torch.bfloat16",
                    velocity_dtype="torch.bfloat16; lossless FP32 storage",seconds=time.perf_counter()-start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                C.write(dest/(variant+"_return.json"),records[variant]);paired[variant]=artifact;nprobe+=1
            if records["m095"]["free_hidden_sha256"]!=records["p105"]["free_hidden_sha256"]:
                raise ValueError("Same F has different pre-backbone free embedding")
            minus,plus=paired["m095"],paired["p105"]
            embedding_changed=bool(np.any(minus["known_hidden_bf16_bits"]!=plus["known_hidden_bf16_bits"]))
            receipt=dict(status="COMPLETE",parent_id=base["parent_id"],base_request_id=base["base_request_id"],
                base_condition_id=base["base_condition_id"],candidate=base["candidate"],K=K,N=N,step=step,
                time=float(np.float32(step/50)),base_raw_sha256=brec["raw_sha256"],base_raw_path=brec["raw_path"],
                free_state_sha256=C.array_sha(state[K:]),free_embedding_bitwise_equal=True,known_embedding_changed=embedding_changed,
                interpretability="DISTINGUISHABLE" if embedding_changed else "UNRESOLVED_INPUT_INDISTINGUISHABLE",
                forwards=2,records=records)
            C.write(dest/"receipt.json",receipt)
            C.write(C.OUT/"execution/probe_progress.json",dict(completed_pairs=nprobe//2,registered_pairs=64,seconds=time.perf_counter()-tick),replace=True)
        final_state_hash=loader.state_hash(model.state_dict())
        if final_state_hash!=identity["generator_state_sha256"]:raise ValueError("Frozen model state changed")
        C.write(C.OUT/"execution/generator_state_after.json",dict(status="PASS",generator_state_sha256=final_state_hash))
        snap("generator_finish")
        C.write(C.OUT/"execution/generator_complete.json",dict(status="COMPLETE",new_rollouts_this_invocation=nroll,
            new_sampling_F=50*nroll,new_probe_F=nprobe,seconds=time.perf_counter()-tick,
            training=0,backward=0,checkpoints_modified=False))
        return dict(status="COMPLETE",new_rollouts=nroll,new_probe_F=nprobe)
    except BaseException as exc:
        event("GENERATOR_STOPPED",error=repr(exc),traceback=traceback.format_exc(),returned_rollouts=nroll,returned_probe_F=nprobe)
        raise
    finally:
        del model;torch.cuda.empty_cache()

def postprocess():
    check_code();r=reg();records=collect_raw();C.package()
    from meshflow_control import denoise,geo_c,geo_io,source_winding
    from meshflow_control import denoise_model
    import torch
    torch.set_num_threads(2)
    model=None;rows=[];calls=0;tick=time.perf_counter()
    try:
        for rawrec in records:
            dest=C.OUT/"evaluation/post"/LABEL/rawrec["request_id"];rp=dest/"receipt.json"
            if rp.exists():
                record=C.read(rp)
                if record["raw_sha256"]!=rawrec["raw_sha256"] or record["source_sha256"]!=rawrec["condition_source_sha256"]:
                    raise ValueError("POST input identity drift")
                for p,h in record["files"].items():
                    if C.sha(p)!=h:raise ValueError("POST artifact changed")
                rows.append(record);continue
            raw=arrays(rawrec["raw_path"])["raw"].reshape(-1,3,3);K=rawrec["K"];src=source(rawrec,raw[:K])
            pp=dest/"proposal.npz";pr=dest/"proposal_receipt.json"
            if (dest/"geometry_started.json").exists():raise RuntimeError("Incomplete geometry requires recorded engineering recovery")
            if pr.exists():
                proposal=denoise.cached_proposal(pp,pr,raw,rawrec["raw_sha256"]);newcall=0
            elif dest.exists():raise RuntimeError("Unresolved D0 attempt; no silent retry")
            else:
                if model is None:
                    snap("D0_before_load",3000.)
                    binding=r["d0_checkpoint"]
                    model,mi=denoise_model.load_denoiser(C.resolve(binding["path"]),device="cuda",expected_sha256=binding["sha256"])
                    C.write(C.OUT/"execution/D0_identity.json",mi);snap("D0_resident")
                dest.mkdir(parents=True,exist_ok=False)
                C.write(dest/"D0_started.json",dict(status="RESERVED",raw_sha256=rawrec["raw_sha256"],checkpoint_sha256=r["d0_checkpoint"]["sha256"]))
                event("D0_FORWARD_RESERVED",request_id=rawrec["request_id"])
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                proposal=denoise_model.predict_proposal(model,raw)
                torch.cuda.synchronize();calls+=1;newcall=1
                save(pp,dict(official_unconstrained_proposal=proposal))
                C.write(pr,dict(status="PASS",raw_sha256=rawrec["raw_sha256"],checkpoint_sha256=r["d0_checkpoint"]["sha256"],
                    output_sha256=C.sha(pp),model_forwards=1,seconds=time.perf_counter()-start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                event("D0_FORWARD_RETURNED",request_id=rawrec["request_id"])
                if calls==1:snap("D0_first_actual_forward")
            denoise.validate_proposal(proposal,raw)
            C.write(dest/"geometry_started.json",dict(status="STARTED",new_D0_F=newcall))
            den=proposal.copy();den[:K]=raw[:K]
            save(dest/"pre_geo.npz",dict(output=den,C=raw[:K],known_faces=np.asarray(K)))
            geometry=geo_c.process(den,K);canonical=geo_io._validate_result(den,K,geometry)
            save(dest/"post_canonical.npz",canonical)
            result=source_winding.export_source_winding(canonical,src);post=result["oriented_arrays"]
            save(dest/"post.npz",post)
            geo_io.write_indexed_ply(dest/"post.ply",post["vertices"],post["faces"])
            vertices,faces=geo_io.read_indexed_ply(dest/"post.ply")
            if not C.same(vertices,post["vertices"]) or not C.same(faces,post["faces"]):raise ValueError("POST PLY readback differs")
            C.write(dest/"geometry.json",{k:v for k,v in geometry.items() if not isinstance(v,np.ndarray)})
            C.write(dest/"source_restoration.json",result["restoration_summary"])
            C.write(dest/"orientation.json",{k:v for k,v in result["orientation_result"].items() if not isinstance(v,np.ndarray)})
            rec={k:v for k,v in rawrec.items() if k not in ("files","contracts")}
            rec.update(status="COMPLETE",source_sha256=rawrec["condition_source_sha256"],proposal_path=str(pp),proposal_receipt=str(pr),
                post_path=str(dest/"post.npz"),post_sha256=C.sha(dest/"post.npz"),canonical_path=str(dest/"post_canonical.npz"),
                pre_geo_path=str(dest/"pre_geo.npz"),source_restoration=result["restoration_summary"],orientation=result["orientation_result"]["summary"],
                new_D0_F=newcall,GEO_calls=1,files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
            C.write(rp,rec);rows.append(rec)
            C.write(C.OUT/"execution/D0_progress.json",dict(completed=len(rows),registered=48,new_D0_F=calls,seconds=time.perf_counter()-tick),replace=True)
            print(json.dumps(dict(stage="POST",completed=len(rows),request_id=rawrec["request_id"])),flush=True)
        snap("D0_finish")
        if (C.OUT/"evaluation/post_index.json").exists():
            previous=C.read(C.OUT/"evaluation/post_index.json")
            if previous["rows"]!=rows or previous["status"]!="COMPLETE":raise ValueError("Completed POST index differs")
            return dict(status="COMPLETE",POST=len(rows),new_D0_F=0)
        result=dict(status="COMPLETE",rows=rows,new_D0_F_this_invocation=calls,seconds=time.perf_counter()-tick)
        C.write(C.OUT/"evaluation/post_index.json",result)
        C.write(C.OUT/"execution/D0_complete.json",{k:v for k,v in result.items() if k!="rows"})
        return dict(status="COMPLETE",POST=len(rows),new_D0_F=calls)
    except BaseException as exc:
        event("POST_STOPPED",error=repr(exc),traceback=traceback.format_exc(),completed=len(rows),D0_returned=calls)
        raise
    finally:
        if model is not None:del model;torch.cuda.empty_cache()
def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("freeze-code","sample-probe","postprocess","collect"),required=True)
    a=p.parse_args()
    if a.stage=="freeze-code":value=freeze_code()
    elif a.stage=="sample-probe":
        with gpu_owner():value=sample_and_probe()
    elif a.stage=="postprocess":
        with gpu_owner():value=postprocess()
    else:value=collect_raw()
    print(json.dumps(dict(stage=a.stage,result=value if not isinstance(value,list) else len(value))),flush=True)
if __name__=="__main__":main()
