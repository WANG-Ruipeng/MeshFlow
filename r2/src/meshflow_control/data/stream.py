"""A shared, deterministic 1000-update HYBRID stream with 8000 bounded OT maps."""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
from .dataset import TrainingDataset, gaussian, training_time, permutations
from .io import digest, array_hash, json_hash, read_json, freeze_json, save_npz, append_event, atomic_json, bits
from ..training.coupling import CONTRACT, nested_ot_map, validate_map, to_noise_slots, to_target_slots, coupling_path

NAMESPACE = "NATIVE_MESH_DIRECT_CONTEXT_READOUT_V1/train/v1"
SEED = int.from_bytes(hashlib.sha256(NAMESPACE.encode()).digest()[:4], "little")
ARMS = ("A", "B", "C")

def domain_seed(domain, *identity):
    data = json.dumps([NAMESPACE, SEED, domain, *identity], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "little")

def make_plan(manifest):
    parents = sorted(manifest["splits"]["train"])
    if len(parents) != 32:
        raise ValueError("Original 32-parent training split required")
    tasks = defaultdict(lambda:defaultdict(list))
    for r in manifest["tasks"]:
        if r["role"]=="train" and r["status"]=="READY":
            tasks[r["uid"]][r["ratio"]].append(r)
    if set(tasks)!=set(parents) or sum(len(v) for d in tasks.values() for v in d.values())!=453:
        raise ValueError("Original 453 READY tasks required")
    domains=("parent_schedule","task_schedule","alpha","alpha_slot_shuffle")
    rng={d:np.random.RandomState(domain_seed(d)) for d in domains}
    schedule=[parents[int(i)] for _ in range(250) for i in rng["parent_schedule"].permutation(32)]
    records=[]
    for batch in range(1000):
        alpha=np.r_[np.ones(4,np.float32),rng["alpha"].uniform(.9,1.1,4).astype(np.float32)]
        alpha=alpha[rng["alpha_slot_shuffle"].permutation(8)]
        for slot in range(8):
            i=batch*8+slot; uid=schedule[i]
            ratios=sorted(tasks[uid]); ratio=ratios[int(rng["task_schedule"].randint(len(ratios)))]
            options=tasks[uid][ratio]; task=options[int(rng["task_schedule"].randint(len(options)))]
            records.append(dict(sample_index=i,effective_batch_index=batch,slot_index=slot,
                step=batch+1,conditional_total_step=5001+batch,uid=uid,task_id=task["task_id"],
                N=task["N"],K=task["K"],ratio=ratio,alpha=float(alpha[slot]),
                gaussian_seed=domain_seed("gaussian",i),epsilon_seed=domain_seed("epsilon",i),
                permutation_seed=domain_seed("permutation",i),t_seed=domain_seed("time",i)))
        if len({r["uid"] for r in records[-8:]})!=8 or sum(r["alpha"]==1. for r in records[-8:])!=4:
            raise ValueError("Eight distinct parents and four exact alpha=1 required")
    return dict(schema="meshflow_control_train_plan_v1",namespace=NAMESPACE,seed=SEED,
        effective_batch_count=1000,batch_size=8,microbatch_size=1,sample_count=8000,
        RNG_mode="stateless_per_sample_frozen_seeds",coupling_lambda=.25,
        parent_distribution="250 shuffled full 32-parent cycles; each 8-item batch has distinct parents",
        context_distribution="uniform available original ratio; uniform original C within ratio",
        object_counts=dict(Counter(r["uid"] for r in records)),records=records)

def prepare_training_plan(data_root, stream_root):
    data=TrainingDataset(data_root); out=Path(stream_root).resolve()
    plan=dict(make_plan(data.manifest),data_manifest_sha256=data.manifest_sha256)
    sha=freeze_json(out/"train_plan.json",plan)
    return {"path":str(out/"train_plan.json"),"sha256":sha,"seed":SEED,"namespace":NAMESPACE,
            "samples":8000,"updates_per_arm":1000,"data_manifest_sha256":data.manifest_sha256}

class HybridStream:
    def __init__(self,data_root,stream_root,arm="A"):
        if arm not in ARMS: raise ValueError("Arm must be A, B or C")
        self.data=TrainingDataset(data_root); self.out=Path(stream_root).resolve(); self.arm=arm
        self.plan_path=self.out/"train_plan.json"
        self.plan=read_json(self.plan_path); self.plan_sha256=digest(self.plan_path)
        if self.plan!=dict(make_plan(self.data.manifest),data_manifest_sha256=self.data.manifest_sha256):
            raise ValueError("Frozen stream plan does not reproduce")
        self.records=self.plan["records"]; self.batch_index=0
        self.cache_dir=self.out/"ot_cache"
        self.counts={"actual_OT_attempts":0,"actual_OT_returns":0,"OT_cache_hits":0}
        self.implementation_sha256=json_hash({
            p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name("dataset.py"),
              Path(__file__).parents[1]/"training/coupling.py")})

    def state_dict(self):
        return dict(schema="meshflow_control_stream_state_v1",arm=self.arm,batch_index=self.batch_index,
            sample_index=8*self.batch_index,namespace=NAMESPACE,seed=SEED,
            plan_sha256=self.plan_sha256,data_manifest_sha256=self.data.manifest_sha256,
            coupling_contract=CONTRACT,lambda_value=.25,implementation_sha256=self.implementation_sha256)

    def load_state_dict(self,state):
        expected=self.state_dict()
        for key in expected:
            if key not in ("batch_index","sample_index") and state.get(key)!=expected[key]:
                raise ValueError("Stream restore identity differs: "+key)
        b=state["batch_index"]
        if type(b)!=int or not 0<=b<=1000 or state["sample_index"]!=8*b:
            raise ValueError("Invalid stream cursor")
        self.batch_index=b

    def _mapping(self,pre,noise,source_ids,corners,known_ids,record):
        identity=dict(contract=CONTRACT,implementation_sha256=self.implementation_sha256,
            pre_target=array_hash(pre),raw_noise=array_hash(noise),source_ids=array_hash(source_ids),
            corners=array_hash(corners),known_ids=array_hash(known_ids),alpha=record["alpha"],
            task_id=record["task_id"],sample_index=record["sample_index"],
            plan_sha256=self.plan_sha256,data_manifest_sha256=self.data.manifest_sha256)
        key=json_hash(identity)
        path=self.cache_dir/(str(record["sample_index"]).zfill(4)+"_"+key+".npz")
        attempt=path.with_suffix(".attempt.json")
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                face,corner=validate_map(z["face"],z["corner"])
                if str(z["key"].item())!=key or json.loads(str(z["identity"].item()))!=identity:
                    raise ValueError("OT cache identity differs")
                if str(z["map_sha256"].item())!=json_hash([array_hash(face),array_hash(corner)]):
                    raise ValueError("OT cache map contents differ")
                face,corner=face.copy(),corner.copy()
            self.counts["OT_cache_hits"]+=1
            return face,corner,key,"HIT"
        if attempt.exists():
            raise RuntimeError("Incomplete registered OT attempt; refusing silent re-OT: "+str(attempt))
        # Exclusive reservation makes concurrency safe across distinct sample IDs.
        with attempt.open("x",encoding="utf-8") as f:
            json.dump({"identity":identity,"key":key,"time_unix":time.time(),"pid":os.getpid()},f,sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        self.counts["actual_OT_attempts"]+=1
        begin=time.perf_counter()
        try:
            face,corner=nested_ot_map(pre,noise)
            self.counts["actual_OT_returns"]+=1
            sha=save_npz(path,face=face,corner=corner,key=np.asarray(key),
                identity=np.asarray(json.dumps(identity,sort_keys=True)),
                map_sha256=np.asarray(json_hash([array_hash(face),array_hash(corner)])))
            freeze_json(path.with_suffix(".return.json"),{"key":key,"npz_sha256":sha,
                "seconds":time.perf_counter()-begin,"status":"RETURN"})
        except BaseException as exc:
            atomic_json(path.with_suffix(".failed.json"),{"key":key,"status":"FAILED","error":repr(exc)})
            raise
        return face,corner,key,"MISS"

    def sample(self,record):
        case=self.data.task(record["task_id"],record["alpha"])
        n,k=case["N"],case["K"]; known_ids,free_ids=case["source_face_ids"],case["free_source_ids"]
        free_order,corners,order=permutations(record["permutation_seed"],n,k)
        ordered_ids=free_ids[free_order]
        pre=case["pre_ot_full"][ordered_ids][np.arange(n-k)[:,None],corners[k:]]
        noise=gaussian(record["gaussian_seed"],n-k)
        face,corner,key,status=self._mapping(pre,noise,ordered_ids,corners[k:],known_ids,record)
        target=to_noise_slots((pre*np.float32(2)).astype(np.float32),face,corner)
        epsilon=gaussian(record["epsilon_seed"],n-k)
        t=np.asarray(training_time(record["t_seed"]),np.float32)
        mixed,_,_=coupling_path(noise,target,epsilon,t)
        x0=np.zeros((n,3,3),np.float32); x1=np.zeros_like(x0); context=np.zeros_like(x0)
        x0[k:]=mixed; x1[k:]=target
        context[:k]=case["full_target"].reshape(n,3,3)[known_ids][np.arange(k)[:,None],corners[:k]]
        ids=np.r_[known_ids,ordered_ids[np.argsort(face)]]
        mapped_corners=np.concatenate((corners[:k],to_noise_slots(corners[k:],face,corner)))
        known=(np.arange(n)<k)[order]
        x0,x1,context=[a[order].reshape(n,9) for a in (x0,x1,context)]
        ids,mapped_corners=ids[order],mapped_corners[order]
        u=x1-x0; xt=(np.float32(1)-t)*x0+t*x1; xt[known]=context[known]
        expected=case["full_target"].reshape(n,3,3)[ids][np.arange(n)[:,None],mapped_corners].reshape(n,9)
        if not bits(expected[~known],x1[~known]) or not bits(expected[known],context[known]):
            raise ValueError("OT face/corner labels do not reconstruct original target")
        values=dict(x0=x0,x1=x1,u=u,xt=xt,context=context,t=t,y=np.asarray(n,np.int64),
            known_mask=known,valid_mask=np.ones(n,bool),source_face_ids=ids,
            corner_permutations=mapped_corners,face_permutation=order,
            object_id=record["uid"],task_id=record["task_id"],K=k,alpha=record["alpha"],ratio=record["ratio"],
            free_gaussian_before_OT=noise,free_epsilon=epsilon,free_target_before_OT=pre,
            ot_face_map=face,ot_corner_map=corner,ot_cache_key=key,ot_cache_status=status,
            sample_index=record["sample_index"],step=record["step"],free_coordinate_count=9*(n-k))
        shared={q:array_hash(values[q]) for q in ("x0","x1","u","xt","context","t","y","known_mask","valid_mask",
            "source_face_ids","corner_permutations","face_permutation","free_gaussian_before_OT","free_epsilon",
            "free_target_before_OT","ot_face_map","ot_corner_map")}
        values["shared_input_sha256"]=json_hash(dict(shared,task_id=record["task_id"],alpha=record["alpha"]))
        values["coupled_input_sha256"]=values["shared_input_sha256"]
        values["input_bytehash"]=json_hash({q:shared[q] for q in ("xt","t","y","known_mask","valid_mask")})
        values["label_bytehash"]=array_hash(u)
        return values

    _sample=sample

    def next_effective_batch(self):
        if not 0<=self.batch_index<1000: raise RuntimeError("Frozen stream exhausted")
        start=8*self.batch_index
        samples=[self.sample(r) for r in self.records[start:start+8]]
        self.batch_index+=1
        self.last_audit={"batch_index":self.batch_index,
            "shared_batch_sha256":json_hash([s["shared_input_sha256"] for s in samples]),
            "input_batch_sha256":json_hash([s["input_bytehash"] for s in samples]),
            "label_batch_sha256":json_hash([s["label_bytehash"] for s in samples]),
            "effective_free_coordinates":sum(s["free_coordinate_count"] for s in samples)}
        return samples

def precompute(data_root,stream_root,workers=4):
    """Compute exactly the registered 8000 maps, with no model or sample generation."""
    if not 1<=workers<=8: raise ValueError("CPU worker budget is 1..8")
    out=Path(stream_root).resolve()
    stream=HybridStream(data_root,stream_root,"A")
    # Check all incomplete attempts before dispatching any additional map.
    for p in stream.cache_dir.glob("*.attempt.json"):
        if not p.with_suffix("").with_suffix(".npz").exists():
            raise RuntimeError("Prior incomplete OT attempt: "+str(p))
    tick=time.perf_counter(); initial=dict(stream.counts)
    def compute(record):
        s=stream.sample(record)
        return {key:s[key] for key in ("sample_index","task_id","shared_input_sha256","input_bytehash",
            "label_bytehash","ot_cache_key","ot_cache_status","free_coordinate_count")}
    # Each frozen record is dispatched exactly once; no worker-derived seed.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        items=list(pool.map(compute,stream.records))
    if len(items)!=8000 or [s["sample_index"] for s in items]!=list(range(8000)):
        raise ValueError("Registered OT map count/order differs")
    # Count actual outcome from immutable per-map records, not unsynchronized counters.
    actual_new=sum(s["ot_cache_status"]=="MISS" for s in items)
    receipt={"status":"PASS","namespace":NAMESPACE,"seed":SEED,"workers":workers,
        "samples":8000,"actual_OT_attempts_this_run":actual_new,"actual_OT_returns_this_run":actual_new,
        "cache_hits_this_run":8000-actual_new,"wall_seconds":time.perf_counter()-tick,
        "model_forwards":0,"model_backwards":0,"optimizer_updates":0,"generation_requests":0,
        "plan_sha256":stream.plan_sha256,"data_manifest_sha256":stream.data.manifest_sha256,
        "implementation_sha256":stream.implementation_sha256}
    freeze_json(out/"paired_inputs.json",{"schema":"meshflow_control_paired_inputs_v1",
        "plan_sha256":stream.plan_sha256,"data_manifest_sha256":stream.data.manifest_sha256,
        "records":[{k:v for k,v in s.items() if k!="ot_cache_status"} for s in items]})
    atomic_json(out/"precompute_receipt.json",receipt)
    return receipt

def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True);p.add_argument("--out",required=True)
    p.add_argument("--precompute",action="store_true");p.add_argument("--workers",type=int,default=4)
    a=p.parse_args(argv)
    print(json.dumps(prepare_training_plan(a.data,a.out),indent=2),flush=True)
    if a.precompute: print(json.dumps(precompute(a.data,a.out,a.workers),indent=2),flush=True)

if __name__=="__main__":main()
