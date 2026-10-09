"""Independent preregistered stream; inherits unchanged HYBRID coupling mathematics."""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib, json, time
import numpy as np
from . import stream as base
from .dataset import TrainingDataset
from .io import digest, json_hash, read_json, freeze_json, atomic_json

NAMESPACE="CHAIR_CONTEXTUAL_ALIGNMENT_SIE_PILOT_V1/train/v1"
SEED=int.from_bytes(hashlib.sha256(NAMESPACE.encode()).digest()[:4],"little")
def domain_seed(domain,*identity):
    return int.from_bytes(hashlib.sha256(json.dumps([NAMESPACE,SEED,domain,*identity],separators=(",",":")).encode()).digest()[:4],"little")

def make_plan(manifest):
    parents=sorted(manifest["splits"]["train"])
    if len(parents)!=32: raise ValueError("Expected unchanged 32 training parents")
    tasks=defaultdict(lambda:defaultdict(list))
    for r in manifest["tasks"]:
        if r["role"]=="train" and r["status"]=="READY": tasks[r["uid"]][r["ratio"]].append(r)
    if set(tasks)!=set(parents) or sum(len(v) for d in tasks.values() for v in d.values())!=453:
        raise ValueError("Expected unchanged 453 READY conditions")
    domains=("parent_schedule","task_schedule","alpha","alpha_slot_shuffle")
    rng={d:np.random.RandomState(domain_seed(d)) for d in domains}
    schedule=[parents[int(i)] for _ in range(250) for i in rng["parent_schedule"].permutation(32)]
    records=[]
    for batch in range(1000):
        alpha=np.r_[np.ones(4,np.float32),rng["alpha"].uniform(.9,1.1,4).astype(np.float32)]
        alpha=alpha[rng["alpha_slot_shuffle"].permutation(8)]
        for slot in range(8):
            i=batch*8+slot;uid=schedule[i]
            ratios=sorted(tasks[uid]);ratio=ratios[int(rng["task_schedule"].randint(len(ratios)))]
            options=tasks[uid][ratio];task=options[int(rng["task_schedule"].randint(len(options)))]
            records.append(dict(sample_index=i,effective_batch_index=batch,slot_index=slot,
                step=batch+1,conditional_total_step=5001+batch,uid=uid,task_id=task["task_id"],
                N=task["N"],K=task["K"],ratio=ratio,alpha=float(alpha[slot]),
                gaussian_seed=domain_seed("gaussian",i),epsilon_seed=domain_seed("epsilon",i),
                permutation_seed=domain_seed("permutation",i),t_seed=domain_seed("time",i)))
        if len({r["uid"] for r in records[-8:]})!=8 or sum(r["alpha"]==1 for r in records[-8:])!=4:
            raise ValueError("Batch contract differs")
    return dict(schema="meshflow_contextual_alignment_train_plan_v1",namespace=NAMESPACE,seed=SEED,
        effective_batch_count=1000,batch_size=8,microbatch_size=1,sample_count=8000,
        RNG_mode="stateless_per_sample_frozen_seeds",coupling_lambda=.25,
        parent_distribution="250 shuffled full 32-parent cycles; each 8-item batch has distinct parents",
        context_distribution="uniform available original ratio; uniform original C within ratio",
        object_counts=dict(Counter(r["uid"] for r in records)),records=records)

def prepare_training_plan(data_root,stream_root):
    data=TrainingDataset(data_root)
    plan=dict(make_plan(data.manifest),data_manifest_sha256=data.manifest_sha256)
    sha=freeze_json(Path(stream_root)/"train_plan.json",plan)
    return dict(path=str(Path(stream_root)/"train_plan.json"),sha256=sha,seed=SEED,namespace=NAMESPACE,
        samples=8000,updates_per_arm=1000,data_manifest_sha256=data.manifest_sha256)

class HybridStream(base.HybridStream):
    def __init__(self,data_root,stream_root,arm="A"):
        if arm not in ("A","B","C"):raise ValueError("Unknown arm")
        self.data=TrainingDataset(data_root);self.out=Path(stream_root).resolve();self.arm=arm
        self.plan_path=self.out/"train_plan.json";self.plan=read_json(self.plan_path);self.plan_sha256=digest(self.plan_path)
        if self.plan!=dict(make_plan(self.data.manifest),data_manifest_sha256=self.data.manifest_sha256):
            raise ValueError("Frozen independent plan does not reproduce")
        self.records=self.plan["records"];self.batch_index=0;self.cache_dir=self.out/"ot_cache"
        self.counts={"actual_OT_attempts":0,"actual_OT_returns":0,"OT_cache_hits":0}
        self.implementation_sha256=json_hash({str(p.name):digest(p) for p in
            (Path(__file__),Path(base.__file__),Path(__file__).with_name("dataset.py"),Path(__file__).parents[1]/"training/coupling.py")})
    def state_dict(self):
        s=super().state_dict();s.update(schema="meshflow_contextual_alignment_stream_state_v1",namespace=NAMESPACE,seed=SEED);return s

def precompute(data_root,stream_root,workers=4):
    if not 1<=workers<=8:raise ValueError("CPU workers outside budget")
    out=Path(stream_root);stream=HybridStream(data_root,stream_root,"A")
    for p in stream.cache_dir.glob("*.attempt.json"):
        if not p.with_suffix("").with_suffix(".npz").exists():raise RuntimeError("Unresolved prior OT attempt: "+str(p))
    tick=time.perf_counter()
    def compute(r):
        s=stream.sample(r)
        return {k:s[k] for k in ("sample_index","task_id","shared_input_sha256","input_bytehash","label_bytehash","ot_cache_key","ot_cache_status","free_coordinate_count")}
    with ThreadPoolExecutor(max_workers=workers) as pool:items=list(pool.map(compute,stream.records))
    if [r["sample_index"] for r in items]!=list(range(8000)):raise ValueError("Incomplete OT preparation")
    new=sum(r["ot_cache_status"]=="MISS" for r in items)
    freeze_json(out/"paired_inputs.json",dict(schema="meshflow_contextual_alignment_paired_inputs_v1",
        plan_sha256=stream.plan_sha256,data_manifest_sha256=stream.data.manifest_sha256,
        records=[{k:v for k,v in r.items() if k!="ot_cache_status"} for r in items]))
    receipt=dict(status="PASS",namespace=NAMESPACE,seed=SEED,workers=workers,samples=8000,
        actual_OT_attempts_this_run=new,actual_OT_returns_this_run=new,cache_hits_this_run=8000-new,
        wall_seconds=time.perf_counter()-tick,model_forwards=0,model_backwards=0,optimizer_updates=0,
        plan_sha256=stream.plan_sha256,data_manifest_sha256=stream.data.manifest_sha256,
        implementation_sha256=stream.implementation_sha256)
    atomic_json(out/"precompute_receipt.json",receipt);return receipt

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--out",required=True)
    p.add_argument("--precompute",action="store_true");p.add_argument("--workers",type=int,default=4);a=p.parse_args()
    print(json.dumps(prepare_training_plan(a.data,a.out)),flush=True)
    if a.precompute:print(json.dumps(precompute(a.data,a.out,a.workers)),flush=True)
