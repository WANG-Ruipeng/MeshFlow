"""Strict external START import and fully resumable candidate checkpoints."""
from . import runtime as _runtime
from pathlib import Path
import copy, hashlib, json, os, random
import numpy as np
import torch
from .config import MODEL_CONFIG, config_hash
from .artifacts import file_sha256
SCHEMA="native_mesh_direct_context_readout_checkpoint_v1"
START_FILE_SHA256="d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80"
START_STATE_SHA256="166c71fef888f26494fc4a0af1776b77a484efa72a6f3ce06552c604fbecc972"
MODES=("none","mixed","clean")

def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str((tuple(value.shape),value.dtype)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(kind=n[0],keys=n[1].tolist(),pos=n[2],has_gauss=n[3],cached=n[4]),
        torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
def restore_rng(value):
    random.setstate(value["python"])
    n=value["numpy"];np.random.set_state((n["kind"],np.asarray(n["keys"],dtype=np.uint32),n["pos"],n["has_gauss"],n["cached"]))
    torch.set_rng_state(value["torch_cpu"])
    if value["torch_cuda"]:
        if not torch.cuda.is_initialized(): raise RuntimeError("CUDA RNG restore requires owned GPU context")
        torch.cuda.set_rng_state_all(value["torch_cuda"])
def _make(mode,trainable):
    from .models.native import build_model
    if mode not in MODES: raise ValueError("Unsupported readout mode")
    return build_model(copy.deepcopy(MODEL_CONFIG),readout_mode=mode,readout_seed=20261007,trainable=trainable).eval()
def _identity(payload):
    if payload.get("schema")!="chair_hybrid_coupling_checkpoint_v1" or payload.get("update_in_progress") is not False:
        raise ValueError("Not a completed original START")
    meta=payload["metadata"]
    for key,value in dict(arm="OT_HYBRID",pilot_step=1000,conditional_total_step=5000,model_state_sha256=START_STATE_SHA256).items():
        if meta.get(key)!=value: raise ValueError("Wrong START identity: "+key)
    old=payload["model"]
    if len(old)!=194 or sum(x.numel() for x in old.values())!=130401155 or state_hash(old)!=START_STATE_SHA256:
        raise ValueError("START model hash/count mismatch")
    if any(x.dtype!=torch.float32 or not torch.isfinite(x).all() for x in old.values()):
        raise ValueError("START model must be finite FP32")
    groups=payload["optimizer"]["param_groups"]
    if len(groups)!=1 or len(groups[0]["params"])!=194 or len(set(groups[0]["params"]))!=194:
        raise ValueError("Unexpected original optimizer ownership")
    g=groups[0]
    if (g["lr"],tuple(g["betas"]),g["weight_decay"])!=(1e-5,(.9,.95),0.): raise ValueError("Wrong START AdamW")
    return old,g
def load_start(path,mode="none",device="cpu",trainable=True):
    path=Path(path)
    if file_sha256(path)!=START_FILE_SHA256: raise ValueError("External START file SHA256 differs")
    payload=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    old,group=_identity(payload);model=_make(mode,trainable);expected=model.state_dict()
    original_names=[n for n in expected if not n.startswith("readouts.")]
    if original_names!=list(old): raise ValueError("Original parameter registration identity/order differs")
    if set(expected)-set(old)!={n for n in expected if n.startswith("readouts.")}: raise ValueError("Unregistered parameters")
    model.load_state_dict({**expected,**old},strict=True)
    if any(not torch.equal(model.state_dict()[n],v) for n,v in old.items()): raise ValueError("Tensor migration not exact")
    new_names=[n for n in expected if n.startswith("readouts.")]
    extra=0 if mode=="none" else 2359296
    if sum(expected[n].numel() for n in new_names)!=extra or len(new_names)!=(0 if mode=="none" else 16):
        raise ValueError("Readout capacity mismatch")
    model.to(device).set_trainable(trainable).eval();named=dict(model.named_parameters())
    mapping=[dict(old_name=n,new_name=n,saved_optimizer_id=s) for n,s in zip(original_names,group["params"])]
    optimizer=None
    if trainable:
        groups=[dict(params=[named[n] for n in original_names])]
        if new_names: groups.append(dict(params=[named[n] for n in new_names]))
        optimizer=torch.optim.AdamW(groups,lr=1e-5,betas=(.9,.95),weight_decay=0.)
        for k,v in group.items():
            if k!="params": optimizer.param_groups[0][k]=copy.deepcopy(v)
        for row in mapping:
            n,sid=row["new_name"],row["saved_optimizer_id"];src=payload["optimizer"]["state"][sid];p=named[n]
            if set(src)!={"step","exp_avg","exp_avg_sq"} or float(src["step"])!=5000: raise ValueError("Original AdamW step")
            if any(src[k].shape!=p.shape or src[k].dtype!=torch.float32 for k in ("exp_avg","exp_avg_sq")):
                raise ValueError("Original AdamW moment identity")
            optimizer.state[p]={k:(v.detach().clone() if k=="step" else v.detach().to(device).clone()) for k,v in src.items()}
            if any(not torch.equal(optimizer.state[p][k].cpu(),src[k]) for k in src): raise ValueError("AdamW migration not equal")
        for n in new_names:
            p=named[n];optimizer.state[p]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
    audit=dict(schema_version=SCHEMA,source_start_sha256=START_FILE_SHA256,source_model_state_sha256=START_STATE_SHA256,
        readout_mode=mode,old_tensor_count=194,old_parameter_count=130401155,new_parameter_count=extra,
        parameter_name_mapping=mapping,old_tensors_equal=True,optimizer_restored=trainable,
        original_optimizer_step=5000,new_optimizer_step=0,new_optimizer_group=bool(new_names and trainable),
        new_readout_state_sha256=state_hash({n:model.state_dict()[n] for n in new_names}),
        model_config=copy.deepcopy(MODEL_CONFIG),inference_targets=False)
    model.import_audit=audit
    return model,optimizer,audit

def save_checkpoint(path,model,optimizer,*,step,stream_state,config,data_manifest_hash,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not 0<=int(step)<=1000: raise ValueError("Pilot step exceeds protocol")
    names={id(v):k for k,v in model.named_parameters()}
    groups=[[names[id(p)] for p in g["params"]] for g in optimizer.param_groups]
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    payload=dict(schema_version=SCHEMA,model_config=copy.deepcopy(MODEL_CONFIG),
        readout=dict(mode=model.readout_mode,layers=[3,6,9,12],dimension=192,heads=4,seed=20261007),
        model=state,model_state_sha256=state_hash(state),optimizer=optimizer.state_dict(),optimizer_parameter_names=groups,
        rng=rng_state(),stream_state=stream_state,pilot_step=int(step),conditional_total_step=5000+int(step),
        source_start_sha256=START_FILE_SHA256,source_model_state_sha256=START_STATE_SHA256,
        training_config=config,training_config_hash=config_hash(config),data_manifest_hash=data_manifest_hash,
        update_in_progress=False,extra=extra or {})
    temp=path.with_suffix(path.suffix+".partial");torch.save(payload,temp);os.replace(temp,path)
    return dict(path=str(path),pilot_step=int(step),sha256=file_sha256(path),model_state_sha256=payload["model_state_sha256"])

def load_checkpoint(path,device="cpu",trainable=False):
    payload=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    required={"schema_version","model_config","readout","model","model_state_sha256","optimizer","optimizer_parameter_names",
        "rng","stream_state","pilot_step","conditional_total_step","source_start_sha256","source_model_state_sha256",
        "training_config","training_config_hash","data_manifest_hash","update_in_progress","extra"}
    if set(payload)!=required or payload["schema_version"]!=SCHEMA or payload["update_in_progress"] is not False:
        raise ValueError("Candidate checkpoint schema/key mismatch")
    r=payload["readout"]
    if (payload["model_config"]!=MODEL_CONFIG or r!={"mode":r.get("mode"),"layers":[3,6,9,12],"dimension":192,"heads":4,"seed":20261007}
        or r["mode"] not in MODES or payload["source_start_sha256"]!=START_FILE_SHA256
        or payload["source_model_state_sha256"]!=START_STATE_SHA256 or not 0<=payload["pilot_step"]<=1000
        or payload["conditional_total_step"]!=5000+payload["pilot_step"]
        or config_hash(payload["training_config"])!=payload["training_config_hash"]): raise ValueError("Candidate identity/config mismatch")
    if state_hash(payload["model"])!=payload["model_state_sha256"]: raise ValueError("Candidate tensor hash mismatch")
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in payload["model"].values()):
        raise ValueError("Candidate state not finite FP32")
    model=_make(r["mode"],trainable);model.load_state_dict(payload["model"],strict=True)
    model.to(device).set_trainable(trainable).eval();optimizer=None
    if trainable:
        named=dict(model.named_parameters());groups=payload["optimizer_parameter_names"];flat=[n for g in groups for n in g]
        if len(flat)!=len(set(flat)) or set(flat)!=set(named): raise ValueError("Optimizer mapping mismatch")
        optimizer=torch.optim.AdamW([{"params":[named[n] for n in g]} for g in groups],lr=1e-5,betas=(.9,.95),weight_decay=0.)
        optimizer.load_state_dict(payload["optimizer"])
        for g,group_names in zip(optimizer.param_groups,groups):
            if (g["lr"],tuple(g["betas"]),g["weight_decay"])!=(1e-5,(.9,.95),0.): raise ValueError("Optimizer settings mismatch")
            for p,n in zip(g["params"],group_names):
                entry=optimizer.state[p];expected=payload["pilot_step"]+(0 if n.startswith("readouts.") else 5000)
                if float(entry["step"])!=expected: raise ValueError("Optimizer step mismatch")
                for k in ("exp_avg","exp_avg_sq"):
                    if entry[k].shape!=p.shape or entry[k].dtype!=torch.float32: raise ValueError("Optimizer moment mismatch")
    return model,optimizer,payload
