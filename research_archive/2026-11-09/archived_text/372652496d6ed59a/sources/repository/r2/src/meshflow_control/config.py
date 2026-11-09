"""Fixed scientific configuration; paths arrive only through CLI/config."""
from pathlib import Path
import copy
import hashlib
import json
import yaml

MODEL_CONFIG = dict(model_type="equidit", use_qknorm=True, use_rmsnorm=True,
    hidden_dim=768, num_heads=12, max_length=800, num_layers=12,
    gradient_checkpointing=False, use_coord_encoding=True, version=3, pe_freq=20,
    mixed_precision="bf16", use_dit_like_pe=False, face_cond=True, face_bin=20)
ARM_MODES = {"A":"none", "B":"mixed", "C":"clean"}
def config_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def load_config(path):
    path=Path(path); value=yaml.safe_load(path.read_text(encoding="utf-8"))
    if "extends" in value:
        base=load_config(path.parent/value.pop("extends"))
        for key,item in value.items():
            if isinstance(item,dict) and isinstance(base.get(key),dict): base[key].update(item)
            else: base[key]=item
        value=base
    return validate_config(value)
def validate_config(value):
    v=copy.deepcopy(value)
    if v.get("model",MODEL_CONFIG)!=MODEL_CONFIG: raise ValueError("Frozen START architecture differs")
    v.setdefault("model",copy.deepcopy(MODEL_CONFIG))
    train=v.setdefault("training",{})
    expected=dict(updates=1000,effective_batch=8,microbatch=1,lr=1e-5,betas=[.9,.95],weight_decay=0.0,clip=1.0,save_steps=[500,1000])
    for key,item in expected.items():
        train.setdefault(key,item)
        if train[key]!=item: raise ValueError("Frozen training setting differs: "+key)
    read=v.setdefault("readout",{})
    for key,item in dict(dimension=192,heads=4,layers=[3,6,9,12],seed=20261007).items():
        read.setdefault(key,item)
        if read[key]!=item: raise ValueError("Frozen readout setting differs: "+key)
    mode=v.setdefault("readout_mode","none")
    if mode not in ARM_MODES.values(): raise ValueError("Unknown readout mode")
    if v.get("arm") and ARM_MODES.get(v["arm"])!=mode: raise ValueError("Arm/mode mismatch")
    return v
