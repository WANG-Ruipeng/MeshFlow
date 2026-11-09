
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import json,sys
from pathlib import Path
repo=Path("/mnt/e/MeshFlow")
sys.path.insert(0,str(repo))
import numpy as np
import torch
from native_t1.portable_checkpoint import load_trained_checkpoint
from native_t1.context_geometry import attach_context_encoder
from native_t1.model import native_execution
from native_t1.artifacts import array_hash,state_sha256
out=repo/"output/tgeo_publish_20260930_165000"
run=repo/"native_t1/runs/geometry_context_revised_20260930_153000"
model,audit=load_trained_checkpoint(run/"T_base/additional_step500_cumulative_step2000.pt")
with np.load(run/"raw/T_geo_P0_patch0_seed9701.npz") as value:
    x=torch.from_numpy(value["path"][25:26].copy()).cuda()
known=torch.zeros((1,112),device="cuda",dtype=torch.bool);known[:,:12]=True
valid=torch.ones_like(known)
t=torch.tensor([.5],device="cuda",dtype=torch.float32)
y=torch.tensor([112],device="cuda")
backbone_before=state_sha256(model.backbone)
role_before=model.role_embedding.detach().clone()
with torch.no_grad(),native_execution(model,coordinates=x):
    pure=model(x,t,y,valid,known).float()
rng=torch.get_rng_state().clone()
cuda_rng=torch.cuda.get_rng_state().clone()
metadata=attach_context_encoder(model,"geo",seed=1010)
assert torch.equal(rng,torch.get_rng_state())
assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
with torch.no_grad(),native_execution(model,coordinates=x):
    geo=model(x,t,y,valid,known).float()
assert torch.equal(pure,geo)
assert state_sha256(model.backbone)==backbone_before
assert torch.equal(role_before,model.role_embedding)
record=dict(status="PASS",model_forwards=2,model_backwards=0,optimizer_updates=0,OT_calls=0,complete_rollouts=0,
            pure_and_zero_geo_bitwise_equal=True,base_parameters_unchanged=True,
            caller_CPU_and_CUDA_RNG_unchanged=True,velocity_sha256=array_hash(pure),metadata=metadata,load=audit)
(out/"zero_branch.json").write_text(json.dumps(record,indent=2))
print(json.dumps(record))
