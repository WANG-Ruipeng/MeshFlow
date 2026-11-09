
import argparse, json, os, sys, time
from pathlib import Path
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
parser=argparse.ArgumentParser()
parser.add_argument("--package-root",required=True)
parser.add_argument("--label",required=True)
args=parser.parse_args()
repo=Path("/mnt/e/MeshFlow")
sys.path.insert(0,str(repo))
sys.path.insert(0,args.package_root)
import numpy as np
import torch
from native_t1.geometry_checkpoint import load_geometry_checkpoint
from native_t1.model import native_execution
from native_t1.artifacts import array_hash, state_sha256
out=repo/"output/tgeo_publish_20260930_165000"
run=repo/"native_t1/runs/geometry_context_revised_20260930_153000"
model,audit=load_geometry_checkpoint(run/"T_geo/additional_step500_cumulative_step2000.pt",repo/"configs/snet/base-120m-ot-v-chair.yaml")
# A fixed interior sampling state from an existing output, no new sampling or OT.
with np.load(run/"raw/T_geo_P0_patch0_seed9701.npz") as value:
    x=torch.from_numpy(value["path"][25:26].copy()).cuda()
known=torch.zeros((1,112),device="cuda",dtype=torch.bool);known[:,:12]=True
valid=torch.ones_like(known)
t=torch.tensor([.5],device="cuda",dtype=torch.float32)
y=torch.tensor([112],device="cuda")
before=state_sha256(model)
with torch.no_grad(), native_execution(model,coordinates=x):
    velocity=model(x,t,y,valid,known).float()
torch.cuda.synchronize()
assert state_sha256(model)==before
np.save(out/(args.label+"_velocity.npy"),velocity.cpu().numpy())
record=dict(label=args.label,status="PASS",module_root=args.package_root,load=audit,
            velocity_sha256=array_hash(velocity),state_unchanged=True,
            model_forwards=1,model_backwards=0,optimizer_updates=0,complete_rollouts=0,OT_calls=0)
(out/(args.label+"_forward.json")).write_text(json.dumps(record,indent=2))
print(json.dumps(record))
