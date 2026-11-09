from pathlib import Path
import json,torch
from meshflow_control.checkpoints import load_checkpoint
from meshflow_control.artifacts import atomic_json,file_sha256
from meshflow_control.data.stream import HybridStream
root=Path("/mnt/e/MeshFlow-Control-artifacts");p=root/"checkpoints/B_step0500.pt"
model,optimizer,payload=load_checkpoint(p,device="cpu",trainable=True)
stream=HybridStream(root/"data",root/"training_stream","B");stream.load_state_dict(payload["stream_state"])
steps={group:sorted({float(optimizer.state[v]["step"]) for n,v in model.named_parameters() if n.startswith("readouts.")==(group=="new")}) for group in ("old","new")}
assert payload["pilot_step"]==500 and payload["conditional_total_step"]==5500
assert steps=={"old":[5500.],"new":[500.]} and stream.batch_index==500 and model.readout_mode=="mixed"
assert not torch.cuda.is_initialized()
result=dict(status="PASS",checkpoint_sha256=file_sha256(p),mode=model.readout_mode,pilot_step=500,
    AdamW_steps=steps,optimizer_groups=len(optimizer.param_groups),data_cursor=stream.batch_index,
    parameter_tensors=len(list(model.parameters())),CUDA_initialized=False,model_forwards=0,optimizer_updates=0)
atomic_json(root/"audit/B500_fresh_cpu_load.json",result);print(json.dumps(result))
