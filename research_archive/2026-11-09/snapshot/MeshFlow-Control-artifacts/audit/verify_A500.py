from pathlib import Path
import json,torch
from meshflow_control.checkpoints import load_checkpoint
from meshflow_control.artifacts import atomic_json,file_sha256
from meshflow_control.data.stream import HybridStream
root=Path("/mnt/e/MeshFlow-Control-artifacts")
p=root/"checkpoints/A_step0500.pt"
model,optimizer,payload=load_checkpoint(p,device="cpu",trainable=True)
stream=HybridStream(root/"data",root/"training_stream","A");stream.load_state_dict(payload["stream_state"])
steps={float(v["step"]) for v in optimizer.state.values()}
assert payload["pilot_step"]==500 and payload["conditional_total_step"]==5500
assert steps=={5500.} and stream.batch_index==500 and model.readout_mode=="none"
assert not torch.cuda.is_initialized()
atomic_json(root/"audit/A500_fresh_cpu_load.json",dict(status="PASS",checkpoint_sha256=file_sha256(p),
    mode=model.readout_mode,pilot_step=500,old_steps=sorted(steps),data_cursor=stream.batch_index,
    original_parameters=len(list(model.parameters())),CUDA_initialized=False,model_forwards=0,optimizer_updates=0))
print(json.dumps(dict(status="PASS",pilot_step=500,old_adamw_steps=sorted(steps),data_cursor=stream.batch_index)))
