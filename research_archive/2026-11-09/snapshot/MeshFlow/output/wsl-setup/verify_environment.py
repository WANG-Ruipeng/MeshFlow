import json
import sys
from pathlib import Path

import torch

print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(0))
print('CUDA architectures:', torch.cuda.get_arch_list())
x = torch.randn(512, 512, device='cuda', requires_grad=True)
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print('GPU BF16 forward/backward: PASS')

import inference
import train
from datasets.mesh_dataset import ObjaverseDataset, collate_fn
assert inference.CHAMFER_AVAILABLE is None, 'Chamfer should not load on import'
data = ObjaverseDataset('downloaded_data/ss_overfit', training=True, noise_sort='ot')
assert len(data) > 0
batch = collate_fn([data[0]])
print('Train samples:', len(data))
print('Batch:', {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)})
assert torch.isfinite(batch['tokens']).all()
assert torch.isfinite(batch['noise']).all()
print('Imports and OT data preparation: PASS')