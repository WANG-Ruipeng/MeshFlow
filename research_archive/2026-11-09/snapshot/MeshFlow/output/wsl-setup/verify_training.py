import json
import sys
from pathlib import Path
import numpy as np
import torch
import trimesh
from datasets.mesh_dataset import ObjaverseDataset, collate_fn, save_mesh
from inference import build_model_from_config, _make_sample_fn
from flow_matching import create_transport

checkpoint_path = Path(sys.argv[1] if len(sys.argv) > 1 else 'output/wsl5080-smoke/checkpoints/00000100.pt')
experiment_dir = checkpoint_path.parent.parent
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
config = checkpoint['config']
model = build_model_from_config(config).cuda().eval()
np.random.seed(123)
data = ObjaverseDataset(config['data']['data_path'], training=False,
                        noise_sort='ot', use_permut_aug=False)
batch = collate_fn([data[0]])
x1 = batch['tokens'].cuda().float()
x0 = batch['noise'].cuda().float()
mask = batch['masks'].cuda().bool()
y = batch['num_faces'].cuda()
target = x1 - x0
baseline = target[mask].square().mean().item()
results = {'zero_velocity_mse': baseline, 'checkpoint': str(checkpoint_path)}
with torch.no_grad():
    for state_key in ('model', 'ema'):
        model.load_state_dict(checkpoint[state_key])
        losses = []
        for time in (0.1, 0.3, 0.5, 0.7, 0.9):
            t = torch.full((1,), time, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model((1-time)*x0 + time*x1, t, y=y, mask=mask)
            losses.append((prediction.float()-target)[mask].square().mean().item())
        results[state_key + '_velocity_mse'] = sum(losses)/len(losses)
mesh_stats = []
for path in sorted((experiment_dir / ('infer_' + checkpoint_path.stem)).glob('*.obj')):
    mesh = trimesh.load(path, force='mesh', process=False)
    assert len(mesh.faces) > 0 and np.isfinite(mesh.vertices).all()
    mesh_stats.append({'path': str(path), 'vertices': len(mesh.vertices), 'faces': len(mesh.faces)})
assert len(mesh_stats) == 2
results['generated_meshes'] = mesh_stats
if '--preview' in sys.argv:
    # Short runs have a cold EMA; inspect current trained weights separately.
    model.load_state_dict(checkpoint['model'])
    transport = create_transport(**config['transport'])
    sample_config = dict(config['sample'], num_sampling_steps=50)
    sample_fn = _make_sample_fn(transport, sample_config, 0)
    torch.manual_seed(42)
    z = torch.randn(1, x1.shape[1], 9, device='cuda')
    with torch.no_grad():
        sample = sample_fn(torch.cat([z, z]), model.forward_with_cfg,
                           y=y, mask=mask, cfg_scale=sample_config['cfg_scale'])[-1][0]
    preview_dir = experiment_dir / 'current_weights_preview'
    preview_dir.mkdir(exist_ok=True)
    save_mesh(sample.float().cpu().numpy(), str(preview_dir / 'sample.obj'), max_val=1/0.3762)
    save_mesh(x1[0].cpu().numpy(), str(preview_dir / 'ground_truth.obj'), max_val=1/0.3762)
    results['current_weights_preview'] = str(preview_dir / 'sample.obj')
print(json.dumps(results, indent=2))
(experiment_dir / 'verification.json').write_text(json.dumps(results, indent=2))