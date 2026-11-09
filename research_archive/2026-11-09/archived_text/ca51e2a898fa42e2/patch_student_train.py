from pathlib import Path
p=Path('/mnt/e/MeshFlow/.worktrees/a-layout-value-20260924/experiments/a_radial/layout_student/train.py')
s=p.read_text()
s=s.replace("data = [torch.as_tensor(x, dtype=torch.float32, device=dev) for x in (u0, u1, features)]", "data = [torch.as_tensor(u0, dtype=torch.float64, device=dev),\n            torch.as_tensor(u1, dtype=torch.float64, device=dev),\n            torch.as_tensor(features, dtype=torch.float32, device=dev)]")
s=s.replace("'bridge_steps': 32, 'training_time': 'uniform_integer_0_to_31_div32',", "'bridge_steps': 32, 'bridge_computation_dtype': 'float64',\n              'training_time': 'uniform_integer_0_to_31_div32',")
s=s.replace("positions, target_velocity, _, t = discrete_bridge(start_batch, end_batch, k)\n            prediction", "positions, target_velocity, _, t = discrete_bridge(start_batch, end_batch, k)\n            positions = positions.float()\n            target_velocity = target_velocity.float()\n            t = t.float()\n            prediction")
p.write_text(s)
