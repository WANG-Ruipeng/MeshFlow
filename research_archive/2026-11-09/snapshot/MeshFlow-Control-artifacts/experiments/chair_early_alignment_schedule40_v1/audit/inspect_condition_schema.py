from pathlib import Path
import numpy as np
for p in [next(Path('/mnt/e/MeshFlow-Control-artifacts/data/evaluation/evaluation_reference').glob('341b*/condition.npz')), Path('/mnt/e/MeshFlow/native_t1/runs/chair_domain_control_v1/inputs/generation/A_33ec1e64a02ff718faaadcdc0221eb72_A40_a100_n0.npz')]:
    with np.load(p) as a: print(str(p), {k:(str(a[k].dtype),list(a[k].shape)) for k in a.files})