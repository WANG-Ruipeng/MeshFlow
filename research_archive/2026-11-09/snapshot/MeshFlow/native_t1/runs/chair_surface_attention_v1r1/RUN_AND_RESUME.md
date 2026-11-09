# Finite repair entry

From WSL:

```bash
cd /mnt/e/MeshFlow
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.chair_surface_attention_v1r1.run_all --resume
```

The completed GPU phase is reused, never rerun. It verifies the protected original run and frozen repair hashes, then regenerates the report from stored results. The independently frozen CPU reference and its ledgers are already complete and never silently replayed. An interrupted dispatch requires inspecting its durable reservations before an explicit repair; no uncounted restart is allowed.

Executed schedule: 76 new forwards / 20 zero-update backwards, cumulative 86 / 21 of the original 96 / 24 hard limits. H_CONT and SURFACE cost have two measurements per task; MOMENT one, disclosed in the amendment. Unused allowances do not authorize new scientific jobs. Formal training and new generation have no entry in this repair.

The old failed experiment remains read-only. Its known baseline raw is referenced, not copied or regenerated. Source diffs, hashes, resources, CPU reference protocols and all measurement CSV/NPZ evidence are retained here.
