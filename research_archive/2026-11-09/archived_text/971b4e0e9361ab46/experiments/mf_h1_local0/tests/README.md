# Independent CPU acceptance checks

Run from the repository root with the existing MeshFlow Python environment:

```powershell
wsl -d Ubuntu -- /home/warpwang/miniforge3/envs/meshflow/bin/python -m experiments.mf_h1_local0.tests.run_units --output <new-report-path>.json
```

The runner hides CUDA before importing PyTorch, uses FP64 geometry and FP32
adapter arithmetic, enables deterministic algorithms, and refuses to overwrite
a report. Every check records its input/seed, observed error, fixed threshold,
and PASS/FAIL. Numerical failures remain serializable in the report.

`geometry_oracle.py` is an independent NumPy scalar implementation: it enumerates
boundary active sets and solves the interior least-squares problem. It imports
no training geometry code. Fixtures cover finite segment/triangle interior,
boundary, vertices, repeated points, collinearity, near degeneracy, far queries,
1,000 seeded random comparisons, all triangle permutations, and batched shapes.

Adapter checks use the specified 768 hidden width and four 128-wide adapters.
The symmetry and padding checks use an explicitly nonzero output projection;
the single-token checks inspect nonzero messages before the zero projection.
They verify exact Q/G/P initial tensors, grouped point means, P group mass,
current Xt/hidden/time dependence, Q/G dynamic-input differences confined to the
relation block, permutations, padding including NaN sentinels, empty rows,
m=1/m=3 mixed batches, query chunking, gradients, and serialized 10-step versus
5+resume+5 equality of model, optimizer, all CPU RNG streams and output.

Frozen affine layers appear only in an autograd unit fixture; they are **not an
official backbone or a synthetic chair proxy**, and produce no geometry quality
result. Official strict-load, official identity/rollout, frozen official hash,
and official sampler call counters require the separate official test runner.
The 60 optimizer steps per invocation are disposable CPU probe steps, not
Phase 2 training. No pretrained weight or dataset is touched by this suite.

Executed reports retained:

- `results/cpu_units_attempt01.json`: 191 PASS, 0 FAIL.
- `results/cpu_units_attempt02.json`: 201 PASS, 0 FAIL; added pooled-token equality
  and mixed empty/nonempty batch checks. Geometry oracle max absolute error
  1.5543122344752192e-15 (threshold 1e-10); nonzero FP32 face permutation maximum
  8.195638656616211e-8; m=1/m=3 batch maximum 7.450580596923828e-8; restored state,
  optimizer, RNG and output exactly equal. S parameters: 950,144; each of Q/G/P:
  1,178,496.

Across those two actual invocations: 120 CPU probe optimizer updates, 704 adapter
layer forward calls, and 319 encoder calls. Official model forward calls,
complete official samples, and Phase 2 optimizer updates in this suite: zero.
The later report is the final suite result; both runs remain auditable.
