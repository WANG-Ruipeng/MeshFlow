# Frozen HINGE seam-side subsegment CPU test

This is a finite local research test, not the maintained sampling default.
It reads the 28 saved `HINGE_OPP015` meshes, original `PRE_GEO` cluster
identities and frozen query/scoring assets. Both split caps start from that same
baseline. It never invokes a generator, D0, GPU, optimizer or new random stream.

Run in the existing WSL environment from the repository root:

```powershell
wsl -d Ubuntu -- env PYTHONPATH=/mnt/e/MeshFlow/r2/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.tools.hinge_subsegment_v1.resume
```

The completed entry validates all source/input/output hashes and exits.
Partially completed scientific attempts remain protected: no silent overwrite,
threshold relaxation or retry. Missing protected assets produce an error;
do not regenerate them to make the test run. This package references local
frozen assets and does not pretend to be a mesh-only repair tool.

Stages:

1. `census`: real shared-index seam-side / bulk-open-edge eligibility.
2. `engine`: up to eight registered formal trials at 1.5%L and 2.5%L, with
   child shape, whole moved-neighborhood, topology, collision and lineage guards.
3. `scoring` / `collect_scores`: original geometric definitions on actual child
   triangles, plus fixed-original-parent affected unions. Identity outputs inherit
   exact old scores after byte and lineage verification. Original requested N
   and RAW-slot identities are never fabricated.
4. Independent census, split and metric audits; complete paired tables, fixed
   original-camera section views and closeout.

`trial_gallery` visualizes already-saved rejected trials; it is not a second
experiment and rejected trial meshes are not final outputs. The final readable
version lives in `trial_gallery/v2_readable/`.

Result: only `85/n0` accepts the same local split in both settings; its bulk
was already connected. DEV gains no connection. The slight actual-face FF
decrease is denominator dilution; fixed-parent FF and I do not improve.
No promotion or default switch. See
`experiments/r2_hinge_subsegment_v1/REPORT.md` and all per-output data.
