# Bench real-part condition data

CPU preparation only. No generator/denoiser, optimizer, noise sampling, OT,
new weight or model-default changes. The frozen 64 TRAIN / 8 DEV Bench parents
and their original mesh versions are reused from `r2_bench_control_transfer_v1`.

The authoritative output is
`experiments/r2_bench_parts_v1/data/manifest.json`.
This contains **reviewed source inputs**, not generated completions.
More conditions do not mean more independent training objects.

## Pipeline

1. `construct.py` proposes whole original source-face subsets for seat surfaces,
   legs/support frames and back surfaces. This is a geometric proposal, not a
   semantic classifier. A narrow-board exception is independently recorded.
2. `quality.py` verifies bitwise C/Y partition, source identity, local C topology,
   finite C-C intersections and source-adjacent interface labels. The original
   detector definitions are imported read-only. No source mesh is repaired.
3. `render.py` produces depth-correct opposite views. An actual AI_PROXY review
   determines the visible part identity and extent. Hidden/ambiguous geometry
   is not automatically accepted. The review records the exact image hash.
4. `publish.py combine` unions accepted same-parent face IDs into at most three
   combination proposals per parent. Each combination receives a separate
   geometry check and actual image review; accepted components do not guarantee
   an acceptable union.
5. `publish.py finalize` exports exact paired data, inference-only inputs,
   per-condition tables, compatibility partitions and source-hash verification.

## Data contracts

- `data/inputs/*.npz`: only `C`, indexed `C_vertices`, original/model C face
  arrays, reversible within-face corner permutations, requested total `N`, and
  `K`. No Y, source owner/alias labels, part names or complete source mesh.
- `data/inputs/*.obj`: C only, in model coordinates with original source
  winding. It is a convenient geometry preview/import; NPZ is authoritative
  for bitwise preservation. Original winding is not an outward-normal proof.
- `data/paired/*.npz`: C and exact complement Y plus face/vertex identity and
  pre-OT arrays. C and Y reconstruct the unchanged source by original face IDs.
  These are training-data artifacts, not a unique-source reconstruction metric.
- `quality/**/supervision_only/*.npz`: labels derived from source Y, for
  training/diagnostic use only. Never use them as C-only inference predictions.
- `data/conditions.csv`: actual reviewed role, visible extent, N/K, area/face
  fractions, components, boundary count, compatibility and paths. Proposal IDs
  retain their old names for traceability; use `part_type`, not the filename,
  for the corrected semantic category.
- `data/current_seam_compatible.json`: subset satisfying the current Bench
  entry and current single-edge slot data contract. This says nothing about
  trained performance on the new parts.
- `data/native_requires_entry_extension.json`: valid input regions needing an
  entry/layout extension, e.g. a closed leg with no boundary. They are retained
  intact; no artificial cuts are introduced to satisfy the current SEAM code.

An isolated leg surface, a complete indexed leg, a U-shaped support, and a
seat-plus-support assembly are not interchangeable. The extent field and
component count retain that distinction. Part names are bookkeeping, not
additional model features. A backless bench has no invented back.

C validity does not certify a clean Y. Source duplicate faces, internal
intersections and other target defects are not silently removed; this task
does not run an exhaustive full-Y intersection audit. Area fractions use the
sum of the original source-face areas, including source duplicate sheets.

For a later authorized experiment, select parents uniformly first, then
explicitly choose parts/variants within the parent. Sampling flat condition
rows would overexpose parents that admit more variants. Freeze train/evaluation
requests and handle compatibility before model use; this preparation runs no
training and does not replace the old experiment's frozen panel.

## Resume / verify

From the repository root in the existing WSL environment:

```bash
PYTHONPATH=/mnt/e/MeshFlow/r2/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B \
  -m r2.data_tools.bench_parts_v1.run
```

Add `--resume` to finish missing CPU stages. Completed outputs are reused and
checked; mismatched outputs are never overwritten. Missing actual image-review
records stop their dependent publication phase. The entry never invents reviews
and contains no training or sampling command. `verify.py` is the already-run
12-case regression fixture, not an instruction to repeatedly re-audit history.
