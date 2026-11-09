# Fixed Bench adaptation evaluation

These entry points consume `experiments/r2_bench_adaptation_strength_v1/registration.json`
and its copied `eval_manifest.json`. They never change training, choose a checkpoint,
evaluate a layout head, draw new noise, or write into a historical experiment.

Run from the repository root in the maintained Python environment with `r2/src`
on the import path. The controller must inspect resources before GPU dispatch and
incrementally choose concurrency. Each worker owns its model and label directory.

```text
python -m r2.pilots.bench_adaptation_strength_v1.eval_plan
python -m r2.pilots.bench_adaptation_strength_v1.sample --label COMMON_S0
python -m r2.pilots.bench_adaptation_strength_v1.sample --label OFFICIAL_S0
python -m r2.pilots.bench_adaptation_strength_v1.sample --label A_S500
python -m r2.pilots.bench_adaptation_strength_v1.sample --label B_S500
python -m r2.pilots.bench_adaptation_strength_v1.sample --label C_S500
python -m r2.pilots.bench_adaptation_strength_v1.sample --label A_S1500
python -m r2.pilots.bench_adaptation_strength_v1.sample --label B_S1500
python -m r2.pilots.bench_adaptation_strength_v1.sample --label C_S1500
```

The first five labels each have the fixed 16 process requests; the last three
each have all 48 requests. This is exactly 224 RAW / 11,200 generator forwards.
A/B share `COMMON_S0`; no duplicate B step-zero sampling is authorized.
Generator exports need `generator_state`, `generator_state_sha256`,
`model_config`, `local_step`, `global_step`, `EMA=False`, and a `.json` SHA receipt.
The registered export path is authoritative.

Only **after all 224 RAW receipts and their artifacts verify**, run once for each
of the eight labels:

```text
python -m r2.pilots.bench_adaptation_strength_v1.post_d0 --label A_S1500
```

This dispatches the unchanged single official Bench D0, hard C/seam restoration,
GEO_C, source C winding restoration and fixed-C free orientation. The wrapper
uses a private function namespace to support different panel sizes; it does not
change historical modules or their output globals. A saved D0 proposal may be
used to finish pending CPU geometry; an unresolved neural attempt is blocked.

CPU stages, after their input producers complete:

```text
python -m r2.pilots.bench_adaptation_strength_v1.evaluation prepare
python -m r2.pilots.bench_adaptation_strength_v1.evaluation raw --workers 2
python -m r2.pilots.bench_adaptation_strength_v1.evaluation post --workers 2
python -m r2.pilots.bench_adaptation_strength_v1.evaluation collect
python -m r2.pilots.bench_adaptation_strength_v1.gallery prepare
python -m r2.pilots.bench_adaptation_strength_v1.gallery run --workers 2
```

`raw` and `post` optionally take `--labels` to dispatch disjoint label sets.
Worker count is measured and capped, not a universal two-worker requirement.
`post` runs the unchanged HINGE_OPP015 → ALPHA → HOLE chain, with original
179-field BASE and FINAL scoring plus preserved candidate and connectivity
ledgers. Closed-C seam metrics remain NA. Every score uses the original query
arrays; source reconstruction is not a success criterion.

Outputs:

- `evaluation/{label}/{request}/raw/`: raw mesh, full trajectory, C source metadata,
  forward ledger and hash receipt.
- `evaluation/{label}/{request}/post/`: official proposal, PRE_GEO, canonical and
  source-oriented BASE mesh, geometry ledger and one-D0 receipt.
- `evaluation/scores/{label}/{request}/RAW/`: full original score and pair details.
- `cpu/{label}/{request}/`: BASE score, three CPU stages, final score, full
  candidate records and direct change ledger.
- `results/`: all 672 scored rows; per-condition and per-parent tables; parent
  macro means with valid denominators; full-field paired B−A and C−B primary
  contrasts, with C−A retained as an additional contrast.
- `gallery/`: 48 final RAW/BASE sheets, 48 final POST sheets and 48 process sheets
  (16 requests × RAW/BASE/FINAL), all with the original 3D cameras and exact
  front/side section positions. Rendering is **not** an actual visual review.

The registered process subset happens to cover LEG_SUPPORT and BACK_SURFACE;
it does not provide a process learning curve for seat inputs. All final48 retain
the original part coverage. Do not change that subset after looking at outputs.

There is no automatic replay of an unresolved forward, score or geometry attempt,
no extra checkpoint selection, and no automatic training extension.
