# Optional training losses

The architecture is Native T1 + Geo. Choose its training objective explicitly with `train --context-encoder geo --loss-recipe NAME`:

| Name | Objective | Selection |
| --- | --- | --- |
| `fm` | Original masked free-coordinate FM | Default; no auxiliary target preparation or backward |
| `surface` | FM + gated/ramped L5 surface-distribution supervision | Optional |
| `edge5` | FM + gated/ramped finite interface-edge loss + L5 | Optional JEdge5; `train-edge5` is a shorthand |

The registered names and frozen metadata live in `native_t1/objectives.py`. The CLI reads that same name list. Selecting a recipe does not select new model layers or change the sampler. The current coefficients, FP32 time gate, ramp and normalization are recorded in each geometry-objective checkpoint and validated on loading. They are historical pilot settings, not universal defaults for future losses.

For example, replace `NAME` with one of the three registered names and use a different output directory for each branch:

```bash
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-geo native_t1/checkpoints/tgeo_g0_cumulative2500.pt \
  --context-encoder geo --loss-recipe NAME \
  --continue-stream --updates 500 \
  --out native_t1/runs/my_NAME_trial
```

This example requires a locally supplied checkpoint. A fresh clone can build one from the [official-asset training sequence](fm_baseline.md#build-from-official-assets). All branches create fresh AdamW and restore only the saved input stream. Nothing schedules multiple experiments automatically.

## Adding a future candidate

Future implemented candidates should use this same explicit selector. A name alone does not implement a loss: complete the following code and validation together.

1. Add the loss and any training-only target construction under `losses/`. Keep source IDs and GT out of `model_inputs`, the Geo encoder and the sampler. Predicted geometry must retain gradients; preparing labels must not advance the training RNG.
2. Add a new recipe name and versioned metadata in `objectives.py`, with its own agreed formula, coefficient, gate, ramp and normalization. Implement its target preparation and auxiliary dispatch. Preserve the meaning of existing recipes so old checkpoints retain their identity. If a candidate needs different timing or denominators, represent them explicitly instead of inheriting JEdge5 settings by accident.
3. Extend the allowed geometry recipes in `geometry_checkpoint.py` and the training-objective constructor. Checkpoints must save and validate the complete recipe. FM schemas must continue to reject auxiliary-loss metadata; unsupported recipes must fail rather than fall back.
4. Run CPU mathematical fixtures and gradient checks, target/permutation alignment, normalization and checkpoint-identity tests before an explicitly budgeted model run. Compare against FM from the same initialization and input stream with a matched update budget. Record failed gates instead of silently substituting a formula or coefficient.
5. Document the new option and evidence. Keep weights, generated outputs, caches and experiment controllers outside Git. Do not change the default recipe based on a single successful case.

The training hook is `train_step(..., objective=..., additional_step=...)`: an objective prepares batch targets and returns a differentiable auxiliary term for each microbatch. It shares the original model forward and backward with FM. Inference continues through the same 50-step clamped Euler sampler and has no loss computation. New objective checkpoints can use the generic `sample --profile trained-geo --checkpoint ...` route once their recipe is registered and validated; a new sampling algorithm is unnecessary.

Only the three names in the table are currently implemented in the portable trainer. Retired experimental losses and the Boundary Cancellation candidate that failed its CPU gate are not selectable options.
