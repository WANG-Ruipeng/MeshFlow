# Optional JEdge5: finite interface supervision

JEdge5 is an optional loss recipe on the FM baseline: Native In-Context T1 with the T_geo context encoder, trained with free-face flow matching, a finite interface-edge objective and a surface-measure objective. Its explicit shortcut is `train-edge5`; sampling requires `--profile edge5`. The registered task remains two chair objects with N112 and mixed K4/8/12; this is an executable research recipe, not an industrial or generalization claim.

## Model and fixed objective

Known triangles use time 1; free triangles use sampled flow time t. T_geo encodes each known triangle's centroid, sorted edge lengths, area and unsigned normal outer product. It adds 149,888 parameters with a zero-initialized exit when first attached. Its P=I operator passes no messages between different known faces.

For update s and effective batch B=8, training uses:

~~~text
x_hat = x_t + (1 - t) * v_theta
g(t) = 1[t >= 0.5]
r(s) = min(s / 50, 1)

L = L_FM
    + r(s) / 8 * sum_b g(t_b) *
      (4.071385484299878 * D_edge(b)
       + 0.3266104383520167 * L_surface(b))
~~~

The original masked FM denominator covers all valid free coordinates. The auxiliary denominator remains 8, including inactive samples; no compensation rescales the active fraction. Coefficients are frozen from the existing pilot and not recalibrated during portable training. The ramp starts with each new invocation. There is no time reweighting of FM, scheduler or new EMA.

`D_edge` uses GT source connectivity only during training. An eligible source edge has exactly two incident faces: one known and one free. The corresponding predicted free edge is compared with the fixed known edge in both directions using finite-segment distances. Five quadrature positions 0, 0.25, 0.5, 0.75 and 1 have weights 1/8, 1/4, 1/4, 1/4 and 1/8. Fixed GT edge lengths weight the eligible-edge mean. Coordinates and thresholds use the fixed GT bbox diagonal. There is no predicted nearest-edge matching.

Segment projection uses a midpoint-centered parameter clamped to the finite segment, with squared-length denominator epsilon=1e-12. Short or collapsed predicted edges remain in the loss. A segment collapsed exactly at the symmetric GT midpoint can have positive loss and zero gradient; the objective does not guarantee recovery from every collapsed state.

`L_surface` is the existing L5 area/unsigned-normal surface-measure objective. Each free triangle contributes three fixed barycentric quadrature points. A spatial/normal kernel is averaged over spatial scales 0.02, 0.05 and 0.10, with normal scale 0.5, after bbox normalization. Predicted area weights are divided by fixed GT area, not renormalized to predicted unit mass. Predicted positions, areas and unsigned normals retain gradients. This direct kernel objective uses no Sinkhorn solver.

The endpoint estimate comes from the existing training forward. Source identities and free GT supervise the losses; they are never model inputs. At inference there is no extra loss evaluation, guidance, projection, welding or repair.

## Start from the local G0 checkpoint

~~~bash
python -B -m native_t1 train-edge5 \
  --inputs native_t1/data/chair_n112.npz \
  --init-geo native_t1/checkpoints/tgeo_g0_cumulative2500.pt \
  --continue-stream --updates 500 \
  --out native_t1/runs/jedge5_3000
~~~

This expands to `train --context-encoder geo --loss-recipe edge5`. The local G0 file SHA256 is `3a66659e86586f54d2ec30ed3d4a18275ab69ba0f8cba2866ef67b340c481c0e`; generator state SHA256 is `6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392`. Continuation restores its seed1010 stream at batch1000/sample8000 and constructs fresh AdamW.

The new run has a fixed +500 budget and cumulative3000 endpoint. Checkpoint filenames record additional and cumulative updates. The output directory must not already exist; the command does not launch generation or expand its budget.

## Build the model from official assets

Fresh clones have no local weights. Install the [environment and official assets](../README.md#environment-and-assets), prepare `native_t1/data/chair_n112.npz`, then run the stages below. Each invocation creates fresh AdamW: lr=1e-5, betas=(0.9,0.95), weight_decay=0, clip=1, effective batch8/microbatch1. Saved optimizer momentum is not resumed.

This sequence documents the existing initialization lineage. It produces your own weights; it does not confer historical hashes or performance or promise bitwise equality across hardware/software environments.

1. Official chair EMA to pure Native500, fresh seed10 stream.

~~~bash
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --official-checkpoint /weights/last.pt \
  --context-encoder none --loss-recipe fm --seed 10 --updates 500 \
  --out native_t1/runs/t1_500
~~~

2. Pure500 to pure1000, another fresh seed10 stream.

~~~bash
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-t1 native_t1/runs/t1_500/additional_step500_cumulative_step500.pt \
  --context-encoder none --loss-recipe fm --seed 10 --updates 500 \
  --out native_t1/runs/a_1000
~~~

3. Pure1000 to pure1500, continuing that saved input stream.

~~~bash
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-t1 native_t1/runs/a_1000/additional_step500_cumulative_step1000.pt \
  --context-encoder none --loss-recipe fm --continue-stream --updates 500 \
  --out native_t1/runs/a_1500
~~~

4. Attach a fresh geo encoder at pure1500; train FM to geo2000 with a fresh seed1010 stream.

~~~bash
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-t1 native_t1/runs/a_1500/additional_step500_cumulative_step1500.pt \
  --context-encoder geo --encoder-seed 1010 --loss-recipe fm \
  --seed 1010 --updates 500 --out native_t1/runs/geo_2000
~~~

5. Continue geo FM to G0_2500, preserving the trained encoder and input stream.

~~~bash
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-geo native_t1/runs/geo_2000/additional_step500_cumulative_step2000.pt \
  --context-encoder geo --loss-recipe fm --continue-stream --updates 500 \
  --out native_t1/runs/g0_2500
~~~

6. Continue G0_2500 with JEdge5 for the final +500 updates.

~~~bash
python -B -m native_t1 train-edge5 \
  --inputs native_t1/data/chair_n112.npz \
  --init-geo native_t1/runs/g0_2500/additional_step500_cumulative_step2500.pt \
  --continue-stream --updates 500 \
  --out native_t1/runs/jedge5_3000
~~~

`--continue-stream` requires the identical prepared archive and cannot be combined with `--seed`. `--encoder-seed` applies only in stage4, where the branch is first created. Loading a trained geo checkpoint never resets its learned encoder.

## Sample and preserve checkpoint identity

~~~bash
python -B -m native_t1 sample --profile edge5 \
  --checkpoint native_t1/runs/jedge5_3000/additional_step500_cumulative_step3000.pt \
  --condition native_t1/data/chair_n112.npz \
  --condition-key parent0_K12_constraints \
  --seed 10101 --out native_t1/runs/sample_jedge5
~~~

When explicitly selected without a checkpoint override, the edge5 profile resolves `native_t1/checkpoints/jedge5_cumulative3000.pt`. This local export uses `native_t1_geo_objective_training_v1` and preserves the generator, objective metadata and input stream while omitting optimizer state. A different serialized file SHA from the training checkpoint is expected; the generator state SHA identifies unchanged learned tensors. No weight is bundled or downloaded automatically.

Sampling uses only C and noise. Its 50 Euler steps retain known coordinates exactly at each recorded state, while free triangles remain a triangle soup. Known-coordinate preservation is not a shared-edge or watertightness guarantee.

## Ablation appendix

The explicit training command supports `--loss-recipe fm`, `surface` and `edge5`. Controlled comparisons start from the same G0_2500, saved stream, +500 budget and fresh AdamW, then use fixed evaluation noises:

| Recipe | Auxiliary objective | Ablation meaning |
| --- | --- | --- |
| `fm` | None | Without auxiliary geometry supervision; retains T_geo |
| `surface` | Gated/ramped lambda_surface * L_surface | Without interface-edge supervision |
| `edge5` | Gated/ramped lambda_edge * D_edge + lambda_surface * L_surface | Optional JEdge5 |

Use `train --context-encoder geo --loss-recipe RECIPE` with an independent output directory. These are optional explicitly budgeted controls, not an automatic sweep. A pure T1 model without the geo encoder is a separate architectural control; it is not the `fm` row above. To sample an ablation, use `sample --profile trained-geo --checkpoint PATH`; the explicit `edge5` profile rejects other objective recipes. The retained local ablation filenames and hashes are listed in the [compact checkpoint index](../DEPENDENCIES.md#local-compact-checkpoint-index).

## Existing evidence and limits

The completed pilot compared five fixed models on two already observed objects, three fixed K12 patches per object and four new noise seeds, with matched update counts and unchanged sampling. The six parent/patch cells were equally weighted. It did not establish new-object generalization.

| Existing macro result | L0: FM only | JEdge5 |
| --- | ---: | ---: |
| Free symmetric surface RMS (% bbox) | 1.090078 | 0.871574 |
| Boundary Gamma RMS (% bbox) | 1.041462 | 0.760189 |
| Free/free illegal intersection pairs | 72.541667 | 67.583333 |

These averages have costs. Compared with the combined L3+L5 control, JEdge5 improved macro RMS and FF counts but reduced Gamma coverage. Its P0 patch2 coverage worsened, and one output had a large triangle-shape tail. P1 patch2 accounted for about 97.8% of the net Gamma improvement over that combined control. Surface-only retained a lower average FF count than JEdge5. That original pilot did not contain a D_edge-only arm; the later fixed four-cell check below fills that cell.

Exact known/free shared edges and vertices remained zero in audited outputs. JEdge5 is geometry supervision, not demonstrated connectivity or watertightness repair. The current sampling default is [Chair HYBRID START](chair_start.md); JEdge5 and surface-only remain explicit N112 comparison recipes.

The historical seven-loss screen is not a public training entry point. Local archives and a cleanup maintenance index may retain provenance after temporary checkpoints are deleted; no historical result JSON, maintenance index or removed checkpoint is required by the commands above. This code/documentation cleanup performs no new training, sampling or performance experiment.

## Later fixed Edge x L5 check

The subsequent fixed-coefficient experiment reused the same starting G0, input stream and 500-update budget, added Edge-only, and freshly generated all four models at seeds10201-10204. New-context macro results (patch3/4 on the same two seen parents) were:

| Metric | L0 FM | Edge-only | L5 | JEdge5 |
| --- | ---: | ---: | ---: | ---: |
| Remaining symmetric RMS (% bbox) | 2.30916 | 2.56477 | 2.63505 | 2.58270 |
| Gamma p95 (% bbox) | 4.58340 | 4.75867 | 4.77175 | 5.08931 |
| FF illegal pairs | 87.0625 | 92.8750 | 92.4375 | 85.7500 |

Relative to Edge-only, L5 reduced mean FF pairs by7.125 on new contexts, but worsened mean RMS and Gamma p95. Joint new-context precision gains over L0 did not replicate, including leave-one-seed-out means. Old-context averages retained partial gains and P0/patch2 remained a failure case. Patch3 overlaps a historical context by11/12 faces; patch4 has lower overlap. These results support retaining an optional quality/precision tradeoff, not universal superiority or necessity.

The current main workflow is [Native T1 + Geo + FM](fm_baseline.md). No method change is implied by preserving the JEdge5 implementation. The new Boundary Cancellation proposal failed its CPU crack/overlap gates before any model update and was not added to this public loss menu.
