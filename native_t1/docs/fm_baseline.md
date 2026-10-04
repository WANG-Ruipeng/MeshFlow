# Native T1 + Geo + FM baseline

This historical N112 baseline uses pure free-face flow matching with the Native known/free-time construction and the C-only T_geo encoder. The local L0 cumulative3000 endpoint is the explicit legacy `fm-geo` sampling profile. The current default is [Chair HYBRID START](chair_start.md). It is not the original unconditional MeshFlow checkpoint, nor a JEdge5, FC, angular or boundary-loss model.

This is a two-observed-chair N112 sandbox. Choosing a stable comparison baseline does not establish universal superiority or generalization. JEdge5 remains an explicit optional loss recipe; new losses should be compared with FM from the same initialization, saved input stream and update budget.

## Local endpoint

- Path: `native_t1/checkpoints/ablations/fm_cumulative3000.pt` (the historical directory name is retained to avoid copying or moving weights).
- File SHA256: `0f22ce498ca2c27dc74b43890fb8aad20f0f3e7153aa74ee6d3c14ea445a2660`.
- Generator state SHA256: `f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b`.
- Cumulative updates: 3000. Optimizer momentum was stripped; model tensors and input RNG were retained exactly. Any continuation uses fresh AdamW.
- `fm-geo` validates `expected_recipe="fm"`, so it cannot silently load a surface-only or JEdge5 objective checkpoint.

## Build from official assets

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

6. Continue G0_2500 with pure FM for the final +500 updates.

~~~bash
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-geo native_t1/runs/g0_2500/additional_step500_cumulative_step2500.pt \
  --context-encoder geo --loss-recipe fm --continue-stream --updates 500 \
  --out native_t1/runs/fm_3000
~~~

`--continue-stream` requires the identical prepared archive and cannot be combined with `--seed`. `--encoder-seed` applies only in stage4, where the branch is first created. Loading a trained geo checkpoint never resets its learned encoder.

## Optional losses and evidence

`train --context-encoder geo --loss-recipe fm` keeps the original FM objective. `--loss-recipe surface` and `--loss-recipe edge5` are explicit options; `train-edge5` remains an optional shorthand. They preserve the architecture and sampler. No command searches losses or chooses a checkpoint automatically.

The completed Edge x L5 check on new seeds10201-10204 found the following new-context macro values across two parents and patch3/4:

| Metric | L0 FM | Edge-only | L5 | JEdge5 |
| --- | ---: | ---: | ---: | ---: |
| Remaining symmetric RMS (% bbox) | 2.30916 | 2.56477 | 2.63505 | 2.58270 |
| Gamma p95 (% bbox) | 4.58340 | 4.75867 | 4.77175 | 5.08931 |
| FF illegal pairs | 87.0625 | 92.875 | 92.4375 | 85.7500 |

JEdge5 retained an FF tradeoff but did not reproduce aggregate new-context precision gains. Patch3 overlaps a historical context by11/12 faces; patch4 has lower overlap. This is new-context evidence on seen shapes, not new-object generalization. The Boundary Cancellation candidate subsequently failed its required CPU mathematical gates before any model training. Neither result changes the baseline architecture or FM loss.

This documentation/default-profile change runs no new experiment or training. Historical reports and raw arrays remain local; they are not runtime dependencies.
