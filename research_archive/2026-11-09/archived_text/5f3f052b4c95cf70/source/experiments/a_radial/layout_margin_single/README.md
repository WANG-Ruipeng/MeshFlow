# Fixed paired CPU single-parent diagnosis

This independent study uses the four preregistered training parents and one training seed (10). Each parent has two targets: the frozen old teacher and the frozen strict-floor teacher-start endpoint. Every one of the eight jobs uses the same original 29,907-parameter model, fresh CPU AdamW, batch4 and 20,000 updates. The paired bridge-time draw hashes must match. The new target changes both the optimization trajectory and quality margin; the comparison is not a pure scalar-margin intervention or a generalization result.

The saved reference is the complete 2,050-node / 4,096-face surface of the **same frozen radial field**, exactly `RadialSupport.mesh(logr)`. It is not the original raw object mesh. Surface-error Monte Carlo samples are independent of the optimizer's samples; the geometric reference is the same. The deployment network receives only its original proxy256 features.

Preparation and CPU smoke must precede the approved driver. `initial.pt` is an actual fresh CPU step-zero state; it does not reuse the old checkpoint's update counter or optimizer/RNG. Every process hides CUDA and saves only Python/NumPy/Torch CPU RNG. Training is CPU1; deployment is the unchanged CPU4 guarded32-step path.

Run the gated training driver with `python -B -m experiments.a_radial.layout_margin_single.run_all --run RUN`. `follow_evaluation --run RUN` waits for completed training reports, then executes all three fixed observations (0, 5k, 20k), all20 final endpoints, reports, figures and independent read-back. It does not select weights or retry failed attempts.

Individual commands:

- `evaluate --run RUN --mode diagnostics --job JOB --update 5000` reads only the train-label archive and an already-existing preregistered checkpoint.
- `evaluate --run RUN --mode final` requires all eight final checkpoints and their final diagnostics. It retains every Fixed/old-teacher/margin-teacher/old-final/margin-final slot.
- `report --run RUN --render` writes the per-parent summary and four scientific figures.
- `review --run RUN` independently replays all successful CPU diagnostics, recomputes endpoint radii, quality and exact audit, and checks stored MC evidence without resampling.

All directions, final radii, logical int32 connectivity hashes, stored connectivity hashes and whole-file hashes remain distinct. Original-Fixed and Fixed-plus0.1degree angular gates are reported independently. No failed learned endpoint is replaced by Fixed. A zero-motion teacher has an undefined relative-error denominator, with absolute error retained.

Completion or read-back PASS means accounting and reproducibility, not successful quality or learned-method superiority. Runtime is not an isolated benchmark when other experiments are active. Existing source runs and their labels remain immutable.
