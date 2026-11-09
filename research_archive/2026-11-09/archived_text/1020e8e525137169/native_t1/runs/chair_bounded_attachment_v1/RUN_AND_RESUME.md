# Run and resume

CPU environment: CUDA_VISIBLE_DEVICES empty; OMP/MKL/OPENBLAS threads=2.

Entry: python -B -m experiments.mf_h1_local0.chair_bounded_attachment_v1.execute --through all

The controller verifies frozen inputs and completed results. It never repeats pending/failed calls automatically. Stage 2 is authorized only by phase1_gate.json status PASS; a failed or incomplete phase 1 ends the fixed task with explicit NOT_RUN trajectory cells. Do not modify matching, thresholds, geometry limits, sampling math, model, or query definitions to retry.

Read-only report regeneration: python -B -m experiments.mf_h1_local0.chair_bounded_attachment_v1.reporting
Offline gallery: python -B -m experiments.mf_h1_local0.chair_bounded_attachment_v1.gallery
