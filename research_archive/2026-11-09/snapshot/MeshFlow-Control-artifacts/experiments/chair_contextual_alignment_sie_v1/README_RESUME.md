# Run and resume this fixed pilot

Canonical artifact root: /mnt/e/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1
Source repository: /mnt/e/MeshFlow-Control
Training Python: /mnt/e/MeshFlow-Control-artifacts/venv/bin/python

Read TASK_REQUEST.txt, protocol.json and METHOD_AND_DEVIATIONS.md first. Keep START and all historical artifacts read-only. The public model default is unchanged. No git commit/push is part of this run.

The durable total entry is:

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B \
  /mnt/e/MeshFlow-Control/tools/run_contextual_alignment.py \
  --stage all --resume \
  --out /mnt/e/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1 \
  --data /mnt/e/MeshFlow-Control-artifacts/data \
  --start /mnt/e/MeshFlow/native_t1/checkpoints/chair_hybrid_start_total5000.pt \
  --teacher-python /mnt/e/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1/software/teacher_venv/bin/python
```

A completed fixed task must remain stopped; this is a recovery entry, not permission to repeat the experiment. Existing teacher cache, preflight, checkpoints and sampled cells are reused only through the controller's checks. Do not launch an additional copy while its recorded process is alive.

Stages are teacher, preflight, run, evaluate, or all. All does not redownload assets or choose a new dataset. CPU registration/OT and reviewed teacher rendering are already completed. It resumes the prescribed remaining computation and produces the full registered output manifest. The fixed GPU lock is /tmp/wrp_meshflow_jedge5_diag_gpu0.lock; cumulative ownership, including loading/saving/waits, is limited to21600 seconds in audit/budget/gpu_occupancy.jsonl.

## Actual recovery state

- orchestration/status.json and processes.jsonl identify controller phase and child process.
- audit/budget/calls.jsonl reserves each real call before execution and retains returns/errors.
- training/A, B, C/status.json record whole completed optimizer steps. Ordinary SIGINT/SIGTERM is handled at an update boundary and writes a recoverable checkpoint.
- INTERRUPTED_SAFE can resume the exact saved state, data cursor and RNG. Completed1000 endpoints are hash-checked and reused.
- Incomplete updates, partially written outputs or unresolved budget reservations require explicit engineering review. Never restart from START, redraw a batch, delete a ledger, or relabel a retry to hide consumed calls.
- Checkpoint500 is a recovery/statistics checkpoint only. There are no500-step generated panels.
- outputs/<label>/<request_id>/ stores each genuinely new RAW NPZ,51-state trajectory, PLY and receipt.32 START cells remain verified read-only references to original outputs.
- outputs/manifest.json contains all192 cells. Missing or failed metrics are retained in tables. A zero scoring exit code alone is not completion; analysis/results.json must report192 scored outputs and288 complete diversity pairs.
- AI_VISUAL_REVIEW requires actual image inspection after gallery rendering. Human usability remains NOT_RUN.

## Pure inference exports

inference/A_generator_step1000.pt, B_generator_step1000.pt and C_generator_step1000.pt use the strict meshflow_contextual_alignment_generator_v1 schema. They contain194 generator tensors only, no aligner/teacher/optimizer or inference READOUT. tools/sample_contextual_alignment.py loads these with training.context_checkpoint.load_generator; the sampling process asserts that alignment-head and teacher modules were not imported. All registered rollouts use the unchanged Euler50 path.

Large training checkpoint files preserve the generator, head where applicable, full AdamW, RNG, cursor, config and source identity. Do not confuse a candidate endpoint with original START or use strict=False.

## Main results and visual evidence

EXPERIMENT_REPORT.md is the final conclusion. per_output_metrics.csv, paired_noise_metrics.csv, paired_parent_metrics.csv and sie_summary.csv retain the complete numeric results, including bounds/statuses. analysis/ contains thresholds, tails, leave-one-parent sensitivity and diversity. OPEN_GALLERY.html provides all source-hidden pages first, with source-reference pages separate. visual_review.csv records actual observations without inventing human ratings.

teacher_manifest.json identifies official assets, preprocessing, geometry binding, encoded calls and feature statistics. teacher_render_contact_sheet.png is a review image only and was never fed into the teacher. code.diff and source hash snapshots preserve the executed implementation. Engineering fixes retain their pre-fix source/diff/receipts under engineering/ or teacher/engineering/.
