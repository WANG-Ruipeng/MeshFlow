# Frozen Chair BACK_FULL transfer check

This experiment is a zero-training scope check of the retained PLAIN_S2000
SEAM generator with the retained LONG C-only layout predictor. The authoritative
budget is eight conditions and sixteen fixed-noise requests. It does not change
the maintained sampling default.

Results: `experiments/r2_chair_back_transfer_v1/REPORT.md`.
The gallery contains every request, with RAW, BASE_POST, HINGE_OPP015 and
FINAL_POST at shared view/scale and fixed C-centered section positions.

From the repository, using the existing WSL environment:

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.chair_back_transfer_v1.resume
```

The default command prints status only. `--execute` resumes only missing
computational stages of the frozen manifest, refuses concurrent writers and
unresolved neural attempts, and never trains or replaces noises. Completed
artifacts are reused. Do not use a new experiment directory to silently repeat
a failed call. The fixed run is complete; no resume is currently necessary.

Data selection and input hashes are in `data_registration.json` and
`eval_manifest.json`. `registration.json` freezes generation and scientific
source hashes; `evaluation/score_registration.json` versions the unchanged
metric definitions and bridge bookkeeping. Pre-execution wrapper fixes are
preserved under `audits/score_preflight_before_execution/`.

`worker.py` executes the retained rollout, one official Chair D0 per output,
and the original protected geometry export. `post_cpu.py` invokes the retained
HINGE/ALPHA/HOLE engines. `evaluate.py` preserves per-output results, identities,
face pairs and fixed original-face denominators. `render.py` invokes the existing
renderer. `audit.py` checks identities and ledgers without model calls.

`summarize.py`, `record_dev_review.py`, and `closeout.py` only summarize saved
evidence; they are not scientific reruns. Visual observations are actual,
nonblind AI_PROXY reviews, not scores automatically interpreted as adoption.
Their files use exclusive creation to protect completed evidence.

All inherited weights, source meshes and prior experiment records are read-only.
No weights are copied here. No installation, training, cleanup, commit or push
is part of this check.
