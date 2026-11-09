# Fixed bounded attachment pilot

Main results: [report](BOUNDED_ATTACHMENT_CORRECTION_REPORT.md), [all outputs](per_output_metrics.csv), [parent pairs](aggregation/paired_parent.csv), [noise pairs](aggregation/paired_noise.csv), [16-row gallery](gallery/index.html), [mesh index](output_asset_index.csv).

From E:\MeshFlow in PowerShell:

```powershell
wsl -e /home/warpwang/miniforge3/envs/meshflow/bin/python -B -u -m experiments.mf_h1_local0.chair_bounded_attachment_v1.execute
```

This entry validates frozen identities and reuses completed projection/scoring/rendering outputs. It never silently replays a failed or unfinished geometric/model call. Actual calls are in ledger/calls.jsonl and accounting.json; finite CPU fixtures are separately counted in tests/project/attempt1/results.json. The sixteen original Gaussians are preserved under inputs/inference/.

The executed phase-1 gate did not authorize phase 2. All 16 TRAJECTORY cells explicitly say NOT_RUN_NO_PHASE1_SIGNAL; no trajectory mesh or score is fabricated. Do not modify the gate or expand this cohort. This delivery does not implement or dispatch an unused GPU experiment after a failed gate.

Old model/raw/scorers/reports were read only. code.diff contains isolated new files. source_snapshots/dependencies/ includes the actual reused scoring/classifier code. No training, default switch, commit or push occurred.
