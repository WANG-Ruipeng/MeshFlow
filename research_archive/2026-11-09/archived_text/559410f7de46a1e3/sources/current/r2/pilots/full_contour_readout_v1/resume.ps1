param([ValidateSet("status","all")][string]$Stage="status")
$ErrorActionPreference="Stop"
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.full_contour_readout_v1.orchestrate --stage $Stage
if ($LASTEXITCODE -ne 0) { throw "Pilot stopped; inspect durable job receipts before resuming." }
