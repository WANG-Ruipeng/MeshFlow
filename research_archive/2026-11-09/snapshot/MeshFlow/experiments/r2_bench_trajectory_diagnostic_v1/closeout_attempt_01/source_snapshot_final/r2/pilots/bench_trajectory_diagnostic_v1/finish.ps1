$ErrorActionPreference = 'Stop'
# Read-only final verification. Does not resume models, score meshes, or train.
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_trajectory_diagnostic_v1.closeout --verify-only
if ($LASTEXITCODE -ne 0) { throw "Bench diagnostic verification failed: $LASTEXITCODE" }
