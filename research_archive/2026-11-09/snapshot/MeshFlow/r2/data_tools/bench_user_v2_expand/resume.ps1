# Completed data-only task. Default recovery verifies sealed artifacts; never trains.
$ErrorActionPreference = 'Stop'
& wsl.exe -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_user_v2_expand.finalize verify
if ($LASTEXITCODE -ne 0) { throw "Bench v2 data verification failed (exit $LASTEXITCODE). Inspect preserved receipts." }
