param([ValidateSet("status", "all")][string]$Stage = "status")
$ErrorActionPreference = "Stop"
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- env PYTHONPATH=/mnt/e/MeshFlow/r2/src:/mnt/e/MeshFlow OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.nft_training_c_frozen_v1.run --stage $Stage
exit $LASTEXITCODE
