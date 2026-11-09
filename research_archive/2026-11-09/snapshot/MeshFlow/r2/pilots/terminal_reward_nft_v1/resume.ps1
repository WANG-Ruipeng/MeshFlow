param(
 [ValidateSet('status','pipeline','prepare','learn')][string]$Stage='status',
 [ValidateRange(0,15)][int]$Round=0
)
$ErrorActionPreference='Stop'
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- env PYTHONPATH=/mnt/e/MeshFlow/r2/src:/mnt/e/MeshFlow OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -u -B -m r2.pilots.terminal_reward_nft_v1.run --stage $Stage --round $Round
exit $LASTEXITCODE
