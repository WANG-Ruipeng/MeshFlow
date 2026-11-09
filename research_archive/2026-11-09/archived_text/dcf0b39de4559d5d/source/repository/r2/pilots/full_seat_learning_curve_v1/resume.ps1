param(
    [ValidateSet('status','all')]
    [string]$Stage = 'status'
)
$ErrorActionPreference = 'Stop'
# This entrypoint is synchronous. It does not open a visible background window,
# change a scientific worker count, retry failures or modify registered code.
$TaskWorkspace = 'E:\MeshFlow'
$TaskPython = '/mnt/e/MeshFlow-Control-artifacts/venv/bin/python'
$TaskModule = 'r2.pilots.full_seat_learning_curve_v1.orchestrate'
& wsl.exe -d Ubuntu --cd /mnt/e/MeshFlow -- $TaskPython -B -m $TaskModule --stage $Stage
if ($LASTEXITCODE -ne 0) {
    throw "FULL seat continuation entrypoint stopped with exit code $LASTEXITCODE. Existing call ledgers and child receipts are preserved; no automatic retry was attempted."
}