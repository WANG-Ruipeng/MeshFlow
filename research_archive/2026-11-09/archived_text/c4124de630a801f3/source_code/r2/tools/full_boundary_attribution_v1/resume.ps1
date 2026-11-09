$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'E:\MeshFlow'
$stages = @('prepare','run','aggregate','audit','render','report','package')
foreach ($stage in $stages) {
    $module = 'r2.tools.full_boundary_attribution_v1.' + $stage
    if ($stage -eq 'audit') {
        & wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m $module --stage all
    } else {
        & wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m $module
    }
    if ($LASTEXITCODE -ne 0) { throw ('Stopped at ' + $stage + '; inspect preserved ledger/failure. No automatic retry.') }
}