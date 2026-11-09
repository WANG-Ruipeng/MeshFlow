param([ValidateSet('Verify','ValidatePending')][string]$Stage='Verify')
$ErrorActionPreference='Stop'
if ($Stage -eq 'Verify') {
    wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_user_relation_rebuild_v1.closeout verify_delivery
} else {
    if (Test-Path -LiteralPath 'E:\MeshFlow\experiments\r2_bench_user_relation_rebuild_v1\DATASET_READY.json') {
        throw 'Sealed dataset exists. Use Verify; no repeated OT or mutable ledger after seal.'
    }
    wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_user_v2.validate --manifest experiments/r2_bench_user_relation_rebuild_v1/manifest.json --out experiments/r2_bench_user_relation_rebuild_v1/validation
}
if ($LASTEXITCODE -ne 0) { throw "CPU dataset stage failed: $LASTEXITCODE" }
