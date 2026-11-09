param([ValidateSet('verify','analyze','render','supplement')][string]$Stage='verify')
$ErrorActionPreference='Stop'
if ($Stage -eq 'supplement') {
    wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_relation_coverage_v1.supplement
} else {
    wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_relation_coverage_v1.run $Stage
}
if ($LASTEXITCODE -ne 0) { throw "Bench CPU coverage stage failed: $LASTEXITCODE" }
