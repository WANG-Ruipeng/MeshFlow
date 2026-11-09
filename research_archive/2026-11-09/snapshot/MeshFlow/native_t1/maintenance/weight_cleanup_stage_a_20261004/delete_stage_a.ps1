$ErrorActionPreference = 'Stop'
$workspacePath = [IO.Path]::GetFullPath('E:\MeshFlow')
$auditPath = [IO.Path]::GetFullPath($PSScriptRoot)
if (-not $auditPath.StartsWith($workspacePath + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Audit outside workspace' }
$prepared = Get-Content -LiteralPath (Join-Path $auditPath 'prepared.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($prepared.status -ne 'PREPARED' -or $prepared.candidate_count -ne 7) { throw 'Missing validated preparation' }
$candidates = Get-Content -LiteralPath (Join-Path $auditPath 'deletion_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$receiptPath = Join-Path $auditPath 'deletion_receipts.jsonl'
if (Test-Path -LiteralPath $receiptPath) { throw 'Existing deletion ledger; inspect before resuming' }
$allowed = @(
    'native_t1/runs/chair_domain_control_v1/checkpoints/Full_latest.pt',
    'native_t1/runs/chair_domain_control_v1/checkpoints/Geo_latest.pt',
    'native_t1/runs/chair_domain_control_v1/checkpoints/T1_latest.pt',
    'native_t1/runs/control_mvp48_budget_single_v1/checkpoints/Full_latest.pt',
    'native_t1/runs/control_mvp48_budget_single_v1/checkpoints/Geo_latest.pt',
    'native_t1/runs/control_mvp48_budget_single_v1/checkpoints/T1_latest.pt',
    'native_t1/runs/control_mvp48_budget_single_v1/checkpoints/T1_fit2_probe_latest.pt'
)
if ($candidates.Count -ne 7 -or @(Compare-Object $allowed @($candidates.path)).Count -ne 0) { throw 'Allowlist mismatch' }
$verified = @()
foreach ($item in $candidates) {
    $targetPath = (Resolve-Path -LiteralPath (Join-Path $workspacePath $item.path)).Path
    if (-not $targetPath.StartsWith($workspacePath + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Target outside workspace' }
    if ($targetPath -ne $item.absolute_path -or $item.tier -ne 'A_STALE_RECOVERY') { throw 'Target identity changed' }
    $fileInfo = Get-Item -LiteralPath $targetPath -Force
    if ($fileInfo.PSIsContainer -or (([int]$fileInfo.Attributes -band 1024) -ne 0)) { throw 'Expected ordinary checkpoint file' }
    if ($fileInfo.Length -ne $item.bytes) { throw 'Size mismatch' }
    $digest = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($digest -ne $item.sha256) { throw 'Hash mismatch' }
    $verified += [PSCustomObject]@{path=$targetPath; relative_path=$item.path; bytes=$item.bytes; sha256=$digest}
}
if (($verified | Measure-Object -Property bytes -Sum).Sum -ne 10950114685) { throw 'Total size mismatch' }

# All seven exact paths, types and full hashes are validated before the first deletion.
# No recursion, shell handoff, wildcard deletion or manifest mutation is performed.
foreach ($item in $verified) {
    Remove-Item -LiteralPath $item.path -Force
    if (Test-Path -LiteralPath $item.path) { throw 'Deletion failed' }
    $receipt = [ordered]@{time=(Get-Date -Format o); status='DELETED_AUTHORIZED_STAGE_A'; relative_path=$item.relative_path; bytes=$item.bytes; sha256=$item.sha256}
    ($receipt | ConvertTo-Json -Compress) | Add-Content -LiteralPath $receiptPath -Encoding UTF8
}
[PSCustomObject]@{deleted_files=$verified.Count; deleted_bytes=($verified | Measure-Object -Property bytes -Sum).Sum; recursive_deletions=0} | ConvertTo-Json
