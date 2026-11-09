$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ExperimentRoot = 'E:\MeshFlow\experiments\r2_bench_control_transfer_v1'
$EvidenceRoot = Join-Path $ExperimentRoot 'cleanup\final_rolling_retirement'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
$Targets = @(
    'E:\MeshFlow\experiments\r2_bench_control_transfer_v1\training\PLAIN\rolling.pt',
    'E:\MeshFlow\experiments\r2_bench_control_transfer_v1\training\SEAM\rolling.pt'
)
$Protected = @(
    'E:\MeshFlow\experiments\r2_bench_control_transfer_v1\training\PLAIN\step1000.pt',
    'E:\MeshFlow\experiments\r2_bench_control_transfer_v1\training\PLAIN\step3000.pt',
    'E:\MeshFlow\experiments\r2_bench_control_transfer_v1\training\SEAM\step1000.pt',
    'E:\MeshFlow\experiments\r2_bench_control_transfer_v1\training\SEAM\step3000.pt'
)

function Assert-SafeFile([string]$Path) {
    $Item = Get-Item -LiteralPath $Path -Force
    if ($Item.PSIsContainer) { throw "Not an individual file: $Path" }
    $Resolved = (Resolve-Path -LiteralPath $Path).ProviderPath
    $Boundary = $ExperimentRoot.TrimEnd('\') + '\'
    if (-not $Resolved.StartsWith($Boundary, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes experiment: $Resolved"
    }
    $Current = $Item
    while ($null -ne $Current) {
        if (($Current.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse component is forbidden: $($Current.FullName)"
        }
        if ($Current.FullName -eq $ExperimentRoot) { break }
        if ($Current -is [IO.FileInfo]) { $Current = $Current.Directory } else { $Current = $Current.Parent }
    }
    if ($null -eq $Current) { throw "Did not reach experiment boundary: $Path" }
    return $Resolved
}

function Write-NewJson([string]$Path, $Object) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to replace evidence: $Path" }
    [IO.File]::WriteAllText($Path, ($Object | ConvertTo-Json -Depth 20) + "`n", $Utf8)
}

function Record-File([string]$Path) {
    $Resolved = Assert-SafeFile $Path
    $Item = Get-Item -LiteralPath $Resolved
    return [pscustomobject]@{
        path = $Resolved
        bytes = $Item.Length
        sha256 = (Get-FileHash -LiteralPath $Resolved -Algorithm SHA256).Hash.ToLowerInvariant()
        last_write_utc = $Item.LastWriteTimeUtc.ToString('o')
    }
}

if (Test-Path -LiteralPath $EvidenceRoot) { throw 'Retirement directory already exists; no implicit retry.' }
$AuditPath = Join-Path $ExperimentRoot 'training_audit\checkpoint_audit.json'
$Audit = Get-Content -LiteralPath $AuditPath -Encoding utf8 -Raw | ConvertFrom-Json
if (@($Audit).Count -ne 2) { throw 'Expected two final checkpoint audits' }
foreach ($Arm in @('PLAIN','SEAM')) {
    $AuditedArm = @($Audit | Where-Object { $_.arm -eq $Arm })
    if ($AuditedArm.Count -ne 1 -or $AuditedArm[0].step -ne 3000 -or -not $AuditedArm[0].CPU_rng_roundtrip) {
        throw "Final CPU audit incomplete: $Arm"
    }
    $Training = Join-Path $ExperimentRoot "training\$Arm"
    $Status = Get-Content -LiteralPath (Join-Path $Training 'status.json') -Encoding utf8 -Raw | ConvertFrom-Json
    if ($Status.status -ne 'COMPLETE' -or $Status.step -ne 3000 -or $Status.updates -ne 3000) {
        throw "Final training endpoint not complete: $Arm"
    }
    foreach ($Stage in @('raw','post')) {
        $Completion = Get-Content -LiteralPath (Join-Path $ExperimentRoot "evaluation\$Arm\${Stage}_complete.json") -Encoding utf8 -Raw | ConvertFrom-Json
        if ($Completion.status -ne 'COMPLETE' -or $Completion.outputs -ne 24) { throw "Incomplete $Stage : $Arm" }
    }
}

$RetainedBefore = @($Protected | ForEach-Object { Record-File $_ })
foreach ($Record in $RetainedBefore) {
    $SidecarPath = [IO.Path]::ChangeExtension($Record.path, '.json')
    $Sidecar = Get-Content -LiteralPath $SidecarPath -Encoding utf8 -Raw | ConvertFrom-Json
    if ($Record.sha256 -ne $Sidecar.sha256) { throw "Retained hash mismatch: $($Record.path)" }
    if ([IO.Path]::GetFileName($Record.path) -eq 'step3000.pt') {
        $Arm = Split-Path -Leaf (Split-Path -Parent $Record.path)
        $FinalAudit = $Audit | Where-Object { $_.arm -eq $Arm }
        if ($Record.sha256 -ne $FinalAudit.file_sha256) { throw "Final CPU audit hash mismatch: $Arm" }
    }
}
$DeletionRecords = @()
$SidecarsBefore = @()
foreach ($Target in $Targets) {
    $Record = Record-File $Target
    $SidecarPath = [IO.Path]::ChangeExtension($Target, '.json')
    $Sidecar = Get-Content -LiteralPath $SidecarPath -Encoding utf8 -Raw | ConvertFrom-Json
    if ($Sidecar.step -ne 2900 -or $Record.sha256 -ne $Sidecar.sha256) { throw "Unexpected rolling identity: $Target" }
    $DeletionRecords += $Record
    $SidecarsBefore += Record-File $SidecarPath
}

New-Item -ItemType Directory -Path $EvidenceRoot | Out-Null
$Manifest = [pscustomobject]@{
    status = 'VERIFIED_BEFORE_DELETE'
    authorization = 'User requested unnecessary-weight cleanup; parent task explicitly permits only the two superseded step2900 rolling checkpoints.'
    created_utc = [DateTime]::UtcNow.ToString('o')
    experiment_root = $ExperimentRoot
    recursive = $false
    deletion_files = $DeletionRecords
    protected_checkpoints_before = $RetainedBefore
    protected_rolling_sidecars_before = $SidecarsBefore
    checkpoint_audit_sha256 = (Get-FileHash -LiteralPath $AuditPath -Algorithm SHA256).Hash.ToLowerInvariant()
    official_weights_touched = $false
    evidence_RAW_POST_source_data_touched = $false
}
Write-NewJson (Join-Path $EvidenceRoot 'deletion_manifest.json') $Manifest

$Deleted = @()
foreach ($Record in $DeletionRecords) {
    $Resolved = Assert-SafeFile $Record.path
    if ($Resolved -notin $Targets -or $Resolved -in $Protected) { throw "Not whitelisted: $Resolved" }
    $Item = Get-Item -LiteralPath $Resolved
    if ($Item.Length -ne $Record.bytes -or $Item.LastWriteTimeUtc.ToString('o') -ne $Record.last_write_utc) {
        throw "Target changed after verification: $Resolved"
    }
    Remove-Item -LiteralPath $Resolved -Force
    if (Test-Path -LiteralPath $Resolved) { throw "Delete not confirmed: $Resolved" }
    $Deleted += [pscustomobject]@{ path=$Resolved; bytes=$Record.bytes; sha256=$Record.sha256; deleted_utc=[DateTime]::UtcNow.ToString('o') }
    Write-NewJson (Join-Path $EvidenceRoot ("deleted_" + $Deleted.Count + '.json')) $Deleted[-1]
}

$RetainedAfter = @($Protected | ForEach-Object { Record-File $_ })
for ($i=0; $i -lt $RetainedBefore.Count; $i++) {
    if ($RetainedBefore[$i].sha256 -ne $RetainedAfter[$i].sha256 -or $RetainedBefore[$i].bytes -ne $RetainedAfter[$i].bytes) {
        throw 'Protected checkpoint changed during deletion'
    }
}
foreach ($Record in $SidecarsBefore) {
    if ((Get-FileHash -LiteralPath $Record.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Record.sha256) {
        throw 'Historical rolling sidecar changed'
    }
}
$Bytes = [int64](($Deleted | Measure-Object -Property bytes -Sum).Sum)
$Receipt = [pscustomobject]@{
    status='COMPLETE'; deleted=$Deleted; deleted_files=$Deleted.Count; bytes_released_logical=$Bytes
    GiB_released_logical=$Bytes/1GB; protected_checkpoints_after=$RetainedAfter
    protected_before_after_identical=$true; old_rolling_json_unchanged=$true
    retired_scope='Only training/PLAIN/rolling.pt and training/SEAM/rolling.pt step2900 recovery backups. Their preserved rolling.json records now describe retired historical files. Use retained matching step3000 checkpoints and sidecars for final endpoint identity.'
    final_step=3000; official_weights_touched=$false; evidence_RAW_POST_source_data_touched=$false
}
Write-NewJson (Join-Path $EvidenceRoot 'cleanup_receipt.json') $Receipt
[IO.File]::WriteAllText((Join-Path $EvidenceRoot 'RETIRED_REFERENCES.md'), "# Superseded rolling recovery references`n`nThe two preserved training/{PLAIN,SEAM}/rolling.json records remain immutable historical identities for retired step2900 recovery backups. Only their rolling.pt files were removed after final step3000 checkpoint audit, all 48 RAWs and all 48 POSTs completed. Both step1000.pt and step3000.pt per arm remain protected and were hashed before and after deletion. This retirement does not remove official weights, sources, datasets, reports, RAW or POST. See deletion_manifest.json and cleanup_receipt.json for exact paths and hashes.`n", $Utf8)
$Receipt | ConvertTo-Json -Depth 12
