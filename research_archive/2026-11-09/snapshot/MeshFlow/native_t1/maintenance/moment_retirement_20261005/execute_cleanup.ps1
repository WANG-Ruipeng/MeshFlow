$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath 'E:\MeshFlow').Path.TrimEnd('\')
$maintenanceRoot = Join-Path $workspaceRoot 'native_t1\maintenance\moment_retirement_20261005'
$plan = Get-Content -LiteralPath (Join-Path $maintenanceRoot 'delete_plan.json') -Raw | ConvertFrom-Json
if ($plan.status -ne 'READY') { throw 'Retirement plan is not ready.' }
$archive = Join-Path $maintenanceRoot 'retired_activity_source.zip'
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $plan.source_archive_sha256) { throw 'Source archive differs.' }
$allowedDirectories = @(
    (Join-Path $workspaceRoot 'experiments\mf_h1_local0\chair_moment_channel_v2'),
    (Join-Path $workspaceRoot 'experiments\mf_h1_local0\chair_surface_attention_v1'),
    (Join-Path $workspaceRoot 'experiments\mf_h1_local0\chair_surface_attention_v1r1')
)
$checkpointRoot = Join-Path $workspaceRoot 'native_t1\runs\chair_moment_channel_v2\checkpoints'
$cacheRoot = Join-Path $workspaceRoot 'native_t1\runs\chair_moment_channel_v2\ot_cache'
$checked = @()
foreach ($row in $plan.files) {
    $resolved = (Resolve-Path -LiteralPath $row.path).Path
    if (-not $resolved.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Target escaped workspace: $resolved" }
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Unexpected target type: $resolved" }
    if ($item.Length -ne $row.bytes) { throw "Target size changed: $resolved" }
    $allowed = $false
    if ($row.group -like '*_active_package') {
        foreach ($directory in $allowedDirectories) {
            if ($resolved.StartsWith($directory + '\', [StringComparison]::OrdinalIgnoreCase)) { $allowed = $true }
        }
    } elseif ($row.group -like '*_checkpoint_weights') {
        $allowed = ($item.DirectoryName -eq $checkpointRoot -and $item.Name -match '^(H_CONT|MOMENT_XY)_moment(200|400|600|800|1000)_total(5200|5400|5600|5800|6000)\.pt$')
    } elseif ($row.group -eq 'V2_OT_mapping_NPZ') {
        $allowed = ($resolved.StartsWith($cacheRoot + '\', [StringComparison]::OrdinalIgnoreCase) -and $item.Extension -eq '.npz')
    }
    if (-not $allowed) { throw "Target outside precise approved groups: $resolved" }
    $checked += [PSCustomObject]@{ path = $resolved; bytes = [int64]$item.Length; group = $row.group; sha256 = $row.sha256 }
}
if ($checked.Count -ne 8061) { throw 'Deletion file count differs.' }
$logPath = Join-Path $maintenanceRoot 'deletions.jsonl'
if (Test-Path -LiteralPath $logPath) { throw 'Deletion log already exists; inspect before recovery.' }
$writer = [IO.StreamWriter]::new($logPath, $false, [Text.UTF8Encoding]::new($false))
try {
    foreach ($row in $checked) {
        Remove-Item -LiteralPath $row.path -Force -ErrorAction Stop
        $record = [ordered]@{ action='DELETE_FILE'; path=$row.path; bytes=$row.bytes; group=$row.group; sha256_before=$row.sha256; utc=[DateTime]::UtcNow.ToString('o'); status='DONE' }
        $writer.WriteLine(($record | ConvertTo-Json -Compress)); $writer.Flush()
    }
} finally { $writer.Dispose() }
# All files were enumerated and removed individually. Remove only the verified
# now-empty package directories; never recursively delete a computed run path.
foreach ($directory in $allowedDirectories) {
    $resolved = (Resolve-Path -LiteralPath $directory).Path
    if ($allowedDirectories -notcontains $resolved) { throw 'Directory identity changed.' }
    if (@(Get-ChildItem -LiteralPath $resolved -Force).Count -ne 0) { throw "Unexpected remaining source entry: $resolved" }
    Remove-Item -LiteralPath $resolved -Force -ErrorAction Stop
}
$total = ($checked | Measure-Object -Property bytes -Sum).Sum
[ordered]@{status='DONE';deleted_files=$checked.Count;deleted_bytes=[int64]$total;source_directories_removed=3;model_calls=0;user_authorized=$true} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $maintenanceRoot 'cleanup_result.json') -Encoding UTF8
Write-Output "CLEANUP_DONE files=$($checked.Count) bytes=$([int64]$total)"
