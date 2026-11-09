$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath 'E:\MeshFlow').Path.TrimEnd('\')
$maintenanceDir = Join-Path $workspaceRoot 'native_t1\maintenance\jedge5_cleanup_20261001_115000'
$manifestPath = Join-Path $maintenanceDir 'cleanup_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$migration = Get-Content -LiteralPath (Join-Path $maintenanceDir 'checkpoint_migration.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.status -ne 'REGISTERED_NOT_DELETED' -or $migration.status -ne 'PASS') {
    throw 'Registered manifest and completed migration are required.'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Save-Manifest {
    [IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 60), $utf8)
}
function Resolve-CheckedFile([string]$relative) {
    $absolute = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $relative.Replace('/', '\')))
    if (-not $absolute.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escaped workspace: $relative"
    }
    $resolved = (Resolve-Path -LiteralPath $absolute).Path
    if ($resolved -ne $absolute) { throw "Resolved path differs: $relative" }
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item.PSIsContainer) { throw "Expected file: $relative" }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Reparse file: $relative"
    }
    $cursor = $item.Directory
    while ($null -ne $cursor -and $cursor.FullName -ne $workspaceRoot) {
        if (($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse directory: $relative"
        }
        $cursor = $cursor.Parent
    }
    if ($null -eq $cursor) { throw "Ancestor check failed: $relative" }
    return $resolved
}
function Check-Hash([string]$path, [string]$expected) {
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "SHA256 changed: $path" }
}

# Verify all replacements and source archive before checking deletion candidates.
if (@($migration.checkpoint_exports).Count -ne 3) { throw 'Expected three replacements.' }
foreach ($record in $migration.checkpoint_exports) {
    if (-not $record.every_tensor_bitwise_equal -or -not $record.input_rng_identical -or $record.strict_load.status -ne 'PASS') {
        throw 'Replacement not verified.'
    }
    Check-Hash (Resolve-CheckedFile $record.destination) $record.sha256
}
if (-not $manifest.source_archive.verified) { throw 'Source archive not verified.' }
Check-Hash (Resolve-CheckedFile $manifest.source_archive.path) $manifest.source_archive.sha256

$checkpointCount = 0
$resolvedCandidates = @()
foreach ($entry in $manifest.entries) {
    $relative = $entry.path
    if ($entry.category -eq 'checkpoint') {
        $oldMatch = $relative -match '^experiments/mf_h1_local0/outputs/loss_screen_20260930_232200/L[0-6]/additional_step(250_cumulative_step2750|500_cumulative_step3000)\.pt$'
        $pilotMatch = $relative -match '^native_t1/runs/l3_l5_interface_pilot_20261001_104600/(J35|JEdge5)/additional_step(250_cumulative_step2750|500_cumulative_step3000)\.pt$'
        if (-not ($oldMatch -or $pilotMatch)) { throw "Checkpoint outside exact scope: $relative" }
        $checkpointCount++
    } elseif ($entry.category -eq 'retired_source') {
        if ($relative -notmatch '^experiments/mf_h1_local0/(loss_screen|l3_l5_interface_pilot)/') {
            throw "Source outside exact scope: $relative"
        }
    } else { throw "Unapproved category: $($entry.category)" }
    $resolved = Resolve-CheckedFile $relative
    if ((Get-Item -LiteralPath $resolved).Length -ne $entry.bytes) { throw "Size changed: $relative" }
    Check-Hash $resolved $entry.sha256
    $resolvedCandidates += $resolved
}
if ($checkpointCount -ne 18 -or ($resolvedCandidates | Select-Object -Unique).Count -ne $resolvedCandidates.Count) {
    throw 'Exact candidate count or uniqueness differs.'
}
$manifest.status = 'VERIFIED_REMOVAL_IN_PROGRESS'
$manifest | Add-Member -NotePropertyName free_bytes_before_removal -NotePropertyValue (Get-PSDrive -Name E).Free
$manifest | Add-Member -NotePropertyName deleted_bytes -NotePropertyValue ([int64]0)
Save-Manifest
try {
    for ($i = 0; $i -lt $manifest.entries.Count; $i++) {
        $entry = $manifest.entries[$i]
        # Re-resolve at the point of deletion; use native PowerShell on exact files only.
        $resolved = Resolve-CheckedFile $entry.path
        if ($resolved -ne $resolvedCandidates[$i]) { throw 'Path changed after verification.' }
        Remove-Item -LiteralPath $resolved -Force
        if (Test-Path -LiteralPath $resolved) { throw "Removal failed: $resolved" }
        $entry.status = 'DELETED'
        $manifest.deleted_bytes += [int64]$entry.bytes
        Save-Manifest
        Write-Output ("Deleted " + $entry.path)
    }
    $manifest.status = 'DELETED_PENDING_PROTECTED_EVIDENCE_CHECK'
} catch {
    $manifest.status = 'STOPPED_PARTIAL_REMOVAL'
    $manifest | Add-Member -NotePropertyName error -NotePropertyValue $_.Exception.Message -Force
    throw
} finally {
    $manifest | Add-Member -NotePropertyName free_bytes_after_removal -NotePropertyValue (Get-PSDrive -Name E).Free -Force
    Save-Manifest
}
