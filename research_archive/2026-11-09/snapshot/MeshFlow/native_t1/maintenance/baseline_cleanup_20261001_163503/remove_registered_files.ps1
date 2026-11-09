$ErrorActionPreference = 'Stop'
$workspaceRootPath = [IO.Path]::GetFullPath('E:\MeshFlow').TrimEnd('\')
$maintenancePath = Join-Path $workspaceRootPath 'native_t1\maintenance\baseline_cleanup_20261001_163503'
$manifestPath = Join-Path $maintenancePath 'cleanup_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.status -ne 'REGISTERED_NOT_DELETED') { throw 'Cleanup was not freshly registered' }
if ($manifest.checkpoint_compaction.status -ne 'PASS') { throw 'Compact checkpoint not verified' }
$allowedDirectories = @(
 'experiments\mf_h1_local0\edge_factorial_pilot',
 'experiments\mf_h1_local0\boundary_cancellation_pilot',
 'output\t1_publish_20260930\checkout',
 'output\tgeo_publish_20260930_165000\clean_checkout'
) | ForEach-Object { [IO.Path]::GetFullPath((Join-Path $workspaceRootPath $_)).TrimEnd('\') }
$allowedExact = @(
 'native_t1\runs\edge_factorial_20261001_123000\Edge_only\additional_step250_cumulative_step2750.pt',
 'native_t1\runs\edge_factorial_20261001_123000\Edge_only\additional_step500_cumulative_step3000.pt',
 'native_t1\runs\edge_factorial_20261001_123000\fixed_seed10201_patch0.png',
 'native_t1\runs\edge_factorial_20261001_123000\fixed_seed10201_patch1.png',
 'native_t1\runs\edge_factorial_20261001_123000\fixed_seed10201_patch2.png',
 'native_t1\runs\edge_factorial_20261001_123000\fixed_seed10201_patch3.png',
 'native_t1\runs\edge_factorial_20261001_123000\fixed_seed10201_patch4.png',
 'output\patch_tgeo_m10_driver.py'
) | ForEach-Object { [IO.Path]::GetFullPath((Join-Path $workspaceRootPath $_)) }
function Assert-WorkspacePath([string]$pathValue) {
 $absolute = [IO.Path]::GetFullPath($pathValue)
 if (-not $absolute.StartsWith($workspaceRootPath + '\',[StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $absolute" }
 $cursor = Get-Item -LiteralPath $absolute -Force
 while ($null -ne $cursor -and $cursor.FullName -ne $workspaceRootPath) {
  if (($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse point rejected: $($cursor.FullName)" }
  $cursor = $cursor.Parent
  if ($null -eq $cursor) { $cursor = (Get-Item -LiteralPath $absolute).Directory; $absolute = $cursor.FullName }
 }
}
$archivePath = [IO.Path]::GetFullPath($manifest.source_archive.path)
Assert-WorkspacePath $archivePath
if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.source_archive.sha256) { throw 'Source ZIP changed' }
$compactPath = Join-Path $workspaceRootPath 'native_t1\checkpoints\ablations\edge_only_cumulative3000.pt'
Assert-WorkspacePath $compactPath
if ((Get-FileHash -LiteralPath $compactPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.checkpoint_compaction.destination.sha256) { throw 'Compact changed' }
$targets = @()
foreach ($entry in $manifest.remove) {
 $target = [IO.Path]::GetFullPath($entry.path)
 Assert-WorkspacePath $target
 $expected = [IO.Path]::GetFullPath((Join-Path $workspaceRootPath $entry.relative))
 if (-not $target.Equals($expected,[StringComparison]::OrdinalIgnoreCase)) { throw 'Relative/absolute mismatch' }
 $inAllowed = $allowedExact -contains $target
 foreach ($directory in $allowedDirectories) {
  if ($target.StartsWith($directory+'\',[StringComparison]::OrdinalIgnoreCase)) { $inAllowed = $true }
 }
 if (-not $inAllowed) { throw "Unapproved cleanup scope: $target" }
 $item = Get-Item -LiteralPath $target -Force
 if ($item.PSIsContainer -or $item.Length -ne $entry.bytes) { throw "File identity differs: $target" }
 if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Hash differs: $target" }
 $targets += $target
}
if ($targets.Count -ne 240 -or ($targets | Select-Object -Unique).Count -ne 240) { throw 'Unexpected cleanup count' }
Write-Output "Validated 240 explicit file targets inside approved workspace directories; compact and ZIP hashes match."
$deleted = @()
try {
 foreach ($target in $targets) {
  Remove-Item -LiteralPath $target -Force
  $deleted += $target
 }
 foreach ($directory in $allowedDirectories) {
  if (Test-Path -LiteralPath $directory) {
   Assert-WorkspacePath $directory
   $children = @(Get-ChildItem -LiteralPath $directory -Directory -Recurse -Force | Sort-Object { $_.FullName.Length } -Descending)
   foreach ($child in $children) {
    Assert-WorkspacePath $child.FullName
    if (@(Get-ChildItem -LiteralPath $child.FullName -Force).Count -eq 0) { Remove-Item -LiteralPath $child.FullName -Force }
   }
   if (@(Get-ChildItem -LiteralPath $directory -Force).Count -eq 0) { Remove-Item -LiteralPath $directory -Force }
  }
 }
 $status = 'PASS'
} catch {
 $status = 'STOPPED_PARTIAL'
 $failure = $_.Exception.Message
 throw
} finally {
 $result = [ordered]@{ status=$status; deleted_count=$deleted.Count; deleted_paths=$deleted; registered_removed_bytes=$manifest.total_removed_bytes; error=$failure; recursive_delete_used=$false; timestamp=(Get-Date -Format o) }
 $encoding = New-Object System.Text.UTF8Encoding($false)
 [IO.File]::WriteAllText((Join-Path $maintenancePath 'deletion_result.json'),(($result | ConvertTo-Json -Depth 8)+[Environment]::NewLine),$encoding)
}
Write-Output "Removed $($deleted.Count) registered files; scientific records and retained checkpoints remain."