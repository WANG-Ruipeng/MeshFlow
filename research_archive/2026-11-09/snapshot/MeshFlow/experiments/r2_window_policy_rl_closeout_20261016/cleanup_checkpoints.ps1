param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath('E:\MeshFlow')
$runRoot = Join-Path $workspace 'experiments\r2_window_policy_rl_v1'
$trainingRoot = Join-Path $runRoot 'training'
$outRoot = Join-Path $workspace 'experiments\r2_window_policy_rl_closeout_20261016'
$utf8 = [Text.UTF8Encoding]::new($false)
function Write-JsonFile($Path, $Value) {
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 15), $utf8)
}
function File-Record($Path) {
    $item = Get-Item -LiteralPath $Path
    [pscustomobject]@{ path=$item.FullName; bytes=[long]$item.Length; sha256=(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
}
function Assert-SafeFile($Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($trainingRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside training root: $absolute" }
    if ($absolute -notmatch '\\batch_0(?:[0-5][0-9]|6[0-2])\\update\\checkpoint\.pt$') { throw "Not an allowed intermediate checkpoint: $absolute" }
    $file = Get-Item -LiteralPath $absolute
    if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse file: $absolute" }
    # Check every ancestor directory as well as the file itself.
    $dir = Get-Item -LiteralPath (Split-Path -Parent $absolute)
    while ($null -ne $dir -and $dir.FullName.StartsWith($workspace, [StringComparison]::OrdinalIgnoreCase)) {
        if (($dir.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse directory: $($dir.FullName)" }
        $dir = $dir.Parent
    }
    return $absolute
}
$receiptPath = Join-Path $outRoot 'cleanup_receipt.json'
if (Test-Path -LiteralPath $receiptPath) { throw 'Cleanup receipt already exists; inspect it rather than rerunning deletion.' }
$completion = Get-Content -LiteralPath (Join-Path $trainingRoot 'final.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($completion.status -ne 'COMPLETE' -or $completion.batches -ne 64 -or $completion.records.Count -ne 64) { throw 'Training is not complete.' }
$windowsProcesses = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' } | Select-Object ProcessId,Name,CommandLine)
$wslProcesses = @(& wsl -d Ubuntu --exec ps -eo pid,comm,args)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect WSL processes.' }
if (($windowsProcesses | Where-Object { $_.CommandLine -match 'window_policy_rl_v1|train_controller|sample_worker' }).Count -gt 0) { throw 'Related Windows model process is active.' }
if (($wslProcesses | Select-String -Pattern 'window_policy_rl_v1|train_controller|sample_worker').Count -gt 0) { throw 'Related WSL model process is active.' }
$diskBefore = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='E:'"
Write-JsonFile (Join-Path $outRoot 'resource_check.json') ([ordered]@{
    timestamp_utc=[DateTime]::UtcNow.ToString('o'); windows_python_processes=$windowsProcesses
    wsl_relevant_processes=@($wslProcesses | Select-String -Pattern 'python|train_controller|sample_worker' | ForEach-Object { $_.Line })
    disk_free_bytes=[long]$diskBefore.FreeSpace; gpu_query=@(& nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader)
    purpose='Read-only cleanup safety observation; zero model calls or GPU jobs.'
})
$actualWeights = @(Get-ChildItem -LiteralPath $runRoot -Recurse -File -Filter '*.pt')
if ($actualWeights.Count -ne 64) { throw "Expected 64 checkpoint files, found $($actualWeights.Count)." }
$finalPath = Join-Path $trainingRoot 'batch_063\update\checkpoint.pt'
$protectedRelative = @('REPORT.md','README_RESUME.md','resume.ps1','registration.json','training_config.json','code.diff',
    'R2_WINDOW_POLICY_RL_V1_REVIEW.zip','training\final.json','training\batch_063\update\checkpoint.pt',
    'engineering\TRAINING_EXECUTION_AUDIT.json','engineering\FINAL_QUANTITATIVE_AUDIT.json','engineering\FINAL_REVIEW_AUDIT.json',
    'evaluation\analysis\summary.json','evaluation\analysis\per_output_review_metrics.csv')
$protectedBefore = @($protectedRelative | ForEach-Object { File-Record (Join-Path $runRoot $_) })
$finalRecord = $protectedBefore | Where-Object { $_.path -eq $finalPath }
if ($finalRecord.sha256 -ne $completion.checkpoint_sha256) { throw 'Final checkpoint hash mismatch.' }
$allowlist = @(0..62 | ForEach-Object {
    $index = $_
    $absolute = Assert-SafeFile (Join-Path $trainingRoot ('batch_{0:D3}\update\checkpoint.pt' -f $index))
    $record = File-Record $absolute
    $historical = @($completion.records | Where-Object { $_.batch_index -eq $index })
    if ($historical.Count -ne 1 -or $record.sha256 -ne $historical[0].checkpoint_sha256) { throw "Historical checkpoint hash mismatch: $absolute" }
    [pscustomobject]@{ batch_index=$index; path=$record.path; bytes=$record.bytes; sha256=$record.sha256 }
})
$totalBytes = [long](($allowlist | Measure-Object -Property bytes -Sum).Sum)
if ($totalBytes -ne 605818757) { throw "Unexpected cleanup bytes: $totalBytes" }
$inventoryBefore = @(Get-ChildItem -LiteralPath $runRoot -Recurse -File | Select-Object FullName,Length,LastWriteTimeUtc)
$inventoryBefore | Export-Csv -LiteralPath (Join-Path $outRoot 'original_tree_before.csv') -NoTypeInformation -Encoding UTF8
$allowlist | Export-Csv -LiteralPath (Join-Path $outRoot 'checkpoint_allowlist.csv') -NoTypeInformation -Encoding UTF8
Write-JsonFile (Join-Path $outRoot 'cleanup_plan.json') ([ordered]@{
    status='PREPARED'; authorization='User requested organizing this RL result and deleting corresponding unnecessary weights.'
    scope='Exactly batch_000..062 checkpoint.pt; no recursive deletion, no other files.'
    count=$allowlist.Count; logical_bytes=$totalBytes; files=$allowlist; protected_files=$protectedBefore
    historical_replay_retired=$true; final_inference_dependencies_retained=$true
})
if (-not $Execute) { Write-Output "Prepared $($allowlist.Count) files / $totalBytes bytes. No deletion."; exit 0 }
$eventsPath = Join-Path $outRoot 'deletion_events.jsonl'
if (Test-Path -LiteralPath $eventsPath) { throw 'Existing deletion journal; inspect partial state before recovery.' }
$deleted = @()
foreach ($record in $allowlist) {
    $absolute = Assert-SafeFile $record.path
    if ((Get-Item -LiteralPath $absolute).Length -ne $record.bytes) { throw "Size changed: $absolute" }
    Remove-Item -LiteralPath $absolute -Force
    if (Test-Path -LiteralPath $absolute) { throw "Deletion failed: $absolute" }
    $event = [ordered]@{ status='DELETED'; timestamp_utc=[DateTime]::UtcNow.ToString('o'); batch_index=$record.batch_index; path=$absolute; bytes=$record.bytes; sha256_before=$record.sha256 }
    [IO.File]::AppendAllText($eventsPath, (($event | ConvertTo-Json -Compress) + [Environment]::NewLine), $utf8)
    $deleted += [pscustomobject]$event
}
$protectedAfter = @($protectedRelative | ForEach-Object { File-Record (Join-Path $runRoot $_) })
foreach ($before in $protectedBefore) {
    $after = $protectedAfter | Where-Object { $_.path -eq $before.path }
    if ($after.bytes -ne $before.bytes -or $after.sha256 -ne $before.sha256) { throw "Protected file changed: $($before.path)" }
}
$deleteSet = @{}; foreach ($entry in $allowlist) { $deleteSet[$entry.path] = $true }
$preservedCount = 0
foreach ($entry in $inventoryBefore) {
    if ($deleteSet.ContainsKey($entry.FullName)) { continue }
    $after = Get-Item -LiteralPath $entry.FullName
    if ($after.Length -ne $entry.Length -or $after.LastWriteTimeUtc -ne $entry.LastWriteTimeUtc) { throw "Unlisted original file changed: $($entry.FullName)" }
    $preservedCount++
}
$remainingWeights = @(Get-ChildItem -LiteralPath $runRoot -Recurse -File -Filter '*.pt')
if ($remainingWeights.Count -ne 1 -or $remainingWeights[0].FullName -ne $finalPath) { throw 'Unexpected remaining checkpoints.' }
$diskAfter = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='E:'"
Write-JsonFile $receiptPath ([ordered]@{
    status='COMPLETE'; timestamp_utc=[DateTime]::UtcNow.ToString('o'); deleted_count=$deleted.Count
    deleted_logical_bytes=$totalBytes; deleted_MiB=$totalBytes/1MB; deleted_GiB=$totalBytes/1GB
    disk_free_before_bytes=[long]$diskBefore.FreeSpace; disk_free_after_bytes=[long]$diskAfter.FreeSpace
    observed_disk_free_delta_bytes=([long]$diskAfter.FreeSpace-[long]$diskBefore.FreeSpace)
    disk_delta_note='Free-space delta includes other processes and newly written cleanup logs; logical deleted bytes are exact.'
    deleted=$deleted; retained_checkpoint=$finalRecord; protected_hashes_after=$protectedAfter
    other_original_files_preserved_size_and_mtime=$preservedCount
    preservation_scope='All other original files checked for presence/size/mtime; selected protected files additionally SHA256 checked. No claim all buffers were rehashed.'
    models_loaded=0; gpu_jobs=0; training_updates=0; new_samples=0; rescoring_calls=0; commit_or_push=$false
    retired='Historical intermediate-model replay and old controller full-chain checkpoint verification/resume.'
})
Write-Output "COMPLETE: removed $($deleted.Count) intermediate checkpoints; $totalBytes bytes. Final retained and hashes unchanged."
