$ErrorActionPreference = 'Stop'
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$checkpointRoot = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot 'experiments\r2_terminal_reward_nft_v1\checkpoints'))
$manifestPath = Join-Path $PSScriptRoot 'cleanup_manifest.json'
$receiptPath = Join-Path $PSScriptRoot 'cleanup_receipt.json'
$journalPath = Join-Path $PSScriptRoot 'deletion_journal.jsonl'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$review = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'checkpoint_review.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($review.status -ne 'PASS') { throw 'Independent review did not pass.' }
if (Test-Path -LiteralPath $receiptPath) { throw 'Cleanup already has a receipt. Do not repeat.' }
if (Test-Path -LiteralPath $journalPath) { throw 'Partial deletion journal exists; inspect manually, no automatic retry.' }
if ($manifest.candidates.Count -ne 30) { throw 'Unexpected candidate count.' }
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w|[0-9.]*)?\.exe$' -and $_.CommandLine -match 'terminal_reward_nft_v1|nft_training_c_frozen_v1'
} | Select-Object ProcessId,Name)
$wslRows = @(wsl -d Ubuntu -- ps -eo pid,comm,args)
if ($LASTEXITCODE -ne 0) { throw 'WSL process check failed.' }
$activeWsl = @($wslRows | Where-Object { $_ -match '\s(python\S*)\s' -and $_ -match '(terminal_reward_nft_v1|nft_training_c_frozen_v1)' })
if ($active.Count -gt 0 -or $activeWsl.Count -gt 0) { throw 'NFT Python job active; do not remove checkpoints.' }
$validated = @()
foreach ($row in $manifest.candidates) {
    if ($row.path -notmatch '^experiments/r2_terminal_reward_nft_v1/checkpoints/(NFT|SELF_FM)/round(0[0-9]|1[0-4])\.pt$') { throw "Path outside deletion whitelist: $($row.path)" }
    $absolute = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $row.path))
    if (-not $absolute.StartsWith($checkpointRoot + '\',[System.StringComparison]::OrdinalIgnoreCase)) { throw 'Escaping checkpoint directory.' }
    $resolved = (Resolve-Path -LiteralPath $absolute).ProviderPath
    if ($resolved -ne $absolute) { throw 'Unexpected resolved path.' }
    $cursor = $absolute
    while ($cursor.StartsWith($workspaceRoot,[System.StringComparison]::OrdinalIgnoreCase)) {
        $item = Get-Item -LiteralPath $cursor -Force
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse point prohibited: $cursor" }
        if ($cursor -eq $workspaceRoot) { break }
        $cursor = [System.IO.Path]::GetDirectoryName($cursor)
    }
    $item = Get-Item -LiteralPath $absolute
    if ($item.PSIsContainer -or $item.Length -ne $row.bytes) { throw 'Candidate size/type mismatch.' }
    $digest = (Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($digest -ne $row.sha256) { throw "Candidate changed: $absolute" }
    $validated += [pscustomobject]@{ path=$row.path; absolute=$absolute; bytes=[int64]$row.bytes; sha256=$digest }
}
if (($validated.path | Sort-Object -Unique).Count -ne 30) { throw 'Duplicate whitelist paths.' }
$reviewPaths = @($review.deletable_paths | ForEach-Object { [System.IO.Path]::GetFullPath($_) } | Sort-Object)
$actualPaths = @($validated.absolute | Sort-Object)
if (@(Compare-Object $reviewPaths $actualPaths).Count -ne 0) { throw 'Independent whitelist differs.' }
$driveBefore = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='E:'"
$started = [DateTime]::UtcNow.ToString('o')
$utf8 = [System.Text.UTF8Encoding]::new($false)
foreach ($row in $validated) {
    # Exact file removal only. No recursion, wildcard, cross-shell file mutation or directory delete.
    Remove-Item -LiteralPath $row.absolute -Force
    if (Test-Path -LiteralPath $row.absolute) { throw 'Deletion did not remove file.' }
    $entry = @{ path=$row.path; bytes=$row.bytes; sha256=$row.sha256; status='DELETED'; utc=[DateTime]::UtcNow.ToString('o') }
    [System.IO.File]::AppendAllText($journalPath, ($entry | ConvertTo-Json -Compress) + "`n", $utf8)
}
$retained = @()
foreach ($row in $manifest.retained_hash_checked) {
    $absolute = Join-Path $workspaceRoot $row.path
    $digest = (Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($digest -ne $row.sha256) { throw "Retained weight changed: $absolute" }
    $retained += @{ path=$row.path; sha256=$digest; status='HASH_UNCHANGED' }
}
$driveAfter = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='E:'"
$total = [int64](($validated | Measure-Object bytes -Sum).Sum)
$receipt = [ordered]@{
    schema='nft_closeout_cleanup_v1'; status='COMPLETE'; started_utc=$started; completed_utc=[DateTime]::UtcNow.ToString('o');
    deleted_files=$validated.Count; logical_deleted_bytes=$total; logical_deleted_MiB=$total / 1MB;
    free_bytes_before=[int64]$driveBefore.FreeSpace; free_bytes_after=[int64]$driveAfter.FreeSpace;
    observed_free_bytes_change=([int64]$driveAfter.FreeSpace - [int64]$driveBefore.FreeSpace);
    disk_change_caveat='Concurrent unrelated filesystem activity may affect observed free-space change.';
    candidate_hashes_verified_before_removal=$true; exact_file_whitelist_only=$true;
    active_NFT_processes_before=0; retained_hash_checked=$retained;
    deleted=$validated | Select-Object path,bytes,sha256;
    model_calls=0; training_updates=0; new_generations=0; git_commit=$false; git_push=$false;
    replay_limit='Intermediate round00..14 weight replay retired. Final endpoints and all evidence retained.'
}
[System.IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 8) + "`n", $utf8)
[pscustomobject]$receipt | Select-Object status,deleted_files,logical_deleted_bytes,logical_deleted_MiB,observed_free_bytes_change | ConvertTo-Json
