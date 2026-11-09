param([switch]$Apply)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = 'E:\MeshFlow'
$out = Join-Path $root 'maintenance\bench_weight_cleanup_20261020_2241'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$adapt = 'experiments\r2_bench_adaptation_strength_v1'
$clean = 'experiments\r2_bench_clean_multipart_seam_v1'
$parts = 'experiments\r2_bench_parts_convergence_v1'
$arms = @('A_COMMON_LR1E5','B_COMMON_LR2E6','C_OFFICIAL_LR2E6')

function Write-Json($path, $value) {
    [IO.File]::WriteAllText($path, ($value | ConvertTo-Json -Depth 12), $utf8)
}
function Assert-File($path) {
    $absolute = [IO.Path]::GetFullPath($path)
    if (-not $absolute.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $absolute" }
    $item = Get-Item -LiteralPath $absolute -Force
    if ($item.PSIsContainer) { throw "Not a file: $absolute" }
    $cursor = $item
    while ($null -ne $cursor) {
        if (($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse point: $($cursor.FullName)" }
        if ($cursor.FullName -eq $root) { break }
        if ($cursor -is [IO.DirectoryInfo]) { $cursor = $cursor.Parent } else { $cursor = $cursor.Directory }
    }
    return $item
}
function Record-File($relative, $reason, $requireReceipt) {
    $item = Assert-File (Join-Path $root $relative)
    $hash = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $receiptPath = [IO.Path]::ChangeExtension($item.FullName, '.json')
    if ($requireReceipt) {
        $receipt = Get-Content -LiteralPath $receiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($receipt.sha256 -ne $hash -or [long]$receipt.bytes -ne $item.Length) { throw "Receipt mismatch: $relative" }
    } else { $receiptPath = $null }
    return [pscustomobject]@{ path=$item.FullName; relative_path=$relative; bytes=$item.Length; sha256=$hash; last_write_utc=$item.LastWriteTimeUtc.ToString('o'); reason=$reason; receipt_path=$receiptPath }
}
function Assert-Completed {
    $statuses = @("$clean\training\status.json")
    foreach ($arm in $arms) { $statuses += "$adapt\training\$arm\status.json" }
    foreach ($relative in $statuses) {
        $s = Get-Content -LiteralPath (Join-Path $root $relative) -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($s.status -ne 'COMPLETE' -or $s.local_step -ne 1500) { throw "Not completed: $relative" }
        if (-not $s.checkpoint.path.EndsWith('/checkpoint_step1500.pt')) { throw "Unexpected active checkpoint: $relative" }
    }
}

$deleteSpecs = @()
foreach ($arm in $arms) {
    foreach ($name in @('checkpoint_step500.pt','checkpoint_step1000.pt','rolling.pt')) {
        $deleteSpecs += [pscustomobject]@{relative="$adapt\training\$arm\$name"; reason='Completed arm; superseded intermediate/1250 rolling checkpoint. Final1500 retained.'}
    }
    foreach ($step in @(0,500)) {
        $deleteSpecs += [pscustomobject]@{relative="$adapt\inference\${arm}_S${step}_generator.pt"; reason='Early evaluation export retired; generated outputs and numeric evidence retained.'}
    }
}
$deleteSpecs += [pscustomobject]@{relative="$clean\training\rolling.pt"; reason='Completed CLEAN; stale1250 rolling checkpoint superseded by retained final1500.'}
if ($deleteSpecs.Count -ne 16) { throw 'Unexpected whitelist size' }

Assert-Completed
$planPath = Join-Path $out 'prepared_manifest.json'
if (-not $Apply) {
    if (Test-Path -LiteralPath $planPath) { throw 'Prepared manifest already exists; do not overwrite.' }
    $deletes = @()
    foreach ($spec in $deleteSpecs) {
        $deletes += Record-File $spec.relative $spec.reason $true
        Write-Output "Verified delete $($deletes.Count)/16: $($spec.relative)"
    }
    $keeps = @()
    foreach ($arm in $arms) {
        $keeps += Record-File "$adapt\training\$arm\checkpoint_step1500.pt" 'Final full generator/optimizer/RNG checkpoint' $true
        $keeps += Record-File "$adapt\inference\${arm}_S1500_generator.pt" 'Final standalone inference export' $true
        $keeps += Record-File "$adapt\training\$arm\checkpoint_step0.pt" 'Exact initial parameter reference required by training resume integrity check' $true
    }
    $keeps += Record-File "$clean\training\checkpoint_step1500.pt" 'CLEAN final full checkpoint' $true
    $keeps += Record-File "$clean\inference\BENCH_CLEAN_MULTIPART_S1500_generator.pt" 'CLEAN final generator' $true
    $keeps += Record-File "$clean\inference\BENCH_CLEAN_MULTIPART_S1500_occupancy.pt" 'CLEAN final occupancy head' $true
    foreach ($stage in @('COMMON','PLAIN','SEAM')) {
        $stageDir = Join-Path $root "$parts\training\$stage"
        foreach ($item in (Get-ChildItem -LiteralPath $stageDir -File -Filter '*.pt')) {
            $keeps += Record-File $item.FullName.Substring($root.Length+1) 'Historical Bench named reference; protected' $false
        }
    }
    $smallProtected = @()
    foreach ($experiment in @($adapt,$clean,$parts)) {
        foreach ($item in (Get-ChildItem -LiteralPath (Join-Path $root $experiment) -File)) {
            if ($item.Extension -in @('.md','.json','.jsonl','.csv')) {
                $smallProtected += Record-File $item.FullName.Substring($root.Length+1) 'Original report/manifest/ledger; read-only' $false
            }
        }
    }
    $nonWeightInventory = @()
    foreach ($experiment in @($adapt,$clean)) {
        foreach ($item in (Get-ChildItem -LiteralPath (Join-Path $root $experiment) -File -Recurse)) {
            if ($item.Extension -ne '.pt') {
                $nonWeightInventory += [pscustomobject]@{path=$item.FullName;bytes=$item.Length;last_write_ticks=$item.LastWriteTimeUtc.Ticks}
            }
        }
    }
    Write-Json (Join-Path $out 'nonweight_inventory_before.json') $nonWeightInventory
    Write-Json $planPath ([ordered]@{
        schema='bench_weight_cleanup_v1'; prepared_at=(Get-Date).ToString('o');
        authorization='User explicitly requested cleanup of redundant weights on 2026-10-20'; status='PREPARED';
        scope=@($adapt,$clean); delete=$deletes; keep=$keeps; protected_documents=$smallProtected;
        delete_count=$deletes.Count; delete_bytes=($deletes | Measure-Object bytes -Sum).Sum;
        nonweight_file_count=$nonWeightInventory.Count;
        free_bytes_preparation=[IO.DriveInfo]::new('E:\').AvailableFreeSpace
    })
    Write-Output "PREPARED: $($deletes.Count) files; $([math]::Round(($deletes | Measure-Object bytes -Sum).Sum / 1GB,4)) GiB; retained weights $($keeps.Count)."
    exit 0
}

$plan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
$receiptPath = Join-Path $out 'cleanup_receipt.json'
if (Test-Path -LiteralPath $receiptPath) { throw 'Receipt exists; do not rerun cleanup.' }
$expected = @($deleteSpecs | ForEach-Object { [IO.Path]::GetFullPath((Join-Path $root $_.relative)) })
if ($plan.delete.Count -ne 16) { throw 'Unexpected prepared count' }
foreach ($entry in $plan.delete) {
    if ($entry.path -notin $expected -or [IO.Path]::GetExtension($entry.path) -ne '.pt') { throw "Not on exact whitelist: $($entry.path)" }
    $item = Assert-File $entry.path
    if ($item.Length -ne $entry.bytes -or $item.LastWriteTimeUtc.ToString('o') -ne $entry.last_write_utc) { throw "Changed after preparation: $($entry.path)" }
}
foreach ($entry in $plan.keep) { $null = Assert-File $entry.path }
$freeBefore = [IO.DriveInfo]::new('E:\').AvailableFreeSpace
$removed = @()
try {
    foreach ($entry in $plan.delete) {
        $null = Assert-File $entry.path
        Remove-Item -LiteralPath $entry.path -ErrorAction Stop
        if (Test-Path -LiteralPath $entry.path) { throw "Still exists: $($entry.path)" }
        $removed += $entry
        Write-Json (Join-Path $out 'deletion_progress.json') ([ordered]@{at=(Get-Date).ToString('o');deleted=$removed})
    }
    foreach ($entry in @($plan.keep) + @($plan.protected_documents)) {
        $item = Assert-File $entry.path
        if ($item.Length -ne $entry.bytes -or (Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Protected file changed: $($entry.path)" }
    }
    $inventory = Get-Content -LiteralPath (Join-Path $out 'nonweight_inventory_before.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($entry in $inventory) {
        $item = Get-Item -LiteralPath $entry.path
        if ($item.Length -ne $entry.bytes -or $item.LastWriteTimeUtc.Ticks -ne $entry.last_write_ticks) { throw "Nonweight asset changed: $($entry.path)" }
    }
    $freeAfter = [IO.DriveInfo]::new('E:\').AvailableFreeSpace
    Write-Json $receiptPath ([ordered]@{
        schema='bench_weight_cleanup_v1'; status='COMPLETE'; finished_at=(Get-Date).ToString('o');
        deleted_count=$removed.Count; logical_bytes_removed=($removed | Measure-Object bytes -Sum).Sum;
        free_bytes_before=$freeBefore; free_bytes_after=$freeAfter; observed_free_bytes_delta=($freeAfter-$freeBefore);
        observed_delta_note='Volume-wide free space includes concurrent filesystem activity and ledger writes; logical bytes are exact file lengths.';
        deleted=$removed; retained_weights_sha256_verified=$plan.keep.Count;
        original_documents_sha256_verified=$plan.protected_documents.Count;
        nonweight_size_mtime_verified=$inventory.Count; nonweight_check='Size and mtime; not content rehash of every mesh.';
        model_loads=0; model_forwards=0; training_updates=0; generation=0;
        lifecycle='Deleted intermediate replay is retired. Final full checkpoints and exports retained. Do not regenerate missing checkpoints or rerun historical full-inventory audits as if these files remain.';
        git_commit=$false; git_push=$false
    })
    Write-Output "COMPLETE: $($removed.Count) weights removed; $([math]::Round(($removed | Measure-Object bytes -Sum).Sum/1GB,4)) GiB."
} catch {
    Write-Json (Join-Path $out 'cleanup_failure.json') ([ordered]@{at=(Get-Date).ToString('o');error=$_.Exception.Message;deleted=$removed})
    throw
}
