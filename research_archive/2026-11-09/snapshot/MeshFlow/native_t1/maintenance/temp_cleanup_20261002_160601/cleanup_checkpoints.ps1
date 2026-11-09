param([ValidateSet('Prepare','Apply','Verify')][string]$Mode = 'Prepare')
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath('E:\MeshFlow')
$audit = Join-Path $root 'native_t1\maintenance\temp_cleanup_20261002_160601'
$latest = 'native_t1\runs\equal_nfe_confirmation_20261002_152000'
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Save-Json($value, $name) {
    [IO.File]::WriteAllText((Join-Path $audit $name), (ConvertTo-Json -InputObject $value -Depth 12), $utf8)
}
function Safe-Path([string]$relative) {
    $full = [IO.Path]::GetFullPath((Join-Path $root $relative))
    if (-not $full.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $relative" }
    $current = $full
    while ($current -ne $root) {
        if (Test-Path -LiteralPath $current) {
            $entry = Get-Item -LiteralPath $current -Force
            if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $current" }
        }
        $current = Split-Path -Parent $current
    }
    return $full
}
function File-State([string]$relative) {
    $path = Safe-Path $relative
    $item = Get-Item -LiteralPath $path -Force
    if ($item.PSIsContainer) { throw "Expected file: $relative" }
    return [pscustomobject]@{relative=$relative; bytes=$item.Length; modified_utc=$item.LastWriteTimeUtc.ToString('o'); sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
if ($Mode -eq 'Prepare') {
    if (Test-Path -LiteralPath (Join-Path $audit 'deletion_manifest.json')) { throw 'Manifest already exists; do not overwrite' }
    $rows = New-Object System.Collections.Generic.List[object]
    $recommendations = Get-Content -LiteralPath (Join-Path $audit 'experiments_checkpoint_recommendations.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($row in $recommendations.checkpoints) {
        if ($row.action -eq 'delete') { $rows.Add([pscustomobject]@{relative=$row.relative.Replace('/','\'); reason=$row.reason; counterpart=$null; kind='checkpoint'}) }
    }
    foreach ($arm in @('R_B','R_J')) {
        foreach ($name in @('raw_additional250_cumulative2750.pt','EMA_additional250_cumulative2750.pt','raw_additional500_cumulative3000.pt')) {
            $reason = if ($name -like '*250*') {'Retired replay intermediate; training logs, raw outputs and final EMA retained'} else {'Replay raw final has canonical generator state; canonical FM/JEdge5 retained; wrapper is not byte-identical'}
            $rows.Add([pscustomobject]@{relative="native_t1\runs\jedge5_diagnosis_20261002_102215\stage_d\$arm\$name"; reason=$reason; counterpart=$null; kind='checkpoint'})
        }
    }
    foreach ($arm in @('T_base','T_geo','T_graph')) {
        $rows.Add([pscustomobject]@{relative="native_t1\runs\geometry_context_revised_20260930_153000\$arm\additional_step250_cumulative_step1750.pt"; reason='Retired intermediate; Base/Geo final endpoints and scientific evidence retained'; counterpart=$null; kind='checkpoint'})
    }
    $rows.Add([pscustomobject]@{relative='native_t1\runs\geometry_context_revised_20260930_153000\T_graph\additional_step500_cumulative_step2000.pt'; reason='Retired graph candidate outside maintained model and next M1/M2/M3 comparison'; counterpart=$null; kind='checkpoint'})
    $numeric = 'native_t1\runs\jedge5_numeric_reference_A_20261002_115426'
    $scratchPairs = @(@('code_diff_roundtrip_73k1rebr','source_snapshot'), @('code_diff_roundtrip_ctt65la_','delivery_before_layout_fix\source_snapshot'))
    foreach ($pair in $scratchPairs) {
        $dir = Safe-Path "$numeric\$($pair[0])"
        foreach ($file in Get-ChildItem -LiteralPath $dir -File -Recurse -Force) {
            $suffix = $file.FullName.Substring($dir.Length + 1)
            $rows.Add([pscustomobject]@{relative=$file.FullName.Substring($root.Length+1); reason='Completed git-apply roundtrip scratch; retained source snapshot has identical bytes'; counterpart="$numeric\$($pair[1])\$suffix"; kind='temporary_duplicate'})
        }
    }
    $testPairs = @(@('cpu_tests_preflight.json','sampler_initial.json'), @('cpu_tests_preflight_v2.json','sampler_final.json'))
    foreach ($pair in $testPairs) {
        $rows.Add([pscustomobject]@{relative="experiments\mf_h1_local0\equal_nfe_confirmation\$($pair[0])"; reason='Duplicate CPU test record already retained in completed run'; counterpart="$latest\tests\$($pair[1])"; kind='temporary_duplicate'})
    }
    foreach ($name in @('mesh_dataset','__init__')) {
        $rows.Add([pscustomobject]@{relative="datasets\__pycache__\$name.cpython-310.pyc"; reason='Regenerable Python bytecode; source retained'; counterpart="datasets\$name.py"; kind='bytecode_cache'})
    }
    $manifest = @()
    foreach ($row in $rows) {
        if ($row.relative -match '^(\.git|\.worktrees|native_t1\\checkpoints)\\' -or $row.relative.StartsWith($latest)) { throw "Protected target: $($row.relative)" }
        $state = File-State $row.relative
        if ($row.counterpart) {
            $copy = File-State $row.counterpart
            if ($row.kind -eq 'temporary_duplicate' -and $copy.sha256 -ne $state.sha256) { throw "Copy differs: $($row.relative)" }
        }
        $manifest += [pscustomobject]@{relative=$state.relative; bytes=$state.bytes; modified_utc=$state.modified_utc; sha256=$state.sha256; kind=$row.kind; reason=$row.reason; counterpart=$row.counterpart}
    }
    if (@($manifest.relative | Select-Object -Unique).Count -ne $manifest.Count) { throw 'Duplicate targets' }
    Save-Json $manifest 'deletion_manifest.json'
    $protected = New-Object System.Collections.Generic.List[string]
    foreach ($dir in @('native_t1\checkpoints',$latest,'experiments\mf_h1_local0\assets','downloaded_data')) {
        foreach ($file in Get-ChildItem -LiteralPath (Safe-Path $dir) -File -Recurse -Force) { $protected.Add($file.FullName.Substring($root.Length+1)) }
    }
    foreach ($row in $recommendations.checkpoints) {
        if ($row.action -eq 'keep' -and [long]$row.bytes -gt 1GB) { $protected.Add($row.relative.Replace('/','\')) }
    }
    $protected.Add('output\120m-ot-v-bench\checkpoints\01000000.pt')
    $protected.Add('output\t1_publish_20260930\chair_n112.npz')
    foreach ($arm in @('T_base','T_geo')) { $protected.Add("native_t1\runs\geometry_context_revised_20260930_153000\$arm\additional_step500_cumulative_step2000.pt") }
    foreach ($arm in @('R_B','R_J')) { $protected.Add("native_t1\runs\jedge5_diagnosis_20261002_102215\stage_d\$arm\EMA_additional500_cumulative3000.pt") }
    $snapshot = @($protected | Sort-Object -Unique | ForEach-Object { File-State $_ })
    Save-Json $snapshot 'protected_before.json'
    Save-Json ([pscustomobject]@{timestamp=(Get-Date -Format o); targets=$manifest.Count; checkpoint_count=@($manifest | Where-Object kind -eq 'checkpoint').Count; bytes=($manifest | Measure-Object bytes -Sum).Sum; protected_files=$snapshot.Count; drive_free=(Get-PSDrive E).Free; recursive_delete=$false; ntfs_compression=$false}) 'plan_summary.json'
    Get-Content -LiteralPath (Join-Path $audit 'plan_summary.json') -Raw
} elseif ($Mode -eq 'Apply') {
    $manifest = Get-Content -LiteralPath (Join-Path $audit 'deletion_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not (Test-Path -LiteralPath (Join-Path $audit 'protected_before.json'))) { throw 'Protected snapshot missing' }
    $manifestHash = (Get-FileHash -LiteralPath (Join-Path $audit 'deletion_manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($manifestHash -ne '6635a6c96f61b0ac356184b3e6715baf759dd0b9cd4e3bde41f7c1ff72849547') { throw 'Manifest differs from reviewed approval list' }
    if ($manifest.Count -ne 57 -or @($manifest.relative | Select-Object -Unique).Count -ne 57 -or ($manifest | Measure-Object bytes -Sum).Sum -ne 35971879861) { throw 'Reviewed target count or total differs' }
    $protectedRows = Get-Content -LiteralPath (Join-Path $audit 'protected_before.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $protectedPaths = @{}
    foreach ($item in $protectedRows) { $protectedPaths[$item.relative] = $true }
    foreach ($item in $manifest) { if ($protectedPaths.ContainsKey($item.relative)) { throw 'Deletion overlaps protected snapshot' } }
    $receiptPath = Join-Path $audit 'deletion_receipts.jsonl'
    if (Test-Path -LiteralPath $receiptPath) { throw 'Receipt exists; no automatic replay' }
    # All targets are checked before any mutation. Exact files only; no recursive removal.
    foreach ($row in $manifest) {
        $state = File-State $row.relative
        if ($state.bytes -ne $row.bytes -or $state.sha256 -ne $row.sha256) { throw "Target changed: $($row.relative)" }
        if ($row.kind -eq 'temporary_duplicate' -and (File-State $row.counterpart).sha256 -ne $row.sha256) { throw 'Retained counterpart changed' }
        if ($row.kind -eq 'bytecode_cache') { $null = Safe-Path $row.counterpart; if (-not (Test-Path -LiteralPath (Join-Path $root $row.counterpart) -PathType Leaf)) { throw 'Cache source missing' } }
    }
    foreach ($row in $manifest) {
        $path = Safe-Path $row.relative
        Remove-Item -LiteralPath $path -Force
        if (Test-Path -LiteralPath $path) { throw "Deletion failed: $path" }
        $receipt = [pscustomobject]@{relative=$row.relative; bytes=$row.bytes; sha256=$row.sha256; deleted_at=(Get-Date -Format o)} | ConvertTo-Json -Compress
        [IO.File]::AppendAllText($receiptPath, $receipt + [Environment]::NewLine, $utf8)
    }
    [pscustomobject]@{deleted=$manifest.Count; bytes=($manifest | Measure-Object bytes -Sum).Sum; drive_free=(Get-PSDrive E).Free} | ConvertTo-Json
} else {
    $manifest = Get-Content -LiteralPath (Join-Path $audit 'deletion_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $protected = Get-Content -LiteralPath (Join-Path $audit 'protected_before.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $errors = New-Object System.Collections.Generic.List[string]
    foreach ($row in $manifest) { if (Test-Path -LiteralPath (Join-Path $root $row.relative)) { $errors.Add("Not deleted: $($row.relative)") } }
    foreach ($row in $protected) {
        $state = File-State $row.relative
        if ($state.bytes -ne $row.bytes -or $state.sha256 -ne $row.sha256) { $errors.Add("Protected content changed: $($row.relative)") }
    }
    $targets = @{}
    foreach ($row in $manifest) { $targets[$row.relative]=$true }
    $missing = @()
    foreach ($row in Import-Csv -LiteralPath (Join-Path $audit 'inventory_before.csv')) {
        if (-not $targets.ContainsKey($row.relative) -and -not (Test-Path -LiteralPath (Join-Path $root $row.relative) -PathType Leaf)) { $missing += $row.relative }
    }
    if ($missing.Count) { $errors.Add('Unexpected missing files') }
    $result = [pscustomobject]@{status=$(if ($errors.Count) {'FAIL'} else {'PASS'}); protected_hashes_verified=$protected.Count; expected_deleted=$manifest.Count; unexpected_missing=$missing; errors=@($errors); drive_free=(Get-PSDrive E).Free; training_updates=0; model_forward_calls=0; timestamp=(Get-Date -Format o)}
    Save-Json $result 'verification.json'
    $result | ConvertTo-Json -Depth 8
    if ($errors.Count) { throw 'Verification failed' }
}
