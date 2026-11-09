param([ValidateSet('Prepare','Delete','Verify')][string]$Phase = 'Prepare')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$workspaceRoot = 'E:\MeshFlow'
$recordRoot = 'E:\MeshFlow\native_t1\maintenance\local_anchor_retirement_20261006'
$codeRoot = 'E:\MeshFlow\experiments\mf_h1_local0\chair_bounded_attachment_v1'
$runRoot = 'E:\MeshFlow\native_t1\runs\chair_bounded_attachment_v1'
$boundaryRoot = 'E:\MeshFlow\native_t1\runs\chair_boundary_attachment_v1'
$cacheRoot = Join-Path $boundaryRoot 'ot_cache'
$weightRoot = Join-Path $boundaryRoot 'checkpoints'
$archivePath = Join-Path $recordRoot 'LOCAL_ANCHOR_RETIRED_SOURCE_AND_EVIDENCE.zip'
$planPath = Join-Path $recordRoot 'deletion_manifest.json'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Write-Json($Path, $Value) {
    [IO.File]::WriteAllText($Path, (ConvertTo-Json -InputObject $Value -Depth 40), $utf8)
}
function Read-Json($Path) { Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json }
function Hash($Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Local-Path($Path) {
    if ($Path.StartsWith('/mnt/e/MeshFlow/')) { return Join-Path $workspaceRoot $Path.Substring(16).Replace('/','\') }
    return [IO.Path]::GetFullPath($Path)
}
function Assert-WorkspacePath($Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $full" }
    $walk = $full
    while ($walk.Length -gt $workspaceRoot.Length) {
        if (Test-Path -LiteralPath $walk) {
            $item = Get-Item -LiteralPath $walk -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse path forbidden: $walk" }
        }
        $walk = [IO.Path]::GetDirectoryName($walk)
    }
    return $full
}
function File-Row($File, $Category) {
    [pscustomobject]@{path=$File.FullName; relative=$File.FullName.Substring($workspaceRoot.Length+1).Replace('\','/'); bytes=[long]$File.Length; sha256=(Hash $File.FullName); category=$Category}
}
function Assert-Archive($ExpectedRows) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        if ($zip.Entries.Count -ne $ExpectedRows.Count) { throw 'Archive entry count differs' }
        foreach ($row in $ExpectedRows) {
            $entry = $zip.GetEntry($row.relative)
            if ($null -eq $entry -or $entry.Length -ne $row.bytes) { throw "Archive entry missing/size mismatch: $($row.relative)" }
            $stream = $entry.Open(); $algorithm = [Security.Cryptography.SHA256]::Create()
            try { $actual = [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
            finally { $algorithm.Dispose(); $stream.Dispose() }
            if ($actual -ne $row.sha256) { throw "Archive bytes differ: $($row.relative)" }
        }
    } finally { $zip.Dispose() }
}

Assert-WorkspacePath $recordRoot | Out-Null
if ($Phase -eq 'Prepare') {
    if (Test-Path -LiteralPath $planPath) { throw 'Prepared manifest exists; inspect it rather than overwrite' }
    $archiveRows = @()
    foreach ($target in @($codeRoot,$runRoot)) {
        Assert-WorkspacePath $target | Out-Null
        foreach ($file in @(Get-ChildItem -LiteralPath $target -Recurse -File -Force)) {
            Assert-WorkspacePath $file.FullName | Out-Null
            $archiveRows += File-Row $file 'archived_local_anchor'
        }
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::Open($archivePath, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($row in $archiveRows) {
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$row.path,$row.relative,[IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    } finally { $zip.Dispose() }
    Assert-Archive $archiveRows
    Write-Json (Join-Path $recordRoot 'archive_manifest.json') @{files=$archiveRows; archive_sha256=(Hash $archivePath); verified_entries=$archiveRows.Count; status='PASS'}

    $evidenceRoot = Join-Path $recordRoot 'evidence'
    New-Item -ItemType Directory -Path $evidenceRoot -Force | Out-Null
    foreach ($name in @('BOUNDED_ATTACHMENT_CORRECTION_REPORT.md','phase1_gate.json','accounting.json','per_output_metrics.csv','quality_cost.csv','code.diff','source_hashes.json')) {
        Copy-Item -LiteralPath (Join-Path $runRoot $name) -Destination (Join-Path $evidenceRoot $name)
    }
    Copy-Item -LiteralPath (Join-Path $runRoot 'aggregation') -Destination (Join-Path $evidenceRoot 'aggregation') -Recurse

    $weights = @()
    foreach ($arm in @('A_FM','B_BOUNDARY','C_SEAM','D_BOUNDARY_SEAM')) {
        foreach ($step in @(200,400,600,800,1000)) {
            if ($arm -eq 'A_FM' -and $step -eq 1000) { continue }
            $name = $arm + '_attachment' + $step + '_total' + (5000+$step) + '.pt'
            $path = Assert-WorkspacePath (Join-Path $weightRoot $name)
            $file = Get-Item -LiteralPath $path
            $row = File-Row $file 'unused_boundary_weight'
            $meta = Read-Json ([IO.Path]::ChangeExtension($path,'.json'))
            if ($meta.checkpoint_sha256 -ne $row.sha256) { throw "Weight identity mismatch: $path" }
            $weights += $row
            Write-Output "WEIGHT_IDENTITY $name"
        }
    }
    if ($weights.Count -ne 19) { throw 'Expected fixed 19 weights' }
    $cache = @()
    Assert-WorkspacePath $cacheRoot | Out-Null
    foreach ($file in @(Get-ChildItem -LiteralPath $cacheRoot -Recurse -File -Force)) {
        Assert-WorkspacePath $file.FullName | Out-Null
        if ($file.Extension -notin @('.npz','.json')) { throw "Unexpected cache file: $($file.FullName)" }
        $cache += File-Row $file 'derived_OT_cache'
    }
    $protected = @{}
    $prior = Read-Json (Join-Path $runRoot 'audit\protected_dependencies.json')
    foreach ($row in $prior.rows) { $protected[(Local-Path $row.path)] = $row.after_sha256 }
    $protected[(Join-Path $workspaceRoot 'native_t1\checkpoints\chair_hybrid_start_total5000.pt')] = 'd640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
    $protected[(Join-Path $weightRoot 'A_FM_attachment1000_total6000.pt')] = '27e5076eda8763a99b7973c9a637ff01f9f466ae94fbae123d2b945c3307f7eb'
    foreach ($path in @(git -C $workspaceRoot ls-files native_t1 models)) {
        $p = Join-Path $workspaceRoot $path
        if (Test-Path -LiteralPath $p -PathType Leaf) { $protected[$p] = Hash $p }
    }
    foreach ($file in @(Get-ChildItem -LiteralPath (Join-Path $boundaryRoot 'outputs') -Recurse -File -Filter raw.npz)) {
        $protected[$file.FullName] = Hash $file.FullName
    }
    $protectedRows = @()
    foreach ($path in @($protected.Keys | Sort-Object)) {
        $actual = Hash $path
        if ($actual -ne $protected[$path]) { throw "Protected original differs before cleanup: $path" }
        $protectedRows += [pscustomobject]@{path=$path; sha256=$actual; bytes=(Get-Item -LiteralPath $path).Length}
    }
    Write-Json (Join-Path $recordRoot 'protected_before.json') $protectedRows
    $removeRows = @($archiveRows) + @($weights) + @($cache)
    $plan = @{status='PREPARED'; workspace=$workspaceRoot; tree_targets=@($codeRoot,$runRoot,$cacheRoot); weight_targets=@($weights.path); files=$removeRows;
        file_count=$removeRows.Count; bytes=($removeRows | Measure-Object bytes -Sum).Sum; weights=$weights.Count; weight_bytes=($weights | Measure-Object bytes -Sum).Sum;
        archive=$archivePath; archive_sha256=(Hash $archivePath); protected_files=$protectedRows.Count;
        retained_A_control='A_FM_attachment1000_total6000.pt'; START_and_original_RAW_preserved=$true; model_loads=0; model_forwards=0; training=0}
    Write-Json $planPath $plan
    $plan | Select-Object status,file_count,bytes,weights,weight_bytes,protected_files | ConvertTo-Json
}

if ($Phase -eq 'Delete') {
    $plan = Read-Json $planPath
    if ($plan.status -ne 'PREPARED' -or $plan.weights -ne 19) { throw 'Wrong deletion manifest' }
    if ((Hash $archivePath) -ne $plan.archive_sha256) { throw 'Verified archive changed' }
    Assert-Archive (Read-Json (Join-Path $recordRoot 'archive_manifest.json')).files
    $actualTrees = @($plan.tree_targets)
    if (($actualTrees -join '|') -ne (@($codeRoot,$runRoot,$cacheRoot) -join '|')) { throw 'Unexpected recursive targets' }
    foreach ($target in $actualTrees) {
        $resolved = (Resolve-Path -LiteralPath (Assert-WorkspacePath $target)).Path
        if ($resolved -ne $target) { throw "Resolved target changed: $target" }
        $expected = @($plan.files | Where-Object { $_.path.StartsWith($target+'\',[StringComparison]::OrdinalIgnoreCase) })
        $current = @(Get-ChildItem -LiteralPath $target -Recurse -File -Force)
        if ($expected.Count -ne $current.Count) { throw "File count changed before removal: $target" }
        $index = @{}; foreach ($r in $expected) { $index[$r.path]=$r }
        foreach ($file in $current) {
            Assert-WorkspacePath $file.FullName | Out-Null
            if (-not $index.ContainsKey($file.FullName) -or $file.Length -ne $index[$file.FullName].bytes) { throw 'Unregistered file in deletion target' }
            if ((Hash $file.FullName) -ne $index[$file.FullName].sha256) { throw 'Deletion source changed' }
        }
    }
    foreach ($path in $plan.weight_targets) {
        Assert-WorkspacePath $path | Out-Null
        if ([IO.Path]::GetDirectoryName($path) -ne $weightRoot -or [IO.Path]::GetFileName($path) -eq 'A_FM_attachment1000_total6000.pt') { throw 'Invalid weight target' }
        $row = @($plan.files | Where-Object path -eq $path)[0]
        if ((Get-Item -LiteralPath $path).Length -ne $row.bytes) { throw 'Weight changed before delete' }
    }
    foreach ($path in $plan.weight_targets) {
        Remove-Item -LiteralPath $path -Force
        [IO.File]::AppendAllText((Join-Path $recordRoot 'operations.jsonl'),((@{event='DELETED_WEIGHT'; path=$path; utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress)+[Environment]::NewLine),$utf8)
    }
    foreach ($target in $actualTrees) {
        # Full absolute target and every descendant were verified above using the same shell.
        Remove-Item -LiteralPath $target -Recurse -Force
        [IO.File]::AppendAllText((Join-Path $recordRoot 'operations.jsonl'),((@{event='DELETED_TREE'; path=$target; utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress)+[Environment]::NewLine),$utf8)
    }
    Write-Json (Join-Path $recordRoot 'deletion_complete.json') @{status='DELETED'; files=$plan.file_count; logical_bytes=$plan.bytes; weights=19; archive_retained=$true; utc=[DateTime]::UtcNow.ToString('o')}
    Write-Output 'DELETE_COMPLETE'
}

if ($Phase -eq 'Verify') {
    $plan = Read-Json $planPath
    $rows = @()
    $protectedBefore = Read-Json (Join-Path $recordRoot 'protected_before.json')
    foreach ($before in $protectedBefore) {
        $after = Hash $before.path
        $rows += [pscustomobject]@{path=$before.path; before_sha256=$before.sha256; after_sha256=$after; unchanged=($before.sha256 -eq $after)}
    }
    $remaining = @($plan.files | Where-Object { Test-Path -LiteralPath $_.path })
    $changed = @($rows | Where-Object { -not $_.unchanged })
    if ($remaining.Count -gt 0 -or $changed.Count -gt 0) { throw 'Post-cleanup verification failed' }
    if ((Hash $archivePath) -ne $plan.archive_sha256) { throw 'Archive changed' }
    Assert-Archive (Read-Json (Join-Path $recordRoot 'archive_manifest.json')).files
    $summary = @{status='PASS'; deleted_files=$plan.file_count; deleted_logical_bytes=$plan.bytes; removed_weights=$plan.weights; weight_bytes=$plan.weight_bytes;
        protected_files=$rows.Count; protected_changed=$changed.Count; deletion_targets_remaining=$remaining.Count; archive_bytes=(Get-Item -LiteralPath $archivePath).Length;
        archive_sha256=$plan.archive_sha256; model_loads=0; model_forwards=0; new_generations=0; training=0; git_commit=0; git_push=0}
    Write-Json (Join-Path $recordRoot 'protected_after.json') $rows
    Write-Json (Join-Path $recordRoot 'verification.json') $summary
    $summary | ConvertTo-Json
}
