$ErrorActionPreference = 'Stop'
$workspacePath = [IO.Path]::GetFullPath('E:\MeshFlow')
$auditPath = [IO.Path]::GetFullPath($PSScriptRoot)
if (-not $auditPath.StartsWith($workspacePath + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Audit path outside workspace' }
$manifestPath = Join-Path $auditPath 'cache_deletion_candidates.json'
$candidates = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$receiptPath = Join-Path $auditPath 'deletion_receipts.jsonl'
if (Test-Path -LiteralPath $receiptPath) { throw 'Deletion receipt already exists; do not silently repeat cleanup' }

# Validate every literal target, hash and type BEFORE the first deletion.
$verifiedPaths = @()
foreach ($item in $candidates) {
    $targetPath = (Resolve-Path -LiteralPath $item.path).Path
    if (-not $targetPath.StartsWith($workspacePath + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Target outside workspace' }
    if ($targetPath -notmatch '\\__pycache__\\[^\\]+\.pyc$') { throw 'Unexpected cache type' }
    if ($targetPath -match '\\\.(git|worktrees|codex|agents)\\') { throw 'Protected repository area' }
    $entry = Get-Item -LiteralPath $targetPath -Force
    if ($entry.PSIsContainer -or (([int]$entry.Attributes -band [int][IO.FileAttributes]::ReparsePoint) -ne 0)) { throw 'Expected ordinary cache file' }
    if ($entry.Length -ne $item.bytes) { throw 'Cache size changed' }
    $currentHash = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($currentHash -ne $item.sha256) { throw 'Cache hash changed since inventory' }
    $verifiedPaths += [PSCustomObject]@{path=$targetPath; bytes=$item.bytes; sha256=$currentHash}
}

foreach ($item in $verifiedPaths) {
    Remove-Item -LiteralPath $item.path -Force
    if (Test-Path -LiteralPath $item.path) { throw 'Cache deletion did not complete' }
    $receipt = [ordered]@{time=(Get-Date -Format o); path=$item.path; bytes=$item.bytes; sha256=$item.sha256; status='DELETED_REBUILDABLE_PYC'}
    ($receipt | ConvertTo-Json -Compress) | Add-Content -LiteralPath $receiptPath -Encoding UTF8
}

# Only remove explicitly checked empty cache directories; no recursive deletion.
$parents = @($verifiedPaths | ForEach-Object { Split-Path -LiteralPath $_.path } | Sort-Object -Unique)
foreach ($dirPath in $parents) {
    $resolvedDir = (Resolve-Path -LiteralPath $dirPath).Path
    if (-not $resolvedDir.StartsWith($workspacePath + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Directory outside workspace' }
    if ((Split-Path -Leaf $resolvedDir) -ne '__pycache__') { throw 'Unexpected parent' }
    if (@(Get-ChildItem -LiteralPath $resolvedDir -Force).Count -eq 0) { Remove-Item -LiteralPath $resolvedDir -Force }
}
[PSCustomObject]@{deleted_files=$verifiedPaths.Count; deleted_bytes=($verifiedPaths | Measure-Object -Property bytes -Sum).Sum; recursive_deletions=0} | ConvertTo-Json
