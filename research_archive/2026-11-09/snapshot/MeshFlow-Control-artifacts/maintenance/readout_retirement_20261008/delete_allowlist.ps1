$ErrorActionPreference = 'Stop'
$auditDir = 'E:\MeshFlow-Control-artifacts\maintenance\readout_retirement_20261008'
$prepared = Get-Content -LiteralPath (Join-Path $auditDir 'prepared.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($prepared.status -ne 'PREPARED') { throw 'Expected completed read-only preparation' }
$manifest = Get-Content -LiteralPath (Join-Path $auditDir 'delete_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.count -ne 8003) { throw 'Unexpected deletion scope' }
$weightRoot = [System.IO.Path]::GetFullPath('E:\MeshFlow-Control-artifacts\checkpoints')
$cacheRoot = [System.IO.Path]::GetFullPath('E:\MeshFlow-Control-artifacts\training_stream\ot_cache')
foreach ($dir in @('E:\MeshFlow-Control-artifacts',$weightRoot,'E:\MeshFlow-Control-artifacts\training_stream',$cacheRoot)) {
 $dirItem = Get-Item -LiteralPath $dir
 if ($dirItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Reparse directory refused: $dir" }
}
# Verify every resolved absolute target is exactly in one of the two authorized old directories.
foreach ($row in $manifest.files) {
 $absolute = [System.IO.Path]::GetFullPath([string]$row.windows_path)
 if ($absolute -ne $row.windows_path) { throw "Noncanonical target: $absolute" }
 $parent = [System.IO.Path]::GetDirectoryName($absolute)
 $name = [System.IO.Path]::GetFileName($absolute)
 $allowed = (($parent -eq $weightRoot) -and ($name -cmatch '^[ABC]_step1000\.pt$')) -or (($parent -eq $cacheRoot) -and ($name.EndsWith('.npz')))
 if (-not $allowed) { throw "Outside old READOUT allowlist: $absolute" }
 $entry = Get-Item -LiteralPath $absolute
 if ($entry.PSIsContainer -or ($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) { throw "Nonregular target: $absolute" }
 if ($entry.Length -ne $row.bytes) { throw "Target size changed: $absolute" }
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$journalPath = Join-Path $auditDir 'deletion.jsonl'
$beforeFree = (Get-PSDrive -Name E).Free
$count = 0
$total = [Int64]0
foreach ($row in $manifest.files) {
 $absolute = [System.IO.Path]::GetFullPath([string]$row.windows_path)
 $actualHash = (Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToLowerInvariant()
 if ($actualHash -ne $row.sha256) { throw "Target hash changed: $absolute" }
 $entry = Get-Item -LiteralPath $absolute
 if ($entry.PSIsContainer -or ($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) { throw "Target became nonregular: $absolute" }
 $intent = [ordered]@{event='DELETE_INTENT';path=$absolute;bytes=$row.bytes;sha256=$actualHash;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress
 [System.IO.File]::AppendAllText($journalPath,$intent+[Environment]::NewLine,$utf8)
 Remove-Item -LiteralPath $absolute -Force
 if (Test-Path -LiteralPath $absolute) { throw "Deletion did not complete: $absolute" }
 $done = [ordered]@{event='DELETED';path=$absolute;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress
 [System.IO.File]::AppendAllText($journalPath,$done+[Environment]::NewLine,$utf8)
 $count++
 $total += [Int64]$row.bytes
 if (($count % 1000) -eq 0) { Write-Output "Deleted $count / 8003 authorized files" }
}
$receipt = [ordered]@{status='DELETED_ALLOWLIST_ONLY';files=$count;logical_bytes=$total;volume_free_before=$beforeFree;volume_free_after=(Get-PSDrive -Name E).Free;recursive_deletion=$false;git_commit_or_push=$false;utc=[DateTime]::UtcNow.ToString('o')}
$receiptText = $receipt | ConvertTo-Json
[System.IO.File]::WriteAllText((Join-Path $auditDir 'deletion_receipt.json'),$receiptText,$utf8)
Write-Output $receiptText
