$ErrorActionPreference='Stop'
$workspaceRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$manifest=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$receipt=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup_receipt.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$violations=[System.Collections.Generic.List[string]]::new()
$checked=0
foreach($row in $manifest.original_nonweight_inventory){
  $p=Join-Path $workspaceRoot $row.path
  $f=[System.IO.FileInfo]::new($p)
  if(-not $f.Exists){$violations.Add('MISSING: '+$row.path);continue}
  if($f.Length -ne $row.bytes){$violations.Add('SIZE_CHANGED: '+$row.path)}
  $ns=[int64](($f.LastWriteTimeUtc.Ticks - [int64]621355968000000000)*100)
  if($ns -ne $row.mtime_ns){$violations.Add('MTIME_CHANGED: '+$row.path)}
  $checked++
}
foreach($row in $manifest.critical_files_hash_checked){
  $digest=(Get-FileHash -LiteralPath (Join-Path $workspaceRoot $row.path) -Algorithm SHA256).Hash.ToLowerInvariant()
  if($digest -ne $row.sha256){$violations.Add('CRITICAL_HASH_CHANGED: '+$row.path)}
}
foreach($row in $manifest.candidates){if(Test-Path -LiteralPath (Join-Path $workspaceRoot $row.path)){$violations.Add('DELETE_TARGET_STILL_EXISTS: '+$row.path)}}
foreach($row in $manifest.retained_hash_checked){
  $digest=(Get-FileHash -LiteralPath (Join-Path $workspaceRoot $row.path) -Algorithm SHA256).Hash.ToLowerInvariant()
  if($digest -ne $row.sha256){$violations.Add('RETAINED_WEIGHT_HASH_CHANGED: '+$row.path)}
}
$row=$manifest.official_denoiser_protected
$f=[System.IO.FileInfo]::new((Join-Path $workspaceRoot $row.path))
$ns=[int64](($f.LastWriteTimeUtc.Ticks - [int64]621355968000000000)*100)
if(-not $f.Exists -or $f.Length -ne $row.bytes -or $ns -ne $row.mtime_ns){$violations.Add('OFFICIAL_D0_STAT_CHANGED')}
$remaining=@(Get-ChildItem -LiteralPath (Join-Path $workspaceRoot 'experiments/r2_terminal_reward_nft_v1/checkpoints') -Recurse -File -Filter '*.pt')
if($remaining.Count -ne 2 -or @($remaining | Where-Object Name -ne 'round15.pt').Count -ne 0){$violations.Add('UNEXPECTED_REMAINING_WEIGHTS')}
$journal=@(Get-Content -LiteralPath (Join-Path $PSScriptRoot 'deletion_journal.jsonl') -Encoding UTF8)
if($journal.Count -ne 30 -or $receipt.deleted_files -ne 30){$violations.Add('DELETION_LEDGER_COUNT')}
$answer=[ordered]@{
 status=$(if($violations.Count -eq 0){'PASS'}else{'FAIL'});
 verified_utc=[DateTime]::UtcNow.ToString('o');
 original_nonweight_files_verified_exists_size_mtime=$checked;
 critical_original_files_sha256_unchanged=$manifest.critical_files_hash_checked.Count;
 retained_weights_sha256_unchanged=$manifest.retained_hash_checked.Count;
 official_D0_scope='exists/size/mtime; previous SHA recorded, not rehashed';
 deletion_whitelist_absence_verified=30; remaining_NFT_SELF_checkpoints=$remaining.Count;
 raw_mesh_rescore_or_model_calls=0; violations=@($violations);
 verification_scope='All pre-existing nonweight files by exists/size/mtime; reports/ZIP/execution source by SHA256. Does not claim all RAW bytes rehashed.'
}
[System.IO.File]::WriteAllText((Join-Path $PSScriptRoot 'verification.json'),($answer | ConvertTo-Json -Depth 8)+"`n",[System.Text.UTF8Encoding]::new($false))
[pscustomobject]$answer | ConvertTo-Json -Depth 8
if($violations.Count -ne 0){throw 'Cleanup verification failed; inspect preserved violations.'}