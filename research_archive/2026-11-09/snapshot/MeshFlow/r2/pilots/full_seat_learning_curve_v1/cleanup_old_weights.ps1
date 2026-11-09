$ErrorActionPreference='Stop'
$folder='E:\MeshFlow\experiments\r2_full_seat_learning_curve_v1\cleanup'
$plan=Get-Content -LiteralPath ($folder+'\plan.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$receiptPath=$folder+'\completion.json'
if(Test-Path -LiteralPath $receiptPath){throw 'Cleanup already recorded; do not repeat'}
$roots=@('E:\MeshFlow','E:\MeshFlow-Control-artifacts\experiments')
$removePaths=@{}
foreach($entry in $plan.remove){
 $resolved=(Resolve-Path -LiteralPath $entry.path).Path
 $absolute=[IO.Path]::GetFullPath($resolved)
 $inside=$false
 foreach($root in $roots){if($absolute.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)){$inside=$true}}
 if(-not $inside -or $absolute -match '\\.worktrees\\' -or [IO.Path]::GetExtension($absolute) -notin @('.pt','.pth','.ckpt','.safetensors')){throw ('Target outside identified weight scope: '+$absolute)}
 if(-not [String]::Equals($absolute,[IO.Path]::GetFullPath($entry.path),[StringComparison]::OrdinalIgnoreCase)){throw 'Resolved target changed'}
 $item=Get-Item -LiteralPath $absolute
 if($item.PSIsContainer -or $item.Length -ne [long]$entry.bytes){throw ('Target size/type changed: '+$absolute)}
 $removePaths[$absolute]=$true
}
foreach($entry in $plan.retained_large_weights){if($removePaths.ContainsKey($entry.path)){throw 'Retention/deletion overlap'};if(-not(Test-Path -LiteralPath $entry.path -PathType Leaf)){throw ('Required retained weight missing: '+$entry.path)}}
$freeBefore=(Get-PSDrive -Name E).Free
$started=[DateTime]::UtcNow.ToString('o')
$deleted=@()
foreach($entry in $plan.remove){
 $p=[IO.Path]::GetFullPath($entry.path)
 $hash=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()
 $before=[ordered]@{event='DELETE_WEIGHT_BEGIN';utc=[DateTime]::UtcNow.ToString('o');path=$p;bytes=[long]$entry.bytes;sha256=$hash;reason=$entry.reason;link_type=$entry.link_type;link_target=$entry.link_target}
 [IO.File]::AppendAllText($folder+'\deletion_ledger.jsonl',($before|ConvertTo-Json -Compress)+"`n",[Text.UTF8Encoding]::new($false))
 Remove-Item -LiteralPath $p -Force
 if(Test-Path -LiteralPath $p){throw ('Deletion did not complete: '+$p)}
 $before.event='DELETE_WEIGHT_COMPLETE';$before.utc=[DateTime]::UtcNow.ToString('o')
 [IO.File]::AppendAllText($folder+'\deletion_ledger.jsonl',($before|ConvertTo-Json -Compress)+"`n",[Text.UTF8Encoding]::new($false))
 $deleted+=New-Object PSObject -Property $before
 Write-Output ('Removed {0}/66: {1}' -f $deleted.Count,[IO.Path]::GetFileName($p))
}
$retained=@()
foreach($entry in $plan.retained_large_weights){
 $item=Get-Item -LiteralPath $entry.path
 if($item.Length -ne [long]$entry.bytes){throw ('Retained file size changed: '+$entry.path)}
 $hash=(Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant()
 $retained+=[PSCustomObject]@{path=$entry.path;bytes=$item.Length;sha256=$hash;status='PRESENT'}
}
$deleted | Export-Csv -LiteralPath ($folder+'\deleted_weights.csv') -NoTypeInformation -Encoding UTF8
$retained | Export-Csv -LiteralPath ($folder+'\retained_verified.csv') -NoTypeInformation -Encoding UTF8
$freeAfter=(Get-PSDrive -Name E).Free
$receipt=[ordered]@{status='COMPLETE';started_utc=$started;completed_utc=[DateTime]::UtcNow.ToString('o');deleted_files=$deleted.Count;deleted_logical_bytes=($deleted|Measure-Object -Property bytes -Sum).Sum;device_free_before=$freeBefore;device_free_after=$freeAfter;observed_free_delta_bytes=($freeAfter-$freeBefore);retained_large_files=$retained.Count;plan_sha256=(Get-FileHash -LiteralPath ($folder+'\plan.json') -Algorithm SHA256).Hash.ToLowerInvariant();deleted_manifest_sha256=(Get-FileHash -LiteralPath ($folder+'\deleted_weights.csv') -Algorithm SHA256).Hash.ToLowerInvariant();retained_manifest_sha256=(Get-FileHash -LiteralPath ($folder+'\retained_verified.csv') -Algorithm SHA256).Hash.ToLowerInvariant();training_started=$false;raw_logs_reports_data_deleted=$false;note='Only explicit old individual weight allowlist deleted. Logical size includes hardlinked aliases. Free-space delta can include unrelated user disk activity. Old evidence/resume metadata retained; removed historical branches no longer resumable.'}
[IO.File]::WriteAllText($receiptPath,($receipt|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
$receipt | ConvertTo-Json -Depth 6