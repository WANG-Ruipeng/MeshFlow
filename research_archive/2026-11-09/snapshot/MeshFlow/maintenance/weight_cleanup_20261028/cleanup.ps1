param([ValidateSet('Prepare','Apply','Verify')][string]$Stage='Prepare')
$ErrorActionPreference='Stop'
$root='E:\MeshFlow'
$out=Join-Path $root 'maintenance\weight_cleanup_20261028'
$utf8=[Text.UTF8Encoding]::new($false)
function ReadJson([string]$p){Get-Content -LiteralPath $p -Raw -Encoding UTF8|ConvertFrom-Json}
function SaveJson([string]$p,$v){[IO.File]::WriteAllText($p,($v|ConvertTo-Json -Depth 25),$utf8)}
function Hash([string]$p){(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
function Safe([string]$p){
 $full=[IO.Path]::GetFullPath($p)
 if(-not $full.StartsWith($root+'\experiments\',[StringComparison]::OrdinalIgnoreCase)){throw "Outside experiment root: $full"}
 $item=Get-Item -LiteralPath $full -Force
 if($item -isnot [IO.FileInfo]){throw "Not a regular file: $full"}
 $cur=$item
 while($cur){
  if(($cur.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw "Reparse path refused: $full"}
  if($cur.FullName.TrimEnd('\') -eq $root){break}
  $cur=if($cur -is [IO.FileInfo]){$cur.Directory}else{$cur.Parent}
 }
 if(-not $cur){throw "Unexpected ancestry: $full"}
 $full
}
function FileRecord([string]$p,[string]$kind){
 $p=Safe $p;$f=Get-Item -LiteralPath $p
 [pscustomobject]@{Path=$p;Bytes=$f.Length;LastWriteUTC=$f.LastWriteTimeUtc.ToString('o');SHA256=(Hash $p);Kind=$kind}
}
$branches=@(
 [pscustomobject]@{Dir='experiments\r2_bench_exposure_allocation_v1\training\A_50_50';End=1500},
 [pscustomobject]@{Dir='experiments\r2_bench_exposure_allocation_v1\training\B_75_25';End=1500},
 [pscustomobject]@{Dir='experiments\r2_bench_natural_n_recipe_v1\training\CONTROL_OLD';End=1500},
 [pscustomobject]@{Dir='experiments\r2_bench_natural_n_recipe_v1\training\NATURAL_HIGH';End=1500},
 [pscustomobject]@{Dir='experiments\r2_bench_high_joint_coverage_v1\training\WIDE_HIGH';End=1500},
 [pscustomobject]@{Dir='experiments\r2_bench_group_weight_match_v1\training\W_MATCHED';End=1500},
 [pscustomobject]@{Dir='experiments\r2_bench_four_routes_v1\exposure\training\M_PLUS3000';End=3000},
 [pscustomobject]@{Dir='experiments\r2_bench_four_routes_v1\seam\training\SEAM1500';End=1500}
)
$allowed=@($branches|ForEach-Object{Join-Path (Join-Path $root $_.Dir) 'checkpoint_latest.pt'})
if($Stage -eq 'Prepare'){
 if(Test-Path -LiteralPath (Join-Path $out 'prepared_manifest.json')){throw 'Already prepared'}
 $old=ReadJson (Join-Path $root 'maintenance\storage_cleanup_20261025\effective_retained_assets.json')
 foreach($p in $allowed){if(@($old|Where-Object {$_.Path -eq $p}).Count){throw "Previously protected file: $p"}}
 $remove=[Collections.Generic.List[object]]::new();$keep=[Collections.Generic.List[object]]::new();$evidence=@{}
 foreach($b in $branches){
  $dir=Join-Path $root $b.Dir
  $state=ReadJson (Join-Path $dir 'status.json')
  if($state.status -ne 'COMPLETE'){throw "Training not complete: $dir"}
  if(@($state.calls.pending_call_ids).Count -ne 0){throw "Pending scientific call: $dir"}
  $mid=ReadJson (Join-Path $dir 'checkpoint_latest.json')
  $end=ReadJson (Join-Path $dir ('checkpoint_step'+$b.End+'.json'))
  $gen=ReadJson (Join-Path $dir ('generator_step'+$b.End+'.json'))
  if($mid.local_step -ne ($b.End-250) -or $end.local_step -ne $b.End -or $gen.local_step -ne $b.End){throw 'Unexpected step identity'}
  if($end.generator_state_sha256 -ne $gen.generator_state_sha256 -or -not $end.optimizer_state_sha256 -or -not $end.rng_sha256){throw 'Incomplete final recovery identity'}
  $r=FileRecord (Join-Path $dir 'checkpoint_latest.pt') 'RETIRED_INTERMEDIATE_NOT_DUPLICATE'
  if($r.SHA256 -ne $mid.sha256 -or $r.Bytes -ne $mid.bytes){throw 'Intermediate hash/size mismatch'}
  $r|Add-Member LocalStep $mid.local_step
  $r|Add-Member RetainedFinal (Join-Path $dir ('checkpoint_step'+$b.End+'.pt'))
  $r|Add-Member Reason 'Completed run; obsolete rolling recovery point 250 updates before the retained final checkpoint'
  $remove.Add($r)
  foreach($type in @('checkpoint','generator')){
   $meta=if($type -eq 'checkpoint'){$end}else{$gen}
   $kr=FileRecord (Join-Path $dir ($type+'_step'+$b.End+'.pt')) 'RETAIN_FINAL'
   if($kr.SHA256 -ne $meta.sha256 -or $kr.Bytes -ne $meta.bytes){throw 'Retained endpoint hash/size mismatch'}
   $keep.Add($kr)
  }
  foreach($f in Get-ChildItem -LiteralPath $dir -File){if($f.Extension -ne '.pt'){$evidence[$f.FullName]=Hash $f.FullName}}
  if($dir -match '^(E:\\MeshFlow\\experiments\\[^\\]+)'){
   $er=$matches[1]
   foreach($f in Get-ChildItem -LiteralPath $er -File){if($f.Extension -in @('.md','.txt','.json','.csv','.diff')){$evidence[$f.FullName]=Hash $f.FullName}}
  }
  Write-Output ('Verified intermediate and both retained endpoints: '+$b.Dir)
 }
 $inventory=ReadJson (Join-Path $root 'maintenance\weight_inventory_20261028_initial.json')
 $other=@($inventory|Where-Object {$_.Path -notin $allowed})
 foreach($f in $other){
  $live=Get-Item -LiteralPath $f.Path
  if($live.Length -ne $f.Bytes -or $live.LastWriteTimeUtc.ToString('o') -ne $f.Modified){throw 'Other weight changed during inventory'}
 }
 $manifest=[ordered]@{Status='PREPARED';UTC=[DateTime]::UtcNow.ToString('o');Root=$root;
  Authorization='User 2026-10-28: 清理一下一些不用的权重';Scope='Only eight explicitly listed obsolete rolling checkpoints; all final model and recovery states retained';
  Remove=@($remove);KeepHashVerified=@($keep);OtherWeightsStatProtected=$other;EvidenceHashes=$evidence;
  DeleteFiles=$remove.Count;DeleteBytes=($remove|Measure-Object Bytes -Sum).Sum;FreeBytesBefore=[IO.DriveInfo]::new('E:\').AvailableFreeSpace;
  NoModelLoad=$true;NoDatasetOrRawPostDeletion=$true;NoDefaultChange=$true;NoCommitOrPush=$true}
 SaveJson (Join-Path $out 'prepared_manifest.json') $manifest
 [IO.File]::WriteAllText((Join-Path $out 'prepared_manifest.sha256'),(Hash (Join-Path $out 'prepared_manifest.json')),$utf8)
 $remove|Export-Csv -LiteralPath (Join-Path $out 'deletion_plan.csv') -NoTypeInformation -Encoding UTF8
 $keep|Export-Csv -LiteralPath (Join-Path $out 'retained_endpoints.csv') -NoTypeInformation -Encoding UTF8
 [pscustomobject]@{Status='PREPARED';Files=$remove.Count;Bytes=$manifest.DeleteBytes;GiB=$manifest.DeleteBytes/1GB;HashVerifiedFinalFiles=$keep.Count}|ConvertTo-Json
 exit
}
$manifestPath=Join-Path $out 'prepared_manifest.json'
if((Hash $manifestPath) -ne (Get-Content -LiteralPath (Join-Path $out 'prepared_manifest.sha256') -Raw).Trim()){throw 'Changed manifest'}
$m=ReadJson $manifestPath
if($m.Remove.Count -ne 8){throw 'Unexpected removal count'}
foreach($r in $m.Remove){if($r.Path -notin $allowed -or [IO.Path]::GetFileName($r.Path) -ne 'checkpoint_latest.pt'){throw 'Target not explicitly authorized'}}
if($Stage -eq 'Apply'){
 if(Test-Path -LiteralPath (Join-Path $out 'cleanup_receipt.json')){throw 'Already applied; use Verify'}
 $prior=@()
 $ledgerPath=Join-Path $out 'deletion_ledger.jsonl'
 if(Test-Path -LiteralPath $ledgerPath){$prior=@(Get-Content -LiteralPath $ledgerPath -Encoding UTF8|ForEach-Object{$_|ConvertFrom-Json})}
 foreach($r in $m.KeepHashVerified){$p=Safe $r.Path;if((Hash $p) -ne $r.SHA256){throw 'Retained final changed'}}
 foreach($p in $m.EvidenceHashes.PSObject.Properties){if((Hash $p.Name) -ne $p.Value){throw ('Evidence changed: '+$p.Name)}}
 foreach($r in $m.Remove){
  if(-not (Test-Path -LiteralPath $r.Path)){
   if(@($prior|Where-Object {$_.Path -eq $r.Path -and $_.Status -eq 'DELETED'}).Count -ne 1){throw 'Missing file without deletion receipt'}
   continue
  }
  $p=Safe $r.Path;$f=Get-Item -LiteralPath $p
  if($f.Length -ne $r.Bytes -or $f.LastWriteTimeUtc.ToString('o') -ne $r.LastWriteUTC -or (Hash $p) -ne $r.SHA256){throw 'Deletion target changed'}
  # Native PowerShell, exact file only; never recursive and no cross-shell deletion.
  Remove-Item -LiteralPath $p -Force -ErrorAction Stop
  if(Test-Path -LiteralPath $p){throw 'Deletion failed'}
  $row=[ordered]@{Status='DELETED';UTC=[DateTime]::UtcNow.ToString('o');Path=$p;Bytes=$r.Bytes;SHA256=$r.SHA256;LocalStep=$r.LocalStep;RetainedFinal=$r.RetainedFinal;Reason=$r.Reason}
  [IO.File]::AppendAllText($ledgerPath,(($row|ConvertTo-Json -Compress)+[Environment]::NewLine),$utf8)
  Write-Output ('Deleted obsolete intermediate: '+$p)
 }
 foreach($b in $branches){
  $dir=Join-Path $root $b.Dir;$notice=Join-Path $dir 'INTERMEDIATE_RETIRED_20261028.txt'
  $text="2026-10-28 user-authorized weight cleanup.`r`nOnly checkpoint_latest.pt (local step $($b.End-250)) was retired after this run completed.`r`nRetained checkpoint_step$($b.End).pt includes the matching optimizer/RNG; generator_step$($b.End).pt is also retained.`r`nOld checkpoint_latest.json and historical receipts remain immutable evidence of the retired asset.`r`nDo not automatically resume from the removed intermediate or replay training to recreate it. Use the explicit retained final endpoint under new authorization.`r`nCleanup ledger: E:\MeshFlow\maintenance\weight_cleanup_20261028\deletion_ledger.jsonl`r`n"
  if(-not (Test-Path -LiteralPath $notice)){[IO.File]::WriteAllText($notice,$text,$utf8)}
 }
}
foreach($r in $m.Remove){if(Test-Path -LiteralPath $r.Path){throw 'Removal incomplete'}}
foreach($r in $m.KeepHashVerified){if((Hash $r.Path) -ne $r.SHA256){throw 'Retained final hash mismatch after cleanup'}}
foreach($p in $m.EvidenceHashes.PSObject.Properties){if((Hash $p.Name) -ne $p.Value){throw 'Evidence hash mismatch after cleanup'}}
foreach($f in $m.OtherWeightsStatProtected){$live=Get-Item -LiteralPath $f.Path;if($live.Length -ne $f.Bytes -or $live.LastWriteTimeUtc.ToString('o') -ne $f.Modified){throw 'Other weight altered'}}
$ledger=@(Get-Content -LiteralPath (Join-Path $out 'deletion_ledger.jsonl') -Encoding UTF8|ForEach-Object{$_|ConvertFrom-Json})
if($ledger.Count -ne 8 -or @($ledger.Path|Sort-Object -Unique).Count -ne 8){throw 'Unexpected deletion ledger'}
$free=[IO.DriveInfo]::new('E:\').AvailableFreeSpace
$receipt=[ordered]@{Status='COMPLETE';UTC=[DateTime]::UtcNow.ToString('o');DeletedFiles=8;DeletedLogicalBytes=$m.DeleteBytes;DeletedGiB=$m.DeleteBytes/1GB;
 FreeBytesBefore=$m.FreeBytesBefore;FreeBytesAfter=$free;ObservedFreeDelta=$free-$m.FreeBytesBefore;
 RetainedFinalHashChecks=$m.KeepHashVerified.Count;OtherWeightsStatChecks=$m.OtherWeightsStatProtected.Count;
 EvidenceHashChecks=@($m.EvidenceHashes.PSObject.Properties).Count;ManifestSHA256=(Hash $manifestPath);LedgerSHA256=(Hash (Join-Path $out 'deletion_ledger.jsonl'));
 Caveat='Drive free-space delta may include unrelated writes; logical deletion count/bytes and exact paths are authoritative';
 NoModelLoads=$true;NoTraining=$true;NoDatasetsOrRawPostDeleted=$true;NoFinalCheckpointDeleted=$true;NoDefaultChange=$true;NoCommitOrPush=$true}
if($Stage -eq 'Apply'){SaveJson (Join-Path $out 'cleanup_receipt.json') $receipt}
else{SaveJson (Join-Path $out 'verification_receipt.json') $receipt}
$receipt|ConvertTo-Json -Depth 3
