param([ValidateSet('Prepare','Execute')][string]$Stage='Prepare')
$ErrorActionPreference='Stop'
$root='E:\MeshFlow'
$out=Join-Path $root 'maintenance\storage_cleanup_20261022'
$dirs=@(
 'experiments\r2_bench_state_schedule_v1\training\PERTURBED_STATE',
 'experiments\r2_bench_state_schedule_v1\training\ASYNC_LEAD_SEAM',
 'experiments\r2_bench_coverage_expansion_v1\training\COVERAGE_SEAM',
 'experiments\r2_bench_user_v2_training_v1\training\USER_V2_SEAM',
 'experiments\r2_bench_early_fm_v1\training\EARLY_FM')
function Read-Json([string]$p){Get-Content -LiteralPath $p -Encoding UTF8 -Raw | ConvertFrom-Json}
function Safe-Path([string]$p){
 $full=[System.IO.Path]::GetFullPath($p)
 if(-not $full.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Outside intended root: $full"}
 $cur=Get-Item -LiteralPath $full -Force
 while($cur.FullName -ne $root){
  if(($cur.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0){throw "Reparse path refused: $full"}
  if($cur -is [System.IO.FileInfo]){$cur=$cur.Directory}else{$cur=$cur.Parent}
  if($null -eq $cur){throw "Unexpected path ancestry: $full"}
 }
 return $full
}
function Hash([string]$p){(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
if($Stage -eq 'Prepare'){
 if(Test-Path -LiteralPath ($out+'\prepared_manifest.json')){throw 'Prepared manifest already exists'}
 $remove=[System.Collections.Generic.List[object]]::new();$keep=@{};$metadata=[System.Collections.Generic.List[object]]::new()
 foreach($relative in $dirs){
  $d=Join-Path $root $relative;$status=Read-Json ($d+'\status.json')
  if($status.status -ne 'COMPLETE' -or $status.local_step -ne 1500){throw "Not complete: $d"}
  $names=@('rolling.pt')
  if($relative -match '(COVERAGE_SEAM|USER_V2_SEAM)$'){$names+=@('checkpoint_step500.pt')}
  foreach($name in $names){
   $p=Safe-Path ($d+'\'+$name);$info=Get-Item -LiteralPath $p
   $side=Read-Json ([System.IO.Path]::ChangeExtension($p,'.json'))
   $expectedStep=if($name -eq 'rolling.pt'){1250}else{500}
   if($side.local_step -ne $expectedStep){throw "Wrong intermediate step: $p"}
   $h=Hash $p;if($h -ne $side.sha256){throw "Sidecar hash mismatch: $p"}
   $remove.Add([pscustomobject]@{Path=$p;Bytes=$info.Length;SHA256=$h;Reason='Completed branch intermediate recovery point';LocalStep=$expectedStep})
  }
  foreach($name in @('checkpoint_step0.pt','checkpoint_step1500.pt','generator_step1500.pt')){
   $p=$d+'\'+$name;if(Test-Path -LiteralPath $p){$keep[$p]=Hash $p}
  }
  $experiment=Split-Path (Split-Path $d -Parent) -Parent
  foreach($f in Get-ChildItem -LiteralPath $experiment -Recurse -File){
   if($f.Extension -ne '.pt'){$metadata.Add([pscustomobject]@{Path=$f.FullName;Bytes=$f.Length;LastWriteUTC=$f.LastWriteTimeUtc.ToString('o')})}
  }
 }
 $review=Read-Json ($out+'\cache_review.json')
 foreach($candidate in $review.candidates){
  $croot=Safe-Path $candidate.path
  $allowedCacheRoots=@(($root+'\deliverables\R2_32_67_ANALYSIS_MINI_V1\_work\unzip_validation'),($root+'\r2\build'))
  if($croot -notin $allowedCacheRoots){throw 'Unexpected cache root'}
  $files=@(Get-ChildItem -LiteralPath $croot -Recurse -File)
  if($files.Count -ne $candidate.count){throw 'Cache file inventory changed'}
  foreach($f in $files){
   $p=Safe-Path $f.FullName;$h=Hash $p
   if($croot -like '*unzip_validation'){
    $match=@($candidate.files | Where-Object {$_.cache_path -eq $p})
    if($match.Count -ne 1 -or $h -ne $match[0].cache_sha256 -or (Hash $match[0].retained_path) -ne $h){throw 'Duplicate validation failed'}
    $keep[$match[0].retained_path]=$h
   }
   $remove.Add([pscustomobject]@{Path=$p;Bytes=$f.Length;SHA256=$h;Reason=$candidate.reason;LocalStep=$null})
  }
 }
 $metadata | Export-Csv -LiteralPath ($out+'\nonweight_assets_before.csv') -NoTypeInformation -Encoding UTF8
 $sum=[long]0;foreach($r in $remove){$sum+=$r.Bytes}
 [ordered]@{Schema='EXACT_ALLOWLIST_STORAGE_CLEANUP_V1';Root=$root;PreparedUTC=[DateTime]::UtcNow.ToString('o');Remove=@($remove);KeepHashes=$keep;DeleteBytes=$sum;DeleteFiles=$remove.Count;CacheRoots=@($review.candidates.path);NoOfficialOrFinalWeights=$true;NonweightInventorySHA256=(Hash ($out+'\nonweight_assets_before.csv'))} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath ($out+'\prepared_manifest.json') -Encoding UTF8
 [pscustomobject]@{Status='PREPARED';Files=$remove.Count;Bytes=$sum;GiB=$sum/1GB;ProtectedHashes=$keep.Count;NonWeightAssets=$metadata.Count} | ConvertTo-Json
 exit
}
$plan=Read-Json ($out+'\prepared_manifest.json')
if(Test-Path -LiteralPath ($out+'\cleanup_receipt.json')){throw 'Already executed; do not delete again'}
foreach($r in $plan.Remove){$p=Safe-Path $r.Path;if((Get-Item -LiteralPath $p).Length -ne $r.Bytes -or (Hash $p) -ne $r.SHA256){throw "Planned file changed: $p"}}
foreach($p in $plan.KeepHashes.PSObject.Properties){if((Hash $p.Name) -ne $p.Value){throw "Protected changed: $($p.Name)"}}
$deleted=[System.Collections.Generic.List[object]]::new()
try{
 foreach($r in $plan.Remove){
  $p=Safe-Path $r.Path
  Remove-Item -LiteralPath $p -Force
  if(Test-Path -LiteralPath $p){throw "Deletion failed: $p"}
  $deleted.Add($r)
  ([ordered]@{UTC=[DateTime]::UtcNow.ToString('o');Path=$p;Bytes=$r.Bytes;SHA256=$r.SHA256}|ConvertTo-Json -Compress) | Add-Content -LiteralPath ($out+'\deletion_ledger.jsonl') -Encoding UTF8
 }
 # Remove only empty directories inside two already checked exact cache roots.
 foreach($c in $plan.CacheRoots){
  $cp=Safe-Path $c
  foreach($d in @(Get-ChildItem -LiteralPath $cp -Recurse -Directory | Sort-Object {$_.FullName.Length} -Descending)){
   $dp=Safe-Path $d.FullName
   if(@(Get-ChildItem -LiteralPath $dp -Force).Count -eq 0){Remove-Item -LiteralPath $dp -Force}
  }
  if(@(Get-ChildItem -LiteralPath $cp -Force).Count -eq 0){Remove-Item -LiteralPath $cp -Force}
 }
 foreach($p in $plan.KeepHashes.PSObject.Properties){if((Hash $p.Name) -ne $p.Value){throw "Protected changed after cleanup: $($p.Name)"}}
 foreach($r in Import-Csv -LiteralPath ($out+'\nonweight_assets_before.csv')){
  $f=Get-Item -LiteralPath $r.Path
  if($f.Length -ne [long]$r.Bytes -or $f.LastWriteTimeUtc.ToString('o') -ne $r.LastWriteUTC){throw "Nonweight asset changed: $($r.Path)"}
 }
 $state='COMPLETE'
}catch{$state='PARTIAL_FAILURE';$failure=$_.Exception.Message;throw}
finally{
 $sum=[long]0;foreach($r in $deleted){$sum+=$r.Bytes}
 [ordered]@{Status=$state;FinishedUTC=[DateTime]::UtcNow.ToString('o');DeletedFiles=$deleted.Count;DeletedBytes=$sum;LogicalGiB=$sum/1GB;ProtectedSHA256Checked=@($plan.KeepHashes.PSObject.Properties).Count;Error=$failure;PlanSHA256=(Hash ($out+'\prepared_manifest.json'));NoTrainingOrModelCalls=$true} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath ($out+'\cleanup_receipt.json') -Encoding UTF8
}
Get-Content -LiteralPath ($out+'\cleanup_receipt.json') -Encoding UTF8
