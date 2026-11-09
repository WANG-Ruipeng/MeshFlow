param([ValidateSet('Prepare','Execute','Verify')][string]$Stage='Prepare')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$archive=$PSScriptRoot
$artifactRoot='E:\MeshFlow-Control-artifacts'
$sourceRoot='E:\MeshFlow-Control'
$legacyRoot='E:\MeshFlow'
$utf8=[System.Text.UTF8Encoding]::new($false)
$targetPaths=@(
 'E:\MeshFlow-Control-artifacts\checkpoints\A_step0500.pt',
 'E:\MeshFlow-Control-artifacts\checkpoints\B_step0500.pt',
 'E:\MeshFlow-Control-artifacts\checkpoints\C_step0500.pt',
 'E:\MeshFlow-Control\.pytest_cache\.gitignore',
 'E:\MeshFlow-Control\.pytest_cache\CACHEDIR.TAG',
 'E:\MeshFlow-Control\.pytest_cache\README.md',
 'E:\MeshFlow-Control\.pytest_cache\v\cache\lastfailed',
 'E:\MeshFlow-Control\.pytest_cache\v\cache\nodeids'
)
function Write-Json($Name,$Value) {
 [IO.File]::WriteAllText((Join-Path $archive $Name),($Value | ConvertTo-Json -Depth 12),$utf8)
}
function Hash-File([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Assert-Target([string]$Path) {
 $full=[IO.Path]::GetFullPath($Path)
 if($full -cne $Path -or $targetPaths -cnotcontains $full) { throw "Target outside explicit allowlist: $Path" }
 $base=if($Path.StartsWith($artifactRoot+'\',[StringComparison]::OrdinalIgnoreCase)){$artifactRoot}else{$sourceRoot}
 if(-not $full.StartsWith($base+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Target escaped root: $Path"}
 $ancestor=$full
 while($ancestor.Length -ge $base.Length){
  if(Test-Path -LiteralPath $ancestor){
   $item=Get-Item -LiteralPath $ancestor -Force
   if(($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw "Reparse point forbidden in deletion path: $ancestor"}
  }
  if($ancestor -eq $base){break}
  $ancestor=[IO.Path]::GetDirectoryName($ancestor)
 }
}
function Protected-Paths {
 $paths=[Collections.Generic.List[string]]::new()
 Get-ChildItem -LiteralPath $artifactRoot -File -Force -Recurse | Where-Object {
  -not $_.FullName.StartsWith($artifactRoot+'\venv\',[StringComparison]::OrdinalIgnoreCase) -and $targetPaths -notcontains $_.FullName
 } | ForEach-Object {$paths.Add($_.FullName)}
 Get-ChildItem -LiteralPath $sourceRoot -File -Force -Recurse | Where-Object {
  -not $_.FullName.StartsWith($sourceRoot+'\.git\',[StringComparison]::OrdinalIgnoreCase) -and $targetPaths -notcontains $_.FullName
 } | ForEach-Object {$paths.Add($_.FullName)}
 Get-ChildItem -LiteralPath ($legacyRoot+'\native_t1\checkpoints') -File -Recurse | ForEach-Object {$paths.Add($_.FullName)}
 $paths.Add($legacyRoot+'\native_t1\runs\chair_hybrid_coupling_v1\checkpoints\OT_HYBRID_pilot1000_total5000.pt')
 $paths.Add($legacyRoot+'\native_t1\runs\chair_domain_control_v1\checkpoints\Geo_phase3000_total4000.pt')
 $paths.Add($legacyRoot+'\WORKSPACE.md')
 $paths.Add($legacyRoot+'\AGENTS.md')
 $paths | Sort-Object -Unique
}
if($Stage -eq 'Prepare'){
 if(Test-Path -LiteralPath (Join-Path $archive 'delete_manifest.csv')){throw 'Already prepared; refusing to replace original ledger'}
 $candidates=foreach($path in $targetPaths){
  Assert-Target $path
  $item=Get-Item -LiteralPath $path -Force
  [pscustomobject]@{path=$path;bytes=$item.Length;sha256=(Hash-File $path);reason=$(if($path.EndsWith('.pt')){'Completed-study secondary step500 checkpoint; step1000 retained'}else{'Regenerable pytest cache; actual test evidence retained'})}
 }
 $candidates | Export-Csv -LiteralPath (Join-Path $archive 'delete_manifest.csv') -NoTypeInformation -Encoding UTF8
 Write-Output ('Delete plan: '+$candidates.Count+' files; '+($candidates|Measure-Object bytes -Sum).Sum+' bytes')
 $paths=@(Protected-Paths)
 $protected=[Collections.Generic.List[object]]::new()
 $n=0
 foreach($path in $paths){
  $item=Get-Item -LiteralPath $path -Force
  $protected.Add([pscustomobject]@{path=$path;bytes=$item.Length;sha256=(Hash-File $path)})
  $n++
  if($n % 10000 -eq 0){Write-Output ('Preservation hashes: '+$n+'/'+$paths.Count)}
 }
 $protected | Export-Csv -LiteralPath (Join-Path $archive 'preserved_before.csv') -NoTypeInformation -Encoding UTF8
 $gitLegacy=@(& git -C $legacyRoot status --short)
 $gitSource=@(& git -C $sourceRoot status --short)
 $links=foreach($p in $targetPaths | Where-Object { $_.EndsWith('.pt') }){[pscustomobject]@{path=$p;hardlinks=@(& fsutil hardlink list $p)}}
 Write-Json 'plan.json' ([ordered]@{status='PREPARED';created_utc=[DateTime]::UtcNow.ToString('o');authorization='User requested cleanup of unused data and weights on 2026-10-08';delete_files=$candidates.Count;delete_logical_bytes=($candidates|Measure-Object bytes -Sum).Sum;preserved_files=$protected.Count;artifact_root=$artifactRoot;source_root=$sourceRoot;retained_weights=@('START','A_step1000.pt','B_step1000.pt','C_step1000.pt','official and prior maintained weights');ot_cache='RETAIN_ALL_8000_NPZ_AND_ATTEMPT_RETURN_LOGS';hardlinks=$links;git_legacy=$gitLegacy;git_source=$gitSource;disk_free_before_bytes=(Get-PSDrive E).Free;models_loaded=0;forwards=0;training_updates=0;new_generations=0})
 Write-Output ('Prepared '+$protected.Count+' preserved hashes. No deletion yet.')
 exit
}
$plan=Get-Content -LiteralPath (Join-Path $archive 'plan.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$candidates=@(Import-Csv -LiteralPath (Join-Path $archive 'delete_manifest.csv'))
if($candidates.Count -ne $targetPaths.Count){throw 'Manifest count mismatch'}
foreach($row in $candidates){Assert-Target $row.path}
$ledgerPath=Join-Path $archive 'deletion_ledger.jsonl'
if($Stage -eq 'Execute'){
 $active=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(3)?\.exe$' -and $_.CommandLine -match 'MeshFlow-Control' })
 if($active.Count){throw 'Active related Python process; cleanup stopped'}
 $knownDeleted=@{}
 if(Test-Path -LiteralPath $ledgerPath){Get-Content -LiteralPath $ledgerPath -Encoding UTF8 | ForEach-Object {$e=$_|ConvertFrom-Json;if($e.status -eq 'DELETED'){$knownDeleted[$e.path]=$true}}}
 foreach($row in $candidates){
  if(-not(Test-Path -LiteralPath $row.path)){
   if($knownDeleted.ContainsKey($row.path)){continue}
   throw "Unrecorded missing candidate: $($row.path)"
  }
  $item=Get-Item -LiteralPath $row.path -Force
  if($item.PSIsContainer -or $item.Length -ne [long]$row.bytes -or (Hash-File $row.path) -ne $row.sha256){throw "Candidate identity changed: $($row.path)"}
  $event=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');path=$row.path;bytes=[long]$row.bytes;sha256=$row.sha256;status='DELETE_INTENT'}
  [IO.File]::AppendAllText($ledgerPath,(ConvertTo-Json $event -Compress)+[Environment]::NewLine,$utf8)
  Remove-Item -LiteralPath $row.path -Force
  if(Test-Path -LiteralPath $row.path){throw "Deletion failed: $($row.path)"}
  $event.utc=[DateTime]::UtcNow.ToString('o');$event.status='DELETED'
  [IO.File]::AppendAllText($ledgerPath,(ConvertTo-Json $event -Compress)+[Environment]::NewLine,$utf8)
  Write-Output ('Deleted '+$row.path)
 }
 Write-Json 'execution.json' ([ordered]@{status='DELETED_PENDING_VERIFY';utc=[DateTime]::UtcNow.ToString('o');files=$candidates.Count;logical_bytes=($candidates|Measure-Object bytes -Sum).Sum;disk_free_after_bytes=(Get-PSDrive E).Free})
 exit
}
$before=@(Import-Csv -LiteralPath (Join-Path $archive 'preserved_before.csv'))
$after=[Collections.Generic.List[object]]::new();$n=0
foreach($row in $before){
 $exists=Test-Path -LiteralPath $row.path -PathType Leaf
 $hash=if($exists){Hash-File $row.path}else{''}
 $bytes=if($exists){(Get-Item -LiteralPath $row.path -Force).Length}else{0}
 $after.Add([pscustomobject]@{path=$row.path;bytes=$bytes;sha256=$hash;status=$(if(-not$exists){'MISSING'}elseif($bytes -ne [long]$row.bytes -or $hash -ne $row.sha256){'CHANGED'}else{'UNCHANGED'})})
 $n++;if($n % 10000 -eq 0){Write-Output ('Verify: '+$n+'/'+$before.Count)}
}
$after | Export-Csv -LiteralPath (Join-Path $archive 'preserved_after.csv') -NoTypeInformation -Encoding UTF8
$problems=@($after | Where-Object {$_.status -ne 'UNCHANGED'})
$stillPresent=@($candidates | Where-Object {Test-Path -LiteralPath $_.path})
$gitLegacy=@(& git -C $legacyRoot status --short);$gitSource=@(& git -C $sourceRoot status --short)
$gitPass=(($gitLegacy -join "`n") -ceq ($plan.git_legacy -join "`n")) -and (($gitSource -join "`n") -ceq ($plan.git_source -join "`n"))
$otCount=@(Get-ChildItem -LiteralPath ($artifactRoot+'\training_stream\ot_cache') -File -Filter '*.npz').Count
$remaining=@(Get-ChildItem -LiteralPath ($artifactRoot+'\checkpoints') -File | Select-Object Name,Length)
$result=[ordered]@{status=$(if($problems.Count -eq 0 -and $stillPresent.Count -eq 0 -and $gitPass -and $otCount -eq 8000){'PASS'}else{'FAIL'});utc=[DateTime]::UtcNow.ToString('o');preserved_files=$before.Count;unchanged_files=$after.Count-$problems.Count;changed_or_missing=$problems;undeleted_targets=$stillPresent;git_status_unchanged=$gitPass;ot_npz_count=$otCount;remaining_checkpoints=$remaining;deleted_files=$candidates.Count;deleted_logical_bytes=($candidates|Measure-Object bytes -Sum).Sum;disk_free_final_bytes=(Get-PSDrive E).Free;models_loaded=0;forwards=0;training_updates=0;new_generations=0}
Write-Json 'verification.json' $result
$result | ConvertTo-Json -Depth 6
if($result.status -ne 'PASS'){throw 'Preservation verification failed'}