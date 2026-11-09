param([ValidateSet('Inventory','Delete')][string]$Stage='Inventory')
$ErrorActionPreference='Stop'
$runRoot=[IO.Path]::GetFullPath('E:\MeshFlow-Control-artifacts\experiments\chair_condition_recipe_factorial_v1')
$auditRoot='E:\MeshFlow-Control-artifacts\maintenance\recipe_non_r2_retirement_20261009'
$utf8=[Text.UTF8Encoding]::new($false)
function Save-Json($Path,$Value){[IO.File]::WriteAllText($Path,($Value | ConvertTo-Json -Depth 12),$utf8)}
function Check-Path($Path){
 $full=[IO.Path]::GetFullPath($Path)
 if(-not $full.StartsWith($runRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Outside allowed experiment: $full"}
 if([IO.Path]::GetExtension($full) -ne '.pt'){throw "Not a PT file: $full"}
 $node=Get-Item -LiteralPath $full
 while($node){
  if(($node.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw "Reparse point: $($node.FullName)"}
  if($node.FullName -eq $runRoot){break}
  $node=Get-Item -LiteralPath ([IO.Path]::GetDirectoryName($node.FullName))
 }
 return $full
}
function Links($Path){
 $lines=@(& fsutil hardlink list $Path)
 if($LASTEXITCODE -ne 0){throw "Cannot inspect hardlinks: $Path"}
 $result=@($lines | ForEach-Object {$s=$_.Trim(); if($s.StartsWith('\')){'E:'+$s}} | Sort-Object -Unique)
 if($result.Count -eq 0){throw "No hardlinks: $Path"}
 return $result
}
$relative=@()
foreach($arm in @('R3_40_H_LATE','R4_MIX_H_LATE','C20_40_MIX','C40_20_MIX')){
 $relative+=@("checkpoints\${arm}_step6000.pt","inference\${arm}_generator_step6000.pt","training\$arm\latest.pt")
}
$relative+=@('confirmation\checkpoints\C20_40_MIX_step6000.pt','confirmation\inference\C20_40_MIX_generator_step6000.pt','confirmation\training\C20_40_MIX\latest.pt')
$targets=@($relative | ForEach-Object {[IO.Path]::GetFullPath((Join-Path $runRoot $_))})
if($Stage -eq 'Inventory'){
 if(Test-Path -LiteralPath ($auditRoot+'\inventory.json')){throw 'Inventory already exists'}
 [IO.Directory]::CreateDirectory($auditRoot) | Out-Null
 $all=@(Get-ChildItem -LiteralPath $runRoot -File -Recurse)
 $binding=Get-Content -LiteralPath ($runRoot+'\analysis\model_index.json') -Raw -Encoding UTF8 | ConvertFrom-Json
 $hashCache=@{}
 $deleteRows=@(foreach($p in $targets){
  $p=Check-Path $p
  $item=Get-Item -LiteralPath $p
  $links=@(Links $p)
  foreach($link in $links){if($targets -notcontains $link){throw "Deletion inode has unlisted retained link: $link"}}
  $key=$links -join '|'
  if(-not $hashCache.ContainsKey($key)){$hashCache[$key]=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower()}
  $expected=@($binding.instances | Where-Object {($_.checkpoint_path -replace '^/mnt/e/','E:/' -replace '/','\') -eq $p})
  if($expected.Count -gt 0 -and $expected[0].checkpoint_sha256 -ne $hashCache[$key]){throw "Frozen model identity mismatch: $p"}
  [pscustomobject]@{path=$p;bytes=$item.Length;sha256=$hashCache[$key];hardlinks=$links;inode_group=$key}
 })
  $metadataLines=@($all | Where-Object {$targets -notcontains $_.FullName} | Sort-Object FullName | ForEach-Object {$_.FullName+'|'+$_.Length+'|'+$_.LastWriteTimeUtc.Ticks})
 $metadataHash=([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes(($metadataLines -join [char]10))))).Replace('-','').ToLower()
 $protectedPaths=@($all.FullName | Where-Object {$targets -notcontains $_ -and $_ -notmatch '\\streams\\(?:new20_ot_cache|paired_inputs|precompute_attempts|confirmation_ot_cache)\\'})
 $protectedPaths+=@($binding.instances | Where-Object {$_.checkpoint_path -notlike '/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1/*'} | ForEach-Object {$_.checkpoint_path -replace '^/mnt/e/','E:/' -replace '/','\'})
 $protected=@(foreach($p in ($protectedPaths | Sort-Object -Unique)){
  $f=Get-Item -LiteralPath $p
  [pscustomobject]@{path=$f.FullName;bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower()}
 })
 $unique=@($deleteRows | Group-Object inode_group | ForEach-Object {$_.Group[0]})
 $plan=[pscustomobject]@{status='READY';utc=[DateTime]::UtcNow.ToString('o');authorization='User selects R2 recipe and explicitly requests removing other weights from just-completed experiment';run_root=$runRoot;metadata_preserved_files=$metadataLines.Count;metadata_before_sha256=$metadataHash;hash_scope="All retained files except bulk derived stream cache directories; all retained files separately inventoried by path, size and modification time";delete=$deleteRows;protected=$protected;delete_paths=$deleteRows.Count;unique_weight_files=$unique.Count;logical_path_bytes=($deleteRows|Measure-Object bytes -Sum).Sum;unique_file_bytes=($unique|Measure-Object bytes -Sum).Sum;free_bytes_before=(Get-PSDrive E).Free;retained='Both main and confirm R2 complete training checkpoints, latest aliases and generator exports; reused R1/HIST/START; all non-weight experiment files';model_calls=0;training_updates=0}
 Save-Json ($auditRoot+'\inventory.json') $plan
 [pscustomobject]@{status=$plan.status;delete_paths=$plan.delete_paths;unique_weight_files=$plan.unique_weight_files;logical_path_bytes=$plan.logical_path_bytes;unique_file_bytes=$plan.unique_file_bytes;protected_files=$protected.Count;inventory=($auditRoot+'\inventory.json')}|ConvertTo-Json
}else{
 $plan=Get-Content -LiteralPath ($auditRoot+'\inventory.json') -Raw -Encoding UTF8 | ConvertFrom-Json
 if($plan.status -ne 'READY' -or $plan.delete_paths -ne 15){throw 'Unexpected inventory'}
 if(Test-Path -LiteralPath ($auditRoot+'\receipt.json')){throw 'Already completed'}
 foreach($r in $plan.delete){
  $p=Check-Path $r.path
  if($targets -notcontains $p){throw 'Manifest target not on hardcoded list'}
  if((Get-Item -LiteralPath $p).Length -ne $r.bytes -or (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash -ne $r.sha256){throw "File changed: $p"}
  $now=@(Links $p)
  if(($now -join '|') -ne ($r.hardlinks -join '|')){throw "Hardlink membership changed: $p"}
 }
 foreach($r in $plan.delete){
  Remove-Item -LiteralPath $r.path -Force
  if(Test-Path -LiteralPath $r.path){throw "Deletion failed: $($r.path)"}
  $line=([pscustomobject]@{utc=[DateTime]::UtcNow.ToString('o');action='DELETE_PT_ONLY';path=$r.path;bytes=$r.bytes;sha256=$r.sha256}|ConvertTo-Json -Compress)
  [IO.File]::AppendAllText($auditRoot+'\deletions.jsonl',$line+[Environment]::NewLine,$utf8)
 }
 $mismatches=@(foreach($r in $plan.protected){
  if(-not(Test-Path -LiteralPath $r.path)){$r.path}
  elseif((Get-FileHash -LiteralPath $r.path -Algorithm SHA256).Hash -ne $r.sha256){$r.path}
 })
  $filesAfter=@(Get-ChildItem -LiteralPath $runRoot -Recurse -File)
 $metadataLinesAfter=@($filesAfter | Sort-Object FullName | ForEach-Object {$_.FullName+'|'+$_.Length+'|'+$_.LastWriteTimeUtc.Ticks})
 $metadataHashAfter=([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes(($metadataLinesAfter -join [char]10))))).Replace('-','').ToLower()
 if($metadataHashAfter -ne $plan.metadata_before_sha256){$mismatches+=@('Retained file metadata inventory changed')}
 $remaining=@($filesAfter | Where-Object {$_.Extension -eq '.pt'} | Select-Object FullName,Length)
 $status=if($mismatches.Count -eq 0 -and $remaining.Count -eq 6){'COMPLETE'}else{'VERIFY_FAILED'}
 $receipt=[pscustomobject]@{status=$status;utc=[DateTime]::UtcNow.ToString('o');deleted_paths=$plan.delete_paths;unique_deleted_files=$plan.unique_weight_files;removed_unique_file_bytes=$plan.unique_file_bytes;logical_path_bytes=$plan.logical_path_bytes;free_bytes_before=$plan.free_bytes_before;free_bytes_after=(Get-PSDrive E).Free;protected_files_verified=$plan.protected.Count;metadata_preserved_files=$plan.metadata_preserved_files;metadata_before_sha256=$plan.metadata_before_sha256;metadata_after_sha256=$metadataHashAfter;hash_scope=$plan.hash_scope;protected_mismatches=$mismatches;remaining_weights=$remaining;model_calls=0;training_updates=0;git_operations=0;default_changed=$false}
 Save-Json ($auditRoot+'\receipt.json') $receipt
 $receipt | ConvertTo-Json -Depth 4
 if($status -ne 'COMPLETE'){throw 'Post-cleanup verification failed'}
}
