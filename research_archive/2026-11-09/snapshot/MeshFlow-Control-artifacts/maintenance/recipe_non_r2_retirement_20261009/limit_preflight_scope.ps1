$ErrorActionPreference='Stop'
$path='E:\MeshFlow-Control-artifacts\maintenance\recipe_non_r2_retirement_20261009\cleanup.ps1'
$content=Get-Content -LiteralPath $path -Raw -Encoding UTF8
$old='$protectedPaths=@($all.FullName | Where-Object {$targets -notcontains $_})'
$new=@'
 $metadataLines=@($all | Where-Object {$targets -notcontains $_.FullName} | Sort-Object FullName | ForEach-Object {$_.FullName+'|'+$_.Length+'|'+$_.LastWriteTimeUtc.Ticks})
 $metadataHash=([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes(($metadataLines -join [char]10))))).Replace('-','').ToLower()
 $protectedPaths=@($all.FullName | Where-Object {$targets -notcontains $_ -and $_ -notmatch '\\streams\\(?:new20_ot_cache|paired_inputs|precompute_attempts|confirmation_ot_cache)\\'})
'@
if(-not $content.Contains($old)){throw 'Patch target absent'}
$content=$content.Replace($old,$new)
$content=$content.Replace('run_root=$runRoot;delete=$deleteRows','run_root=$runRoot;metadata_preserved_files=$metadataLines.Count;metadata_before_sha256=$metadataHash;hash_scope="All retained files except bulk derived stream cache directories; all retained files separately inventoried by path, size and modification time";delete=$deleteRows')
$old='$remaining=@(Get-ChildItem -LiteralPath $runRoot -Recurse -File | Where-Object {$_.Extension -eq ''.pt''} | Select-Object FullName,Length)'
$new=@'
 $filesAfter=@(Get-ChildItem -LiteralPath $runRoot -Recurse -File)
 $metadataLinesAfter=@($filesAfter | Sort-Object FullName | ForEach-Object {$_.FullName+'|'+$_.Length+'|'+$_.LastWriteTimeUtc.Ticks})
 $metadataHashAfter=([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes(($metadataLinesAfter -join [char]10))))).Replace('-','').ToLower()
 if($metadataHashAfter -ne $plan.metadata_before_sha256){$mismatches+=@('Retained file metadata inventory changed')}
 $remaining=@($filesAfter | Where-Object {$_.Extension -eq '.pt'} | Select-Object FullName,Length)
'@
if(-not $content.Contains($old)){throw 'Second patch target absent'}
$content=$content.Replace($old,$new)
$content=$content.Replace('protected_files_verified=$plan.protected.Count;protected_mismatches=$mismatches','protected_files_verified=$plan.protected.Count;metadata_preserved_files=$plan.metadata_preserved_files;metadata_before_sha256=$plan.metadata_before_sha256;metadata_after_sha256=$metadataHashAfter;hash_scope=$plan.hash_scope;protected_mismatches=$mismatches')
[IO.File]::WriteAllText($path,$content,[Text.UTF8Encoding]::new($false))
