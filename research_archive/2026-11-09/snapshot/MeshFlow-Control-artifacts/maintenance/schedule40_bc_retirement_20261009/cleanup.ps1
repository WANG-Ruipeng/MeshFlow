$ErrorActionPreference='Stop'
$run=[IO.Path]::GetFullPath('E:\MeshFlow-Control-artifacts\experiments\chair_early_alignment_schedule40_v1')
$out=[IO.Path]::GetFullPath('E:\MeshFlow-Control-artifacts\maintenance\schedule40_bc_retirement_20261009')
$null=New-Item -ItemType Directory -Path $out -Force
if(Test-Path -LiteralPath (Join-Path $out 'cleanup_result.json')){throw 'Existing cleanup result; do not repeat'}
$rels=@('checkpoints\EARLY_step6000.pt','training\EARLY\latest.pt','inference\EARLY_generator_step6000.pt','checkpoints\LATE_step6000.pt','training\LATE\latest.pt','inference\LATE_generator_step6000.pt')
$targets=@()
foreach($rel in $rels){
 $abs=[IO.Path]::GetFullPath((Join-Path $run $rel));$f=Get-Item -LiteralPath $abs
 if(-not $abs.StartsWith($run+'\',[StringComparison]::OrdinalIgnoreCase) -or $f.PSIsContainer -or (($f.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) -or $f.FullName -ne $abs){throw 'Unsafe target'}
 $targets += [pscustomobject]@{Relative=$rel;Path=$abs;Bytes=$f.Length;SHA256=(Get-FileHash -LiteralPath $abs -Algorithm SHA256).Hash.ToLower();Links=@()}
}
foreach($t in $targets){
 $links=@(& fsutil hardlink list $t.Path)
 if($LASTEXITCODE -ne 0){throw 'Cannot enumerate hardlinks'}
 $t.Links=@($links|Where-Object {$_.Trim()}|ForEach-Object {'E:'+ $_.Trim()})
 foreach($link in $t.Links){if([IO.Path]::GetFullPath($link) -notin @($targets.Path)){throw ('External hardlink: '+$link)}}
}
$protected=@()
foreach($dir in @('evaluation','outputs','gallery','source_code','audit','training','sampling')){
 $protected += @(rg --files --hidden (Join-Path $run $dir)|Where-Object {-not $_.EndsWith('.pt',[StringComparison]::OrdinalIgnoreCase)})
}
$protected += @(Get-ChildItem -LiteralPath $run -File|Select-Object -ExpandProperty FullName)
$protected += @((Join-Path $run 'checkpoints\COMMON_step5000.pt'),(Join-Path $run 'checkpoints\FM_ONLY_step6000.pt'),(Join-Path $run 'inference\FM_ONLY_generator_step6000.pt'),(Join-Path $run 'stream\train_plan.json'),(Join-Path $run 'stream\paired_inputs.json'),(Join-Path $run 'stream\precompute_receipt.json'),'E:\MeshFlow\native_t1\checkpoints\chair_hybrid_start_total5000.pt','E:\MeshFlow\native_t1\runs\control_mvp48_v1\checkpoints\official_chair_last.pt','E:\MeshFlow-Control-artifacts\experiments\chair_contextual_alignment_sie_v1\inference\A_generator_step1000.pt')
$before=@()
foreach($p in @($protected|Sort-Object -Unique)){
 if($p -in @($targets.Path)){throw 'Protected target overlap'}
 $f=Get-Item -LiteralPath $p
 $before += [pscustomobject]@{Path=$f.FullName;Bytes=$f.Length;SHA256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower()}
}
$groups=@{};foreach($t in $targets){$groups[(@($t.Links|Sort-Object)-join '|')]=$t.Bytes}
$physical=($groups.Values|Measure-Object -Sum).Sum;$logical=($targets|Measure-Object Bytes -Sum).Sum
$plan=[ordered]@{Status='VALIDATED_BEFORE_DELETE';Scope='User authorized immediately preceding BC-related weights: EARLY/LATE B branches only';Targets=$targets;Protected=$before;LogicalBytes=$logical;UniquePhysicalBytes=$physical;UniquePhysicalFiles=$groups.Count;TimestampUtc=[DateTime]::UtcNow.ToString('o')}
[IO.File]::WriteAllText((Join-Path $out 'cleanup_manifest.json'),($plan|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
foreach($t in $targets){
 $resolved=(Resolve-Path -LiteralPath $t.Path).Path
 if($resolved -ne $t.Path -or -not $resolved.StartsWith($run+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Pre-delete resolved target check failed'}
 Remove-Item -LiteralPath $t.Path
 $entry=([ordered]@{Path=$t.Path;SHA256=$t.SHA256;Deleted= -not(Test-Path -LiteralPath $t.Path);Utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress)+[char]10
 [IO.File]::AppendAllText((Join-Path $out 'deletions.jsonl'),$entry,[Text.UTF8Encoding]::new($false))
}
$changed=@()
foreach($x in $before){
 if(-not(Test-Path -LiteralPath $x.Path)){$changed += $x.Path;continue}
 if((Get-Item -LiteralPath $x.Path).Length -ne $x.Bytes -or (Get-FileHash -LiteralPath $x.Path -Algorithm SHA256).Hash.ToLower() -ne $x.SHA256){$changed += $x.Path}
}
$remaining=@($targets|Where-Object {Test-Path -LiteralPath $_.Path})
$result=[ordered]@{Status=$(if($changed.Count -eq 0 -and $remaining.Count -eq 0){'COMPLETE'}else{'FAIL'});DeletedPaths=6-$remaining.Count;UniquePhysicalFiles=$groups.Count;LogicalBytes=$logical;UniquePhysicalBytes=$physical;ProtectedHashChecks=$before.Count;ChangedProtected=$changed;RemainingTargets=$remaining;ModelCalls=0;TimestampUtc=[DateTime]::UtcNow.ToString('o')}
[IO.File]::WriteAllText((Join-Path $out 'cleanup_result.json'),($result|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
$notice="# Schedule40 B-branch weight retirement"+[char]10+[char]10+"EARLY/LATE training latest, endpoint hardlinks and inference weights deleted under explicit user authorization 2026-10-09: six paths, four unique physical files, $physical bytes. Historical reports/source/RAW/CSV/gallery/ZIP remain unchanged; old weight-existence statements are historical. FM_ONLY, COMMON, official, START and historical FM remain. No model calls. See cleanup_manifest.json, deletions.jsonl and cleanup_result.json. Do not automatically regenerate retired weights."+[char]10
[IO.File]::WriteAllText((Join-Path $out 'CLEANUP_REPORT.md'),$notice,[Text.UTF8Encoding]::new($false))
$result|ConvertTo-Json -Depth 4
if($result.Status -ne 'COMPLETE'){throw 'Cleanup verification failed'}