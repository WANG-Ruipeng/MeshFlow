param([ValidateSet('Verify','Delete')][string]$Mode='Verify')
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$workspaceRoot='E:\MeshFlow'
$outDir=Join-Path $workspaceRoot 'experiments\r2_bench_control_transfer_v1\cleanup'
$utf8=New-Object System.Text.UTF8Encoding($false)
function Save-Json([string]$name,$value) {
 [IO.File]::WriteAllText((Join-Path $outDir $name),(ConvertTo-Json -InputObject $value -Depth 30),$utf8)
}
function Event($value) {
 [IO.File]::AppendAllText((Join-Path $outDir 'ledger.jsonl'),(ConvertTo-Json -InputObject $value -Compress -Depth 15)+[Environment]::NewLine,$utf8)
}
function Safe-File([string]$path) {
 $abs=[IO.Path]::GetFullPath($path)
 if(-not $abs.StartsWith($workspaceRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Outside workspace: $abs"}
 $resolved=(Resolve-Path -LiteralPath $abs).ProviderPath
 if(-not $resolved.Equals($abs,[StringComparison]::OrdinalIgnoreCase)){throw "Resolution changed: $abs"}
 $item=Get-Item -LiteralPath $resolved -Force
 if($item.PSIsContainer){throw "Directory prohibited: $resolved"}
 $cursor=$abs
 while($cursor.Length -ge $workspaceRoot.Length){
  $part=Get-Item -LiteralPath $cursor -Force
  if(($part.Attributes -band [IO.FileAttributes]::ReparsePoint)-ne 0){throw "Reparse point prohibited: $cursor"}
  $cursor=[IO.Path]::GetDirectoryName($cursor)
 }
 return $item
}
$targets=@(
 'experiments\r2_full_contour_readout_v1\checkpoints\CONTOUR_step1000.pt',
 'experiments\r2_full_contour_readout_v1\checkpoints\CONTOUR_step1114.pt',
 'experiments\r2_full_contour_readout_v1\checkpoints\CONTOUR_step3000.pt',
 'experiments\r2_full_contour_readout_v1\checkpoints\CONTOUR_step6000.pt',
 'experiments\r2_full_contour_readout_v1\inference\CONTOUR_S1000_generator.pt',
 'experiments\r2_full_contour_readout_v1\inference\CONTOUR_S3000_generator.pt',
 'experiments\r2_seam_hinge_quality_v1\checkpoints\CLEAN_EDGE_COH_rolling.pt',
 'experiments\r2_seam_hinge_quality_v1\checkpoints\CLEAN_EDGE_COH_step1000.pt',
 'experiments\r2_seam_hinge_quality_v1\checkpoints\SMOOTH_PRIOR_rolling.pt',
 'experiments\r2_seam_hinge_quality_v1\checkpoints\SMOOTH_PRIOR_step1000.pt',
 'experiments\r2_seam_quality_wave1_wave2_v1\checkpoints\CLEAN_BULK_FM_rolling.pt',
 'experiments\r2_seam_quality_wave1_wave2_v1\checkpoints\SURGICAL_LATE4_rolling.pt',
 'experiments\r2_seam_pool47_couple_v1\checkpoints\COUPLE3000_step3000.pt'
) | ForEach-Object { Join-Path $workspaceRoot $_ }
$protected=@(
 'experiments\r2_full_contour_readout_v1\inference\CONTOUR_S6000_generator.pt',
 'experiments\r2_seam_hinge_quality_v1\inference\CLEAN_EDGE_COH_generator.pt',
 'experiments\r2_seam_hinge_quality_v1\inference\SMOOTH_PRIOR_generator.pt',
 'experiments\r2_seam_pool47_couple_v1\inference\COUPLE3000_generator.pt',
 'experiments\r2_seam_quality_wave1_wave2_v1\checkpoints\CLEAN_BULK_FM_step1000.pt',
 'experiments\r2_seam_quality_wave1_wave2_v1\checkpoints\SURGICAL_LATE4_step1000.pt',
 'experiments\r2_seam_quality_wave1_wave2_v1\inference\CLEAN_BULK_FM_generator.pt',
 'experiments\r2_seam_quality_wave1_wave2_v1\inference\SURGICAL_LATE4_generator.pt',
 'experiments\r2_seam_pool47_couple_v1\checkpoints\PLAIN3000_step2000.pt',
 'experiments\r2_seam_quality_wave1_wave2_v1\inference\PLAIN_S2000_generator.pt',
 'experiments\r2_seam_quality_four_v1\checkpoints\LONG_step1000.pt',
 'experiments\r2_seam_quality_four_v1\inference\LONG_occupancy.pt',
 'experiments\r2_seam_abc_v1\checkpoints\B_POOL47_step1000.pt',
 'experiments\r2_seam_slot_c200_v1\checkpoints\SEAM_SLOT_C200_step200.pt',
 'experiments\r2_seam_slot_c200_v1\checkpoints\FULL4_C200_step200.pt',
 'experiments\r2_full_seat_learning_curve_v1\checkpoints\FULL_SEAT_step6000.pt',
 'experiments\r2_user_data_scaling_32_67_v1\checkpoints\USER_EXPANDED_67_step6000.pt',
 'native_t1\checkpoints\chair_hybrid_start_total5000.pt',
 'native_t1\runs\control_mvp48_v1\checkpoints\official_chair_last.pt',
 'experiments\r2_postprocess_avmr_pmr_v1\assets\chair_denoiser_0030000.pt',
 'experiments\r2_full_contour_readout_v1\REPORT.md',
 'experiments\r2_seam_hinge_quality_v1\REPORT.md',
 'experiments\r2_seam_quality_wave1_wave2_v1\REPORT.md',
 'experiments\r2_seam_pool47_couple_v1\REPORT.md'
) | ForEach-Object { Join-Path $workspaceRoot $_ }
$manifestPath=Join-Path $outDir 'deletion_manifest.json'
if($Mode -eq 'Verify'){
 if(Test-Path -LiteralPath $manifestPath){throw 'Manifest exists; refusing overwrite.'}
 $records=@()
 foreach($p in $targets){
  $f=Safe-File $p
  $reason=if($p -match 'rolling'){ 'Obsolete recovery checkpoint of completed run; final endpoint preserved.' }elseif($p -match 'CONTOUR'){ 'Completed non-promoted contour route; final inference export and all evidence preserved.' }else{ 'Completed non-promoted candidate optimizer checkpoint; final inference export and all evidence preserved.' }
  $records += [pscustomobject][ordered]@{path=$p;bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant();last_write_ticks=$f.LastWriteTimeUtc.Ticks;reason=$reason}
 }
 $keep=@()
 foreach($p in $protected){$f=Safe-File $p; $keep += [ordered]@{path=$p;bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant();last_write_ticks=$f.LastWriteTimeUtc.Ticks}}
 $sum=[long](($records | Measure-Object bytes -Sum).Sum)
 if($records.Count -ne 13 -or $sum -ne 17602302735){throw 'Unexpected count/bytes.'}
 $inventory=@(rg --files -uu experiments native_t1 r2 -g '*.pt' -g '*.pth' -g '*.ckpt' -g '*.safetensors' | ForEach-Object {$f=Get-Item -LiteralPath $_; [ordered]@{path=$f.FullName;bytes=$f.Length;last_write_ticks=$f.LastWriteTimeUtc.Ticks}})
 Save-Json 'weight_inventory_before.json' $inventory
 Save-Json 'deletion_manifest.json' ([ordered]@{schema='negative_weight_cleanup_v1';created_utc=[DateTime]::UtcNow.ToString('o');authorization='Current user requested old unused negative-result weight cleanup before non-chair training; conservative scope keeps compact inference endpoints.';root=$workspaceRoot;expected_bytes=$sum;targets=$records;protected=$keep;reports_read=@('experiments/r2_full_contour_readout_v1/REPORT.md','experiments/r2_seam_hinge_quality_v1/REPORT.md','experiments/r2_seam_quality_wave1_wave2_v1/REPORT.md','experiments/r2_seam_pool47_couple_v1/REPORT.md');resume_retirement='Full optimizer continuation and intermediate weight replay retired only for explicitly deleted files. Historical reports remain unchanged; archived results remain valid.';no_recursive_deletion=$true;no_model_calls=$true})
 Event ([ordered]@{event='PRECHECK_COMPLETE';utc=[DateTime]::UtcNow.ToString('o');count=$records.Count;bytes=$sum;protected_count=$keep.Count})
 Write-Output ('VERIFIED '+$records.Count+' files; '+$sum+' bytes; '+$keep.Count+' protected SHA256 identities.')
 exit
}
if(Test-Path -LiteralPath (Join-Path $outDir 'cleanup_receipt.json')){throw 'Completed receipt exists; refusing repeat deletion.'}
$m=Get-Content -LiteralPath $manifestPath -Encoding UTF8 -Raw | ConvertFrom-Json
if($m.targets.Count -ne 13 -or $m.expected_bytes -ne 17602302735){throw 'Manifest mismatch.'}
foreach($r in $m.targets){
 if($targets -notcontains $r.path){throw 'Target not in exact whitelist.'}
 $f=Safe-File $r.path
 if($f.Extension -ne '.pt' -or $f.Length -ne $r.bytes -or $f.LastWriteTimeUtc.Ticks -ne $r.last_write_ticks){throw "Changed deletion target: $($r.path)"}
}
$deletions=@()
foreach($r in $m.targets){
 Remove-Item -LiteralPath $r.path -Force
 if(Test-Path -LiteralPath $r.path){throw "Still exists: $($r.path)"}
 $entry=[pscustomobject][ordered]@{event='DELETED';utc=[DateTime]::UtcNow.ToString('o');path=$r.path;bytes=$r.bytes;sha256=$r.sha256}
 $deletions += $entry; Event $entry
}
$kept=@()
foreach($r in $m.protected){
 $f=Safe-File $r.path
 if($f.Length -ne $r.bytes -or $f.LastWriteTimeUtc.Ticks -ne $r.last_write_ticks){throw "Protected identity changed: $($r.path)"}
 $sha=(Get-FileHash -LiteralPath $r.path -Algorithm SHA256).Hash.ToLowerInvariant()
 if($sha -ne $r.sha256){throw "Protected hash changed: $($r.path)"}
 $kept += [ordered]@{path=$r.path;bytes=$r.bytes;sha256=$sha;status='HASH_UNCHANGED'}
}
$old=Get-Content -LiteralPath (Join-Path $outDir 'weight_inventory_before.json') -Encoding UTF8 -Raw | ConvertFrom-Json
foreach($r in $old){if($targets -notcontains $r.path){$f=Safe-File $r.path; if($f.Length -ne $r.bytes -or $f.LastWriteTimeUtc.Ticks -ne $r.last_write_ticks){throw "Non-target weight changed: $($r.path)"}}}
Save-Json 'cleanup_receipt.json' ([ordered]@{status='COMPLETE';finished_utc=[DateTime]::UtcNow.ToString('o');deleted_count=$deletions.Count;logical_bytes_freed=[long](($deletions | Measure-Object bytes -Sum).Sum);actual_filesystem_free_delta='NA_NOT_MEASURED';deletions=$deletions;protected_after=$kept;non_target_weight_count=($old.Count-$targets.Count);non_target_weight_size_and_timestamp_unchanged=$true;raw_reports_sources_deleted=0;external_paths_deleted=0;recursive_deletions=0;model_calls=0;resume_retirement=$m.resume_retirement})
Event ([ordered]@{event='COMPLETE';utc=[DateTime]::UtcNow.ToString('o');deleted_count=$deletions.Count;logical_bytes_freed=$m.expected_bytes})
Write-Output ('DELETED '+$deletions.Count+' files; '+$m.expected_bytes+' logical bytes; protected SHA256 unchanged.')
