$ErrorActionPreference = 'Stop'
$exp = 'E:\MeshFlow\experiments\r2_bench_parts_convergence_v1'
$out = $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding($false)
$sources = [System.Collections.Generic.HashSet[string]]::new()
function ReadJson([string]$path) { [void]$sources.Add($path); Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json }
function ReadCsv([string]$path) { [void]$sources.Add($path); Import-Csv -LiteralPath $path }
function SaveJson([string]$name, $obj) { [IO.File]::WriteAllText((Join-Path $out $name), ($obj|ConvertTo-Json -Depth 20), $utf8) }
$registration=ReadJson "$exp\registration.json"
$frozen=ReadJson "$exp\evaluation_inputs\manifest.json"
[void]$sources.Add("$exp\REPORT.md")
[void]$sources.Add("$exp\DELIVERY_INDEX.md")
[void]$sources.Add('E:\MeshFlow\experiments\r2_bench_post_rescue_diagnostic_v1\point_contact\per_output.csv')
$tables=@{}; $galleries=@{}; $summary=@(); $checks=@()
foreach($s in @(15000,21000)) {
  $tables[$s]=@(ReadCsv "$exp\evaluation\tables\NATIVE_S${s}__SEAM_S${s}_full\per_output.csv")
  $galleries[$s]=ReadJson "$exp\gallery_manifests\NATIVE_S${s}__SEAM_S${s}_full.json"
  foreach($arm in @('NATIVE','SEAM')) { foreach($split in @('TRAIN','DEV','ALL')) {
    $rows=@($tables[$s]|Where-Object {$_.label -eq "${arm}_S${s}" -and $_.stage -eq 'POST' -and ($split -eq 'ALL' -or $_.split -eq $split)})
    $rawSum=($rows|ForEach-Object {[int]$_.N-[int]$_.K}|Measure-Object -Sum).Sum
    $delSum=($rows|ForEach-Object {[int]$_.deleted_free_faces}|Measure-Object -Sum).Sum
    $retSum=($rows|ForEach-Object {[int]$_.actual_free_faces}|Measure-Object -Sum).Sum
    if($rawSum -ne $delSum+$retSum) { throw 'Count accounting mismatch' }
    $summary += [pscustomobject]@{step=$s;arm=$arm;split=$split;requests=$rows.Count;raw_free_mean=$rawSum/$rows.Count;deleted_free_mean=$delSum/$rows.Count;retained_free_mean=$retSum/$rows.Count;raw_free_total=$rawSum;deleted_free_total=$delSum;retained_free_total=$retSum;pooled_deleted_fraction=$delSum/$rawSum}
  }}
}
$identity1=@($galleries[15000].pages|ForEach-Object {$_|Select-Object request_id,condition_id,parent_id,noise_index,N,K,split|ConvertTo-Json -Compress})
$identity2=@($galleries[21000].pages|ForEach-Object {$_|Select-Object request_id,condition_id,parent_id,noise_index,N,K,split|ConvertTo-Json -Compress})
if(@(Compare-Object $identity1 $identity2).Count) { throw 'Gallery request identity mismatch' }
foreach($q in $frozen.requests) {
  $r=@{}; $p=@{}
  foreach($s in @(15000,21000)) {
    $r[$s]=ReadJson "$exp\evaluation\SEAM_S$s\$($q.request_id)\raw\receipt.json"
    $p[$s]=ReadJson "$exp\evaluation\SEAM_S$s\$($q.request_id)\post\receipt.json"
  }
  $sourceHashes=@(foreach($s in @(15000,21000)) { ($r[$s].files.PSObject.Properties|Where-Object {$_.Name.EndsWith('/source_condition.npz')}).Value })
  $checks += [pscustomobject]@{request_id=$q.request_id;split=$q.split;part=$q.part_type;N=$q.N;K=$q.K;noise_index=$q.noise_index;same_noise=($r[15000].noise_sha256 -eq $r[21000].noise_sha256 -and $r[21000].noise_sha256 -eq $q.noise_sha256);same_saved_source_C=($sourceHashes[0] -eq $sourceHashes[1]);same_D0=($p[15000].official_D0_sha256 -eq $p[21000].official_D0_sha256);same_post_schema=($p[15000].schema -eq $p[21000].schema);D0_once_both=($p[15000].D0_forwards -eq 1 -and $p[21000].D0_forwards -eq 1);Euler50_both=($r[15000].generator_forwards -eq 50 -and $r[21000].generator_forwards -eq 50);source_labels_read=($r[15000].source_labels_read -or $r[21000].source_labels_read)}
}
$pair=@()
foreach($q in $tables[21000]|Where-Object {$_.label -eq 'SEAM_S21000' -and $_.stage -eq 'POST'}) {
  $p=$tables[15000]|Where-Object {$_.label -eq 'SEAM_S15000' -and $_.stage -eq 'POST' -and $_.request_id -eq $q.request_id}
  $pair += [pscustomobject]@{request_id=$q.request_id;split=$q.split;part=$q.part_type;N=[int]$q.N;K=[int]$q.K;raw_free=[int]$q.N-[int]$q.K;deleted15000=[int]$p.deleted_free_faces;deleted21000=[int]$q.deleted_free_faces;retained21000=[int]$q.actual_free_faces;deleted_fraction21000=[double]$q.deleted_free_faces/([int]$q.N-[int]$q.K);delete_delta=[int]$q.deleted_free_faces-[int]$p.deleted_free_faces}
}
$summary|Export-Csv -LiteralPath "$out\deletion_summary_existing.csv" -NoTypeInformation -Encoding UTF8
$pair|Sort-Object request_id|Export-Csv -LiteralPath "$out\seam_48_request_delete_pair.csv" -NoTypeInformation -Encoding UTF8
$checks|Export-Csv -LiteralPath "$out\request_identity_checks.csv" -NoTypeInformation -Encoding UTF8
SaveJson 'checks.json' @{scope='CPU reads of existing reports, manifests, tables and receipts; no model or rescoring';request_count=$checks.Count;gallery_identity_difference_count=0;failed_checks=@($checks|Where-Object {-not ($_.same_noise -and $_.same_saved_source_C -and $_.same_D0 -and $_.same_post_schema -and $_.D0_once_both -and $_.Euler50_both) -or $_.source_labels_read});initialization=$registration.initialization;head_policy=$registration.head_policy;head_Chair_difference=$registration.head_Chair_difference}
$selected=@('f5d370e9fe5b7452dbbc9440457e303e__BACK_SURFACE_1__n1','9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n0','9d56087c9f98bf90717cca22daef45e7__LEG_SUPPORT_1__n0','1dd7409c95d59d0da192483aa282f8e5__SEAT_SURFACE_1__n0')
foreach($rid in $selected) { [void]$sources.Add("$exp\gallery\NATIVE_S21000__SEAM_S21000_full\$rid.png") }
[void]$sources.Add("$exp\gallery\NATIVE_S15000__SEAM_S15000_full\$($selected[0]).png")
[void]$sources.Add("$out\collect_context.ps1")
$hashes=@(foreach($path in @($sources)|Sort-Object) { $i=Get-Item -LiteralPath $path; [pscustomobject]@{path=$i.FullName;bytes=$i.Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()} })
SaveJson 'source_hashes.json' $hashes
SaveJson 'typical_cases.json' @($pair|Where-Object {$_.request_id -in $selected})
[pscustomobject]@{status='COMPLETE';sources=$hashes.Count;request_checks=$checks.Count;summary_rows=$summary.Count;paired_requests=$pair.Count;selected_cases=$selected.Count}|ConvertTo-Json
