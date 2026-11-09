param([ValidateSet('Verify','Delete')][string]$Mode = 'Verify')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Repo = 'E:\MeshFlow'
$Out = 'E:\MeshFlow\experiments\r2_seam_quality_wave1_wave2_v1\cleanup'
$PromptPath = 'C:\Users\warpwang\.codex\attachments\3a4e99e9-7378-4c0f-88fa-0fb549027563\Pasted text.txt'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
$Experiments = @('r2_seam_slot_c200_v1','r2_seam_quality_four_v1','r2_seam_abc_v1','r2_seam_pool47_couple_v1')
$DeleteRelative = @(
 'r2_seam_pool47_couple_v1\checkpoints\COUPLE3000_step1000.pt',
 'r2_seam_pool47_couple_v1\checkpoints\COUPLE3000_step2000.pt',
 'r2_seam_pool47_couple_v1\checkpoints\PLAIN3000_rolling.pt',
 'r2_seam_pool47_couple_v1\checkpoints\COUPLE3000_rolling.pt',
 'r2_seam_slot_c200_v1\checkpoints\FULL4_C200_rolling.pt',
 'r2_seam_slot_c200_v1\checkpoints\SEAM_SLOT_C200_rolling.pt',
 'r2_seam_quality_four_v1\checkpoints\LONG_rolling.pt',
 'r2_seam_quality_four_v1\checkpoints\LOCAL_NOISE_rolling.pt',
 'r2_seam_quality_four_v1\checkpoints\ANTI_COVER_rolling.pt',
 'r2_seam_abc_v1\checkpoints\A_COUPLE_rolling.pt',
 'r2_seam_abc_v1\checkpoints\B_POOL47_rolling.pt',
 'r2_seam_abc_v1\checkpoints\C_SHARED_TIP_rolling.pt',
 'r2_seam_quality_four_v1\checkpoints\LOCAL_NOISE_step1000.pt',
 'r2_seam_quality_four_v1\checkpoints\ANTI_COVER_step1000.pt',
 'r2_seam_abc_v1\checkpoints\A_COUPLE_step1000.pt',
 'r2_seam_abc_v1\checkpoints\C_SHARED_TIP_step1000.pt'
)
$DeletePaths = @($DeleteRelative | ForEach-Object { Join-Path ($Repo + '\experiments') $_ })
$AllowedDirs = @($Experiments | ForEach-Object { $Repo + '\experiments\' + $_ + '\checkpoints' })
$ExpectedDeleteBytes = [long]25041197776
$Metadata = @{}
function To-Windows([string]$p) { return $p.Replace('/mnt/e/','E:\').Replace('/','\') }
function Read-Json([string]$p) { return Get-Content -LiteralPath $p -Encoding UTF8 -Raw | ConvertFrom-Json }
function Write-Json([string]$name, $obj) {
 [System.IO.File]::WriteAllText((Join-Path $Out $name), (ConvertTo-Json -InputObject $obj -Depth 30), $Utf8)
}
function Append-Event($obj) {
 [System.IO.File]::AppendAllText((Join-Path $Out 'ledger.jsonl'), (ConvertTo-Json -InputObject $obj -Depth 10 -Compress) + [Environment]::NewLine, $Utf8)
}
function Save-Csv([string]$name, $rows) {
 [System.IO.File]::WriteAllText((Join-Path $Out $name), (($rows | ConvertTo-Csv -NoTypeInformation) -join [Environment]::NewLine) + [Environment]::NewLine, $Utf8)
}
function Assert-SafeFile([string]$p, [bool]$deletion = $false) {
 $full = [System.IO.Path]::GetFullPath($p)
 if (-not $full.StartsWith($Repo + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $full" }
 $resolved = (Resolve-Path -LiteralPath $full).ProviderPath
 if (-not $resolved.Equals($full, [StringComparison]::OrdinalIgnoreCase)) { throw "Resolved path differs: $full => $resolved" }
 $fi = Get-Item -LiteralPath $resolved -Force
 if ($fi.PSIsContainer) { throw "Not a regular file: $resolved" }
 $cursor = $full
 while ($null -ne $cursor -and $cursor.Length -ge 3) {
  $component = Get-Item -LiteralPath $cursor -Force
  if (($component.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse point is prohibited: $cursor" }
  $parent = [System.IO.Directory]::GetParent($cursor)
  if ($null -eq $parent) { break }
  $cursor = $parent.FullName
 }
 if ($deletion) {
  if ($DeletePaths -notcontains $full) { throw "Not in exact deletion whitelist: $full" }
  if ($AllowedDirs -notcontains $fi.DirectoryName) { throw "Not in approved checkpoint directory: $full" }
  if ($fi.Extension -ne '.pt') { throw "Unexpected deletion extension: $full" }
 }
 return $fi
}
function Add-Expected($entry, [string]$source) {
 $p = To-Windows ([string]$entry.path)
 if (-not $p.StartsWith($Repo + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Unexpected metadata path: $p" }
 $sha = [string]$entry.sha256
 if ($sha -notmatch '^[a-f0-9]{64}$') { throw "No valid SHA metadata: $p" }
 $bytes = if ($entry.PSObject.Properties.Name -contains 'bytes') { [long]$entry.bytes } else { [long](Get-Item -LiteralPath $p).Length }
 if ($Metadata.ContainsKey($p)) {
  if ($Metadata[$p].sha256 -ne $sha -or $Metadata[$p].bytes -ne $bytes) { throw "Metadata conflict: $p" }
 } else {
  $Metadata[$p] = [pscustomobject]@{path=$p;bytes=$bytes;sha256=$sha;metadata_source=$source;original_metadata=$entry}
 }
}
if ($Mode -eq 'Verify') {
 if (Test-Path -LiteralPath (Join-Path $Out 'precheck_complete.json')) { throw 'Precheck already exists; do not silently overwrite.' }
 [void](New-Item -ItemType Directory -Path $Out -Force)
 [void](New-Item -ItemType Directory -Path (Join-Path $Out 'retired') -Force)
 Append-Event ([ordered]@{event='VERIFY_BEGIN';utc=[DateTime]::UtcNow.ToString('o');pid=$PID;model_loads=0;model_forwards=0;training_updates=0})
 try {
  foreach ($ex in $Experiments) {
   $trainDir = $Repo + '\experiments\' + $ex + '\training'
   foreach ($armDir in (Get-ChildItem -LiteralPath $trainDir -Directory)) {
    $historyPath = Join-Path $armDir.FullName 'checkpoint_history.jsonl'
    if (-not (Test-Path -LiteralPath $historyPath)) { continue }
    $latest = @{}
    foreach ($line in (Get-Content -LiteralPath $historyPath -Encoding UTF8)) {
     $record = $line | ConvertFrom-Json
     $latest[$record.path] = $record
    }
    foreach ($record in $latest.Values) { Add-Expected $record $historyPath }
    $statusPath = Join-Path $armDir.FullName 'status.json'
    $status = Read-Json $statusPath
    if ($status.status -ne 'COMPLETE') { throw "Branch not complete: $statusPath" }
    foreach ($field in @('inference_export','occupancy_export','pointer_export')) {
     if (($status.PSObject.Properties.Name -contains $field) -and $null -ne $status.$field) { Add-Expected $status.$field $statusPath }
    }
   }
   foreach ($sub in @('checkpoints','inference')) {
    foreach ($file in (Get-ChildItem -LiteralPath ($Repo + '\experiments\' + $ex + '\' + $sub) -File -Filter '*.pt')) {
     if (-not $Metadata.ContainsKey($file.FullName)) { throw "Unregistered recent weight: $($file.FullName)" }
    }
   }
  }
  foreach ($arm in @('CONTEXT','MLP')) {
   $dir = $Repo + '\experiments\r2_owner_layout_v1\heads\' + $arm
   $mPath = Join-Path $dir 'complete.json'
   $m = Read-Json $mPath
   Add-Expected ([pscustomobject]@{path=(Join-Path $dir 'step2000.pt');sha256=$m.checkpoint_sha256}) $mPath
  }
  $maintenance = $Repo + '\experiments\r2_full_seat_learning_curve_v1\cleanup\retained_verified.csv'
  $corePaths = @(
   ($Repo + '\experiments\r2_postprocess_avmr_pmr_v1\assets\chair_denoiser_0030000.pt'),
   ($Repo + '\experiments\r2_user_data_scaling_32_67_v1\checkpoints\USER_EXPANDED_67_step6000.pt'),
   ($Repo + '\experiments\r2_user_data_scaling_32_67_v1\inference\USER_EXPANDED_67_generator_step6000.pt'),
   ($Repo + '\native_t1\checkpoints\chair_hybrid_start_total5000.pt'),
   ($Repo + '\native_t1\runs\chair_hybrid_coupling_v1\checkpoints\OT_HYBRID_pilot1000_total5000.pt'),
   ($Repo + '\native_t1\runs\control_mvp48_v1\checkpoints\official_chair_last.pt')
  )
  $retainedOld = Import-Csv -LiteralPath $maintenance -Encoding UTF8
  foreach ($p in $corePaths) {
   $r = @($retainedOld | Where-Object path -eq $p)
   if ($r.Count -ne 1) { throw "Missing core maintenance identity: $p" }
   Add-Expected $r[0] $maintenance
  }
  $regPath = $Repo + '\experiments\r2_seam_slot_c200_v1\registration.json'
  $reg = Read-Json $regPath
  Add-Expected ([pscustomobject]@{path=$reg.source_checkpoint;sha256=$reg.source_checkpoint_sha256}) $regPath
  $fullInferPath = $Repo + '\experiments\r2_full_seat_learning_curve_v1\inference\FULL_S6000_generator.json'
  Add-Expected (Read-Json $fullInferPath) $fullInferPath
  if ($Metadata.Count -ne 52) { throw "Expected 52 checked weight identities, got $($Metadata.Count)" }
  $targetSum = [long]0
  foreach ($p in $DeletePaths) {
   $fi = Assert-SafeFile $p $true
   if (-not $Metadata.ContainsKey($p)) { throw "Deletion has no historical identity: $p" }
   if ($fi.Length -ne $Metadata[$p].bytes) { throw "Deletion size mismatch: $p" }
   $targetSum += $fi.Length
  }
  if ($DeletePaths.Count -ne 16 -or $targetSum -ne $ExpectedDeleteBytes) { throw 'Deletion count/bytes mismatch.' }
  $checks = @()
  foreach ($p in @($Metadata.Keys | Sort-Object)) {
   $fi = Assert-SafeFile $p ($DeletePaths -contains $p)
   $m = $Metadata[$p]
   if ($fi.Length -ne $m.bytes) { throw "Weight size mismatch: $p" }
   $sha = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()
   if ($sha -ne $m.sha256) { throw "Weight SHA mismatch: $p" }
   $kind = if ($DeletePaths -contains $p) { 'DELETE_WHITELIST' } else { 'PROTECTED_RETAIN' }
   $r = [pscustomobject]@{path=$p;bytes=$fi.Length;sha256=$sha;expected_sha256=$m.sha256;metadata_source=$m.metadata_source;kind=$kind;last_write_utc_ticks=$fi.LastWriteTimeUtc.Ticks;status='VERIFIED_BEFORE_DELETE'}
   $checks += $r
   Append-Event ([ordered]@{event='WEIGHT_HASH_VERIFIED';utc=[DateTime]::UtcNow.ToString('o');path=$p;bytes=$fi.Length;sha256=$sha;kind=$kind})
   Write-Output ('VERIFIED ' + $kind + ' ' + $fi.Name)
  }
  $retained = @($checks | Where-Object kind -eq 'PROTECTED_RETAIN')
  $deleting = @($checks | Where-Object kind -eq 'DELETE_WHITELIST')
  if ($retained.Count -ne 36 -or $deleting.Count -ne 16) { throw 'Protected/deletion cardinality mismatch.' }
  foreach ($d in $deleting) {
   if ($retained.sha256 -contains $d.sha256) { throw "Deletion content aliases a protected identity: $($d.path)" }
  }
  $endpointProof = @()
  foreach ($pair in @(
   @('r2_seam_quality_four_v1','LOCAL_NOISE'),
   @('r2_seam_quality_four_v1','ANTI_COVER'),
   @('r2_seam_abc_v1','A_COUPLE'),
   @('r2_seam_abc_v1','C_SHARED_TIP')
  )) {
   $cp = $Repo + '\experiments\' + $pair[0] + '\checkpoints\' + $pair[1] + '_step1000.pt'
   $ip = $Repo + '\experiments\' + $pair[0] + '\inference\' + $pair[1] + '_generator.pt'
   $cm = $Metadata[$cp].original_metadata
   $im = $Metadata[$ip].original_metadata
   if ($cm.generator_state_sha256 -ne $im.generator_state_sha256) { throw "Endpoint/inference state identity mismatch: $cp" }
   if ($retained.path -notcontains $ip) { throw "Missing retained inference: $ip" }
   $endpointProof += [pscustomobject]@{checkpoint=$cp;retained_generator=$ip;generator_state_sha256=$cm.generator_state_sha256;inference_sha256=$Metadata[$ip].sha256;status='ACTUAL_FILE_HASH_AND_RECORDED_GENERATOR_STATE_MATCH'}
  }
  Save-Csv 'planned_deleted_weights.csv' $deleting
  Save-Csv 'retained_verified_before.csv' $retained
  Write-Json 'deletion_manifest.json' ([ordered]@{schema='r2_wave_cleanup_v1';authorization_prompt=$PromptPath;authorization_prompt_sha256=(Get-FileHash -LiteralPath $PromptPath -Algorithm SHA256).Hash.ToLowerInvariant();expected_count=16;expected_bytes=$ExpectedDeleteBytes;paths=$DeletePaths;files=$deleting;protected=$retained;endpoint_inference_proof=$endpointProof;original_metadata=@($deleting | ForEach-Object {$Metadata[$_.path]});scope='Only the exact16 checkpoint paths. No directory recursion, no inference/data/source deletion.'})
  Copy-Item -LiteralPath $PromptPath -Destination (Join-Path $Out 'authorized_task_prompt.txt')
  $manifestSha=(Get-FileHash -LiteralPath (Join-Path $Out 'deletion_manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant()
  Write-Json 'precheck_complete.json' ([ordered]@{status='PASS_READY_FOR_EXPLICIT_DELETE_STAGE';utc=[DateTime]::UtcNow.ToString('o');manifest_sha256=$manifestSha;deleted_count=0;verified_delete_targets=16;verified_protected_weights=36;expected_delete_bytes=$ExpectedDeleteBytes;expected_delete_GiB=$ExpectedDeleteBytes/1GB;script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant();scientific_calls=0})
  Write-Output ('PRECHECK PASS; manifest SHA ' + $manifestSha)
 } catch {
  Write-Json 'verification_failure.json' ([ordered]@{status='BLOCKED_BEFORE_DELETE';utc=[DateTime]::UtcNow.ToString('o');message=$_.Exception.Message;deleted_count=0})
  throw
 }
 exit 0
}
if (-not (Test-Path -LiteralPath (Join-Path $Out 'precheck_complete.json'))) { throw 'Precheck required.' }
if (Test-Path -LiteralPath (Join-Path $Out 'delete_stage_started.json')) { throw 'Deletion stage already started; do not silently retry.' }
$pre = Read-Json (Join-Path $Out 'precheck_complete.json')
$manifestPath = Join-Path $Out 'deletion_manifest.json'
if ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pre.manifest_sha256) { throw 'Frozen deletion manifest changed.' }
if ((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pre.script_sha256) { throw 'Cleanup script changed since precheck.' }
$manifest = Read-Json $manifestPath
if ($manifest.files.Count -ne 16 -or [long]$manifest.expected_bytes -ne $ExpectedDeleteBytes) { throw 'Manifest count/budget mismatch.' }
if (@(Compare-Object $DeletePaths @($manifest.files.path)).Count -ne 0) { throw 'Manifest path set differs from hardcoded whitelist.' }
foreach ($r in @($manifest.files) + @($manifest.protected)) {
 $fi = Assert-SafeFile $r.path ($DeletePaths -contains $r.path)
 if ($fi.Length -ne [long]$r.bytes -or $fi.LastWriteTimeUtc.Ticks -ne [long]$r.last_write_utc_ticks) { throw "Weight changed since precheck: $($r.path)" }
}
$freeBefore = (New-Object System.IO.DriveInfo('E:\')).AvailableFreeSpace
Write-Json 'delete_stage_started.json' ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');manifest_sha256=$pre.manifest_sha256;free_bytes_before=$freeBefore;pid=$PID;planned_files=16;planned_bytes=$ExpectedDeleteBytes})
$deleted = @()
try {
 foreach ($r in $manifest.files) {
  $fi = Assert-SafeFile $r.path $true
  $sha = (Get-FileHash -LiteralPath $r.path -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($sha -ne $r.sha256 -or $fi.Length -ne [long]$r.bytes) { throw "Immediate deletion identity mismatch: $($r.path)" }
  $source = @($manifest.original_metadata | Where-Object path -eq $r.path)[0]
  $name = $fi.Directory.Parent.Name + '__' + $fi.Name + '.retired.json'
  Write-Json ('retired\' + $name) ([ordered]@{status='AUTHORIZED_RETIREMENT_PENDING_DELETE';path=$r.path;bytes=$r.bytes;sha256=$sha;metadata=$source;manifest_sha256=$pre.manifest_sha256;retired_capability='Exact historical checkpoint resume/replay; inference exports and evidence remain protected.'})
  Append-Event ([ordered]@{event='DELETE_INTENT';utc=[DateTime]::UtcNow.ToString('o');path=$r.path;bytes=$r.bytes;sha256=$sha})
  Remove-Item -LiteralPath $r.path -Force
  if (Test-Path -LiteralPath $r.path) { throw "Deletion did not remove exact target: $($r.path)" }
  $receipt = [pscustomobject]@{path=$r.path;bytes=[long]$r.bytes;sha256_before_delete=$sha;deleted_utc=[DateTime]::UtcNow.ToString('o');status='DELETED_EXACT_WHITELIST'}
  $deleted += $receipt
  Save-Csv 'deleted_weights.csv' $deleted
  Write-Json ('retired\' + $name) ([ordered]@{status='DELETED_AND_RETIRED';path=$r.path;bytes=$r.bytes;sha256_before_delete=$sha;deleted_utc=$receipt.deleted_utc;metadata=$source;manifest_sha256=$pre.manifest_sha256;retired_capability='Exact historical checkpoint resume/replay; inference exports and evidence remain protected.'})
  Append-Event ([ordered]@{event='DELETE_DONE';utc=$receipt.deleted_utc;path=$r.path;bytes=$r.bytes;sha256=$sha})
  Write-Output ('DELETED ' + $fi.Name)
 }
 $verified = @()
 foreach ($r in $manifest.protected) {
  $fi = Assert-SafeFile $r.path $false
  $sha = (Get-FileHash -LiteralPath $r.path -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($fi.Length -ne [long]$r.bytes -or $sha -ne $r.sha256) { throw "Protected post-delete mismatch: $($r.path)" }
  $verified += [pscustomobject]@{path=$r.path;bytes=$fi.Length;sha256_before=$r.sha256;sha256_after=$sha;metadata_source=$r.metadata_source;status='PROTECTED_SHA256_UNCHANGED'}
  Append-Event ([ordered]@{event='PROTECTED_POST_HASH_VERIFIED';utc=[DateTime]::UtcNow.ToString('o');path=$r.path;bytes=$fi.Length;sha256=$sha})
 }
 Save-Csv 'retained_verified.csv' $verified
 $freeAfter = (New-Object System.IO.DriveInfo('E:\')).AvailableFreeSpace
 $deletedSum = [long]($deleted | Measure-Object bytes -Sum).Sum
 if ($deleted.Count -ne 16 -or $deletedSum -ne $ExpectedDeleteBytes -or $verified.Count -ne 36) { throw 'Final count/bytes mismatch.' }
 $retirement = @'
# Exact checkpoint retirement
The user-authorized stage0 retired only the16 paths in deletion_manifest.json.
The rolling files were older states (C200 local150; quality-four/ABC local750; pool47 continuation local2900), not byte-identical final checkpoints.
COUPLE local1000/2000 and the complete LOCAL_NOISE, ANTI_COVER, A_COUPLE, C_SHARED_TIP endpoints no longer support exact AdamW/RNG resume from those deleted states.
All inference generator exports, occupancy/pointer heads, core B/LONG/PLAIN1000/2000/3000/COUPLE3000/SEAM200/FULL4 states and protected START/EXP67/FULL/official Chair/D0 assets remain.
Original reports and manifests remain historical records and were not rewritten. Old all-files verification or recovery entrypoints requiring retired paths may fail; do not silently reconstruct or retrain them.
RAW/POST, scores, images, data, noise, source code and default-model settings were untouched. No model load, forward, backward, training, sampling, commit or push occurred in this cleanup.
Logical deleted bytes are authoritative; whole-drive free-space change can include concurrent external filesystem activity.
'@
 [System.IO.File]::WriteAllText((Join-Path $Out 'CHECKPOINT_RETIREMENT.md'), $retirement, $Utf8)
 Write-Json 'completion.json' ([ordered]@{status='COMPLETE';stage='0_CLEANUP';utc=[DateTime]::UtcNow.ToString('o');deleted_files=$deleted.Count;deleted_bytes=$deletedSum;deleted_GiB=$deletedSum/1GB;retained_verified_files=$verified.Count;retained_sha256_unchanged=$true;free_bytes_before=$freeBefore;free_bytes_after=$freeAfter;observed_free_delta_bytes=$freeAfter-$freeBefore;manifest_sha256=$pre.manifest_sha256;deleted_csv_sha256=(Get-FileHash -LiteralPath (Join-Path $Out 'deleted_weights.csv') -Algorithm SHA256).Hash.ToLowerInvariant();retained_csv_sha256=(Get-FileHash -LiteralPath (Join-Path $Out 'retained_verified.csv') -Algorithm SHA256).Hash.ToLowerInvariant();script_sha256=$pre.script_sha256;model_loads=0;model_forwards=0;training_updates=0;other_source_or_asset_edits=0;git_commit_or_push=$false})
 Append-Event ([ordered]@{event='CLEANUP_COMPLETE';utc=[DateTime]::UtcNow.ToString('o');deleted_files=16;deleted_bytes=$deletedSum;protected_verified=36})
 Write-Output ('COMPLETE: deleted16, bytes=' + $deletedSum + ', protected36 SHA unchanged.')
} catch {
 Write-Json 'delete_stage_failure.json' ([ordered]@{status='STOPPED_NO_AUTOMATIC_RETRY';utc=[DateTime]::UtcNow.ToString('o');message=$_.Exception.Message;deleted_so_far=$deleted.Count;bytes_so_far=($deleted | Measure-Object bytes -Sum).Sum})
 throw
}

