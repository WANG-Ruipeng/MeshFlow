param([ValidateSet('prepare','execute')][string]$Action='prepare')
$ErrorActionPreference='Stop'
$workspaceRoot=[IO.Path]::GetFullPath('E:\MeshFlow')
$auditRoot=Join-Path $workspaceRoot 'maintenance\loop_motif_closeout_20261029'
New-Item -ItemType Directory -Force -Path $auditRoot | Out-Null
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class MeshCleanupSize {
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
 static extern uint GetCompressedFileSizeW(string path, out uint high);
 public static ulong Allocated(string path) {
  uint high; uint low=GetCompressedFileSizeW(path,out high);
  if(low==0xffffffff && Marshal.GetLastWin32Error()!=0) throw new System.ComponentModel.Win32Exception();
  return ((ulong)high << 32) | low;
 }
}
'@
function SafePath([string]$p) {
 $full=[IO.Path]::GetFullPath($p)
 if(-not $full.StartsWith($workspaceRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Outside workspace: $full"}
 $item=Get-Item -LiteralPath $full
 $cursor=$item
 while($cursor.FullName -ne $workspaceRoot){
  if($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint){throw "Reparse point: $($cursor.FullName)"}
  $cursor=if($cursor -is [IO.DirectoryInfo]){$cursor.Parent}else{$cursor.Directory}
  if($null -eq $cursor){throw 'Ancestor check failed'}
 }
 return $item.FullName
}
function Record([string]$p) {
 $p=SafePath $p;$item=Get-Item -LiteralPath $p
 return [PSCustomObject]@{path=$p;bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower();allocated=[MeshCleanupSize]::Allocated($p)}
}
function JsonFile($p,$o){$o|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $p -Encoding UTF8}
$manifestPath=Join-Path $auditRoot 'prepared_manifest.json'
if($Action -eq 'prepare') {
 if(Test-Path -LiteralPath $manifestPath){throw 'Manifest already exists; inspect before another run'}
 $deleted=@();$protected=@()
 foreach($arm in @('PLAIN200','FLIP200')) {
  $base=Join-Path $workspaceRoot "experiments\r2_bench_c_edgeflip_v1\training\$arm"
  $old=Get-Content -LiteralPath (Join-Path $base 'checkpoint_latest.json') -Raw|ConvertFrom-Json
  $retired=Get-Content -LiteralPath (Join-Path $base 'rolling_checkpoint_retirement.json') -Raw|ConvertFrom-Json
  if($old.local_step -ne 150 -or $retired.status -ne 'SUPERSEDED_BY_FORMAL_ENDPOINT' -or $retired.active_checkpoint.local_step -ne 200){throw 'Checkpoint retirement identity mismatch'}
  $r=Record (Join-Path $base 'checkpoint_latest.pt')
  if($r.sha256 -ne $old.sha256){throw 'Old checkpoint hash mismatch'}
  $deleted+=[PSCustomObject]@{path=$r.path;bytes=$r.bytes;sha256=$r.sha256;allocated=$r.allocated;reason='superseded_step150_rolling_not_duplicate'}
  foreach($name in @('checkpoint_step200.pt','generator_step200.pt')){$protected+=Record (Join-Path $base $name)}
 }
 $cachePaths=@(& rg --files --no-ignore r2 native_t1 models -g '*.pyc' -g '!**/.git/**')
 foreach($p in $cachePaths){
  if($p -notmatch '(^|[\\/])__pycache__[\\/]'){continue}
  $r=Record (Join-Path $workspaceRoot $p)
  $deleted+=[PSCustomObject]@{path=$r.path;bytes=$r.bytes;sha256=$r.sha256;allocated=$r.allocated;reason='rebuildable_python_bytecode'}
 }
 $compressionRoots=@();$compress=@()
 foreach($exp in @('r2_bench_full_prior_motif_guidance_v1','r2_bench_loop_motif_mini_v1')) {
  foreach($sub in @('base_pair_caches','cpu','raw_scores')) {
   $p=SafePath (Join-Path $workspaceRoot "experiments\$exp\$sub");$compressionRoots+=$p
   foreach($f in Get-ChildItem -LiteralPath $p -File -Recurse -Filter '*.json'){$compress+=Record $f.FullName}
  }
  foreach($name in @('REPORT.md','COMPLETE.json','registration.json')){
   $p=Join-Path $workspaceRoot "experiments\$exp\$name"
   if(Test-Path -LiteralPath $p){$protected+=Record $p}
  }
  foreach($f in Get-ChildItem -LiteralPath (Join-Path $workspaceRoot "experiments\$exp") -File -Filter '*.zip'){$protected+=Record $f.FullName}
 }
 $weights=@(& rg --files --hidden experiments r2 native_t1 checkpoints output -g '*.pt' -g '*.pth' -g '*.safetensors' -g '!**/.git/**' | ForEach-Object {Get-Item -LiteralPath $_} | ForEach-Object {[PSCustomObject]@{path=$_.FullName;bytes=$_.Length;mtime=$_.LastWriteTimeUtc.Ticks}})
 JsonFile $manifestPath ([PSCustomObject]@{utc=[DateTime]::UtcNow.ToString('o');scope='two superseded rolling checkpoints, bytecode, lossless compression of sealed motif JSON caches';delete=$deleted;protect=$protected;weights=$weights;compression_roots=$compressionRoots;compress=$compress})
 [PSCustomObject]@{delete_files=$deleted.Count;delete_GiB=($deleted|Measure-Object bytes -Sum).Sum/1GB;compress_files=$compress.Count;compress_logical_GiB=($compress|Measure-Object bytes -Sum).Sum/1GB;protected=$protected.Count}|ConvertTo-Json
 exit
}
if(Test-Path -LiteralPath (Join-Path $auditRoot 'cleanup_receipt.json')){throw 'Cleanup already sealed'}
$plan=Get-Content -LiteralPath $manifestPath -Raw|ConvertFrom-Json
$diskBefore=(Get-PSDrive E).Free
foreach($p in $plan.protect){if((Record $p.path).sha256 -ne $p.sha256){throw 'Protected asset changed before cleanup'}}
foreach($p in $plan.delete){
 $current=Record $p.path
 if($current.sha256 -ne $p.sha256 -or $current.bytes -ne $p.bytes){throw "Deletion target changed: $($p.path)"}
}
foreach($p in $plan.delete){
 $full=SafePath $p.path
 Remove-Item -LiteralPath $full -Force
 [PSCustomObject]@{utc=[DateTime]::UtcNow.ToString('o');path=$full;bytes=$p.bytes;sha256=$p.sha256;reason=$p.reason;action='DELETED'}|ConvertTo-Json -Compress|Add-Content -LiteralPath (Join-Path $auditRoot 'deletion_ledger.jsonl') -Encoding UTF8
}
$i=0
foreach($p in $plan.compression_roots){
 $full=SafePath $p
 & compact.exe /C /I /Q "/S:$full" '*.json' *> (Join-Path $auditRoot "compact_$i.log")
 if($LASTEXITCODE -ne 0){throw "Compression command failed: $full"}
 $i++
}
$compressedAfter=0L
foreach($p in $plan.compress){
 $r=Record $p.path
 if($r.sha256 -ne $p.sha256 -or $r.bytes -ne $p.bytes){throw "Lossless compression validation failed: $($p.path)"}
 $compressedAfter+=$r.allocated
 [PSCustomObject]@{path=$p.path;sha256=$r.sha256;bytes=$r.bytes;allocated_before=$p.allocated;allocated_after=$r.allocated}|ConvertTo-Json -Compress|Add-Content -LiteralPath (Join-Path $auditRoot 'compression_ledger.jsonl') -Encoding UTF8
}
foreach($p in $plan.protect){if((Record $p.path).sha256 -ne $p.sha256){throw 'Protected asset changed after cleanup'}}
$deleteNames=@($plan.delete.path)
foreach($p in $plan.weights){
 if($p.path -in $deleteNames){continue}
 $item=Get-Item -LiteralPath $p.path
 if($item.Length -ne $p.bytes -or $item.LastWriteTimeUtc.Ticks -ne $p.mtime){throw "Unrelated weight changed: $($p.path)"}
}
$r=[PSCustomObject]@{status='COMPLETE_VERIFIED';utc=[DateTime]::UtcNow.ToString('o');deleted_files=$plan.delete.Count;deleted_bytes=($plan.delete|Measure-Object bytes -Sum).Sum;deleted_allocated_bytes=($plan.delete|Measure-Object allocated -Sum).Sum;compressed_files=$plan.compress.Count;compressed_logical_bytes=($plan.compress|Measure-Object bytes -Sum).Sum;compression_allocated_before=($plan.compress|Measure-Object allocated -Sum).Sum;compression_allocated_after=$compressedAfter;compression_saved_bytes=(($plan.compress|Measure-Object allocated -Sum).Sum-$compressedAfter);protected_hashes_pass=$plan.protect.Count;other_weight_metadata_pass=$plan.weights.Count-2;disk_free_before=$diskBefore;disk_free_after=(Get-PSDrive E).Free;training=0;model_calls=0;source_dataset_deletion=0;RAW_POST_deletion=0;scoring_evidence_deletion=0}
JsonFile (Join-Path $auditRoot 'cleanup_receipt.json') $r
$r|ConvertTo-Json
