param([ValidateSet('Prepare','Workspace','External','Verify')][string]$Stage='Prepare')
$ErrorActionPreference='Stop'
$root='E:\MeshFlow'
$external='E:\MeshFlow-Control-artifacts'
$out=Join-Path $root 'maintenance\storage_cleanup_20261025'
$utf8=[System.Text.UTF8Encoding]::new($false)
function Read-Json([string]$p){Get-Content -LiteralPath $p -Encoding UTF8 -Raw | ConvertFrom-Json}
function Save-Json([string]$p,$v){[IO.File]::WriteAllText($p,($v | ConvertTo-Json -Depth 20),$utf8)}
function Normal([string]$p){
 if($p.StartsWith('/mnt/e/')){$p='E:\'+$p.Substring(7)}
 if(-not [IO.Path]::IsPathRooted($p)){$p=Join-Path $root $p}
 [IO.Path]::GetFullPath($p.Replace('/','\'))
}
function Safe-File([string]$p){
 $full=Normal $p
 $anchor=if($full.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)){$root}elseif($full.StartsWith($external+'\',[StringComparison]::OrdinalIgnoreCase)){$external}else{throw "Outside audited roots: $full"}
 if($full -match '\\\.(git|agents|codex|aws)\\'){throw "Protected metadata path: $full"}
 $cur=Get-Item -LiteralPath $full -Force
 if($cur -isnot [IO.FileInfo]){throw "Not a regular file: $full"}
 while($cur){
  if(($cur.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw "Reparse path refused: $full"}
  if($cur.FullName.TrimEnd('\') -eq $anchor){break}
  $cur=if($cur -is [IO.FileInfo]){$cur.Directory}else{$cur.Parent}
 }
 if($null -eq $cur){throw "Unexpected ancestry: $full"}
 $full
}
function Hash([string]$p){(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
function Space(){[IO.DriveInfo]::new('E:\').AvailableFreeSpace}
function Ledger($v){[IO.File]::AppendAllText(($out+'\deletion_ledger.jsonl'),(($v|ConvertTo-Json -Compress)+[Environment]::NewLine),$utf8)}
if($Stage -eq 'Prepare'){
 if(Test-Path -LiteralPath ($out+'\prepared_manifest.json')){throw 'Already prepared'}
 $chair=Read-Json ($out+'\chair_audit.json')
 $bench=Import-Csv -LiteralPath ($out+'\backend_scope_checkpoint_decisions.csv')
 $package=Read-Json ($out+'\dataset_archived_package_whitelist.json')
 $deps=Read-Json ($out+'\dataset_active_dependencies.json')
 $remove=@{};$protected=@{};$keepReasons=@{}
 $multipart=Normal 'experiments/r2_chair_multipart_seam_v1/training/checkpoint_step1500.pt'
 foreach($r in $chair.rows){
  $p=Normal $r.path
  if($r.action -eq 'RETIRE_OBSOLETE_WEIGHT' -and $p -ne $multipart){$remove[$p]=[pscustomobject]@{Path=$p;Bytes=[long]$r.bytes;Reason=$r.reason;Kind='RETIRED_WEIGHT';ExpectedHash=$null}}
  elseif($r.action -like 'KEEP*' -or $p -eq $multipart){$keepReasons[$p]=if($p -eq $multipart){'Retain complete recovery for distinct Chair BACK positive'}else{$r.reason}}
 }
 foreach($r in $bench){
  $p=Normal $r.path
  if($r.action -eq 'DELETE_CANDIDATE'){$remove[$p]=[pscustomobject]@{Path=$p;Bytes=[long]$r.bytes;Reason=$r.reason;Kind='RETIRED_WEIGHT';ExpectedHash=$null}}
  elseif($r.action -eq 'KEEP'){$keepReasons[$p]=$r.reason}
 }
 foreach($r in $package.files){
  $p=Normal $r.path
  if(-not $p.StartsWith($root+'\experiments\r2_single_step_denoiser_distillation_v1\review_package\',[StringComparison]::OrdinalIgnoreCase)){throw 'Unexpected expanded package candidate'}
  $remove[$p]=[pscustomobject]@{Path=$p;Bytes=[long]$r.bytes;Reason=('Byte-identical expanded review copy; preserved ZIP member '+$r.archive_member);Kind='ARCHIVED_DUPLICATE';ExpectedHash=$r.sha256}
 }
 $keepReasons[(Normal $package.archive)]='Verified lossless original review ZIP'
 foreach($r in $package.keep_expanded_metadata){$p=if($r -is [string]){$r}else{$r.path};$keepReasons[(Normal $p)]='Required historical archive verification manifest'}
 # All large files outside the removal list stay protected, including third-party teacher assets.
 foreach($r in (Import-Csv -LiteralPath ($out+'\weight_inventory.csv'))){$p=Normal $r.Path;if(-not $remove.ContainsKey($p) -and [long]$r.Bytes -gt 1MB -and -not $keepReasons.ContainsKey($p)){$keepReasons[$p]='Protected remaining official asset or small fixture; not retired'}}
 $keepReasons[(Normal 'deliverables/R2_BENCH_RAW_LOCAL_REPAIR_V1_REVIEW_SMALL.zip')]='Verified current experiment review archive'
 foreach($r in $deps.manifests){$keepReasons[(Normal $r.path)]='Active dataset or baseline manifest'}
 # Preserve original top-level reports and completion records in every affected experiment.
 $experimentRoots=@{}
 foreach($p in $remove.Keys){
  if($p -match '^(E:\\MeshFlow(?:-Control-artifacts)?\\experiments\\[^\\]+)\\'){$experimentRoots[$Matches[1]]=$true}
 }
 foreach($d in $experimentRoots.Keys){foreach($f in Get-ChildItem -LiteralPath $d -File){if($f.Extension -in @('.md','.json','.csv','.diff')){$keepReasons[$f.FullName]='Original experiment evidence; immutable during weight retirement'}}}
 foreach($p in $keepReasons.Keys){if($remove.ContainsKey($p)){throw "Keep/delete overlap: $p"}}
 $deleteRows=[Collections.Generic.List[object]]::new();$i=0
 foreach($r in ($remove.Values | Sort-Object Path)){
  $p=Safe-File $r.Path;$f=Get-Item -LiteralPath $p
  if($f.Length -ne $r.Bytes){throw "Size changed: $p"}
  $h=Hash $p;if($r.ExpectedHash -and $h -ne $r.ExpectedHash){throw "Duplicate hash mismatch: $p"}
  $deleteRows.Add([pscustomobject]@{Path=$p;Bytes=$f.Length;SHA256=$h;LastWriteUTC=$f.LastWriteTimeUtc.ToString('o');Kind=$r.Kind;Reason=$r.Reason;Scope=$(if($p.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)){'Workspace'}else{'External'})})
  $i++;if($i%25 -eq 0){Write-Output "Hashed removal candidates $i / $($remove.Count)"}
 }
 $keepRows=[Collections.Generic.List[object]]::new();$i=0
 foreach($p in ($keepReasons.Keys | Sort-Object)){
  $p=Safe-File $p;$f=Get-Item -LiteralPath $p
  $keepRows.Add([pscustomobject]@{Path=$p;Bytes=$f.Length;SHA256=(Hash $p);Reason=$keepReasons[$p]})
  $i++;if($i%50 -eq 0){Write-Output "Hashed protected assets $i / $($keepReasons.Count)"}
 }
 $archiveRow=@($keepRows|Where-Object Path -eq (Normal $package.archive))[0]
 if($archiveRow.SHA256 -ne $package.archive_sha256){throw 'Retained ZIP mismatch'}
 $manifest=[ordered]@{Schema='EXACT_FILE_RETIREMENT_V1';CreatedUTC=[DateTime]::UtcNow.ToString('o');Authorization='Latest user request to archive current round and delete obsolete weights/datasets, retain best setups';Root=$root;ExternalRoot=$external;Remove=@($deleteRows);Keep=@($keepRows);DeleteFiles=$deleteRows.Count;DeleteBytes=($deleteRows|Measure-Object Bytes -Sum).Sum;FreeBytesAtPrepare=(Space);ExperimentRoots=@($experimentRoots.Keys|Sort-Object);NoTraining=$true;NoDefaultChange=$true;NoOriginalDatasetDeletion=$true;RawPostEvidenceRetained=$true;Override='Keep MULTIPART_S1500 complete resume as well as its inference/head'}
 Save-Json ($out+'\prepared_manifest.json') $manifest
 [IO.File]::WriteAllText(($out+'\prepared_manifest.sha256'),(Hash ($out+'\prepared_manifest.json')),$utf8)
 $deleteRows | Export-Csv -LiteralPath ($out+'\deletion_plan.csv') -NoTypeInformation -Encoding UTF8
 $keepRows | Export-Csv -LiteralPath ($out+'\protected_assets.csv') -NoTypeInformation -Encoding UTF8
 [pscustomobject]@{Status='PREPARED';Files=$deleteRows.Count;GiB=$manifest.DeleteBytes/1GB;Protected=$keepRows.Count} | ConvertTo-Json
 exit
}
$planPath=$out+'\prepared_manifest.json'
if((Hash $planPath) -ne [IO.File]::ReadAllText($out+'\prepared_manifest.sha256').Trim()){throw 'Plan changed'}
$plan=Read-Json $planPath
if($Stage -eq 'Verify'){
 $ledger=@(Get-Content -LiteralPath ($out+'\deletion_ledger.jsonl') -Encoding UTF8 | ForEach-Object {$_|ConvertFrom-Json})
 if($ledger.Count -ne $plan.Remove.Count){throw 'Ledger count mismatch'}
 foreach($r in $plan.Remove){if(Test-Path -LiteralPath $r.Path){throw "Removed path exists: $($r.Path)"};$entry=@($ledger|Where-Object Path -eq $r.Path);if($entry.Count -ne 1 -or $entry[0].SHA256 -ne $r.SHA256){throw 'Ledger mismatch'}}
 $i=0;foreach($r in $plan.Keep){$p=Safe-File $r.Path;if((Hash $p) -ne $r.SHA256){throw "Protected asset changed: $p"};$i++;if($i%100 -eq 0){Write-Output "Verified protected assets $i / $($plan.Keep.Count)"}}
 $free=Space
 $receipt=[ordered]@{Status='COMPLETE_VERIFIED';UTC=[DateTime]::UtcNow.ToString('o');DeletedFiles=$ledger.Count;LogicalBytes=($ledger|Measure-Object Bytes -Sum).Sum;LogicalGiB=(($ledger|Measure-Object Bytes -Sum).Sum/1GB);FreeBytesBefore=$plan.FreeBytesAtPrepare;FreeBytesAfter=$free;ObservedFreeDeltaBytes=$free-$plan.FreeBytesAtPrepare;ObservedFreeDeltaGiB=($free-$plan.FreeBytesAtPrepare)/1GB;ProtectedHashChecks=$plan.Keep.Count;PreparedManifestSHA256=(Hash $planPath);LedgerSHA256=(Hash ($out+'\deletion_ledger.jsonl'));NoTraining=$true;NoGeneration=$true;NoOriginalDatasetDeletion=$true;NoDefaultChange=$true;NoGitCommitPush=$true;PhysicalSpaceCaveat='Observed volume free-space delta can include other processes. Logical sizes include hardlinked paths; do not equate with unique physical allocation.'}
 Save-Json ($out+'\cleanup_receipt.json') $receipt
 $receipt|ConvertTo-Json
 exit
}
$scopeRows=@($plan.Remove|Where-Object Scope -eq $Stage)
if($scopeRows.Count -eq 0){throw "No rows for scope $Stage"}
$previous=@{}
if(Test-Path -LiteralPath ($out+'\deletion_ledger.jsonl')){foreach($line in Get-Content -LiteralPath ($out+'\deletion_ledger.jsonl') -Encoding UTF8){$r=$line|ConvertFrom-Json;$previous[$r.Path]=$r}}
# Only a recorded, verified deletion can be skipped on recovery.
$start=Space;$i=0
foreach($r in $scopeRows){
 if($previous.ContainsKey($r.Path)){if((Test-Path -LiteralPath $r.Path) -or $previous[$r.Path].SHA256 -ne $r.SHA256){throw 'Recovery ledger mismatch'};continue}
 $p=Safe-File $r.Path
 if($Stage -eq 'External' -and -not $p.StartsWith($external+'\experiments\',[StringComparison]::OrdinalIgnoreCase)){throw 'External scope mismatch'}
 $f=Get-Item -LiteralPath $p
 if($f.Length -ne $r.Bytes -or (Hash $p) -ne $r.SHA256){throw "Candidate changed before deletion: $p"}
 Remove-Item -LiteralPath $p -Force
 if(Test-Path -LiteralPath $p){throw "Deletion failed: $p"}
 Ledger ([ordered]@{UTC=[DateTime]::UtcNow.ToString('o');Path=$p;Bytes=$r.Bytes;SHA256=$r.SHA256;Kind=$r.Kind;Scope=$Stage;Reason=$r.Reason})
 $i++;if($i%25 -eq 0){Write-Output "Deleted verified $Stage files: $i"}
}
Save-Json ($out+"\scope_${Stage}_receipt.json") ([ordered]@{UTC=[DateTime]::UtcNow.ToString('o');Status='SCOPE_COMPLETE';Scope=$Stage;NewDeletions=$i;ListedFiles=$scopeRows.Count;FreeBefore=$start;FreeAfter=(Space)})
Write-Output "Scope $Stage complete: $i new deletions"
