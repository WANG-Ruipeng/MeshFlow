param([ValidateSet('Prepare','Execute')][string]$Stage='Prepare')
$ErrorActionPreference='Stop'
$root='E:\MeshFlow'
$out=$root+'\maintenance\storage_cleanup_20261025'
$utf8=[Text.UTF8Encoding]::new($false)
function ReadJson($p){Get-Content -LiteralPath $p -Raw -Encoding UTF8 | ConvertFrom-Json}
function Hash($p){(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
function Save($p,$v){[IO.File]::WriteAllText($p,($v|ConvertTo-Json -Depth 15),$utf8)}
function Safe($p){
 $p=[IO.Path]::GetFullPath($p.Replace('/','\'))
 $ok=$false
 foreach($allowed in @('E:\MeshFlow\.worktrees\a-layout-value-20260924\output\','E:\MeshFlow\.worktrees\ponq-gate1-20260926\output\')){if($p.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){$ok=$true}}
 if(-not $ok -or [IO.Path]::GetExtension($p) -ne '.pt'){throw "Outside exact legacy scope: $p"}
 $f=Get-Item -LiteralPath $p -Force
 if($f -isnot [IO.FileInfo]){throw 'Not file'}
 $c=$f
 while($c -and $c.FullName -ne $root){if(($c.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw 'Reparse path'};$c=if($c -is [IO.FileInfo]){$c.Directory}else{$c.Parent}}
 if($null -eq $c){throw 'Bad ancestry'}
 return $p
}
$main=ReadJson ($out+'\prepared_manifest.json')
if($Stage -eq 'Prepare'){
 if(Test-Path -LiteralPath ($out+'\legacy_small_plan.json')){throw 'Already prepared'}
 $audit=ReadJson ($out+'\legacy_small_weights_audit.json')
 $known=@{};foreach($r in $main.Keep){$known[$r.Path]=$r}
 $rows=[Collections.Generic.List[object]]::new()
 foreach($r in $audit.rows){
  if($r.action -ne 'RETIRE_OBSOLETE_WEIGHT'){continue}
  $p=Safe $r.path;$f=Get-Item -LiteralPath $p
  if(-not $known.ContainsKey($p)){throw "Not pre-hashed: $p"}
  if($f.Length -ne [long]$r.bytes -or $f.Length -ge 100MB -or $f.Length -le 1MB){throw "Unexpected size: $p"}
  $h=Hash $p;if($h -ne $known[$p].SHA256){throw "Hash changed: $p"}
  $rows.Add([pscustomobject]@{Path=$p;Bytes=$f.Length;SHA256=$h;Reason=$r.reason})
 }
 if($rows.Count -eq 0){throw 'No audited removals'}
 Save ($out+'\legacy_small_plan.json') ([ordered]@{Schema='LEGACY_SMALL_SUPPLEMENT_V1';CreatedUTC=[DateTime]::UtcNow.ToString('o');Remove=@($rows);Supersedes='Only these first-pass generic KEEP entries; official/current retained models unchanged';AuditSHA256=(Hash ($out+'\legacy_small_weights_audit.json'));FreeBefore=[IO.DriveInfo]::new('E:\').AvailableFreeSpace})
 [IO.File]::WriteAllText(($out+'\legacy_small_plan.sha256'),(Hash ($out+'\legacy_small_plan.json')),$utf8)
 [pscustomobject]@{Status='PREPARED';Files=$rows.Count;GiB=($rows|Measure-Object Bytes -Sum).Sum/1GB}|ConvertTo-Json
 exit
}
$pplan=$out+'\legacy_small_plan.json'
if((Hash $pplan) -ne [IO.File]::ReadAllText($out+'\legacy_small_plan.sha256').Trim()){throw 'Plan changed'}
$plan=ReadJson $pplan
if(Test-Path -LiteralPath ($out+'\legacy_small_receipt.json')){throw 'Already complete'}
$prior=@{}
if(Test-Path -LiteralPath ($out+'\legacy_small_deletion_ledger.jsonl')){foreach($line in Get-Content -LiteralPath ($out+'\legacy_small_deletion_ledger.jsonl')){$r=$line|ConvertFrom-Json;$prior[$r.Path]=$r}}
foreach($r in $plan.Remove){
 if($prior.ContainsKey($r.Path)){if((Test-Path -LiteralPath $r.Path) -or $prior[$r.Path].SHA256 -ne $r.SHA256){throw 'Recovery mismatch'};continue}
 $p=Safe $r.Path
 if((Hash $p) -ne $r.SHA256){throw "Changed before delete: $p"}
 Remove-Item -LiteralPath $p -Force
 if(Test-Path -LiteralPath $p){throw 'Deletion failed'}
 [IO.File]::AppendAllText(($out+'\legacy_small_deletion_ledger.jsonl'),(([ordered]@{UTC=[DateTime]::UtcNow.ToString('o');Path=$p;Bytes=$r.Bytes;SHA256=$r.SHA256;Reason=$r.Reason}|ConvertTo-Json -Compress)+[Environment]::NewLine),$utf8)
}
$removed=@{};foreach($r in $plan.Remove){$removed[$r.Path]=$true}
$count=0
foreach($r in $main.Keep){if($removed.ContainsKey($r.Path)){continue};if(-not (Test-Path -LiteralPath $r.Path) -or (Get-Item -LiteralPath $r.Path).Length -ne $r.Bytes){throw "Preserved file missing/changed: $($r.Path)"};$count++}
$free=[IO.DriveInfo]::new('E:\').AvailableFreeSpace
$result=[ordered]@{Status='COMPLETE_VERIFIED';UTC=[DateTime]::UtcNow.ToString('o');DeletedFiles=$plan.Remove.Count;LogicalBytes=($plan.Remove|Measure-Object Bytes -Sum).Sum;FreeBefore=$plan.FreeBefore;FreeAfter=$free;ObservedFreeDeltaBytes=$free-$plan.FreeBefore;RetainedExistenceSizeChecks=$count;ProtectedHashEvidence='First pass full hashes; supplemental deletion only exact separately audited legacy paths';PlanSHA256=(Hash $pplan);LedgerSHA256=(Hash ($out+'\legacy_small_deletion_ledger.jsonl'));NoTraining=$true;NoGeneration=$true;OriginalEvidenceUnchanged=$true}
Save ($out+'\legacy_small_receipt.json') $result
$result|ConvertTo-Json
