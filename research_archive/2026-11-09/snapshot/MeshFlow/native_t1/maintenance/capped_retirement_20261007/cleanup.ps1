param([ValidateSet('Prepare','Delete','Verify')][string]$Phase='Prepare')
$ErrorActionPreference='Stop'
$workspaceRoot='E:\MeshFlow'
$recordRoot='E:\MeshFlow\native_t1\maintenance\capped_retirement_20261007'
$runNames=@('chair_control_feedback_auto_v1','chair_capped500_gradbudget_v1','chair_capped500_half_v1')
$runRoots=@($runNames | ForEach-Object { Join-Path $workspaceRoot ('native_t1\runs\'+$_) })
$codeRoots=@($runNames | ForEach-Object { Join-Path $workspaceRoot ('experiments\mf_h1_local0\'+$_) })
$cacheRoots=@((Join-Path $runRoots[0] 'streams\explore\ot_cache'),(Join-Path $runRoots[1] 'streams\repeat\ot_cache'))
$startPaths=@((Join-Path $workspaceRoot 'native_t1\checkpoints\chair_hybrid_start_total5000.pt'),(Join-Path $workspaceRoot 'native_t1\runs\chair_hybrid_coupling_v1\checkpoints\OT_HYBRID_pilot1000_total5000.pt'))
$startSha='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
$utf8=[Text.UTF8Encoding]::new($false)
$checkedDirs=@{}
function HashFile([string]$Path){
    $stream=[IO.File]::OpenRead($Path);$algo=[Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algo.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally {$algo.Dispose();$stream.Dispose()}
}
function WriteJson([string]$Path,$Value){[IO.File]::WriteAllText($Path,(ConvertTo-Json -InputObject $Value -Depth 12),$utf8)}
function ReadJson([string]$Path){Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json}
function SafePath([string]$Path){
    $full=[IO.Path]::GetFullPath($Path)
    if(-not $full.StartsWith($workspaceRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Outside workspace: $full"}
    $walk=[IO.Path]::GetDirectoryName($full)
    while($walk.Length -ge $workspaceRoot.Length){
        if(-not $checkedDirs.ContainsKey($walk)){
            $item=Get-Item -LiteralPath $walk -Force
            if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw "Reparse directory: $walk"}
            $checkedDirs[$walk]=$true
        }
        if($walk -eq $workspaceRoot){break}
        $walk=[IO.Path]::GetDirectoryName($walk)
    }
    return $full
}
function FileRow($Item,[string]$Category){
    $full=SafePath $Item.FullName
    if($Item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw "Reparse file: $full"}
    return [pscustomobject]@{path=$full;bytes=[long]$Item.Length;sha256=(HashFile $full);category=$Category}
}
function AllowedDeletion([string]$Path){
    $full=SafePath $Path;$parent=[IO.Path]::GetDirectoryName($full);$name=[IO.Path]::GetFileName($full)
    if($cacheRoots -contains $parent){return $name.EndsWith('.npz',[StringComparison]::OrdinalIgnoreCase)}
    foreach($root in $runRoots){if($parent -eq (Join-Path $root 'checkpoints')){return $name.EndsWith('.pt',[StringComparison]::OrdinalIgnoreCase)}}
    return $false
}
function Event($Value){[IO.File]::AppendAllText((Join-Path $recordRoot 'operations.jsonl'),((ConvertTo-Json -InputObject $Value -Compress -Depth 6)+[Environment]::NewLine),$utf8)}
function SnapshotRows($Files,[string]$Category){
    $result=[Collections.Generic.List[object]]::new();$i=0
    foreach($f in $Files){$result.Add((FileRow $f $Category));$i++;if($i%4000 -eq 0){Write-Host "$Category $i"}}
    return $result.ToArray()
}

SafePath (Join-Path $recordRoot 'cleanup.ps1') | Out-Null
$planPath=Join-Path $recordRoot 'deletion_manifest.json'
if($Phase -eq 'Prepare'){
    if(Test-Path -LiteralPath $planPath){throw 'Manifest exists; do not overwrite'}
    $targets=[Collections.Generic.List[object]]::new()
    foreach($root in $runRoots){
        $folder=Join-Path $root 'checkpoints';SafePath (Join-Path $folder 'placeholder') | Out-Null
        foreach($file in @(Get-ChildItem -LiteralPath $folder -File -Filter '*.pt')){
            $row=FileRow $file 'retired_feedback_weight';$side=ReadJson ([IO.Path]::ChangeExtension($file.FullName,'.json'))
            if($side.checkpoint_sha256 -ne $row.sha256){throw "Weight differs from recorded identity: $($row.path)"}
            $targets.Add($row);Write-Host "WEIGHT_VERIFIED $($file.Name)"
        }
    }
    if($targets.Count -ne 22){throw 'Expected exactly22 retired weights'}
    foreach($root in $cacheRoots){
        SafePath (Join-Path $root 'placeholder') | Out-Null
        $files=@(Get-ChildItem -LiteralPath $root -File -Filter '*.npz')
        if($files.Count -ne 8000){throw "Expected8000 derived OT files: $root"}
        foreach($r in @(SnapshotRows $files 'derived_OT_cache')){$targets.Add($r)}
    }
    $targetSet=@{};foreach($r in $targets){if(-not (AllowedDeletion $r.path)){throw 'Unapproved location'};$targetSet[$r.path]=$true}
    $keep=@{}
    foreach($root in ($runRoots+$codeRoots)){
        foreach($f in @(Get-ChildItem -LiteralPath $root -Recurse -File -Force)){
            if(-not $targetSet.ContainsKey($f.FullName)){$keep[$f.FullName]=$f}
        }
    }
    foreach($p in @(git -C $workspaceRoot ls-files native_t1)){
        $full=Join-Path $workspaceRoot $p
        if(Test-Path -LiteralPath $full -PathType Leaf){$keep[$full]=Get-Item -LiteralPath $full}
    }
    foreach($f in @(Get-ChildItem -LiteralPath (Join-Path $workspaceRoot 'native_t1\checkpoints') -File -Recurse -Force)){$keep[$f.FullName]=$f}
    foreach($p in ($startPaths+@((Join-Path $workspaceRoot 'configs\snet\base-120m-ot-v-chair.yaml'),(Join-Path $workspaceRoot 'AGENTS.md')))){$keep[$p]=Get-Item -LiteralPath $p}
    $protected=@(SnapshotRows @($keep.Values | Sort-Object FullName) 'retained_evidence_or_baseline')
    foreach($p in $startPaths){if((HashFile $p) -ne $startSha){throw 'START identity mismatch'}}
    WriteJson (Join-Path $recordRoot 'protected_before.json') ([ordered]@{files=$protected;count=$protected.Count})
    $groups=@($targets | Group-Object category | ForEach-Object {[pscustomobject]@{category=$_.Name;files=$_.Count;bytes=($_.Group|Measure-Object bytes -Sum).Sum}})
    WriteJson $planPath ([ordered]@{status='FROZEN';authorization='User requested unused temporary files and weights removed; return to original START baseline.';files=$targets.ToArray();groups=$groups;logical_bytes=($targets|Measure-Object bytes -Sum).Sum;retained_files=$protected.Count;preserve_OT_attempts=$true;no_directory_deletion=$true;plan_source_sha256=(HashFile $PSCommandPath)})
    Copy-Item -LiteralPath (Join-Path $workspaceRoot 'WORKSPACE.md') -Destination (Join-Path $recordRoot 'WORKSPACE_before.md')
    [IO.File]::WriteAllLines((Join-Path $recordRoot 'git_status_before.txt'),@(git -C $workspaceRoot status --short),$utf8)
    Write-Output ($groups | ConvertTo-Json -Compress)
    Write-Output "PREPARED protected=$($protected.Count) targets=$($targets.Count)"
}
if($Phase -eq 'Delete'){
    $plan=ReadJson $planPath
    if($plan.plan_source_sha256 -ne (HashFile $PSCommandPath)){throw 'Cleanup code changed after registration'}
    $deleted=@{}
    $journal=Join-Path $recordRoot 'operations.jsonl'
    if(Test-Path -LiteralPath $journal){foreach($line in [IO.File]::ReadLines($journal)){if($line){$e=$line|ConvertFrom-Json;if($e.event -eq 'DELETE_OK'){$deleted[$e.path]=$true}}}}
    foreach($p in $startPaths){if((HashFile $p) -ne $startSha){throw 'START protection mismatch before deletion'}}
    $count=0
    foreach($r in $plan.files){
        if(-not (AllowedDeletion $r.path)){throw "Forbidden deletion target: $($r.path)"}
        if(-not (Test-Path -LiteralPath $r.path)){
            if(-not $deleted.ContainsKey($r.path)){throw "Unexpected pre-existing absence: $($r.path)"}
            continue
        }
        $file=Get-Item -LiteralPath $r.path -Force
        if($file.PSIsContainer -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw "Unsafe target type: $($r.path)"}
        if($file.Length -ne $r.bytes -or (HashFile $r.path) -ne $r.sha256){throw "Target changed: $($r.path)"}
        Event @{event='DELETE_BEGIN';path=$r.path;bytes=$r.bytes;sha256=$r.sha256;utc=[DateTime]::UtcNow.ToString('o')}
        Remove-Item -LiteralPath $r.path -Force -ErrorAction Stop
        if(Test-Path -LiteralPath $r.path){throw "Deletion failed: $($r.path)"}
        Event @{event='DELETE_OK';path=$r.path;bytes=$r.bytes;sha256=$r.sha256;utc=[DateTime]::UtcNow.ToString('o')}
        $count++;if($count%2000 -eq 0){Write-Output "DELETED $count"}
    }
    WriteJson (Join-Path $recordRoot 'deletion_complete.json') @{status='PASS';files=$plan.files.Count;logical_bytes=$plan.logical_bytes;groups=$plan.groups}
    Write-Output "DELETE_COMPLETE $($plan.files.Count)"
}
if($Phase -eq 'Verify'){
    $plan=ReadJson $planPath;$before=ReadJson (Join-Path $recordRoot 'protected_before.json')
    $issues=[Collections.Generic.List[object]]::new();$after=[Collections.Generic.List[object]]::new();$i=0
    foreach($r in $plan.files){if(Test-Path -LiteralPath $r.path){$issues.Add(@{issue='Deletion target remains';path=$r.path})}}
    foreach($r in $before.files){
        if(-not (Test-Path -LiteralPath $r.path -PathType Leaf)){$issues.Add(@{issue='Protected file missing';path=$r.path});continue}
        $actual=HashFile $r.path;$after.Add([pscustomobject]@{path=$r.path;sha256=$actual;unchanged=($actual -eq $r.sha256)})
        if($actual -ne $r.sha256){$issues.Add(@{issue='Protected bytes changed';path=$r.path})}
        $i++;if($i%4000 -eq 0){Write-Output "VERIFIED $i"}
    }
    WriteJson (Join-Path $recordRoot 'protected_after.json') @{files=$after.ToArray();count=$after.Count}
    $status=if($issues.Count -eq 0){'PASS'}else{'FAIL'}
    $result=@{status=$status;protected_files=$before.files.Count;unchanged_files=$after.Count;deleted_files=$plan.files.Count;logical_bytes=$plan.logical_bytes;issues=$issues.ToArray();model_forwards=0;training_updates=0;generation=0}
    WriteJson (Join-Path $recordRoot 'verification.json') $result
    Write-Output ($result|ConvertTo-Json -Compress)
    if($issues.Count){throw 'Final protection verification failed'}
}
