$ErrorActionPreference = 'Stop'
$experimentRoot = 'E:\MeshFlow\experiments\r2_bench_control_transfer_v1'
$deadlineUtc = [DateTimeOffset]::Parse('2026-10-18T13:12:32Z')
$monitorLog = Join-Path $experimentRoot 'resource_monitor.jsonl'
$utf8 = New-Object System.Text.UTF8Encoding($false)
while ([DateTimeOffset]::UtcNow -lt $deadlineUtc) {
    $states = @()
    foreach ($arm in @('PLAIN','SEAM')) {
        $statePath = Join-Path $experimentRoot ('training\'+$arm+'\status.json')
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $states += [pscustomobject]@{arm=$arm;status=$state.status;step=$state.step;elapsed_seconds=$state.elapsed_seconds;last_update_seconds=$state.last.update_seconds}
    }
    $gpu = (& nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader) -join ' '
    $record = [ordered]@{utc=[DateTimeOffset]::UtcNow.ToString('o');training=$states;gpu=$gpu;action='READ_ONLY_MONITOR'}
    $line = $record | ConvertTo-Json -Depth 5 -Compress
    [System.IO.File]::AppendAllText($monitorLog, $line+[Environment]::NewLine, $utf8)
    Write-Output $line
    if (($states | Where-Object {$_.status -eq 'COMPLETE'}).Count -eq 2) {exit 0}
    if (($states | Where-Object {$_.status -notin @('TRAINING','COMPLETE')}).Count -gt 0) {exit 2}
    Start-Sleep -Seconds 50
}
Write-Output 'Monitor deadline reached; no process was stopped or restarted.'
exit 3
