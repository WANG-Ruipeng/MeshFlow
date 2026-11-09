param([string]$WorkspaceRoot = 'E:\MeshFlow')
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $WorkspaceRoot).Path
$dest = Join-Path $root 'maintenance\research_synthesis_20261028'
New-Item -ItemType Directory -Path $dest -Force | Out-Null
$rows = @()
$experimentRoot = Join-Path $root 'experiments'
foreach ($dir in Get-ChildItem -LiteralPath $experimentRoot -Directory | Where-Object Name -Like 'r2_*' | Sort-Object Name) {
    foreach ($file in Get-ChildItem -LiteralPath $dir.FullName -File | Where-Object { $_.Name -match '^(REPORT.*|NEGATIVE_RESULT_REPORT|DATA_REPORT|ANSWERS)\.(md|txt)$' } | Sort-Object Name) {
        $title = Get-Content -LiteralPath $file.FullName -Encoding UTF8 -TotalCount 1
        $rows += [pscustomobject]@{
            experiment = $dir.Name
            report = $file.FullName.Substring($root.Length + 1).Replace('\','/')
            title = [string]$title
            bytes = $file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            evidence_scope = 'DISCOVERY_INDEX_NOT_A_NEW_REVIEW_OR_SCORE'
        }
    }
}
$rows | Export-Csv -LiteralPath (Join-Path $dest 'report_inventory.csv') -NoTypeInformation -Encoding UTF8
$registryPath = Join-Path $root 'r2\docs\research_route_registry.json'
$registry = Get-Content -LiteralPath $registryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$registry.routes | Select-Object id,name,status,evidence,retry_boundary,@{Name='source_ids';Expression={$_.source_ids -join ';'}} | Export-Csv -LiteralPath (Join-Path $dest 'historical_route_summary.csv') -NoTypeInformation -Encoding UTF8
[pscustomobject]@{
    generated_at_utc = [DateTime]::UtcNow.ToString('o')
    mode = 'READ_ONLY_EVIDENCE_SYNTHESIS_NO_MODELS_NO_RESCORING'
    scan_scope = 'Immediate REPORT*/NEGATIVE_RESULT_REPORT/DATA_REPORT/ANSWERS md/txt files under experiments/r2_*; not all workspace reports'
    indexed_reports = $rows.Count
    indexed_experiment_directories = @($rows.experiment | Select-Object -Unique).Count
    historical_registry = 'r2/docs/research_route_registry.json'
    historical_registry_sha256 = (Get-FileHash -LiteralPath $registryPath -Algorithm SHA256).Hash.ToLowerInvariant()
    historical_registry_counts = $registry.counts
    limitation = 'Historical family counts and indexed reports overlap and must not be added as experiment counts. An indexed report is not a claim of a fresh full audit. Core decision evidence is linked in RESEARCH_SYNTHESIS.md.'
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $dest 'inventory_scope.json') -Encoding UTF8
Get-Content -LiteralPath (Join-Path $dest 'inventory_scope.json') -Encoding UTF8
