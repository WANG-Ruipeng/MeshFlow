$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$reviewDir = $PSScriptRoot
function Read-Json($path) {
    return Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
}
function Sha256($path) {
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Win-Path($path) {
    if ($path.StartsWith('/mnt/e/MeshFlow/')) {
        return Join-Path 'E:\MeshFlow' $path.Substring('/mnt/e/MeshFlow/'.Length).Replace('/', '\')
    }
    return $path
}
$protocolPath = Join-Path $reviewDir 'protocol.json'
$assignmentPath = Join-Path $reviewDir 'assignments.json'
$protocol = Read-Json $protocolPath
$assignments = Read-Json $assignmentPath
$observations = Read-Json (Join-Path $reviewDir 'training_observations_16.json')
$expected = Read-Json (Join-Path $reviewDir 'expected_cells.json')
$requests = @($assignments.assignments.training)
$trainingCells = @($expected.rows | Where-Object reviewer -eq 'training')
if ($requests.Count -ne 16 -or $observations.Count -ne 16 -or $trainingCells.Count -ne 144) {
    throw 'Incomplete assignment or observation counts'
}
$models = @('A_S1500', 'B_S1500', 'C_S1500')
$stages = @('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')
$axes = @('overall_shape', 'C_use', 'support_retention', 'visible_fragment_or_gap', 'layering_or_fold')
$maps = @(
    @{R='REASONABLE'; I='INCOMPLETE'; U='UNCERTAIN'},
    @{P='PLAUSIBLE'; X='REPLACED_OR_IGNORED'; U='UNCERTAIN'; N='NA'},
    @{P='PRESENT'; M='MISSING_OR_DAMAGED'; U='UNCERTAIN'; N='NA'},
    @{Y='YES'; N='NO_OBVIOUS'; U='UNCERTAIN'},
    @{Y='YES'; N='NO_OBVIOUS'; U='UNCERTAIN'}
)
$rows = @()
$imageHashes = @{}
$definitionHash = Sha256 (Join-Path $taskRoot 'gallery\definition.json')
for ($i = 0; $i -lt 16; $i++) {
    $obs = $observations[$i]
    if ($obs.i -ne $i -or $obs.rows.Count -ne 9) { throw "Bad observations index $i" }
    $rid = $requests[$i]
    for ($j = 0; $j -lt 9; $j++) {
        $model = $models[$j % 3]
        $stage = $stages[[math]::Floor($j / 3)]
        $cells = @($trainingCells | Where-Object {
            $_.request_id -eq $rid -and $_.model -eq $model -and $_.stage -eq $stage
        })
        if ($cells.Count -ne 1) { throw "Missing or duplicate expected cell $rid $model $stage" }
        $cell = $cells[0]
        $imagePath = Win-Path $cell.image
        $receiptPath = Win-Path $cell.image_receipt
        $receipt = Read-Json $receiptPath
        if (-not $imageHashes.ContainsKey($imagePath)) {
            $imageHashes[$imagePath] = Sha256 $imagePath
        }
        $imageHash = $imageHashes[$imagePath]
        if ($receipt.image_sha256 -ne $imageHash -or $receipt.definition_sha256 -ne $definitionHash) {
            throw "Unauthenticated image $imagePath"
        }
        if ($receipt.request_id -ne $rid) { throw "Receipt request mismatch $rid" }
        $producerCell = @($receipt.cells | Where-Object { $_.model -eq $model -and $_.stage -eq $stage })
        if ($producerCell.Count -ne 1) { throw "Receipt cell mismatch $rid $model $stage" }
        $rating = $obs.rows[$j]
        if ($rating.Count -ne 6 -or [string]::IsNullOrWhiteSpace($rating[5])) {
            throw "Missing actual note $rid $model $stage"
        }
        $row = [ordered]@{
            request_id=$rid
            model=$model
            stage=$stage
            image_sha256=$imageHash
            actually_viewed=$true
            views_and_sections_checked=$true
        }
        for ($a = 0; $a -lt 5; $a++) {
            $axis = $axes[$a]
            $value = $maps[$a][$rating[$a]]
            if ($null -eq $value -or $value -notin $protocol.axes.$axis) {
                throw "Invalid rating $rid $model $stage $axis"
            }
            $row[$axis] = $value
        }
        $row.notes = $rating[5]
        $rows += [pscustomobject]$row
    }
}
if ($rows.Count -ne 144 -or $imageHashes.Count -ne 32) { throw 'Incomplete final review' }
$keys = @($rows | ForEach-Object { "$($_.request_id)|$($_.model)|$($_.stage)" } | Sort-Object -Unique)
if ($keys.Count -ne 144) { throw 'Duplicate reviews' }
$envelope = [ordered]@{
    schema=$protocol.schema
    reviewer='training'
    reviewer_kind='AI_PROXY'
    blind=$false
    protocol_sha256=(Sha256 $protocolPath)
    assignments_sha256=(Sha256 $assignmentPath)
    rows=$rows
}
$outputPath = Join-Path $reviewDir 'review_training.json'
if (Test-Path -LiteralPath $outputPath) { throw 'Refusing to overwrite an existing completed review' }
$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($outputPath, ($envelope | ConvertTo-Json -Depth 20), $utf8)
$check = [ordered]@{
    status='PASS'
    reviewer='training'
    request_count=16
    actual_image_sheets=32
    actual_cell_records=144
    source_observations_sha256=(Sha256 (Join-Path $reviewDir 'training_observations_16.json'))
    review_sha256=(Sha256 $outputPath)
    image_hashes=$imageHashes
    scope='Self-validation only; does not collect or certify other reviewers or the experiment.'
}
[System.IO.File]::WriteAllText((Join-Path $reviewDir 'training_self_validation.json'), ($check | ConvertTo-Json -Depth 20), $utf8)
[pscustomobject]$check | Select-Object status, request_count, actual_image_sheets, actual_cell_records, review_sha256 | ConvertTo-Json
