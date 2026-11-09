param([Parameter(Mandatory=$true)][string]$Batch)
$ErrorActionPreference = 'Stop'
$root = 'E:\MeshFlow\experiments\r2_bench_joint_shift_v1\visual_review'
$a = Get-Content -Encoding UTF8 -Raw -LiteralPath (Join-Path $root 'assignments.json') | ConvertFrom-Json
$incoming = Get-Content -Encoding UTF8 -Raw -LiteralPath $Batch | ConvertFrom-Json
$dest = Join-Path $root 'dev_a_records.json'
$records = @()
if (Test-Path -LiteralPath $dest) { $records = @((Get-Content -Encoding UTF8 -Raw -LiteralPath $dest | ConvertFrom-Json).records) }
$maps = @(@{R='REASONABLE';I='INCOMPLETE';U='UNCERTAIN'},@{P='PLAUSIBLE';R='REPLACED_OR_IGNORED';U='UNCERTAIN'},@{P='PRESENT';M='MISSING_OR_DAMAGED';U='UNCERTAIN'},@{Y='YES';N='NO_OBVIOUS';U='UNCERTAIN'},@{Y='YES';N='NO_OBVIOUS';U='UNCERTAIN'})
foreach ($entry in $incoming) {
    $page = $a.tasks.dev_a.pages[[int]$entry.index]
    $receipt = Get-Content -Encoding UTF8 -Raw -LiteralPath ([System.IO.Path]::ChangeExtension($page.image_path,'.json')) | ConvertFrom-Json
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $page.image_path).Hash.ToLowerInvariant()
    if ($hash -ne $receipt.image_sha256 -or $page.page_id -ne $receipt.page_id) { throw 'Gallery identity mismatch' }
    if ($entry.cells.Count -ne 3) { throw 'Three freshly reviewed columns required' }
    for ($j=0;$j -lt 3;$j++) {
        $cell = $entry.cells[$j]; $codes = $cell.code.ToCharArray()
        if ($codes.Count -ne 5) { throw 'Five axes required' }
        $vals = @()
        for ($k=0;$k -lt 5;$k++) { $val=$maps[$k][[string]$codes[$k]];if(-not $val){throw 'Invalid axis code'};$vals+=$val }
        $label = $a.labels[$j]
        if (@($records | Where-Object { $_.page_id -eq $page.page_id -and $_.label -eq $label }).Count) { throw 'Duplicate review cell' }
        $records += [ordered]@{request_id=$page.request_id;parent_id=$page.parent_id;condition_id=$page.condition_id;label=$label;stage=$page.stage;page_id=$page.page_id;image_path=$page.image_path;image_sha256=$hash;actually_viewed=$true;actual_view_source='tools.view_image full comparison PNG';overall_shape=$vals[0];C_use=$vals[1];support_retention=$vals[2];visible_fragment_or_gap=$vals[3];layering_or_fold=$vals[4];note=$cell.note}
    }
}
$obj=[ordered]@{reviewer_kind='AI_PROXY';blind=$false;group='dev_a';protocol_sha256=$a.protocol_sha256;records=$records}
[System.IO.File]::WriteAllText($dest,(($obj | ConvertTo-Json -Depth 8)+[Environment]::NewLine),(New-Object System.Text.UTF8Encoding($false)))
Write-Output ('Saved ' + $records.Count + ' records')
