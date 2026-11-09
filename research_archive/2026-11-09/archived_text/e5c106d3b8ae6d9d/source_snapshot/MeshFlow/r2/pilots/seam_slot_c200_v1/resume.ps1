param([ValidateSet('status','train-full','train-seam','sample-full','sample-seam','post-full','post-seam','score','render')][string]$Stage='status')
$ErrorActionPreference='Stop'
$Workspace='E:\MeshFlow'
if ($Stage -eq 'status') {
  $RunRoot=Join-Path $Workspace 'experiments\r2_seam_slot_c200_v1'
  foreach ($Arm in @('FULL4_C200','SEAM_SLOT_C200')) {
    $StatusFile=Join-Path $RunRoot "training\$Arm\status.json"
    if (Test-Path -LiteralPath $StatusFile) {
      $State=Get-Content -LiteralPath $StatusFile -Encoding UTF8 -Raw|ConvertFrom-Json
      [pscustomobject]@{Arm=$Arm;Status=$State.status;LocalStep=$State.local_step;GlobalStep=$State.global_step}
    } else { [pscustomobject]@{Arm=$Arm;Status='NOT_STARTED';LocalStep=0;GlobalStep=12000} }
  }
  $Completion=Join-Path $RunRoot 'completion.json'
  if(Test-Path -LiteralPath $Completion){Get-Content -LiteralPath $Completion -Encoding UTF8}
  exit 0
}
$Common=@('-d','Ubuntu','--cd','/mnt/e/MeshFlow','--','env','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONPATH=/mnt/e/MeshFlow/r2/src:/mnt/e/MeshFlow','/mnt/e/MeshFlow-Control-artifacts/venv/bin/python','-B','-m')
switch($Stage) {
  'train-full' {$Module='train';$Extra=@('--arm','FULL4_C200')}
  'train-seam' {$Module='train';$Extra=@('--arm','SEAM_SLOT_C200')}
  'sample-full' {$Module='generate';$Extra=@('--arm','FULL4_C200')}
  'sample-seam' {$Module='generate';$Extra=@('--arm','SEAM_SLOT_C200')}
  'post-full' {$Module='postprocess';$Extra=@('--arm','FULL4_C200')}
  'post-seam' {$Module='postprocess';$Extra=@('--arm','SEAM_SLOT_C200')}
  'score' {$Module='evaluate';$Extra=@('--stage','all','--workers','2')}
  'render' {$Module='render';$Extra=@()}
}
& wsl @Common "r2.pilots.seam_slot_c200_v1.$Module" @Extra
exit $LASTEXITCODE
