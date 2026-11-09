param([ValidateSet('prepare','calibration-sample','calibration-d0','calibration-cpu','calibration-gallery','calibration-analyze','train','final-sample','final-d0','final-cpu','final-gallery','final-analyze')][string]$Phase='prepare')
$ErrorActionPreference='Stop'
$Module='r2.pilots.window_policy_rl_v1.run'
$ModuleArgs=@($Phase)
switch ($Phase) {
  'train' { $Module='r2.pilots.window_policy_rl_v1.train_controller'; $ModuleArgs=@() }
  'calibration-gallery' { $Module='r2.pilots.window_policy_rl_v1.gallery'; $ModuleArgs=@('--phase','calibration','--workers','2') }
  'final-gallery' { $Module='r2.pilots.window_policy_rl_v1.gallery'; $ModuleArgs=@('--phase','final','--workers','2') }
  'calibration-analyze' { $Module='r2.pilots.window_policy_rl_v1.analyze'; $ModuleArgs=@('--reviews','experiments/r2_window_policy_rl_v1/calibration/blind_gallery/reviews_actual_v2.json') }
  'final-analyze' { $Module='r2.pilots.window_policy_rl_v1.final_analysis'; $ModuleArgs=@() }
}
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- env PYTHONPATH=/mnt/e/MeshFlow/r2/src:/mnt/e/MeshFlow OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m $Module @ModuleArgs
if ($LASTEXITCODE -ne 0) { throw "Phase failed without automatic retry: $Phase" }
