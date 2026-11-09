# Execution and recovery

Run in the existing WSL environment:

```powershell
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/pilots/user_mix_from_official_v1/run.py --stage status
```

The controller root is E:/MeshFlow/experiments/r2_user_mix_from_official_v1. The original data and prior experiment remain outside this directory and read-only.

Executed preparation: register, prepare-data, prepare-eval, freeze. The single training arm starts with --stage train. Do not dispatch a second trainer while status is TRAINING. Completed stages and their receipts are immutable; do not restart or regenerate them.

After training is COMPLETE, the authorized stages are:

```powershell
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/pilots/user_mix_from_official_v1/run.py --stage sample
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/pilots/user_mix_from_official_v1/score_parallel.py --stage run --out /mnt/e/MeshFlow/experiments/r2_user_mix_from_official_v1 --workers 2 --controller-owned
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/pilots/user_mix_from_official_v1/run.py --stage render
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/pilots/user_mix_from_official_v1/run.py --stage verify-protected
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/pilots/user_mix_from_official_v1/audit_run.py --out /mnt/e/MeshFlow/experiments/r2_user_mix_from_official_v1
```

Review all24 rendered PNGs /12 requests, preserving AI_PROXY and uncertainty. Save per-output and paired observations, then write the report and final delivery hashes. No new generation is authorized after the registered12 requests.

Training recovery is allowed only from status INTERRUPTED_SAFE and its exact checkpoint hash. Use --stage train --resume <that checkpoint path> --attempt <new unique ID>. The trainer verifies stream, AdamW, RNG, code, data and every consumed call. A regular rolling checkpoint is not permission to replay updates completed after it. OOM/numerical failure stops the experiment; no automatic retries or scientific-setting changes.

The train source manifest was frozen before GPU dispatch. The independent CPU scoring helper and final audit helper have separate source identities; they do not change the frozen trainer/evaluator. Do not overwrite the train manifest to admit source changes.