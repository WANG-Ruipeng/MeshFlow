# Run and resume

Run from WSL using the existing environment:

```bash
cd /mnt/e/MeshFlow
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.chair_surface_attention_v1.run_all --resume
```

The controller reuses durable completed stage results. An attempted failed or interrupted scientific call is never replayed automatically. The current numerical gate is IMPLEMENTATION_BLOCKED; resume verifies identities and rebuilds the report only, and cannot start any formal training arm. A numerical redesign requires a separately authorized protocol, not editing this frozen run.

The one GPU audit worker acquires `/tmp/wrp_meshflow_jedge5_diag_gpu0.lock` before CUDA initialization. No unrelated process is killed and the lock is never deleted. Real forward/backward calls are reserved in ledger JSONL before execution, including failed calls. CPU module tests have separate attempt records. Known old checkpoint/raw files are references only.

Formal training code is implemented but NOT_RUN. No new fixed endpoint or generated SURFACE trajectory exists. All three arms' training, quality comparisons and trajectory quadrature remain blocked by the frozen gate. START uses 138 independently verified old raw files and six original inapplicable requests, with no replay.
