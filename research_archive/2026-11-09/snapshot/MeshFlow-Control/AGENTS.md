# MeshFlow-Control maintenance
This is an independent research fork, not an official MeshFlow author release.
Read README.md, docs/method.md, docs/metrics.md and docs/reproduction.md before changes.
Keep weights, raw meshes, private data and logs outside this source repository.
The original sibling MeshFlow workspace is read-only for this task.
Do not silently change model defaults, metric definitions or checkpoint selection.
The fixed Direct Context Readout V1 study uses one GPU worker, the shared lock,
three 1000-update arms and an external durable budget ledger. No new experiment,
training update or sample is authorized merely by a maintenance request.
Do not automatically commit, push, delete retained assets or restore retired methods.
