Run in WSL from /mnt/e/MeshFlow:

`/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.tools.c_restore_bounded_v1.run --stage all`

CPU only. Successful stages are hash-verified and reused. An incomplete attempted scientific stage blocks rather than silently replaying. Originals remain read-only. Final reporting/audit/packaging never dispatch models.
