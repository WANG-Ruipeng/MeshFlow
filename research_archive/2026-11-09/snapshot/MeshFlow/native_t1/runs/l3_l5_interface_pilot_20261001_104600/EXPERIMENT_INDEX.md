# Experiment index

Current default: Native T1 + Geo, original FM. Public G0 cumulative2500 is unchanged; new candidates are not defaults.

Historical six-loss screen is read-only: /mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/loss_screen_20260930_232200
Historical results SHA256: ffaf7a01f62fdd8a1aa02eb5f826e942701f8ff8588a564472241460ce3813e0
No old checkpoint, raw output, score, source snapshot or failure record is moved, deleted or overwritten.

This pilot trains only J35 and JEdge5, each from the same G0_2500 for500 updates, and evaluates only L0/L3/L5/J35/JEdge5.
L1/L2/L4/L6 remain historical candidates and are absent from this pilot default run list.

| Role | Path | State SHA256 |
| --- | --- | --- |
| Common initialization / current default reference | /mnt/e/MeshFlow/native_t1/checkpoints/tgeo_g0_cumulative2500.pt | 6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392 |
| Read-only L03000 | /mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/loss_screen_20260930_232200/L0/additional_step500_cumulative_step3000.pt | f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b |
| Read-only L33000 | /mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/loss_screen_20260930_232200/L3/additional_step500_cumulative_step3000.pt | 663c2ed010b8a5363e34984eefb42eb1243391292741423937d4377e41e57026 |
| Read-only L53000 | /mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/loss_screen_20260930_232200/L5/additional_step500_cumulative_step3000.pt | 3c596db36fd7d5836427906fedecd7485b7be3e62cc6f605dbcb25b62c0c6641 |

Pilot protocol: protocol.json. New endpoints are recorded in results.json; no historical weights are copied.
