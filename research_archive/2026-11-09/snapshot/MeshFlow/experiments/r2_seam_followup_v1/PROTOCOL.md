# SEAM frozen follow-up V1

Frozen before model dispatch, 2026-10-17. User authorized experimentation on attachment `3701b7dc-17e4-4786-848f-526b132f75fc/Pasted text.txt`.

## Scope and stopping point

Execute the attachment's two immediate frozen diagnostics. Zero training, zero optimizer updates, no new loss or pointer. Ideas 3/4 receive feasibility/history review only; no training budget was specified for them. Do not infer success, change maintained defaults, delete weights, commit or push.

1. **B_LONG_LAYOUT**: frozen final B_POOL47 generator, exact cached LONG occupancy/layout on all 28 registered ABC requests. Original C/N/Gaussian arrays, Euler50, deterministic BF16 forward/FP32 integration. No occupancy/pointer head calls. Exactly 28 new RAW rollouts (1400 generator forwards), then 28 single official D0 calls after all RAW are saved. Same protected GEO_C/source winding export as ABC. Compare to cached B-own-layout and LONG, TRAIN16 and DEV12 separately. This is an inference-layout intervention, not a proof that occupancy caused all B degradation or that the expanded generator is universally better.
2. **CPU binding**: frozen LONG and A_COUPLE POST representatives/actual GEO_C cluster membership and PRE_GEO corner positions; 28 requests each, two predeclared per-corner displacement caps 0.015L and 0.020L, 112 variants. L remains the fixed ABC scale sqrt(3)*2/0.3762. Original RAW/D0 remain untouched. No additional neural calls.

## CPU intervention definition

Candidates target free seam-tip clusters and open seam lateral edges, pairing with bulk open-boundary clusters/edges. Tip-only midpoint merges and full lateral-edge correspondence are separately identified in one deterministic candidate stream. Sharing a tip alone is not counted as shared-edge or bulk reach success. Two free representative clusters meet at their midpoint; a cluster containing a C anchor is immovable. No radius re-clustering: operate original GEO_C identities and transport all members together. Lateral edges test both endpoint permutations and unsigned axis angle <30 degrees. Sort by maximum actual member displacement, then stable candidate type and source identities. Representative separation cannot exceed twice the displacement cap. Every moved original PRE_GEO corner, including members of deleted faces, must meet the cap. Record both representative separation and actual displacement.

Reject anchor conflicts, same-face collapse/area <=1e-12 L², new exact duplicate faces, new/worse <5-degree angles or q>10 skinny faces, physical geometric normal rotation >60 degrees, newly illegal CF pairs, new nonmanifold edges, or loss of baseline legal CF edges. Checks cover all affected faces; winding re-expression is distinct from physical geometric rotation. Existing defects are not silently repaired or discarded. Track old legal FF connectivity losses and newly illegal FF pairs rather than assuming cluster assignment is harmless. Exact cleanup and C-fixed free orientation follow the existing export contract. Keep accepted/rejected candidate ledger and deleted/source face identities.

## Evaluation and evidence

Use unchanged existing 124 numeric fields/scorers, fixed query points and denominators; RAW/POST separate. Main reading: actual legal seam-bulk edges, free-only reached bulk area, CF/FF/I, small angles, retained/deleted free faces and area, components, nonmanifold edges, old CF/FF preservation, newly illegal FF pairs. Compare all paired requests, not only visually selected wins. C preservation must pass. Visual review is actual nonblind AI_PROXY, with uncertain labels retained; it does not certify topology. Preserve per-output/per-parent tables, candidate ledger, meshes, source hashes, implementation snapshots, report and resumable entry point.

Closest routes: R31 SEAM bounded positive; ABC A/B/C; R2_PARTIAL_EDGE_BINDING_V1 (C endpoint target, now seam free-tip/lateral target); R31 geometry guards (capacity gains with FF costs); R07/R10 for coherence/partner training proposals. No physical upper bound, no information-theoretic impossibility, no zero-crack guarantee. B failure can have both generator and layout contributions. Original parent topology is supervision/history only, never new inference oracle.

## Resources and failures

Inspect actual WSL resources before each dispatch wave. First snapshot RTX5080 total16303MiB, free13017MiB, utilization10%, 24logicalCPU, ~61GiB availableRAM, no compute process listed. Keep >=2GiB extra free headroom. Initial one GPU sampler; independent CPU geometry overlaps. Comparable sampling estimate2.6GiB/worker; D0 estimate3.5GiB/worker includes conservative context allowance. WDDM process VRAM may be NA, never equate allocated memory with process memory. Additional workers only after representative measurement and throughput justification; frozen seeds/output ownership independent. CPU threads bounded. No synthetic GPU probes.

On OOM/numerical/model failure stop new dispatch, preserve attempted calls, do not silently retry/change scientific settings. Engineering fixes before model calls are logged. Once the fixed matrices and review are complete, stop.
