# I cost decomposition from existing evidence

This is re-aggregation of 64 existing score.json, pairs.json, vectors.npz records for 3f/73b, two models, two stages and eight frozen noises. No model, geometric detector or scorer was called. The saved query masks and weights reproduce every I component within 1e-12.

| Parent (POST, mean over 8 noises) | SMALL I % | CREF I % | SMALL cross % | CREF cross % | CREF near-parallel dup % | CREF cross AND dup % |
|---|---:|---:|---:|---:|---:|---:|
| 3f | 0.7996 | 14.4348 | 0.7996 | 3.5461 | 11.7310 | 0.8423 |
| 73b | 1.7761 | 16.7419 | 1.7761 | 1.4465 | 15.4358 | 0.1404 |

I = cross + dup - their overlap. The increase is dominated by the near-parallel term; for 73b the actual penetration-band term declines on average. These I values must not all be described as certified illegal intersections.

The frozen scorer calls dup a proximity proxy: C queries at least 0.01L from the open C boundary, within strictly 0.005L of a finite F triangle, with absolute normal dot meeting the 15-degree criterion. The absolute dot accepts both aligned and reversed normals. The cross term covers points within 0.005L of an already witnessed proper CF penetration segment. Both use the total weight of all 2048 C queries as denominator, including border-excluded samples. The source is MeshFlow-Control/src/meshflow_control/evaluation/control.py, constants at lines 23-29 and conflict_metrics at lines 243-281. No thresholds have changed.

The near-parallel proxy does not certify exact duplicate faces or positive-area coplanarity. Across these 64 outputs the exact duplicate face counts are zero. POST has only two CF positive-area coplanar pairs in 3f CREF (across all eight noises) and zero in 73b CREF. In particular, 73b n4 has zero cross-band coverage and I=43.3594%, entirely due to near-parallel proximity. Saved masks do not identify the nearest qualifying F face or its signed side, so this package cannot separate legitimate thin underside from a redundant parallel seat without an additional targeted geometry inspection; none was run here.

The means also have substantial tails. 3f CREF POST ranges from 1.1230% to 76.1719%, with median 5.5908%; n2 alone contributes about two thirds of its summed I. 73b ranges from 0.0977% to 75.6836%, with median 3.2959%; n4 and n6 dominate the sum. Nevertheless, I rises versus SMALL on 7/8 noises for 3f and 6/8 for 73b, so the change is not confined to a single output.

All three RAW NO-to-YES adoption transitions in these two parents coexist with POST I increases: 3f n2 goes 0 -> 76.1719%; 73b n3 goes 1.6113 -> 5.9082%; 73b n4 goes 5.5176 -> 43.3594%. This is an observed paired tradeoff, not a causal claim that adoption requires high conflict. Uncertain ratings are retained separately.

An independent geometric cost remains: 73b POST FF affected-free fraction grows from 11.6092% to 31.9794%, with proper FF pair counts summed over noises increasing 102 -> 533. For 3f it moves 11.2259% -> 12.0341%, pair counts 94 -> 123. These affected-face denominators differ with POST face deletion and are recorded per output; they are not directly interchangeable with C-query-weight I.

See I_cost_paired_noise.csv for all paired noises, I_cost_per_output.csv for components/counts/denominators and adoption labels, I_cost_per_C_face.csv for existing query-mask coverage grouped by C face, I_pair_classifications.csv for saved CF/FF classifications, and I_COST_BREAKDOWN.json for provenance hashes and limitations. Original score files remain unchanged.
