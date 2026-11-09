# V2 independent real-precheck review

**Result: PASS.** This review only read persisted evidence on CPU. It did not run a model, CUDA, relation computation, backward, or optimizer update.

The ledger contains 120 returned forwards and 44 returned zero-update backwards, with zero failed or unresolved calls. Each of the 12 measured MOMENT cost backwards and four warmup backwards has exactly the registered 16 new-parameter gradient keys: all fc2 gradients are finite and nonzero; all fc1 gradients are zero at zero initialization.

Eight zero-initialization raw comparisons across four tasks and FP32/BF16 are byte-identical. Sixteen same-shape padding-value raw comparisons are byte-identical. All eight persisted real padding-input backward probes report zero gradients. The gradient arrays were not saved, and this review did not repeat backward.

For each of the four FP32 original-versus-padded task pairs, all 88 saved stage tensors are byte-identical: compact identities/inputs/R, valid QKV, phi, logits, softmax, aggregation, projected correction, and valid hidden states through layer 12. There is therefore no first difference to attribute in these runs. The JSON reconstructs site/layer labels from the frozen source emission order because flat_debug omitted scalar metadata. This finding is limited to the tested four inputs and 11 additional padding slots.

All four nonzero MOMENT FP32 permutations still fail the old elementwise allclose test, while passing the preregistered normalized contract. Maximum e_inf = 2.61414892e-06; maximum e_rms = 6.69782412e-07. Historical FAIL results and current old-test flags remain unchanged.

BF16 nonzero permutation differences are descriptive only: maximum e_inf = 0.0132450331; maximum e_rms = 0.00617492578. This does not establish strict equivariance, and the FP32 thresholds are not applied.

| Fixed task | MOMENT/H forward median | MOMENT/H F+B median |
|---|---:|---:|
| ordinary_20 | 1.447711 | 1.870242 |
| maximum_20 | 1.986526 | 2.074119 |
| ordinary_40 | 2.118054 | 1.900284 |
| maximum_40 | 2.354588 | 2.092785 |

Each selected-arm median uses three measurements for the same task. All ratios are below the preregistered 3x limit and include the new b/u telemetry. The old reference has only one measured pair and lacks the new u telemetry; those measurements cannot establish a controlled pure-kernel speedup.

The minimum sampled free GPU memory was 7926 MiB, above the 2445.45 MiB reserve. Unavailable process memory remains NA. Sampling does not certify that every transient peak was observed.

Both H_CONT and MOMENT restored model/optimizer hashes matching their initial metadata, with the persisted RNG-restoration conjunction true. This audit found no implementation or numerical reason to block the authorized training. Separate initial Euler replay and other parent-flow gates remain required.

Encoding repair: the unreadable prior Markdown is preserved as precheck_review.encoding_failed.md. The underlying JSON, scientific evidence, and source code were not changed; the audit was not rerun.
