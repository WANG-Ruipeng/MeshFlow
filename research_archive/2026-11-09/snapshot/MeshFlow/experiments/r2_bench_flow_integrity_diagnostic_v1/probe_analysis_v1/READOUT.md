# Frozen source-bridge probe readout

200 frozen forwards; zero updates/backward. These measurements use source-target interpolation states. They do not measure autonomous completion, certify generated topology, or require a unique source style.

Native models share exactly the same 60 input states. SEAM has a distinct source-owner layout and is descriptive only; its errors are not a same-input architecture comparison.

| Model | Free velocity MSE | Both-free shared-pair difference MSE | One-fixed shared-pair difference MSE |
|---|---:|---:|---:|
| COMMON_S12000 | 0.85095319 | 0.35991909 | 0.3289255 |
| NATIVE_S6000 | 0.85507146 | 0.35734836 | 0.26695487 |
| COVERAGE_S1500 | 0.80918136 | 0.32549121 | 0.24576904 |
| SEAM_S3000 | 0.14417961 | 0.23044364 | 0.1670336 |

Means: equal five fixed times, equal noises, equal conditions within parent, then equal parents. MSE divides by three scalar coordinates per corner/pair. No valence-squared pooling. Nonapplicable strata stay NA.

The primary pair mask excludes duplicate/degenerate source faces, incidence other than two, nonmanifold endpoint links, and unverified/non-legal adjacent contacts. Legality is pair-local, not a whole-source clean certificate. Slender-triangle proxy means minimum altitude / longest edge <0.1 on either incident triangle; it is not a semantic support label.

Fixed network predictions are projected to zero for the exact production-motion identity and separately logged. Source face and corner metadata reconstruct x1 exactly. The endpoint-gap identity is checked per pair, with FP32 bridge rounding explicitly tolerated.

Do not infer a broken coupling, numerical integration error, or improved free-generation quality from a change in these MSEs alone. Use the separately frozen rollout and step-sensitivity results for those questions.
