# 1B fixed-budget necessary condition

This is an analytical bound derived from the immutable endpoint diagnostic CSV, not a new optimization, intersection classification or model run. It does not change the registered transform or binding limits.

For an original pre-GEO corner x, its globally affine-transformed position T(x), and a target C vertex c, triangle inequality gives:

`||x-c|| <= ||x-T(x)|| + ||T(x)-c|| <= 0.035 L + 0.02 L = 0.055 L`.

Both endpoints of any accepted existing open free edge must satisfy this. Consequently the best original whole-edge endpoint assignment, minimized over both endpoint pairings and **all axes** of the original eligible open edges, must be at most 5.5%L. This relaxes the angle, full-cluster displacement, capacity, shape and intersection guards: passing is only necessary, never sufficient.

The fixed 1B contract bounds all original pre-GEO free corners as well as POST representatives. Original C remains fixed, original open-edge topology supplies the candidates, and B2 measures its displacement against the transformed pre-GEO positions. Thus the bound applies even if a global scaling changes edge-axis qualification. It would not apply unchanged to a different contract that inserts vertices, splits edges or expands the displacement allowance.

Existing data exclude five of the eight 608/610 requests within this budget: all four 610 noises and 608 n1. The remaining 608 n0/n2/n3 are not excluded; this does not establish simultaneous feasibility of a common six-parameter transform or a legal seam.

See `affine_budget_necessary_condition.csv` for all eight rows and `affine_budget_necessary_condition.json` for the source hash. No failed optimizer result is needed for these five exclusions. This is a limited displacement/whole-edge result, not a physical impossibility theorem for mesh completion.
