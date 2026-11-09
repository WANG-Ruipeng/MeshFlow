# Metric numerical implementation revision 002

This revision preserves the registered metric formulas, tau/theta/rho values, degeneracy and ray tolerances, query seeds/counts, and fixed 26/74 directions. It makes two uncertainty paths conservative.

A determinant rejection does not prove ray/triangle parallelism. A finite forward ray-AABB possibility is now UNKNOWN. The analytic skinny triangle regression has area 0.4 and an interior hit at t=1, but the original implementation returned VISIBLE; v1r1 returns UNKNOWN.

Undefined C query normals leave reference eligibility unresolved. Their presence now widens the auxiliary global exposure interval to [0,1]. The primary value on the definite-reference denominator and NA when that denominator is empty are retained.

The original ten required fixtures and seven supplemental tests all pass with identical values. Four revision regressions pass. No model, backward, optimizer, generation, or formal model raw operation occurred. Original source, test receipts and definitions remain in this directory; the old formal scoring caches remain intact. New references and per-output controls must use separate v1r1 directories and be recomputed before reporting active results.
