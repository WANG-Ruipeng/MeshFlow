# HINGE opposite-winding guard

Retained research profile: **PLAIN_S2000 + HINGE 1.5%L**, with **2.5%L** as a comparison. This does not change the maintained sampling default.

`engine.decode(..., require_opposite_winding=True)` checks **every newly induced two-face indexed edge**, including collateral CF/FF edges produced by a whole-cluster merge. The nominated hinge was already checked by the preceding implementation. Existing same-directed defects are retained; this operation does not reorient or delete faces.

## Fresh checkout

The experiment historically imported local CPU modules not installed as part of the maintained package. The byte-preserving dependency closure, original scorers, explicit array replay entry point and full results are published in [the research archive](../../research/hinge_opposite_v1/README.md). Use its `replay_array.py` for a standalone CPU array replay; it explicitly loads the bundled legacy mathematics rather than relying on another checkout on the author's machine.

Inputs must include the original GEO corner-cluster mapping, original source face/corner identity, seam anchors, pre-GEO coordinates and saved pair classifications. A mesh file alone is insufficient. Missing assets are blocked; the code does not run a model to reconstruct them. No checkpoint is included.

## Original workspace recovery

With the frozen experiment assets and historical local adapters present:

```bash
python -B -m r2.tools.hinge_opposite_v1.run run --workers 4
python -B -m r2.tools.hinge_opposite_v1.analyze
python -B -m r2.tools.hinge_opposite_v1.audit
python -B -m r2.tools.hinge_opposite_v1.gallery
python -B -m r2.tools.hinge_opposite_v1.report
```

The finite runner verifies completed cells and original hashes, skips completed work and refuses silent retries of incomplete attempts. It processes only the registered 28 saved S2000 requests at the two caps. There are no GPU, model, noise-generation or training calls.

## Evidence boundary

The new 1.5%L endpoint improves aggregate FF/CF, singleton components and legal shared-edge counts on the fixed panel. All 4,624 original legal edges are retained, with 105 added legal edges and no new same-directed two-face edge, illegal pair or deleted face. This is **not all-metric dominance**: one I tail and some q/angle/area-ratio costs remain; DEV bulk reach at least 30% remains 0/12. See the full per-output tables and report in the archive.
