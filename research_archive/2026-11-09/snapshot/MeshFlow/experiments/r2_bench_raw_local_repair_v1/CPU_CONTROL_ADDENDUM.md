# Necessary unchanged-RAW processing control

Registered before target results or new full scoring, 2026-10-24.

The existing GEO_ONLY cache stops at GEO/source orientation. It does **not**
contain the HINGE_OPP015 -> ALPHA -> HOLE FINAL used for the new targets.
Comparing a local target without D0 to an official D0 result would therefore
confound local editing with omission of D0. This addendum completes the missing
control within the authorized CPU data feasibility work.

For the same 12 requests, reuse the frozen GEO_ONLY PRE, BASE and full pair cache
without repeating GEO or changing RAW. Run only the original remaining cascade
and original scoring, under label RAW_IDENTITY. Keep its outputs in a separate
directory. This adds 12 CPU cascade jobs and 24 BASE/FINAL scoring rows; duplicated
BASE metric values must match the existing GEO_ONLY score values. There remain
only 12 new target proposals and 12 target GEO jobs. Total new full score rows:
48. Model calls, GPU calls, new generation, noise and queries remain zero.

Primary editing comparison is LOCAL_REPAIR vs RAW_IDENTITY at the same stage.
Official/SHAPE/RELDEF are contextual references. A recovered thin support that
already survives GEO(RAW) must not be claimed as an editing gain. Likewise,
RAW-level collision removal that unchanged GEO already accomplishes is a real
coordinate modification but not evidence of additional final-backend utility.

This addendum does not change target selection, displacement bounds, repair
strategy or the majority readiness gate. It prevents a processing-chain
confound and preserves all negative results.
