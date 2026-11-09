# Integration into the original fork
No original workspace file was changed by this task. The independent package can be used beside the original installation with explicit external asset paths.

A later, separately authorized integration should copy the small models/readout.py module and the after-block hook in models/native.py, preserve the original variable-length coordinate/role/Geo path, and choose none/mixed/clean through a single model configuration. Four modules after blocks 3/6/9/12 must retain their own projection weights. Do not replace the old joint attention or clamp hidden known states.

Port the new versioned checkpoint schema and parameter-identity optimizer mapping together. The original strict START loader should keep its original identity restriction. The new loader must validate all candidate schema fields and tensor keys rather than use strict=False. Keep original START as the default until a separate decision authorizes promotion.

Retain the external-data importer, common frozen HYBRID stream, durable accounting, lossless NPZ/PLY export and public C/N/noise sampling interface. Never introduce runtime imports of historical runs. Preserve old scientific metric definitions and mark source-only metrics appropriately.

Validation when integration is explicitly requested: synthetic CPU contracts; exact START zero-branch comparisons; bounded real FP32 permutation/padding checks; new checkpoint strict load in a new process; exact C coordinates, winding and multiplicity through sampling/export. Reuse the recorded evidence here where identities match. Integration itself does not authorize new training or an expanded experiment matrix.
