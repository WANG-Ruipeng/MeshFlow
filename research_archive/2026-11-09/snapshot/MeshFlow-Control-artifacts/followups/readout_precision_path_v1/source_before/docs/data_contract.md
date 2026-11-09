# Frozen Chair data and training contract

This document describes the implemented import and training stream. [reproduction.md](reproduction.md) provides the actual CLI commands. Data, receipts and caches belong in the external artifact directory; the original repository and its assets remain read-only.

## Imported assets and identities

The import tool reads the original repair_v2 arrays and 453 READY tasks for the same 32 training parents. It copies only those parent arrays and C partitions into the configured external data directory. It does not execute historical Python, renormalize meshes, select new C patches or repair geometry. Parent NPZ bytes, FP32 target coordinates, original pre-OT coordinates, source corner indices and retained-to-original face maps are checked. Runtime paths in the imported manifest are relative to that manifest; original source paths occur only in the external provenance receipt.

The task bank has 250 READY tasks at 20% and 203 at 40%. These percentages refer to face counts: K = floor(ratio*N + .5). The historical available ratio is sampled uniformly for each parent, then one original task within that ratio is sampled uniformly. Known patches are connected through complete source edges. Original interfaces are reference metadata, never model inputs.

The recorded import produced 32 parents and 453 tasks with before/after hashes for the actual source dependencies. It built no model and performed no OT. `train_manifest.json`, `source_identity.json` and `import_receipt.json` are the authoritative local evidence; copying this document does not recreate those assets.

## Frozen random stream and augmentation

The namespace is `NATIVE_MESH_DIRECT_CONTEXT_READOUT_V1/train/v1`. Its SHA256 first four bytes, interpreted little endian, give seed **142732150**. Independent domain hashes select parents, tasks, alpha, Z, epsilon, time and permutations.

There are 250 shuffled cycles over all 32 parents: 1,000 batches of eight different parents, exactly 250 exposures per parent. Each batch has four alpha=1 samples and four independent Uniform[.9,1.1] samples, shuffled within the batch. FP32 alpha scales only X in the saved target/pre-OT/vertex arrays; there is no new normalization. The same transformed source defines both the known condition and its supervised free target.

Z and epsilon use independent local NumPy MT19937 standard-normal draws, originally float64. Time uses an independent MT19937 normal, passed through sigmoid and converted to scalar FP32. Augmentation uses the original three cyclic corner choices and the final full-face permutation. Randomness is keyed by registered identity, not worker ID, arm, completion order or model output.

## Matching and HYBRID path

OT compares original FP32 pre-OT free targets with float64 Z using the original Euclidean cdist, six triangle-corner permutations and Hungarian face assignment. The final target is multiplied by two only after this matching, then moved into the original iid Z slots using inverse face and corner maps.

The mixture is evaluated in FP32:

```text
Z_h = sqrt(.75) * Z + .5 * epsilon
X_t = (1 - t) * Z_h + t * Y_OT
u   = Y_OT - Z_h
```

There is no second OT after mixing. This is a training coupling; inference starts from the registered Gaussian and does not perform OT or the training mixture.

Exactly 8,000 maps are reserved in this run's external OT directory. A/B/C share the same maps and frozen input/label hashes. Missing maps may be built once; an attempt without a completed map blocks silent rematching. Completed cache entries are validated before reuse. CPU scheduling can vary within the declared worker limit, but map keys and tensor identities cannot.

The recorded preparation completed 8,000 matching calls with zero failures and zero model calls. `train_plan.json`, `paired_inputs.json`, `implementation_frozen.json` and `precompute_receipt.json` retain the exact identities and counts. Historical matching caches are not silently substituted.

## Model input boundary and objective

The positional model-input allowlist is `(xt, t, y, valid_mask, known_mask)`, where y is explicit total N. Known C coordinates occupy the known slots of xt. Supervision dictionaries are never passed to the model. UID, full target, Gamma, source mappings and source bounding box are not model inputs.

Each update contains eight microbatches of one example. The FM sum of squared free-coordinate residuals is divided by the **total valid free scalar count across all eight examples**. It is not the unweighted mean of eight per-example means. Known and padding coordinates have zero contribution and zero direct FM gradient.

Training keeps every parameter trainable in eval mode, with FP32 parameters, the original BF16 backbone execution and strict AdamW moments. Existing parameters begin at optimizer step 5,000; the added B/C readout group begins at step zero. AdamW uses lr 1e-5, betas (.9,.95), weight decay zero and gradient clipping at 1. There is no auxiliary loss, scheduler, EMA or feedback replay. Each update has eight forwards, eight backwards and one optimizer update.

## Checkpoints and recovery

The fixed checkpoints are pilot 500 and 1,000, corresponding to conditional totals 5,500 and 6,000. Requested signal-safe update boundaries can also save a recovery checkpoint. Each checkpoint contains model, optimizer, RNG, stream cursor, configuration and data identities.

Resume requires exact config/data/stream identity and durable logs covering every completed update. Incomplete or uncovered execution cannot be silently replayed. Each actual microbatch is checked against its frozen input, label, matching and free-coordinate identities before use.

Synthetic CPU tests build no real model, load no checkpoint and perform no optimizer update. They cover independent random streams, matching direction, HYBRID ordering, free/padding normalization and recovery contracts. The importer and OT preparation are also model-free. These checks establish implementation contracts, not successful GPU execution or generation quality.

## Evaluation and export are separate contracts

P1 uses the eight registered DEV parents with original conditions and frozen OLD/NEW Gaussian identities. P2 uses the fixed designed control meshes and has no source completion GT. Evaluation reference data remain separate from the model input bank.

At inference, C must be finite FP32 with shape [K,3,3] or [K,9]; explicit total N is 128..256 and 0 < K < N. All 51 Euler states and the final NPZ/PLY preserve the known triangle coordinates, order, winding and multiplicity exactly. Export uses a triangle soup with explicit face indices; it does not claim retention of the original shared vertex-ID graph. No welding, repair, hole filling, smoothing or quantization is applied. Exact C preservation does not guarantee that generated free faces connect to it.
