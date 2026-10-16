# SEAM computational core

This package preserves the actual `SEAM_SLOT_C200` computational definitions.
The five Python modules were copied byte for byte from the completed local
prototype. [source_manifest.json](source_manifest.json) records their original
paths and SHA256 values. No runtime import refers to the original prototype or
an experiment directory. Install the parent project with `pip install ./r2`.

This is an optional research API, not a new default model or a general-purpose
end-to-end training command. Checkpoints, source meshes and experiment outputs
are separate assets. It introduces no new loss, occupancy threshold or weights.

## Modules and contracts

| Module | Responsibility |
| --- | --- |
| `slot_data` | Extract source-oriented C boundary edges and 24 C-only features; construct fixed-N layouts; build supervised owner/bulk targets with restricted bulk OT. |
| `occupancy` | Existing 24→32→1 occupancy head (833 parameters); weighted BCE with explicit frozen class weights; unchanged `logit >= 0` decision. |
| `model_adapter` | Reuse supplied Native/T1 + role + Geo generator parameters with corner-level known-time modulation and the independent-coordinate FM denominator. |
| `corner_contract` | NumPy mask validation, exact anchor restoration, Euler algebra and explicit shared-index decoding. No welding or learned computation. |
| `inference` | C-only occupancy/layout entry, rejecting extra source labels, Y and parent identity fields. It calls the supplied head, but does not load or call a generator or D0. |

At inference the only bundle keys are `C_vertices`, `C_faces_original`,
`C_faces_model` and `total_N`. Both C face tables refer to the same supplied
vertex table; the model table may permute each face's corners but cannot change
its vertex set. Boundary extraction uses the original, consistently oriented
C topology. The caller preserves this original table for final export.

Each open C edge reserves one candidate free-face slot. An occupied candidate
references the two endpoints in reverse C-edge order and evolves only its third
corner. An unoccupied candidate returns to ordinary three-corner bulk; it is
not removed. Thus total N stays fixed, with `N` in 128..256. With `K` C faces
and `S` active seams, FM has exactly `9*(N-K)-6*S` independent scalar coordinates.
Anchored and padding coordinates are excluded from numerator and denominator.
Microbatch accumulation uses the sum of independent scalars across the entire
effective batch.

Supervised `supervised_sample` is deliberately separate from C-only inference.
It uses source occupancy and legal owner identities to allocate targets to seam
slots. Those owners do not participate in bulk matching. Ordinary bulk and
returned Null slots use the existing nested face/corner OT, with the original
pre-OT half-scale, six corner permutations, and post-OT HYBRID lambda 0.25.
Multiple source edges owned by one triangle, and third-corner references to C,
are explicitly unsupported in this prototype and raise errors.

`seam_forward` is parameter-free and retains the generator's state-dict keys.
It must execute inside the package's existing `native_execution` context.
Only original C faces enter Geo and known-time face attention; seam anchor
corners use known-time MLP/final modulation while seam face attention stays at
free time. The no-seam case delegates directly to the original generator.
The supplied occupancy head has its own state and is not part of the generator
state dict. The adapter does not create optimizers, clamp integration states,
choose a checkpoint or run denoising.

`decode_shared_indices` preserves explicit C references and gives each
independent corner a distinct index, even when coordinates coincide. It does
not restore original C corner order by itself, weld free vertices, fix winding,
delete faces or certify surface quality. Downstream D0/GEO/source-winding
processing must preserve the source mapping and explicit anchor provenance;
this core alone is not a substitute for that complete export contract.

## Bounded evidence

The original experiment is a four-TRAIN-condition, 200-update pilot. It has
positive connection and visual evidence, but FF/I did not improve together.
Source-oracle occupancy is a diagnostic condition and is not the C-only method.
Shared C/F edges alone do not certify connection into the bulk, a complete
surface, freedom from intersections, or DEV generalization.

The packaged CPU tests exercise synthetic layout/OT, shared-index, source-order
and FM algebra contracts. They do not call an occupancy head, real generator
or denoiser, and do not validate generation quality or neural BF16 arithmetic:

```bash
CUDA_VISIBLE_DEVICES='' python -B -m pytest r2/tests/test_seam_contract.py -q
```
