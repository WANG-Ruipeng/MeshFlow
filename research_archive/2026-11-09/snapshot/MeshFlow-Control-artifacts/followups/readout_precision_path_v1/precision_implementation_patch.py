from pathlib import Path
import hashlib
root=Path('/mnt/e/MeshFlow-Control')
expected={'src/meshflow_control/models/native.py':'f3ad51957e673e8ae9bc22253df2f536f9320ef1de1d6a8d0e0778707364955f','src/meshflow_control/models/readout.py':'047b4eee800c36de5e7e35b0b3f10d31ffbea02ac261752b9a674fd3c6331f1c'}
for rel,h in expected.items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h, 'Original source changed before minimal patch: '+rel
p=root/'src/meshflow_control/models/native.py'
s=p.read_text()
s=s.replace('from .. import runtime as _stable\n','from .. import runtime as _stable\nfrom ..precision import current_precision, readout_execution_options\n',1)
s=s.replace("native_precision='bf16' if _BF16.get() else 'fp32_gate_only',","native_precision=current_precision() or ('bf16' if _BF16.get() else 'fp32_gate_only'),",1)
s=s.replace('    ``autocast=False`` is reserved for the predefined FP32 implementation gate.','    ``autocast=False`` supports the predefined FP32 gate and the explicitly\n    selected FP32_REFERENCE fixed-weight precision context.',1)
old='                    source_mode=self.readout_mode)'
assert s.count(old)==1
s=s.replace(old,'                    source_mode=self.readout_mode, **readout_execution_options())',1)
p.write_text(s)
p=root/'src/meshflow_control/models/readout.py'
s=p.read_text()
old='                  observer=None, layer=None, source_mode=None, reference=False):'
assert s.count(old)==1
s=s.replace(old,'                  observer=None, layer=None, source_mode=None, reference=False,\n                  addition_mode="on", capture=None):',1)
old='    if hidden.ndim != 3 or hidden.shape[1] != 3 * known_mask.shape[1] or known_mask.shape != valid_mask.shape:'
assert s.count(old)==1
s=s.replace(old,'    if addition_mode not in ("on", "off", "zero"):\n        raise ValueError("Unknown readout addition mode")\n    if capture is not None and not callable(capture):\n        raise TypeError("capture must be callable or None")\n'+old,1)
old='    if memory is None:\n        memory = gather_known_corners(hidden, known_mask, valid_mask)'
assert s.count(old)==1
s=s.replace(old,'    if addition_mode == "off":\n        # The original backbone/Geo/role/CLEAN-H0 path still executes.\n        return hidden\n'+old,1)
old='        updated = query + residual.to(dtype=hidden.dtype)\n        rows.append(hidden[b].index_copy(0, indices, updated))'
assert s.count(old)==1
new='''        cast = residual.to(dtype=hidden.dtype)
        applied = torch.zeros_like(cast) if addition_mode == "zero" else cast
        updated = query + applied
        rows.append(hidden[b].index_copy(0, indices, updated))
        if capture is not None:
            # Actual free-only operands, after the output row has been copied.
            # Retain real r/cast under ZERO and separately expose applied zeros.
            capture(dict(site="readout_addition", batch_index=b, layer=layer,
                source_mode=source_mode, addition_mode=addition_mode,
                free_corner_indices=indices.detach(), before=query.detach(),
                residual=residual.detach(), cast=cast.detach(),
                applied=applied.detach(), after=updated.detach()))'''
s=s.replace(old,new,1)
p.write_text(s)
# Restore exactly the caller's declared matmul selection after explicit FP32.
p=root/'src/meshflow_control/precision.py'
s=p.read_text()
s=s.replace('        # Native policy intentionally leaves deterministic/TF32 flags strict,\n        # even if restoring an old \'high\' selection would otherwise allow TF32.\n        torch.backends.cuda.matmul.allow_tf32 = False\n        torch.backends.cudnn.allow_tf32 = False\n','')
p.write_text(s)
print('Patched native/readout explicit options; no backbone formula or source change.')
