"""Exact inherited Native runtime; reuse prior max-N qualification, no probes."""
from pathlib import Path
from r2.pilots.bench_natural_n_recipe_v1 import runtime as inherited
from r2.pilots.bench_natural_n_recipe_v1.runtime import (
    NativeModel, build_model, native_execution, edge_features, inference_layout,
    supervised_sample, sample_for_training, verify_sample, VERSION, N_MIN, N_MAX,
)


def cpu_gate():
    from . import common as C
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only gate must not initialize CUDA')
    prior_paths=[C.WIDE_ROOT/'runtime_cpu_gate.json',C.WIDE_ROOT/'training_cpu_gate.json']
    receipts=[C.read(p) for p in prior_paths]
    for receipt in receipts:
        if receipt['status']!='PASS':raise ValueError('Prior runtime/maximum-N gate was not passed')
        for p,h in receipt['source_hashes'].items():
            if C.sha(p)!=h:raise ValueError('Previously qualified source changed: '+p)
    if receipts[0]['manifest_sha256'] != C.sha(C.MANIFEST):
        raise ValueError('Data changed after maximum-N qualification')
    names=('NativeModel','build_model','native_execution','edge_features','inference_layout',
           'supervised_sample','sample_for_training','verify_sample')
    if any(globals()[n] is not getattr(inherited,n) for n in names):
        raise ValueError('Runtime object identity changed')
    result=dict(status='PASS',schema='bench_group_weight_match_runtime_gate_v1',
        runtime_version=VERSION,N_contract=[128,799],same_function_objects=True,
        reused_maximum_N=776,reused_boundary_N=799,
        inherited_gate_hashes={str(p):C.sha(p) for p in prior_paths},
        source_hashes={str(Path(__file__)):C.sha(__file__),str(Path(inherited.__file__)):C.sha(inherited.__file__)},
        manifest_sha256=C.sha(C.MANIFEST),neural_model_constructions=0,
        model_forwards=0,model_backwards=0,optimizer_updates=0,production_OT_calls=0,
        tensor_weight_loads=0,CUDA_initialized=False)
    C.write_new(C.OUT/'runtime_cpu_gate.json',result)
    return result
