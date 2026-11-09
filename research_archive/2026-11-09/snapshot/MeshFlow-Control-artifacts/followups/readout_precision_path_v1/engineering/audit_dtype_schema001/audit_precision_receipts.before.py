"""CPU-only independent audit of completed precision receipts and saved arrays."""
from pathlib import Path
from collections import Counter, defaultdict
import ast, copy, hashlib, json
import numpy as np

ROOT = Path('/mnt/e/MeshFlow-Control-artifacts/followups/readout_precision_path_v1')
REPO = Path('/mnt/e/MeshFlow-Control')
OLD = ROOT.parent/'readout_effective_injection_v1'
LAYERS = [3, 6, 9, 12]
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
_hash_cache = {}
def sha(p):
    p = str(p)
    if p not in _hash_cache:
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
        _hash_cache[p] = h.hexdigest()
    return _hash_cache[p]
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def same(a, b): return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
def load_array(c, key):
    assert sha(c['array_path']) == c['array_sha256']
    with np.load(c['array_path'], allow_pickle=False) as z: return z[key].copy()
def identity(c): return c['label'], c['request_id'], c['step'], c['mode']
def old_cell(c, mode):
    return OLD/'cells'/f"{c['label']}__{c['request_id']}__s{c['step']:02d}__{mode}.json"
def node(source, name, cls=None):
    tree = ast.parse(source)
    scope = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls).body if cls else tree.body
    return next(x for x in scope if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == name)
def dump(n): return ast.dump(n, include_attributes=False)
def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def main():
    plan = read(ROOT/'plan.json')
    pfstatus = read(ROOT/'preflight/status.json')
    assert pfstatus['status'] == 'PASS'
    receipt_paths = sorted((ROOT/'preflight/cells').glob('*.json')) + sorted((ROOT/'fixed/cells').glob('*.json'))
    cells = [(p, read(p)) for p in receipt_paths]
    preflight = [(p,c) for p,c in cells if p.parent.parent.name == 'preflight']
    fixed = [(p,c) for p,c in cells if p.parent.parent.name == 'fixed']
    assert len(preflight) == 12 and len(fixed) == 48
    expected = {(l,r['request_id'],s,m) for l in plan['labels'] for r in plan['requests'] for s in [0,25,45] for m in ['on','off']}
    assert {identity(c) for p,c in fixed} == expected
    current_sources = read(ROOT/'execution_source_hashes.json')
    own = read(ROOT/'precision_source_hashes.json')['files']
    for r in own:
        assert sha(REPO/r['path']) == r['sha256']
    source_paths = [REPO/x for x in [
        'src/meshflow_control/precision.py','src/meshflow_control/models/native.py',
        'src/meshflow_control/models/readout.py','src/meshflow_control/models/geometry.py',
        'src/meshflow_control/models/backbone/equidit.py','src/meshflow_control/models/backbone/attention.py',
        'src/meshflow_control/models/backbone/utils.py','src/meshflow_control/runtime.py',
        'tools/run_readout_precision.py']]
    for p in source_paths: assert sha(p) == current_sources[str(p)]

    # Original block, final head, embedding and forward control flow are intact.
    native_path = 'src/meshflow_control/models/native.py'
    before = (ROOT/'source_before'/native_path).read_text()
    after = (REPO/native_path).read_text()
    ast_checks = {}
    for name, cls in [('native_block_forward',None),('native_final_forward',None),('_embed','NativeModel')]:
        ast_checks[name] = dump(node(before,name,cls)) == dump(node(after,name,cls))
    a = copy.deepcopy(node(after,'forward','NativeModel'))
    removed = 0
    for item in ast.walk(a):
        if isinstance(item,ast.Call) and isinstance(item.func,ast.Name) and item.func.id == 'apply_readout':
            options = [k for k in item.keywords if k.arg is None and isinstance(k.value,ast.Call)
                       and isinstance(k.value.func,ast.Name) and k.value.func.id == 'readout_execution_options']
            assert len(options) == 1
            item.keywords.remove(options[0]); removed += 1
    ast_checks['forward_identical_except_explicit_readout_options'] = removed == 1 and dump(a) == dump(node(before,'forward','NativeModel'))
    assert all(ast_checks.values())

    receipt_summaries=[]; probe_cells=[]; layers=[]; fp32_raw_layers=[]
    metadata_caveats=[]
    for path,c in cells:
        assert c['status'] == 'COMPLETE'
        assert c['checkpoint_sha256'] == plan['checkpoints'][c['label']]['sha256']
        assert sha(c['array_path']) == c['array_sha256']
        is_fixed = path.parent.parent.name == 'fixed'
        assert c['precision'] == ('FP32_REFERENCE' if is_fixed else c['precision'])
        expected_dtype = 'torch.float32' if c['precision'] == 'FP32_REFERENCE' else 'torch.bfloat16'
        assert c['velocity_dtype'] == expected_dtype
        with np.load(c['array_path'],allow_pickle=False) as z:
            state=z['state'];velocity=z['velocity'];nxt=z['next_state']
            assert state.dtype == velocity.dtype == nxt.dtype == np.float32
            assert state.shape == velocity.shape == nxt.shape == (1,c['N'],9)
            assert ah(state) == c['state_sha256'] and ah(nxt) == c['next_state_sha256']
            assert np.isfinite(state).all() and np.isfinite(velocity).all() and np.isfinite(nxt).all()
            K=c['K'];assert same(state[:,:K],nxt[:,:K])
            expect=state.copy();expect[:,K:]=state[:,K:]+velocity[:,K:]*(1/50)
            assert same(expect,nxt)
            if is_fixed:
                old=read(old_cell(c,c['mode']))
                assert ah(state) == old['input_state_sha256']
                assert c['checkpoint_sha256'] == old['checkpoint_sha256']
            if is_fixed and c['mode'] == 'on':
                assert [r['layer'] for r in c['layer_metrics']] == LAYERS
                for r in c['layer_metrics']:
                    assert all(r[k] == 'torch.float32' for k in ['hidden_dtype','residual_dtype','cast_dtype','output_dtype'])
                    assert r['corner_count'] == 3*(c['N']-c['K']) and r['hidden_width'] == 768
                    assert r['element_count'] == 3*(c['N']-c['K'])*768
                    assert r['array_hashes']['residual'] == r['array_hashes']['cast']
                    if r['arrays_saved']:
                        values={k:z[f"l{r['layer']}_{k}"].copy() for k in ['before','residual','cast','after']}
                        assert all(a.dtype == np.float32 and a.shape == (3*(c['N']-c['K']),768) for a in values.values())
                        assert all(ah(v) == r['array_hashes'][k] for k,v in values.items())
                        assert same(values['residual'],values['cast'])
                        assert same(np.add(values['before'],values['cast'],dtype=np.float32),values['after'])
                        fp32_raw_layers.append(dict(call_id=c['call_id'],layer=r['layer'],actual_addition_cpu_replay_bitwise=True))
                    layers.append(dict(call_id=c['call_id'],label=c['label'],request_id=c['request_id'],
                                       step=c['step'],layer=r['layer'],actual_dtypes={k:r[k] for k in ['hidden_dtype','residual_dtype','cast_dtype','output_dtype']},
                                       free_corners=r['corner_count'],arrays_saved=r['arrays_saved'],source_hashes=r['array_hashes']))
            else: assert not c['layer_metrics']
        records=c['dtype_records']
        if records:
            sites=[r['site'] for r in records]
            required=['backbone.x_embedder','context_encoder','native_embedding_','backbone.t_embedder',
                      'backbone.y_embedder','backbone.final_layer.norm_final','backbone.final_layer.linear','native_final_final']
            for i in range(12):
                required += [f'backbone.layers.{i}.'+suffix for suffix in ['norm1','norm2','attn','attn.qkv_proj','attn.out_proj','mlp.net.0','mlp.net.2','adaLN_modulation.1']]
                required += [f'native_block_{i+1}']
            if c['label'] == 'C1000': required += ['native_clean_memory_']
            assert set(required) <= set(sites)
            actual_readouts=sorted({int(s.split('.')[1]) for s in sites if s.startswith('readouts.')})
            assert actual_readouts == ([] if c['mode']=='off' else LAYERS)
            if c['mode'] != 'off':
                for layer in LAYERS:
                    assert all(f'readouts.{layer}.{x}' in sites for x in ['norm','W_Q','W_K','W_V','W_O'])
            floats=sorted({d for r in records for k in ['input_float_dtypes','output_float_dtypes'] for d in r[k]})
            if c['precision']=='FP32_REFERENCE': assert floats == ['torch.float32']
            else: assert floats == ['torch.bfloat16','torch.float32']
            probe_cells.append(dict(call_id=c['call_id'],precision=c['precision'],mode=c['mode'],label=c['label'],
                request_id=c['request_id'],step=c['step'],record_count=len(records),
                actual_floating_dtypes=floats,readout_layers_actually_executed=actual_readouts,required_sites_all_observed=True,
                receipt_path=str(path),receipt_sha256=sha(path)))
        if not is_fixed and c['cell_id'].endswith('__off') and c['mode']!='off':
            metadata_caveats.append(dict(call_id=c['call_id'],field='cell_id',saved=c['cell_id'],
                interpretation='Legacy metadata helper suffix is off; authoritative call_id, mode, precision, path and numerical comparison identify this preflight ON/ZERO call. Original receipt remains unchanged.'))
        receipt_summaries.append(dict(path=str(path),sha256=sha(path),call_id=c['call_id'],
                precision=c['precision'],mode=c['mode'],velocity_dtype=c['velocity_dtype'],
                array_path=c['array_path'],array_sha256=c['array_sha256'],dtype_records=len(records)))

    assert len(layers)==96 and len(fp32_raw_layers)==16 and len(probe_cells)==20
    index={identity(c):c for p,c in fixed}
    replay=[];zero=[]
    for path,c in preflight:
        ref=read(old_cell(c,'on' if c['mode']=='on' else 'off')) if c['precision']=='LEGACY_BF16' else index[(c['label'],c['request_id'],c['step'],'off')]
        exact=same(load_array(c,'velocity'),load_array(ref,'velocity'))
        assert exact and same(load_array(c,'state'),load_array(ref,'state'))
        record=dict(call_id=c['call_id'],label=c['label'],request_id=c['request_id'],precision=c['precision'],
                    same_state=True,velocity_bitwise=True,reference_array=ref['array_path'],reference_array_sha256=ref['array_sha256'])
        (replay if c['mode']=='on' else zero).append(record)
    assert len(replay)==4 and len(zero)==8
    # ON/OFF execute the identical set and multiplicity of non-readout modules.
    nonreadout_pairs=[]
    for p,c in fixed:
        if c['mode']!='on' or not c['dtype_records']: continue
        off=index[(c['label'],c['request_id'],c['step'],'off')]
        get=lambda x:Counter(r['site'] for r in x['dtype_records'] if not r['site'].startswith('readouts.'))
        assert get(c)==get(off)
        nonreadout_pairs.append(dict(label=c['label'],request_id=c['request_id'],step=c['step'],
                                    identical_nonreadout_site_multiplicity=True,
                                    off_readout_module_records=0))
    assert len(nonreadout_pairs)==4

    legacy=[]
    for label,request,step,mode in sorted(expected):
        if mode!='on':continue
        c=read(OLD/'cells'/f'{label}__{request}__s{step:02d}__on.json')
        assert c['readout_sites_visited']==LAYERS
        assert c['prediction_dtype']=='torch.bfloat16'
        for r in c['layer_checks']:
            assert r['hidden_dtype']==r['output_dtype']=='torch.bfloat16'
            assert r['residual_dtype']=='torch.float32'
            assert r['query_is_actual_free'] and r['known_padding_unchanged'] and r['actual_add_matches']
            legacy.append(dict(label=label,request_id=request,step=step,layer=r['layer'],
                               hidden_dtype=r['hidden_dtype'],residual_dtype=r['residual_dtype'],output_dtype=r['output_dtype'],
                               evidence='Original saved per-layer runtime receipt; not rerun'))
    assert len(legacy)==96
    integrity=[]
    for phase in ['preflight','fixed-state']:
        for label in plan['labels']:
            p=ROOT/'integrity'/f'{phase}__{label}.json';d=read(p)
            assert d['unchanged'] and not d['optimizer'] and d['before']==d['after'] and d['RNG_before']==d['RNG_after']
            integrity.append(dict(path=str(p),sha256=sha(p),phase=phase,label=label,parameter_state_unchanged=True,RNG_unchanged=True))
    site_summary=defaultdict(lambda:dict(input=set(),output=set(),occurrences=0))
    for path,c in cells:
        for r in c['dtype_records']:
            s=site_summary[(c['precision'],r['site'])];s['input'].update(r['input_float_dtypes']);s['output'].update(r['output_float_dtypes']);s['occurrences']+=1
    summary=[dict(precision=p,site=site,input_float_dtypes=sorted(v['input']),output_float_dtypes=sorted(v['output']),occurrences=v['occurrences']) for (p,site),v in sorted(site_summary.items())]
    audit=dict(status='PASS',schema='readout_precision_dtype_audit_v1',auditor='coupling independent CPU receipt/array/source audit',
        model_calls=0,backwards=0,optimizer_updates=0,new_noise=0,
        source_script=str(Path(__file__).resolve()),source_script_sha256=sha(__file__),
        counts=dict(new_receipts=60,fixed_FP32_receipts=48,preflight_receipts=12,dtype_probed_calls=20,
                    FP32_dtype_probed_calls=12,legacy_dtype_probed_calls=8,new_FP32_ON_layer_records=96,
                    representative_FP32_full_layers_CPU_addition_verified=16,cached_BF16_ON_layer_receipts=96,
                    legacy_replays_bitwise=4,zero_matches_same_precision_OFF_bitwise=8),
        scope='Full dtype hooks on the first P1 and first P2 request at t=.5 for each model; all48 fixed outputs check FP32 velocity and all24ON cells check four-layer actual dtypes. Other40 fixed calls do not have complete module-hook traces; do not claim they do.',
        dtype_probe_schema='Root DtypeProbe: site,input_float_dtypes,output_float_dtypes; dtype sets do not encode every tensor shape/value.',
        site_summary=summary,probed_calls=probe_cells,layer_records=layers,representative_saved_addition_checks=fp32_raw_layers,
        legacy_layer_receipts=legacy,legacy_replay_checks=replay,zero_checks=zero,nonreadout_path_checks=nonreadout_pairs,
        integrity_receipts=integrity,source_AST_checks=ast_checks,receipt_sources=receipt_summaries,
        source_files=[dict(path=str(p),sha256=sha(p)) for p in source_paths],
        metadata_caveats=metadata_caveats,
        limitations=['Kernel internal accumulation is inferred from frozen PyTorch operators/math-SDPA policy, not hardware-instruction tracing.',
          'FP32 is a numerical reference, not a correct-chair target. Its whole-forward intervention changes h and r, not only one residual addition.',
          'Zero applied tensors are established by frozen source, 8 CPU contracts and actual ZERO=OFF predictions. GPU ZERO probes did not save separate applied-zero arrays.',
          'Legacy default numerical identity was actually replayed on four registered t=.5 fixtures, not every possible input or every old trajectory step.'])
    contract=dict(status='VERIFIED_FOR_DECLARED_SCOPE',schema='readout_precision_contract_v1',
        modes=dict(LEGACY_BF16=dict(parameters='float32',coordinates_and_Euler='float32',backbone='legacy CUDA BF16 autocast; FP32 stable suboperations retained',readout='FP32 LN/Q/K/V/math-SDPA/O',addition='query + residual.to(hidden.dtype)',observed_output='torch.bfloat16'),
                   FP32_REFERENCE=dict(parameters='same float32 values',coordinates_and_Euler='float32',autocast='CUDA and caller CPU autocast explicitly disabled',key_floating_path='coordinate embedding, role/Geo fusion, time/count, modulation, norm/attention/MLP, hidden, readout/addition, final head actually float32 on instrumented calls',matmul='highest, TF32 disabled during execution',observed_output='torch.float32; not a final BF16-to-FP32 relabel')),
        readout=dict(on='Four original learned residuals at blocks3/6/9/12',off='Return unchanged hidden at only those four extra apply_readout sites; branch computation skipped',zero='Compute true branch/cast, add torch.zeros_like(cast) at same site; no parameter mutation',
                     clean='Fresh known H0 after coordinate+role+Geo in the same forward, not detached or cross-step cached',
                     mixed='Known hidden at the corresponding original block output'),
        unchanged=dict(original_block_final_embedding_AST=True,forward_AST_except_explicit_readout_options=True,
                       attention_backend='same math SDPA; no FlashAttention switch',prediction='native velocity; no extra x1/velocity conversion',
                       integration='x_free + velocity_free.float()*(1/50), original C written unchanged',
                       parameters_and_rng='Saved preflight/fixed-state cryptographic integrity receipts unchanged, no optimizer constructed'),
        audit_path=str(ROOT/'dtype_audit.json'),audit_sha256=None,
        numerical_gate=dict(status='PASS',legacy_ON_exact_replays=4,zero_equals_same_precision_OFF=8,FP32_OFF_references_consumed_from_fixed48=4,
                            scope='B/C x first P1/P2 x t=.5. This preserves default behavior in those four tested legacy fixtures, not a universal bitwise proof.'),
        interpretation=['OFF is not unconditional, A or START: the backbone was jointly trained with readout.',
                        'Changes under FP32_OFF measure backbone precision effects; ON/OFF interaction includes the full precision pathway.',
                        'Larger branch differences or more faithfully represented additions do not prove useful condition integration or better geometry.',
                        'Complete rollout and visual/geometry results are separate evidence and are not judged by this dtype contract.'],
        source_files=audit['source_files'],source_script_sha256=sha(__file__))
    write(ROOT/'dtype_audit.json',audit)
    contract['audit_sha256']=sha(ROOT/'dtype_audit.json')
    write(ROOT/'precision_contract.json',contract)
    print(json.dumps(dict(status='PASS',counts=audit['counts'],dtype_audit_sha256=sha(ROOT/'dtype_audit.json'),
                          precision_contract_sha256=sha(ROOT/'precision_contract.json'),metadata_caveats=len(metadata_caveats)),indent=2))

if __name__=='__main__':
    try:main()
    except BaseException as ex:
        write(ROOT/'precision_contract_audit_error.json',dict(status='BLOCKED',error=repr(ex),source_script_sha256=sha(__file__),model_calls=0))
        raise
