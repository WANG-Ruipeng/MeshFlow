"""Session B CPU-only receipt of the frozen repair_v2 data; zero OT/model calls."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
import json
import time
from collections import Counter
from pathlib import Path
from native_t1 import runtime
import torch
from native_t1.artifacts import atomic_json,file_sha256
from .train_contract import ROOT,read_binding
from .train_stream import FrozenStream
from .train_io import resource_snapshot
from .model_mvp import json_hash

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if torch.cuda.is_initialized(): raise RuntimeError('Data receipt must not initialize CUDA')
    root=ROOT
    before=resource_snapshot()
    atomic_json(root/'registration/train_data_resources.json',before)
    tick=time.perf_counter()
    binding,protocol,plan_path,data_root=read_binding(root)
    ready=json.loads((root/'DATA_READY.json').read_text())
    if ready.get('data_revision')!='repair_v2':
        raise ValueError('This receipt explicitly acknowledges repair_v2')
    stream=FrozenStream(binding,plan_path,data_root)
    counts=Counter((r['uid'],r['patch_id']) for r in stream.records)
    assert len(stream.objects)==32 and len(stream.cases)==128 and len(stream.records)==8000
    assert all(n in (62,63) for n in counts.values())
    assert stream.counts==dict(actual_OT_attempts=0,actual_OT_returns=0,OT_cache_hits=0)
    assert not torch.cuda.is_initialized()
    receipt=dict(schema='control_mvp48_train_data_ack_v1',owner='Session B',
        status='DATA_BOUND_CPU_CONSUMER_PASS',data_revision=ready['data_revision'],
        data_ready_sha256=binding['data_ready_sha256'],contract_sha256=binding['accepted_contract_sha256'],
        manifest_sha256=protocol['manifest_sha256'],train_plan_sha256=protocol['plan_sha256'],
        binding_path='registration/train_data_binding.json',binding_sha256=protocol['binding_sha256'],
        files_verified=len(binding['files']),splits=ready['splits'],
        loaded_training_objects=len(stream.objects),loaded_training_patches=len(stream.cases),
        effective_batches=1000,effective_batch_size=8,sample_records=8000,
        plan_N_range=[min(r['N'] for r in stream.records),max(r['N'] for r in stream.records)],
        plan_K_range=[min(r['K'] for r in stream.records),max(r['K'] for r in stream.records)],
        pair_usage_range=[min(counts.values()),max(counts.values())],
        all_record_order_N_K_checked=True,training_evaluation_disjoint=True,
        coordinate_contract='pre_ot_full * 2 == model target; C/full_target not rescaled',
        label_contract='source_face_ids use retained dense rows; original_* are provenance only',
        feature_statistics_fitted=False,actual_OT_calls=0,model_forwards=0,optimizer_updates=0,
        CUDA_initialized=False,seconds=time.perf_counter()-tick,
        supersedes_data_blocked=ready.get('supersedes_blocked'),
        remaining_blockers=['OFFICIAL_CHAIR_ASSET_MISSING'],
        next_action='Resolve and SHA256-verify official Chair; then original bounded precheck and single-worker 2500-update tree.')
    atomic_json(root/'registration/train_data_ack.json',receipt)
    atomic_json(root/'registration/train_protocol.json',dict(protocol=protocol,protocol_sha256=json_hash(protocol),
        status='DATA_BOUND_AWAITING_OFFICIAL_CHAIR'))
    handoff_path=root/'registration/train_handoff.json'
    previous=json.loads(handoff_path.read_text())
    history=root/'registration/train_handoff_before_repair_v2.json'
    if not history.exists(): atomic_json(history,previous)
    previous.update(status='WAITING_OFFICIAL_CHAIR',data_revision='repair_v2',DATA_READY_exists=True,
        accepted_data_contract=dict(path=ready['contract']['path'],sha256=ready['contract']['sha256']),
        data_ready_sha256=binding['data_ready_sha256'],data_binding='registration/train_data_binding.json',
        data_ack='registration/train_data_ack.json',remaining_blockers=receipt['remaining_blockers'])
    atomic_json(handoff_path,previous)
    missing=json.loads((root/'registration/train_dependency_audit.json').read_text())
    history=root/'registration/train_dependency_before_repair_v2.json'
    if not history.exists(): atomic_json(history,missing)
    missing.update(status='WAITING_OFFICIAL_CHAIR',data_status='BOUND_REPAIR_V2',
        data_ready_sha256=binding['data_ready_sha256'],data_ack='registration/train_data_ack.json')
    if 'data_blocked' in missing: missing['historical_data_blocked_superseded']=missing.pop('data_blocked')
    missing['action']=receipt['next_action']
    atomic_json(root/'registration/train_dependency_audit.json',missing)
    lines=['# Session B data requirements and receipt','',
        'Current data: repair_v2, all '+str(len(binding['files']))+' frozen file hashes and sizes verified.',
        'B consumed 32 training objects, 128 training patches and 8000 fixed records using FrozenStream.',
        'No OT, model forward, optimizer update or CUDA initialization occurred in this receipt.','',
        '| Input | Required role | Current status |','|---|---|---|',
        '| DATA_READY.json | Freeze identity and hash inventory | Bound |',
        '| registration/data_contract_repair_v2.json | Data API and retained/source index semantics | Bound |',
        '| data/manifest.json, splits.json, patches.json | Object identity, 32/8/8 split, patch membership/interfaces | Verified |',
        '| data/meshes/<uid>.npz | Original provenance, retained topology, FP32 target/C/pre-OT coordinates | 48 frozen, train32 consumed |',
        '| data/train_plan.json | 1000 batches x8; UID/patch/Gaussian/t/permutation seeds | 8000 records validated |',
        '| data/load_sample.py | Shared exact loaders and stateless RNG recipes | Hash pinned and consumed |',
        '| data/independent_repair_validation.json | Independent audit | PASS |',
        '| Official Chair last.pt | Official EMA backbone only | Missing |','',
        'Per-record plan fields: sample_index, effective_batch_index, slot_index, uid, patch_id, N, K, gaussian_seed, t_seed, permutation_seed.',
        'Mesh/patch fields consumed: model_vertices, source_face_vertex_ids, pre_ot_full, full_target, constraints, source_face_ids, free_source_ids, bbox_diagonal, interface.',
        'N counts retained genuine faces. source_face_ids/free_source_ids index retained dense rows; original_* fields are traceability only.',
        'pre_ot_full is captured before official final x2. full_target and C already include x2; they are never rescaled.',
        'Validation/test files are hash-verified but never used to fit statistics, select training batches, or train. Geo fits no statistics.','',
        'Remaining asset: 2,084,287,208 bytes; SHA256 bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d.',
        'Historical registered location: /mnt/c/Users/warpwang/Downloads/last.pt. Bench/old tuned models cannot substitute.',
        'After this exact asset is restored, the registered real-data prechecks and 2500 updates can run; no extra arm or retry was added.','',
        'Receipt: registration/train_data_ack.json. Binding: registration/train_data_binding.json.']
    (root/'training/TRAIN_DATA_REQUIREMENTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
