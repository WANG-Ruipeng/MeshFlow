"""Independent read-only input/cache/contract audit; no scoring or model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import collections
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/e/MeshFlow')
RUN = ROOT / 'experiments/r2_seam_abc_v1'
OLD = ROOT / 'experiments/r2_seam_quality_four_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def verify(files):
    for p,d in files.items():assert sha(p)==d, 'Hash mismatch: '+str(p)

def main():
    t=time.perf_counter()
    out=RUN/'audit/inputs.json'
    if out.exists():
        previous=read(out);verify(previous['protected_hashes'])
        print(json.dumps({'status':'VERIFIED_EXISTING','path':str(out)}));return
    import torch
    assert not torch.cuda.is_initialized()
    # Abort if any imported API accidentally dispatches a neural module or CUDA.
    def forbidden(*a,**kw):raise AssertionError('Model/CUDA call forbidden in this audit')
    torch.nn.Module.__call__=forbidden
    torch.cuda.init=forbidden
    from r2.pilots.seam_abc_v1 import generate as G
    from r2.pilots.user_condition_finetune_v1 import evaluation as E
    root,reg,m=G.load_registered(RUN,require_execution_freeze=True)
    freeze=read(RUN/'execution_freeze.json')
    protected={**freeze['code_hashes'],**freeze['input_hashes'],str(RUN/'execution_freeze.json'):sha(RUN/'execution_freeze.json')}
    oldm=read(OLD/'eval_manifest.json')
    assert sha(OLD/'eval_manifest.json')==m['original_manifest_sha256']
    assert sha(RUN/'B_EVAL_DEV_ASSETS.json')==m['dev_manifest_sha256']
    train=[q for q in m['requests'] if q['split']=='TRAIN']
    dev=[q for q in m['requests'] if q['split']=='DEV']
    assert len(train)==16 and len(dev)==12
    assert {q['parent_id'] for q in train}.isdisjoint({q['parent_id'] for q in dev})
    assert {q['request_id']:q for q in train}=={q['request_id']:q for q in oldm['requests']}
    assert len({q['parent_id'] for q in train})==4 and len({q['parent_id'] for q in dev})==3
    input_rows=[]
    for q in m['requests']:
        bundle=G.read_bundle(q); source=G.source_from_bundle(q,bundle); z=G.read_noise(q)
        canonical=bundle['C_vertices'][bundle['C_faces_model']]
        original=bundle['C_vertices'][bundle['C_faces_original']]
        query=E.load_queries(q['query_descriptor'])
        assert E.array_sha(canonical)==q['query_descriptor']['identity']['C_sha256']
        assert len(query['arrays']['C_points'])==2048
        noise_copied=None
        if q['split']=='DEV':
            assert sha(q['source_input_path'])==q['source_input_sha256']
            saved=arrays(q['source_input_path'])
            noise_copied=same(z,saved['z'].reshape(q['N'],3,3)[q['K']:])
            assert noise_copied and same(saved['C'].reshape(q['K'],3,3),canonical)
            protected[q['source_input_path']]=q['source_input_sha256']
        input_rows.append(dict(request_id=q['request_id'],condition_id=q['condition_id'],split=q['split'],
            N=q['N'],K=q['K'],noise_index=q['noise_index'],seed=q['seed'],bundle_keys=sorted(bundle),
            input_sha256=sha(q['inference_npz']),noise_sha256=sha(q['noise_npz']),
            metadata_sha256=sha(q['metadata_npz']),source_corner_roundtrip='PASS',
            C_canonical_sha256=E.array_sha(canonical),C_source_order_sha256=E.array_sha(original),
            query_sha256=sha(q['query_descriptor']['path']),query_count=2048,
            DEV_noise_exact_source_z_free=noise_copied))
    cache_rows=[]
    for q in train:
        bundle=G.read_bundle(q); original=bundle['C_vertices'][bundle['C_faces_original']]
        for stage in ('raw','post'):
            old_dir=OLD/'evaluation/LONG'/q['request_id']/stage
            new_dir=RUN/'evaluation/LONG'/q['request_id']/stage
            oldr=read(old_dir/'receipt.json');newr=read(new_dir/'receipt.json')
            verify(oldr['files']);verify(newr['files'])
            assert newr['cache_origin']==str(old_dir)
            assert newr['cache_origin_receipt_sha256']==sha(old_dir/'receipt.json')
            assert newr['newly_generated'] is False and newr['new_rollouts']==0 and newr['new_D0_forwards']==0
            files={p.name:sha(p) for p in old_dir.iterdir() if p.is_file() and p.name!='receipt.json'}
            assert files=={p.name:sha(p) for p in new_dir.iterdir() if p.is_file() and p.name!='receipt.json'}
            a=arrays(new_dir/(stage+'.npz'))
            assert same(a['output'].reshape(-1,3,3)[:q['K']],original)
            assert same(a['vertices'][a['faces']],a['output'].reshape(-1,3,3))
            if stage=='raw':
                assert newr['noise_sha256']==sha(q['noise_npz'])
                assert newr['input_bundle_sha256']==sha(q['inference_npz'])
                assert newr['audit']['nfe']==50 and newr['source_labels_read'] is False
            protected.update({str(old_dir/'receipt.json'):sha(old_dir/'receipt.json'),str(new_dir/'receipt.json'):sha(new_dir/'receipt.json')})
            protected.update(newr['files'])
            cache_rows.append(dict(request_id=q['request_id'],stage=stage,status='BYTE_IDENTICAL',file_count=len(files),
                origin=str(old_dir),new=str(new_dir),files=files,C_source_order='EXACT',new_model_calls=0))
    layout_rows=[]
    for d in sorted((RUN/'evaluation/LONG/layouts').glob('*/receipt.json')):
        r=read(d)
        if 'cache_origin_receipt' not in r:continue
        op=Path(r['cache_origin_receipt']);o=read(op);verify(r['files']);verify(o['files'])
        assert r['condition_id']==o['condition_id'] and r['occupancy_forward_calls']==0
        for name in ('layout.npz','source_condition.npz'):assert sha(d.parent/name)==sha(op.parent/name)
        layout_rows.append(dict(condition_id=r['condition_id'],status='BYTE_IDENTICAL',new_occupancy_calls=0))
    assert len(layout_rows)==4
    oldscore=read(OLD/'evaluation/score_registration.json')
    newscore_path=RUN/'evaluation/score_registration.json'
    newscore=read(newscore_path)
    verify(newscore['source_hashes']);verify(newscore['protected'])
    assert newscore['manifest_sha256']==sha(RUN/'eval_manifest.json')
    overlap=set(oldscore['source_hashes']) & set(newscore['source_hashes'])
    assert overlap and all(oldscore['source_hashes'][p]==newscore['source_hashes'][p] for p in overlap)
    assert newscore['quality_definition']==oldscore['quality_definition']
    assert newscore['boundary_settings']==oldscore['boundary_settings']
    assert newscore['candidate_graph']==oldscore['candidate_graph']
    assert newscore['cells']==224 and newscore['registered_requests']==28
    protected.update(newscore['source_hashes']);protected.update(newscore['protected'])
    protected[str(newscore_path)]=sha(newscore_path)
    assert reg['new_rollouts']==reg['new_D0_forwards']==96
    assert reg['cached_LONG_TRAIN']==16 and reg['expected_occupancy_forwards']==24 and reg['expected_pointer_forwards']==28
    assert reg['all_RAW_before_D0'] is True
    protected[str(__file__)]=sha(__file__)
    assert not torch.cuda.is_initialized()
    value=dict(status='PASS',version='seam_abc_independent_input_cache_audit_v1',seconds=time.perf_counter()-t,
        model_calls=0,GPU_calls=0,training_updates=0,new_samples=0,rescoring_calls=0,new_noise_draws=0,
        scope='Read-only code/input/cache contracts; no claim about uncompleted outputs or final score quality',
        conditions=7,TRAIN_conditions=4,DEV_conditions=3,requests=28,arms=m['arms'],
        expected_new_rollouts=96,expected_new_D0_forwards=96,cached_LONG_requests=16,
        expected_new_occupancy_forwards=24,expected_new_pointer_forwards=28,expected_scoring_cells=224,
        inputs=input_rows,LONG_cached_stages=cache_rows,LONG_cached_layouts=layout_rows,
        scorer=dict(common_original_source_files=len(overlap),all_common_original_hashes_match=True,
            quality_definition_unchanged=True,boundary_settings_unchanged=True,free_only_graph_definition_unchanged=True),
        static_contract_review=dict(C_only='bundle contains only C vertices/faces and total N; model layouts/pointer do not read labels/Y',
            source_metadata='C-only source identity/order for export; no source free geometry used in generation',
            pointer='predicted occupancy + initial Gaussian, Null argmax; no target labels; shared tip guarantees vertex identity only',
            GEO='original GEO_C process/validator and source-winding exporter; C_SHARED_TIP clamps learned aliases before GEO',
            D0='one official proposal per newly generated request; all RAW completion barrier before any D0',
            provenance='cached LONG historical generator_forwards are historical; new_rollouts/new_D0_forwards are explicitly zero'),
        caveats=['C_SHARED_TIP alias restoration is the registered representation change, not an unchanged geometric output.',
            'B_POOL47 source-owner alias extension and exposure changes prevent describing B as pure parent-count-only ablation.',
            'Four TRAIN C were seen; three DEV C are held out from TRAIN47. This is not a large unseen-chair test.'],
        protected_hashes=protected)
    out.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:value[k] for k in ('status','seconds','requests','model_calls','rescoring_calls')}))

if __name__=='__main__':main()
