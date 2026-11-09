"""CPU interface check, no models, generated noise, scores or formal panel writes."""
import ast, hashlib, json, sys, time, types
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/e/MeshFlow');sys.path[:0]=[str(ROOT),str(ROOT/'r2/src'),str(ROOT/'native_t1/src')]
from r2.pilots.bench_seat_coverage_v1 import panel as P, eval_runtime as E
from r2.pilots.bench_parts_convergence_v1 import data as D
from r2.pilots.seam_slot_c200_v1.generate import raw_arrays
from r2.pilots.seam_slot_c200_v1.postprocess import geometry_from_proposal
from r2.pilots.bench_adaptation_strength_v1.collect_v2 import numeric_fields
from r2.pilots.bench_seat_coverage_v1 import evaluate as V,collect as C,gallery as G

def run():
    start=time.perf_counter();out=E.OUT/'engineering/eval_interface_cpu_audit.json'
    if out.exists():raise RuntimeError('Do not overwrite previous interface audit')
    files=[Path(P.__file__).with_name(n+'.py') for n in ('panel','eval_runtime','sample','post_d0','evaluate','collect','gallery')]
    for p in files:ast.parse(p.read_text())
    selected=[r for r in P.read(E.OUT/'data/candidate_manifest.json')['conditions'] if r['parent_id'].startswith(('a8e987','27eb79','df6fd3'))]
    assert len(selected)==3
    records=[]
    for source in selected:
        row=dict(source,eval_group='NEW_PARENT_HOLDOUT' if source['parent_id'].startswith('df6') else 'NEW_TRAIN',
                 split='NEW_PARENT_HOLDOUT' if source['parent_id'].startswith('df6') else 'NEW_TRAIN',
                 panel_origin='NEW_COVERAGE',baseline_reuse=False)
        memory={}
        def array_file(path,values):memory[str(path)]={k:v.copy() for k,v in values.items()}
        def protect(protected,path,expected=None):
            if str(path) in memory:
                h=hashlib.sha256()
                for key,val in sorted(memory[str(path)].items()):
                    h.update(key.encode());h.update(str(val.dtype).encode());h.update(repr(val.shape).encode());h.update(val.tobytes())
                digest=h.hexdigest()
            else:
                digest=P.sha(path)
                if expected is not None:assert digest==expected
            protected[str(path)]=digest;return digest
        ns=dict(vars(P.old),OUT=Path('/virtual/no_persistent_panel'),P=types.SimpleNamespace(array_file=array_file,FAMILIES=P.P.FAMILIES),protect=protect)
        fn=types.FunctionType(P.old.new_bundle.__code__,ns)
        q=fn(row,{})
        inp=memory[q['inference_npz']];src=memory[q['source_npz']];pair=P.arrays(row['paired_path'])
        edges,_=D.edge_features(inp['C_vertices'],inp['C_faces_original'],row['N'])
        layout=D.inference_layout(inp['C_vertices'],inp['C_faces_original'],inp['C_faces_model'],row['N'],np.zeros(len(edges),bool))
        assert not len(layout['seam_slots']) and len(layout['bulk_slots'])==row['N']-row['K']
        target=np.concatenate((pair['C'],pair['Y'])).astype(np.float32)
        raw=raw_arrays(target,layout,np.full(len(edges),-1,np.float32),src)
        assert not raw['raw_seam_slot_mask'].any()
        assert raw['C'].tobytes()==src['C_original'].tobytes()
        assert np.all(raw['raw_anchor_ids'][row['K']:]<0)
        # Identity proposal is a source-array schema check, not a D0 result or quality score.
        _,_,post,_,_,anchors=geometry_from_proposal(raw,target.copy(),src)
        assert post['C'].tobytes()==src['C_original'].tobytes()
        assert anchors['C_vertex_ids_bijective'] and anchors['input_active_seam_faces']==0
        assert set(inp)=={'C_vertices','C_faces_original','C_faces_model','N','K'}
        records.append(dict(condition_id=row['condition_id'],N=row['N'],K=row['K'],boundary_edges=len(edges),
            in_memory_bundle_only=True,source_corner_restore=True,all_free_native_layout=True,
            target_identity_proposal_geometry_schema=True,model_calls=0,score_calls=0))
    old=P.read(E.SOURCE_ROOT/'eval_manifest.json');rows=[]
    for origin in old['requests']:
        q=dict(origin,panel_origin='INHERITED24',baseline_reuse=True)
        for label in (P.LABELS[0],P.LABELS[2]):
            E.verify_one_raw(q,label)
            for stage in V.STAGES:
                folder=V.score_dir(label,q,stage);r=P.read(folder/'row.json');s=P.read(folder/'score.json')['seam_to_bulk']['status']
                decorated=C.decorate_with_status(r,label,q,stage,s)
                assert decorated['boundary_edges']==r['boundary_edges']
                assert all(decorated[k]==v for k,v in r.items() if k not in {'label','stage','part_type','closed_C','seam_applicability'})
                rows.append(decorated)
                cell=G.cell(q,label,stage);assert E.resolve(cell['path']).is_file() and E.resolve(cell['receipt']).is_file()
    metrics=numeric_fields(rows);assert len(metrics)==177,(len(metrics),metrics)
    assert 'torch' not in sys.modules
    result=dict(status='PASS',new_conditions=records,existing_score_rows_verified=len(rows),original_numeric_fields=len(metrics),
        in_memory_query_bank_checks=3,persisted_new_queries=0,new_noise_arrays=0,
        source_identity_geometry_checks=3,scoring_calls=0,model_calls=0,GPU_calls=0,torch_imported=False,
        formal_panel_written=False,old_assets_written=False,seconds=time.perf_counter()-start,
        code_sha256={str(p):P.sha(p) for p in files},audit_script_sha256=P.sha(__file__))
    P.P.immutable(out,result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
