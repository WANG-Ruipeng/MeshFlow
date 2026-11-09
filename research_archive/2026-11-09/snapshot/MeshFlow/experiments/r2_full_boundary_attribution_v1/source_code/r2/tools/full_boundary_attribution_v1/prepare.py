"""Freeze the 11 preselected requests and all read-only evidence before analysis."""
from . import common as C
from . import geometry as G
import numpy as np
SELECT={'3f7417':[0,1,2,3],'73b369':[2],'2d08a6':[2,3],'85b169':[0,1,2,3]}
def main():
    target=C.OUT/'registration.json'
    if target.exists():
        r=C.read(target)
        for p,h in r['protected_hashes'].items():assert C.sha(p)==h,p
        for c in r['cells']:
            assert C.sha(c['geometry_path'])==c['geometry_sha256']
            assert C.sha(c['inherited_path'])==c['inherited_sha256']
        print('Frozen registration verified; no analysis restarted.');return r
    protected={}
    def protect(p,expected=None):
        p=C.resolve(p);h=C.sha(p)
        if expected:assert h==expected,str(p)
        protected[str(p)]=h;return p
    def read(p):return C.read(protect(p))
    base=read(C.BASE/'registration.json')
    requests=[q for q in base['requests'] if q['parent_id'][:6] in SELECT and int(q['candidate']) in SELECT[q['parent_id'][:6]]]
    requests.sort(key=lambda q:(list(SELECT).index(q['parent_id'][:6]),int(q['candidate'])))
    assert len(requests)==11
    raw={(x['request_id'],x['label']):x for x in read(C.BASE/'evaluation/raw_index.json')['rows']}
    post={(x['request_id'],x['label']):x for x in read(C.BASE/'evaluation/post_index.json')['rows']}
    scores={(x['request_id'],x['label'],x['stage']):x for x in read(C.BASE/'evaluation/score_completion.json')['receipts']}
    num=C.csvread(protect(C.BASE/'evaluation/per_output.csv'))
    vis=C.csvread(protect(C.BASE/'evaluation/gallery/AI_PROXY_reviews_v2.csv'))
    numeric={(x['request_id'],x['label'],x['stage']):x for x in num}
    reviews={(x['request_id'],x['label'],x['stage']):x for x in vis}
    cells=[]
    for q in requests:
        srcpath=protect(q['condition_source'],q['condition_source_sha256']);src=C.arrays(srcpath)
        inputpath=protect(q['input_path'],q['input_sha256']);inp=C.arrays(inputpath)
        K=int(q['K']);assert C.same(inp['C'],src['C'])
        for label in ['SMALL_C200','CREF_C200']:
            for stage in ['RAW','POST']:
                key=(q['request_id'],label,stage);rec=scores[key];scorepath=protect(rec['path'],rec['sha256']);score=C.read(scorepath)
                files={C.resolve(p).name:protect(p,h) for p,h in score['files'].items()}
                sourcegeo=files['RAW_source_direction.npz'] if stage=='RAW' else protect(post[key[:2]]['post_path'],post[key[:2]]['post_sha256'])
                a=C.arrays(sourcegeo);assert np.array_equal(a['vertices'][a['faces']],a['output'])
                assert C._unordered_equal(a['output'][:K],src['C_original'])
                assert C.same(a['output'][:K],src['C_original'])
                if stage=='RAW':
                    rawp=protect(raw[key[:2]]['raw_path'],raw[key[:2]]['raw_sha256']);old=C.arrays(rawp)
                    assert C._unordered_equal(a['output'],old['raw'].reshape(-1,3,3))
                cellid=f"{q['parent_id'][:6]}_n{q['candidate']}__{label}__{stage}"
                geom=C.OUT/'inputs/cells'/cellid/'geometry.npz';inherited=C.OUT/'inputs/cells'/cellid/'inherited.json'
                C.frozen_npz(geom,dict(C=src['C'],C_original=src['C_original'],C_source_vertex_ids=src['C_original_source_vertex_ids'],
                    C_source_corner_permutations=src['C_source_corner_permutations'],C_original_source_face_ids=src['C_original_source_face_ids'],
                    F=a['output'][K:],mesh_vertices=a['vertices'],mesh_faces=a['faces'],known_mask=np.arange(len(a['faces']))<K,
                    original_free_slot_ids=(a['source_face_ids'][K:] if stage=='POST' else np.arange(K,len(a['faces'])))))
                inherited_value=dict(score=score,pairs=C.read(files['pairs.json']),valid_edges=C.csvread(files['valid_edges.csv']),
                    old_boundary_vertices=C.csvread(files['boundary_vertices.csv']),old_boundary_distances=C.csvread(files['descriptive_boundary_distance_points.csv']),
                    numeric=numeric[key],review=reviews[key],source_evidence_files={str(p):C.sha(p) for p in files.values()})
                C.write(inherited,inherited_value)
                cell=dict(cell_id=cellid,request_id=q['request_id'],parent_id=q['parent_id'],candidate=q['candidate'],label=label,stage=stage,K=K,N=q['N'],
                    geometry_path=str(geom),geometry_sha256=C.sha(geom),inherited_path=str(inherited),inherited_sha256=C.sha(inherited),
                    source_geometry_path=str(sourcegeo),source_geometry_sha256=C.sha(sourcegeo),score_path=str(scorepath),score_sha256=C.sha(scorepath),
                    condition_source_path=str(srcpath),condition_source_sha256=C.sha(srcpath),input_path=str(inputpath),input_sha256=C.sha(inputpath),
                    graph_identity='POST_SAVED_ACTUAL_VERTEX_INDICES' if stage=='POST' else 'RAW_SAVED_EXACT_COORDINATE_OBSERVATIONAL_INDEX_SOURCE_DIRECTION')
                cells.append(cell)
    assert len(cells)==44
    cfg=dict(G.DEFAULTS);assert cfg['distance_thresholds_over_L']==[.001,.005,.01,.02]
    deps=[C.CODE/'geometry.py',C.CODE/'common.py',C.CODE/'prepare.py',C.ROOT/'r2/tools/cref_layer_audit_v1/geometry.py']
    r=dict(version=C.VERSION,status='FROZEN',utc=C.utc(),requests=requests,cells=cells,protected_hashes=protected,
      scientific_code_hashes={str(p):C.sha(p) for p in deps},geometry_config=cfg,L=C.L,
      selection=SELECT,selection_policy='Preselected observed cases for mechanism attribution, not a new prevalence/generalization evaluation.',
      zero_budget=dict(training_updates=0,model_forwards=0,new_noise=0,new_mesh_generation=0,mesh_edits=0,old_quality_rescans=0),
      sampling='Per original source-ID open edge: ceil(length/(.002L)) deterministic equal-length midpoint bins; weight=bin length. Unique boundary vertices separately zero-weight.',
      query_target='ONLY F finite triangles, all finite F edges, F-open edges, F vertices; C excluded from all distance and normal-line targets.',
      topology='Saved RAW exact-coordinate observational indices and POST actual indices; no welding or repairs. Existing score legal-edge rows and CF/FF pair decisions inherited.',
      required_interface_status='NA_NO_REQUIRED_INTERFACE; no S or SJoin success score. All-boundary distances descriptive.',
      inference_limits='Selected endpoint geometry diagnoses do not identify training/architecture causes. Source normal side is not inside/outside. Nearness does not establish safe welding or required attachment.',
      resource_plan='CPU only; 2 threads per process. First actual registered cell measures RSS/time, then at most 2 independent CPU workers if headroom. GPU untouched.',
      protocol='All raw/post identities and sampling definitions fixed before production. Engineering exceptions recorded; no silent retry or result replacement.')
    C.write(target,r);C.csvwrite(C.OUT/'inputs/cells.csv',cells)
    C.event('REGISTRATION_FROZEN',requests=11,cells=44,protected_files=len(protected),registration_sha256=C.sha(target))
    print('FROZEN',len(cells),'cells;',len(protected),'protected files');return r
if __name__=='__main__':main()