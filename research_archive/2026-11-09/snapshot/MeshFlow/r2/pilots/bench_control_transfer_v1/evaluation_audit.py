"""CPU input/metric transport audit and explicit official-split result strata."""
from __future__ import annotations
import argparse, collections, csv, json
from pathlib import Path
import numpy as np
from . import evaluate as V

C = V.C


def input_audit(root):
    root = C.resolve(root); reg = V.prepare(root)
    frozen = C.read(root/'evaluation_inputs/manifest.json')
    by_request = {row['request_id']:row for row in frozen['requests']}
    old = V.E.helpers()[0]
    checks = []; strata = collections.Counter()
    for q in reg['requests']:
        generated = by_request[q['request_id']]
        package = C.arrays(q['inference_npz']); metadata = C.arrays(q['metadata_npz'])
        c_only = C.arrays(generated['inference_npz']); source = C.arrays(generated['source_npz'])
        original = package['C_vertices'][package['C_faces_original']]
        canonical = package['C_vertices'][package['C_faces_model']]
        ids = metadata['C_source_vertex_ids'][package['C_faces_original']]
        assert np.array_equal(source['C_original'], original)
        assert np.array_equal(source['C'], canonical)
        assert np.array_equal(source['C_original_source_vertex_ids'], ids)
        for key in ('C_vertices','C_faces_original','C_faces_model'):
            assert np.array_equal(package[key], c_only[key]), (q['request_id'], key)
        assert (int(c_only['N']),len(c_only['C_faces_model'])) == (q['N'],q['K'])
        assert generated['noise_seed'] == q['noise_seed']
        noise = C.arrays(generated['noise_npz'])['z']
        assert noise.shape == (q['N']-q['K'],3,3) and noise.dtype == np.float32
        reference = old.load_queries(q['query_descriptor']); a = reference['arrays']
        reconstructed = (canonical[a['C_point_faces']].astype(np.float64)*a['C_barycentric'][:,:,None]).sum(axis=1)
        error = float(np.max(np.abs(reconstructed-a['C_points'])))
        assert error < 1e-10
        assert reference['identity']['C_sha256'] == old.array_sha(canonical)
        assert len(a['C_points']) == 2048 and np.all(a['C_weights']>0)
        assert abs(a['C_weights'].sum()-q['query_descriptor']['weight_denominator']) < 1e-10
        strata[q['split'],q['official_split'],q['official_pretraining_exposure']] += 1
        checks.append(dict(request_id=q['request_id'],condition_id=q['condition_id'],parent_id=q['parent_id'],
            split=q['split'],official_split=q['official_split'],official_pretraining_exposure=q['official_pretraining_exposure'],
            K=q['K'],N=q['N'],C_source_ids_match=True,C_canonical_and_original_match=True,
            query_point_max_barycentric_error=error,query_points=2048,
            query_sha256=q['query_descriptor']['sha256'],noise_sha256=generated['noise_sha256']))
    # Unit-level graph identities, not a new sampled/scored scientific mesh.
    triangle=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    mesh=np.stack([triangle,triangle,triangle]); mask=np.array([False,True,False]); ids=np.arange(3)
    edges=[dict(valid_edge=True,incident_face_ids=[0,1],length_model=1.),
           dict(valid_edge=True,incident_face_ids=[0,2],length_model=1.)]
    first,_=V.E.seam_to_bulk(mesh,1,edges,mask,ids,mask,True)
    assert first['free_only_reached_bulk_area_fraction']==0.
    second,_=V.E.seam_to_bulk(mesh,1,edges+[dict(valid_edge=True,incident_face_ids=[1,2],length_model=1.)],mask,ids,mask,True)
    assert second['free_only_reached_bulk_area_fraction']==1.
    plain,_=V.E.seam_to_bulk(mesh,1,edges,mask,ids,mask,False)
    assert plain['status']=='NA_NOT_APPLICABLE_FULL_ARM'
    assert all(plain[key] is None for key in V.E.INTERNAL_FIELDS)
    assert V.API['C'].ARMS==('SEAM',) and V.E.C.ARMS!=('SEAM',)
    assert V.API['score_one'].__code__ is V.E.score_one.__code__
    history=C.ROOT/'experiments/r2_seam_quality_wave1_wave2_v1/evaluation'
    expected=C.read(history/'wave1_summary_completion.json')['numeric_fields']
    assert len(expected)==124 and len(set(expected))==124
    extracted=[]
    for stage in ('RAW','POST'):
        path=history/'scores/PLAIN_S2000/2d08a64e_n0'/stage/'score.json'
        score=C.read(path);row,_,_=V.metric_row(score,C.arrays(score['identity']['mesh_path']),C.read(path.parent/'pairs.json'))
        extracted.append(row)
    actual=sorted(k for k in set().union(*(r.keys() for r in extracted))
        if k not in (*V.E.META,'label','stage') and any(V.number(r.get(k)) for r in extracted))
    assert actual==expected, (set(actual)-set(expected),set(expected)-set(actual))
    split_rows=[dict(split=k[0],official_split=k[1],official_pretraining_exposure=k[2],requests=n,
        conditions=len({r['condition_id'] for r in checks if (r['split'],r['official_split'],r['official_pretraining_exposure'])==k}))
        for k,n in sorted(strata.items())]
    out=root/'evaluation/input_audit';out.mkdir(exist_ok=True)
    C.csvwrite(out/'per_request.csv',checks);C.csvwrite(out/'official_split_strata.csv',split_rows)
    result=dict(status='PASS',requests=len(checks),unique_conditions=len({x['condition_id'] for x in checks}),
        all_C_queries_and_source_ids_match=True,original_numeric_fields=actual,original_numeric_count=len(actual),
        scorer_code_object_unchanged=True,private_namespace_does_not_mutate_original=True,
        PLAIN_structured_seam_metrics='NA_NOT_APPLICABLE_FULL_ARM',ordinary_CF_graph_retained=True,
        graph_unit_tests=dict(no_path_through_C=True,legal_FF_path_required=True,synthetic_graph_cases=3,new_geometry_scans=0),
        scope='C preservation and graph reachability do not certify intended use, full surface completeness, watertightness, or universal control.',
        official_split_strata=split_rows,model_calls=0,new_queries=0,new_quality_scans=0,
        files={str(out/name):C.sha(out/name) for name in ('per_request.csv','official_split_strata.csv')})
    C.write(out/'receipt.json',result);return result


def stratify(root):
    root=C.resolve(root);reg=V.prepare(root)
    requests={q['request_id']:q for q in reg['requests']}
    complete=C.read(root/'evaluation/summary_completion.json');fields=complete['numeric_fields']
    with (root/'evaluation/per_output.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    groups=collections.defaultdict(list)
    for row in rows:
        q=requests[row['request_id']]
        groups[row['label'],row['stage'],q['split'],q['official_split'],q['official_pretraining_exposure']].append(row)
    result_rows=[]
    for key,rr in sorted(groups.items()):
        for metric in fields:
            values=[]
            for row in rr:
                value=row.get(metric,'')
                if value in ('True','False'):values.append(float(value=='True'))
                elif value not in ('','None'):values.append(float(value))
            result_rows.append(dict(label=key[0],stage=key[1],split=key[2],official_split=key[3],
                official_pretraining_exposure=key[4],metric=metric,n_registered=len(rr),n_valid=len(values),
                mean=float(np.mean(values)) if values else None,status='PASS' if len(values)==len(rr) else 'NA_OR_PARTIAL'))
    dest=root/'evaluation/official_split_summary.csv';C.csvwrite(dest,result_rows)
    result=dict(status='COMPLETE',rows=len(result_rows),metrics=len(fields),new_scans=0,model_calls=0,
        policy='Condition-training held-out DEV is split by official dataset train/test metadata. Test membership does not independently prove absence from every upstream pretraining source.',
        files={str(dest):C.sha(dest)},source_per_output_sha256=C.sha(root/'evaluation/per_output.csv'))
    C.write(root/'evaluation/official_split_summary_receipt.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(V.ROOT));p.add_argument('--stage',choices=('inputs','strata'),required=True)
    a=p.parse_args();r=input_audit(a.root) if a.stage=='inputs' else stratify(a.root)
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','original_numeric_fields')}))


if __name__=='__main__':main()
