"""Ninety frozen extractions: 89 enclosing interventions and one fixed replay."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,time,traceback,shutil
from pathlib import Path
import numpy as np
from common import ROOT,dump,sha,save_mesh,load_mesh,mesh_audit,metrics,sample
from run_clay import strict_links,sorted_simplex,array_hash
from exact_audit import exact_mesh
from safe_extract_domain import extract_domain,corner_domain


def read(path):return json.loads(Path(path).read_text())
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    protocol=read(ROOT/'FROZEN_PROTOCOL.json');expected_sha=read(ROOT/'FROZEN_PROTOCOL_SHA256.json')['sha256']
    assert sha(ROOT/'FROZEN_PROTOCOL.json')==expected_sha
    rows=[];checks=[]
    for spec in protocol['slots']:
        rd=ROOT/'runs'/spec['id'];rd.mkdir();start=time.perf_counter()
        row=dict(id=spec['id'],group=spec['group'],domain=spec['domain'],arm=spec.get('arm'),sample_id=spec.get('sample_id'),
            target=spec['target'],source_parameters=spec['parameters'],source_parameters_sha256=spec['parameters_sha256'],
            baseline_fixed=spec['baseline_fixed'],requested_extractions=1,extraction_completed=False,geometry_audited=False,
            valid=False,status='STARTED',parameter_intervention=False,reference_used_to_set_domain=False)
        stage='load_parameters'
        try:
            source=Path(spec['parameters']);assert sha(source)==spec['parameters_sha256']
            shutil.copy2(source,rd/'parameters.npz');p=arrays(rd/'parameters.npz')
            before={k:array_hash(a) for k,a in p.items()}
            corners,domain_audit=corner_domain(p['v'],spec['domain']);row.update(domain_audit)
            dump(rd/'DOMAIN.json',domain_audit)
            stage='extract';et=time.perf_counter();v,f,ids,trace,audit=extract_domain(p,domain=spec['domain']);row['extraction_seconds']=time.perf_counter()-et
            row['extraction_completed']=True;row['vertices']=len(v);row['faces']=len(f)
            row.update(audit);dump(rd/'extractor_audit.json',audit);np.savez_compressed(rd/'extractor_trace.npz',**trace)
            save_mesh(rd/'mesh.obj',v,f);np.savez_compressed(rd/'mesh.npz',vertices=v,faces=f,input_vertex_ids=ids)
            row['mesh_sha256']=sha(rd/'mesh.obj');row['surface_input_faces_sha256']=array_hash(sorted_simplex(ids[f]))
            integrity=dict(parameters_bytes_unchanged=sha(source)==sha(rd/'parameters.npz')==spec['parameters_sha256'],
                all_input_arrays_unchanged=before=={k:array_hash(a) for k,a in p.items()},
                all_original_Delaunay_sites_unchanged=np.array_equal(trace['delaunay_points'][:len(p['v'])],p['v']),
                anchor_coordinates_expected=np.array_equal(trace['delaunay_points'][-8:],corners),
                output_ids_map_vertices=np.array_equal(v,trace['delaunay_points'][ids]),
                output_ids_map_faces=np.array_equal(trace['surface_faces_input_ids'],ids[f]),
                actual_capacities_finite_nonnegative=bool(np.isfinite(trace['graph_capacities']).all() and np.all(trace['graph_capacities']>=0)))
            row['integrity']=integrity;checks.append(dict(id=spec['id'],checks=integrity,passed=all(integrity.values())))
            old=arrays(spec['fixed_mesh_npz']);sf=sorted_simplex(ids[f]);oldsf=sorted_simplex(old['input_vertex_ids'][old['faces']])
            row['same_input_id_surface_as_fixed']=bool(np.array_equal(sf,oldsf))
            row['same_vertices_as_fixed']=bool(np.array_equal(v,old['vertices']))
            row['same_faces_as_fixed']=bool(np.array_equal(f,old['faces']))
            row['same_mesh_bytes_as_fixed']=sha(rd/'mesh.obj')==spec['fixed_mesh_sha256']
            if spec['domain']=='fixed':
                row['fixed_identity_passed']=all(row[k] for k in ['same_vertices_as_fixed','same_faces_as_fixed','same_mesh_bytes_as_fixed'])
                if spec.get('fixed_trace'):
                    oldtrace=arrays(spec['fixed_trace'])
                    row['fixed_trace_unchanged']=all(k in trace and np.array_equal(a,trace[k],equal_nan=True) for k,a in oldtrace.items())
                    row['fixed_identity_passed']=row['fixed_identity_passed'] and row['fixed_trace_unchanged']
                checks.append(dict(id=spec['id']+'_identity',passed=row['fixed_identity_passed']))
            stage='geometry_audit'
            if not len(v) or not len(f):raise ValueError('Empty extracted surface: geometry audit undefined; retained as failure')
            row.update(mesh_audit(v,f));link=strict_links(f);row.update(link);dump(rd/'vertex_link_witnesses.json',link)
            exact=exact_mesh(v,f);row.update(exact);dump(rd/'exact_audit.json',dict(source_sha256=row['mesh_sha256'],**exact))
            row['geometry_audited']=True;row['betti_match']=[row['beta'+str(i)] for i in range(3)]==[1,0,1]
            row['valid']=bool(row['closed_oriented_2manifold'] and row['strict_all_vertex_links_simple_cycles'] and row['betti_match']
                and row['duplicate_faces']==0 and row['exact_intersection_pairs']==0 and row['exact_degenerate_faces']==0)
            stage='same_target_metrics';ref=ROOT/'references'/spec['reference_id'];gv,gf=load_mesh(ref/'reference.obj')
            with np.load(ref/'eval_samples.npz') as z:gt=z['points'];gn=z['normals']
            mt=time.perf_counter();row['same_target_geometry']=metrics(v,f,gv,gf,gt,gn,spec['metric_seed'],n=8192);row['metric_seconds']=time.perf_counter()-mt
            row['same_target_rms_pct_bboxdiag']=row['same_target_geometry']['rms_pct_bboxdiag'];row['status']='EXTRACTED'
        except Exception as exc:
            row.update(status='EXTRACTION_OR_AUDIT_OR_EVALUATION_FAILED',failure_stage=stage,error=repr(exc),traceback=traceback.format_exc())
            if not row.get('geometry_audited'):row['valid']=False
            if hasattr(exc,'audit'):dump(rd/'extractor_audit.json',exc.audit);row['failure_extractor_audit']=exc.audit
            if hasattr(exc,'trace_arrays'):np.savez_compressed(rd/'extractor_trace.npz',**exc.trace_arrays)
        row['total_seconds']=time.perf_counter()-start;rows.append(row)
        dump(rd/'result.json',row);dump(ROOT/'LEDGER_PARTIAL.json',rows)
        print('DOMAIN_RESULT',row['id'],row['status'],'valid',row['valid'],'faces',row.get('faces'),'identity',row.get('fixed_identity_passed'),flush=True)
    dump(ROOT/'LEDGER.json',rows)
    summary={}
    for group in ['generated','oracle24','single_target','fixed_identity']:
        rr=[r for r in rows if r['group']==group]
        summary[group]=dict(requested=len(rr),extraction_completed=sum(r['extraction_completed'] for r in rr),
            pipeline_completed=sum(r['status']=='EXTRACTED' for r in rr),valid=sum(bool(r['valid']) for r in rr),
            baseline_fixed_valid=sum(bool(r['baseline_fixed']['valid']) for r in rr),
            changed_mesh_bytes=sum(r.get('same_mesh_bytes_as_fixed') is False for r in rr),
            changed_input_id_surface=sum(r.get('same_input_id_surface_as_fixed') is False for r in rr),
            actual_faces=[r.get('faces') for r in rr],
            outside_actual_fixed_cube_samples=sum(r.get('input_sites_outside_actual_fixed_cube',0)>0 for r in rr),
            outside_actual_fixed_cube_points=sum(r.get('input_sites_outside_actual_fixed_cube',0) for r in rr),
            legacy_half_cube_exceedance_samples=sum(r.get('legacy_half_cube_exceedance_count',0)>0 for r in rr),
            metric_evaluable=sum('same_target_rms_pct_bboxdiag' in r for r in rr))
    by_arm={}
    for arm in ['direct','transport_residual']:
        rr=[r for r in rows if r['group']=='generated' and r['arm']==arm]
        by_arm[arm]=dict(requested=32,baseline_fixed_valid=sum(r['baseline_fixed']['valid'] for r in rr),
            enclosing_valid=sum(r['valid'] for r in rr),extraction_completed=sum(r['extraction_completed'] for r in rr),
            actual_faces=[r.get('faces') for r in rr],actual_fixed_cube_outside_samples=sum(r.get('input_sites_outside_actual_fixed_cube',0)>0 for r in rr),
            actual_fixed_cube_outside_points=sum(r.get('input_sites_outside_actual_fixed_cube',0) for r in rr))
    summary_doc=dict(requested_new_extractions=90,actual_rows=len(rows),groups=summary,generated_by_arm=by_arm,
        original_fixed_corners=np.array(protocol['fixed_corners']).tolist(),
        note='Domain intervention changes Delaunay structure and external labels; no generation/quadric/normal repair or retraining. Fixed 64 reused, enclosing64 newly extracted.',
        fixed_domain_correction='Official fixed anchors are (+/-1,+/-1,+/-1), not +/-0.5. Prior half-cube exceedance did not establish actual-domain violation.')
    dump(ROOT/'SUMMARY.json',summary_doc)
    hard=dict(passed=len(rows)==90 and len({r['id'] for r in rows})==90 and all(c['passed'] for c in checks),
        requested=90,actual=len(rows),checks=checks,geometry_failures_do_not_fail_integrity=True,
        missing_trace_slots=[r['id'] for r in rows if not (ROOT/'runs'/r['id']/'extractor_trace.npz').exists()])
    dump(ROOT/'audit/INTEGRITY.json',hard)
    print(json.dumps(summary_doc,indent=2),flush=True)
    if not hard['passed']:raise SystemExit(1)
if __name__=='__main__':main()