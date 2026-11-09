"""Freeze 24 refit targets and verify their legal full-quadric chart before FM."""
import sys,json,shutil,time,traceback
from pathlib import Path
import numpy as np
from common import ROOT,dump,sha,load_mesh,save_mesh,mesh_audit
from run_clay import strict_links,sorted_simplex
from safe_extract import extract
from exact_audit import exact_mesh
from fm_state import encode_state,decode_state
from summarize_clay import method_failures

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def main():
    parent=Path(sys.argv[1]).resolve();gate=Path(sys.argv[2]).resolve()
    gate_rows=read(gate/'LEDGER.json')
    assert len(gate_rows)==45 and all(not method_failures(r) for r in gate_rows),'Cumulative geometry gate did not pass'
    assert read(gate/'audit/ACCUM_VERIFY.json')['passed'],'Cumulative algebra gate did not pass'
    gc=read(gate/'FROZEN_PROTOCOL.json')['config']['gates']
    ur=next(r for r in gate_rows if r['source_object']=='thin_u' and r['direction']=='forward' and r['frame']==7)
    assert ur['rms_pct_bboxdiag']<=gc['thin_u_incremental_endpoint_rms_max'],'Thin-U retention gate did not pass'
    records=[r for r in read(parent/'LEDGER.json') if r['state']=='refit']
    assert len(records)==24 and all(not method_failures(r) for r in records)
    records.sort(key=lambda r:(r['source_object'],r['frame']))
    states=[];reports=[];start=time.perf_counter()
    for row in records:
        name=row['object'];source=parent/'runs'/f'{name}__refit';target=ROOT/'targets'/name;target.mkdir(parents=True)
        try:
            p=arrays(source/'parameters.npz');state,diag=encode_state(p)
            decoded=decode_state(state,normalize_normals=False)
            np.savez_compressed(target/'parameters.npz',**decoded)
            v,f,ids,trace,audit=extract(decoded)
            save_mesh(target/'mesh.obj',v,f)
            np.savez_compressed(target/'mesh.npz',vertices=v,faces=f,input_vertex_ids=ids)
            ex=exact_mesh(v,f);geo=mesh_audit(v,f);links=strict_links(f)
            valid=(geo['closed_oriented_2manifold'] and [geo['beta'+str(i)] for i in range(3)]==[1,0,1]
                and links['strict_all_vertex_links_simple_cycles'] and ex['exact_intersection_pairs']==0 and ex['exact_degenerate_faces']==0 and geo['duplicate_faces']==0)
            orig=arrays(source/'mesh.npz')
            same_surface=np.array_equal(sorted_simplex(ids[f]),sorted_simplex(orig['input_vertex_ids'][orig['faces']]))
            Qerr=float(np.linalg.norm(decoded['Q']-p['Q'])/np.linalg.norm(p['Q']))
            report=dict(object=name,source_object=row['source_object'],frame=row['frame'],source_parameters=str(source/'parameters.npz'),
                source_parameters_sha256=sha(source/'parameters.npz'),source_mesh_sha256=sha(source/'mesh.obj'),
                output_mesh_sha256=sha(target/'mesh.obj'),same_surface_input_faces=same_surface,valid=bool(valid),
                Q_world_relative=Qerr,encoding=diag,geometry=geo,links=links,exact=ex,extractor=audit,
                normal_bytes_unchanged=np.array_equal(decoded['n'],p['n']))
            dump(target/'ROUNDTRIP.json',report);reports.append(report);states.append(state)
        except Exception as exc:
            report=dict(object=name,valid=False,error=repr(exc),traceback=traceback.format_exc())
            reports.append(report);dump(target/'ROUNDTRIP.json',report)
        print('TARGET_ROUNDTRIP',name,reports[-1].get('valid'),reports[-1].get('same_surface_input_faces'),flush=True)
    if len(states)!=24:
        dump(ROOT/'DATA_CHECK.json',dict(passed=False,targets=24,records=reports,error='One or more target roundtrips failed'))
        raise RuntimeError('Target roundtrip errors recorded; no training')
    states=np.stack(states);mean=states.mean(axis=(0,1));scale=np.empty(22)
    for a,b in [(0,3),(3,6),(6,22)]:scale[a:b]=np.sqrt(np.mean((states[...,a:b]-mean[a:b])**2))
    assert np.isfinite(states).all() and np.all(scale>0)
    passed=all(r['valid'] and r['same_surface_input_faces'] and r['normal_bytes_unchanged'] and r['Q_world_relative']<1e-10 for r in reports)
    dump(ROOT/'DATA_CHECK.json',dict(passed=passed,targets=24,parent_objects=3,source_state='reference_assisted_refit',
        normal_normalization=False,roundtrip_extractions=24,records=reports,seconds=time.perf_counter()-start,
        note='All states are known development objects. Roundtrip must retain valid canonical input-ID surface; this is an encoding gate, not a performance result.'))
    if not passed:raise RuntimeError('FM encoding geometry gate failed; preserve result and do not train')
    np.savez_compressed(ROOT/'data.npz',states=states,mean=mean,scale=scale,object_names=np.array([r['object'] for r in reports]))
    print('DATA_READY',states.shape,flush=True)
if __name__=='__main__':main()
