"""All72 RAW first, one official D0, fixed GEO/source export and CPU cascade.

PATCH preserves its candidate shared-free decoder after D0. The official
unprojected proposal remains immutable; affected geometry and index survival
are separately audited. No extra denoiser pass or coordinate-neighbour fallback.
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
from . import common as C, evaluation as E

VERSION='bench_three_mechanisms_single_D0_shared_patch_v1'


def shared_index_audit(post,pairs):
    import numpy as np
    face_rows={int(f):i for i,f in enumerate(post['source_face_ids'])}
    records=[]
    for pair in np.asarray(pairs,np.int64):
        vids=[]
        for corner in pair:
            face,local=divmod(int(corner),3)
            if face not in face_rows:vids.append(None);continue
            row=face_rows[face];cols=np.flatnonzero(post['source_corner_ids'][row]==local)
            if len(cols)!=1:raise ValueError('Source corner identity not a permutation')
            vids.append(int(post['faces'][row,int(cols[0])]))
        status='FACE_DELETED' if None in vids else ('SHARED' if vids[0]==vids[1] else 'SPLIT')
        records.append(dict(raw_corner_pair=pair.tolist(),post_vertex_ids=vids,status=status))
    split=sum(r['status']=='SPLIT' for r in records)
    if split:raise ValueError('GEO split a retained explicit PATCH shared endpoint')
    return dict(status='PASS',pairs=len(records),retained_shared=sum(r['status']=='SHARED' for r in records),
        pairs_with_deleted_face=sum(r['status']=='FACE_DELETED' for r in records),split_pairs=split,records=records,
        deleted_faces_are_not_shared_connectivity_success=True)


def projection_audit(before,after,k,pairs):
    import numpy as np
    a,b=np.asarray(before,np.float64),np.asarray(after,np.float64)
    def facts(x):
        cross=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]);length=np.linalg.norm(cross,axis=1)
        return length/2,cross/np.maximum(length[:,None],1e-30)
    aa,na=facts(a);ab,nb=facts(b);delta=np.linalg.norm(b-a,axis=-1)
    affected=np.unique(np.asarray(pairs,np.int64).reshape(-1)//3)
    deg=1e-12*C.L*C.L
    angles=np.degrees(np.arccos(np.clip((na*nb).sum(-1),-1,1)))
    return dict(status='RECORDED',affected_faces=affected.tolist(),C_changed=bool(np.any(delta[:k])),
        max_free_corner_displacement_L=float(delta[k:].max(initial=0)/C.L),
        rms_free_corner_displacement_L=float(np.sqrt(np.mean(delta[k:]**2))/C.L),
        free_area_before=float(aa[k:].sum()),free_area_after=float(ab[k:].sum()),
        affected_area_before=float(aa[affected].sum()),affected_area_after=float(ab[affected].sum()),
        newly_degenerate_faces=np.flatnonzero((aa>deg)&(ab<=deg)).tolist(),
        affected_normal_gt60_faces=affected[angles[affected]>60].tolist(),
        per_affected_face=[dict(raw_slot=int(i),area_before=float(aa[i]),area_after=float(ab[i]),
            normal_change_degrees=float(angles[i]),max_corner_displacement_L=float(delta[i].max()/C.L)) for i in affected],
        scope='Shared projection only, after common C restore; not a collision detector. Final quality uses unchanged177 scorer.')


def patch_d0(estimated_peak_mib=3500):
    import numpy as np
    from . import patch
    from r2.pilots.bench_control_transfer_v1 import postprocess as OLD
    from r2.pilots.seam_slot_c200_v1.postprocess import geometry_from_proposal
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    from meshflow_control.geo_io import write_indexed_ply
    label='PATCH200';barrier=E.all_raw_barrier();out=C.OUT/'evaluation'/label
    final=out/'post_complete_full.json'
    if final.exists():
        done=C.read(final);C.verify(done['files'])
        for path in done['files']:C.verify(C.read(path)['files'])
        return done
    if list(out.glob('postprocess_attempt_*')):raise RuntimeError('Prior PATCH D0 attempt needs explicit recovery; no retry')
    code=E._code_hashes({str(__file__):C.sha(__file__),str(patch.__file__):C.sha(patch.__file__)},'D0')
    C.write(C.OUT/'evaluation/inference_registrations/PATCH200__D0.json',dict(status='FROZEN',label=label,
        outputs=12,new_D0_budget=12,new_GEO_budget=12,code=code,D0_checkpoint=str(OLD.D0),D0_checkpoint_sha256=OLD.D0_SHA,
        RAW_barrier={str(p):C.sha(p) for p in barrier},
        contract='singleD0 -> restoreC -> average sharedfree -> originalGEO_C/sourcewinding; explicit index audit'))
    attempt=out/('postprocess_attempt_'+str(time.time_ns()));attempt.mkdir(parents=True)
    resource=C.resources();C.write(attempt/'resources_before.json',resource);E.dispatch_guard(resource,estimated_peak_mib)
    observer=GPUObserver(C.EXPERIMENT+'_PATCH_D0',interval=1.);observer.__enter__()
    model=None;calls=count=0;started=time.perf_counter()
    try:
        import torch
        from meshflow_control.denoise_model import predict_proposal
        model,identity=OLD.load_bench_denoiser(OLD.D0);C.write(attempt/'model_identity.json',identity)
        C.write(attempt/'resources_loaded.json',C.resources())
        for q in E.requests():
            rawdir=out/q['request_id']/'raw';dest=out/q['request_id']/'post'
            if dest.exists():raise RuntimeError('Unresolved PATCH post output; no overwrite')
            raw=C.arrays(rawdir/'raw.npz');source=C.arrays(rawdir/'source_condition.npz')
            layout=C.arrays(out/'layouts'/q['condition_id']/'layout.npz');dest.mkdir()
            C.write(dest/'attempt.json',dict(label=label,request=q,raw_sha256=C.sha(rawdir/'raw.npz'),D0_budget=1,GEO_budget=1))
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            C.event(dest/'calls.jsonl',dict(event='D0_FORWARD_STARTED'))
            proposal=predict_proposal(model,raw['raw']);torch.cuda.synchronize();calls+=1
            C.event(dest/'calls.jsonl',dict(event='D0_FORWARD_RETURNED',seconds=time.perf_counter()-tick))
            C.npz(dest/'proposal.npz',dict(official_unconstrained_proposal=proposal))
            C.write(dest/'proposal_receipt.json',dict(status='COMPLETE',D0_forwards=1,raw_sha256=C.sha(rawdir/'raw.npz'),
                proposal_sha256=C.sha(dest/'proposal.npz'),official_checkpoint_sha256=OLD.D0_SHA,seconds=time.perf_counter()-tick,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            restored=proposal.copy();anchor=raw['raw_anchor_ids'];fixed=anchor>=0
            restored[fixed]=raw['C_vertices'][anchor[fixed]]
            projected=patch.project_coordinates(restored,layout)
            projection=projection_audit(restored,projected,int(q['K']),layout['shared_corner_pairs'])
            if projection['C_changed']:raise AssertionError('PATCH projection moved C')
            C.npz(dest/'shared_projected_proposal.npz',dict(output=projected))
            C.write(dest/'shared_projection.json',projection)
            C.write(dest/'geometry_attempt.json',dict(calls=1,wall_time=time.time()))
            pre,canonical,post,geometry,oriented,anchors=geometry_from_proposal(raw,projected,source)
            shared=shared_index_audit(post,layout['shared_corner_pairs'])
            post.update(raw_shared_corner_pairs=layout['shared_corner_pairs'],raw_patch_ids=layout['patch_ids'],
                raw_patch_state_ids=layout['state_ids'],patch_count=layout['patch_count'])
            C.npz(dest/'pre_geo.npz',dict(output=pre,C=raw['raw'][:int(q['K'])],known_faces=raw['known_faces'],
                raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask'],
                raw_shared_corner_pairs=layout['shared_corner_pairs']))
            C.npz(dest/'post_canonical.npz',canonical);C.npz(dest/'post.npz',post)
            write_indexed_ply(dest/'post.ply',post['vertices'],post['faces'])
            C.write(dest/'geometry.json',dict(stats=geometry['stats'],face_records=geometry['face_records']))
            C.write(dest/'anchor_identity.json',anchors);C.write(dest/'shared_index_audit.json',shared)
            C.write(dest/'source_restoration.json',oriented['restoration_summary'])
            ori=oriented['orientation_result'];C.write(dest/'orientation.json',dict(summary=ori['summary'],components=ori['component_rows'],edges=ori['edge_rows']))
            C.write(dest/'receipt.json',dict(status='COMPLETE',schema=VERSION,label=label,arm=label,request_id=q['request_id'],
                condition_id=q['condition_id'],post_sha256=C.sha(dest/'post.npz'),raw_sha256=C.sha(rawdir/'raw.npz'),
                official_D0_sha256=OLD.D0_SHA,D0_forwards=1,GEO_calls=1,source_labels_read=False,anchors=anchors,shared=shared,
                files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()}))
            count+=1
            if count in (1,2,12):C.write(attempt/f'resources_output{count}.json',C.resources())
            print(json.dumps(dict(stage='POST',label=label,completed=count,request_id=q['request_id'])),flush=True)
        C.verify(code)
        receipts=[out/q['request_id']/'post/receipt.json' for q in E.requests()]
        result=dict(status='COMPLETE',label=label,subset='full',outputs=12,new_D0_forwards=calls,new_GEO_calls=count,
            seconds=time.perf_counter()-started,barrier_labels=list(E.NEW_LABELS),files={str(p):C.sha(p) for p in receipts})
        C.write(final,result);return result
    except BaseException as exc:
        C.write(attempt/'failure.json',dict(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc(),
            returned_D0_calls=calls,completed_outputs=count,no_automatic_retry=True));raise
    finally:
        observer.__exit__(None,None,None)
        C.write(attempt/'resources.json',dict(summary=observer.summary(),samples=observer.samples))
        if model is not None:
            import torch
            del model;torch.cuda.empty_cache()


def d0(label,estimated_peak_mib=3500):
    if label not in E.NEW_LABELS:raise ValueError('Baseline D0 rerun forbidden')
    E.all_raw_barrier()
    return patch_d0(estimated_peak_mib) if label=='PATCH200' else E.engine().d0(label,estimated_peak_mib)


def cpu(mode,labels=None,workers=2):
    if mode not in ('raw','post') or workers not in (1,2):raise ValueError('Fixed CPU scope')
    E.all_raw_barrier()
    return E.engine().run_cpu(list(E.NEW_LABELS if labels is None else labels),mode,workers)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('verify','d0','raw','post'))
    p.add_argument('--label',choices=E.NEW_LABELS);p.add_argument('--labels',nargs='+',choices=E.NEW_LABELS)
    p.add_argument('--workers',type=int,choices=(1,2),default=2);p.add_argument('--estimated-peak-mib',type=float,default=3500)
    a=p.parse_args()
    if a.action=='verify':r=dict(status='PASS',RAW=len(E.all_raw_barrier()))
    elif a.action=='d0':
        if not a.label:p.error('--label required')
        r=d0(a.label,a.estimated_peak_mib)
    else:r=cpu(a.action,a.labels,a.workers)
    print(json.dumps({k:v for k,v in r.items() if k!='files'},ensure_ascii=False))


if __name__=='__main__':main()
