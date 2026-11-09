"""Freeze all logical candidates before dispatch; no models or scorer calls."""
from . import common as C
import csv, hashlib, io, json
import numpy as np


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def immutable_npz(path, values):
    if path.exists():
        prior=C.arrays(path)
        if set(prior)!=set(values) or any(prior[k].dtype!=v.dtype or prior[k].shape!=v.shape or prior[k].tobytes()!=v.tobytes() for k,v in values.items()):
            raise ValueError('Prepared arrays changed: '+str(path))
    else:C.npz(path,values)


def immutable_csv(path, rows):
    if not path.exists():C.csvwrite(path,rows);return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    if path.read_bytes()!=buffer.getvalue().encode('utf-8-sig'):raise ValueError('Prepared table changed: '+str(path))


def boundary_ledger(output, selected, ids, known):
    flat=output.reshape(-1,3).astype(float)
    choice=np.repeat(selected,3)
    free=np.arange(len(flat))>=known*3
    result=[]
    for cid in np.unique(ids):
        members=np.flatnonzero(ids.reshape(-1)==cid)
        a=members[choice[members]]
        b=members[~choice[members]&free[members]]
        c=members[~free[members]]
        if len(a) and (len(b) or len(c)):
            other=np.concatenate((b,c))
            distances=np.linalg.norm(flat[a,None]-flat[None,other],axis=2)
            result.append(dict(cluster_id=int(cid),RAW_flat_corners=a.tolist(),D0_flat_corners=b.tolist(),C_flat_corners=c.tolist(),
                minimum_gap=float(distances.min()),maximum_gap=float(distances.max()),
                exact_pairs=int((distances==0).sum()),total_pairs=int(distances.size),
                all_members_same_position=bool(np.all(flat[members]==flat[members[0]]))))
    return result


def run():
    from . import coverage, runtime
    from meshflow_control import geo_c
    from r2.pilots.bench_four_routes_v1.hybrid.selector import quantities
    from r2.pilots.bench_group_weight_match_v1 import evaluation as E
    target=C.OUT/'jobs.json'
    if target.exists():
        r=C.read(target);C.verify(r['protected']);C.verify(r['source_hashes']);return r
    panel_path=C.OUT/'panel/manifest.json'
    panel=C.read(panel_path)
    old_path=C.OLD/'registration.json';old=C.read(old_path)
    C.verify(panel.get('protected',{}));C.verify(old['protected']);C.verify(old['source_hashes'])
    tasks={t['request_id']:t for t in old['tasks']}
    assert len(panel['requests'])==6
    protected={str(panel_path):C.sha(panel_path),str(C.OUT/'PROTOCOL.md'):C.sha(C.OUT/'PROTOCOL.md')}
    protected.update({str(C.resolve(x)):C.sha(x) for x in panel.get('protected',{})})
    jobs=[];references=[];selections=[]
    for index,item in enumerate(panel['requests']):
        rid=item['request_id'];task=tasks[rid];q=task['request'];paths=task['paths']
        if item['request']!=q:raise ValueError('Panel request was altered: '+rid)
        for p in paths.values():protected[str(C.resolve(p))]=C.sha(p)
        raw=C.arrays(paths['raw'])['raw'];pre=C.arrays(paths['pre'])['output'];k=int(q['K']);n=len(raw)
        if raw[:k].tobytes()!=pre[:k].tobytes():raise AssertionError('C mismatch')
        oldsel_path=C.OLD/'outputs'/rid/'selection.npz'
        old_receipt_path=oldsel_path.with_name('complete.json')
        old_receipt=C.read(old_receipt_path)
        if old_receipt['status']!='COMPLETE' or old_receipt['registration_sha256']!=C.sha(old_path):
            raise ValueError('Old RAW-GEO producer identity differs')
        C.verify(old_receipt['files']);protected[str(old_receipt_path)]=C.sha(old_receipt_path)
        oldsel=C.arrays(oldsel_path);protected[str(oldsel_path)]=C.sha(oldsel_path)
        ids=oldsel['cluster_ids'];base=C.arrays(paths['original_base'])['output']
        area,_=quantities(raw)
        masks={}
        for region in item['regions']:
            mask=np.zeros(n,bool);slots=np.asarray(region['source_face_ids'],int)
            if len(slots)==0 or np.any(slots<k) or np.any(slots>=n) or len(set(slots))!=len(slots):raise ValueError('Invalid frozen region')
            mask[slots]=True;masks[region['region_id']]=mask
        if set(masks)!={'P1','P2'}:raise ValueError('Expected two frozen regions')
        if np.any(masks['P1']&masks['P2']):raise ValueError('Assisted regions must be disjoint')
        auto=coverage.select(raw,pre,k,base,ids,L=C.L,R=geo_c.RADIUS)
        cdir=C.OUT/'selectors'/rid;cdir.mkdir(parents=True,exist_ok=True)
        # Complete machine-readable automatic selector, including rejected components.
        for key,value in auto.items():
            if key not in ('output',) and not isinstance(value,np.ndarray):
                C.write(cdir/('auto_'+key+'.json'),value)
        immutable_npz(cdir/'auto_arrays.npz',{key:v for key,v in auto.items() if isinstance(v,np.ndarray)})
        automatic=auto.get('selected_RAW',auto.get('selected_raw'))
        if automatic is None:raise KeyError('Automatic selector must expose selected_RAW')
        rawmask=np.arange(n)>=k
        variants=[('RAW_ALL',rawmask),('ASSIST10',masks['P1']),('ASSIST01',masks['P2']),
                  ('ASSIST11',masks['P1']|masks['P2']),('AUTO_COVER',automatic)]
        seen={}
        for label,mask in variants:
            mask=np.asarray(mask,bool);assert mask.shape==(n,) and not mask[:k].any()
            proposal=pre.copy();proposal[mask]=raw[mask];proposal[:k]=raw[:k]
            if label=='AUTO_COVER' and not np.array_equal(proposal,auto['output']):raise AssertionError('AUTO contract mismatch')
            jid=f'J{index:02d}_{label}';dest=C.OUT/'proposals'/jid
            immutable_npz(dest/'proposal.npz',dict(output=proposal,known_faces=np.asarray(k),raw_slot_ids=np.arange(n),
                  source_corner_ids=np.tile(np.arange(3),(n,1)),selected_RAW=mask))
            bh=boundary_ledger(proposal,mask,ids,k);C.write(dest/'mixed_boundary.json',bh)
            immutable_csv(dest/'selection_per_face.csv',[dict(raw_slot=f,known=f<k,selected_RAW=bool(mask[f]),
                raw_area=float(area[f]),max_shift_from_D0=float(np.linalg.norm(proposal[f].astype(float)-pre[f],axis=1).max())) for f in range(n)])
            detail=dict(job_id=jid,request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],label=label,
                selected_RAW_faces=int(mask.sum()),free_faces=n-k,selected_RAW_fraction=float(mask.sum()/(n-k)),
                selected_RAW_area=float(area[mask].sum()),raw_free_area=float(area[k:].sum()),
                selected_RAW_area_fraction=float(area[mask].sum()/area[k:].sum()),
                mixed_boundary_clusters=len(bh),RAW_D0_boundary_clusters=sum(bool(x['D0_flat_corners']) for x in bh),
                noncoincident_RAW_D0_clusters=sum(bool(x['D0_flat_corners']) and not x['all_members_same_position'] for x in bh),
                C_bytes_preserved=proposal[:k].tobytes()==raw[:k].tobytes(),proposal_array_sha256=array_hash(proposal))
            selections.append(detail);C.write(dest/'selection_summary.json',detail)
            job=dict(job_id=jid,label=label,request=q,paths=paths,
                proposal=dict(path=str(dest/'proposal.npz'),sha256=C.sha(dest/'proposal.npz'),output_key='output'),
                selection_summary=str(dest/'selection_summary.json'),mixed_boundary=str(dest/'mixed_boundary.json'),reuse_of=None)
            digest=array_hash(proposal)
            if digest in seen:
                job['reuse_of']=seen[digest]
            elif proposal.tobytes()==pre.tobytes():
                job['reuse_original_d0']=True
            elif proposal.tobytes()==C.arrays(C.OLD/'outputs'/rid/'pre_geo.npz')['output'].tobytes():
                receipt=C.OLD/'outputs'/rid/'complete.json'
                job['reuse_of_external']=dict(registration_path=str(old_path),registration_sha256=C.sha(old_path),request_id=rid,
                    complete_path=str(receipt),complete_sha256=C.sha(receipt),cpu_dir=str(C.OLD/'cpu/HYBRID'/rid))
            seen[digest]=jid;jobs.append(job)
        references.append(task)
    if len(jobs)!=30:raise AssertionError('Logical budget must be30')
    sources={str(p):C.sha(p) for p in C.CODE.glob('*.py')}
    sources.update(old['source_hashes'])
    sources.update(runtime.definition_hashes())
    from r2.tools.cref_layer_audit_v1 import geometry as finite_geometry
    sources[str(C.resolve(finite_geometry.__file__))]=C.sha(finite_geometry.__file__)
    sources[str(C.resolve(geo_c.__file__))]=C.sha(geo_c.__file__)
    physical=[j for j in jobs if not j.get('reuse_of') and not j.get('reuse_of_external') and not j.get('reuse_original_d0')]
    registration=dict(status='FROZEN',schema='bench_regional_mix_dual_v1',utc=C.utc(),jobs=jobs,references=references,
        panel_sha256=C.sha(panel_path),protocol_sha256=C.sha(C.OUT/'PROTOCOL.md'),protected=protected,source_hashes=sources,
        metric_fields=C.read(E.MANIFEST)['metric_fields'],L=C.L,R=geo_c.RADIUS,
        budget=dict(logical_jobs=30,physical_jobs=len(physical),new_score_rows=2*len(physical),
                    maximum_full_cascades=30,model_calls=0,D0_calls=0,training_updates=0,new_noise=0),
        scheduling=dict(maximum_workers=2,threads_per_worker=1,first_measured_authorized_job=True),
        no_GT_success=True,no_result_based_selection=True)
    immutable_csv(C.OUT/'selection_per_output.csv',selections)
    C.write(target,registration)
    return registration


if __name__=='__main__':
    r=run();print(json.dumps(dict(status=r['status'],budget=r['budget']),ensure_ascii=False))
