"""Read-only CPU comparison against historical EXP67 and protected-file audit.

No scorer, model, CUDA, generation, geometry, or replay is imported or invoked.
--compare refuses to finalize until the complete new RAW index is available.
--protect reads each protected file once, including the two model checkpoints;
an existing receipt is returned without another multi-GB hash pass.
"""
import argparse, collections, json, time
import numpy as np
import common as C

DEST = C.OUT/'audit'

def diff(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:
        return dict(shape_equal=False,dtype_equal=a.dtype==b.dtype,bitwise_equal=False,
                    numeric_equal=False,maxabs=None,rms=None,new_shape=list(a.shape),old_shape=list(b.shape))
    d=a.astype(np.float64)-b.astype(np.float64)
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    return dict(shape_equal=True,dtype_equal=a.dtype==b.dtype,bitwise_equal=C.same(a,b),
        numeric_equal=bool(np.array_equal(a,b)),finite=finite,
        maxabs=float(np.max(np.abs(d))) if finite and d.size else (0. if not d.size else None),
        rms=float(np.sqrt(np.mean(d*d))) if finite and d.size else (0. if not d.size else None),
        differing_numeric_elements=int(np.count_nonzero(a!=b)),new_array_sha256=C.array_sha(a),old_array_sha256=C.array_sha(b))

def compare():
    receipt=DEST/'baseline_reproduction.json'
    if receipt.exists(): return C.read(receipt)
    indexpath=C.OUT/'evaluation/raw_index.json'
    if not indexpath.is_file(): return dict(status='NOT_READY',reason='Complete new RAW index not yet available',model_calls=0)
    index=C.read(indexpath)
    if index.get('status')!='COMPLETE': return dict(status='NOT_READY',reason='RAW index not COMPLETE',model_calls=0)
    start=time.perf_counter();regpath=C.OUT/'registration.json';reg=C.read(regpath)
    byid={x['request_id']:x for x in index['rows']};base=[q for q in reg['requests'] if q['variant']=='base']
    if len(base)!=16 or len(index['rows'])!=48 or any(q['request_id'] not in byid for q in base):
        return dict(status='NOT_READY',reason='Need complete48 RAW and exactly16 registered base requests',model_calls=0)
    rows=[];states=[];full=[];artifacts={str(regpath):C.sha(regpath),str(indexpath):C.sha(indexpath)}
    for q in base:
        new=byid[q['request_id']]
        oldreceipt=C.OLD/'evaluation/raw/USER_EXPANDED_67'/q['base_request_id']/'receipt.json'
        old=C.read(oldreceipt);artifacts[str(oldreceipt)]=C.sha(oldreceipt)
        n,k=q['N'],q['K']
        ni=C.arrays(q['input_path']);oi=C.arrays(q['original_input_path'])
        na=C.arrays(new['raw_path']);oa=C.arrays(old['raw_path'])
        nt=na['trajectory'].reshape(51,n,9);ot=oa['trajectory'].reshape(51,n,9)
        nr=na['raw'].reshape(n,9);orr=oa['raw'].reshape(n,9)
        rawhashes={}
        for tag,row in [('new',new),('old',old)]:
            h=C.sha(row['raw_path']);rawhashes[tag]=dict(path=row['raw_path'],expected=row['raw_sha256'],actual=h,match=h==row['raw_sha256'])
            artifacts[row['raw_path']]=h
        checks=dict(checkpoint_same=new['checkpoint_sha256']==old['checkpoint_sha256']==reg['checkpoint']['sha256'],
            N_K_same=(new['N'],new['K'])==(old['N'],old['K'])==(n,k),
            source_input_C_equal=C.same(ni['C'],oi['C']),source_input_z_equal=C.same(ni['z'],oi['z']),
            input_N_K_equal=C.same(ni['N'],oi['N']) and C.same(ni['K'],oi['K']),
            receipt_C_identity=new['C_sha256']==old['C_sha256']==q['C_sha256'],
            receipt_Gaussian_identity=new['Gaussian_sha256']==old['Gaussian_sha256']==q['Gaussian_sha256'],
            output_C_matches_input=C.same(na['C'],ni['C']) and C.same(oa['C'],oi['C']),
            output_z_matches_input=C.same(na['z'],ni['z']) and C.same(oa['z'],oi['z']),
            raw_artifact_hashes_match=all(x['match'] for x in rawhashes.values()),
            initial_free_matches_noise=C.same(nt[0,k:].reshape(1,n-k,9),ni['z'][:,k:]) and C.same(ot[0,k:].reshape(1,n-k,9),oi['z'][:,k:]),
            both_final_trajectory_equal_raw=C.same(nt[-1],nr) and C.same(ot[-1],orr))
        td=diff(nt,ot);rd=diff(nr,orr);cd=diff(na['C'],oa['C']);zd=diff(na['z'],oa['z'])
        statebits=[]
        for i in range(51):
            sd=diff(nt[i],ot[i]);statebits.append(sd['bitwise_equal'])
            states.append(dict(request_id=q['request_id'],base_request_id=q['base_request_id'],parent_id=q['parent_id'],candidate=q['candidate'],
                state_index=i,time=i/50,bitwise_equal=sd['bitwise_equal'],numeric_equal=sd['numeric_equal'],maxabs=sd['maxabs'],rms=sd['rms'],differing_numeric_elements=sd['differing_numeric_elements']))
        passed=all(checks.values()) and all(x['bitwise_equal'] for x in (td,rd,cd,zd))
        row=dict(request_id=q['request_id'],base_request_id=q['base_request_id'],parent_id=q['parent_id'],base_condition_id=q['base_condition_id'],candidate=q['candidate'],
            condition_type=q['condition_type'],N=n,K=k,status='PASS_BITWISE' if passed else 'MISMATCH_NO_REPLAY',
            checkpoint_same=checks['checkpoint_same'],input_identity_pass=all(checks.values()),
            trajectory_all51_bitwise=td['bitwise_equal'],matching_states=sum(statebits),trajectory_maxabs=td['maxabs'],trajectory_rms=td['rms'],
            final_raw_bitwise=rd['bitwise_equal'],final_raw_maxabs=rd['maxabs'],final_raw_rms=rd['rms'],C_bitwise=cd['bitwise_equal'],z_bitwise=zd['bitwise_equal'])
        rows.append(row);full.append(dict(**row,checks=checks,trajectory=td,final_raw=rd,C=cd,z=zd,raw_files=rawhashes,
            new_input_path=q['input_path'],old_input_path=q['original_input_path'],old_receipt_path=str(oldreceipt)))
    status='PASS_BITWISE' if all(r['status']=='PASS_BITWISE' for r in rows) else 'MISMATCH_NO_REPLAY'
    result=dict(status=status,utc=C.utc(),registered_bases=16,compared=16,whole_trajectories_bitwise=sum(r['trajectory_all51_bitwise'] for r in rows),
        final_raws_bitwise=sum(r['final_raw_bitwise'] for r in rows),states_compared=len(states),states_bitwise=sum(r['bitwise_equal'] for r in states),
        max_trajectory_abs=max(r['trajectory_maxabs'] for r in rows),max_final_raw_abs=max(r['final_raw_maxabs'] for r in rows),
        rows=full,source_hashes=artifacts,implementation_sha256=C.sha(__file__),seconds=time.perf_counter()-start,
        model_calls=0,geometry_calls=0,replays=0,scope='Exact registered base inputs/noise/checkpoint and full51 saved states; no additional sampling')
    C.csvwrite(DEST/'baseline_reproduction.csv',rows);C.csvwrite(DEST/'baseline_reproduction_per_state.csv',states);C.write(receipt,result)
    return result

def protect():
    dest=DEST/'protected_asset_hashes.json'
    if dest.exists(): return C.read(dest)
    start=time.perf_counter();reg=C.read(C.OUT/'registration.json');sel=C.read(C.OUT/'selection.json')
    expected={};kinds=collections.defaultdict(set)
    def add(path,sha,kind):
        key=str(C.resolve(path).resolve())
        if key in expected and expected[key]!=sha:raise ValueError('Conflicting inherited hashes: '+key)
        expected[key]=sha;kinds[key].add(kind)
    for key in ('preparation_source_hashes','source_documents'):
        for p,h in reg[key].items():add(p,h,key)
    for q in sel['candidates']:add(q['source_path'],q['source_sha256'],'all64_original_candidate_C_NPZ')
    for q in reg['requests']:
        add(q['original_input_path'],q['original_input_sha256'],'original_selected_C_and_noise')
        add(q['original_condition_source'],q['original_condition_source_sha256'],'original_selected_source_mapping')
    for q in reg['queries']:add(q['inherited_query_path'],q['inherited_query_sha256'],'original_selected_C_queries')
    for key in ('checkpoint','d0_checkpoint'):
        a=reg[key];add(a['path'],a['sha256'],'protected_'+key)
    rows=[]
    for path in sorted(expected):
        p=C.resolve(path);exists=p.is_file();actual=C.sha(p) if exists else None
        rows.append(dict(path=path,kinds=sorted(kinds[path]),bytes=p.stat().st_size if exists else None,
            expected_sha256=expected[path],actual_sha256=actual,status='PASS' if exists and actual==expected[path] else 'MISMATCH_OR_MISSING'))
    checkpoints=[r for r in rows if any(k.startswith('protected_') for k in r['kinds'])]
    result=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'MISMATCH_OR_MISSING',utc=C.utc(),
        files=len(rows),total_bytes=sum(r['bytes'] or 0 for r in rows),read_hash_passes_per_unique_file=1,
        actual_full_checkpoint_hashes=checkpoints,rows=rows,seconds=time.perf_counter()-start,implementation_sha256=C.sha(__file__),
        policy='Original files read-only; no model loading. Existing completed audit receipt reused without another multi-GB hash pass.',model_calls=0)
    C.csvwrite(DEST/'protected_asset_hashes.csv',rows);C.write(dest,result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--compare',action='store_true');p.add_argument('--protect',action='store_true');a=p.parse_args()
    if not a.compare and not a.protect:p.error('Specify --compare and/or --protect')
    if a.compare:
        r=compare();print(json.dumps({k:v for k,v in r.items() if k not in ('rows','source_hashes')},ensure_ascii=False))
    if a.protect:
        r=protect();print(json.dumps({k:v for k,v in r.items() if k not in ('rows',)},ensure_ascii=False))

if __name__=='__main__':main()
