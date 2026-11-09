"""CPU read-only trajectory inventory. No project/model/scorer imports."""
import collections,csv,hashlib,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_bench_flow_integrity_diagnostic_v1/audits/trajectory_inventory'
CUR=ROOT/'experiments/r2_bench_seat_coverage_v1'
FULL=ROOT/'experiments/r2_bench_full_seat_curriculum_v1'
OLD=ROOT/'experiments/r2_bench_trajectory_diagnostic_v1'
CROSS=ROOT/'experiments/r2_bench_trajectory_crossover_v1'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(p,data):
    raw=(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode()
    if p.exists():assert p.read_bytes()==raw,str(p)
    else:p.write_bytes(raw)
def csvwrite(p,rows):
    import io
    text=io.StringIO(newline='');keys=list(dict.fromkeys(k for r in rows for k in r))
    w=csv.DictWriter(text,fieldnames=keys);w.writeheader();w.writerows({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in rows)
    raw=('\ufeff'+text.getvalue()).encode()
    if p.exists():assert p.read_bytes()==raw,str(p)
    else:p.write_bytes(raw)

def inspect(q,label,root,layout,manifest,ckpt=None,L=None,kind='CURRENT'):
    rid=q['request_id'];d=root/'evaluation'/label/rid/'raw';p=d/'trajectory.npz'
    tr=arrays(p);state=tr.get('state',tr.get('states'));lay=arrays(layout);raw=arrays(d/'raw.npz')
    N,K=int(q['N']),int(q['K']);fixed=lay['anchor_ids']>=0
    assert state.shape==(51,N,3,3) and state.dtype==np.float32 and np.isfinite(state).all(),str(p)
    noise=arrays(q['noise_npz'])['z'];initial=np.zeros((N,3,3),np.float32);initial[K:]=noise
    initial[fixed]=lay['C_vertices'][lay['anchor_ids'][fixed]]
    assert np.array_equal(initial,state[0]) and np.array_equal(state[-1],raw['raw'])
    assert all(np.array_equal(x[fixed],initial[fixed]) for x in state)
    assert np.array_equal(lay['free_corner_mask'],~fixed)
    assert np.array_equal(lay['known_mask'],np.arange(N)<K)
    assert np.array_equal(raw['raw_anchor_ids'],lay['anchor_ids'])
    if 'times' in tr:assert np.array_equal(tr['times'],np.arange(51)/50)
    hasv='velocities' in tr
    if hasv:
        assert tr['velocities'].shape==(50,N,3,3)
        for j,v in enumerate(tr['velocities']):
            nx=state[j]+np.float32(.02)*v;nx[fixed]=initial[fixed]
            assert np.array_equal(nx,state[j+1])
    receipt=read(d/'receipt.json');h=sha(p)
    registered=[v for k,v in receipt.get('files',{}).items() if Path(k).name=='trajectory.npz']
    if registered:assert registered==[h]
    return dict(cohort=kind,label=label,parent_id=q['parent_id'],condition_id=q['condition_id'],request_id=rid,
        eval_group=q.get('eval_group',q.get('split')),noise_index=q['noise_index'],N=N,K=K,
        trajectory_path=str(p),trajectory_sha256=h,trajectory_array_key='state' if 'state'in tr else 'states',
        trajectory_shape=list(state.shape),nodes=51,t0=0,t1=1,dt=.02,stored_velocity=hasv,
        stored_times='times'in tr,all51_finite=True,all51_fixed_corners_exact=True,initial_noise_writeback_exact=True,
        final_equals_raw_canonical=True,stored_Euler_velocity_recurrence_exact=True if hasv else 'NA_NO_STORED_VELOCITY',
        identity_transport='fixed slot and corner addresses across all51 states; no per-time OT',
        free_corner_count=int((~fixed).sum()),free_coordinate_count=int((~fixed).sum()*3),
        seam_slots=len(lay['seam_slots']),bulk_slots=len(lay['bulk_slots']),fixed_free_corners=int(fixed[K:].sum()),
        layout_path=str(layout),layout_sha256=sha(layout),layout_keys=list(lay),
        free_mask_field='free_corner_mask (equals anchor_ids<0)',role_fields='known_mask, seam_slots, bulk_slots, anchor_ids, edge_records, occupied',
        noise_path=q['noise_npz'],noise_sha256=sha(q['noise_npz']),input_path=q['inference_npz'],input_sha256=sha(q['inference_npz']),
        source_condition_path=q.get('source_npz'),metadata_path=q.get('metadata_npz'),case_path=q.get('case_path'),
        raw_path=str(d/'raw.npz'),raw_sha256=sha(d/'raw.npz'),receipt_path=str(d/'receipt.json'),receipt_sha256=sha(d/'receipt.json'),
        manifest_path=str(manifest),manifest_sha256=sha(manifest),checkpoint=ckpt,L=L,L_policy='frozen prior score reference, not parent bbox',
        geometry_frame='original model/world coordinates, canonical C in state; raw.output restores source C corner order',
        scientific_execution_this_inventory='0 model,0 scorer,0 new noise,0 OT')

def run():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];weights=[]
    curman=read(CUR/'eval_manifest.json');fullman=read(FULL/'eval_manifest.json')
    for root,labels,qs,kind,manifest in [
        (FULL,('COMMON_S12000','NATIVE_S6000','SEAM_S3000'),fullman['requests'],'FULL_SEAT_CURRICULUM',FULL/'eval_manifest.json'),
        (CUR,('NATIVE_S6000','COVERAGE_FINAL'),curman['requests'],'CURRENT_COVERAGE',CUR/'eval_manifest.json')]:
        for label in labels:
            reg=read(root/'evaluation/inference_registrations'/(label+'__RAW.json'))
            spec=reg['spec'];cp=Path(spec['generator_path'])
            weights.append(dict(cohort=kind,label=label,path=str(cp),exists=cp.is_file(),bytes=cp.stat().st_size if cp.exists() else None,
                registered_sha256=reg['protected'].get(str(cp)),hash_policy='copied from immutable inference registration; no expensive weight reread or load',spec=spec,
                registration_path=str(root/'evaluation/inference_registrations'/(label+'__RAW.json')),registration_sha256=sha(root/'evaluation/inference_registrations'/(label+'__RAW.json'))))
            for q in qs:
                if root==CUR and label=='NATIVE_S6000' and q['panel_origin']=='INHERITED24':continue
                lp=root/'evaluation'/label/'layouts'/q['condition_id']/'layout.npz'
                rows.append(inspect(q,label,root,lp,manifest,weights[-1],curman['L'] if root==CUR else fullman['L'],kind))
    oldreg=read(OLD/'registration.json');oldman=read(OLD/'evaluation_inputs/manifest.json')
    oldscore=read(OLD/'evaluation/score_registrations/COMMON.json')
    oldL=oldscore.get('L',float(np.sqrt(3)*5.31488037109375))
    for label in ('COMMON','USER'):
        cp=Path(oldreg['checkpoints'][label]);weight=dict(cohort='OLD_COMMON_USER',label=label,path=str(cp),exists=cp.exists(),bytes=cp.stat().st_size if cp.exists() else None,
            registered_sha256=oldreg['files'].get(str(cp)),hash_policy='copied from frozen registration; no weight loading or rehash',registration_path=str(OLD/'registration.json'),registration_sha256=sha(OLD/'registration.json'))
        weights.append(weight)
        for q in oldman['requests']:
            rows.append(inspect(q,label,OLD,OLD/'layouts'/q['condition_id']/'layout.npz',OLD/'evaluation_inputs/manifest.json',weight,oldL,'OLD_COMMON_USER'))
    crosses=[];creg=read(CROSS/'registration.json')
    for combo in ('CU','UC'):
        for q in creg['requests']:
            p=CROSS/'evaluation'/combo/q['request_id']/'raw/trajectory.npz';tr=arrays(p)
            assert tr['states'].shape==(26,q['N'],3,3) and tr['velocities'].shape==(25,q['N'],3,3)
            assert np.array_equal(tr['times'],np.arange(25,51)/50)
            crosses.append(dict(combo=combo,request_id=q['request_id'],path=str(p),sha256=sha(p),nodes=26,t0=.5,t1=1,stored_velocity=True,eligible_pure_model_smoothness=False,
                exclusion='intentional model switch at .5; not a pure single-model flow; first half is read-only baseline reference'))
    # Exact identity pairing within current Native/Candidate; older model pairs use a different panel/representation.
    pairs=[]
    for q in curman['requests']:
        a=next(r for r in rows if r['label']=='NATIVE_S6000' and r['request_id']==q['request_id'])
        b=next(r for r in rows if r['label']=='COVERAGE_FINAL' and r['request_id']==q['request_id'])
        for key in ('noise_sha256','input_sha256','layout_sha256','N','K','free_corner_count'):assert a[key]==b[key],(q['request_id'],key)
        pairs.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],eval_group=q['eval_group'],native_trajectory=a['trajectory_path'],candidate_trajectory=b['trajectory_path'],noise_sha256=a['noise_sha256'],layout_sha256=a['layout_sha256'],exact_same_current_condition_noise_layout=True))
    recommendations=[]
    reason={'a8e987':'new trained regular slat seat; large POST loss', '27eb79':'new trained curved seat; RAW organized but thin supports lost after processing','838c48':'old trained reasonable slab/dual support control','6b894c':'old trained candidate FF/I tail and visibly torn infill','f495bd':'old DEV C underneath/replaced by gray deck','df6fd3':'current-lineage-unseen holdout; only one parent, two noises'}
    for q in curman['requests']:
        prefix=q['parent_id'][:6]
        if prefix not in reason:continue
        rec={k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','N','K','eval_group','case_path','inference_npz','metadata_npz','source_npz','noise_npz','query_descriptor')}
        rec.update(reason=reason[prefix],selection_scope='post-observation mechanism panel, not a new unbiased success-rate test',new_model_calls_needed_for_trajectory=0,
            trajectories=[r for r in rows if r['request_id']==q['request_id'] and r['label']in ('NATIVE_S6000','COVERAGE_FINAL','SEAM_S3000')])
        recommendations.append(rec)
    extras=[]
    for p in sorted((OLD/'execution').glob('*/*/*/*trajectory.npz')):
        with np.load(p,allow_pickle=False) as z:
            extras.append(dict(path=str(p),keys=z.files,shapes={k:list(z[k].shape) for k in z.files},scope='source-owner oracle, not production layout; excluded from pure C-only primary panel'))
    csvwrite(OUT/'per_trajectory.csv',rows);save(OUT/'per_trajectory.json',rows)
    csvwrite(OUT/'cross_suffixes.csv',crosses);save(OUT/'checkpoint_identities.json',weights);save(OUT/'current_identity_pairs.json',pairs)
    save(OUT/'recommended_panel.json',dict(status='PROPOSED_NOT_NEW_DISPATCH',parents=6,requests=len(recommendations),requests_data=recommendations,selection_after_known_results=True))
    save(OUT/'extra_source_oracle_assets.json',extras)
    summary=dict(status='PASS',model_calls=0,scoring_calls=0,training_updates=0,new_noise=0,new_OT=0,full51_trajectories=len(rows),
        counts=dict(collections.Counter(r['cohort']+'/'+r['label'] for r in rows)),stored50_velocity_trajectories=sum(r['stored_velocity'] for r in rows),
        suffix26_mixed_model_trajectories=len(crosses),current_native_candidate_exact_pairs=len(pairs),all_requested_assets_full51=True,
        sparse_snapshots_found_in_requested_raw_trajectories=0,raw_endpoint_fixed_mask_noise_checks='PASS',
        reference_scales={'CURRENT':curman['L'],'OLD':oldL,'scale_rounding_difference':abs(curman['L']-oldL)},
        source_code_paths=['r2/pilots/bench_control_transfer_v1/sample.py','r2/pilots/bench_trajectory_diagnostic_v1/gpu.py','r2/pilots/bench_trajectory_crossover_v1/prepare.py','r2/pilots/bench_full_seat_curriculum_v1/audit_all_raw_cache.py'],
        script_sha256=sha(__file__),notes=['No current trajectory velocities saved: diff(state)/.02 is realized discrete velocity, not bitwise recovery of BF16 network output.', 'Use existing role masks and per-model free coordinate denominator. Current SEAM changes fixed corners; do not compare whole arrays without role/free masking.', 'Historical CC/UU refer directly to OLD COMMON/USER full trajectories. CU/UC cannot estimate a single-model smoothness.', 'Source-informed continuation_t*.npz and prediction_t*.npz are not full noise-to-output production trajectories.'])
    save(OUT/'receipt.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
    assert 'torch' not in sys.modules

if __name__=='__main__':run()
