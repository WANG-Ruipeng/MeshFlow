"""Package existing decision evidence. No torch, model, training or scorer calls."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import shutil
import zipfile
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RUN=ROOT/'native_t1/runs/chair_moment_channel_v2'
CHAIR=ROOT/'native_t1/runs/chair_domain_control_v1'
UID='c1b8c2f0c4f3cc43aff7c08b060f5ed6'
LABELS=['START','H_CONT_END','MOMENT_XY_END']
ZIPNAME='CHAIR_MOMENT_CHANNEL_V2_DECISION_MATERIALS'
SOURCES=[]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    h.update(str(a.dtype).encode('ascii'));h.update(str(a.shape).encode('ascii'));h.update(a.tobytes())
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def write(rel,obj):
    p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def csvwrite(rel,rows):
    p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def source_path(v):
    if v.startswith('/mnt/e/MeshFlow/'):
        p=ROOT/v[len('/mnt/e/MeshFlow/'):]
    else:p=Path(v) if Path(v).is_absolute() else CHAIR/v
    p=p.resolve();assert p.is_relative_to(ROOT.resolve());return p

def copy(src,rel):
    src=Path(src);dst=OUT/rel;before=sha(src)
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    assert sha(dst)==sha(src)==before
    SOURCES.append({'source':src.relative_to(ROOT).as_posix(),'bundle_path':str(rel).replace('\\','/'),'sha256':before,'bytes':src.stat().st_size})
    return dst

def objwrite(rel,triangles):
    p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    a=np.asarray(triangles).reshape(-1,3,3)
    with p.open('w',encoding='ascii',newline='\n') as f:
        f.write('# Exact triangle-soup export; no welding, cleanup, repair or coordinate transform.\n')
        for v in a.reshape(-1,3):f.write('v '+' '.join(format(float(x),'.9g') for x in v)+'\n')
        for i in range(len(a)):f.write(f'f {3*i+1} {3*i+2} {3*i+3}\n')

def prepare():
    for f in sorted((RUN/'paired_results').glob('*.csv')):
        if f.stem in {'cost_summary','sampling_cost_per_output','precheck_cost_summary','numerical_comparisons'}:continue
        copy(f,'tables/'+f.name)
    for rel in ['protocol.json','protocol.sha256','checkpoint_manifest.json','accounting.json','RUN_COMPLETE.json','registration/evaluation_records.json','registration/training_selection.json','train_plan.json','independent_audit/quality_review.json','independent_audit/quality_review.md','independent_audit/final_audit.md','independent_audit/training_first_updates_MOMENT_XY.json']:
        copy(RUN/rel,'provenance/'+rel)
    for f in sorted((RUN/'source_code').rglob('*.py')):copy(f,'source_code/'+f.relative_to(RUN/'source_code').as_posix())
    copy(RUN/'code.diff','source_code/code.diff')
    for arm in ['H_CONT','MOMENT_XY']:
        for suffix in ['training_log.csv','sample_exposure.csv','events.jsonl','summary.json']:
            copy(RUN/f'training/{arm}_{suffix}',f'training/{arm}_{suffix}')
    copy(RUN/'training/shared_batch_hashes.jsonl','training/shared_batch_hashes.jsonl')
    observations=[]
    with (RUN/'training/MOMENT_XY_events.jsonl').open(encoding='utf-8') as f:
        for line in f:
            e=json.loads(line)
            if e.get('event')=='UPDATE_DONE':
                for r in e['relation_observations']:
                    observations.append(dict(moment_step=e['moment_step'],conditional_total_step=e['conditional_total_step'],**r))
    assert len(observations)==32000
    csvwrite('training/MOMENT_XY_relation_observations.csv',observations)

    prefix='geometry/'+UID
    reference=CHAIR/f'inputs/references/{UID}_A20_a100.npz'
    refmeta=read(reference.with_suffix('.json'))
    copy(reference,prefix+'/original/reference.npz');copy(reference.with_suffix('.json'),prefix+'/original/reference.json')
    with np.load(reference,allow_pickle=False) as z:query=z['coverage_points'].copy()
    assert array_sha(query)==refmeta['array_sha256']['coverage_points']
    csvwrite(prefix+'/frozen_queries.csv',[{'query_index':i,'x':float(q[0]),'y':float(q[1]),'z':float(q[2])} for i,q in enumerate(query)])
    per=read(RUN/'paired_results/per_output.json')
    generated={label:read(RUN/f'evaluation/{label}/generated_rows.json')['rows'] for label in LABELS}
    distance_rows=[];geometry_receipts=[];banks=[]
    for noise in (0,1):
        case=f'A_{UID}_A20_a100_n{noise}'
        selected={label:next(r for r in generated[label] if r['case_id']==case) for label in LABELS}
        bankpath=source_path(selected['START']['input_path'])
        copy(bankpath,prefix+f'/original/input_noise{noise}.npz')
        with np.load(bankpath,allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
        banks.append(bank)
        mask=bank['known_mask'];L=float(bank['L']);L0=float(bank['L0']);assert L==L0
        vectors={}
        for label in LABELS:
            row=selected[label];rawpath=source_path(row['raw_path']);assert sha(rawpath)==row['raw_sha256']
            copy(rawpath,prefix+f'/noise{noise}/{label}/original_raw.npz')
            write(prefix+f'/noise{noise}/{label}/generation_identity.json',row)
            vecpath=RUN/f'evaluation/{label}/scoring/fixed_vectors/{case}.npz'
            scorepath=RUN/f'evaluation/{label}/scoring/scores/{case}.json'
            copy(vecpath,prefix+f'/noise{noise}/{label}/original_fixed_vectors.npz')
            copy(scorepath,prefix+f'/noise{noise}/{label}/original_score.json')
            with np.load(rawpath,allow_pickle=False) as z:
                raw=z['raw'].copy().reshape(-1,3,3)
                assert np.array_equal(z['known_mask'],mask)
                assert np.array_equal(raw[mask],bank['C'])
                free=raw[~mask]
            target=OUT/prefix/f'noise{noise}/{label}/free_triangles.npy'
            np.save(target,free,allow_pickle=False)
            objwrite(prefix+f'/noise{noise}/{label}/free_mesh.obj',free)
            with np.load(vecpath,allow_pickle=False) as z:
                d=z['distances_L'].copy();assert str(z['query_sha256'])==array_sha(query)
            score=read(scorepath)['score']['detail']['fixed_GT_coverage']
            assert np.allclose(d,np.array(score['per_point_distances'])/L,rtol=1e-14,atol=0)
            pro=next(r for r in per if r['checkpoint_label']==label and r['case_id']==case)
            mse=float(np.mean(d*d));assert math.isclose(mse,float(pro['fixed_query_MSE']),rel_tol=1e-12,abs_tol=1e-15)
            vectors[label]=d
            geometry_receipts.append({'label':label,'case_id':case,'noise_index':noise,'N':len(raw),'K':int(mask.sum()),'free_faces':len(free),'query_count':len(query),'query_sha256':array_sha(query),'fixed_query_MSE':mse,'raw_file_sha256':sha(rawpath),'exported_free_array_sha256':array_sha(free),'known_coordinates_match':True})
        for i,q in enumerate(query):
            row=dict(noise_index=noise,query_index=i,x=float(q[0]),y=float(q[1]),z=float(q[2]),L=L,L0=L0)
            for label in LABELS:
                d=float(vectors[label][i]);row[label+'_distance_L']=d;row[label+'_distance_world']=d*L;row[label+'_squared_distance_L0']=d*d
            distance_rows.append(row)
    for key in ['C','known_mask','valid_mask','full_target','gt_free','source_face_ids','free_source_ids']:
        assert np.array_equal(banks[0][key],banks[1][key]),key
    shared={k:banks[0][k] for k in ['C','known_mask','valid_mask','full_target','gt_free','source_face_ids','free_source_ids','L','L0','N','K']}
    np.savez_compressed(OUT/prefix/'shared_C_GT.npz',**shared)
    objwrite(prefix+'/C_known.obj',shared['C']);objwrite(prefix+'/GT_free.obj',shared['gt_free']);objwrite(prefix+'/GT_full.obj',shared['full_target'])
    csvwrite(prefix+'/per_query_distances_START_H_M.csv',distance_rows)
    write(prefix+'/geometry_validation.json',{'status':'PASS','cases':geometry_receipts,'same_C_GT_across_noises':True,'query_order':'Same frozen coverage_points row index; no point resampling or distance recomputation.','mesh_export':'Boolean free-slot selection preserving original order; no repair or welding.','new_model_forwards':0})
    for r in SOURCES:assert sha(ROOT/r['source'])==r['sha256']
    write('source_copy_manifest.json',SOURCES)
    write('preparation_validation.json',{'status':'PASS','copied_source_files':len(SOURCES),'six_saved_mesh_outputs':len(geometry_receipts),'frozen_queries':len(query),'paired_query_rows':len(distance_rows),'training_relation_observations':len(observations),'source_files_unchanged':True,'new_training_updates':0,'new_model_forwards':0,'new_generations':0,'scoring_calls':0})
    print('PREPARE PASS',len(SOURCES),'source files; 6 meshes; 1024 paired query rows; 32000 existing telemetry rows')

def pack():
    assert read(OUT/'preparation_validation.json')['status']=='PASS'
    assert (OUT/'README.md').is_file()
    assert list((OUT/'panels').glob('*.csv')) and list((OUT/'panels').glob('*.json'))
    for r in read(OUT/'source_copy_manifest.json'):
        assert sha(ROOT/r['source'])==sha(OUT/r['bundle_path'])==r['sha256']
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='MANIFEST.sha256.json']
    assert not any(p.suffix in {'.pt','.pth','.ckpt'} for p in files)
    manifest={p.relative_to(OUT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    write('MANIFEST.sha256.json',manifest);files.append(OUT/'MANIFEST.sha256.json')
    archive=OUT.parent/(ZIPNAME+'.zip');assert not archive.exists(),'Do not overwrite previous delivery.'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,ZIPNAME+'/'+p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.infolist())==len(files)
        for p in files:
            h=hashlib.sha256()
            with z.open(ZIPNAME+'/'+p.relative_to(OUT).as_posix()) as f:
                for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
            assert h.hexdigest()==sha(p)
    receipt={'status':'PASS','zip':archive.name,'bytes':archive.stat().st_size,'MiB':archive.stat().st_size/2**20,'sha256':sha(archive),'file_count':len(files),'crc_and_all_member_hashes_verified':True,'no_large_weights':True,'new_model_calls':0}
    archive.with_suffix('.verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','pack']);args=parser.parse_args()
    prepare() if args.phase=='prepare' else pack()
