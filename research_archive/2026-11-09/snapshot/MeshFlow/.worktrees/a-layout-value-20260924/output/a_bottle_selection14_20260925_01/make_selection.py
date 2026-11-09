"""Freeze all14 already qualified bottles after explicit14x3x1 scope revision.
No new geometry screening, network, optimizer, MC, or old-output mutation.
"""
import argparse, hashlib, json, tarfile
from pathlib import Path
import numpy as np

SCREEN_P_SHA='47a8da2e1c772725287f48e20e2f86ff518a2f203918e53f5e69d56a3931dae3'
SCREEN_S_SHA='fe42a0ba6cd5a1b97c88b340d49d5adec917f7942264218c1f3668cc126b78bd'
SCREEN_C_SHA='1abc168ad75421ccb083c66b752e1c19eaa4f4314171c476ad018c92d4cc0621'
SCREEN_M_SHA='944ed7012ecafa0c7f74cde96c7285c3f73fe565f66b53c8a8d73fbc815ca354'
SOURCE_SHA='93a7d0e3ab3f3caeb3ddb5e368c8e1f6384e638e2e5296a58d64c86c08b5330e'
PSALT='a-condition16-parent-v1:'
DSALT='a-condition16-donor-v1:'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def arrhash(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a)
        h.update(str(a.dtype).encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def ordered(uids,salt): return sorted(uids,key=lambda u:(hashlib.sha256((salt+u).encode()).hexdigest(),u))
def write(path,data):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=True,indent=2,allow_nan=False);f.write('\n')
def require(ok,msg):
    if not ok: raise ValueError(msg)

def run(screen,study,study_sha):
    root=Path(__file__).resolve().parent
    screen,study=Path(screen).resolve(),Path(study).resolve()
    require(not (root/'SELECTION.json').exists(),'Selection already frozen')
    source_bindings={}
    for name,digest in [('SCREEN_PROTOCOL.json',SCREEN_P_SHA),('SELECTION.json',SCREEN_S_SHA),('COMPLETE.json',SCREEN_C_SHA),('FILE_MANIFEST.json',SCREEN_M_SHA)]:
        require(sha(screen/name)==digest,'Frozen screen changed: '+name)
        source_bindings[str(screen/name)]=digest
    require(sha(study)==study_sha,'Root revised study hash mismatch')
    source_bindings[str(study)]=study_sha
    p,s,c=read(screen/'SCREEN_PROTOCOL.json'),read(screen/'SELECTION.json'),read(screen/'COMPLETE.json')
    manifest=read(screen/'FILE_MANIFEST.json')['files']
    for name,digest in manifest.items():
        require(sha(screen/name)==digest,'Screen manifest mismatch: '+name)
        source_bindings[str(screen/name)]=digest
    source_bindings[str(screen/'READBACK.json')]=sha(screen/'READBACK.json')
    for name,digest in p['sources_sha256'].items(): require(sha(name)==digest,'Historical frozen source changed: '+name)
    require(s['status']==c['status']=='INSUFFICIENT_ELIGIBLE','Original16 protocol must stay insufficient')
    require(c['screened_count']==225 and c['eligible_count']==14 and c['runtime_errors_preserved']==0,'Wrong original screen result')
    require(s['selected16']==[] and s['data']=={} and s['donor_mapping']=={},'Original screen must not have substituted14')
    rows=[json.loads(line) for line in (screen/'SCREEN_ROWS.jsonl').read_text().splitlines()]
    require(len(rows)==225 and {x['uid'] for x in rows}==set(p['train_uids']) and len({x['uid'] for x in rows})==225,'225 denominator mismatch')
    require(not any(x['status']=='error' for x in rows),'Runtime error is not geometric rejection')
    candidates={x['uid']:x for x in rows if x['status']=='eligible'}
    uids=ordered(candidates,PSALT)
    require(len(uids)==14 and uids==s['eligible_uids_hash_order'],'All14 qualified parents must be included')
    donor_order=ordered(uids,DSALT)
    donors={uid:donor_order[(i+1)%len(donor_order)] for i,uid in enumerate(donor_order)}
    require(set(donors)==set(donors.values())==set(uids) and all(k!=v for k,v in donors.items()),'Bad donor permutation')
    require(not set(uids)&set(p['old_development_uids']),'Old development UID overlap')
    require(not {x['mainfield_hash'] for x in candidates.values()}&set(p['old_development_mainfield_hashes'].values()),'Old field alias')
    require(sha(p['source_archive'])==SOURCE_SHA,'Raw archive changed')
    source_bindings[p['source_archive']]=SOURCE_SHA
    data={}
    (root/'data').mkdir(exist_ok=False);(root/'raw').mkdir(exist_ok=False)
    with tarfile.open(p['source_archive'],'r:gz') as tar:
        for uid in uids:
            row=candidates[uid];src=Path(row['candidate_data_path'])
            require(sha(src)==row['candidate_data_sha256'],'Candidate changed')
            with np.load(src,allow_pickle=False) as z:
                require(set(z.files)=={'logr','base_u','faces','ref_vertices','ref_faces'},'Candidate schema changed')
                require(z['logr'].shape==(2050,) and z['logr'].dtype==np.float32,'Field shape/dtype changed')
                require(z['base_u'].shape==(258,3) and z['base_u'].dtype==np.float64,'Base shape/dtype changed')
                require(z['faces'].shape==(512,3) and z['faces'].dtype==np.int32,'Final template changed')
                require(z['ref_vertices'].shape==(2050,3) and z['ref_vertices'].dtype==np.float64,'Reference vertices changed')
                require(z['ref_faces'].shape==(4096,3) and z['ref_faces'].dtype==np.int64,'Reference faces changed')
                require(arrhash(z['logr'])==row['mainfield_hash'],'Mainfield typed hash changed')
                require(arrhash(z['ref_vertices'],z['ref_faces'])==row['reference_hash'],'Reference typed hash changed')
            dest=root/'data'/(uid+'.npz')
            with dest.open('xb') as f: f.write(src.read_bytes())
            raw_blob=tar.extractfile(row['archive_member']).read()
            require(hashlib.sha256(raw_blob).hexdigest()==row['source_sha256'],'Source mesh bytes changed')
            raw=root/'raw'/(uid+'.npz')
            with raw.open('xb') as f:f.write(raw_blob)
            require(sha(dest)==row['candidate_data_sha256'],'Byte copy changed')
            data[uid]={'path':str(dest),'sha256':sha(dest),'source_path':str(raw),'source_sha256':row['source_sha256'],
                       'source_candidate_path':str(src),'source_candidate_sha256':row['candidate_data_sha256'],
                       'archive_member':row['archive_member'],'source_faces':row['faces'],'source_mesh_hash':row['source_mesh_hash'],
                       'mainfield_hash':row['mainfield_hash'],'reference_hash':row['reference_hash'],
                       'split':'new_bottle_train_geometry_screened_no_model_use'}
    selection={'schema':'A-BOTTLE-ALL14-SELECTION-v1','status':'SELECTED_ALL14','role':'all14 geometry-qualified bottles, once-per-arm crosscategory',
        'root_study_protocol':str(study),'root_study_protocol_sha256':study_sha,'source_screen_root':str(screen),'source_screen_run':str(screen),
        'source_screen_complete_sha256':SCREEN_C_SHA,'source_screen_selection_sha256':SCREEN_S_SHA,'source_screen_file_manifest_sha256':SCREEN_M_SHA,
        'protocol_path':str(screen/'SCREEN_PROTOCOL.json'),'protocol_sha256':SCREEN_P_SHA,
        'source_selection_path':str(screen/'SELECTION.json'),'source_selection_sha256':SCREEN_S_SHA,
        'source_complete_path':str(screen/'COMPLETE.json'),'source_complete_sha256':SCREEN_C_SHA,
        'source_screen_status':'INSUFFICIENT_ELIGIBLE','source_archive_sha256':SOURCE_SHA,
        'source_original_requirement':16,'source_original_shortfall':2,'source_original_selected_count':0,
        'user_scope_revision':'explicitly use every14 qualified bottle, three arms once per parent,42 total; not a successful16-parent screen',
        'screened_count':225,'eligible_count':14,'selected_count':14,'selected_uids':uids,'selected14':uids,
        'lex_sorted_uids':sorted(uids),'eligible_uids_hash_order':uids,'train_uids':p['train_uids'],
        'parent_hash_salt':PSALT,'donor_hash_salt':DSALT,'donor_mapping':donors,'donor_uids_hash_order':donor_order,
        'data':data,'counts':s['counts'],'repeats':1,'planned_cases':42,'no_subset_of_qualified14':True,
        'source_F_le512_required':False,'final_output_faces':512,'post_selection_substitution_allowed':False,
        'no_new_geometry_screening':True,'models_optimizer_MC_calls':0,'old_screen_unchanged':True,
        'reference_semantics':p['reference_semantics'],'source_bindings_sha256':source_bindings,
        'preparation_source_sha256':sha(__file__)}
    write(root/'SELECTION.json',selection)
    for name,digest in source_bindings.items():require(sha(name)==digest,'Original binding changed during copy: '+name)
    files={str(path.relative_to(root)):sha(path) for path in sorted(root.rglob('*')) if path.is_file()}
    write(root/'FILE_MANIFEST.json',{'files':files})
    complete={'status':'READY_ALL14_NO_MODEL_RUN','selected_count':14,'planned_cases':42,'selection_sha256':sha(root/'SELECTION.json'),
              'file_manifest_sha256':sha(root/'FILE_MANIFEST.json'),'files':len(files),'all_source_bindings_unchanged':True,
              'eligible_source_face_range':[min(x['source_faces'] for x in data.values()),max(x['source_faces'] for x in data.values())],
              'eligible_source_faces_above512':sum(x['source_faces']>512 for x in data.values())}
    write(root/'COMPLETE.json',complete)
    print(json.dumps(complete))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--screen',required=True);a.add_argument('--study',required=True);a.add_argument('--study-sha',required=True);args=a.parse_args()
    run(args.screen,args.study,args.study_sha)