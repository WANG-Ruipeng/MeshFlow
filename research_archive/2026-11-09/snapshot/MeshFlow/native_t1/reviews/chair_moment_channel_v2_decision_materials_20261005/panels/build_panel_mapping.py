"""CPU-only identity extraction from frozen files; no sampler/scorer/model imports."""
from pathlib import Path
import argparse, csv, hashlib, json, shutil
from collections import Counter
import numpy as np

DEFAULT_ROOT = Path('/mnt/e/MeshFlow')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()

def legacy_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((str(a.dtype),tuple(a.shape))).encode()+a.tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();root=args.workspace;out=args.out
    out.mkdir(parents=True,exist_ok=True);(out/'source_tables').mkdir(exist_ok=True)
    v=root/'native_t1/runs/chair_moment_channel_v2'
    chair=root/'native_t1/runs/chair_domain_control_v1'
    old=root/'native_t1/runs/control_mvp48_v1'
    records_path=v/'registration/evaluation_records.json'
    panels_path=chair/'panels.json';patches_path=old/'data/patches.json'
    rows=json.loads(records_path.read_text());panels=json.loads(panels_path.read_text())
    patches=json.loads(patches_path.read_text())['records']
    sources={}
    def source(p,role):
        p=Path(p);key=str(p)
        if key not in sources:sources[key]=dict(path=key,workspace_relative=str(p.relative_to(root)),sha256=digest(p),bytes=p.stat().st_size,role=role)
        return sources[key]['sha256']
    for p,role in [(records_path,'V2 frozen evaluation registration'),(panels_path,'original A/B/C panels'),(patches_path,'C holdout original patch/source-face identity'),(v/'protocol.json','V2 experiment definition')]:source(p,role)
    shutil.copyfile(records_path,out/'source_tables/evaluation_records.json')
    shutil.copyfile(v/'protocol.json',out/'source_tables/protocol.json')
    (out/'source_tables/original_panel_records.json').write_text(json.dumps({t:panels[t] for t in ('A','B','C')},indent=2)+'\n')
    selected_patches=[p for p in patches if any(r['table']=='C' and r['uid']==p['uid'] and r['patch_id']==p['patch_id'] for r in rows)]
    (out/'source_tables/C_holdout_patch_records.json').write_text(json.dumps(selected_patches,indent=2)+'\n')
    patch_lookup={(p['uid'],p['patch_id']):p for p in selected_patches}
    contexts=[];slot_details={};bank={};checks=[]
    for index,r in enumerate(rows):
        base={k:r.get(k) for k in ('table','split','uid','task_id','patch_id','case_id','alpha','ratio','noise_index','N','K','status')}
        base.update(registration_row_index=index,input_path=r.get('input_path'),registered_input_sha256=r.get('input_sha256'))
        if r['status']!='READY':
            base.update(identity_status='NA',reason=r.get('reason'));contexts.append(base);continue
        p=Path(r['input_path']);p=p if p.is_absolute() else chair/p
        h=source(p,'frozen model input; read arrays only')
        if h!=r['input_sha256']:raise ValueError('Input file hash mismatch: '+str(p))
        with np.load(p,allow_pickle=False) as z:
            C=z['C'].copy();known=z['known_mask'].copy();valid=z['valid_mask'].copy()
            ids=z['source_face_ids'].copy() if 'source_face_ids' in z.files else None
        slots=np.flatnonzero(known).tolist();K=len(C);N=len(known)
        original_ids=None;ids_origin='input NPZ source_face_ids'
        if r['table']=='C':
            patch=patch_lookup[(r['uid'],r['patch_id'])]
            ids=np.asarray(patch['source_face_ids'],dtype=np.int64)
            original_ids=patch.get('original_source_face_ids')
            ids_origin='control_mvp48_v1/data/patches.json, keyed by uid and patch_id'
            checks.append(dict(case_id=r['case_id'],name='legacy C and known-mask hash',pass_=legacy_hash(C)==r['C_sha256'] and legacy_hash(known)==r['known_mask_sha256']))
        checks.append(dict(case_id=r['case_id'],name='known slot contract',pass_=slots==list(range(K)) and K==r['K'] and N==r['N'] and bool(valid.all()) and len(ids)==K))
        base.update(identity_status='PASS',input_path=str(p),input_file_sha256=h,C_shape=list(C.shape),C_dtype=str(C.dtype),C_array_sha256=array_hash(C),C_legacy_array_sha256=legacy_hash(C),known_mask_shape=list(known.shape),known_mask_dtype=str(known.dtype),known_mask_sha256=array_hash(known),known_mask_legacy_sha256=legacy_hash(known),valid_mask_sha256=array_hash(valid),known_slot_first=slots[0],known_slot_last=slots[-1],known_slot_count=len(slots),source_face_ids_sha256=array_hash(ids),source_face_ids_origin=ids_origin)
        slot_details[r['case_id']]=dict(known_model_slots=slots,source_face_ids=ids.tolist(),original_source_face_ids=original_ids,source_face_ids_origin=ids_origin,original_ids_status='AVAILABLE' if original_ids is not None else 'NOT_EXTRACTED',slot_to_source_face=list(zip(slots,ids.tolist())))
        contexts.append(base);bank[r['case_id']]=(C,known,ids)
    def uids(t,split=None,ratio=None):
        return sorted({r['uid'] for r in rows if r['table']==t and (split is None or r['split']==split) and (ratio is None or r['ratio']==ratio)})
    A=uids('A','validation',.2);B=uids('B');BV=uids('B','validation');C=uids('C')
    all_uids=sorted(set(r['uid'] for r in rows));membership=[]
    for uid in all_uids:
        membership.append(dict(uid=uid,A20_validation=uid in A,B_train=uid in uids('B','train'),B_validation=uid in BV,C_holdout=uid in C,registered_panels=';'.join(sorted({r['table']+'/'+r['split'] for r in rows if r['uid']==uid}))))
    comparisons=[]
    for uid in sorted(set(uids('A',ratio=.2))&set(B)):
        ar=next(r for r in rows if r['table']=='A' and r['uid']==uid and r['ratio']==.2 and r['noise_index']==0)
        br=next(r for r in rows if r['table']=='B' and r['uid']==uid and r['alpha']==1 and r['noise_index']==0)
        ac,ak,ai=bank[ar['case_id']];bc,bk,bi=bank[br['case_id']]
        aset,bset=set(ai.tolist()),set(bi.tolist())
        comparisons.append(dict(uid=uid,split=ar['split'],A20_case_id=ar['case_id'],B_alpha1_case_id=br['case_id'],A20_task_id=ar['task_id'],B_task_id=br['task_id'],same_known_model_mask=np.array_equal(ak,bk),same_C_bytes=ac.shape==bc.shape and ac.dtype==bc.dtype and ac.tobytes()==bc.tobytes(),same_source_face_order=np.array_equal(ai,bi),same_source_face_set=aset==bset,shared_source_faces=len(aset&bset),union_source_faces=len(aset|bset),source_face_jaccard=len(aset&bset)/len(aset|bset),A20_C_sha256=array_hash(ac),B_alpha1_C_sha256=array_hash(bc)))
    # Same B patch/noise copies must share C; alpha is part of the context identity.
    for r in rows:
        if r['table']!='B' or r['noise_index']!=0:continue
        q=next(s for s in rows if s['table']=='B' and s['task_id']==r['task_id'] and s['alpha']==r['alpha'] and s['noise_index']==1)
        checks.append(dict(case_id=r['case_id'],name='B context shared across registered noises',pass_=all(np.array_equal(a,b) for a,b in zip(bank[r['case_id']],bank[q['case_id']]))))
    target='c1b8c2f0c4f3cc43aff7c08b060f5ed6'
    summary=dict(schema='chair_v2_panel_identity_v1',status='PASS' if all(c['pass_'] for c in checks) else 'BLOCKED',C_word_disambiguation={'known_context':'The saved C triangle array and known_mask/model slots of each input.','panel_C':'The previously observed holdout panel, distinct from the mathematical known context C.'},A20_validation_uids=A,B_train_uids=uids('B','train'),B_validation_uids=BV,C_holdout_uids=C,A20_validation_B_overlap=sorted(set(A)&set(B)),A20_validation_not_in_B=sorted(set(A)-set(B)),A20_validation_C_overlap=sorted(set(A)&set(C)),B_C_overlap=sorted(set(B)&set(C)),c1b_membership=dict(A20_validation=target in A,B_any=target in B,C_holdout=target in C),registered_request_counts={t:dict(Counter(r['status'] for r in rows if r['table']==t)) for t in ('A','B','C')},comparison_conclusion='Matching parent UID or identical model known-mask slots does not identify the same known geometry/patch. Compare C bytes and original source-face IDs separately.',A20_B_alpha1_context_comparisons=comparisons,checks=checks,source_manifest='sources.csv',context_schema={'hash_current':'SHA256(str((array.shape, str(array.dtype))).encode() + C-contiguous array bytes), same as HYBRID/V2 array_hash','hash_legacy':'SHA256(str((str(array.dtype), tuple(array.shape))).encode() + C-contiguous bytes), same as old eval_prepare array_hash','source_face_ids':'Retained source mesh face rows, not model slots. C holdout also supplies original source-mesh row indices from frozen patch metadata.','known_model_slots':'0-based model face slots; C occupies 0 through K-1 in these frozen inputs.','C_shape':'[K,3,3] float32','known_mask_shape':'[N] bool','registration_row_index':'0-based entry in the exact bundled source_tables/evaluation_records.json','identity_status':'NA retains original NOT_APPLICABLE registrations without inferring missing input identity.'},scope='Existing-file reading, copying and hashing only; no model/geometry scoring, sampling, RNG or training.')
    def write_json(name,obj):
        (out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    def write_csv(name,values):
        keys=list(dict.fromkeys(k for r in values for k in r))
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
            for r in values:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
    write_json('panel_identity_summary.json',summary);write_json('context_slots.json',slot_details)
    write_csv('panel_parent_membership.csv',membership);write_csv('known_contexts.csv',contexts);write_csv('A20_B_context_comparison.csv',comparisons);write_csv('sources.csv',list(sources.values()))
    (out/'README.md').write_text('# Frozen panel and known-context identities\n\nRead `panel_identity_summary.json` first. This folder distinguishes known-context C from panel C (observed holdout).\n\n`c1b8c2f0c4f3cc43aff7c08b060f5ed6` is in A20 validation, but is absent from the entire B control panel and C holdout. B validation has four parents, all a subset of A20 validation eight; A20/B shared parents have different registered known patches. Do not attribute c1b width-response behavior from the B control curves.\n\n`panel_parent_membership.csv` lists panel membership. `known_contexts.csv` identifies every registered input, preserving unavailable rows, with input/C/mask/source-face hashes. `context_slots.json` maps 0-based model slots to source face IDs. `A20_B_context_comparison.csv` compares A20 with B alpha=1 for shared parents. All eight shared A20/B parents (four train and four validation) have equal model known-mask slots but different C byte hashes and different source-face sets at alpha=1. Model known-mask equality does not imply C geometry or source-face equality.\n\nAll C arrays are float32 [K,3,3], known/valid masks bool [N]; model known slots are 0..K-1. Historical and current array hash headers differ and are explicitly named. Source-face IDs address retained source mesh rows; C holdout additionally includes original pre-cleanup face IDs from frozen patch records. Original input arrays were read only, with no regenerated geometry or scores.\n\nOriginal identity sources are `native_t1/runs/chair_domain_control_v1/panels.json`, `native_t1/runs/chair_moment_channel_v2/registration/evaluation_records.json`, and `native_t1/runs/control_mvp48_v1/data/patches.json`. `source_tables/` contains the exact V2 evaluation manifest and protocol plus extracted original A/B/C records and C holdout patch records. `sources.csv` gives original absolute/relative paths, file hashes and sizes. `package_files.json` hashes every file here except itself. Reproduce using `python -B build_panel_mapping.py --workspace /mnt/e/MeshFlow --out <new_folder>` in an environment with NumPy. Reproduction reads original paths; this packaged mapping and source extracts are self-contained for reviewing the UID/context identity conclusions.\n',encoding='utf-8')
    manifest={str(p.relative_to(out)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='package_files.json'}
    write_json('package_files.json',manifest)
    print(json.dumps({k:summary[k] for k in ('status','A20_validation_B_overlap','A20_validation_not_in_B','c1b_membership','registered_request_counts')}))
    print('A20_B_contexts',json.dumps([{k:r[k] for k in ('uid','same_known_model_mask','same_C_bytes','same_source_face_set','source_face_jaccard')} for r in comparisons]))

if __name__=='__main__':main()
