"""Independent CPU data/evaluation/A0 identity gate; never imports a model.

Default execution waits if reviewed data is absent. Completed receipts can be
reverified without repeating OT probes. Source assets remain read-only.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,collections,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
OUT=ROOT/'experiments/r2_bench_coverage_expansion_v1'
GATES=OUT/'preparation/gates'
MAN=OUT/'data/manifest.json'
SCHEDULE=OUT/'data/schedule.json'
OLD=ROOT/'experiments/r2_bench_parts_convergence_v1/data/manifest.json'
OLD_EVAL=ROOT/'experiments/r2_bench_adaptation_strength_v1/eval_manifest.json'
OLD_START=ROOT/'experiments/r2_bench_adaptation_strength_v1/training/A_COMMON_LR1E5/checkpoint_step0.pt'
START_SHA='75d54c15f3ff2ee1328dddca1269352db48f5ecbca4d459eaff8c603c2297398'
REG=OUT/'training_registration.json'

def resolve(p):
    s=str(p).replace('\\','/')
    for prefix in ('E:/MeshFlow/','/mnt/e/MeshFlow/'):
        if s.startswith(prefix):return ROOT/s[len(prefix):]
    p=Path(s);return p if p.is_absolute() else ROOT/p

def sha(p):
    h=hashlib.sha256()
    with resolve(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read(p):return json.loads(resolve(p).read_text(encoding='utf-8-sig'))

def write(p,value):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def arrays(p):
    with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

class Audit:
    def __init__(self):self.checks=collections.Counter();self.inputs={}
    def require(self,ok,name):
        if not ok:raise ValueError('CPU_GATE_FAILED: '+name)
        self.checks[name]+=1
    def protect(self,p,expected=None):
        p=resolve(p);h=self.inputs.setdefault(str(p),sha(p))
        self.require(expected is None or h==expected,'registered_input_hash')
        return p
    def unchanged(self):
        for p,h in self.inputs.items():self.require(sha(p)==h,'protected_hash_unchanged')

def edgebook(f):
    book=collections.defaultdict(list)
    for fi,tri in enumerate(f):
        for k in range(3):
            u,v=int(tri[k]),int(tri[(k+1)%3]);book[tuple(sorted((u,v)))].append((fi,u,v))
    return book

def source_case_checks(a,row,c,s,pair):
    n,k=int(c['N']),int(c['K']);known=c['known_ids'];free=c['free_ids']
    a.require(128<=n<=256 and 2<=k<n,'N_K_contract')
    a.require(n==row['N'] and k==row['K'],'row_N_K_identity')
    a.require(np.isfinite(c['full_target']).all() and c['full_target'].shape==(n,3,3),'finite_target_shape')
    a.require(np.array_equal(c['full_target'],s['full_target']),'untouched_full_source_target')
    a.require(np.array_equal(c['pre_ot_full']*np.float32(2),c['full_target']),'pre_OT_half_scale')
    a.require(np.array_equal(np.sort(np.r_[known,free]),np.arange(n)),'known_free_source_partition')
    a.require(np.array_equal(c['C'],c['full_target'][known]) and np.array_equal(c['Y'],c['full_target'][free]),'exact_C_Y_complement')
    a.require(np.array_equal(c['C_vertices'][c['C_faces_model']],c['C']),'canonical_C_roundtrip')
    for key in ('known_ids','free_ids','C','Y','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_faces_original'):
        a.require(np.array_equal(c[key],pair[key]),'independent_pair_'+key)
    a.require(np.array_equal(c['C_source_faces_original'],s['source_faces'][known]),'source_C_original_winding')
    a.require(np.array_equal(c['C_source_vertex_ids'][c['C_faces_original']],s['source_faces'][known]),'source_C_vertex_ID_roundtrip')
    book=edgebook(c['C_faces_original'])
    a.require(all(len(x)<=2 for x in book.values()),'C_no_nonmanifold_edge')
    a.require(all(len(x)!=2 or x[0][1:]==x[1][1:][::-1] for x in book.values()),'C_internal_opposite_winding')
    edges=np.asarray([x[0] for _,x in sorted(book.items()) if len(x)==1],np.int64).reshape(-1,3)
    a.require(np.array_equal(edges,c['edge_records']),'independent_C_open_edge_order')
    b=len(edges);labels=np.asarray(c['occupied_labels']);support=np.asarray(c['source_supported_labels']);aliases=np.asarray(c['source_alias_labels'])
    a.require(labels.dtype==np.bool_ and labels.shape==support.shape==aliases.shape==(b,),'source_role_shapes')
    a.require(np.array_equal(labels|aliases,support) and not np.any(labels&aliases),'owner_alias_partition')
    a.require(b<=n-k,'boundary_slot_capacity')
    sourcebook=edgebook(s['source_faces']);ownerlist=[]
    for ei,(cf,u,v) in enumerate(edges):
        if not support[ei]:continue
        su,sv=int(c['C_source_vertex_ids'][u]),int(c['C_source_vertex_ids'][v]);fi=int(c['edge_owner_face'][ei]);tr=c['edge_source_triple'][ei]
        a.require(fi in set(map(int,free)),'supported_owner_is_free_source_face')
        a.require(len(sourcebook[tuple(sorted((su,sv)))])==2,'source_interface_two_face_incidence')
        a.require((fi,sv,su) in sourcebook[tuple(sorted((su,sv)))],'source_interface_reverse_edge')
        a.require(np.array_equal(tr[:2],[sv,su]) and set(map(int,tr))==set(map(int,s['source_faces'][fi])),'owner_source_triple_identity')
        matching=np.flatnonzero(support&(c['edge_owner_face']==fi))
        a.require(bool(labels[ei])==(int(ei)==int(matching.min())),'owner_is_smallest_supported_edge_alias_rule')
        if labels[ei]:ownerlist.append(fi)
    a.require(len(set(ownerlist))==len(ownerlist),'one_owner_per_source_face')
    a.require(int(labels.sum())==row['source_owner_count'],'row_owner_count')
    return dict(condition_id=row['condition_id'],parent_id=row['parent_id'],group=row['group'],N=n,K=k,boundary_edges=b,source_owners=int(labels.sum()),path='CLOSED_NATIVE' if b==0 else ('OPEN_ALL_BULK' if not labels.any() else 'SEAM_OWNER_ONLY'))

def cached_verify():
    latest=read(GATES/'latest.json');done=read(latest['completion'])
    if done['status']!='PASS':return None
    for p,h in read(done['input_hashes']).items():
        if sha(p)!=h:raise ValueError('Completed CPU gate source changed: '+p)
    if sha(__file__)!=done['gate_script_sha256']:raise ValueError('CPU gate code changed after completed receipt')
    return dict(status='PASS',mode='CACHED_HASH_REVERIFIED_NO_OT',completion=latest['completion'],new_model_calls=0,new_GPU_calls=0,new_OT_calls=0)

def run():
    GATES.mkdir(parents=True,exist_ok=True)
    missing=[str(p) for p in (MAN,SCHEDULE,OLD,OLD_EVAL,OLD_START,OLD_START.with_suffix('.json')) if not p.exists()]
    if missing:
        result=dict(status='WAITING_FOR_FROZEN_INPUTS',missing=missing,model_calls=0,GPU_calls=0,OT_calls=0)
        write(GATES/('waiting_'+str(time.time_ns())+'.json'),result);return result
    if (GATES/'latest.json').exists():
        result=cached_verify()
        if result:return result
    started=time.time();adir=GATES/('audit_'+str(time.time_ns()));adir.mkdir()
    audit=Audit();probe_records=[];ot_calls=0
    try:
        for p in (MAN,SCHEDULE,OLD,OLD_EVAL,OLD_START.with_suffix('.json'),Path(__file__),ROOT/'r2/pilots/bench_coverage_expansion_v1/data.py',ROOT/'r2/pilots/bench_parts_convergence_v1/data.py',ROOT/'r2/pilots/seam_abc_v1/slot_data_extension.py',ROOT/'r2/src/meshflow_control/seam/slot_data.py',ROOT/'r2/src/meshflow_control/training/coupling.py'):audit.protect(p)
        m=read(MAN);sch=read(SCHEDULE);old=read(OLD);panel=read(OLD_EVAL)
        if (OUT/'eval_manifest.json').exists():
            audit.protect(OUT/'eval_manifest.json')
            audit.require(read(OUT/'eval_manifest.json')==panel,'new_eval_manifest_exact_old_A')
        audit.require(len(old['train'])==277 and len(old['dev'])==39,'frozen_old277_39_counts')
        old_by={r['condition_id']:r for r in old['rows']};by={r['condition_id']:r for r in m['rows']}
        audit.require(len(by)==len(m['rows']),'unique_condition_ids')
        audit.require(len(m['dev'])==39 and m['dev']==old['dev'],'old_DEV39_rows_unchanged')
        for row in old['rows']:audit.require(by[row['condition_id']]==row,'original316_row_dictionaries_unchanged')
        trainp={r['parent_id'] for r in m['train']};devp={r['parent_id'] for r in m['dev']};oldtrainp={r['parent_id'] for r in old['train']}
        audit.require(not trainp&devp,'TRAIN_DEV_parent_disjoint')
        audit.require(oldtrainp<=trainp and 64<=len(trainp)<=79,'old64_retained_and_no_more79TRAIN')
        audit.require(all(r['official_split']=='train' for r in m['train']),'official_train_only')
        audit.require(len(m['train'])==277+len(m['new_train']),'old277_plus_reviewed_new_only')
        audit.require({r['condition_id'] for r in m['train']}|{r['condition_id'] for r in m['dev']}==set(by),'rows_train_dev_identity')
        cases={};sources={};case_records=[]
        for row in m['rows']:
            for key,hkey in (('case_path','case_sha256'),('source_case_path','source_case_sha256'),('paired_path','paired_sha256'),('supervision_file','supervision_sha256'),('input_path','input_sha256')):audit.protect(row[key],row[hkey])
            c=arrays(row['case_path']);cases[row['condition_id']]=c
            sp=str(resolve(row['source_case_path']))
            if sp not in sources:sources[sp]=arrays(sp)
            s=sources[sp];pair=arrays(row['paired_path']);sup=arrays(row['supervision_file'])
            for key in ('occupied_labels','edge_owner_face','edge_source_triple','source_supported_labels','source_alias_labels','edge_records'):audit.require(np.array_equal(c[key],sup[key]),'case_matches_frozen_supervision_'+key)
            case_records.append(source_case_checks(audit,row,c,s,pair))
            if row['condition_id'] not in old_by:
                rv=row['review'];audit.require(rv['actually_viewed'] and rv['decision']=='ACCEPT' and rv['human_plausible_input']=='YES' and rv['reviewer'].startswith('AI_PROXY'),'new_case_actual_image_review')
                audit.protect(rv['image_path'],rv['image_sha256'])
                audit.require(row['geometry_status'] in ('PASS_PENDING_SEMANTIC_REVIEW','PASS'),'new_C_geometry_gate')
        for p,h in m.get('protected_hashes',{}).items():audit.protect(p,h)
        audit.protect(m['schedule'],m['schedule_sha256'])
        steps=sch['steps'];audit.require(len(steps)==1500 and [e['step'] for e in steps]==list(range(1,1501)),'1500_registered_updates')
        exposure=collections.Counter();parents=collections.Counter();first={}
        for entry in steps:
            batch=entry['samples'];audit.require(len(batch)==8 and len({r['parent_id'] for r in batch})==4,'batch8_four_distinct_parents')
            audit.require([r['augmentation'] for r in batch]==['CLEAN']*4+['ANISOTROPIC']*4,'paired_augmentation_order')
            for mi,row in enumerate(batch):
                target=m['train'][row['condition_index']]
                audit.require((row['condition_id'],row['parent_id'],row['part_type'])==(target['condition_id'],target['parent_id'],target['part_type']),'schedule_index_identity')
                scale=np.asarray(row['scale_xyz']);audit.require(scale.shape==(3,) and np.isfinite(scale).all() and (scale>=.92).all() and (scale<=1.08).all(),'positive_bounded_anisotropic_scale')
                audit.require(row['sample_index']==mi,'microbatch_index')
                seed=10000000+entry['step']*1000+mi
                audit.require(all(row[k]==seed+offset for k,offset in [('noise_seed',0),('epsilon_seed',100000000),('time_seed',200000000),('permutation_seed',300000000)]),'same_A_seed_formula')
                exposure[row['condition_id']]+=1;parents[row['parent_id']]+=1;first.setdefault(row['condition_id'],row)
            for a,b in zip(batch[:4],batch[4:]):audit.require(a['condition_id']==b['condition_id'] and a['scale_xyz']==[1.,1.,1.],'identity_plus_augmented_same_condition')
        audit.require(sum(exposure.values())==12000 and dict(exposure)==sch['condition_exposure'],'exact12000_sample_exposure')
        audit.require(set(parents)==trainp and max(parents.values())-min(parents.values())<=2,'parent_balanced_exposure')
        audit.require(len(panel['requests'])==48 and len({r['request_id'] for r in panel['requests']})==48,'fixed48_unique_requests')
        for req in panel['requests']:
            cid=req['condition_id'];c=cases[cid];audit.require(req['N']==int(c['N']) and req['K']==int(c['K']),'eval_C_N_K_same_as_source')
            audit.require(not req.get('source_labels_available_to_models',True),'eval_no_Y_source_labels')
            if req['group']=='DEV':audit.require(req['parent_id'] not in trainp,'eval_DEV_never_train')
            for k,hk in [('inference_npz','inference_npz_sha256'),('noise_npz','noise_sha256'),('metadata_npz','metadata_sha256'),('source_npz','source_sha256'),('old_layout_npz','old_layout_sha256')]:audit.protect(req[k],req[hk])
            audit.protect(req['query_descriptor']['path'],req['query_descriptor']['sha256'])
            inp=arrays(req['inference_npz']);noise=arrays(req['noise_npz']);lay=arrays(req['old_layout_npz'])
            for key in ('C_vertices','C_faces_original','C_faces_model','N','K'):audit.require(np.array_equal(inp[key],c[key]) and np.array_equal(lay[key],c[key]),'eval_input_and_OLD_layout_'+key)
            audit.require(noise['z'].shape==(req['N']-req['K'],3,3) and np.isfinite(noise['z']).all() and int(noise['seed'])==req['seed'],'frozen_noise_shape_seed')
            audit.require(np.array_equal(lay['free_corner_mask'],lay['anchor_ids']<0),'old_layout_free_mask')
        audit.protect(OLD_START,START_SHA);receipt=read(OLD_START.with_suffix('.json'))
        audit.require(receipt['sha256']==START_SHA and receipt['local_step']==0 and receipt['global_step']==12000 and receipt['bytes']==OLD_START.stat().st_size,'exact_A0_checkpoint_file_identity')
        if REG.exists():
            reg=read(REG)
            audit.require(reg['status']=='FROZEN' and reg['updates']==1500 and reg['arms']==['COVERAGE_SEAM'],'registration_single_arm1500')
            audit.require(read(reg['eval_manifest'])==panel,'registered_eval_manifest_exact_old48')
            audit.require(resolve(reg['source_checkpoint'])==OLD_START and reg['source_checkpoint_sha256']==START_SHA,'registered_A0_source')
        # Three real scheduled cases, one for each existing data path. These are
        # bounded CPU OT/mask checks, not neural fixed probes or training calls.
        from . import data as D
        for category in ('CLOSED_NATIVE','OPEN_ALL_BULK','SEAM_OWNER_ONLY'):
            selected=next(r for r in case_records if r['group']=='TRAIN' and r['path']==category and r['condition_id'] in first)
            cid=selected['condition_id'];case=cases[cid];row=first[cid];sample=D.sample_for_training(case,row);ot_calls+=int(sample['production_OT_calls'])
            scaled=D.scaled_case(case,row['scale_xyz']);n,k=int(sample['N']),int(sample['K']);fixed=sample['anchor_ids']>=0
            expected=scaled['full_target'][sample['source_face_ids'][:,None],sample['source_corner_permutations']]
            audit.require(np.array_equal(expected,sample['x1']),'probe_OT_target_source_identity')
            audit.require(np.array_equal(np.sort(sample['source_face_ids']),np.arange(n)),'probe_all_source_faces_once')
            audit.require(np.array_equal(np.sort(sample['source_corner_permutations'],axis=1),np.tile(np.arange(3),(n,1))),'probe_corner_bijection')
            audit.require(np.array_equal(sample['free_corner_mask'],~fixed),'probe_free_mask_identity')
            audit.require(np.array_equal(sample['x0'][fixed],sample['C_vertices'][sample['anchor_ids'][fixed]]) and not sample['u'][fixed].any(),'probe_exact_anchor_zero_velocity')
            audit.require(int(sample['free_coordinate_count'])==int((~fixed).sum())*3==9*(n-k)-6*int(case['occupied_labels'].sum()),'probe_free_coordinate_denominator')
            audit.require(all(np.isfinite(sample[x]).all() for x in ('x0','x1','xt','u')),'probe_finite_all_arrays')
            audit.require(np.array_equal(sample['u'],sample['x1']-sample['x0']),'probe_FM_target_identity')
            t=np.float32(sample['t']);xt=(np.float32(1)-t)*sample['x0']+t*sample['x1'];xt[fixed]=sample['x0'][fixed]
            audit.require(np.array_equal(xt,sample['xt']),'probe_FM_path_identity')
            z=np.random.RandomState(row['noise_seed']).standard_normal((n-k,3,3)).astype(np.float32);eps=np.random.RandomState(row['epsilon_seed']).standard_normal(z.shape).astype(np.float32)
            mixed=np.float32(np.sqrt(.75))*z+np.float32(.5)*eps;inv=np.argsort(sample['face_permutation']);base=sample['x0'][inv];mask=sample['free_corner_mask'][inv]
            audit.require(np.array_equal(base[k:][mask[k:]],mixed[mask[k:]]),'probe_HYBRID025_noise_unscaled')
            probe_records.append(dict(path=category,condition_id=cid,N=n,K=k,owners=int(case['occupied_labels'].sum()),free_coordinates=int(sample['free_coordinate_count']),production_OT_calls=int(sample['production_OT_calls']),status='PASS'))
        audit.require('torch' not in sys.modules,'no_torch_import')
        audit.unchanged()
        write(adir/'input_hashes.json',audit.inputs);write(adir/'probe_records.json',probe_records)
        with (adir/'case_checks.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(case_records[0]));w.writeheader();w.writerows(case_records)
        result=dict(status='PASS',rows=len(m['rows']),train_conditions=len(m['train']),train_parents=len(trainp),dev_conditions=len(m['dev']),new_conditions=len(m['new_train']),
            parent_exposure_min=min(parents.values()),parent_exposure_max=max(parents.values()),registered_updates=1500,registered_samples=12000,
            closed_train=sum(r['group']=='TRAIN' and r['path']=='CLOSED_NATIVE' for r in case_records),open_zeroowner_train=sum(r['group']=='TRAIN' and r['path']=='OPEN_ALL_BULK' for r in case_records),
            eval_requests=48,old_eval_identity='EXACT_FROZEN_MANIFEST_AND_ALL_ARRAY_HASHES',initialization='A0_FILE_HASH_AND_REGISTERED_RECEIPT; NO_CHECKPOINT_DESERIALIZATION',source_checkpoint_sha256=START_SHA,
            input_hashes=str(adir/'input_hashes.json'),gate_script_sha256=sha(__file__),checks=dict(audit.checks),OT_calls=ot_calls,
            model_calls=0,model_imports=0,GPU_calls=0,training_updates=0,neural_forwards=0,neural_probe_calls=0,new_generation=0,all_protected_inputs_unchanged=True,wall_seconds=time.time()-started)
        write(adir/'completion.json',result);write(GATES/'latest.json',dict(status='PASS',completion=str(adir/'completion.json'),sha256=sha(adir/'completion.json')));write(OUT/'cpu_gate.json',result);return result
    except BaseException as exc:
        write(adir/'failure.json',dict(status='FAILED',error=repr(exc),checks=dict(audit.checks),OT_calls=ot_calls,model_calls=0,GPU_calls=0,wall_seconds=time.time()-started))
        raise

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(),ensure_ascii=False,indent=2))
