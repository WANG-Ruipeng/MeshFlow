"""Read original archive membership/metadata, prior screening and weight provenance.
No mesh preprocessing, geometry filters, model inference, optimization or MC.
"""
from pathlib import Path
import collections,hashlib,io,json,pickletools,re,tarfile,zipfile
import numpy as np
ROOT=Path.cwd();DOWNLOAD=Path('/mnt/c/Users/warpwang/Downloads')
OUT=ROOT/'output/a_condition_data_audit_20260925_01'
HAND=DOWNLOAD/'A_MESHFLOW_CODEX_HANDOFF_20260924/A_MESHFLOW_CODEX_HANDOFF_20260924'

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def array_hash(*values):
 h=hashlib.sha256()
 for a in values:
  a=np.ascontiguousarray(a);h.update(str(a.dtype).encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def split_ids(blob):
 ids=set();headers={}
 with zipfile.ZipFile(io.BytesIO(blob)) as z:
  for n in z.namelist():
   f=io.BytesIO(z.read(n));version=np.lib.format.read_magic(f)
   shape,fortran,dtype=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
   headers[n]={'shape':list(shape),'dtype':str(dtype),'fortran':fortran}
   if dtype.hasobject:
    # Parse literal string opcodes only; never execute a pickle or load test meshes.
    for op,arg,pos in pickletools.genops(f.read()):
     if isinstance(arg,str) and re.fullmatch('[0-9a-f]{16,64}',arg):ids.add(arg)
   else:
    a=np.load(io.BytesIO(z.read(n)),allow_pickle=False)
    ids.update(str(x) for x in np.asarray(a).reshape(-1) if re.fullmatch('[0-9a-f]{16,64}',str(x)))
 return sorted(ids),headers

parents=read(ROOT/'experiments/a_radial/assets/parents.json');old={x['uid']:x for x in parents['objects']}
recovery_path=HAND/'evidence/REAL16_RECOVERY.json';recovery=read(recovery_path)
seen_geometry={x['uid']:x for x in recovery['rows']}
archives={};speaker_members={};full_speaker_members={}
# Stream archives; retain speaker train numeric arrays solely for source identity.
for filename in ['shapenet-rebuttal2.tar.gz','shapenet-rebuttal.tar.gz','shapenet-cls.tar.gz','shapenet.tar.gz']:
 p=DOWNLOAD/filename;prefix_counts=collections.Counter();category_uids=collections.defaultdict(set);splits={};file_count=0
 with tarfile.open(p,'r|gz') as t:
  for m in t:
   if not m.isfile():continue
   file_count+=1;prefix=m.name.split('/')[0];prefix_counts[prefix]+=1
   if '/split/' in m.name and m.name.endswith('.npz'):
    blob=t.extractfile(m).read();uids,headers=split_ids(blob)
    splits[m.name]={'uids':uids,'uid_count':len(uids),'header':headers,'sha256':sha_bytes(blob)}
   if '/objaverse_occ_v5_ids/' not in m.name or not m.name.endswith('.npz'):continue
   uid=Path(m.name).stem;category_uids[prefix].add(uid)
   is_speaker=filename=='shapenet-rebuttal2.tar.gz' and prefix=='shapenet-03691459'
   is_full=filename=='shapenet.tar.gz' and uid in speaker_members
   if not (is_speaker or is_full):continue
   blob=t.extractfile(m).read();row={'uid':uid,'member':m.name,'bytes':len(blob),'npz_sha256':sha_bytes(blob)}
   # Official test members are identified/hashes only, no numeric geometry read.
   if uid in seen_geometry:
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
     v=np.asarray(z['vertices'],np.float64);f=np.asarray(z['faces'],np.int64)
     row.update(vertices=len(v),faces=len(f),vertices_hash=array_hash(v),faces_hash=array_hash(f),geometry_hash=array_hash(v,f))
   (speaker_members if is_speaker else full_speaker_members)[uid]=row
 archives[filename]={'path':str(p),'bytes':p.stat().st_size,'sha256':sha_file(p),
    'file_count':file_count,'prefix_file_counts':dict(prefix_counts),
    'categories':{k:{'object_files':len(v),'uids':sorted(v)} for k,v in category_uids.items()},'splits':splits}
 print('ARCHIVE_READ',filename,file_count,flush=True)
assert archives['shapenet-rebuttal2.tar.gz']['sha256']==parents['source_archive_sha256']
a=archives['shapenet-rebuttal2.tar.gz'];train=a['splits']['shapenet-03691459/split/train.npz']['uids'];test=a['splits']['shapenet-03691459/split/test.npz']['uids']
assert set(train)==set(seen_geometry) and len(train)==376 and len(test)==41 and not(set(train)&set(test))
assert set(speaker_members)==set(full_speaker_members)==set(train)|set(test)
full=archives['shapenet.tar.gz'];fulltrain=set(full['splits']['shapenet-new/split/train.npz']['uids']);fulltest=set(full['splits']['shapenet-new/split/test.npz']['uids'])
assert fulltrain & set(speaker_members)==set(train) and fulltest & set(speaker_members)==set(test)
for uid,obj in old.items():assert speaker_members[uid]['npz_sha256']==obj['sha256']
passed=[x for x in recovery['rows'] if x['phase']=='passed'];eligible=[x for x in passed if 20<x['faces']<800]
assert len(passed)==17 and {x['uid'] for x in eligible}==set(old)
for uid,row in speaker_members.items():
 row['original_split']='train' if uid in seen_geometry else 'test'
 row['previous_geometry_screen']=seen_geometry.get(uid)
 row['current16_model_teacher_development_pool']=uid in old
 if uid in old:row['matches_current_raw_source_file']=True
comparisons=[{'uid':u,'same_npz_bytes':speaker_members[u]['npz_sha256']==full_speaker_members[u]['npz_sha256'],
 'same_vertices_array':speaker_members[u].get('vertices_hash')==full_speaker_members[u].get('vertices_hash'),
 'same_faces_array':speaker_members[u].get('faces_hash')==full_speaker_members[u].get('faces_hash'),
 'same_geometry_arrays':speaker_members[u].get('geometry_hash')==full_speaker_members[u].get('geometry_hash')}
 for u in train]
# Only current run protocol files, not arbitrary personal files or generated shapes.
history=[];history_union=set()
def exact_known_strings(x):
 if isinstance(x,str):return {x} & set(speaker_members)
 if isinstance(x,list):return set().union(*(exact_known_strings(v) for v in x)) if x else set()
 if isinstance(x,dict):return set().union(*(exact_known_strings(k)|exact_known_strings(v) for k,v in x.items())) if x else set()
 return set()
for p in sorted((ROOT/'output').glob('*/RUN_PROTOCOL.json')):
 data=read(p);uids=exact_known_strings(data)
 if uids:
  history.append({'path':str(p),'sha256':sha_file(p),'speaker_UIDs_referenced':sorted(uids)})
  history_union|=uids
assert history_union<=set(old),sorted(history_union-set(old))
student=read(ROOT/'output/a_taskaligned_20260924_01/RUN_PROTOCOL.json')
# Bind current raw model file to historical ZIP bytes, without loading Torch.
model_path=ROOT/'experiments/a_radial/assets/original_A_10.pt';model_sha=sha_file(model_path)
restart=DOWNLOAD/'A_RESEARCH_STATE_20260923_RESTART.zip';reading=DOWNLOAD/'A_RESEARCH_STATE_20260923_READING.zip'
weight_proofs=[];documents=[];manifest_candidates=[];relevant_sources=[]
with zipfile.ZipFile(restart) as z:
 for e in z.infolist():
  if e.filename.endswith('original_A_10.pt'):
   digest=sha_bytes(z.read(e));assert digest==model_sha
   weight_proofs.append({'archive_member':e.filename,'sha256':digest,'bytes':e.file_size})
  if e.filename.endswith(('01_FULL_HANDOFF_zh.md','02_IMPLEMENTATION_AND_REPRO.md')) and '/A_domain_transport_handoff_20260918/' in e.filename and '/A_readonly_review_' not in e.filename:
   blob=z.read(e);text=blob.decode('utf-8-sig');selected=[{'line':i+1,'text':line} for i,line in enumerate(text.splitlines()) if any(k in line for k in ('192','0..191','3000','3,000','程序化','没有真实资产','29,907','A_teachers'))]
   documents.append({'member':e.filename,'sha256':sha_bytes(blob),'selected_evidence':selected})
  if any(k in e.filename for k in ('A_teachers','dataset.npz','target.npz','train_probe.py','FULL_SOURCE_MANIFEST')):manifest_candidates.append(e.filename)
  if e.filename.endswith(('A_SOURCE_VERIFIED.json','A_SOURCE_EXTRACT.json')):
   blob=z.read(e);d=json.loads(blob)
   if d.get('models/original_A_10.pt',{}).get('sha256')==model_sha:relevant_sources.append({'member':e.filename,'sha256':sha_bytes(blob),'model_binding':d['models/original_A_10.pt']})
provenance={'current_weight_path':str(model_path),'current_weight_sha256':model_sha,'restart_zip':str(restart),
 'restart_zip_sha256':sha_file(restart),'reading_zip':str(reading),'reading_zip_sha256':sha_file(reading),
 'matching_weight_members':weight_proofs,'source_binding_reports':relevant_sources,'training_documents':documents,
 'declared_original_A_training':{'objects':192,'object_ids':'0..191','kind':'procedural smooth/feature radial surfaces','teacher_faces':128,'A_updates':3000},
 'actual_training_sample_or_teacher_manifest_members_found':manifest_candidates,
 'full_training_samples_manifest_reverified':False,'claim_level':'training documentation plus exact weight-file provenance, not complete sample-level retraining audit'}
# Existing template and build source only, no function invocation.
code_paths=[HAND/'tools/recover16.py',ROOT/'experiments/a_radial/prepare_data.py',ROOT/'experiments/a_radial/mesh_ops.py',ROOT/'experiments/a_radial/exporter.py',ROOT/'experiments/a_radial/assets/support.npz']
report={'status':'AUDIT_COMPLETE_NO_NOVEL_ELIGIBLE_SPEAKER_POOL','scope':'Read-only local dataset/source identity and existing geometric-screen evidence; no new geometric filter, field build, model forward, optimizer, MC, training, test mesh geometry or data extraction.',
 'archives':archives,'speaker_rows':speaker_members,'full_archive_speaker_rows':full_speaker_members,
 'pool_counts':{'verified_speaker_parent_UIDs':417,'original_train_UIDs':376,'official_test_UIDs_excluded':41,
   'previous_geometry_screened_train_UIDs':376,'current16_teacher_model_development_pool':16,
   'outside_current16_train_UIDs_but_already_geometry_screened':360,'historical_geometric_pass':17,
   'historical_strict_eligible20F800':16,'strict_eligible_new_parent_UIDs':0,'requested_new_parents':16,'shortfall':16},
 'previous_screen':{'path':str(recovery_path),'sha256':sha_file(recovery_path),
   'phase_counts':dict(collections.Counter(x['phase'] for x in recovery['rows'])),
   'geometry_pass_but_face_rule_excluded':[x for x in passed if x not in eligible],
   'applies_to_archive_sha256':parents['source_archive_sha256'],'new_filters_rerun':False},
 'archive_set_relations':{'speaker_UID_set_and_train_test_split_identical_in_full_archive':True,
   'speaker_NPZ_bytes_identical_count':sum(speaker_members[u]['npz_sha256']==full_speaker_members[u]['npz_sha256'] for u in speaker_members),
   'train_geometry_array_comparisons':comparisons,'comparison_test_geometry_not_read':True,
   'full_archive_version_is_not_new_parent_and_old_screen_does_not_automatically_transfer':True,
   'full_archive_unlabelled_UIDs_cannot_be_declared_new_speakers_without_category_provenance':True},
 'history':{'current_workspace_RUN_PROTOCOL_sources':history,'referenced_real_speaker_UID_union':sorted(history_union),
   'teacher_and_development16':sorted(old),'early3k_parameter_fit_train12':student['train_uids'],
   'early3k_internal_check4_not_parameter_fit_but_seen_in_development':student['internal_check_uids'],
   'all376_previous_geometry_contact_UIDs':sorted(seen_geometry),'official_test_geometry_not_used_in_recovery':not recovery['source_test_used_for_geometry']},
 'original_A_pretraining_provenance':provenance,
 'mainfield_build_sources':{str(p):sha_file(p) for p in code_paths},
 'unchanged_build_contract':{'bbox_centre':True,'bbox_diagonal_normalization':True,'one_hit_check_directions':10244,
  'support_nodes':2050,'support_reference_faces':4096,'logr':'log(r / sqrt(mean(r*r))) float32',
  'exact_raw_endpoint_intersections_required':True,'topology':'closed oriented single genus-zero',
  'raw_face_range':'20<F<800 in existing recovery; no silent change','512_subdivision':'deterministic shared-edge midpoint; no old-vertex movement; builder can subdivide only and therefore requires input F<=512 for exact512 data',
  'no_holes_filled_no_closest_hit_replacement_no_category_switch':True},
 'future_screening_recommendation':{'authorized_pool_now':None,
  'require_new_same_category_source_and_UID_category_original_split_filehash_manifest':True,
  'exclude_previous376_UIDs_and_all_current16_source_or_geometry_duplicates':True,
  'exclude_official_test':True,'order':"ascending SHA256('a-condition-independent-v1:'+category+':'+uid), fixed before geometry/model inspection",
  'filter_order':['source/category/split/UID/hash eligibility','closed oriented genus-zero topology','bbox-centre10244 single-positive-ray hits','exact degenerate/intersection audit','predeclared face512 construction applicability'],
  'selection':'first16 passing predeclared geometry-only rules; retain all failures and stop/report shortfall if fewer16; never inspect A/Fixed gain during selection',
  'frozen_model_and_mismatched_condition_mapping_must_be_bound_before_model_evaluation':True},
 'limitations':['This is a search of known relevant workspace, Downloads archives and shallow WSL project/data locations, not a forensic whole-disk scan.',
 'Other local category packages exist but are outside current authorization and have no eligibility-screen result for this protocol.',
 'The aggregate package has source variants and unlabelled objects; they do not establish independent loudspeaker parents.',
 'Original_A training origin is document-and-weight-hash bound; full192 sample/teacher manifest was not available for individual verification.',
 'Earlier inventory deliberately excluded these raw tar archives; this audit expands the source scope and does not rewrite it.'],
 'source_files_not_modified':True,'new_candidate_mesh_data_written':False,'executed_audit_script_sha256':sha_file(__file__)}
path=OUT/'AUDIT.json'
with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=True);f.write('\n')
print(json.dumps({'status':report['status'],'pool_counts':report['pool_counts'],'phases':report['previous_screen']['phase_counts'],'AUDIT_sha256':sha_file(path)}),flush=True)
