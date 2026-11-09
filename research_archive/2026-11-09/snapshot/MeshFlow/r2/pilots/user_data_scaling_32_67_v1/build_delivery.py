"""Final read-only-evidence packager. Implemented now; run only after root review.

No model imports, generation, image conversion, source geometry reads, or weight
hashing. Existing weight/array hashes are taken from receipts, with provenance.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import csv
import datetime
import difflib
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import zipfile

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
OUT = ROOT / 'experiments/r2_user_data_scaling_32_67_v1'
OLD = ROOT / 'experiments/r2_user_data_scaling_12h_v1'
OLD_CODE = CODE.parent / 'user_data_scaling_12h_v1'
ARMS = ('USER_SMALL', 'USER_EXPANDED_67')
HASH = re.compile(r'^[0-9a-f]{64}$')
BINARY = {'.pt', '.pth', '.ckpt', '.safetensors', '.npz', '.npy', '.ply', '.obj', '.glb'}
WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors'}
REQUIRED_CSV = ('per_output', 'quality_per_output', 'topology_per_output', 'AVMR_PMR_all8',
                'boundary_vertices', 'paired_per_request', 'paired_per_condition',
                'paired_per_parent', 'paired_summary', 'paired_strata')


class Blocked(RuntimeError): pass
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def resolve(v):
    v = str(v)
    if len(v)>2 and v[1]==':' and os.name!='nt': v='/mnt/'+v[0].lower()+'/'+v[3:].replace('\\','/')
    return Path(v)


def output(p):
    p = Path(p).absolute()
    if p == OUT or any(x.is_symlink() for x in (p,*p.parents)) or not p.resolve().is_relative_to(OUT.resolve()):
        raise Blocked('Unsafe output path: '+str(p))
    return p


def save(p, value):
    p=output(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def text(p,value):
    p=output(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(value)


def csvtext(rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));s=io.StringIO(newline='')
    w=csv.DictWriter(s,keys);w.writeheader()
    for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    return '\ufeff'+s.getvalue()


def require(p):
    p=Path(p)
    if not p.is_file() or not p.stat().st_size:raise Blocked('MISSING_REQUIRED: '+str(p))
    return p


@contextlib.contextmanager
def closed_workers():
    if os.name=='nt':raise Blocked('Run under the registered WSL environment')
    import fcntl
    with contextlib.ExitStack() as stack:
        for p in (OUT/'controller/owner.lock',OUT/'gpu_owner.lock'):
            if p.exists():
                f=stack.enter_context(p.open('rb'))
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:raise Blocked('ACTIVE_CONTROLLER_OR_GPU_WORKER: '+str(p))
        for arm in ARMS:
            s=read(require(OUT/'training'/arm/'status.json'))
            if s.get('status') not in ('COMPLETE','SEGMENT_COMPLETE','TIME_CUTOFF','INTERRUPTED_SAFE'):
                raise Blocked('TRAINING_NOT_CLOSED: '+arm+' '+str(s.get('status')))
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:cmd=p.read_bytes().replace(b'\0',b' ').decode('utf-8','replace')
            except (OSError,PermissionError):continue
            if str(p.parent.name)!=str(os.getpid()) and 'user_data_scaling_32_67_v1/training.py' in cmd:
                raise Blocked('ACTIVE_TRAINING_PROCESS: '+p.parent.name)
        yield


def gates(analysis_name):
    required=[OUT/'REPORT.md',OUT/'README_RESUME.md',OUT/'registration.json',OUT/'PROTOCOL.md',
              OUT/'training_budget_lock.json',OUT/'data/data_freeze.json',OUT/'data/train_manifest.json',
              OUT/'data/panel_manifest.json',OUT/'data/raw_inheritance_hashes.json',
              OUT/'evaluation/aggregate_completion.json',OUT/'evaluation/score_completion.json',
              OUT/'evaluation/gallery/output_review_receipt.json',OUT/'evaluation/gallery/manifest.json',
              OUT/'evaluation/gallery/index.html',OUT/'evaluation/registration.json',
              OUT/'evaluation/raw_index.json',OUT/'evaluation/post_index.json']
    required += [OUT/'evaluation'/(n+'.csv') for n in REQUIRED_CSV]
    required += [OUT/'report_tables'/n for n in (
        'receipt.json','REPORT_TABLES.md','absolute_parent_equal.csv',
        'absolute_per_condition.csv','absolute_per_parent.csv',
        'categorical_status_counts.csv','expanded_minus_small_both_tails.csv',
        'expanded_minus_small_per_parent.csv','expanded_minus_small_summary.csv',
        'metric_directions.csv','two_noise_counts.csv','two_noise_per_condition.csv',
        'two_noise_per_parent.csv')]
    required += [OUT/'review_tables'/n for n in (
        'receipt.json','README.md','classification_counts.csv','original_value_counts.csv',
        'classification_counts_per_parent.csv','original_value_counts_per_parent.csv',
        'expanded_vs_small_per_request.csv','observations_and_notes_index.csv')]
    for p in required:require(p)
    aggregate=read(OUT/'evaluation/aggregate_completion.json');score=read(OUT/'evaluation/score_completion.json')
    if aggregate.get('status')!='COMPLETE' or score.get('status')!='COMPLETE' or aggregate.get('stage_records')!=840:
        raise Blocked('AGGREGATE_OR_SCORING_INCOMPLETE; no success package is allowed')
    reg=read(OUT/'evaluation/registration.json');requests=reg['requests'];labels=reg['labels']
    if len(requests)!=140 or len({q['request_id'] for q in requests})!=140 or len(labels)!=3:
        raise Blocked('Registered 140-request / three-method identity differs')
    rr=read(OUT/'evaluation/gallery/output_review_receipt.json')
    if rr.get('status') not in ('COMPLETE','PARTIAL') or rr.get('review_kind') not in ('AI_PROXY','HUMAN'):
        raise Blocked('Actual visual-review receipt absent or invalid')
    review_file=resolve(rr['review_file']);require(review_file)
    if sha(review_file)!=rr.get('review_file_sha256'):raise Blocked('Actual review file hash mismatch')
    with review_file.open(encoding='utf-8-sig',newline='') as f:review_rows=list(csv.DictReader(f))
    expected={(q['request_id'],a,s) for q in requests for a in labels for s in ('RAW','POST')}
    got=[(r['request_id'],r['label'],r['stage']) for r in review_rows]
    viewed=[r for r in review_rows if r.get('review_status') in ('REVIEWED','REVIEWED_AI_PROXY','REVIEWED_HUMAN')]
    if set(got)!=expected or len(got)!=len(expected) or len(viewed)!=rr.get('actually_reviewed') or not viewed:
        raise Blocked('Review coverage/count mismatch; explicit unreviewed records are required')
    if rr['status']=='COMPLETE' and len(viewed)!=840:raise Blocked('PARTIAL_REVIEW_MISLABELLED_COMPLETE')
    if 'numeric_ranking_seen_before_review' not in rr:raise Blocked('Missing ranking-exposure disclosure')
    observation_summary=read(OUT/'review_tables/receipt.json')
    if observation_summary.get('status')!='COMPLETE_DESCRIPTIVE_OBSERVATION_COUNTS' or observation_summary.get('registered_stage_rows')!=840:
        raise Blocked('Actual observation summary is incomplete; run review_summary.py after the actual review receipt')
    if observation_summary.get('actually_viewed')!=rr['actually_reviewed'] or observation_summary.get('actual_review_status')!=rr['status']:
        raise Blocked('Observation summary does not match the final actual-review receipt')
    if any(not r.get('notes') or not r.get('reviewed_image_sha256') for r in viewed):raise Blocked('Missing actual image evidence')
    with (OUT/'evaluation/per_output.csv').open(encoding='utf-8-sig',newline='') as f:outrows=list(csv.DictReader(f))
    if len(outrows)!=840 or {(r['request_id'],r['label'],r['stage']) for r in outrows}!=expected:
        raise Blocked('Per-output CSV is incomplete')
    gallery=read(OUT/'evaluation/gallery/manifest.json');image_rows=gallery['receipts']
    image_expected={(q['request_id'],s) for q in requests for s in ('RAW','POST')}
    if len(image_rows)!=280 or {(r['identity']['request_id'],r['identity']['stage']) for r in image_rows}!=image_expected:
        raise Blocked('Gallery does not contain all 140 RAW and POST request pages')
    for r in image_rows:
        p=require(resolve(r['image_path']))
        if sha(p)!=r['image_sha256']:raise Blocked('Gallery image changed: '+str(p))
    if analysis_name:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',analysis_name):raise Blocked('Invalid analysis name')
        analysis=OUT/'training_analysis'/analysis_name
    else:
        found=[p.parent for p in (OUT/'training_analysis').glob('*/summary.json')]
        if len(found)!=1:raise Blocked('Specify --training-analysis NAME for exactly one final training accounting directory')
        analysis=found[0]
    a=read(require(analysis/'summary.json'))
    if a.get('status') not in ('PASS_CLOSED_TRAINING_ACCOUNTING','ACCOUNTING_ISSUES_PRESERVED'):
        raise Blocked('Final training accounting not completed')
    for n in ('updates','exposures','per_parent_exposure','diagnostic_per_fixture','diagnostic_groups','diagnostic_pairs'):
        require(analysis/(n+'.csv'))
    return dict(required=required,review_file=review_file,review=rr,gallery=gallery,analysis=analysis,
                training_accounting_status=a['status'],requests=requests,labels=labels)


class References:
    def __init__(self):
        self.hashes={};self.origins={};self.paths=set();self.scoring_sources=set();self.versions={};self.scorer_versions={}
    def add(self,p,h,origin,scorer=False):
        p=resolve(p)
        if not p.is_absolute():return
        p=str(p);self.paths.add(p)
        if isinstance(h,str) and HASH.fullmatch(h):
            self.versions.setdefault(p,{}).setdefault(h,set()).add(origin)
            self.hashes[p]=h;self.origins.setdefault(p,set()).add(origin)
            if scorer and Path(p).suffix=='.py':
                self.scoring_sources.add(p);self.scorer_versions.setdefault(p,set()).add(h)
    def walk(self,obj,origin,scorer=False):
        if isinstance(obj,list):
            for v in obj:self.walk(v,origin,scorer)
        elif isinstance(obj,dict):
            for k,v in obj.items():
                child_scorer=scorer or k=='source_hashes' and '/scores/' in origin
                if isinstance(k,str) and resolve(k).is_absolute() and isinstance(v,str):self.add(k,v,origin,child_scorer)
                if isinstance(v,str) and resolve(v).is_absolute():
                    stem=k[:-5] if k.endswith('_path') else k
                    h=obj.get(stem+'_sha256',obj.get(k+'_sha256'))
                    if k.endswith('_file'):h=h or obj.get(k[:-5]+'_sha256')
                    if k in ('path','npz','file','checkpoint'):h=h or obj.get('sha256') or obj.get('file_sha256')
                    if k=='image_path':h=obj.get('image_sha256')
                    self.add(v,h,origin,child_scorer)
                self.walk(v,origin,child_scorer)



def inherited_data_evidence(add):
    """Copy frozen input evidence; never rerun construction or reinterpret review."""
    prior_manifest=read(require(OLD/'review_file_manifest.json'))
    expected={str(resolve(r['path'])):r['sha256'] for r in prior_manifest['included']}
    provenance=[]
    def old_file(p,expected_sha=None,role='inherited_data_evidence'):
        p=require(p)
        if not p.is_relative_to(OLD) or p.suffix in BINARY:raise Blocked('Invalid inherited evidence: '+str(p))
        h=sha(p);recorded=expected_sha or expected.get(str(p))
        if recorded and h!=recorded:raise Blocked('Inherited evidence differs from frozen receipt: '+str(p))
        arc='inherited/'+p.relative_to(OLD).as_posix();add(p,arc)
        provenance.append(dict(source_path=str(p),archive_path=arc,bytes=p.stat().st_size,sha256=h,
            recorded_sha256=recorded,role=role,receipt_source=str(OLD/'review_file_manifest.json') if recorded else None))
    for rel in ('source_hashes.json','review_file_manifest.json','INPUT_REVIEW_GALLERY.html',
        'data/coverage_matrix.csv','data/qualified_conditions.csv','data/annotated_existing_proposals.csv',
        'data/annotated_existing_proposals.json','data/parent_candidates.json','data/parent_candidates_v2.json',
        'data/preprocessing_equivalence.json','inventory/input_hashes.json','inventory/quality_variant_summary.csv'):
        old_file(OLD/rel)
    # These are the actual prior construction/review/qualification implementations,
    # distinct from current reuse_data.py and from the unused old training pilot.
    names={'README.md','archive_inventory.py','asset_inventory.py','data_prepare.py','surface_proposals.py',
        'existing_review.py','data_review.py','quality_scan.py','data_freeze.py','verification_data.py','finalize.py'}
    codes=read(OLD/'source_hashes.json')['new_code'];seen=set()
    for r in codes:
        if Path(r['path']).name in names:
            p=resolve(r['snapshot']);old_file(p,r['sha256'],'original_constructor_snapshot')
            provenance[-1].update(original_source_path=r['path'],receipt_source=str(OLD/'source_hashes.json'))
            seen.add(p.name)
    if seen!=names:raise Blocked('Missing inherited constructor snapshots: '+str(sorted(names-seen)))
    for name,h in read(OLD/'data/data_freeze.json')['review_sources'].items():
        old_file(resolve(name),h,'frozen_AI_PROXY_input_review')
    for folder,pattern in (('data/proposals_v2','*.json'),('data/existing_region_review_sheets','*.json'),
        ('data/review_sheets','v2_*.json')):
        found=sorted((OLD/folder).glob(pattern))
        if not found:raise Blocked('Missing inherited region metadata: '+folder)
        for p in found:old_file(p,role='region_ids_and_input_sheet_metadata')
    # Keep the old input gallery browsable with its original images, not just an
    # external path. Copy bytes only; no image viewing, rendering or compression.
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.images=set()
        def handle_starttag(self,tag,attrs):
            if tag=='img':self.images.update(v for k,v in attrs if k=='src' and v)
    links=Links();links.feed((OLD/'INPUT_REVIEW_GALLERY.html').read_text('utf-8-sig'))
    if not links.images:raise Blocked('Inherited input gallery has no image references')
    for rel in sorted(links.images):
        p=(OLD/rel).resolve()
        if not p.is_relative_to(OLD.resolve()) or p.suffix not in ('.png','.jpg','.jpeg'):
            raise Blocked('Unsafe inherited image reference: '+rel)
        old_file(p,role='original_input_gallery_image')
    # Copy the small real constructor dependencies without importing/executing them.
    deps=[ROOT/'experiments/mf_h1_local0/control_mvp48/data_prepare.py',
          ROOT/'r2/data_tools/user_conditions_v1/prepare.py',ROOT/'r2/data_tools/user_conditions_v1/quality.py',
          ROOT/'r2/src/meshflow_control/source_winding.py',ROOT/'r2/src/meshflow_control/winding_solver.py']
    quality=deps[2];constants={}
    for node in ast.parse(quality.read_text('utf-8-sig')).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FROZEN_HASHES' for t in node.targets):
            constants=ast.literal_eval(node.value)
    frozen_base=ROOT.parent/'MeshFlow-Control-artifacts/experiments/chair_user_condition_data_audit_v1/geometry/source_snapshot/src/meshflow_control/evaluation'
    if set(constants)!={'pair_audit.py','intersections.py','seams.py'}:raise Blocked('Frozen data detector dependency identity missing')
    deps += [frozen_base/name for name in constants]
    for p in deps:
        require(p);h=sha(p);recorded=constants.get(p.name) if p.parent==frozen_base else None
        if recorded and h!=recorded:raise Blocked('Frozen data detector changed: '+str(p))
        rel=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else 'frozen_detector/'+p.name
        arc='code/inherited_data_dependencies/'+rel;add(p,arc)
        provenance.append(dict(source_path=str(p),archive_path=arc,bytes=p.stat().st_size,sha256=h,
            recorded_sha256=recorded,role='constructor_dependency',
            identity_scope='frozen detector identity checked' if recorded else 'current read-only source snapshot; original linkage retained in constructor'))
    return provenance


def input_review_projection():
    # A column projection only: retain every original accepted/rejected/uncertain
    # row and exact strings. The full 82.6MB CSV stays read-only at its original path.
    p=OLD/'data/condition_decisions.csv';old_limit=csv.field_size_limit()
    try:
        csv.field_size_limit(16*1024**2)
        with p.open(encoding='utf-8-sig',newline='') as f:
            rows=[{k:v for k,v in r.items() if k!='quality'} for r in csv.DictReader(f)]
    finally:csv.field_size_limit(old_limit)
    if not rows:raise Blocked('Inherited input decision CSV has no rows')
    original_hash=sha(p)
    manifest=read(OLD/'review_file_manifest.json')
    expected=next((r['sha256'] for r in manifest['included'] if str(resolve(r['path']))==str(p)),None)
    if expected and original_hash!=expected:raise Blocked('Original decision table changed from prior package receipt')
    return rows,dict(original_path=str(p),original_bytes=p.stat().st_size,original_sha256=original_hash,
        original_included_in_zip=False,original_row_count=len(rows),projection_row_count=len(rows),
        omitted_columns=['quality'],preserved_columns=list(rows[0]),
        statement='Column projection only, every row and original string retained. The omitted quality JSON contains repeated contact details; the complete original CSV remains read-only locally. This projection is not called the complete original table.')


def collect_files(gate):
    files={}
    def add(p,arc=None):
        p=Path(p)
        if not p.is_file() or p.suffix in BINARY:return
        arc=arc or p.relative_to(OUT).as_posix()
        if arc in files and files[arc]!=p:raise Blocked('Archive path collision: '+arc)
        files[arc]=p
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ('.md','.json','.diff','.txt'):add(p)
    for folder in ('data','training_analysis','report_tables','review_tables','review_work','report_fragments','engineering','controller','preflight','diagnostics','initialization','checkpoints','inference'):
        base=OUT/folder
        if base.exists():
            for p in base.rglob('*'):
                if p.suffix in ('.json','.jsonl','.csv','.md','.txt','.diff','.png','.jpg','.jpeg','.py','.yaml','.yml','.log'):add(p)
    for arm in ARMS:
        for p in (OUT/'training'/arm).rglob('*'):
            if p.suffix in ('.json','.jsonl'):add(p)
    for p in (OUT/'evaluation').rglob('*'):
        if p.suffix not in ('.json','.jsonl','.csv','.md','.html','.jpg','.jpeg','.png'):continue
        if p.name in ('pairs.json','shape_per_face.json'):continue
        if p.suffix=='.png' and (p.with_suffix('.jpg').exists() or p.with_suffix('.jpeg').exists()):continue
        add(p)
    # Actual chunks/allocation rules are evidence, including extensionless or
    # TSV artifacts. Do not filter them out merely by the usual report suffixes.
    for base in [OUT/'review_work',*(OUT/'evaluation').rglob('review_work')]:
        if base.is_dir():
            for p in base.rglob('*'):
                if p.is_file() and p.suffix not in BINARY:add(p)
    add(gate['review_file'])
    # Registered image paths override the optional PNG-versus-JPEG shortcut.
    # A PNG referenced by the actual render manifest is never dropped.
    for r in gate['gallery']['receipts']:add(resolve(r['image_path']))
    for p in CODE.glob('*.py'):add(p,'code/new/'+p.name)
    inherited=[OLD/'data/data_freeze.json',OLD/'data/verification_data.json',OLD/'data/verification_scope_note.json',
        OLD/'data/train_manifest.json',OLD/'data/panel_manifest.json',OLD/'data/qualified_condition_manifest.json',
        OLD/'data/parent_decisions.csv',OLD/'data/candidate_selection_rules.json',
        OLD/'inventory/historical_exclusions.json',OLD/'inventory/quality_summary.json']
    for p in inherited:
        require(p);add(p,'inherited/'+p.relative_to(OLD).as_posix())
    gate['inherited_data_provenance']=inherited_data_evidence(add)
    train=read(OUT/'data/train_manifest.json');panel=read(OUT/'data/panel_manifest.json')
    for t in [*train['tasks'].values(),*panel['conditions']]:
        for key in ('annotation_source','quality_path'):
            p=resolve(t[key]);require(p)
            if p.is_relative_to(OLD):add(p,'inherited/'+p.relative_to(OLD).as_posix())
    return files


def portable_gallery(gate):
    base=OUT/'evaluation/gallery';body=(base/'index.html').read_text('utf-8-sig')
    for r in gate['gallery']['receipts']:
        p=resolve(r['image_path'])
        if not p.is_relative_to(base):raise Blocked('Registered image outside gallery: '+str(p))
        rel=p.relative_to(base).as_posix()
        body=body.replace(p.as_uri(),rel).replace(str(p),rel)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=set()
        def handle_starttag(self,tag,attrs):
            self.links.update(v for k,v in attrs if k in ('src','href') and v)
    parser=Links();parser.feed(body)
    for r in gate['gallery']['receipts']:
        rel=resolve(r['image_path']).relative_to(base).as_posix()
        if rel not in parser.links:raise Blocked('Gallery HTML lacks portable image link: '+rel)
    return body


def inventory(refs,hash_budget_mib):
    # Enumerate only registered artifact roots; no disk-wide or OT-cache crawl.
    for folder in ('initialization','checkpoints','inference','evaluation/raw','evaluation/post','evaluation/inputs','evaluation/queries','diagnostics'):
        p=OUT/folder
        if p.exists():
            refs.paths.update(str(q) for q in p.rglob('*') if q.is_file() and q.suffix in BINARY)
    rows=[];hashed=0;limit=hash_budget_mib*1024**2
    for name in sorted(refs.paths):
        p=Path(name)
        if p.suffix not in BINARY:continue
        exists=p.is_file();size=p.stat().st_size if exists else None;h=refs.hashes.get(name)
        state='RECORDED_SHA_NOT_REHASHED' if h else 'NA_NO_FILE_RECEIPT'
        versions=refs.versions.get(name,{})
        if len(versions)>1:
            h=None;state='NA_MULTIPLE_HISTORICAL_HASHES_NO_CURRENT_RECEIPT'
            receipt=p.with_suffix('.json')
            if receipt.is_file():
                r=read(receipt)
                if str(resolve(r.get('path','')))==name and r.get('sha256') in versions:
                    h=r['sha256'];state='CURRENT_ADJACENT_RECEIPT_SHA_NOT_REHASHED'
        if not exists:state='MISSING_LOCAL_FILE'
        elif not h and p.suffix not in WEIGHTS and size<=64*1024**2 and hashed+size<=limit:
            h=sha(p);hashed+=size;state='COMPUTED_ONCE_MISSING_RECEIPT'
        elif not h and p.suffix in WEIGHTS:state='NA_WEIGHT_RECEIPT_MISSING_NO_REHASH'
        elif not h:state='NA_BOUNDED_HASH_BUDGET'
        rows.append(dict(path=name,bytes=size,sha256=h,hash_status=state,
            hash_receipt_sources=sorted(refs.origins.get(name,[])),included_in_zip=False,
            historical_file_hashes={h:sorted(origins) for h,origins in versions.items()},
            reason='Large weights / mesh arrays / source meshes / actual noise / queries remain at original local paths.'))
    return rows,hashed


def build(name,analysis_name,hash_budget_mib):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise Blocked('Simple unique package name required')
    target=output(OUT/'delivery'/name);archive=output(OUT/'delivery'/(name+'.zip'))
    if target.exists() or archive.exists():raise Blocked('Preserve prior delivery; choose a new --name, no overwrite')
    with closed_workers():
        g=gates(analysis_name);files=collect_files(g);refs=References();tick=time.monotonic()
        before={str(OUT/'training'/a/'status.json'):sha(OUT/'training'/a/'status.json') for a in ARMS}
        for arc,p in list(files.items()):
            if arc in ('inherited/review_file_manifest.json','inherited/source_hashes.json'):
                continue  # Broad old package inventories are provenance, not current asset requirements.
            if p.suffix=='.json' and p.stat().st_size<=64*1024**2:
                refs.walk(read(p),str(p))
        registration=read(OUT/'registration.json');initial=read(require(OUT/'initialization/audit.json'))
        refs.add(registration['official_generator_path'],initial['official_file_sha256'],str(OUT/'initialization/audit.json'))
        if not refs.scoring_sources:raise Blocked('Actual per-output scorer source hashes missing; cannot deliver only wrappers')
        source_rows=[]
        for name0,h in sorted(refs.hashes.items()):
            p=Path(name0)
            if p.suffix!='.py':continue
            require(p);actual=sha(p)
            if actual not in refs.versions[name0]:raise Blocked('Actual dependency source changed without recorded identity: '+str(p))
            if name0 in refs.scorer_versions and refs.scorer_versions[name0]!={actual}:
                raise Blocked('Actual scorer source differs between executed stage records or current source: '+str(p))
            rel=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else 'external/'+hashlib.sha256(name0.encode()).hexdigest()[:16]+'/'+p.name
            arc='code/dependencies/'+rel;files[arc]=p
            source_rows.append(dict(source_path=str(p),archive_path=arc,sha256=actual,actual_scorer_dependency=name0 in refs.scoring_sources,
                receipt_sources=sorted(refs.origins[name0]),historical_hashes={h:sorted(origins) for h,origins in refs.versions[name0].items()}))
        for p in sorted(CODE.glob('*.py')):
            source_rows.append(dict(source_path=str(p),archive_path='code/new/'+p.name,sha256=sha(p),actual_scorer_dependency=p.name=='scoring.py',receipt_sources=['current new pilot source']))
        local_rows,hashed=inventory(refs,hash_budget_mib)
        missing=[r for r in local_rows if r['hash_status'].startswith('NA_') or r['hash_status']=='MISSING_LOCAL_FILE']
        # A missing primary checkpoint must never hide behind a successful ZIP.
        critical=[r for r in local_rows if Path(r['path']).suffix in WEIGHTS and (r['hash_status']=='MISSING_LOCAL_FILE' or not r['sha256'])]
        if critical:raise Blocked('Missing local checkpoint or recorded hash: '+json.dumps(critical[:4]))
        target.mkdir(parents=True)
        generated={}
        def product(rel,value,istext=False):
            p=target/rel;(text if istext else save)(p,value);generated[rel]=p
        product('evaluation/gallery/index.html',portable_gallery(g),True)
        product('source_hashes.json',source_rows)
        input_rows,input_projection_receipt=input_review_projection()
        product('inherited/INHERITANCE_PROVENANCE.json',dict(files=g['inherited_data_provenance'],
            no_new_data_construction=True,no_new_input_review=True,
            original_order_deviation_preserved=True,input_projection=input_projection_receipt))
        product('inherited/input_review_decisions_compact.csv',csvtext(input_rows),True)
        product('local_artifacts.csv',csvtext(local_rows),True)
        product('local_artifacts.json',dict(rows=local_rows,missing_or_hash_NA=missing,new_array_hash_bytes=hashed,weight_bytes_rehashed=0))
        diff=[];bases=[]
        for p in sorted(CODE.glob('*.py')):
            old=OLD_CODE/p.name;prior=old.read_text('utf-8-sig').splitlines(True) if old.exists() else []
            current=p.read_text('utf-8-sig').splitlines(True)
            diff.extend(difflib.unified_diff(prior,current,fromfile='prior/'+p.name if old.exists() else '/dev/null',tofile='new/'+p.name))
            bases.append(dict(new_path=str(p),new_sha256=sha(p),base_path=str(old) if old.exists() else None,base_sha256=sha(old) if old.exists() else None))
        product('code.diff',''.join(diff),True);product('code_diff_bases.json',bases)
        product('README_PACKAGE.md',
            '# R2 32-versus-67 review package\n\n'
            'Start with REPORT.md, then evaluation/gallery/index.html for every registered RAW/POST request. '
            'All original per-output, per-condition and per-parent CSVs and actual review records are included.\n\n'
            'This small review archive cannot independently train or regenerate models. Weights, RAW, trajectories, '
            'D0 proposals, POST arrays, source meshes, Gaussian arrays and query arrays remain local. '
            'local_artifacts.csv lists actual paths, sizes, recorded SHA provenance and explicit missing/NA states. '
            'Recorded weight SHA values were not recomputed during packaging.\n\n'
            'The full existing JPEG gallery is copied unchanged; no image was redrawn, cropped or recompressed. '
            'Any existing source PNG remains local. Actual visual-review coverage is '+g['review']['status']+' ('+str(g['review']['actually_reviewed'])+'/840 rows).\n\n'
            'code/new contains this experiment source; code/dependencies contains the actual small source files '
            'named by execution/scoring receipts. code.diff compares each new pilot file with the preserved '
            'same-name prior blocked-run pilot file, or /dev/null for a new file. It is not a fabricated Git commit.\n\n'
            'inherited/ contains the actual old constructor snapshots, region face-ID lists, coverage matrix, '
            'complete acceptance/exclusion tables, all five frozen AI_PROXY input-review files and the original '
            'INPUT_REVIEW_GALLERY.html with its existing images. INHERITANCE_PROVENANCE.json records original '
            'paths, hashes and snapshot identities. input_review_decisions_compact.csv retains every original '
            'decision row and exact strings while omitting only the repeated quality JSON column. The full '
            '82.6MB condition_decisions.csv remains read-only locally, with its original path, size and SHA '
            'in INHERITANCE_PROVENANCE.json; the projection does not replace the full original table. '
            'No input was newly reviewed or reconstructed.\n\n'
            'The earlier blocked run and qualification-order deviation remain unchanged in inherited/. '
            'Use README_RESUME.md for local recovery; this archive authorizes no further scientific work.\n',True)
        for rel,p in generated.items():files[rel]=p
        entries=[]
        for arc,p in sorted(files.items()):
            if p.suffix in BINARY:raise Blocked('Binary scientific asset leaked into archive plan: '+str(p))
            entries.append(dict(archive_path=arc,source_path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
        product('PACKAGE_MANIFEST.json',dict(experiment='R2_USER_DATA_SCALING_32_67_V1',files=entries,
            status='REVIEW_PACKAGE_READY_WITH_DISCLOSED_NA' if missing or g['review']['status']=='PARTIAL' else 'REVIEW_PACKAGE_READY',
            visual_review_status=g['review']['status'],training_accounting_status=g['training_accounting_status'],
            registered_requests=140,gallery_images=280,stage_records=840,local_asset_NA_count=len(missing)))
        files['PACKAGE_MANIFEST.json']=generated['PACKAGE_MANIFEST.json']
        if before!={p:sha(p) for p in before}:raise Blocked('Training status changed during packaging')
        tmp=output(archive.with_suffix('.zip.partial'))
        with zipfile.ZipFile(tmp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for arc,p in sorted(files.items()):z.write(p,arc)
        with zipfile.ZipFile(tmp,'r') as z:
            if len(z.namelist())!=len(files) or len(z.namelist())!=len(set(z.namelist())):raise Blocked('ZIP membership mismatch')
            expected={r['archive_path']:r['sha256'] for r in entries}
            expected['PACKAGE_MANIFEST.json']=sha(generated['PACKAGE_MANIFEST.json'])
            for arc,h in expected.items():
                actual=hashlib.sha256()
                with z.open(arc) as f:
                    for block in iter(lambda:f.read(1024*1024),b''):actual.update(block)
                if actual.hexdigest()!=h:raise Blocked('ZIP content/hash changed: '+arc)
        os.replace(tmp,archive)
        result=dict(status='DELIVERY_COMPLETE_WITH_DISCLOSED_LIMITATIONS' if missing or g['review']['status']=='PARTIAL' or g['training_accounting_status']!='PASS_CLOSED_TRAINING_ACCOUNTING' else 'DELIVERY_COMPLETE',
            utc=utc(),archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),
            visual_review_status=g['review']['status'],actually_reviewed=g['review']['actually_reviewed'],
            local_asset_NA_count=len(missing),actual_scorer_sources=len(refs.scoring_sources),
            wall_seconds=time.monotonic()-tick,new_model_calls=0,new_geometry_calls=0,weight_bytes_rehashed=0,
            statement='Archive packaging complete; scientific conclusion is only in root-authored REPORT.md. No missing artifact is called a successful experiment.')
        save(target/'delivery_receipt.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',action='store_true',help='Perform final gated packaging; default prints instructions only')
    p.add_argument('--name',default='R2_USER_DATA_SCALING_32_67_REVIEW')
    p.add_argument('--training-analysis',help='Existing final training_analysis subdirectory; inferred only if exactly one exists')
    p.add_argument('--inventory-hash-budget-mib',type=int,default=256)
    a=p.parse_args()
    if not a.build:
        print('IMPLEMENTED_NOT_EXECUTED: final use --build after REPORT, actual visual review, aggregate and closed training accounting.');return 0
    try:
        if not 0<=a.inventory_hash_budget_mib<=1024:raise Blocked('Bounded array hash budget must be 0..1024 MiB')
        print(json.dumps(build(a.name,a.training_analysis,a.inventory_hash_budget_mib),ensure_ascii=False,indent=2));return 0
    except (Blocked,KeyError,ValueError,OSError) as exc:
        r=dict(status='DELIVERY_BLOCKED',utc=utc(),reason=str(exc),no_success_claim=True,new_model_calls=0)
        save(OUT/'delivery'/('blocked_'+str(time.time_ns())+'.json'),r)
        print(json.dumps(r,ensure_ascii=False,indent=2));return 2


if __name__=='__main__':raise SystemExit(main())
