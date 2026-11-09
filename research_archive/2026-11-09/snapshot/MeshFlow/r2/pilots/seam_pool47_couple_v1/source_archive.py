"""Final small-source archive; default status, no automatic closure.

Requires a PASS final audit and nonempty REPORT.md. Copies only Python source,
never weights, meshes, caches or dataset arrays. No Git command is executed.
"""
from __future__ import annotations
import argparse
import csv
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
EXPERIMENT=ROOT/'experiments/r2_seam_pool47_couple_v1'
OUT=EXPERIMENT/'source_archive'
VERSION='seam_pool47_couple_final_small_source_archive_v1'
MAX_FILE_BYTES=2<<20
MAX_TOTAL_BYTES=32<<20
# Explicit current-experiment authorities only. No repository-wide traversal.
AUTHORITIES={
    'training_registration.json':('cache_code_hashes','protected_hashes'),
    'training_dispatch_go.json':('code_hashes',),
    'generation_dispatch_go.json':('code_hashes',),
    'evaluation/score_registration.json':('source_hashes',),
    'cpu_binding/registration.json':('code',),
    'endpoint_binding/registration.json':('code',),
    'corner_rule/registration.json':('code',),
}


def resolve(value):
    s=str(value).replace('\\','/')
    if os.name!='nt' and re.match(r'^[A-Za-z]:/',s):s='/mnt/'+s[0].lower()+s[2:]
    if os.name=='nt' and re.match(r'^/mnt/[a-z]/',s):s=s[5].upper()+':/'+s[7:]
    p=Path(s);return (p if p.is_absolute() else ROOT/p).resolve()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest_bytes(value):return hashlib.sha256(value).hexdigest()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def safe_destination(relative):
    p=(OUT/relative).resolve()
    if not p.is_relative_to(OUT.resolve()):raise ValueError('Archive destination escaped experiment')
    return p


def gate(audit_path):
    p=resolve(audit_path);report=EXPERIMENT/'REPORT.md'
    if not p.is_relative_to(EXPERIMENT.resolve()):raise ValueError('Final audit must be inside this experiment')
    if not p.is_file() or read(p).get('status')!='PASS':raise RuntimeError('Not final: final_audit PASS is required')
    if not report.is_file() or not report.read_text(encoding='utf-8-sig').strip():raise RuntimeError('Not final: nonempty REPORT.md is required')
    audit=read(p)
    for key in ('training','generation','scores_and_binding','visuals','hashes'):
        # Actual audit may name its score/binding section differently; all
        # present checks must pass, and the top-level PASS is authoritative.
        if key in audit.get('checks',{}) and audit['checks'][key].get('status')!='PASS':
            raise ValueError('Final audit contains a non-PASS check: '+key)
    return p,report


def collect_sources():
    expected={};authority_sources={};provenance={}
    for relative,fields in AUTHORITIES.items():
        path=EXPERIMENT/relative
        if not path.is_file():raise FileNotFoundError('Required source registration missing: '+str(path))
        value=read(path);authority_sources[str(path)]=digest(path)
        for field in fields:
            mapping=value.get(field)
            if not isinstance(mapping,dict):raise ValueError('Missing registered source hash mapping: '+relative+':'+field)
            for source,h in mapping.items():
                if not str(source).lower().endswith('.py'):continue
                p=resolve(source);key=str(p)
                if not isinstance(h,str) or not re.fullmatch(r'[0-9a-f]{64}',h):raise ValueError('Invalid source hash in '+relative)
                if key in expected and expected[key]!=h:raise ValueError('Contradictory registered source versions: '+key)
                expected[key]=h;provenance.setdefault(key,[]).append(relative+':'+field)
    # This single owned Python package and direct experiment helper scripts are
    # the only discovery locations. Do not walk data/cache/mesh directories.
    new=set(p.resolve() for p in CODE.rglob('*.py'))
    new.update(p.resolve() for p in EXPERIMENT.glob('*.py'))
    sources=set(map(Path,expected))|new
    entries=[];total=0
    for p in sorted(sources,key=str):
        if not p.is_file():raise FileNotFoundError('Registered small source missing: '+str(p))
        size=p.stat().st_size
        if size>MAX_FILE_BYTES:raise ValueError('Source exceeds 2 MiB per-file guard: '+str(p))
        body=p.read_bytes();h=digest_bytes(body);key=str(p)
        if key in expected and h!=expected[key]:raise ValueError('Registered source changed after execution: '+key)
        if p in new:
            if not p.is_relative_to(ROOT.resolve()):raise ValueError('New code outside workspace')
            dest=Path('new_code')/p.relative_to(ROOT.resolve());category='NEW_EXPERIMENT_SOURCE'
        elif p.is_relative_to(ROOT.resolve()):
            dest=Path('reused_code')/p.relative_to(ROOT.resolve());category='REGISTERED_REUSED_SOURCE'
        else:
            identity=digest_bytes(str(p).encode('utf-8'))[:8]
            dest=Path('reused_external')/(h[:16]+'__'+identity)/p.name;category='REGISTERED_EXTERNAL_SMALL_SOURCE'
        safe_destination(dest);total+=len(body)
        entries.append(dict(source_path=key,archive_path=dest.as_posix(),sha256=h,bytes=len(body),category=category,
            registered_sha256=expected.get(key,''),authority_fields=provenance.get(key,[]),body=body))
    if total>MAX_TOTAL_BYTES:raise ValueError('Source archive exceeds 32 MiB source-only guard')
    return entries,authority_sources,total


def added_file_diff(entries):
    chunks=[]
    for entry in entries:
        if entry['category']!='NEW_EXPERIMENT_SOURCE':continue
        relative=Path(entry['archive_path']).relative_to('new_code').as_posix()
        content=entry['body'].decode('utf-8')
        chunks.append('diff --git a/'+relative+' b/'+relative+'\nnew file mode 100644\n')
        if not content:
            # Git's empty new-file form needs no content hunk.
            continue
        for line in difflib.unified_diff([],content.splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+relative):
            chunks.append(line)
            if not line.endswith('\n'):chunks.append('\n\\ No newline at end of file\n')
    return ''.join(chunks).encode('utf-8')


def index_text(entries,audit,report,total):
    count=sum(x['category']=='NEW_EXPERIMENT_SOURCE' for x in entries)
    artifacts=[
        ('Final report','REPORT.md'),('Final audit',audit.relative_to(EXPERIMENT).as_posix()),
        ('Frozen scientific protocol','protocol.json'),('Actual training analysis','training_analysis/complete.json'),
        ('Final 252-output numerical table','results/per_output.csv'),('Per-noise paired results','results/paired_per_noise.csv'),
        ('Per-parent paired results','results/paired_per_parent.csv'),('FF role ledger','results/illegal_pairs_by_role.csv'),
        ('Deletion ledger','results/deleted_faces.csv'),('All-bound connectivity/cost ledger','results/binding_ledger.csv'),
        ('Saved-layout CPU feasibility result','cpu_binding/NUMERICAL_REPORT.md'),
        ('Saved-layout CPU DEV detail','cpu_binding/NUMERICAL_ADDENDUM.md'),
        ('DEV finite-edge and alias records','cpu_binding/dev_diagnostic/completion.json'),
        ('Final same-1.5% binding receipt','endpoint_binding/completion.json'),
        ('RAW/POST gallery','gallery/training/index.html'),('Same-bound gallery','gallery/endpoint_binding/index.html'),
        ('Actual visual-review records','visual_review_endpoint.csv'),('Scientific summary figures','summary_figures/completion.json'),
        ('Resumable explicit stage entry','../../r2/pilots/seam_pool47_couple_v1/resume.py'),
    ]
    lines=['# Artifact index and small-source archive','',
        f'This archive contains {len(entries)} Python source snapshots ({count} new experiment files), {total:,} source bytes. It contains no weights, meshes, noise arrays, training cache or dataset copies.','',
        '`code.diff` is an explicit `/dev/null` → new-file review patch for the new experiment sources. It is not a Git working-tree diff, a diff against historical experiments, or evidence of a commit. Reused source files are preserved as snapshots and are not represented as newly authored code.','',
        'Source selection comes from this experiment\'s registered code hashes, plus its owned Python package and direct experiment helper scripts. The training dispatch registration intentionally covers a dependency envelope; inclusion alone does not prove that every dependency function executed. Exact provenance is retained in `source_hashes.csv`.','',
        '| Artifact | Local path |','|---|---|']
    for label,relative in artifacts:
        target=(EXPERIMENT/relative).resolve()
        status='' if target.exists() else ' — NOT PRESENT'
        lines.append('| '+label+' | [path](../'+relative.replace(' ','%20')+')'+status+' |')
    lines += ['', 'Weights remain in their existing locations and are not read or copied by this archiver:','']
    generation=read(EXPERIMENT/'generation_registration.json')
    for arm,item in sorted(generation.get('models',{}).items()):
        lines.append('- '+arm+': `'+str(item['path'])+'`; registered SHA-256 `'+item['sha256']+'`.')
    lines += ['', 'The gate records the report and final audit hashes in `completion.json`. This archiving action executes no experiment and changes no maintained default. It does not commit, push, remove files or promote a model.','']
    return '\n'.join(lines)


def run(audit_path):
    audit,report=gate(audit_path);completion=OUT/'completion.json'
    if completion.exists():
        prior=read(completion)
        if prior['gate_hashes']!={str(audit):digest(audit),str(report):digest(report)} or prior['source_archive_script_sha256']!=digest(__file__):
            raise ValueError('Existing archive has a different closure identity')
        for path,h in prior['files'].items():
            if digest(path)!=h:raise ValueError('Archived artifact changed: '+path)
        return prior
    if OUT.exists():raise RuntimeError('Partial source archive exists; preserve it, no silent overwrite/retry')
    # Validate and size every selected source before creating an archive.
    entries,authorities,total=collect_sources();patch=added_file_diff(entries)
    if len(patch)>MAX_TOTAL_BYTES:raise ValueError('New-code review patch exceeds 32 MiB guard')
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=False)
    gate_hashes={str(audit):digest(audit),str(report):digest(report)}
    write_json(OUT/'started.json',dict(status='ARCHIVING',version=VERSION,gate_hashes=gate_hashes,authority_hashes=authorities))
    files={};hashrows=[]
    try:
        for entry in entries:
            target=safe_destination(entry['archive_path']);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(entry['body'])
            if digest(target)!=entry['sha256']:raise ValueError('Snapshot copy hash differs')
            files[str(target)]=entry['sha256']
            hashrows.append({k:(';'.join(v) if isinstance(v,list) else v) for k,v in entry.items() if k!='body'})
        p=OUT/'code.diff'
        with p.open('xb') as f:f.write(patch)
        files[str(p)]=digest(p)
        p=OUT/'source_hashes.csv'
        with p.open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(hashrows[0]));writer.writeheader();writer.writerows(hashrows)
        files[str(p)]=digest(p)
        p=OUT/'ARTIFACTS.md'
        with p.open('x',encoding='utf-8') as f:f.write(index_text(entries,audit,report,total))
        files[str(p)]=digest(p)
        for path,h in {**authorities,**gate_hashes}.items():
            if digest(path)!=h:raise ValueError('Closure authority changed while archiving')
        # Small-source race check only. Never hash a weight or mesh here.
        for entry in entries:
            if digest(entry['source_path'])!=entry['sha256']:raise ValueError('Source changed during archive')
        value=dict(status='COMPLETE',version=VERSION,source_archive_script_sha256=digest(__file__),gate_hashes=gate_hashes,
            authority_hashes=authorities,source_files=len(entries),new_source_files=sum(e['category']=='NEW_EXPERIMENT_SOURCE' for e in entries),
            source_bytes=total,patch_bytes=len(patch),seconds=time.perf_counter()-start,files=files,
            models_copied=0,mesh_files_copied=0,dataset_files_copied=0,model_calls=0,scorer_calls=0,git_actions=0,
            diff_semantics='All new experiment Python files shown as /dev/null additions for review, not an actual Git diff')
        write_json(completion,value);return value
    except BaseException as exc:
        write_json(OUT/'failure.json',dict(status='STOPPED',error=repr(exc),seconds=time.perf_counter()-start,
            preserved_partial_archive=True,model_calls=0,git_actions=0));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('status','run'),default='status',nargs='?')
    p.add_argument('--audit',default=str(EXPERIMENT/'final_audit.json'));a=p.parse_args()
    if a.stage=='run':r=run(a.audit);r={k:v for k,v in r.items() if k not in ('files','authority_hashes')}
    else:
        audit=resolve(a.audit)
        r=dict(status='PREPARED_NO_AUTO_ARCHIVE',audit_exists=audit.exists(),report_exists=(EXPERIMENT/'REPORT.md').exists(),
            archive_completed=(OUT/'completion.json').exists(),model_calls=0,git_actions=0)
    print(json.dumps(r,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
