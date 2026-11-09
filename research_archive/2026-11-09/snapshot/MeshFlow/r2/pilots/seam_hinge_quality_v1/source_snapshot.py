"""Final source-only evidence snapshot. Import/status never freezes source.

Explicit --finalize is required. Registered code hashes are read from completed
training/generation/scoring/binding/HINGE registration records; only .py text is
copied. The diff covers this new pilot directory alone, never workspace changes.
"""
from __future__ import annotations
import argparse,collections,datetime,difflib,hashlib,json,os,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
NEW=ROOT/'r2/pilots/seam_hinge_quality_v1'
OUT=ROOT/'experiments/r2_seam_hinge_quality_v1'
VERSION='seam_hinge_quality_final_python_source_snapshot_v1'
HEX=re.compile(r'^[0-9a-fA-F]{64}$')


def resolve(value):
    s=str(value).replace('\\','/')
    if os.name!='nt' and re.match(r'^[A-Za-z]:/',s):s='/mnt/'+s[0].lower()+s[2:]
    elif os.name=='nt' and re.match(r'^/mnt/[a-z]/',s):s=s[5].upper()+':/'+s[7:]
    p=Path(s);return p if p.is_absolute() else ROOT/p


def sha(value):return hashlib.sha256(value).hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def jsonbytes(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')


def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise RuntimeError('Refuse to overwrite different source evidence: '+str(path))
        return
    with path.open('xb') as f:f.write(data)


def registration_paths():
    paths=set(OUT.glob('*registration*.json'))
    paths.update(OUT.glob('*_dispatch_go.json'))
    for folder in ('evaluation','binding','hinge','source_edges'):
        base=OUT/folder
        if base.exists():paths.update(base.rglob('*registration*.json'))
    paths.update((OUT/'hinge/registrations').glob('*.json'))
    return sorted(paths,key=str)


def extract_hashes(value,origin,pointer='$'):
    """Traverse registration metadata, accepting path→SHA and path/SHA records."""
    result=[]
    if isinstance(value,dict):
        for key,val in value.items():
            if isinstance(key,str) and key.lower().endswith('.py') and isinstance(val,str) and HEX.fullmatch(val):
                result.append((resolve(key),val.lower(),origin,pointer+'.'+key))
        for pathkey in ('path','source_path','source_file','code_path','file'):
            p=value.get(pathkey)
            if isinstance(p,str) and p.lower().endswith('.py'):
                for hashkey in ('sha256','source_sha256','code_sha256','hash'):
                    h=value.get(hashkey)
                    if isinstance(h,str) and HEX.fullmatch(h):result.append((resolve(p),h.lower(),origin,pointer+'.'+pathkey))
        for key,val in value.items():result.extend(extract_hashes(val,origin,pointer+'.'+str(key)))
    elif isinstance(value,list):
        for i,val in enumerate(value):result.extend(extract_hashes(val,origin,pointer+'['+str(i)+']'))
    return result


def python_bytes(path):
    if path.suffix.lower()!='.py':raise ValueError('Only .py files may enter code package')
    data=path.read_bytes()
    if b'\0' in data:raise ValueError('Binary/NUL data cannot enter source evidence '+str(path))
    data.decode('utf-8-sig',errors='strict')
    if len(data)>10*1024*1024:raise ValueError('Unexpected oversized Python source '+str(path))
    return data


def verify_completion():
    required={OUT/'hinge/completion.json':168,OUT/'evaluation/main_score_completion.json':112}
    result={}
    for path,n in required.items():
        value=read(path)
        if value.get('status')!='COMPLETE' or value.get('cells')!=n:raise RuntimeError('Final source snapshot requires completed registered matrix '+str(path))
        result[str(path)]=sha(path.read_bytes())
    return result


def added_only_diff(files):
    rel=NEW.relative_to(ROOT).as_posix()
    proc=subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'ls-files','--',rel],cwd=ROOT,capture_output=True,text=True,check=True)
    if proc.stdout.strip():raise RuntimeError('Pilot directory now contains tracked files; cannot describe it as added-only')
    parts=[]
    for path,data in files:
        name=path.relative_to(ROOT).as_posix();text=data.decode('utf-8-sig')
        # Preserve actual line contents; normal source LF is used in a unified diff.
        lines=text.splitlines();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        parts.append(f'diff --git a/{name} b/{name}\nnew file mode 100644\nindex 0000000..{blob[:7]}\n')
        if lines:
            parts.append(f'--- /dev/null\n+++ b/{name}\n@@ -0,0 +1,{len(lines)} @@\n')
            parts.extend('+'+line+'\n' for line in lines)
            if not text.endswith(('\n','\r')):parts.append('\\ No newline at end of file\n')
    return ''.join(parts).encode('utf-8')


def finalize():
    manifest_path=OUT/'source_snapshot.json'
    if manifest_path.exists():
        old=read(manifest_path)
        for path,h in old['output_hashes'].items():
            if sha(resolve(path).read_bytes())!=h:raise RuntimeError('Saved source evidence changed '+path)
        for row in old['files']:
            if sha(resolve(row['source_path']).read_bytes())!=row['sha256']:raise RuntimeError('Source changed after final snapshot '+row['source_path'])
        return dict(status='ALREADY_COMPLETE_VERIFIED',files=len(old['files']),manifest=str(manifest_path))
    completion_hashes=verify_completion();records=registration_paths()
    if not records:raise RuntimeError('No registration provenance found')
    known=collections.defaultdict(list);record_hashes={}
    for record in records:
        record_hashes[str(record)]=sha(record.read_bytes())
        for path,h,origin,pointer in extract_hashes(read(record),str(record)):
            known[path.resolve()].append(dict(expected_sha256=h,registration=origin,json_pointer=pointer))
    newfiles=sorted(NEW.rglob('*.py'),key=str)
    if not newfiles:raise RuntimeError('No new pilot source found')
    values={};failures=[]
    for path in sorted(set(known)|{p.resolve() for p in newfiles},key=str):
        data=python_bytes(path);actual=sha(data);expected={r['expected_sha256'] for r in known[path]}
        if expected and expected!={actual}:
            failures.append(dict(path=str(path),actual_sha256=actual,registered_sha256=sorted(expected),provenance=known[path]))
        values[path]=data
    if failures:
        # Explicit diagnostic only; do not copy today's bytes as an older registered source.
        raise RuntimeError('Registered source identity mismatch; preserve provenance and resolve before finalization: '+json.dumps(failures,ensure_ascii=False))
    if sum(map(len,values.values()))>50*1024*1024:raise RuntimeError('Source-only package unexpectedly exceeds50MiB')
    diff=added_only_diff([(p.resolve(),values[p.resolve()]) for p in newfiles])
    dest=OUT/'review_code'
    if dest.exists():raise RuntimeError('Partial source snapshot exists; no silent overwrite/restart')
    payload={};rows=[]
    for path,data in sorted(values.items(),key=lambda x:str(x[0])):
        is_new=path.is_relative_to(NEW.resolve())
        target=(dest/'new'/path.relative_to(ROOT.resolve())) if is_new else dest/'dependencies'/(sha(data)[:16]+'__'+path.name)
        if target in payload and payload[target]!=data:raise AssertionError('Source evidence destination collision')
        payload[target]=data
        rows.append(dict(source_path=str(path),snapshot_path=str(target),sha256=sha(data),bytes=len(data),
            kind='NEW_PILOT_CODE' if is_new else 'ACTUAL_REGISTERED_REUSED_PYTHON',
            registered=bool(known[path]),provenance=known[path],
            current_only_reason=None if known[path] else 'Final analysis/render/maintenance helper in the new pilot directory; not represented as frozen earlier training code'))
    # Everything is verified before the first output write.
    for path,data in payload.items():write_once(path,data)
    write_once(OUT/'code.diff',diff)
    output_hashes={str(p):sha(data) for p,data in payload.items()};output_hashes[str(OUT/'code.diff')]=sha(diff)
    for path,h in output_hashes.items():
        if sha(resolve(path).read_bytes())!=h:raise AssertionError('Copied source verification failed '+path)
    result=dict(status='COMPLETE',schema=VERSION,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=rows,new_python_files=sum(r['kind']=='NEW_PILOT_CODE' for r in rows),
        registered_reused_python_files=sum(r['kind']=='ACTUAL_REGISTERED_REUSED_PYTHON' for r in rows),
        total_source_bytes=sum(map(len,values.values())),registered_record_hashes=record_hashes,
        completion_hashes=completion_hashes,output_hashes=output_hashes,
        scope='Only .py text, copied byte-for-byte. No weights, meshes, training arrays or dataset files.',
        diff_scope='Added files under r2/pilots/seam_hinge_quality_v1 only. Git read-only ls-files guard; no git index/worktree mutation.',
        limits='Dependency inclusion follows actual registration source hash records, not speculative transitive imports. New current-only helpers are labelled separately.')
    write_once(manifest_path,jsonbytes(result))
    return dict(status='COMPLETE',files=len(rows),new_python_files=result['new_python_files'],registered_reused_python_files=result['registered_reused_python_files'],
        total_source_bytes=result['total_source_bytes'],manifest=str(manifest_path),diff=str(OUT/'code.diff'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');args=p.parse_args()
    print(json.dumps(finalize() if args.finalize else dict(status='PREPARED_ONLY_NOT_RUN',writes=0,required_action='Run --finalize only after all authorized outputs and source edits are complete'),ensure_ascii=False),flush=True)
