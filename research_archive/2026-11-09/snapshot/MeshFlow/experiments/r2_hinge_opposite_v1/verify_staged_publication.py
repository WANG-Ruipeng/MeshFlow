"""Read-only check: exact publication whitelist and Git-blob byte preservation."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
git=['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT)]
names=subprocess.check_output(git+['diff','--cached','--name-only','-z']).decode('utf-8').split('\0')
names=[p for p in names if p]
assert names and '.gitattributes' in names
for name in names:
    assert name=='.gitattributes' or name.startswith(('r2/tools/hinge_opposite_v1/','r2/research/hinge_opposite_v1/')),name
    assert Path(name).suffix.lower() not in {'.pt','.pth','.ckpt','.safetensors','.npz','.npy'},name
stream=subprocess.check_output(git+['cat-file','--batch'],input=('\n'.join(':'+n for n in names)+'\n').encode())
offset=0;records=[]
for name in names:
    end=stream.index(b'\n',offset);head=stream[offset:end].split();assert head[1]==b'blob',(name,head)
    n=int(head[2]);data=stream[end+1:end+1+n];offset=end+1+n+1
    local=(ROOT/name).read_bytes()
    if name=='.gitattributes':assert data.replace(b'\r\n',b'\n')==local.replace(b'\r\n',b'\n')
    else:assert data==local,('GIT_BYTE_CHANGE',name)
    records.append(dict(path=name,bytes=n,sha256=hashlib.sha256(data).hexdigest()))
result=dict(status='PASS_EXPLICIT_WHITELIST_AND_EXACT_GIT_BLOBS',staged_files=len(records),bytes=sum(r['bytes'] for r in records),weight_files=0,unrelated_changes_staged=0,files=records)
(Path(__file__).parent/'staged_publication_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='files'}))
