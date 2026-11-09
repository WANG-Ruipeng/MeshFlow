"""Scientific accounting gates: missing or mismatched outputs do not vanish."""
from pathlib import Path
import json
from .io_utils import file_hash

def validate_internal(manifests):
    if set(manifests)!=set(['fixed','a','qem']):raise ValueError('Exactly three internal manifests required')
    parsed={k:json.loads(Path(p).read_text()) if not isinstance(p,dict) else p for k,p in manifests.items()}
    maps={}
    for arm,m in parsed.items():
        rs=m['records'];ids=[r['sample_id'] for r in rs]
        if len(set(ids))!=len(ids):raise ValueError('duplicate sample IDs')
        if len(ids)!=m['requested_samples']:raise ValueError('silently dropped attempt')
        maps[arm]={r['sample_id']:r for r in rs}
    if not maps['fixed'].keys()==maps['a'].keys()==maps['qem'].keys():raise ValueError('sample IDs differ')
    for sid in maps['fixed']:
        for key in ['source_bundle_sha256','source_nodes_sha256','ae_sha256','fm_sha256','requested_faces']:
            vs=[maps[arm][sid].get(key) for arm in maps]
            if any(v is None for v in vs) and key!='source_nodes_sha256':raise ValueError('missing provenance:'+key)
            if not vs[0]==vs[1]==vs[2]:raise ValueError(f'non-shared {key}: {sid}')
    return {'same_samples':True,'same_checkpoints_and_mainfields':True,'attempts':len(maps['fixed']),
            'failures':{k:sum(r['status']!='ok' for r in v.values()) for k,v in maps.items()}}

def load_attempts(path):
    p=Path(path);m=json.loads(p.read_text());ids=[r['sample_id'] for r in m['records']]
    if len(ids)!=m['requested_samples'] or len(set(ids))!=len(ids):raise ValueError('incomplete/duplicate sample ledger')
    for r in m['records']:
        if r['status']=='ok' and file_hash(p.parent/r['mesh_path'])!=r['mesh_sha256']:raise ValueError('mutated output mesh')
    return m
