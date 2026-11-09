"""Read-only cached geometry helpers; no model or scorer imports."""
from pathlib import Path
import csv, datetime, hashlib, io, json, os, re, subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
CODE=ROOT/'r2/tools/cref_layer_audit_v1'
OUT=ROOT/'experiments/r2_cref_layer_audit_v1'
BASE=ROOT/'experiments/r2_c_reference_frame_v1'
L=float(np.sqrt(3)*2/.3762)
VERSION='R2_CREF_LAYER_AUDIT_V1'
def resolve(path):
    s = str(path).replace('\\', '/')
    if os.name != 'nt' and re.match(r'^[A-Za-z]:/', s):
        s = '/mnt/' + s[0].lower() + s[2:]
    elif os.name == 'nt' and re.match(r'^/mnt/[a-z]/', s):
        s = s[5].upper() + ':/' + s[7:]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p

def checked(path):
    p = resolve(path).resolve()
    if not p.is_relative_to(OUT.resolve()):
        raise ValueError('Write outside this diagnostic OUT: ' + str(p))
    return p

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, Path): return str(value)
    return value

def read(path): return json.loads(resolve(path).read_text(encoding='utf-8-sig'))
def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}

def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''): h.update(part)
    return h.hexdigest()

def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()

def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

def _write_bytes(path, payload, replace=False):
    p = checked(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not replace:
        if p.read_bytes() != payload: raise ValueError('Frozen file differs: ' + str(p))
        return p
    temp = checked(p.with_name(p.name + '.tmp'))
    with temp.open('xb') as f: f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(temp, p)
    return p

def write(path, value, replace=False):
    value = clean(value)
    if checked(path).exists() and not replace:
        if read(path) != value: raise ValueError('Frozen JSON differs: ' + str(path))
        return checked(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    return _write_bytes(path, text.encode('utf-8'), replace=replace)

def frozen_npz(path, value=None, **kwargs):
    data = value if value is not None else kwargs
    if value is not None and kwargs: raise ValueError('Use either mapping or keywords')
    data = {k: np.asarray(v) for k, v in data.items()}
    p = checked(path)
    if p.exists():
        old = arrays(p)
        if set(old) != set(data) or any(not same(old[k], data[k]) for k in data):
            raise ValueError('Frozen NPZ differs: ' + str(p))
        return p
    buffer = io.BytesIO(); np.savez_compressed(buffer, **data)
    return _write_bytes(p, buffer.getvalue())

save_npz = frozen_npz

def csvwrite(path, rows, replace=False):
    rows = clean(list(rows)); keys = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return _write_bytes(path, stream.getvalue().encode('utf-8-sig'), replace=replace)

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def event(event_name, **fields):
    p = checked(OUT/'ledger.jsonl'); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(clean(dict(event=event_name, utc=utc(), **fields)), ensure_ascii=False, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())

def resources(path=None):
    def command(args):
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=20)
            return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except Exception as e: return dict(returncode=-1, stdout='', stderr=repr(e))
    value = dict(utc=utc(), cpu_logical=os.cpu_count(),
        gpu=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv']),
        loadavg=list(os.getloadavg()) if hasattr(os,'getloadavg') else None,
        memory=Path('/proc/meminfo').read_text() if Path('/proc/meminfo').is_file() else None)
    if path is not None: write(path, value)
    return value

def _unordered_equal(a,b):
    if a.shape != b.shape:return False
    return all(sorted(map(tuple,x))==sorted(map(tuple,y)) for x,y in zip(a,b))

def load_geometry(cell):
    for key in ('geometry','condition_source','query','vectors'):
        if cell.get(key+'_path') and cell.get(key+'_sha256'):
            assert sha(cell[key+'_path'])==cell[key+'_sha256'], (cell['cell_id'],key)
    q=arrays(cell['query_path'])
    src=arrays(cell['condition_source_path'])
    with np.load(resolve(cell['input_path']),allow_pickle=False) as z:C=z['C'].copy()
    Corig=src['C_original'].copy()
    assert _unordered_equal(C,Corig), 'source C face-row identity mismatch'
    if cell['stage']=='SOURCE':
        with np.load(resolve(cell['geometry_path']),allow_pickle=False) as z:
            v=z[cell['vertices_key']].copy(); fs=z[cell['faces_key']].copy()
        ids=np.r_[src['source_face_ids'],src['free_source_ids']].astype(np.int64)
        faces=fs[ids];vertices=v;F=vertices[faces[len(C):]]
        assert _unordered_equal(C,vertices[faces[:len(C)]]),'source geometry differs'
        graph='SOURCE_INDEXED_ORIGINAL_VERTEX_IDS'
    elif cell['stage']=='POST':
        with np.load(resolve(cell['geometry_path']),allow_pickle=False) as z:
            F=z['free'].copy();vertices=z['vertices'].copy();faces=z['faces'].copy(); output=z['output'].copy()
        assert np.array_equal(vertices[faces],output)
        assert np.array_equal(output[len(C):],F)
        assert _unordered_equal(C,output[:len(C)])
        graph='POST_SAVED_ACTUAL_VERTEX_INDICES'
    else:
        with np.load(resolve(cell['geometry_path']),allow_pickle=False) as z:F=z['free'].copy();storedC=z['C'].copy()
        assert np.array_equal(storedC,C)
        tri=np.concatenate([C,F]);vertices,inverse=np.unique(tri.reshape(-1,3),axis=0,return_inverse=True)
        faces=inverse.reshape(-1,3);graph='RAW_EXACT_COORDINATE_OBSERVATIONAL_INDEX'
    assert np.isfinite(C).all() and np.isfinite(F).all()
    assert len(q['C_points'])==2048 and np.isclose(q['C_weights'].sum(),1)
    mask=np.arange(len(faces))<len(C)
    old=arrays(cell['vectors_path']) if cell.get('vectors_path') else {}
    return dict(C=C,F=F,C_original=Corig,vertices=vertices,faces=faces,known_mask=mask,
        query=q,L=L,old_conflict_dup=old.get('conflict_dup'),old_conflict_cross=old.get('conflict_cross'),
        old_conflict_union=old.get('conflict_union'),graph_identity=graph)
