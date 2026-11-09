"""Freeze exact retirement targets, archive source, and hash retained evidence."""
from pathlib import Path
import hashlib
import json
import zipfile

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RUN=ROOT/'native_t1/runs/chair_moment_channel_v2'
PACKAGES=['chair_moment_channel_v2','chair_surface_attention_v1','chair_surface_attention_v1r1']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')

def main():
    inventory=json.loads((OUT/'inventory.json').read_text(encoding='utf-8-sig'))
    targets=[]
    for group in inventory['groups']:
        for r in group['files']:
            p=Path(r['path']).resolve();assert p.is_relative_to(ROOT.resolve())
            assert p.is_file() and p.stat().st_size==r['bytes'] and not p.is_symlink()
            rel=p.relative_to(ROOT).as_posix()
            if group['group'].endswith('_active_package'):
                assert any(p.is_relative_to(ROOT/'experiments/mf_h1_local0'/n) for n in PACKAGES)
            elif group['group'].endswith('_checkpoint_weights'):
                assert p.parent==RUN/'checkpoints' and p.name.startswith(('H_CONT_moment','MOMENT_XY_moment')) and p.suffix=='.pt'
            else:assert group['group']=='V2_OT_mapping_NPZ' and p.is_relative_to(RUN/'ot_cache') and p.suffix=='.npz'
            digest=sha(p)
            if p.suffix=='.pt':
                side=json.loads(p.with_suffix('.json').read_text())
                assert digest==side['checkpoint_sha256']
            targets.append(dict(path=str(p),relative_path=rel,bytes=r['bytes'],sha256=digest,group=group['group']))
        print('TARGET_HASHED',group['group'],len(group['files']),flush=True)
    assert len(targets)==8061
    target_paths={r['relative_path'] for r in targets}
    code=[r for r in targets if r['group'].endswith('_active_package')]
    archive=OUT/'retired_activity_source.zip';assert not archive.exists()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for r in code:z.write(ROOT/r['relative_path'],r['relative_path'])
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.infolist())==51
        for r in code:assert hashlib.sha256(z.read(r['relative_path'])).hexdigest()==r['sha256']
    write('source_archive_manifest.json',{'status':'PASS','archive':archive.name,'sha256':sha(archive),'files':code})
    protected={}
    for name in PACKAGES:
        for p in sorted((ROOT/'native_t1/runs'/name).rglob('*')):
            if p.is_file() and p.relative_to(ROOT).as_posix() not in target_paths:
                protected[p.relative_to(ROOT).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
    for rel in ['native_t1/runs/chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt',
                'native_t1/checkpoints/chair_hybrid_start_total5000.pt',
                'native_t1/reviews/CHAIR_MOMENT_CHANNEL_V2_DECISION_MATERIALS.zip']:
        p=ROOT/rel;protected[rel]={'sha256':sha(p),'bytes':p.stat().st_size}
    assert protected['native_t1/checkpoints/chair_hybrid_start_total5000.pt']['sha256']=='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
    write('protected_evidence_before.json',protected)
    write('delete_plan.json',{'status':'READY','authorization':'User explicitly retired MOMENT, requested START rollback and removal of related code and weights.',
        'workspace':str(ROOT),'source_archive_sha256':sha(archive),'files':targets,
        'source_directories':[str((ROOT/'experiments/mf_h1_local0'/n).resolve()) for n in PACKAGES],
        'bytes':sum(r['bytes'] for r in targets),'protected_files':len(protected),'model_calls':0})
    print('RETIREMENT_READY',len(targets),'files;',len(protected),'protected;',sum(r['bytes'] for r in targets),'bytes',flush=True)

if __name__=='__main__':main()
