"""Audit provenance, snapshot exact source, and verify a portable result archive."""
import argparse,json,shutil,hashlib,zipfile,subprocess,datetime
from pathlib import Path
import numpy as np
from ..exporter import ASSETS,RadialSupport
from ..io_utils import file_hash,array_hash,write_json
from ..verify_repo import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    protocol=json.loads((run/'RUN_PROTOCOL.json').read_text());rows=json.loads((run/'results.json').read_text())['rows']
    expected={(u,arm) for u in protocol['uids'] for arm in protocol['arms']}
    found={(r['uid'],r['arm']) for r in rows}
    assert len(rows)==len(found)==96 and found==expected
    assert file_hash(ASSETS/'original_A_10.pt')==protocol['a']['weight_sha256']
    support=RadialSupport();mesh_count=0;query_checks=0
    for uid in protocol['uids']:
        record=next(x for x in protocol['records'] if x['uid']==uid)
        with np.load(run/'inputs'/f'{uid}.npz',allow_pickle=False) as z:y=z['logr'];rv=z['vertices'];rf=z['faces']
        assert array_hash(y)==record['mainfield_hash']
        assert array_hash(rv,rf)==record['ref_mesh_hash']
        fixedfaces=None
        for row in sorted((r for r in rows if r['uid']==uid),key=lambda r:r['arm']!='fixed'):
            assert row['mainfield_hash']==record['mainfield_hash']
            if row['status']!='success':continue
            with np.load(run/row['mesh_file'],allow_pickle=False) as z:
                v=z['vertices'];f=z['faces']
                assert array_hash(v,f)==row['mesh_hash']
                if row['arm']=='fixed':fixedfaces=f
                if row['arm'] in ['fixed','a','opt_fixed','opt_a']:
                    assert np.array_equal(f,fixedfaces)
                    u=z['u']
                    assert np.allclose(np.linalg.norm(u,axis=1),1,rtol=0,atol=1e-12)
                    assert np.allclose(v,u*support.query(y,u)[:,None],rtol=1e-12,atol=1e-12)
                    query_checks+=1
            mesh_count+=1
    identity=verify(Path.cwd())
    write_json(run/'FINAL_AUDIT.json',{'attempts':96,'unique_ids':True,'saved_mesh_hashes_checked':mesh_count,
        'same_connectivity_and_accepted_direction_requeries_checked':query_checks,'frozen_A_hash_unchanged':True,
        'upstream':identity,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    source=Path(__file__).resolve().parents[1]
    shutil.copytree(source,run/'source_snapshot'/'experiments'/'a_radial',
                    ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','build','*.pyc','*.nbi','*.nbc'))
    manifest={str(p.relative_to(run)):file_hash(p) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='MANIFEST_SHA256.json'}
    write_json(run/'MANIFEST_SHA256.json',{'files':manifest},replace=(run/'MANIFEST_SHA256.json').exists())
    archive=run.with_name(run.name+'_results.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(run.rglob('*')):
            if path.is_file():z.write(path,arcname=str(Path(run.name)/path.relative_to(run)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,sha in manifest.items():
            assert hashlib.sha256(z.read(str(Path(run.name)/rel))).hexdigest()==sha
    write_json(archive.with_suffix('.zip.json'),{'archive':str(archive),'bytes':archive.stat().st_size,
        'sha256':file_hash(archive),'member_hashes_verified':len(manifest),'crc_verified':True})
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'hashes_verified':len(manifest),
                      'saved_meshes':mesh_count,'radial_requeries':query_checks}))
if __name__=='__main__':main()

