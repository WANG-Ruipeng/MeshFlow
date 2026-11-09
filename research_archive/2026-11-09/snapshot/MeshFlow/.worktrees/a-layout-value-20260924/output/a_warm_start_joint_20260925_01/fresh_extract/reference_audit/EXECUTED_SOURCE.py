"""Read-only audit of reference geometry identity; no optimization or MC resampling."""
from pathlib import Path
import json, argparse
import numpy as np
from ..exporter import RadialSupport
from ..io_utils import array_hash, file_hash, new_dir, write_json

def run(source, original, out):
    source=Path(source).resolve(); original=Path(original).resolve(); out=new_dir(out)
    manifest=json.loads((source/'DATA_MANIFEST.json').read_text())['files']
    original_manifest=json.loads((original/'MANIFEST_SHA256.json').read_text())['files']
    support=RadialSupport(); rows=[]
    for path in sorted((source/'data').glob('*.npz')):
        rel=path.relative_to(source).as_posix(); before=file_hash(path)
        assert before==manifest[rel]
        with np.load(path,allow_pickle=False) as z:
            y=z['logr']; rv=z['ref_vertices']; rf=z['ref_faces']; uid=str(z['uid'])
        v,f=support.mesh(y)
        assert uid==path.stem and rv.shape==(2050,3) and rf.shape==(4096,3)
        assert np.array_equal(rv,v) and np.array_equal(rf,f)
        rep=original/'representation'/f'{uid}.json'; rep_rel=rep.relative_to(original).as_posix()
        assert file_hash(rep)==original_manifest[rep_rel]
        rep_row=json.loads(rep.read_text())
        rows.append({'uid':uid,'source_archive_sha256':before,'logr_hash':array_hash(y),
            'reference_hash':array_hash(rv,rf),'reconstructed_field_mesh_hash':array_hash(v,f),
            'vertices_bitwise_equal':True,'faces_bitwise_equal':True,
            'reference_vertices':len(rv),'reference_faces':len(rf),
            'separate_old_field_to_raw_asset_metric':rep_row['mainfield_to_original'],
            'separate_metric_sha256':file_hash(rep),'source_unchanged_after_read':file_hash(path)==before})
    assert len(rows)==16
    code_paths=['layout_student/prepare.py','layout_value/run.py','layout_value/optimizer.py','exporter.py']
    root=Path(__file__).resolve().parents[1]
    result={'status':'PASS','objects':16,'all_saved_references_exactly_equal_same_field_mesh':True,
      'semantics':'same 2050-node / 4096-face radial field surface, not raw asset surface',
      'optimizer_uses_same_reference_geometry_reconstructed_from_logr':True,
      'independent_MC_samples_do_not_mean_independent_reference_geometry':True,
      'no_metric_recomputed_no_model_update':True,'rows':rows,
      'source_manifest_sha256':file_hash(source/'DATA_MANIFEST.json'),
      'original_manifest_sha256':file_hash(original/'MANIFEST_SHA256.json'),
      'source_code_hashes':{p:file_hash(root/p) for p in code_paths},
      'audit_code_sha256':file_hash(__file__)}
    write_json(out/'RESULTS.json',result)
    (out/'EXECUTED_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    vals=np.array([r['separate_old_field_to_raw_asset_metric']['rms_bbox_percent'] for r in rows])
    print(json.dumps({'status':'PASS','exact_reference_matches':len(rows),
        'old_field_to_raw_RMS_bbox_percent_median':float(np.median(vals)),
        'old_field_to_raw_RMS_bbox_percent_min':float(vals.min()),
        'old_field_to_raw_RMS_bbox_percent_max':float(vals.max()),
        'result_sha256':file_hash(out/'RESULTS.json')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--original',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.source,a.original,a.out)