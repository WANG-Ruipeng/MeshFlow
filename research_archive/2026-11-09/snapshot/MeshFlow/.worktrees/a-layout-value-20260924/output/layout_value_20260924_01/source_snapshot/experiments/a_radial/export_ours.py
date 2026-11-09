"""Write all3 outlets with the same immutable geometry ID and complete failure ledger."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from .exporter import MeshExporter
from .legacy.exact_accel import audit
from .legacy.common import face_quality,save_obj
from .mesh_ops import topology
from .io_utils import new_dir,write_json,save_npz,file_hash,array_hash

def run(a):
    torch.set_num_threads(a.threads);src=Path(a.samples);manifest=json.loads((src/'SAMPLES.json').read_text());out=new_dir(a.out);ex=MeshExporter(a_seed=a.a_seed)
    ledgers={k:[] for k in ['fixed','a','qem']};bundlehash=file_hash(src/'SAMPLES.json')
    for rec in manifest['records']:
        sid=rec['sample_id']
        if rec['status']=='ok':
            with np.load(src/rec['path'],allow_pickle=False) as z:y=z['nodes'].copy()
            if array_hash(y)!=rec['nodes_hash']:raise ValueError('Mainfield changed')
        for arm in ledgers:
            row={'row':'ours_'+arm,'sample_id':sid,'source_bundle_sha256':bundlehash,'source_nodes_sha256':rec.get('nodes_hash'),'dataset_sha256':manifest['dataset_sha256'],'dataset_manifest_sha256':manifest['dataset_manifest_sha256'],'ae_sha256':manifest['ae_sha256'],'fm_sha256':manifest['fm_sha256'],'requested_faces':a.faces,'seconds_noise_to_nodes':rec.get('seconds_noise_to_nodes'),'a_weight_seed':a.a_seed if arm=='a' else None}
            try:
                if rec['status']!='ok':raise RuntimeError('upstream_failed:'+rec['error'])
                m,meta=ex.export(y,a.faces,arm);p=out/arm/f'{sid}.npz';save_npz(p,**m);save_obj(p.with_suffix('.obj'),m['vertices'],m['faces'])
                quality=face_quality(m['vertices'],m['faces']);top=topology(m['vertices'],m['faces']);exact=audit(m['vertices'],m['faces']) if a.exact else None
                row.update(meta);row.update(status='ok',mesh_path=str(p.relative_to(out)),mesh_sha256=file_hash(p),quality=quality,topology=top,exact=exact,seconds_noise_to_mesh=rec['seconds_noise_to_nodes']+meta['seconds'])
                # A valid file may still fail topology/actual budget; never delete or replace it.
            except Exception as e:row.update(status='failed',error=repr(e))
            ledgers[arm].append(row)
            write_json(out/f'{arm}_MANIFEST.json',{'row':'ours_'+arm,'requested_samples':manifest['requested'],'reference_scope':'closed_set16','records':ledgers[arm]},replace=(out/f'{arm}_MANIFEST.json').exists())
    write_json(out/'COMPLETE.json',{'same_latents':True,'records_per_row':{k:len(v) for k,v in ledgers.items()},'no_silent_retries':True,'exact_scope':'all output objects' if a.exact else 'not run','not_publication_readiness':True})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--samples',required=True);p.add_argument('--out',required=True);p.add_argument('--faces',type=int,default=512);p.add_argument('--a-seed',type=int,choices=[10,11],default=10);p.add_argument('--threads',type=int,default=1);p.add_argument('--exact',action='store_true');run(p.parse_args())
