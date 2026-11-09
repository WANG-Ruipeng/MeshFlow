"""Canonicalize saved part proposals and apply the same full target gates."""
from .common import *
from .targets import save_target

def part27():
    from .part27_regions import run
    region=run();parent=next(p for p in freeze()['parents'] if p['parent_id'].startswith('27eb79'))
    a=load(parent['source']['clean_case_path']);mesh=load(OUT/'fallback27/proposal_mesh.npz')
    arrays={k:a[k].copy() for k in ('source_vertices','normalized_vertices','model_vertices','transform_json')}
    changed=np.any(mesh['vertices']!=a['model_vertices'],axis=1);mv=mesh['vertices'][changed]
    tr=json.loads(str(a['transform_json']));norm=mv*np.float32(.3762/2.)
    src=(norm.astype(float)*tr['divisor3']+np.asarray(tr['center3'])+np.asarray(tr['center2']))*tr['divisor1']+np.asarray(tr['center1'])
    arrays['model_vertices'][changed]=mv;arrays['normalized_vertices'][changed]=norm;arrays['source_vertices'][changed]=src.astype(np.float32)
    arrays['source_faces']=mesh['faces']
    save_target(parent,arrays,region['output_face_source_region_ids'],read(OUT/'fallback27/lineage.json'),'part_rebuild',False)

if __name__=='__main__':part27()
