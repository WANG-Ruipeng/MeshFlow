"""Runs ACTUAL pinned official Dataset/collate/OT, not a rewritten loader."""
from pathlib import Path
import argparse,json
import numpy as np
from .verify_repo import verify
from .io_utils import write_json,file_hash
from scipy.spatial import cKDTree

def compare_token_geometry(target,decoded,tolerance=3e-6):
    """Compare vertex multiplicities and unoriented triangle multisets after shuffling.
    Never rely on lexicographic floating-point sorting across float32/float64.
    Fail ambiguous nearest matching rather than invent vertex equivalence.
    """
    target=np.asarray(target).reshape(-1,3);decoded=np.asarray(decoded).reshape(-1,3)
    if target.shape!=decoded.shape or not np.isfinite(decoded).all():raise AssertionError('invalid tokens')
    unique,ti,counts=np.unique(target,axis=0,return_inverse=True,return_counts=True)
    distance,index=cKDTree(unique).query(decoded,k=2)
    if distance[:,0].max()>tolerance:raise AssertionError('inverse scaling/geometry mismatch')
    if np.any(distance[:,1]-distance[:,0]<1e-10):raise AssertionError('ambiguous nearest vertex mapping')
    pi=index[:,0]
    if not np.array_equal(np.bincount(pi,minlength=len(unique)),counts):raise AssertionError('vertex multiplicity changed')
    def faces(x):
        f=np.sort(x.reshape(-1,3),axis=1)
        return f[np.lexsort(f.T[::-1])]
    if not np.array_equal(faces(ti),faces(pi)):raise AssertionError('triangle surface multiset changed')
    return float(distance[:,0].max())

def audit_loader(repo,data):
    identity=verify(repo)
    from datasets.mesh_dataset import ObjaverseDataset,collate_fn
    ds=ObjaverseDataset(str(Path(data)/'meshflow512'),noise_sort='ot',training=True,do_dataset_normalize=True,
                       use_custom_prior=False,max_face_length=800,use_rot_aug=False,use_scale_aug=False,use_permut_aug=True)
    ds.data.sort(key=lambda z:str(np.asarray(z['uid']).item()))
    expect=json.loads((Path(data)/'DATASET.json').read_text());expected={r['uid'] for r in expect['rows']}
    if len(ds)!=16:raise AssertionError(('loader lost parents',len(ds)))
    rows=[]
    for i in range(len(ds)):
        item=ds[i];uid=str(np.asarray(item['uid']).item());batch=collate_fn([item],max_seq_length=800)
        if uid not in expected:raise AssertionError('unexpected uid')
        if tuple(batch['tokens'].shape)!=(1,512,9) or int(batch['num_faces'][0])!=512 or int(batch['masks'].sum())!=512:raise AssertionError('512 effective tokens/mask contract failed')
        if int(batch['num_faces'][0])//20!=25:raise AssertionError('face condition mismatch')
        with np.load(Path(data)/'meshflow512/objaverse_occ_v5_ids'/f'{uid}.npz',allow_pickle=False) as z:v=z['vertices'];f=z['faces']
        centre=(v.min(0)+v.max(0))/2;v=(v-centre)/np.ptp(v,axis=0).max();target=v[f].reshape(-1,3)
        decoded=batch['tokens'][0].numpy().reshape(-1,3)*(.3762*.5)
        delta=compare_token_geometry(target,decoded)
        rows.append({'uid':uid,'valid_faces':512,'condition_bin':25,'mask_true':512,'token_target_inverse_max_abs':delta})
    if {r['uid'] for r in rows}!=expected:raise AssertionError('parent coverage')
    return ds,{'repo':identity,'dataset_sha256':file_hash(Path(data)/'DATASET.json'),'actual_official_loader_executed':True,'records':rows,'closed_set_reference_not_holdout':True,'pass':True}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--data',required=True);p.add_argument('--report',required=True);a=p.parse_args();ds,r=audit_loader(a.repo,a.data);write_json(a.report,r);print('ALL16 official loader512 checks passed')
