"""Pre-register the frozen-field experiment and verify both supplied data copies."""
import argparse, json, subprocess, platform, hashlib, importlib.metadata, datetime
from pathlib import Path
import numpy as np
import torch
from ..io_utils import file_hash, array_hash, write_json, save_npz, new_dir, read_mesh
from ..exporter import ASSETS, RadialSupport

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--external-data',required=True);p.add_argument('--handoff',required=True)
    a=p.parse_args();data=Path(a.data);out=new_dir(a.out);external=Path(a.external_data);handoff=Path(a.handoff)
    with np.load(data/'radial_train.npz',allow_pickle=False) as z: y=z['logr'];uids=z['uids']
    order=np.argsort(uids);y=y[order];uids=uids[order]
    assert y.shape==(16,2050) and len(set(uids.tolist()))==16
    records=[];checks=[]
    for other in [external,handoff/'evidence/prepared512']:
        with np.load(other/'radial_train.npz',allow_pickle=False) as z:
            oi=np.argsort(z['uids']);oy=z['logr'][oi];ou=z['uids'][oi]
            assert np.array_equal(uids,ou)
            assert np.array_equal(y,oy)
        checks.append({'path':str(other),'radial_array_exact':True,'radial_file_hash':file_hash(other/'radial_train.npz')})
    support=RadialSupport()
    for uid,field in zip(uids,y):
        uid=str(uid);raw=data/'raw'/f'{uid}.npz'
        assert file_hash(raw)==file_hash(external/'raw'/f'{uid}.npz')
        ref_v,ref_f=support.mesh(field)
        save_npz(out/'inputs'/f'{uid}.npz',logr=field,vertices=ref_v,faces=ref_f,uid=np.array(uid))
        v,f=read_mesh(data/'meshflow512/objaverse_occ_v5_ids'/f'{uid}.npz')
        assert len(f)==512
        records.append({'uid':uid,'mainfield_hash':array_hash(field),'ref_mesh_hash':array_hash(ref_v,ref_f),
                        'raw_sha256':file_hash(raw),'source_faces512':len(f),'reference_faces':len(ref_f)})
    protocol={
      'experiment':'A-LAYOUT-VALUE-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'authorization':'User approved layout diagnostic after reviewing feedback; supersedes generation-first default.',
      'scope':'16 seen development loudspeakers, frozen real mainfields; no AE/FM training and no generation claim',
      'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'target_faces':512,'uids':uids.tolist(),'smoke_uids':uids[:4].tolist(),
      'selection':'All16 sorted UID; first4 implementation/timing pilot, then same locked protocol for remaining12',
      'arms':['fixed','a','opt_fixed','opt_a','vtk_qem','cgal_constrained'],
      'a':{'seed':10,'device':'cpu','proxy_points':256,'proxy_seed':4501,'steps':32,
           'weight_sha256':file_hash(ASSETS/'original_A_10.pt'),'original_guard_unmodified':True},
      'optimizer':{'max_steps':500,'max_seconds':300,'device':'cuda','training_points':2048,
        'seed_base':71310,'seed_per_object':'seed_base+sorted_uid_index, same across initializations',
        'snapshot_steps':[0,25,100,250,500],
        'variables':'unit directions only; fixed connectivity and shared piecewise radial query',
        'objective':'bbox-normalized area-weighted symmetric squared point-to-triangle distance',
        'output_selection':'best accepted feasible by optimization objective, never independent evaluation',
        'a_initialization_cost_included':True,'finite_budget_local_reference_not_global_bound':True},
      'quality':{'min_angle_deg_floor':'per-object Fixed minimum angle',
        'angle_q05_deg_floor':'per-object Fixed q05 of triangle minimum angles','tolerance_deg':1e-6,
        'endpoint':'closed oriented manifold, no self-intersection or degenerate triangles, exactly512 faces',
        'infeasible_initialization':'explicit restoration within same time/iteration budget; no silent Fixed fallback'},
      'cgal':{'version':'6.1.1 system package','policy':'GarlandHeckbert_plane_policies',
        'reason_policy_delta':'Available6.1.1 lacks plane-and-line; classic-plane named explicitly, not misrepresented.',
        'filters':'native normal/quality and prospective intersections, details captured in backend metadata',
        'unreachable_target':'retain output and actual face count, no repair or retry'},
      'evaluation':{'points_each_direction':8192,'seed_base':72101,'independent_optimization_samples':True,
        'normalizer':'reference F4096 bbox diagonal','metric':'sqrt(mean of two area-sampled squared point-to-triangle distances)/diag*100',
        'recheck_seed_base':83101,'audit':'shared independent endpoint exact-predicate audit',
        'target':'quality-qualified RMS <=0.9*Fixed RMS; exploratory target, not significance or acceptance criterion',
        'failed_attempts':'all retained; no successful-subset winner'},
      'timing':{'algorithm':'complete per-object preprocessing/query/guard/transfers/filter included',
        'audit':'independent final audit/evaluation reported separately',
        'cold_start':'model/support construction and environment startup reported separately',
        'single_run_optimizer':'no repeated long optimization for timing; all trajectory times retained'},
      'data_dir':str(data.resolve()),'records':records
    }
    write_json(out/'RUN_PROTOCOL.json',protocol)
    write_json(out/'DATA_AUDIT.json',{'provided_copies':checks,'records':records,'max_refinement_radial_delta':
        max(x['max_raw_refined_radial_diff_bbox'] for x in json.loads((data/'DATASET.json').read_text())['rows']),
        'old_unmounted_clean16_manifest_compared':False})
    packages={}
    for name in ['torch','numpy','scipy','trimesh','rtree','vtk','numba','pytest','matplotlib']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    write_json(out/'ENVIRONMENT.json',{'python':platform.python_version(),'platform':platform.platform(),
      'packages':packages,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),
      'capability':torch.cuda.get_device_capability(0),
      'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv,noheader'],text=True)})
    print(json.dumps({'run':str(out),'objects':len(uids),'smoke_uids':uids[:4].tolist(),'copies_arrays_exact':True}))
if __name__=='__main__':main()

