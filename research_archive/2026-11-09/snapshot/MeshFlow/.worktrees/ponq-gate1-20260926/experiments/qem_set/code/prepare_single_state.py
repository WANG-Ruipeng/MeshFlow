"""One valid thin-U frame6 analytic q_only target; no model fitting here."""
import sys,json,shutil
from pathlib import Path
import numpy as np
from common import ROOT,dump,sha
from fm_state import encode_state
from single_state_common import arrays,geometry,target_reference

def main():
    previous=Path(sys.argv[1]);source=Path(sys.argv[2]);targetdir=ROOT/'target';targetdir.mkdir()
    original=arrays(source/'parameters.npz');state,encoding=encode_state(original)
    old=arrays(previous/'data.npz');mean,scale=old['mean'],old['scale']
    np.savez_compressed(ROOT/'data.npz',states=state[None],mean=mean,scale=scale,object_names=np.array(['thin_u_frame06__q_only']))
    shutil.copy2(source/'parameters.npz',targetdir/'original_parameters.npz')
    shutil.copy2(source/'mesh.obj',targetdir/'canonical_mesh.obj')
    shutil.copy2(source/'mesh.npz',targetdir/'canonical_mesh.npz')
    shutil.copy2(previous/'sample_noise.npz',ROOT/'sample_noise.npz')
    probe=np.random.default_rng(260929).standard_normal((32,256,22)).astype(np.float32)
    times=np.array([0.,.25,.5,.75,.9,.99],dtype=np.float32)
    np.savez_compressed(ROOT/'probe_noise.npz',noise=probe,times=times)
    oldnoise=arrays(ROOT/'sample_noise.npz')['noise'];assert not np.array_equal(oldnoise,probe)
    dump(ROOT/'target'/'ENCODING.json',encoding)
    np.savez_compressed(targetdir/'physical_state.npz',state=state)
    reference=target_reference();rows=[]
    for precision in ['float64','float32_chart']:
        physical=state if precision=='float64' else ((((state-mean)/scale).astype(np.float32)*scale.astype(np.float32))+mean.astype(np.float32)).astype(np.float64)
        for domain in ['fixed','enclosing']:
            row=geometry(physical,domain,ROOT/'oracle'/(precision+'_'+domain),reference,304207,
                dict(kind='target_precision_oracle',precision=precision,generated=False),np.arange(256))
            rows.append(row);print('ORACLE',precision,domain,row['status'],row['valid'],flush=True)
    result=dict(passed=all(r['valid'] for r in rows if r['domain']=='fixed'),rows=rows,target_parameters_sha256=sha(source/'parameters.npz'),
        target_mesh_sha256=sha(source/'mesh.obj'),chart_statistics_reused_from=str(previous/'data.npz'),
        sample_noise_sha256=sha(ROOT/'sample_noise.npz'),probe_noise_sha256=sha(ROOT/'probe_noise.npz'),
        sample_noise_reused_byte_identical=sha(ROOT/'sample_noise.npz')==sha(previous/'sample_noise.npz'),
        interpretation='One fixed audited analytic q_only snapshot, not the invalid freeze state. P means extractor v, not fit p.')
    dump(ROOT/'TARGET_CHECK.json',result)
    if not result['passed']:raise RuntimeError('Target encoding/domain oracle failed; no training')
if __name__=='__main__':main()