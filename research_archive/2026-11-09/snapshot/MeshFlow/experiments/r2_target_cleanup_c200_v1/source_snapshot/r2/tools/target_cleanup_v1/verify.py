"""Independent read-only data-reader check, no model/noise/OT/scoring."""
from pathlib import Path
import json, sys
import numpy as np
from r2.tools.target_cleanup_v1.prepare import OUT, ROOT, read, arrays, same, sha, write, coordinate_key
sys.path.insert(0,str(ROOT/'r2/src'))
from meshflow_control.data.dataset import TrainingDataset

def main():
    m=read(OUT/'data_manifest.json');checks=[];loaded={}
    for arm,path in m['arm_manifests'].items():
        manifest=read(path);d=TrainingDataset.__new__(TrainingDataset)
        d.root=Path(path).parent;d.parents=manifest['parents'];d.tasks=manifest['tasks'];d.manifest=manifest
        assert len(d.parents)==len(d.tasks)==16
        assert set(manifest['pools'])=={'COMMON_AFFECTED_16'}
        for tid,t in d.tasks.items():
            c=arrays(t['npz']);p=d.parent(t['uid']);n,k=t['N'],t['K'];known=np.asarray(t['source_face_ids']);free=np.asarray(t['free_source_ids'])
            assert same(p['full_target'].reshape(n,3,3)[known],c['C'])
            assert np.array_equal(np.sort(np.r_[known,free]),np.arange(n))
            for alpha in (1.,.9,1.1):
                sample=d.task(tid,alpha);assert sample['N']==n and sample['K']==k
                assert same(sample['pre_ot_full']*np.float32(2),sample['full_target'].reshape(n,3,3))
                assert same(sample['model_vertices'][sample['source_face_vertex_ids']],sample['full_target'].reshape(n,3,3))
                checks.append(dict(arm=arm,task_id=tid,alpha=alpha,status='PASS'))
            loaded[(arm,tid)]=(p,c)
    for tid in m['main_task_ids']:
        op,oc=loaded['ORIGINAL',tid];cp,cc=loaded['DEDUP',tid]
        assert same(oc['C'],cc['C']) and same(oc['C_original'],cc['C_original'])
        assert same(op['model_vertices'],cp['model_vertices'])
        assert {coordinate_key(tri) for tri in op['full_target'].reshape(-1,3,3)}=={coordinate_key(tri) for tri in cp['full_target'].reshape(-1,3,3)}
    for path,h in {**m['protected_source_hashes'],**m['output_hashes']}.items():assert sha(path)==h,path
    result=dict(status='PASS',training_dataset_reader='Original TrainingDataset.parent/task with only constructor manifest size override',checks=len(checks),parents=16,arms=2,
        alphas=[1.,.9,1.1],C_original_and_canonical_bitwise_preserved=True,geometric_surface_union_exact=True,
        source_and_outputs_hashes_verified=len(m['protected_source_hashes'])+len(m['output_hashes']),no_noise_draws=True,no_OT=True,no_geometry_pair_detector=True,no_model_calls=True,
        source_code_sha256=sha(__file__),data_manifest_sha256=sha(OUT/'data_manifest.json'))
    write(OUT/'audits/training_reader_roundtrip.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
