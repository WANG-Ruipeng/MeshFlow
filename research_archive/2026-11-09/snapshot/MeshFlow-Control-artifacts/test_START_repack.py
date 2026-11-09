from meshflow_control.evaluation.evaluate import *
out=resolve('/mnt/e/MeshFlow-Control-artifacts/tests/evaluation_repack')
r=load_manifest('/mnt/e/MeshFlow-Control-artifacts/data/evaluation/P1.json')[0]
h=r['historical_start'];old=arrays(h['raw_path']);p=out/'raw.npz'
save_npz(p,dict(raw=old['output'].copy(),trajectory=old['path'].copy(),C=old['C'].copy(),z=old['z'].copy(),free=old['output'][r['K']:].reshape(-1,3,3).copy()))
rp=out/'run.json';write_json(rp,dict(export_npz_ply_bitwise=True))
r.update(label='START',raw_path=str(p),raw_sha256=digest(p),result_path=str(rp))
result=score_request(r,out/'scored')
assert result['scientific_scores_recomputed']==0
assert result['historical_repack_audit']['current_file_sha256']!=result['historical_repack_audit']['historical_file_sha256']
assert all(result['historical_repack_audit']['arrays_bitwise_equal'].values())
write_json(out/'receipt.json',dict(status='PASS',array_audit=result['historical_repack_audit'],NPZ_C_bitwise=result['contracts']['NPZ_C_bitwise'],PLY_C_bitwise=result['contracts']['PLY_C_bitwise']))
print('repacked START cache: PASS, scoring recomputation=0')
