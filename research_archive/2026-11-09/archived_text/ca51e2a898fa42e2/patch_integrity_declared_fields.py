from pathlib import Path
p=Path('/mnt/e/MeshFlow/.worktrees/a-layout-value-20260924/experiments/a_radial/layout_student/integrity.py')
s=p.read_text()
s=s.replace("require(identity['mesh_hash']==row['mesh_hash'],'Original ledger mesh hash mismatch')", "require(identity['mesh_hash']==row['mesh_hash'],'Original ledger mesh hash mismatch')\n        for field,value in identity.items():\n            if field in row:\n                require(row[field]==value,'Declared row identity field mismatch: '+field)")
p.write_text(s)
p=Path('/mnt/e/MeshFlow/.worktrees/a-layout-value-20260924/experiments/a_radial/layout_student/test_integrity.py')
s=p.read_text(encoding='utf-8-sig').replace('mesh_identity,generate_integrity,validate_integrity','mesh_identity,generate_integrity,validate_integrity,collect_integrity')
s += '''

def test_future_declared_dual_hash_fields_are_checked(complete_run):
    run=complete_run;ledger=run/'results.json';data=json.loads(ledger.read_text());row=data['rows'][0]
    with np.load(run/'data'/(row['uid']+'.npz')) as d:
        identity=mesh_identity(run/row['mesh_file'],d['faces'])
    row.update(identity)
    attempt_path=run/'attempts'/row['uid']/row['model']/'result.json'
    attempt=json.loads(attempt_path.read_text());attempt['rows'][0].update(identity)
    write_json(attempt_path,attempt);write_json(ledger,data)
    assert collect_integrity(run)['rows_checked']==128
    row['stored_array_hash']='0'*64;attempt['rows'][0]['stored_array_hash']='0'*64
    write_json(attempt_path,attempt);write_json(ledger,data)
    with pytest.raises(ValueError,match='Declared row identity field mismatch'):
        collect_integrity(run)
'''
p.write_text(s,encoding='utf-8')
