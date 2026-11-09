from pathlib import Path
p=Path('/mnt/e/MeshFlow/.worktrees/a-layout-value-20260924/experiments/a_radial/layout_student/package_results.py')
s=p.read_text()
anchor='\n\ndef audit_run(run, smoke_run):'
helper='''

def verify_teacher_arrays(source_mesh, reference_mesh, template_faces,
                          source_mesh_hash, logical_mesh_hash, stored_array_hash):
    """Compare immutable source/reference geometry before comparing typed hashes.

    Historical optimized teachers store int64 indices; the new reference stores
    the original int32 template. The two hashes intentionally retain their own
    dtype and are never treated as interchangeable.
    """
    sv,sf,su=[source_mesh[k] for k in ('vertices','faces','u')]
    v,f,u=[reference_mesh[k] for k in ('vertices','faces','u')]
    require(np.issubdtype(sf.dtype,np.integer) and np.issubdtype(f.dtype,np.integer),
            'Teacher connectivity must have integer dtype')
    require(array_hash(sv,sf)==source_mesh_hash,'Original teacher source array hash mismatch')
    require(np.array_equal(v,sv) and v.dtype==sv.dtype,'Teacher reference vertices changed')
    require(np.array_equal(u,su) and u.dtype==su.dtype,'Teacher reference directions changed')
    require(np.array_equal(f,sf),'Teacher reference connectivity values changed')
    require(np.array_equal(f,template_faces),'Teacher reference differs from original template')
    require(array_hash(v,template_faces)==logical_mesh_hash,'Teacher reference logical hash mismatch')
    require(array_hash(v,f)==stored_array_hash,'Teacher reference storage hash mismatch')
    return {'source_mesh_hash':source_mesh_hash,'reference_logical_mesh_hash':logical_mesh_hash,
        'reference_stored_array_hash':stored_array_hash,'source_faces_dtype':str(sf.dtype),
        'reference_faces_dtype':str(f.dtype),'source_vertices_dtype':str(sv.dtype),
        'reference_vertices_dtype':str(v.dtype),'vertices_exactly_equal':True,
        'directions_exactly_equal':True,'connectivity_values_exactly_equal':True,
        'face_dtype_differs':sf.dtype!=f.dtype,
        'source_and_reference_hashes_keep_original_dtype_semantics':True}
'''
assert anchor in s;s=s.replace(anchor,helper+anchor)
s=s.replace("support=RadialSupport();meshes={};fallback_count=0;unchanged_baseline_count=0", "support=RadialSupport();meshes={};fallback_count=0;unchanged_baseline_count=0;teacher_identity_records=[]")
old="""if row['model']=='teacher_reference':
                require(row['mesh_hash']==teachers[uid]['teacher_mesh_hash'],'Teacher reference changed')"""
new="""if row['model']=='teacher_reference':
                teacher=teachers[uid]
                source_result_path=old/'attempts'/uid/teacher['chosen_arm']/'result.json'
                source_row=read_json(source_result_path)
                source_mesh_path=inside(old,source_row['mesh_file'])
                source_mesh=mesh_arrays(source_mesh_path)
                require(source_row['mesh_hash']==teacher['teacher_mesh_hash'],'Teacher metadata disagrees with original source ledger')
                teacher_identity=verify_teacher_arrays(source_mesh,mesh,f,
                    teacher['teacher_mesh_hash'],row['mesh_hash'],stored_identity['stored_array_hash'])
                teacher_identity_records.append({'uid':uid,'chosen_arm':teacher['chosen_arm'],
                    'source_mesh_file':source_row['mesh_file'],'source_mesh_file_sha256':file_hash(source_mesh_path),
                    'source_result_file':source_result_path.relative_to(old).as_posix(),
                    'source_result_file_sha256':file_hash(source_result_path),
                    'reference_mesh_file':row['mesh_file'],
                    'reference_mesh_npz_sha256':stored_identity['mesh_npz_sha256'],**teacher_identity})"""
assert old in s;s=s.replace(old,new)
s=s.replace("'frozen_source_files_checked':len(protocol['fixed_source_hashes']),'upstream':identity,", "'frozen_source_files_checked':len(protocol['fixed_source_hashes']),\n        'teacher_reference_source_identity_checks':len(teacher_identity_records),\n        'teacher_source_vs_reference_face_dtype_differences':sum(r['face_dtype_differs'] for r in teacher_identity_records),\n        'teacher_source_reference_identity_records':teacher_identity_records,'upstream':identity,")
p.write_text(s)
