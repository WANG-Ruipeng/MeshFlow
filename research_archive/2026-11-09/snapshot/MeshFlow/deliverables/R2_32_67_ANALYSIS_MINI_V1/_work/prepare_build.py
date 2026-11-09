from pathlib import Path
r=Path('/mnt/e/MeshFlow/deliverables/R2_32_67_ANALYSIS_MINI_V1')
p=r/'ANSWERS.md'
s=p.read_text('utf-8')
s=s.replace('68.75190856345242%','约68.7519%').replace('68.56880807034334%','约68.5688%')
p.write_text(s,encoding='utf-8')
p=r/'protocol_and_code/package_mini.py'
s=p.read_text('utf-8')
def replace(old,new):
    global s
    assert old in s,old
    s=s.replace(old,new,1)
replace("    # Merge provenance as declared", """    parent_rows=tables['parents.csv'];condition_rows=tables['conditions.csv']
    assert Counter(r['parent_role'] for r in parent_rows)==Counter(COMMON_SMALL32=32,EXP_ONLY35=35,NEW_DEV32=32,OLD6=6)
    assert len({(r['parent_uid'],r['condition_id']) for r in condition_rows})==264
    assert Counter(r['parent_role'] for r in condition_rows)==Counter(COMMON_SMALL32=91,EXP_ONLY35=103,NEW_DEV32=64,OLD6=6)
    train_groups={r['group_id'] for r in parent_rows if r['parent_role'] in {'COMMON_SMALL32','EXP_ONLY35'}}
    dev_groups={r['group_id'] for r in parent_rows if r['parent_role']=='NEW_DEV32'}
    assert not train_groups&dev_groups
    agg=rows(ROOT/'derived/aggregate_recomputed.csv')
    meddiff=max(abs(float(r['original_parent_median'])-float(r['recomputed_parent_median'])) for r in agg)
    assert meddiff<=1e-12
    checks.update(parent_roles_and_condition_keys_pass=True,train_dev_group_disjoint=True,
                  aggregate_median_comparisons=len(agg),maximum_absolute_median_difference=meddiff)

    # Merge provenance as declared""")
replace("    verification={p.name:load(p) for p in sorted((ROOT/'_work').glob('*verification*.json'))}", """    receipts=sorted(set((ROOT/'_work').glob('*verification*.json'))|set((ROOT/'_work').glob('*validation*.json')))
    verification={p.name:load(p) for p in receipts}""")
replace("    excluded={'_work','__pycache__'}", """    excluded={'_work','__pycache__'}
    duplicate_exclusions={'protocol_and_code/source_quality_rule_functions.txt':'Complete original functions included in original/source_quality_scan.py and original/source_data_freeze.py'}""")
replace("or p.name in {'MANIFEST.json','FILE_INDEX.csv'}:continue", "or p.name in {'MANIFEST.json','FILE_INDEX.csv'} or rel.as_posix() in duplicate_exclusions:continue")
replace("    for part in provenance.values():", "    for namespace,part in provenance.items():")
replace("        for name,record in output.items():metadata[name]=record", """        for name,record in output.items():metadata[name]={**record,'provenance_namespace':namespace}
    metadata['derived/training_quality_exposure.csv']=dict(DERIVED=True,
        provenance_namespace='exposure_quality_provenance',source_refs=['conditions.csv','source_quality_and_interfaces.csv','exposure_by_condition.csv'],
        projection='Join identities; sum observed integer counts, Decimal ratios; original source labels preserved')
    for name in ['parents.csv','conditions.csv','source_quality_and_interfaces.csv']:
        metadata[name]['DERIVED']='MIXED: inherited values plus explicitly named DERIVED columns and documented arithmetic fields'
        metadata[name]['source_refs']='provenance.json/data_provenance/sources; row source IDs resolve within that namespace'
        metadata[name]['projection']=provenance['data_provenance']['projections'][Path(name).stem]
    metadata['source_events.json']['DERIVED']='PROJECTION_ONLY: saved events, no new geometry'
    metadata['source_events.json']['source_refs']='provenance.json/data_provenance/sources; each event record retains origin'
    metadata['source_events.json']['projection']=provenance['data_provenance']['projections']['source_events']""")
replace("else 'CONSOLIDATED_METADATA',DERIVED=derived)", "else 'CONSOLIDATED_METADATA',DERIVED=derived,existing_source_hash='NA_NEW_MINI_FILE',source_path=rel)")
replace("exclusions=['_work','all weights/arrays/meshes/images','old complete ZIP'],", "exclusions=['_work','all weights/arrays/meshes/images','old complete ZIP'],duplicate_exclusions=duplicate_exclusions,")
replace("    receipt=dict(status='MINI_COMPLETE'", """    exposure_check=ROOT/'_work/exposure_from_unzip'
    result=subprocess.run([sys.executable,'-B',str(verify/'protocol_and_code/summarize_quality_exposure.py'),'--out',str(exposure_check)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert sha(exposure_check/'training_quality_exposure.csv')==sha(verify/'derived/training_quality_exposure.csv')
    receipt=dict(status='MINI_COMPLETE'""")
replace("portable_unzip_recompute_byte_identical=True,", "portable_unzip_recompute_byte_identical=True,portable_exposure_recompute_byte_identical=True,")
p.write_text(s,encoding='utf-8')
import ast
ast.parse(s)
print('Root documentation precision corrected; package metadata and portable checks updated. No old paths written.')
