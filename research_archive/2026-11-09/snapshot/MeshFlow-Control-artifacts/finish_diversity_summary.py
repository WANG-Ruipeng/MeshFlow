from pathlib import Path
p=Path('/mnt/e/MeshFlow-Control/tools/summarize_results.py');s=p.read_text()
s=s.replace("table(out/'per_unit_metrics.csv',unit_rows);",'''    # Diversity contrasts use condition means, never individual pair counts as independent units.
    div_paired=[];div_contrasts=[];di={(r['label'],r['panel'],r['cohort'],r['condition_id']):r for r in div_units}
    for panel in LABELS:
        cids=sorted({r['condition_id'] for r in bypanel[panel]})
        for step in ((500,1000) if panel=='P1' else (1000,)):
            comparisons=[(f'B{step}',f'A{step}'),(f'C{step}',f'A{step}'),(f'C{step}',f'B{step}')]+[(f'{a}{step}','START') for a in 'ABC']
            for candidate,reference in comparisons:
                for cohort in COHORTS[panel]:
                    vals=[]
                    for cid in cids:
                        a=di.get((candidate,panel,cohort,cid),{}).get('D_free_mean')
                        b=di.get((reference,panel,cohort,cid),{}).get('D_free_mean')
                        delta=a-b if numeric(a) and numeric(b) else None;vals.append(delta)
                        div_paired.append(dict(contrast=candidate+'-'+reference,panel=panel,cohort=cohort,condition_id=cid,candidate_value=a,reference_value=b,delta=delta))
                    div_contrasts.append(dict(contrast=candidate+'-'+reference,panel=panel,cohort=cohort,metric='D_free_mean',
                        endpoint_role='primary1000' if step==1000 else 'secondary500',**stats(vals,len(cids)),
                        interpretation='Variation difference only; positive is not automatically better'))
    table(out/'diversity_paired.csv',div_paired);table(out/'diversity_contrasts.csv',div_contrasts)
    table(out/'per_unit_metrics.csv',unit_rows);'''.lstrip())
s=s.replace("expected_cells=sum(len(bypanel[p])*len(LABELS[p]) for p in LABELS)\n    result=dict(schema=VERSION,status='COMPLETE_COHORT' if len(index)==expected_cells and all(r.get('status')=='PASS' for r in observed) else 'PARTIAL_COHORT',","expected_keys={(label,r['request_id']) for panel in LABELS for label in LABELS[panel] for r in bypanel[panel]}\n    expected_cells=len(expected_keys)\n    result=dict(schema=VERSION,status='COMPLETE_COHORT' if set(index)==expected_keys and all(r.get('status')=='PASS' for r in observed) else 'PARTIAL_COHORT',")
s=s.replace("registered_requests=len(base),expected_cells=expected_cells,observed_cells=len(index),source_hashes=source_hashes,","registered_requests=len(base),expected_cells=expected_cells,observed_cells=len(index),missing_cells=len(expected_keys-set(index)),unexpected_cells=len(set(index)-expected_keys),source_hashes=source_hashes,")
s=s.replace("contracts=contracts,diversity=div_units,training=training,","contracts=contracts,diversity=div_units,diversity_contrasts=div_contrasts,training=training,")
p.write_text(s)
