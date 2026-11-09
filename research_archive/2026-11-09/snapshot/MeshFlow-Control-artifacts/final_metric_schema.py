from pathlib import Path
root=Path('/mnt/e/MeshFlow-Control/src/meshflow_control/evaluation')
p=root/'aggregate.py';s=p.read_text()
s=s.replace("def scopes(r):",'''def delta_interval(a,b,key):
    upper={'SeamCov':'SeamCov_upper','ConflictCov_C':'ConflictCov_C_upper','ExposedRetain':'ExposedRetain_upper'}.get(key)
    if upper and all(number(v) for v in (a.get(key),a.get(upper),b.get(key),b.get(upper))):
        return float(a[key]-b[upper]),float(a[upper]-b[key])
    return None,None
def scopes(r):''')
s=s.replace("va,vb=a['metrics'].get(k),b['metrics'].get(k)","va,vb=a['metrics'].get(k),b['metrics'].get(k)\n                dlo,dhi=delta_interval(a['metrics'],b['metrics'],k)")
s=s.replace("candidate_value=va,reference_value=vb,delta=float(va-vb) if number(va) and number(vb) else None,","candidate_value=va,reference_value=vb,delta=float(va-vb) if number(va) and number(vb) else None,delta_lower=dlo,delta_upper=dhi,")
s=s.replace("delta=float(np.mean(d)) if d and all(number(v) for v in d) else None,noise_count=len(d)))","delta=float(np.mean(d)) if d and all(number(v) for v in d) else None,noise_count=len(d),\n                            delta_lower=float(np.mean([r['delta_lower'] for r in pr])) if pr and all(number(r['delta_lower']) for r in pr) else None,\n                            delta_upper=float(np.mean([r['delta_upper'] for r in pr])) if pr and all(number(r['delta_upper']) for r in pr) else None))")
p.write_text(s)
p=root/'registry.py';s=p.read_text()
s=s.replace("'PR2048':dict(taus=[.005,.01,.02,.05],comparison='<')}","'PR2048':dict(taus=[.005,.01,.02,.05],comparison='<'),\n                'pair_distance_epsilon':1e-9,'pair_primary_area_epsilon':1e-12,'pair_independent_area_floor':1e-11,'pair_coplanar_area_epsilon':1e-12,\n                'free_shape_near_area':1e-12,'small_angle_degrees':5,'q_actual_threshold':10,'finite_Gamma_coverage_tau':.001}")
p.write_text(s)
p=Path('/mnt/e/MeshFlow-Control/docs/metrics.md');s=p.read_text().replace('- Connectivity:', '- Duplicate excess: free faces are keyed by their three exact numeric coordinates, ignoring corner order/winding; this diagnostic is separate from near-duplicate I and does not remove any faces.\n- Connectivity:')
p.write_text(s)
