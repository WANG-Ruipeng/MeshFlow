import json,numpy as np
from pathlib import Path
from experiments.a_radial.exporter import RadialSupport
r=Path('output/a_taskaligned_20260924_01')
rows=json.loads((r/'results.json').read_text())['rows']; support=RadialSupport()
errors=[]; maxerr=0.; baseline=0; old=Path('output/layout_value_20260924_01')
for x in rows:
    z=np.load(r/x['mesh_file'],allow_pickle=False)
    d=np.load(r/'data'/(x['uid']+'.npz'),allow_pickle=False)
    err=float(np.max(np.abs(z['vertices']-z['u']*support.query(d['logr'],z['u'])[:,None])))
    maxerr=max(maxerr,err)
    if err>1e-12:errors.append((x['uid'],x['model'],err))
    if x['policy']=='raw' and x['model'] in ('fixed','original_a'):
        arm='fixed' if x['model']=='fixed' else 'a'
        row=json.loads((old/'attempts'/x['uid']/arm/'result.json').read_text())
        oz=np.load(old/row['mesh_file'],allow_pickle=False)
        assert np.array_equal(oz['vertices'],z['vertices'])
        assert np.array_equal(oz['faces'],z['faces'])
        baseline+=1
print(json.dumps({'radial_requery_count':len(rows),'max_vertex_abs_diff':maxerr,'requery_failures':errors,'old_baseline_vertices_faces_value_equal':baseline}))
