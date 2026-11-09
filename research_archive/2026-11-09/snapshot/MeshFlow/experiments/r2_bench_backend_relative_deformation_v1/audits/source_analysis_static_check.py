import ast,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/mnt/e/MeshFlow')
import numpy as np
from r2.pilots.bench_backend_relative_deformation_v1 import source_analysis as s
p=Path(s.__file__);ast.parse(p.read_text())
r=np.array([[[0,0,0],[1,0,0],[0,1,0]]],np.float32)
z=np.stack((r[:,1]-r[:,0],r[:,2]-r[:,0]),-1)*np.float32(s.SCALE)
c=dict(face_ids=np.array([0]),target_edges=z,inverse=np.eye(2,dtype=np.float32)[None]/np.float32(s.SCALE))
v,h=s.relative_values(r,c,.1)
assert v[0]==0 and h[0]==0 and 'torch' not in sys.modules
result=dict(status='PASS',source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),AST=True,numpy_identity_zero=True,torch_imported=False,model_calls=0,GEO_calls=0,
 command_only_attempt_note='Initial Powershell -c smoke lost quotes and stopped at Python syntax parsing. The script-path smoke then lacked repository sys.path and stopped before source import. Explicit local repository import root fixed only the smoke launcher; no model/GEO execution or cache mutation in these attempts.')
out=s.OUT/'audits/source_analysis_static_check.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
