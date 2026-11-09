from pathlib import Path
import ast,hashlib,json
OLD=Path('/mnt/e/MeshFlow')
NEW=Path('/mnt/e/MeshFlow-Control')
DST=NEW/'src/meshflow_control/evaluation'
DST.mkdir(parents=True,exist_ok=True)
mapping=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(rel,names):
 p=OLD/rel;s=p.read_text(encoding='utf-8-sig');t=ast.parse(s); chunks=[]
 for node in t.body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:chunks.append(ast.get_source_segment(s,node))
 return '\n\n\n'.join(chunks)+'\n'
def put(name,text,orig,reason):
 p=DST/name;p.write_text(text,encoding='utf-8')
 mapping.append(dict(destination=str(p.relative_to(NEW)),source=orig,source_sha256=sha(OLD/orig),destination_sha256=sha(p),reason=reason))
base='experiments/mf_h1_local0/'
put('surface.py','"""Finite triangle distances and original area sampler; MeshFlow local provenance in source_map."""\nimport numpy as np\nimport trimesh\n\n'+extract(base+'evaluate.py',{'surface_distance_squared','sample_mesh_surface','raw_quality'}),base+'evaluate.py','Extract only CPU geometry functions, mathematical bodies unchanged.')
put('hashing.py','import hashlib\nimport numpy as np\n\n'+extract(base+'native_inpainting_metrics.py',{'array_sha'}),base+'native_inpainting_metrics.py','Extract original dtype/shape-aware array digest.')
s=(OLD/(base+'native_patch_seams.py')).read_text(encoding='utf-8')
t=ast.parse(s); names={n.name for n in t.body if isinstance(n,ast.FunctionDef) and n.lineno<385}
header=s[:s.index('def _triangles')]
put('seams.py',header+extract(base+'native_patch_seams.py',names),base+'native_patch_seams.py','Only finite geometry, boundary sampling, topology and primary classifier; no old run entrypoints.')
s=(OLD/(base+'guided_native_geometry_geometry.py')).read_text();t=ast.parse(s)
names={n.name for n in t.body if isinstance(n,ast.FunctionDef) and (n.name in ('_bump','_triangles') or 197<=n.lineno<=440)}
header='"""Independent CPU triangle verifier; original finite geometry unchanged."""\nimport itertools\nimport math\nimport numpy as np\nfrom .seams import classify_triangle_pair as _primary_classifier\n\ndef _numpy(value):\n    return np.asarray(value)\n\n'
header+=s[s.index('ILLEGAL_CLASSES ='):s.index('def initial_geometry_ledger')]
put('intersections.py',header+extract(base+'guided_native_geometry_geometry.py',names),base+'guided_native_geometry_geometry.py','Keep independent NumPy classifier; remove all torch/autograd potentials; _numpy accepts only CPU arrays.')
header='from collections import Counter\nimport numpy as np\nPROPER="proper_transverse_penetration"\nCOPLANAR="coplanar_positive_area_overlap"\nILLEGAL=(PROPER,COPLANAR)\n\n'
s=header+extract(base+'guided_native_geometry_audit.py',{'_tri','pair_scan'})
s=s.replace('from .guided_native_geometry_geometry import','from .intersections import')
put('pair_audit.py',s,base+'guided_native_geometry_audit.py','Complete pair scan only; local imports.')
s=(OLD/(base+'gpu_guidance_evaluation.py')).read_text()
header='import numpy as np\nQ_CUT=10.\nAREA_FLOOR=1e-10\nQ_FLOOR=1e-12\nNEAR_AREA=1e-12\nPROPER="proper_transverse_penetration"\nCOPLANAR="coplanar_positive_area_overlap"\nILLEGAL=(PROPER,COPLANAR)\n\n'
put('shape.py',header+extract(base+'gpu_guidance_evaluation.py',{'_tri','_distribution','shape_quality','_detector_view'}),base+'gpu_guidance_evaluation.py','Extract triangle quality and detector summaries only.')
edge='native_t1/runs/edge_factorial_20261001_123000/source_snapshot/experiments/mf_h1_local0/edge_factorial_pilot/edge_metric.py'
put('finite_edges.py',(OLD/edge).read_text(),edge,'Small standalone original finite-edge metric, physically copied; no dynamic loading.')
rel=base+'control_mvp48/eval_scoring.py'
s=(OLD/rel).read_text()
start=s.index('SURFACE_POINTS ='); end=s.index('def _tri')
header='"""Migrated original variable-N CPU scorer, with static local dependencies."""\nimport hashlib,json,time\nfrom pathlib import Path\nimport numpy as np\nfrom .surface import sample_mesh_surface,surface_distance_squared,raw_quality\nfrom .hashing import array_sha\nfrom .seams import extract_boundary_segments,_sample_boundary,point_to_triangle_surface_squared\nfrom .pair_audit import pair_scan,PROPER,COPLANAR\nfrom .shape import shape_quality,_detector_view\nfrom . import finite_edges\n\n'
s=header+s[start:end]+extract(rel,{'_tri','_inc','prepare_reference','_check_reference','_weighted_surface','score_free','score_output'})
s=s.replace('edge_module = _edge_module()','edge_module = finite_edges')
put('legacy_quality.py',s,rel,'Preserve scalar formulas and field names; replace old dynamic edge loader with static migrated module; omit model/asset preparation.')
rel=base+'chair_boundary_attachment_v1/control_metrics.py'
s=(OLD/rel).read_text()
s=s.replace('from experiments.mf_h1_local0.evaluate import','from .surface import').replace('from experiments.mf_h1_local0.native_inpainting_metrics import','from .hashing import').replace('from experiments.mf_h1_local0.native_patch_seams import','from .seams import').replace('from experiments.mf_h1_local0.guided_native_geometry_audit import','from .pair_audit import')
t=ast.parse(s);n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='source_identity')
lines=s.splitlines();lines[n.lineno-1:n.end_lineno]=['def source_identity():','    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}'];s='\n'.join(lines)+'\n'
put('control.py',s,rel,'All original v1r1 control math preserved. Only imports and provenance identity made repository-local.')
rel=base+'chair_control_usability_quick_v1/scoring.py'
header='import hashlib\nimport numpy as np\nfrom .surface import sample_mesh_surface,surface_distance_squared\nfrom .seams import nearest_segments\nfrom .shape import shape_quality\nTAUS=(.005,.01,.02,.05)\nATTACH_TAUS=(.001,.003,.005,.01,.02,.05)\n\n'
put('quality.py',header+extract(rel,{'seed','tri','area','tkey','pr_metrics','weighted_stats','gamma_points','interface_metrics','angle_reference'}),rel,'Extract deterministic sampling namespace and reference/finite-distance scalar formulas only.')
(NEW/'docs').mkdir(exist_ok=True)
(NEW/'docs/evaluation_source_map.json').write_text(json.dumps({'schema':'evaluation_source_map_v1','files':mapping},indent=2)+'\n')
print(json.dumps({'status':'PASS','migrated_modules':len(mapping),'lines':sum(len((DST/x['destination'].split('/')[-1]).read_text().splitlines()) for x in mapping)}))
