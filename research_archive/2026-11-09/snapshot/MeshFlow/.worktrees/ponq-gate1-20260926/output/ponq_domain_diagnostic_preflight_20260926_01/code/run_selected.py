"""Run the frozen Gate 1 recipe on the independently selected real meshes."""
import argparse, copy, json
from pathlib import Path
import run_gate1 as gate
from common import load_mesh, sha
p=argparse.ArgumentParser();p.add_argument('selection',type=Path);args=p.parse_args()
selection=json.loads(args.selection.read_text())
gate.SPECS=selection['objects']
assert len(gate.SPECS)==4
gate.CONFIG=copy.deepcopy(gate.CONFIG)
gate.CONFIG.update(version='PONQ_GATE1_LOCAL_REAL_V1',selection_path=str(args.selection.resolve()),
    selection_sha256=sha(args.selection),known_limitations=['single seed','four selected real objects, not population coverage',
    'fitter not proven optimal','full4 Q noise differs from learned 3x3-factor errors'],
    state_groups={'clean':['clean128','clean256'],'mild':['position_light','normal_light','qem_light','joint_light']})
def create_selected_mesh(name):
    spec=gate.SPECS[name];path=Path(spec['path'])
    assert sha(path)==spec['reference_sha256'], 'Frozen reference changed'
    v,f=load_mesh(path)
    return v,f,spec.get('normalization',{'already_normalized':True})
gate.create_mesh=create_selected_mesh
gate.main()