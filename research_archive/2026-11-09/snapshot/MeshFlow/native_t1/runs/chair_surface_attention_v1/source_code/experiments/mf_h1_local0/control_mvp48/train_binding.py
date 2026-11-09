"""Session A readiness verification; no speculative schema or data selection."""
from pathlib import Path
import importlib.util
import json
from native_t1.artifacts import atomic_json, file_sha256
from .model_mvp import json_hash
from .train_contract import PROTOCOL, WaitingDependency, checked_file

def data_api(path):
    spec = importlib.util.spec_from_file_location('control_mvp48_frozen_data', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def bind_data(root):
    root = Path(root)
    ready_path = root / 'DATA_READY.json'
    if not ready_path.is_file():
        blocked=root/'DATA_BLOCKED.json'
        if blocked.is_file():
            reason=json.loads(blocked.read_text()).get('reason', 'See Session A report')
            raise WaitingDependency('BLOCKED_DATA: '+reason+' [DATA_BLOCKED sha256='+file_sha256(blocked)+']')
        raise WaitingDependency('Session A DATA_READY.json is absent')
    ready = json.loads(ready_path.read_text())
    if ready.get('schema') != 'control_mvp48_data_ready_v1' or ready.get('status') != 'READY':
        raise WaitingDependency('Session A has not published the expected READY contract')
    if ready['splits'] != dict(train=32, validation=8, test=8):
        raise ValueError('Frozen split counts changed')
    verified = {}
    for name, record in ready['files'].items():
        path = checked_file(root, dict(path=name, sha256=record['sha256']))
        if path.stat().st_size != record['bytes']:
            raise ValueError('Frozen artifact size mismatch: ' + name)
        verified[name] = record['sha256']
    for name in ('manifest', 'split', 'contract', 'independent_audit'):
        record = ready[name]
        if verified.get(record['path']) != record['sha256']:
            raise ValueError('Readiness references an unhashed/different artifact')
    audit = json.loads((root / ready['independent_audit']['path']).read_text())
    if audit['status'] != 'PASS':
        raise ValueError('Independent data audit did not pass')
    splits = json.loads((root / ready['split']['path']).read_text())
    sets = [set(splits[k]) for k in ('train', 'validation', 'test')]
    if [len(s) for s in sets] != [32, 8, 8] or any(sets[i] & sets[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('Split overlap/count error')
    for record in ready['task_plans'].values():
        if verified.get(record['path']) != record['sha256']:
            raise ValueError('Unbound task plan')
    loader_path = root / 'data/load_sample.py'
    if 'data/load_sample.py' not in verified:
        raise ValueError('Data loader is not frozen')
    binding = dict(schema='control_mvp48_train_binding_v1', status='FROZEN',
        data_ready_sha256=file_sha256(ready_path), accepted_contract_sha256=ready['contract']['sha256'],
        train_object_ids=sorted(sets[0]), evaluation_object_ids=sorted(sets[1] | sets[2]),
        files=verified, loader_path=str(loader_path), data_root=str(root / 'data'),
        manifest=ready['manifest'], plan=ready['task_plans']['train'])
    path = root / 'registration/train_data_binding.json'
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise ValueError('Existing train binding differs; refusing overwrite')
    else:
        atomic_json(path, binding)
    protocol = dict(PROTOCOL, data_ready_sha256=binding['data_ready_sha256'],
        binding_sha256=file_sha256(path), manifest_sha256=ready['manifest']['sha256'],
        plan_sha256=ready['task_plans']['train']['sha256'],
        data_contract_sha256=ready['contract']['sha256'],
        data_rng='Frozen per-sample Session A Gaussian/time/permutation seeds; no worker-derived seeds')
    return binding, protocol, root / ready['task_plans']['train']['path'], root / 'data'
