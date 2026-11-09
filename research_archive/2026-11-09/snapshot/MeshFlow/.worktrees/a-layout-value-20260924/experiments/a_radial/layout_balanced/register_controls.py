"""Register only the two predeclared unweighted240k controls, exactly once.

This command performs all read-only training audits before creating any controls.
Evaluation subsequently reads the copied files and the immutable registry only.
"""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import shutil
from ..io_utils import file_hash, write_json
from ..layout_student.package_results import inside
from ..layout_balanced.prepare import validate_control_contract
from ..layout_balanced.train import config_digest

CONTROL_MODELS = ('unweighted_joint_10', 'unweighted_joint_11')


def require(ok, message):
    if not ok: raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def control_plan(run, protocol):
    """Validate the frozen selection plan without opening unfinished checkpoints."""
    run = Path(run).resolve(); plan = protocol['unweighted_controls']; evaluation = protocol['evaluation']
    require(protocol['experiment'] == 'A-BALANCED-v1', 'Wrong balanced protocol')
    require(protocol['hashes']['config_sha256'] == config_digest(protocol), 'Balanced configuration changed')
    require(plan['registration_exclusive_once'] is True and plan['run_protocol_never_rewritten'] is True
            and plan['registry_required_before_control_evaluation'] is True, 'Control registration scope changed')
    require(tuple(plan['models']) == CONTROL_MODELS and evaluation['unweighted_control_models'] == list(CONTROL_MODELS),
            'Control model set/order changed')
    require(plan['registry_file'] == evaluation['control_registry'] == protocol['paths']['control_registry'] == 'CONTROL_REGISTRY.json',
            'Registry path changed')
    copied_protocol = inside(run, plan['copied_protocol_file'])
    require(plan['copied_protocol_file'] == protocol['paths']['unweighted_control_protocol'], 'Control protocol path differs')
    require(file_hash(copied_protocol) == plan['source_protocol_sha256'] == protocol['hashes']['unweighted_control_protocol_sha256'],
            'Copied control protocol changed')
    cp = read_json(copied_protocol)
    require(cp['hashes']['config_sha256'] == config_digest(cp), 'Control configuration changed')
    source_path = inside(run, protocol['paths']['source_protocol'])
    require(file_hash(source_path) == protocol['hashes']['source_protocol_sha256'], 'Original20k source protocol changed')
    source = read_json(source_path)
    validate_control_contract(cp, source,
        {n: j['initial_checkpoint_sha256'] for n, j in protocol['jobs'].items()},
        {n: j['initial_model_sha256'] for n, j in protocol['jobs'].items()})
    control_root = Path(plan['source_run']).resolve()
    require(Path(plan['source_protocol_file']).resolve() == control_root / 'RUN_PROTOCOL.json', 'Control source protocol path changed')
    for name in CONTROL_MODELS:
        binding = plan['models'][name]; seed = int(name.rsplit('_', 1)[1]); job = f'joint_{seed}'
        require(binding['job_id'] == job and binding['seed'] == seed and binding['update'] == 240000
                and binding['required_arm'] == 'c0' and binding['required_global_exposures_per_parent'] == 80000,
                'Control model identity/budget changed')
        require(binding['source_protocol_sha256'] == plan['source_protocol_sha256'], 'Control protocol binding differs')
        require(Path(binding['source_checkpoint']).resolve() == control_root / 'training' / job / 'final.pt',
                'Control final path was substituted')
        require(binding['local_checkpoint'] == f'controls/{name}_final.pt'
                and evaluation['model_checkpoints'][name] == binding['local_checkpoint'], 'Control copied path changed')
        require(evaluation['model_seeds'][name] == seed, 'Control evaluation seed changed')
        require(binding['initial_checkpoint_sha256'] == protocol['jobs'][job]['initial_checkpoint_sha256']
                and binding['initial_model_sha256'] == protocol['jobs'][job]['initial_model_sha256'],
                'Control and weighted model have different initial states')
    return cp


def load_registry(run, protocol):
    """Portable read-only validation; never reads the external live control run."""
    run = Path(run).resolve(); cp = control_plan(run, protocol)
    registry = read_json(inside(run, protocol['paths']['control_registry']))
    require(registry['schema'] == 'A-BALANCED-CONTROLS-v1' and registry['status'] == 'REGISTERED_ONCE', 'Invalid registry state')
    require(registry['protocol_sha256'] == file_hash(run / 'RUN_PROTOCOL.json'), 'Registry has another balanced protocol')
    require(registry['control_protocol_sha256'] == protocol['unweighted_controls']['source_protocol_sha256'],
            'Registry control protocol changed')
    require(set(registry['models']) == set(CONTROL_MODELS), 'Registry controls missing or substituted')
    require(registry['source_full_state_audit_passed'] is True and registry['source20k_full_import_verified'] is True,
            'Control registration lacks full-state audit')
    for name, entry in registry['models'].items():
        binding = protocol['unweighted_controls']['models'][name]
        require(entry['binding'] == binding, 'Registry substituted a predeclared control')
        rel = binding['local_checkpoint']; path = inside(run, rel)
        require(file_hash(path) == entry['checkpoint_sha256'] == registry['files'][rel], 'Registered control bytes changed')
    return registry, cp


def register_controls(run):
    run = Path(run).resolve(); protocol_path = run / 'RUN_PROTOCOL.json'
    protocol = read_json(protocol_path); ph = file_hash(protocol_path)
    require(not (run / 'CONTROL_REGISTRY.json').exists() and not (run / 'controls').exists(),
            'Control registration is exclusive once; existing/partial files are retained')
    cp = control_plan(run, protocol); plan = protocol['unweighted_controls']; control_root = Path(plan['source_run']).resolve()
    require(file_hash(control_root / 'RUN_PROTOCOL.json') == plan['source_protocol_sha256'], 'Live control protocol changed')
    source_manifest_path = control_root / 'DATA_MANIFEST.json'
    require(file_hash(source_manifest_path) == cp['hashes']['data_manifest_sha256'], 'Control data manifest changed')
    manifest = read_json(source_manifest_path)['files']
    # This independent read-only audit verifies all finals, the 20k imported byte
    # copies, full optimizer/RNG/sampler and every source trace/timing prefix.
    from ..layout_longfit.package_results import training_audit
    training, models, observations = training_audit(control_root, cp, manifest)
    records = {}; before_files = {}
    for name in CONTROL_MODELS:
        binding = plan['models'][name]; job = binding['job_id']; source = Path(binding['source_checkpoint'])
        digest = file_hash(source)
        require(models[job]['checkpoint_sha256'] == digest, 'Audited control checkpoint changed')
        initial = inside(run, protocol['jobs'][job]['initial_checkpoint'])
        require(file_hash(initial) == binding['initial_checkpoint_sha256'] == training[job]['initial_source_sha256'],
                'Audited control has another original20k full state')
        require(training[job]['full_resumable_state_verified'] and training[job]['parent_trace_preserved_exactly'],
                'Control full-state lineage not verified')
        before_files[name] = digest
        records[name] = {'binding': binding, 'checkpoint_sha256': digest, 'model_sha256': models[job]['model_sha256'],
            'control_training_config_sha256': file_hash(control_root / 'training' / job / 'TRAIN_CONFIG.json'),
            'control_import_record_sha256': file_hash(control_root / 'training' / job / 'IMPORT_RECORD.json'),
            'training_audit': training[job], 'source20k_prefix_rows': 20000,
            'fixed_observations': {checkpoint: identity for (n, checkpoint), identity in observations.items() if n == job}}
    require(file_hash(protocol_path) == ph and file_hash(control_root / 'RUN_PROTOCOL.json') == plan['source_protocol_sha256'],
            'Protocol changed during registration audit')
    # All scientific/input preconditions pass before the first new artifact.
    (run / 'controls').mkdir(exist_ok=False)
    files = {}
    for name in CONTROL_MODELS:
        binding = plan['models'][name]; destination = inside(run, binding['local_checkpoint'])
        with Path(binding['source_checkpoint']).open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
        require(file_hash(destination) == before_files[name] == file_hash(binding['source_checkpoint']),
                'Control changed during exclusive copy; partial artifacts retained')
        files[binding['local_checkpoint']] = before_files[name]
    require(file_hash(protocol_path) == ph, 'Balanced protocol changed during copy')
    value = {'schema': 'A-BALANCED-CONTROLS-v1', 'status': 'REGISTERED_ONCE',
        'registered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'protocol_sha256': ph, 'control_protocol_sha256': plan['source_protocol_sha256'],
        'source_full_state_audit_passed': True, 'source20k_full_import_verified': True,
        'source_training_protocol_sha256': protocol['hashes']['source_protocol_sha256'],
        'models': records, 'files': files, 'registration_source_sha256': file_hash(__file__),
        'source_audit_implementation_sha256': file_hash(Path(__file__).parents[1] / 'layout_longfit/package_results.py'),
        'run_protocol_modified': False, 'checkpoint_selection_or_substitution': False,
        'qualification_or_surface_metrics_used_for_selection': False}
    write_json(run / 'CONTROL_REGISTRY.json', value)
    require(file_hash(protocol_path) == ph, 'Registration rewrote the frozen balanced protocol')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', required=True)
    args = parser.parse_args(); value = register_controls(args.run)
    print(json.dumps({'status': value['status'], 'models': {n: {k: r[k] for k in ('checkpoint_sha256','model_sha256')} for n,r in value['models'].items()}}, indent=2))


if __name__ == '__main__': main()
