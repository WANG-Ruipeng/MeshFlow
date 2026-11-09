"""Immutable adapters to the executed v1r1 control and original quality code.

The only metric extension is tau=.001 in a separate globals dictionary attached
to the original SeamCov code object. No old module or old result is modified.
"""
from pathlib import Path
from types import FunctionType
import copy
import numpy as np
from .common import ROOT, read_json, resolve, digest, arrays, array_hash
from experiments.mf_h1_local0.chair_boundary_attachment_v1 import control_metrics as control
from experiments.mf_h1_local0.chair_control_usability_quick_v1 import scoring as quality

VERSION='bounded_attachment_metric_adapter_v1'
BASE_VERSION='chair_boundary_attachment_control_metrics_v1r1'
TAUS=(.001,.005,.01,.02)


def _clone(function,overrides):
    namespace=dict(function.__globals__);namespace.update(overrides)
    result=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=copy.copy(function.__kwdefaults__)
    return result


def source_identity():
    value=control.source_identity()
    for p in (Path(__file__),Path(quality.__file__),ROOT/'experiments/mf_h1_local0/native_inpainting_metrics.py'):
        value[str(p.relative_to(ROOT))]=digest(p)
    return value


def definition():
    assert control.VERSION==BASE_VERSION
    return dict(version=VERSION,base_control_version=BASE_VERSION,base_control=control.frozen_definition(),
        strict_tau=.001,retained_taus=[.005,.01,.02],primary_strict_name='SeamCov_strict',
        primary_old_name='SeamCov',theta_degrees=20.,formula='Unmodified original seam_metrics code object with a private TAUS tuple',
        frozen_reference='Read original v1r1 C points, directions, GT visibility, and original quality GT queries; never draw replacement GT queries',
        predicted_sampling='Original Q_F area sampling and original condition_id/candidate seed rule',
        units=quality.DEFINITIONS,source_identity=source_identity(),model_calls=0)


def checked_path(request,key):
    path=resolve(request[key]);expected=request.get(key.replace('_path','_sha256'))
    if expected is None:raise ValueError('Missing registered file hash: '+key)
    if digest(path)!=expected:raise ValueError('Registered scoring dependency changed: '+key)
    return path


def load_reference(request,condition):
    meta=read_json(checked_path(request,'legacy_control_reference_meta_path'))
    path=checked_path(request,'legacy_control_reference_npz_path')
    if digest(path)!=meta['npz_sha256']:raise ValueError('Old control reference NPZ identity differs')
    ref=copy.deepcopy(meta['reference']);ref['arrays']=arrays(path)
    if ref['identity']['version']!=BASE_VERSION or ref['identity']['density']!='main':raise ValueError('Wrong original control reference')
    if ref['identity']['condition_id']!=request['condition_id']:raise ValueError('Reference condition id differs')
    if ref['identity']['C_sha256']!=control.array_sha(condition['C']) or ref['identity']['GT_free_sha256']!=control.array_sha(condition['GT_free']):
        raise ValueError('Reference geometry differs')
    if meta['cache_identity']['source_identity']!=control.source_identity():raise ValueError('Original control scorer source changed')
    for key,value in ref['arrays'].items():
        if array_hash(value)!=meta['arrays_sha256'][key]:raise ValueError('Frozen control array differs: '+key)
    return ref


def frozen_query_bank(out,request,condition):
    path=checked_path(request,'legacy_quality_query_npz_path')
    meta=read_json(checked_path(request,'legacy_quality_query_meta_path'))
    if meta['npz_sha256']!=digest(path):raise ValueError('Old quality queries changed')
    identity=meta['identity']
    if identity['condition_id']!=request['condition_id'] or identity['GT_free_sha256']!=control.array_sha(condition['GT_free']):
        raise ValueError('Old GT query identity differs')
    bank=arrays(path)
    for key,value in bank.items():
        if control.array_sha(value)!=meta['array_hashes'][key]:raise ValueError('Old quality query array changed')
    if 'Q_Y2048' not in bank or len(bank['Q_Y2048'])!=2048:raise ValueError('Original 2048 GT queries required')
    return bank


def score_quality(out,request):
    """Call old quality scoring with only result-status and GT-bank adapters."""
    result_file=(Path(out)/'outputs'/request['request_id']/'result.json').resolve()
    original_read=quality.read_json
    def adapted_read(path):
        value=original_read(path)
        if Path(path).resolve()==result_file and value.get('status')=='PROJECTED':
            value=dict(value,status='PASS')
        return value
    run=_clone(quality.score_request,dict(query_bank=frozen_query_bank,read_json=adapted_read))
    qrequest=dict(request,sensitivity_check=False)
    if request['label']!='RAW':qrequest.pop('legacy_score_path',None)
    else:checked_path(request,'legacy_score_path')
    return run(out,qrequest,cf_ff=True)


def cf_from_quality(row):
    if not row.get('legacy_score_path'):return None
    score=read_json(resolve(row['legacy_score_path']));score=score.get('score',score)
    value=score.get('detail',{}).get('CF')
    return value if value and value.get('independent_verification') is True else None


def score_control(free,condition,reference,*,cf_scan=None):
    if control.VERSION!=BASE_VERSION:raise ValueError('Frozen control version changed')
    before=control.TAUS
    seam=_clone(control.seam_metrics,dict(TAUS=TAUS))
    score=_clone(control.score_control,dict(seam_metrics=seam))
    value=score(free,condition,reference,cf_scan=cf_scan)
    assert control.TAUS is before and control.TAUS==(.005,.01,.02)
    metrics=value['metrics'];metrics['SeamCov_strict']=metrics.get('SeamCov_0.001')
    metrics['SeamCov_strict_upper']=metrics.get('SeamCov_upper_0.001')
    if metrics['SeamCov_strict'] is None:status='NOT_APPLICABLE'
    else:status='PARTIAL' if metrics['SeamCov_strict_upper']>metrics['SeamCov_strict'] else 'PASS'
    metrics['SeamCov_strict_status']=status;value['statuses']['SeamCov_strict']=status
    value['audit']['adapter']=dict(version=VERSION,source_identity=source_identity(),old_globals_unchanged=True)
    return value


def formula_equivalence_check(free,condition,reference,*,cf_scan=None):
    """Reuse a registered projection fixture; this function constructs none."""
    cf=control.trusted_cf(condition['C'],free,float(condition['L0']),cf_scan)
    old,old_arrays=control.seam_metrics(free,condition,reference,cf)
    new,new_arrays=_clone(control.seam_metrics,dict(TAUS=TAUS))(free,condition,reference,cf)
    for key,value in old.items():
        if new[key]!=value:raise AssertionError('Original Seam metric changed: '+key)
    for key,value in old_arrays.items():
        if not np.array_equal(new_arrays[key],value,equal_nan=True):raise AssertionError('Original Seam vector changed: '+key)
    assert control.TAUS==(.005,.01,.02)
    return dict(status='PASS',original_fields=len(old),original_arrays=len(old_arrays),
        original_taus_unchanged=True,same_function_code=True,strict_tau=.001,
        newly_constructed_geometry_fixtures=0,model_calls=0)
