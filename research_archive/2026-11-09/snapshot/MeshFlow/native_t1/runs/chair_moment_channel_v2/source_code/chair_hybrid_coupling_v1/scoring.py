"""CPU-only scoring of the frozen Chair panels; no model imports or calls.

The historical scorer is read-only. New column statistics and fixed-query
statistics have distinct names and never replace the historical metric fields.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from experiments.mf_h1_local0.control_mvp48.eval_scoring import score_free, source_identity
from experiments.mf_h1_local0.evaluate import surface_distance_squared
from experiments.mf_h1_local0.native_inpainting_metrics import array_sha

ALPHAS = (.9, 1., 1.1)
LABELS = ('START', 'OT_BASE', 'OT_HYBRID')
ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'native_t1/runs/chair_domain_control_v1'
DEFINITION = dict(
    fixed_query_MSE='mean((d(q,F)/L)^2), frozen coverage_points; dimensionless squared L-normalized distance',
    fixed_query_RMS_L='sqrt(fixed_query_MSE)',
    fixed_query_MSE_L0='mean((d(q,F)/L0)^2); separately named diagnostic',
    coverage_fraction='mean(d(q,F)<=0.01*L) on the same frozen coverage_points',
    original_surface_RMS='historical area-sampled prediction and separate frozen pairwise_target_points; %L',
    matrix='E[alpha,beta]=sqrt(mean(d(Q_beta,F_alpha)^2))/L0; row=input, column=target',
    column_advantage='mean(E[alpha!=beta,beta])-E[beta,beta]',
    width='equal triangle-corner X q95-q05, numpy quantile linear',
    r_base='W(F_1)/W(Y_1)',
    b_self='(W(F_1.1)-W(F_.9))/(0.2*W(F_1))',
    b_GT='(W(F_1.1)-W(F_.9))/(0.2*W(Y_1))',
    slope_error='abs(b-1) computed per curve BEFORE any averaging',
    small_width='width <= 1e-8*L0 is UNDEFINED; denominator retained',
    post_scale='fit dot(C1.x,Calpha.x)/dot(C1.x,C1.x) about original origin; only C and own F1',
    reference_contract='No-response/Post-scale use before_C+own_before_output+edited_C; not current-C-only inference',
    watertight='NOT_RUN', primary='A validation ratio=.2 alpha=1; two noises within each of 8 parents, then parents equal',
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: 'NA' if v is None else json.dumps(clean(v), ensure_ascii=False) if isinstance(v, (dict,list)) else v for k,v in clean(row).items()})


def arrays(path):
    with np.load(path, allow_pickle=False) as f: return {k:f[k].copy() for k in f.files}


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root)/path


def verified(root, rec, key, hash_key=None):
    path = resolve(root, rec[key])
    expected = rec.get(hash_key or key.replace('_path','_sha256'))
    if expected and sha(path) != expected: raise ValueError('Frozen source changed: '+str(path))
    return path


def bank_reference(source, record):
    bank = arrays(verified(source, record, 'input_path', 'input_sha256'))
    if 'gt_free' not in bank:
        bank.update(arrays(resolve(source, record.get('target_path', record['ref_npz']))))
    ref = arrays(verified(source, record, 'ref_npz', 'ref_npz_sha256'))
    prefix = record.get('reference_key_prefix')
    if prefix: ref = {k[len(prefix):]:v for k,v in ref.items() if k.startswith(prefix)}
    meta = read_json(verified(source, record, 'ref_json', 'ref_json_sha256'))
    for key, expected in meta['array_sha256'].items():
        if array_sha(ref[key]) != expected: raise ValueError('Reference array identity: '+key)
    if meta['known_sha256'] != array_sha(bank['C'].reshape(-1,3,3)): raise ValueError('Reference C identity')
    if meta['GT_free_sha256'] != array_sha(bank['gt_free'].reshape(-1,3,3)): raise ValueError('Reference GT identity')
    return bank, dict(arrays=ref, audit=meta)


def extent(triangles):
    xyz = np.asarray(triangles, np.float64).reshape(-1,3)
    if not len(xyz) or not np.isfinite(xyz).all(): raise ValueError('Invalid width geometry')
    return np.quantile(xyz,.95,axis=0)-np.quantile(xyz,.05,axis=0), np.median(xyz,axis=0)


def width_metrics(free, gt, L0):
    ps, pm = extent(free); gs, gm = extent(gt)
    result = {}
    for j, axis in enumerate('XYZ'):
        result.update({axis+'_span':float(ps[j]), axis+'_GT_span':float(gs[j]),
            axis+'_span_abs_error_L0':float(abs(ps[j]-gs[j])/L0),
            axis+'_median':float(pm[j]), axis+'_GT_median':float(gm[j]),
            axis+'_median_error_L0':float((pm[j]-gm[j])/L0),
            axis+'_median_abs_error_L0':float(abs(pm[j]-gm[j])/L0)})
    return result


def column_audit(matrix):
    M = np.asarray(matrix, np.float64)
    if M.shape != (3,3) or not np.isfinite(M).all(): raise ValueError('Invalid 3x3 matrix')
    columns = []
    for j,beta in enumerate(ALPHAS):
        other = np.delete(M[:,j],j)
        winners = [ALPHAS[i] for i in range(3) if M[i,j] == M[:,j].min()]
        columns.append(dict(target_beta=beta, diagonal_RMS_L0=float(M[j,j]),
            A_column=float(other.mean()-M[j,j]), column_strict_win=bool(M[j,j]<other.min()),
            column_no_worse=bool(M[j,j]<=other.min()), best_input_alphas=winners,
            A_row_diagnostic=float(np.delete(M[j,:],j).mean()-M[j,j])))
    same = [a for i,a in enumerate(ALPHAS) if all(M[i,j]<np.delete(M[:,j],i).min() for j in range(3))]
    return columns, dict(all_three_strict_win=all(r['column_strict_win'] for r in columns),
        one_input_strictly_wins_all_targets=bool(same), universal_winner_alpha=same[0] if same else None,
        mean_A_column=float(np.mean([r['A_column'] for r in columns])))


def curve_metrics(widths, gt1, L0):
    p, g = float(widths[1.]), float(gt1)
    d = float(widths[1.1]-widths[.9])
    valid_p, valid_g = p>1e-8*L0, g>1e-8*L0
    bself, bgt = d/(.2*p) if valid_p else None, d/(.2*g) if valid_g else None
    return dict(W_09=float(widths[.9]), W_10=p, W_11=float(widths[1.1]), W_GT_10=g,
        r_base=p/g if valid_g else None, baseline_abs_ratio_error=abs(p/g-1) if valid_g else None,
        b_self=bself, b_GT=bgt, b_self_abs_error=abs(bself-1) if bself is not None else None,
        b_GT_abs_error=abs(bgt-1) if bgt is not None else None,
        response_status='PASS' if valid_p and valid_g else 'UNDEFINED_SMALL_WIDTH',
        monotonic=bool(widths[.9]<=p<=widths[1.1]), strict_monotonic=bool(widths[.9]<p<widths[1.1]),
        self_denominator=.2*p, GT_denominator=.2*g)


def fixed_metrics(score, bank, reference):
    L = float(bank['L']); L0 = float(bank.get('L0',L))
    detail = score['detail']['fixed_GT_coverage']
    q = reference['arrays']['coverage_points']
    if detail['points_sha256'] != array_sha(q): raise ValueError('Scored query identity changed')
    d = np.asarray(detail['per_point_distances'],np.float64)
    if d.shape != (len(q),) or not np.isfinite(d).all(): raise ValueError('Invalid fixed distance vector')
    result = dict(fixed_query_MSE=float(np.mean((d/L)**2)), fixed_query_RMS_L=float(np.sqrt(np.mean((d/L)**2))),
        fixed_query_MSE_L0=float(np.mean((d/L0)**2)), fixed_query_p95_L=float(np.quantile(d/L,.95)),
        fixed_query_max_L=float(np.max(d/L)), coverage_fraction=float(np.mean(d<=.01*L)),
        query_sha256=array_sha(q), Q=len(q), L=L,L0=L0,
        pairwise_query_sha256=array_sha(reference['arrays']['pairwise_target_points']))
    if not np.isclose(result['fixed_query_RMS_L']*100,score['metrics']['fixed_R12_coverage_RMS_bbox_pct'],atol=1e-12,rtol=1e-12):
        raise ValueError('Original coverage RMS mismatch')
    return result, d/L


def _event(out, value):
    path = Path(out)/'ledger'/'cpu_scoring.jsonl'; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(clean(dict(time_unix=time.time(),pid=os.getpid(),**value)),ensure_ascii=False,allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())


def _source_score(source, label, case_id, raw_hash, raw_array):
    if label != 'START': return None
    ledger_path = Path(source)/'evaluation/phase3000/Geo/ledger.json'
    if not ledger_path.exists(): return None
    old = next((r for r in read_json(ledger_path)['rows'] if r['case_id']==case_id),None)
    if not old or old.get('score_status')!='PASS': return None
    if old.get('raw_sha256')!=raw_hash:
        # Replay NPZ can carry extra velocities/states; compare actual FP32 raw.
        oldraw=resolve(source,old['raw_path'])
        if sha(oldraw)!=old['raw_sha256']: raise ValueError('Historical raw hash changed')
        if array_sha(arrays(oldraw)['raw'])!=array_sha(raw_array): return None
    path = resolve(source,old['score_path'])
    if sha(path)!=old['score_sha256']: raise ValueError('Historical score hash changed')
    return path


def score_case(source, label, record, generated, folder, out):
    base = {k:record.get(k) for k in ('case_id','table','uid','split','ratio','alpha','noise_index','task_id','patch_id')}
    base.update(checkpoint_label=label, input_sha256=record.get('input_sha256'))
    if record.get('status') in ('NOT_APPLICABLE','NOT_AVAILABLE'):
        return dict(base,status='NOT_APPLICABLE',reason=record.get('reason')), None
    if not generated or generated.get('status') not in ('SAMPLED','CACHED','PASS'):
        return dict(base,status='FAILED' if generated and generated.get('status') in ('FAILED','FAIL') else 'BLOCKED',
            reason=generated.get('reason','Raw output not available') if generated else 'Raw output not available'), None
    rawpath = resolve(out, generated['raw_path']); rawhash=sha(rawpath)
    if rawhash!=generated['raw_sha256']: raise ValueError('Raw file identity mismatch')
    bank, ref = bank_reference(source, record)
    rawfile=arrays(rawpath); raw=rawfile['raw']; N=int(bank['N']); K=len(bank['C'])
    mask=np.asarray(bank['known_mask'],bool).reshape(-1)
    if mask.shape!=(N,) or mask.sum()!=K or not np.array_equal(mask,np.arange(N)<K): raise ValueError('Frozen mask/slot contract changed')
    finite=bool(np.isfinite(raw).all()); sizeok=raw.shape==(N,9)
    keep=sizeok and np.array_equal(raw[mask].reshape(-1,3,3),bank['C'])
    base.update(raw_path=str(rawpath),raw_sha256=rawhash,N_preserved=sizeok,C_preserved=bool(keep),finite=finite,
        N=N,K=K,noise_sha256=array_sha(bank['z']),mask_sha256=array_sha(mask),GT_sha256=array_sha(bank['gt_free']))
    if not sizeok or not keep or not finite: return dict(base,status='INVALID_OUTPUT',reason='N/C/nonfinite contract violation'), None
    if 'noise' in rawfile and not np.array_equal(rawfile['noise'],bank['z']): raise ValueError('Raw noise differs from frozen bank')
    if 'C' in rawfile and not np.array_equal(rawfile['C'],bank['C']): raise ValueError('Raw condition differs from frozen bank')
    free=raw[~mask].reshape(-1,3,3)
    dst=folder/'scores'/(record['case_id']+'.json')
    identity=dict(raw_sha256=rawhash,input_sha256=record['input_sha256'],ref_sha256=sha(resolve(source,record['ref_npz'])),
        score_sources=source_identity())
    cached=False; original_reused=False
    if dst.exists():
        wrapper=read_json(dst)
        if wrapper['identity']!=identity: raise ValueError('CPU score cache identity changed')
        score=wrapper['score']; cached=True
    else:
        started=time.perf_counter(); _event(out,dict(kind='score_attempt',checkpoint_label=label,case_id=record['case_id']))
        historical=_source_score(source,label,record['case_id'],rawhash,raw)
        if historical:
            score=read_json(historical); original_reused=True
            if score['detail']['reference']!=ref['audit']: raise ValueError('Historical score reference identity')
            if score['detail']['predicted_free_sha256']!=array_sha(free): raise ValueError('Historical prediction geometry identity')
        else:
            score=score_free(free,bank['C'],bank['gt_free'],float(bank['L']),reference=ref,include_pair_audit=True)
        write_json(dst,dict(identity=identity,score=score,original_score_reused=original_reused,
            original_score_path=str(historical) if historical else None))
        _event(out,dict(kind='score_return',checkpoint_label=label,case_id=record['case_id'],historical_cache=original_reused,
            elapsed_seconds=time.perf_counter()-started,status=score['status']))
    base.update(score_path=str(dst),score_sha256=sha(dst),status=score['status'],reason=score.get('failure_reason'),
        score_cache_hit=cached,original_score_reused=original_reused if not cached else read_json(dst)['original_score_reused'],
        watertight='NOT_RUN',**score['metrics'])
    base.update(C_preserved=bool(keep),N_preserved=sizeok,finite=finite)
    if score['status']=='PASS':
        fm,dist=fixed_metrics(score,bank,ref); base.update(fm,**width_metrics(free,bank['gt_free'],fm['L0']))
        vpath=folder/'fixed_vectors'/(record['case_id']+'.npz');vpath.parent.mkdir(parents=True,exist_ok=True)
        if not vpath.exists(): np.savez_compressed(vpath,distances_L=dist,query_sha256=np.array(fm['query_sha256']))
        base.update(fixed_vector_path=str(vpath),fixed_vector_sha256=sha(vpath))
    return base, dict(bank=bank,ref=ref,free=free,rawfile=rawfile,record=record,row=base)


def score_width_group(label, state):
    center=state[1.]; rec=center['record'];L0=float(center['bank']['L0'])
    base=dict(checkpoint_label=label,uid=rec['uid'],split=rec['split'],noise_index=rec['noise_index'],task_id=rec['task_id'],L0=L0)
    for alpha,item in state.items():
        for key in ('z','known_mask'):
            if not np.array_equal(item['bank'][key],center['bank'][key]): raise ValueError('Width across-alpha '+key+' mismatch')
        if item['record']['task_id']!=rec['task_id']: raise ValueError('Width task mismatch')
    queries={a:arrays(verified(item['source'],item['record'],'control_points_path','control_points_sha256'))['points'] for a,item in state.items()}
    # Saved queries must be transported original points, not fresh variant-area samples.
    for a in ALPHAS:
        expected=np.asarray(queries[1.]).copy();expected[:,0]*=a
        if not np.allclose(queries[a],expected,atol=1e-6,rtol=0): raise ValueError('Control query transport mismatch')
    C1=np.asarray(center['bank']['C'],np.float64);x=C1[...,0].reshape(-1);den=float(x@x)
    variants={};fit={}
    for name in ('MODEL','No-response','Post-scale'):
        variants[name]={}
        for a,item in state.items():
            if name=='MODEL':free=item['free']
            else:
                free=np.asarray(center['free'],np.float64).copy()
                if name=='Post-scale':
                    if den<=1e-12*L0**2*len(x): continue
                    edit=np.asarray(item['bank']['C'],np.float64)[...,0].reshape(-1)
                    fit[a]=float(x@edit/den);free[...,0]*=fit[a]
            variants[name][a]=free
    output=[];curves=[];matrices=[];columns=[]
    for name,freemap in variants.items():
        b=dict(base,candidate=name,reference_input_contract='current_C_only' if name=='MODEL' else DEFINITION['reference_contract'])
        if len(freemap)!=3:
            curves.append(dict(b,status='UNDEFINED_ALPHA_FIT',alpha_fit_denominator=den));continue
        M=np.empty((3,3));cov=np.empty((3,3));widths={a:extent(f)[0][0] for a,f in freemap.items()}
        for i,a in enumerate(ALPHAS):
            for j,beta in enumerate(ALPHAS):
                d2=surface_distance_squared(np.asarray(queries[beta],np.float64),np.asarray(freemap[a],np.float64))
                M[i,j]=np.sqrt(np.mean(d2))/L0;cov[i,j]=np.mean(d2<=(.01*L0)**2)
                matrices.append(dict(b,alpha=a,target_beta=beta,E_RMS_L0=float(M[i,j]),
                    coverage_1pct_L0_fraction=float(cov[i,j]),query_sha256=array_sha(queries[beta]),Q=len(d2),status='PASS'))
        audits,summary=column_audit(M)
        columns += [dict(b,**r,status='PASS') for r in audits]
        curve=curve_metrics(widths,extent(center['bank']['gt_free'])[0][0],L0)
        curve.update(b,status='PASS',**summary)
        for a,item in state.items():
            wm=width_metrics(freemap[a],item['bank']['gt_free'],L0)
            i=ALPHAS.index(a)
            output.append(dict(b,case_id=item['record']['case_id'],alpha=a,status='PASS',**wm,
                E_diagonal_RMS_L0=float(M[i,i]),alpha_hat=fit.get(a) if name=='Post-scale' else None,
                signed_width_error_L0=(widths[a]-wm['X_GT_span'])/L0,
                baseline_width_component_L0=a*(widths[1.]-curve['W_GT_10'])/L0,
                relative_response_component_L0=(widths[a]-a*widths[1.])/L0))
        curve['width_absolute_error_L0_mean']=float(np.mean([r['X_span_abs_error_L0'] for r in output if r.get('candidate')==name]))
        curves.append(curve)
        if name=='No-response' and (any(r['A_column']!=0 or r['column_strict_win'] for r in audits) or (curve['b_self'] is not None and curve['b_self']!=0)):
            raise AssertionError('No-response positive control failed')
    return output,curves,matrices,columns


def cpu_tests(out):
    # Geometric fixtures only: no GT resampling, RNG, model or CUDA.
    triangles=np.array([[[1.,0,0],[1.,1,0],[1.,0,1]],[[-1.,0,0],[-1.,1,0],[-1.,0,1]]])
    gt={a:triangles*np.array([a,1.,1.]) for a in ALPHAS}
    q={a:g.mean(axis=1) for a,g in gt.items()}
    perfect=np.array([[np.sqrt(surface_distance_squared(q[b],gt[a]).mean())/3 for b in ALPHAS] for a in ALPHAS])
    noresp=np.array([[np.sqrt(surface_distance_squared(q[b],gt[1.]).mean())/3 for b in ALPHAS] for a in ALPHAS])
    pc,ps=column_audit(perfect); nc,ns=column_audit(noresp)
    assert np.max(np.diag(perfect))<1e-12 and ps['all_three_strict_win']
    assert all(r['A_column']==0 and not r['column_strict_win'] for r in nc)
    identical=np.tile([1.,2.,3.],(3,1));col,_=column_audit(identical)
    assert [r['A_column'] for r in col]==[0,0,0]
    assert [r['A_row_diagnostic'] for r in col]==[1.5,0,-1.5]
    widths={a:extent(gt[a])[0][0] for a in ALPHAS};curve=curve_metrics(widths,widths[1.],3.)
    assert abs(curve['b_self']-1)<1e-12 and abs(curve['b_GT']-1)<1e-12
    assert curve_metrics({a:0 for a in ALPHAS},0,3.)['response_status']=='UNDEFINED_SMALL_WIDTH'
    assert np.mean(np.abs(np.array([.2,1.8])-1))>.79 and abs(np.mean([.2,1.8])-1)==0
    result=dict(status='PASS',perfect_matrix=perfect,no_response_matrix=noresp,column_positive_controls=pc,
        no_response_column_controls=nc,identical_rows=col,curve=curve,model_forwards=0,optimizer_updates=0,
        GT_resampling_calls=0,geometry_calls=18)
    write_json(Path(out)/'tests/scoring_cpu_tests.json',result);return clean(result)


def run_checkpoint(out, checkpoint_label, records, generated_rows, source_out=OLD):
    if checkpoint_label not in LABELS: raise ValueError('Unregistered checkpoint label')
    started=time.perf_counter();out=Path(out);source=Path(source_out)
    folder=out/'evaluation'/checkpoint_label/'scoring';folder.mkdir(parents=True,exist_ok=True)
    old_protocol=read_json(source/'protocol.json')
    source_checks={name:sha(ROOT/name)==expected for name,expected in old_protocol['source_scorer_identity'].items()}
    if not all(source_checks.values()): raise ValueError('Historical frozen scoring source changed')
    if len({r['case_id'] for r in records})!=len(records): raise ValueError('Duplicate panel identifier')
    if len({r['case_id'] for r in generated_rows})!=len(generated_rows): raise ValueError('Duplicate generated identifier')
    generated={r['case_id']:r for r in generated_rows};rows=[];width_state=defaultdict(dict)
    for record in records:
        try:
            row,state=score_case(source,checkpoint_label,record,generated.get(record['case_id']),folder,out)
        except (FileNotFoundError,ValueError,AssertionError) as exc:
            _event(out,dict(kind='score_failure',checkpoint_label=checkpoint_label,case_id=record['case_id'],reason=repr(exc)))
            raise
        rows.append(row)
        if state and row['status']=='PASS' and record['table']=='B':
            state['source']=source;width_state[(record['uid'],record['noise_index'])][float(record['alpha'])]=state
        write_json(folder/'per_output.json',rows)
    outputs=[];curves=[];matrices=[];columns=[]
    expected=defaultdict(list)
    for rec in records:
        if rec['table']=='B':expected[(rec['uid'],rec['noise_index'])].append(rec)
    for key,recs in sorted(expected.items()):
        if set(width_state[key])!=set(ALPHAS):
            curves.append(dict(checkpoint_label=checkpoint_label,uid=key[0],noise_index=key[1],split=recs[0]['split'],candidate='MODEL',status='BLOCKED',reason='Incomplete registered alpha triplet'))
            continue
        cache_path=folder/'width_groups'/(key[0]+'_n'+str(key[1])+'.json')
        identity={str(alpha):dict(raw_array_sha256=array_sha(v['rawfile']['raw']),input_sha256=v['record']['input_sha256'],
            query_sha256=v['record']['control_points_sha256']) for alpha,v in width_state[key].items()}
        if cache_path.exists():
            saved=read_json(cache_path)
            if saved['identity']!=identity: raise ValueError('Width analysis cache identity changed')
            o,c,m,a=(saved[n] for n in ('outputs','curves','matrices','columns'))
        else:
            tick=time.perf_counter();_event(out,dict(kind='width_group_attempt',checkpoint_label=checkpoint_label,uid=key[0],noise_index=key[1]))
            o,c,m,a=score_width_group(checkpoint_label,width_state[key])
            write_json(cache_path,dict(identity=identity,outputs=o,curves=c,matrices=m,columns=a))
            _event(out,dict(kind='width_group_return',checkpoint_label=checkpoint_label,uid=key[0],noise_index=key[1],
                elapsed_seconds=time.perf_counter()-tick,surface_distance_calls=len(m),model_forwards=0))
        outputs+=o;curves+=c;matrices+=m;columns+=a
    tables={'per_output':rows,'width_per_output':outputs,'width_per_curve':curves,'cross_matrix':matrices,'target_column_audit':columns}
    for name,values in tables.items():write_json(folder/(name+'.json'),values);write_csv(folder/(name+'.csv'),values)
    summary=dict(checkpoint_label=checkpoint_label,status='COMPLETE' if all(r['status'] in ('PASS','NOT_APPLICABLE') for r in rows) else 'INCOMPLETE',
        requested=len(records),availability=dict(Counter(r['status'] for r in rows)),rows={k:len(v) for k,v in tables.items()},
        elapsed_seconds=time.perf_counter()-started,definition=DEFINITION,source_scoring_sha256=source_identity(),
        historical_scoring_source_checks=source_checks,
        model_forwards=0,model_backwards=0,optimizer_updates=0,GT_query_resampling_calls=0)
    write_json(folder/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--label',choices=LABELS)
    p.add_argument('--source',default=str(OLD));p.add_argument('--records');p.add_argument('--generated');p.add_argument('--cpu-tests',action='store_true')
    args=p.parse_args()
    if args.cpu_tests: print(json.dumps(cpu_tests(args.out)));return
    panels=read_json(args.records or Path(args.source)/'panels.json')
    records=panels if isinstance(panels,list) else [dict(r,table=t) for t in ('A','B','C') for r in panels[t]]
    payload=read_json(args.generated);generated=payload if isinstance(payload,list) else payload['rows']
    print(json.dumps(run_checkpoint(args.out,args.label,records,generated,args.source)))


if __name__=='__main__':main()
