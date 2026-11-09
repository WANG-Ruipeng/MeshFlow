"""Read-only final hashes and actual-array audit; no model imports or scoring."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
REG = OUT / 'registration'
EXPECTED_HEAD = 'e1b6eef0bb2185ffaafd113abd11a26c8f62834b'


def main():
    started = time.monotonic()
    hashes, checks, missing = {}, [], []
    def path(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p
    def sha(p):
        p = path(p).resolve(); k = str(p)
        if k not in hashes:
            try:
                h = hashlib.sha256()
                with p.open('rb') as f:
                    for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
                hashes[k] = dict(sha256=h.hexdigest(), bytes=p.stat().st_size)
            except FileNotFoundError:
                hashes[k] = dict(sha256=None, bytes=0); missing.append(k)
        return hashes[k]['sha256']
    def read(p):
        sha(p)
        return json.loads(path(p).read_text())
    def check(name, ok, **details):
        checks.append(dict(check=name, status='PASS' if bool(ok) else 'BLOCKED', **details))
    def ah(a):
        a = np.ascontiguousarray(a)
        return hashlib.sha256(str((a.shape, str(a.dtype))).encode()+a.tobytes()).hexdigest()
    def same(a,b): return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

    source_rows = read(REG/'source_files.json')
    for row in source_rows:
        actual = sha(row['path'])
        check('registered_old_dependency_unchanged', actual == row['sha256'] and hashes[str(path(row['path']).resolve())]['bytes'] == row['bytes'], path=row['path'], expected_sha256=row['sha256'], actual_sha256=actual, bytes=row['bytes'], role=row['role'])
    generation = read(REG/'generation_sources_before.json')
    check('generation_dependency_count', len(generation) == 9)
    for p, expected in generation.items():
        check('generation_dependency_unchanged', sha(p) == expected, path=p, expected_sha256=expected, actual_sha256=sha(p))
    definitions = read(OUT/'scoring/definitions.json')
    score_sources = definitions['source_identity']
    for p, expected in score_sources.items():
        check('original_scoring_dependency_unchanged', sha(p) == expected, path=p, expected_sha256=expected, actual_sha256=sha(p))
    scoring_source = ROOT/'experiments/mf_h1_local0/chair_control_usability_quick_v1/scoring.py'
    check('executed_quick_scorer_unchanged', sha(scoring_source) == definitions['scoring_sha256'])
    head = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    check('git_HEAD_unchanged', head == EXPECTED_HEAD, expected=EXPECTED_HEAD, actual=head)
    requests = read(OUT/'requests.json'); protocol = read(OUT/'protocol.json')
    check('request_manifest_still_frozen', sha(OUT/'requests.json') == protocol['requests_sha256'])
    check('construction_protocol_still_frozen', sha(REG/'construction_protocol.json') == protocol['construction_protocol_sha256'])
    check('exact_registered_panel_counts', len(requests)==80 and len({r['request_id'] for r in requests})==80 and sum(r['panel']=='main' for r in requests)==64 and sum('multi' in r['panels'] for r in requests)==24)
    earlier = read(REG/'independent_inputs_audit.json')
    check('earlier_input_audit_valid', earlier['status']=='PASS' and all(sha(p)==expected for p,expected in earlier['sources'].items()))
    models = {r['metadata']['arm']:r for r in read(REG/'checkpoints.json')}
    actual_pairs = collections.defaultdict(list); statuses = collections.Counter()
    result_rows = []
    identity_keys = ('request_id','uid','condition_id','candidate','arm','context_level','N','K','input_sha256','condition_sha256','checkpoint_sha256','gaussian_sha256','C_array_sha256','known_mask_sha256')
    contract = dict(status='PASS',solver='original_Euler50',nfe=50,accepted_states=51,integration_dtype='float32',forward_dtype='bfloat16',known_all51_bitwise=True,known_state_checks=51,actual_backbone_calls=50,actual_Geo_recomputations=50,GT_model_inputs=False,geometry_repairs=0)
    for request in requests:
        rid = request['request_id']; rpath = OUT/'outputs'/rid/'result.json'
        try:
            result = read(rpath); statuses[result['status']] += 1
            check('result_registered_identity', all(result.get(k)==request.get(k) for k in identity_keys), request_id=rid)
            check('result_status_and_checkpoint', result['status'] in ('CACHED','SAMPLED') and result['checkpoint_sha256']==models[request['arm']]['checkpoint_sha256'], request_id=rid)
            check('raw_file_hash', sha(result['raw_path'])==result['raw_sha256'],request_id=rid)
            check('input_and_condition_files_unchanged',sha(request['input_npz'])==request['input_sha256'] and sha(request['condition_npz'])==request['condition_sha256'],request_id=rid)
            with np.load(request['input_npz'],allow_pickle=False) as z: bank={k:z[k].copy() for k in z.files}
            with np.load(result['raw_path'],allow_pickle=False) as z: raw={k:z[k].copy() for k in z.files}
            K,N=request['K'],request['N']; meta=result['metadata']
            check('actual_raw_noise_C_mask_equal_registered_bank',same(raw['noise'],bank['z']) and same(raw['C'],bank['C']) and same(raw['known_mask'],bank['known_mask']),request_id=rid)
            check('raw_finite_shape_known_bytes',raw['raw'].shape==(N,9) and raw['raw'].dtype==np.float32 and np.isfinite(raw['raw']).all() and same(raw['raw'][:K],bank['C'].reshape(K,9)),request_id=rid)
            check('actual_array_hashes_and_metadata',ah(raw['noise'])==request['gaussian_sha256']==meta['noise_sha256'] and ah(raw['C'])==request['C_array_sha256'] and ah(raw['C'].reshape(K,9))==meta['condition_sha256'] and ah(raw['known_mask'])==request['known_mask_sha256'],request_id=rid)
            check('recorded_sampler_contract',all(meta.get(k)==v for k,v in contract.items()),request_id=rid)
            if result['status']=='CACHED':
                check('cached_raw_exact_original_source',result['raw_path']==request['legacy_raw'] and result['raw_sha256']==request['legacy_raw_sha256'] and request['cache_status']=='PASS' and result['model_calls']==0,request_id=rid)
            else:
                check('sampled_raw_under_request_directory',Path(result['raw_path']).resolve().parent==rpath.parent and result['model_calls']==50 and ah(raw['raw'])==result['raw_array_sha256'],request_id=rid)
            score=read(OUT/'scoring/outputs'/f'{rid}.json'); si=score['identity']
            check('scored_exact_final_raw_and_frozen_sources',si['raw_sha256']==result['raw_sha256'] and si['condition_sha256']==request['condition_sha256'] and si['source_scorer_identity']==score_sources and si['scoring_sha256']==sha(scoring_source),request_id=rid)
            actual_pairs[(request['condition_id'],request['candidate'])].append(dict(arm=request['arm'],noise=raw['noise'],C=raw['C'],mask=raw['known_mask']))
            result_rows.append(dict(request_id=rid,result_sha256=sha(rpath),status=result['status'],raw_path=result['raw_path'],raw_sha256=result['raw_sha256'],actual_noise_sha256=ah(raw['noise']),actual_C_sha256=ah(raw['C'])))
        except Exception as exc:
            check('result_audit_exception',False,request_id=rid,error=repr(exc))
    check('completed_result_counts',statuses=={'CACHED':32,'SAMPLED':48})
    check('paired_bank_count',len(actual_pairs)==40)
    for (cid,candidate),pair in actual_pairs.items():
        check('two_arm_actual_noise_C_mask_byte_identity',len(pair)==2 and {x['arm'] for x in pair}=={'OT_BASE','OT_HYBRID'} and all(same(pair[0][k],pair[1][k]) for k in ('noise','C','mask')),condition_id=cid,candidate=candidate)
    failed=[c for c in checks if c['status']!='PASS']
    result=dict(status='PASS' if not failed and not missing else 'BLOCKED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Only registered dependencies, nine generation sources, eight original scoring dependencies, current pilot result/input/score files; no repository scan, metric recomputation, or model execution.',old_dependency_files=len(source_rows),old_dependency_bytes=sum(r['bytes'] for r in source_rows),generation_source_files=len(generation),original_scoring_source_files=len(score_sources),unique_hashed_files=len(hashes),unique_hashed_bytes=sum(r['bytes'] for r in hashes.values()),missing_count=len(missing),mismatch_or_failed_check_count=len(failed),missing=missing,failed_checks=failed,head=head,result_counts=dict(statuses),actual_paired_banks=len(actual_pairs),prior_input_audit_sha256=sha(REG/'independent_inputs_audit.json'),model_calls=0,optimizer_updates=0,backward_calls=0,checks=checks,results=result_rows,actual_hashes=hashes,seconds=time.monotonic()-started)
    dest=REG/'final_scoped_source_audit.json'
    if dest.exists(): raise RuntimeError('Refuse to replace existing final audit')
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','old_dependency_files','old_dependency_bytes','generation_source_files','original_scoring_source_files','unique_hashed_files','unique_hashed_bytes','missing_count','mismatch_or_failed_check_count','head','result_counts','actual_paired_banks','seconds')},indent=2))


if __name__=='__main__': main()
