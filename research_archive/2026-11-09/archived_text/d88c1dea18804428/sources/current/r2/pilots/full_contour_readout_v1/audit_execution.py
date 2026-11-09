"""Independent, bounded CPU audit of saved CONTOUR execution evidence.

Default progress mode never imports torch or opens a large checkpoint. Final
mode CPU-loads the final full checkpoint exactly once; no model is constructed,
and no training, OT, sampling, scoring or image interpretation is performed.
Run only after the controller authorizes the selected audit stage.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
import argparse, collections, csv, datetime, hashlib, io, json, time
from pathlib import Path
from . import common as C

VERSION = 'full_contour_saved_execution_audit_v1'
MAIN = ('CONTOUR_S0', 'CONTOUR_S1000', 'CONTOUR_S3000', 'CONTOUR_S6000')
TRAINED = MAIN[1:]
REFERENCES = ('FULL_S0000', 'FULL_S1000', 'FULL_S3000', 'FULL_S6000')
LIMITS = dict(train_forward=48000, train_backward=48000, optimizer_update=6000)
TRAIN = C.OUT / 'training/CONTOUR'
EV = C.OUT / 'evaluation'
REVIEW_FIELDS = ('C_adoption', 'C_floating', 'C_replaced_by_other_surface', 'C_wrong_layer',
                 'whole_body_complete', 'support_missing', 'fragments_or_folded_layers', 'visible_join')


class Audit:
    def __init__(self, stage):
        self.stage = stage; self.final = stage == 'final'; self.errors = []; self.missing = []
        self.checks = 0; self.files = {}; self.parts = {}; self.weight_loads = 0
        self.started = time.perf_counter(); self.hash_cache = {}

    def check(self, value, code, **context):
        self.checks += 1
        if not bool(value): self.errors.append(dict(code=code, **context))
        return bool(value)

    def need(self, path):
        p = C.resolve(path)
        if not p.is_file(): self.missing.append(dict(reason='MISSING_FILE', path=str(p))); return None
        return p

    def digest(self, path):
        p = C.resolve(path); key = str(p)
        if key not in self.hash_cache: self.hash_cache[key] = C.sha(p)
        self.files[key] = self.hash_cache[key]
        return self.hash_cache[key]

    def verify(self, path, expected):
        p = self.need(path)
        if p is not None: return self.check(self.digest(p) == expected, 'FILE_HASH_DIFFERS', path=str(p), expected=expected)
        return False

    def read(self, path):
        p = self.need(path)
        if p is None: return None
        data = p.read_bytes(); self.files[str(p)] = hashlib.sha256(data).hexdigest()
        return json.loads(data.decode('utf-8-sig'))

    def lines(self, path):
        p = self.need(path)
        if p is None: return []
        data = p.read_bytes(); self.files[str(p)] = hashlib.sha256(data).hexdigest(); result = []
        lines = data.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip(): continue
            try: result.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if index == len(lines)-1 and not line.endswith(b'\n'):
                    self.missing.append(dict(reason='INCOMPLETE_JSONL_TAIL', path=str(p), bytes=len(line)))
                else: self.check(False, 'MALFORMED_JSONL', path=str(p), line=index+1)
        return result

    def csv(self, path):
        p = self.need(path)
        if p is None: return []
        data = p.read_bytes(); self.files[str(p)] = hashlib.sha256(data).hexdigest()
        return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))

    def count(self, observed, expected, name):
        self.check(observed <= expected, 'OVER_BUDGET_OR_DUPLICATE', name=name, observed=observed, expected=expected)
        if observed < expected: self.missing.append(dict(reason='INCOMPLETE_COUNT', name=name, observed=observed, expected=expected))

    def phase(self, name, function):
        tick = time.perf_counter()
        try: value = function()
        except Exception as exc:
            self.check(False, 'AUDIT_PHASE_EXCEPTION', phase=name, error=repr(exc)); value = dict(status='INVALID_INPUT_OR_AUDITOR_ERROR')
        self.parts[name] = dict(seconds=time.perf_counter()-tick, result=value)


def contracts(a, reg):
    a.check(reg.get('status') == 'FROZEN', 'REGISTRATION_NOT_FROZEN')
    a.check(reg.get('evaluation_labels') == list(MAIN), 'CHECKPOINT_LABELS_DIFFER')
    for path, key in (('baseline_registration_path', 'baseline_registration_sha256'),
                      ('source_data', 'source_data_sha256'), ('interface_manifest_path', 'interface_manifest_sha256'),
                      ('replay_plan_path', 'replay_plan_sha256'), ('replay_exposure_path', 'replay_exposure_sha256'),
                      ('topology_manifest_path', 'topology_manifest_sha256')):
        a.verify(reg[path], reg[key])
    for path, digest in {**reg['training_code_hashes'], **reg['evaluation_code_hashes']}.items(): a.verify(path, digest)
    topo = a.read(reg['topology_manifest_path'])
    if topo:
        a.check(len(topo['training']) == 47 and len(topo['requests']) == 24, 'TOPOLOGY_SCOPE_DIFFERS')
        a.check(topo['source_Y_used'] is False and topo['coordinate_welding'] is False, 'TOPOLOGY_CONTRACT_DIFFERS')
        for path, digest in topo['source_files'].items(): a.verify(path, digest)
    zero = a.read(C.OUT / 'audits/zero_initialization.json')
    if zero:
        a.check(zero.get('status') == 'PASS' and zero.get('normal_training_microbatches') == 8, 'FIRST_BATCH_PROOF_INVALID')
        for key in ('zero_residual_exact', 'unchanged_hidden_exact', 'first_batch_FM_matches', 'old_state_sha256_matches'):
            a.check(zero.get(key) is True, 'FIRST_BATCH_CHECK_FALSE', field=key)
    if a.final: a.verify(reg['source_checkpoint'], reg['source_checkpoint_sha256'])
    return dict(registration_sha256=a.digest(C.OUT/'registration.json'), source_checkpoint_disk_hash='VERIFIED' if a.final else 'DEFERRED_PROGRESS_MODE',
                official_D0_disk_rehash='NOT_REPEATED; every saved proposal is checked against its pinned hash',
                topology_conditions=len(topo['training']) if topo else None, code_hashes=len({**reg['training_code_hashes'], **reg['evaluation_code_hashes']}))


def training(a, reg):
    plan = a.read(reg['replay_plan_path']); reference = a.lines(reg['replay_exposure_path'])
    exposures = a.lines(TRAIN/'exposure.jsonl'); logs = a.lines(TRAIN/'training_log.jsonl'); calls = a.lines(TRAIN/'calls.jsonl')
    records = plan['records'] if plan else []; a.count(len(records), 48000, 'original_plan_records')
    a.count(len(reference), 48000, 'original_exposures'); a.count(len(exposures), 48000, 'new_exposures'); a.count(len(logs), 6000, 'completed_training_log_updates')
    ref = {int(r['sample_index']): r for r in reference}; seen = set(); parent_counts = collections.Counter()
    fields = ('input_bytehash','label_bytehash','interface_mask_sha256','ot_cache_key','uid','task_id','N','K','alpha','t','free_coordinate_count')
    for row in exposures:
        sid = int(row['sample_index']); a.check(sid not in seen and 0 <= sid < 48000, 'DUPLICATE_OR_INVALID_EXPOSURE', sample_index=sid); seen.add(sid)
        old = ref.get(sid)
        if old is None: a.check(False, 'MISSING_BASELINE_EXPOSURE', sample_index=sid); continue
        for key in fields: a.check(row.get(key) == old.get(key), 'EXPOSURE_PAIR_DIFFERS', sample_index=sid, field=key)
        a.check(row['local_step'] == sid//8+1 and row['global_step'] == 6001+sid//8, 'EXPOSURE_CLOCK_DIFFERS', sample_index=sid)
        a.check(row['condition_type'] == 'SEAT_FULL' and row['K'] >= 2 and bool(row.get('boundary_edge_mask_sha256')), 'EXPOSURE_FULL_TOPOLOGY_MISSING', sample_index=sid)
        if sid < len(records):
            for key in ('uid','task_id','N','K','alpha'): a.check(row.get(key) == records[sid].get(key), 'PLAN_EXPOSURE_DIFFERS', sample_index=sid, field=key)
        parent_counts[row['uid']] += 1
    a.check(sorted(seen) == list(range(len(seen))), 'EXPOSURE_PREFIX_NOT_CONTIGUOUS')
    a.check([int(x['local_step']) for x in logs] == list(range(1, len(logs)+1)), 'UPDATE_LOG_REPLAY_OR_GAP')
    begun = {}; pending = {}; attempted = collections.Counter(); done = collections.Counter(); failed = collections.Counter(); optimizer_steps = []; forward_ids = []; backward_ids = []
    seconds = collections.Counter()
    for row in calls:
        ident, kind, event = row['call_id'], row['kind'], row['event']; context = row.get('context', {})
        a.check(kind in LIMITS and row.get('arm') == 'CONTOUR', 'CALL_CATEGORY_OR_OWNER_DIFFERS', call_id=ident)
        if event == 'CALL_BEGIN':
            a.check(ident not in begun, 'DUPLICATE_CALL_ID', call_id=ident); begun[ident] = row; pending[ident] = row; attempted[kind] += 1
            if kind == 'optimizer_update': optimizer_steps.append(int(context['global_step']))
        elif event in ('CALL_DONE', 'CALL_FAILED'):
            start = pending.pop(ident, None)
            a.check(start is not None and start['kind'] == kind and start.get('context') == context, 'CALL_COMPLETION_WITHOUT_MATCH', call_id=ident)
            (done if event == 'CALL_DONE' else failed)[kind] += 1; seconds[kind] += float(row.get('seconds', 0))
            if event == 'CALL_DONE' and kind in ('train_forward','train_backward'):
                sid = int(context['sample_index']); (forward_ids if kind == 'train_forward' else backward_ids).append(sid)
                if sid in ref:
                    for key in ('input_bytehash','label_bytehash'): a.check(context.get(key) == ref[sid][key], 'CALL_PAIRED_INPUT_DIFFERS', call_id=ident, field=key)
        else: a.check(False, 'UNKNOWN_CALL_EVENT', event=event)
    for kind, limit in LIMITS.items(): a.count(attempted[kind], limit, 'attempted_'+kind); a.count(done[kind], limit, 'successful_'+kind)
    a.check(not failed, 'FAILED_SCIENTIFIC_CALLS', failed=dict(failed))
    a.check(optimizer_steps == list(range(6001,6001+len(optimizer_steps))), 'OPTIMIZER_REPLAY_OR_GAP')
    for name, ids in (('forward',forward_ids),('backward',backward_ids)):
        a.check(ids == list(range(len(ids))), 'MICROBATCH_REPLAY_OR_GAP', kind=name)
    if pending: a.missing.append(dict(reason='PENDING_MODEL_CALLS', call_ids=sorted(pending)))
    status = a.read(TRAIN/'status.json')
    if a.final and status: a.check(status.get('status') == 'COMPLETE' and status.get('local_step') == 6000 and status.get('global_step') == 12000, 'FINAL_TRAINING_STATUS_DIFFERS')
    if (TRAIN/'worker_owner.json').exists(): a.missing.append(dict(reason='TRAINING_OWNER_STILL_PRESENT', path=str(TRAIN/'worker_owner.json')))
    # Read receipts and hash arrays only. Never instantiate Builder or call OT.
    ot = C.BASELINE/'ot/FULL_SEAT'; receipt_count = len(list(ot.glob('[0-9][0-9][0-9][0-9][0-9].json')))
    a.count(receipt_count, 6000, 'original_OT_batch_receipts'); verified_ot = 0
    if a.final:
        for index in range(6000):
            p = ot/f'{index:05d}.json'; receipt = a.read(p)
            if not receipt: continue
            expected_records = hashlib.sha256(json.dumps(records[index*8:(index+1)*8], sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()
            a.check(receipt['identity'] == dict(records_sha256=expected_records, plan_sha256=reg['replay_plan_sha256'], batch_index=index), 'OT_IDENTITY_DIFFERS', batch=index)
            a.check(receipt.get('actual_OT') == 8 and receipt.get('status') == 'PASS', 'ORIGINAL_OT_RECEIPT_INVALID', batch=index)
            if a.verify(ot/f'{index:05d}.npz', receipt['sha256']): verified_ot += 1
    a.check(not list((C.OUT/'ot').rglob('*.npz')) if (C.OUT/'ot').exists() else True, 'UNREGISTERED_NEW_OT_ARRAYS')
    return dict(attempted=dict(attempted), successful=dict(done), failed=dict(failed), pending=sorted(pending), call_seconds=dict(seconds),
                training_log_rows=len(logs), exposure_rows=len(exposures), paired_exposure_rows=len(seen), parent_counts=dict(parent_counts),
                original_OT_receipts=receipt_count, OT_arrays_hash_verified=verified_ot, OT_verification='ALL_FINAL' if a.final else 'RECEIPT_COUNT_ONLY_PROGRESS',
                new_OT_executed_by_auditor=0, training_status=status,
                torch_reserved_peak_bytes=max((x.get('cuda_peak_reserved_bytes',0) for x in logs),default=None),
                torch_allocated_peak_bytes=max((x.get('cuda_peak_allocated_bytes',0) for x in logs),default=None),
                allocator_scope='Training-update PyTorch allocator counters; not full worker process or device VRAM')


def _sampling_calls(a, folder):
    rows = a.lines(folder/'calls.jsonl'); reserve = collections.Counter(); returns = collections.Counter(); errors = 0
    registrations=[]; completions=[]
    for index,row in enumerate(rows):
        kind, event = row.get('kind'), row['event']
        if event == 'CELL_REGISTERED': registrations.append((index,row))
        elif event == 'ROLLOUT_COMPLETE': completions.append((index,row))
        elif event == 'RESERVE':
            reserve[kind] += 1; a.check(row.get('ordinal') == reserve[kind], 'ROLLOUT_RESERVATION_ORDINAL_DIFFERS', path=str(folder), kind=kind)
        elif event == 'RETURN': returns[kind] += 1
        elif event == 'ERROR': errors += 1
        else: a.check(False, 'UNKNOWN_ROLLOUT_EVENT', event=event)
    a.check(errors == 0, 'FAILED_ROLLOUT_CALL', path=str(folder))
    a.check(set(reserve) <= {'rollout_attempt','sampling_forward'}, 'UNREGISTERED_ROLLOUT_KIND', path=str(folder))
    a.check(set(returns) <= {'sampling_forward'}, 'UNREGISTERED_ROLLOUT_RETURN_KIND', path=str(folder))
    a.count(len(registrations),1,'rollout_registration:'+folder.name)
    a.count(len(completions),1,'rollout_completion:'+folder.name)
    if registrations:a.check(registrations[0][0]==0,'ROLLOUT_REGISTRATION_NOT_FIRST',path=str(folder))
    for index,completed in completions:
        a.check(index==len(rows)-1,'ROLLOUT_COMPLETION_NOT_LAST',path=str(folder))
        a.check(completed.get('nfe')==50 and reserve['sampling_forward']==50 and returns['sampling_forward']==50,'ROLLOUT_COMPLETION_NFE_DIFFERS',path=str(folder))
        a.check(completed.get('known_all51_bitwise') is True,'ROLLOUT_COMPLETION_KNOWN_NOT_EXACT',path=str(folder))
        a.check(bool(registrations) and completed.get('context')==registrations[0][1].get('context'),'ROLLOUT_COMPLETION_CONTEXT_DIFFERS',path=str(folder))
        a.check(isinstance(completed.get('seconds'),(int,float)) and completed['seconds']>=0,'ROLLOUT_COMPLETION_TIME_INVALID',path=str(folder))
    a.count(reserve['rollout_attempt'],1,'rollout_attempt:'+folder.name)
    a.count(reserve['sampling_forward'],50,'sampling_reservations:'+folder.name)
    a.count(returns['sampling_forward'],50,'sampling_returns:'+folder.name)
    return reserve['sampling_forward'], returns['sampling_forward']


def evaluation(a, reg):
    qs = {q['request_id']:q for q in reg['requests']}; totals = collections.Counter(); per_label = []
    d0_hash = reg['d0_checkpoint']['sha256']
    for label in MAIN+REFERENCES:
        counts = collections.Counter(); is_new = label in TRAINED
        for rid,q in qs.items():
            rawfolder = EV/'raw'/label/rid; postfolder = EV/'post'/label/rid
            raw = a.read(rawfolder/'receipt.json'); post = a.read(postfolder/'receipt.json')
            for stage, row in (('RAW',raw),('POST',post)):
                if row is None: continue
                counts[stage] += 1
                a.check(row.get('status') == 'COMPLETE' and row.get('label') == label and row.get('request_id') == rid, 'OUTPUT_IDENTITY_DIFFERS', label=label, request_id=rid, stage=stage)
                for field in ('N','K','C_sha256','Gaussian_sha256','condition_source_sha256','input_sha256'):
                    a.check(row.get(field) == q[field], 'FIXED_REQUEST_DIFFERS', label=label, request_id=rid, stage=stage, field=field)
                if not is_new:
                    a.check(row.get('new_sampling_F') == 0 and row.get('new_rollouts') == 0, 'CACHE_CLAIMS_NEW_GENERATION', label=label, request_id=rid)
                    if stage == 'POST': a.check(row.get('new_D0_F') == 0, 'CACHE_CLAIMS_NEW_D0', label=label, request_id=rid)
                    a.verify(row['historical_receipt'],row['historical_receipt_sha256'])
                if a.final:
                    if stage == 'RAW':
                        for field in ('raw','ply','result','velocities'): a.verify(row[field+'_path'],row[field+'_sha256'])
                    else:
                        a.verify(row['post_path'],row['post_sha256'])
                        for path,h in row.get('files',{}).items(): a.verify(path,h)
            if is_new:
                if (rawfolder/'calls.jsonl').exists():
                    tried,done = _sampling_calls(a,rawfolder); totals['sampling_attempted'] += tried; totals['sampling_returned'] += done
                    if raw is None: a.missing.append(dict(reason='RAW_CALLS_WITHOUT_COMMITTED_RECEIPT',label=label,request_id=rid))
                if raw:
                    a.check(raw.get('new_sampling_F') == 50 and raw.get('new_rollouts') == 1,'NEW_ROLLOUT_COUNTS_DIFFER',label=label,request_id=rid)
                    totals['new_RAW'] += 1
                started = postfolder/'D0_started.json'; proposalpath = postfolder/'proposal_receipt.json'
                if started.exists():
                    d = a.read(started); totals['D0_attempted'] += 1
                    a.check(d.get('checkpoint_sha256') == d0_hash, 'D0_SOURCE_DIFFERS',label=label,request_id=rid)
                    if not proposalpath.exists(): a.missing.append(dict(reason='PENDING_D0_RETURN',label=label,request_id=rid))
                if proposalpath.exists():
                    d = a.read(proposalpath); totals['D0_returned'] += int(d.get('model_forwards',0))
                    a.check(started.exists() and d.get('status') == 'PASS' and d.get('model_forwards') == 1 and d.get('checkpoint_sha256') == d0_hash,'D0_RETURN_INVALID',label=label,request_id=rid)
                    if raw: a.check(d.get('raw_sha256') == raw['raw_sha256'],'D0_RAW_LINEAGE_DIFFERS',label=label,request_id=rid)
                if post:
                    a.check(post.get('new_D0_F') == 1 and post.get('GEO_calls') == 1,'POST_CALL_COUNTS_DIFFER',label=label,request_id=rid)
                    restore = post.get('source_restoration',{})
                    a.check(restore.get('status') == 'PASS' and restore.get('source_C_order_fixed_after_restoration') is True,'SOURCE_WINDING_EXPORT_MISSING',label=label,request_id=rid)
                    totals['new_POST'] += 1
            elif raw and post: totals['cached_pairs'] += 1
        a.count(counts['RAW'],24,label+'_RAW'); a.count(counts['POST'],24,label+'_POST')
        per_label.append(dict(label=label,**counts))
        # Table completeness, without importing/running any scorer.
        rows=a.csv(EV/'tables'/label/'per_output.csv'); keys=[(r.get('request_id'),r.get('stage')) for r in rows]
        expected={(rid,stage) for rid in qs for stage in ('RAW','POST')}
        a.check(len(keys)==len(set(keys)) and set(keys)<=expected,'SCORE_TABLE_DUPLICATE_OR_WRONG_IDENTITY',label=label)
        a.count(len(keys),48,label+'_score_rows')
        for row in rows: a.check(row.get('label')==label,'SCORE_LABEL_DIFFERS',label=label)
        for name,countkey in (('score_completion','receipts'),('boundary_completion','receipts')):
            receipt=a.read(EV/name/(label+'.json'))
            if receipt:
                a.check(receipt.get('status')=='COMPLETE','SCORING_COMPLETION_STATUS',label=label,kind=name)
                a.count(len(receipt.get(countkey,[])),48,label+'_'+name)
    for key,n in (('new_RAW',72),('new_POST',72),('sampling_attempted',3600),('sampling_returned',3600),('D0_attempted',72),('D0_returned',72),('cached_pairs',120)):
        a.count(totals[key],n,key)
    return dict(totals=dict(totals),per_label=per_label,step0_reuse=24,baseline_curve_reuse=96,
                semantics='Count original reservations/returns only for new three candidate endpoints; historical cells are verified reuse, not new calls.')


def weights(a, reg):
    receipts = []; final_path = None; final_receipt = None
    for step in (1000,3000,6000):
        label='CONTOUR_S'+str(step)
        for kind,path in (('full',C.OUT/'checkpoints'/f'CONTOUR_step{step}.pt'),('inference',C.OUT/'inference'/f'{label}_generator.pt')):
            receipt=a.read(path.with_suffix('.json'))
            if not receipt:continue
            if a.final:a.verify(path,receipt['sha256'])
            a.check(receipt.get('local_step')==step and receipt.get('global_step')==6000+step and receipt.get('arm')=='CONTOUR','CHECKPOINT_RECEIPT_CLOCK_DIFFERS',path=str(path))
            receipts.append(dict(kind=kind,step=step,path=str(path),sha256=receipt['sha256'],generator_state_sha256=receipt.get('generator_state_sha256'),file_hash_verified=a.final))
            if kind=='inference':
                binding=a.read(EV/'bindings'/f'{label}.json')
                if binding:a.check(binding['sha256']==receipt['sha256'] and binding['generator_state_sha256']==receipt['generator_state_sha256'],'INFERENCE_BINDING_DIFFERS',label=label)
            elif step==6000:final_path,final_receipt=path,receipt
    if not a.final:return dict(status='DEFERRED_FINAL_CHECKPOINT_LOAD',receipts=receipts,checkpoint_loads=0)
    if final_path is None or a.need(final_path) is None:return dict(status='PARTIAL_FINAL_CHECKPOINT_MISSING',receipts=receipts,checkpoint_loads=0)
    cache_path=C.OUT/'audits'/('final_checkpoint_state_'+final_receipt['sha256'][:16]+'.json')
    if cache_path.exists():
        cached=a.read(cache_path)
        a.check(cached.get('checkpoint_file_sha256')==final_receipt['sha256'] and cached.get('registration_sha256')==a.digest(C.OUT/'registration.json') and cached.get('auditor_source_sha256')==C.sha(__file__),'CHECKPOINT_AUDIT_CACHE_IDENTITY_DIFFERS')
        return dict(cached['result'],checkpoint_loads=0,prior_checkpoint_loads=1,reused_verified_state_receipt=str(cache_path))
    errors_before=len(a.errors)
    # No Native imports, model object, CUDA API, prediction, or optimizer step.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2)
    p=torch.load(final_path,map_location='cpu',weights_only=True,mmap=True);a.weight_loads+=1
    def state_hash(state):
        h=hashlib.sha256()
        for key,value in sorted(state.items()):
            value=value.detach().cpu().contiguous();h.update(key.encode());h.update(str((tuple(value.shape),value.dtype)).encode());h.update(value.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    def tree_hash(value):
        h=hashlib.sha256()
        def add(v):
            if isinstance(v,torch.Tensor):
                x=v.detach().cpu().contiguous();h.update(str(('tensor',str(x.dtype),tuple(x.shape))).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(v,dict):
                h.update(b'dict')
                for key in sorted(v,key=lambda k:(type(k).__name__,repr(k))):add(key);add(v[key])
            elif isinstance(v,(list,tuple)):
                h.update(str((type(v).__name__,len(v))).encode())
                for item in v:add(item)
            else:h.update(str((type(v).__name__,repr(v))).encode())
        add(value);return h.hexdigest()
    observed={key:fn(p[key]) for key,fn in (('generator_state',state_hash),('optimizer',tree_hash),('rng',tree_hash))}
    for key,hash_key in (('generator_state','generator_state_sha256'),('optimizer','optimizer_state_sha256'),('rng','rng_sha256')):
        a.check(observed[key]==p[hash_key]==final_receipt[hash_key],'FINAL_INTERNAL_STATE_HASH_DIFFERS',state=key)
    a.check((p.get('local_step'),p.get('global_step'),p.get('arm'),p.get('update_in_progress'))==(6000,12000,'CONTOUR',False),'FINAL_CHECKPOINT_CLOCK_DIFFERS')
    a.check(p.get('protocol')==reg and p.get('plan_sha256')==reg['replay_plan_sha256'],'FINAL_PROTOCOL_DIFFERS')
    state=p['generator_state']; names=p['optimizer_parameter_names']; groups=p['optimizer']['param_groups']; moments=p['optimizer']['state']
    a.check(len(groups)==len(names)==2,'FINAL_OPTIMIZER_GROUPS_DIFFER')
    clocks=[];all_ids=[];name_flat=[]
    for gi,(group,group_names) in enumerate(zip(groups,names)):
        a.check(len(group['params'])==len(group_names),'OPTIMIZER_NAME_COUNT_DIFFERS',group=gi)
        a.check((group['lr'],tuple(group['betas']),group['eps'],group['weight_decay'])==(1e-5,(.9,.95),1e-8,0.),'OPTIMIZER_HYPERPARAMETERS_DIFFER',group=gi)
        expected_clock=12000 if gi==0 else 6000;observed_clocks=[]
        for ident,name in zip(group['params'],group_names):
            all_ids.append(ident);name_flat.append(name);key=name.removeprefix('generator.');value=state[key];moment=moments[ident]
            a.check(key.startswith('contour_readout.')==(gi==1),'OPTIMIZER_GROUP_PARAMETER_OWNERSHIP_DIFFERS',name=name)
            a.check(value.dtype==torch.float32 and bool(torch.isfinite(value).all()),'GENERATOR_PARAMETER_INVALID',name=name)
            a.check(set(moment)=={'step','exp_avg','exp_avg_sq'},'OPTIMIZER_STATE_KEYS_DIFFER',name=name)
            clock=float(moment['step']);observed_clocks.append(clock)
            a.check(clock==expected_clock,'ACTUAL_PARAMETER_ADAMW_CLOCK_DIFFERS',name=name,observed=clock,expected=expected_clock)
            for field in ('exp_avg','exp_avg_sq'):
                v=moment[field];a.check(v.shape==value.shape and v.dtype==torch.float32 and bool(torch.isfinite(v).all()),'OPTIMIZER_MOMENT_INVALID',name=name,field=field)
        clocks.append(dict(group=gi,parameters=len(group_names),expected=expected_clock,actual_unique=sorted(set(observed_clocks))))
    a.check(len(all_ids)==len(set(all_ids))==len(moments),'OPTIMIZER_ID_DUPLICATES_OR_UNOWNED_STATE')
    a.check(set(name_flat)=={'generator.'+key for key in state},'GENERATOR_OPTIMIZER_PARAMETER_COVERAGE_DIFFERS')
    a.check(len(names[0])==194 and sum(v.numel() for n,v in state.items() if n.startswith('contour_readout.'))==412032,'MODEL_CAPACITY_DIFFERS')
    del p
    result=dict(status='AUDITED_FINAL_ONCE',receipts=receipts,checkpoint_loads=a.weight_loads,actual_state_hashes=observed,actual_parameter_clocks=clocks,
                intermediate_state_scope='1000/3000 full and inference files verified against actual saved receipts; only6000 optimizer tensors loaded/rehashed')
    if len(a.errors)==errors_before:
        C.write(cache_path,dict(status='PASS',checkpoint_file_sha256=final_receipt['sha256'],registration_sha256=a.digest(C.OUT/'registration.json'),auditor_source_sha256=C.sha(__file__),result=result))
    return result


def results(a, reg):
    qs={q['request_id']:q for q in reg['requests']};rows=a.csv(EV/'review/AI_PROXY_reviews.csv')
    expected={(label,rid,stage) for label in TRAINED for rid in qs for stage in ('RAW','POST')};seen=set();accepted=0
    for row in rows:
        key=(row.get('label'),row.get('request_id'),row.get('stage'));okay=a.check(key in expected and key not in seen,'REVIEW_IDENTITY_INVALID',identity=key);seen.add(key)
        okay &= a.check(row.get('reviewer_kind')=='AI_PROXY' and bool(row.get('reviewer','').strip()) and row.get('review_status') in ('REVIEWED','COMPLETE'),'REVIEW_NOT_ACTUAL_COMPLETE',identity=key)
        okay &= a.check(bool(row.get('notes','').strip()) and str(row.get('blinded','')).lower() in ('false','0'),'REVIEW_NOTES_OR_NONBLIND_FLAG_MISSING',identity=key)
        for field in REVIEW_FIELDS:okay &= a.check(row.get(field) in ('YES','NO','UNCERTAIN'),'REVIEW_LABEL_MISSING',identity=key,field=field)
        q=qs.get(row.get('request_id'))
        if q:
            okay &= a.check(row.get('parent_id')==q['parent_id'] and int(row['candidate'])==q['candidate'],'REVIEW_PARENT_NOISE_DIFFERS',identity=key)
            for field in ('N','K'):
                if row.get(field):okay &= a.check(int(row[field])==q[field],'REVIEW_NK_DIFFERS',identity=key,field=field)
        for field in ('image','section_image'):
            path,sha=row.get(field+'_path'),row.get(field+'_sha256')
            if path and sha:okay &= a.verify(path,sha)
            else:okay &= a.check(False,'REVIEW_IMAGE_EVIDENCE_MISSING',identity=key,field=field)
        accepted+=int(okay)
    a.count(len(rows),144,'actual_new_AI_PROXY_rows');a.count(accepted,144,'validated_actual_AI_PROXY_rows')
    review_by={(r.get('label'),r.get('request_id'),r.get('stage')):r for r in rows}
    baseline_reviews=a.csv(C.BASELINE/'evaluation/review/AI_PROXY_reviews.csv')
    review_by.update({(r.get('label'),r.get('request_id'),r.get('stage')):r for r in baseline_reviews})
    scores={}
    for label in ('CONTOUR_S6000','FULL_S6000'):
        for row in a.csv(EV/'tables'/label/'per_output.csv'):
            if row.get('stage')=='POST':scores[label,row.get('request_id')]=row
    latest=a.read(EV/'comparison/latest.json');summary=None
    if latest and a.verify(latest['path'],latest['sha256']):summary=a.read(latest['path'])
    tables={}
    if summary:
        if summary.get('status')!='COMPLETE':a.missing.append(dict(reason='PAIRED_COMPARISON_PARTIAL',status=summary.get('status')))
        for path,sha in summary.get('output_hashes',{}).items():a.verify(path,sha)
        folder=C.resolve(summary['out'])
        for name,count,stages in (('final_joint_vs_FULL_S6000.csv',24,False),('final_vs_FULL_S6000.csv',48,True)):
            records=a.csv(folder/name);a.count(len(records),count,name);seen=set()
            for row in records:
                key=(row.get('request_id'),row.get('stage')) if stages else row.get('request_id')
                allowed={(rid,s) for rid in qs for s in ('RAW','POST')} if stages else set(qs)
                a.check(key in allowed and key not in seen,'FINAL_PAIR_IDENTITY_INVALID',table=name,key=key);seen.add(key)
                a.check(row.get('label')=='CONTOUR_S6000' and row.get('baseline')=='FULL_S6000','FINAL_PAIR_COMPARATOR_DIFFERS',table=name,key=key)
                if not stages:
                    pair_status=row.get('status')
                    if pair_status=='PARTIAL':a.missing.append(dict(reason='FINAL_JOINT_PAIR_PARTIAL',request_id=key))
                    else:a.check(pair_status=='COMPLETE','FINAL_JOINT_PAIR_STATUS_INVALID',key=key,status=pair_status)
                    for prefix,label in (('current','CONTOUR_S6000'),('baseline','FULL_S6000')):
                        review=review_by.get((label,key,'RAW'));score=scores.get((label,key))
                        if review and score and score.get('C_F_valid_edge_count') not in ('',None):
                            observed=review.get('C_adoption')=='YES' and float(score['C_F_valid_edge_count'])>0
                            a.check(row.get(prefix+'_joint','').lower()==str(observed).lower(),'JOINT_NOT_SAME_REQUEST_RECOMPUTABLE',request_id=key,label=label)
                        else:a.missing.append(dict(reason='JOINT_RECOMPUTATION_INPUT_MISSING',request_id=key,label=label))
            tables[name]=len(records)
    return dict(actual_review_rows=len(rows),valid_actual_reviews=accepted,final_pair_tables=tables,
                image_scope='Existing image file hashes and completed review records only. Auditor did not view images or generate ratings.',
                scoring_scope='Existing rows/receipts only; no scorer or geometry-analysis function is executed.')


def resources(a):
    stages=[];failures=[];pending=[]
    for folder in sorted((C.OUT/'orchestration/jobs').glob('*')):
        request=a.read(folder/'request.json')
        if not request:continue
        donepath=folder/'done.json'
        if not donepath.exists():pending.append(folder.name);continue
        done=a.read(donepath)
        a.check(done.get('request_sha256')==a.digest(folder/'request.json'),'SCHEDULER_REQUEST_HASH_DIFFERS',job=folder.name)
        if done.get('returncode')!=0:failures.append(dict(job=folder.name,kind=request['kind'],returncode=done.get('returncode')))
        stages.append(dict(job=folder.name,kind=request['kind'],phase=request.get('phase'),label=request.get('label'),elapsed_seconds=done.get('elapsed_seconds'),started=done.get('started'),finished=done.get('finished')))
    if pending:a.missing.append(dict(reason='PENDING_ORCHESTRATION_JOBS',jobs=pending))
    a.check(not failures,'FAILED_ORCHESTRATION_JOBS',failures=failures)
    attempts=[]
    for path in sorted((TRAIN/'attempts').glob('*/final_status.json')):
        row=a.read(path);rs=row.get('resources',{}) if row else {}
        attempts.append(dict(path=str(path),status=row.get('status'),local_step=row.get('local_step'),wall_seconds=row.get('wall_seconds'),resources=rs))
    cpu={}
    try:
        import psutil
        cpu=dict(available_RAM_bytes=psutil.virtual_memory().available,auditor_RSS_bytes=psutil.Process().memory_info().rss,logical_cores=psutil.cpu_count())
    except ImportError:cpu=dict(status='NA_PSUTIL_NOT_AVAILABLE')
    return dict(stage_wall_times=stages,training_attempts=attempts,pending_jobs=pending,failed_jobs=failures,auditor_CPU=cpu,
                timing_scope='Recorded job wall times include loading, disk IO and contention; overlapping phases must not be summed as elapsed experiment wall time.',
                VRAM_scope='Keep saved allocator peaks, matching-process nvidia-smi peaks if available, and device-wide peaks distinct. Device occupancy includes unrelated user processes.')


def run(stage='progress'):
    a=Audit(stage);reg=a.read(C.OUT/'registration.json')
    if reg:
        for name,function in (('contracts',lambda:contracts(a,reg)),('training',lambda:training(a,reg)),('evaluation',lambda:evaluation(a,reg)),
                              ('checkpoints',lambda:weights(a,reg)),('results_and_reviews',lambda:results(a,reg)),('resources_and_times',lambda:resources(a))):
            a.phase(name,function)
    report_hash=None
    if a.final:
        report=a.need(C.OUT/'REPORT.md')
        if report is not None:report_hash=a.digest(report)
    status='FAIL' if a.errors else 'PARTIAL' if a.missing else 'PASS'
    result=dict(version=VERSION,stage=stage,status=status,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=a.checks,
                errors=a.errors,missing=a.missing,blocking_findings=a.errors+a.missing,phases=a.parts,read_file_hashes=a.files,checkpoint_loads=a.weight_loads,
                elapsed_seconds=time.perf_counter()-a.started,auditor_source_sha256=C.sha(__file__),
                model_forwards=0,model_backwards=0,optimizer_updates=0,new_OT=0,new_noise_arrays=0,new_scores=0,new_images=0,new_image_reviews=0)
    if reg:result['registration_sha256']=a.digest(C.OUT/'registration.json')
    if report_hash is not None:result['report_sha256']=report_hash
    tag=hashlib.sha256(json.dumps(C.clean(result),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]
    path=C.OUT/'audits'/('execution_'+stage+'_'+tag+'.json');C.write(path,result)
    C.write(C.OUT/'audits'/('execution_'+stage+'_latest.json'),dict(status=status,path=str(path),sha256=C.sha(path),stage=stage),replace=True)
    return dict(status=status,path=str(path),sha256=C.sha(path),errors=len(a.errors),missing=len(a.missing),checks=a.checks,checkpoint_loads=a.weight_loads,elapsed_seconds=result['elapsed_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('progress','final'),default='progress')
    args=parser.parse_args();result=run(args.stage);print(json.dumps(result,indent=2))
    raise SystemExit(2 if result['status']=='FAIL' else 0)
