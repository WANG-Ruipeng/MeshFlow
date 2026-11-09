"""Independent read-only evidence audit; never constructs or calls a model.

Default prints readiness. ``--run`` writes a new immutable audit under
independent_audit/. A full PASS requires --verify-large --tensors and every
section, including actual visual-review records. Completed Wave1 tensor checks
can be reused through their explicit hash-bound receipt. Historical protected
weights use Stage0's two actual SHA checks plus unchanged size/mtime, while core
sources and new weights are rehashed. --sections permits incomplete
stage checks without implying completion of the full experiment.

NumPy reads saved arrays only. Optional torch usage is CPU mmap checkpoint
inspection, without importing training/model/scoring modules or restoring RNG.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

# CPU audit only; avoid contending with authorized experiment workers.
for _thread_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_thread_key, '1')

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_seam_quality_wave1_wave2_v1'
AUDIT = OUT / 'independent_audit'
WAVES = {'wave1': ('PLAIN_S1000', 'PLAIN_S2000', 'WISE_B_PLAIN_050'),
         'wave2': ('CLEAN_BULK_FM_S1000', 'SURGICAL_LATE4_S1000')}
ARMS = tuple(a for arms in WAVES.values() for a in arms)
TRAIN = ('CLEAN_BULK_FM', 'SURGICAL_LATE4')
LABELS = ('LONG', 'B_LONG_LAYOUT', 'PLAIN_S1000', 'PLAIN_S2000', 'PLAIN3000',
          'WISE_B_PLAIN_050', *WAVES['wave2'])
STAGES = ('RAW', 'POST', 'BOUND015')
SECTIONS = ('cleanup', 'exports', 'training', 'generation_wave1',
            'scores_wave1', 'generation_wave2', 'scores_wave2', 'results', 'visuals')


def resolve(value):
    s = str(value).replace('\\', '/')
    if os.name != 'nt' and re.match(r'^[A-Za-z]:/', s):
        s = '/mnt/' + s[0].lower() + s[2:]
    elif os.name == 'nt' and re.match(r'^/mnt/[a-z]/', s):
        s = s[5].upper() + ':/' + s[7:]
    p = Path(s)
    return (p if p.is_absolute() else ROOT / p).resolve()


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with resolve(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def jsonl(path):
    with resolve(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def truth(value):
    return str(value).strip().lower() in ('true', '1', 'yes')


def same(a, b):
    import numpy as np
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
        np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))


def arrays(path):
    import numpy as np
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def frozen(name):
    m = re.match(r'^backbone\.layers\.([0-9]+)\.', name)
    return bool(m and int(m.group(1)) < 8)


def tree_hash(value):
    """Independent copy of persisted tensor-tree serialization; no model import."""
    import torch
    h = hashlib.sha256()
    def add(v):
        if isinstance(v, torch.Tensor):
            x = v.detach().cpu().contiguous()
            h.update(str(('tensor', str(x.dtype), tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v, key=lambda k: (type(k).__name__, repr(k))):
                add(k); add(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str((type(v).__name__, len(v))).encode())
            for item in v:
                add(item)
        else:
            h.update(str((type(v).__name__, repr(v))).encode())
    add(value)
    return h.hexdigest()


def state_hash(state):
    import torch
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Audit:
    def __init__(self, verify_large=False, tensors=False, visual_file=None):
        self.expected, self.scopes, self.checks = {}, {}, {}
        self.verify_large, self.tensors = verify_large, tensors
        self.visual_file = resolve(visual_file or OUT / 'visual_review.csv')
        self.protocol = read(OUT / 'protocol.json')
        self.qq = self.protocol['evaluation']['requests']
        self.requests = {q['request_id']: q for q in self.qq}
        require(len(self.qq) == len(self.requests) == 28, 'Fixed 28 request identities')
        require(Counter(q['split'] for q in self.qq) == Counter(TRAIN=16, DEV=12), 'Fixed split')
        self.hashes(self.protocol['protected_files'], 'Protocol protected input/source')
        self.core_large = {str(resolve(p)) for p in self.protocol['sources'].values()}
        for wave in WAVES:
            registration = OUT/('generation_'+wave+'_registration.json')
            if registration.exists():
                self.core_large.add(str(resolve(read(registration)['d0_checkpoint']['path'])))
        # Avoid re-reading unrelated historical weights already hashed twice in
        # the immediately preceding authorized cleanup. Never use this shortcut
        # for the current source/checkpoint, official D0, or any new weight.
        cleanup = read(OUT/'cleanup/completion.json')
        manifest_path = OUT/'cleanup/deletion_manifest.json'
        kept_path = OUT/'cleanup/retained_verified.csv'
        require(sha(manifest_path)==cleanup['manifest_sha256'] and sha(kept_path)==cleanup['retained_csv_sha256'],
                'Stage0 retained verification identity')
        before = {str(resolve(r['path'])):r for r in read(manifest_path)['protected']}
        self.retained = {}
        for item in rows(kept_path):
            key = str(resolve(item['path'])); original = before[key]
            require(item['sha256_before']==item['sha256_after']==original['sha256'], 'Stage0 double hash differs')
            self.retained[key] = original
        self.hashes({str(manifest_path):cleanup['manifest_sha256'],str(kept_path):cleanup['retained_csv_sha256']},
                    'Stage0 twice-actual-hash reuse evidence')
        self.latest_work = 0.

    def hashes(self, mapping, scope):
        for path, value in mapping.items():
            p = str(resolve(path))
            require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA: ' + p)
            require(p not in self.expected or self.expected[p] == value, 'Conflicting hash: ' + p)
            self.expected[p] = value
            self.scopes.setdefault(p, set()).add(scope)

    def receipt(self, path, scope):
        value = read(path)
        self.hashes(value.get('files', {}), scope)
        return value

    def registered(self, path, fields):
        r = read(path)
        for f in fields:
            self.hashes(r[f], str(path) + ':' + f)
        return r

    def cleanup(self):
        done = read(OUT / 'cleanup/completion.json')
        require(done['status'] == 'COMPLETE' and done['deleted_files'] == 16
                and done['retained_verified_files'] == 36, 'Cleanup count/status')
        deleted = rows(OUT / 'cleanup/deleted_weights.csv')
        kept = rows(OUT / 'cleanup/retained_verified.csv')
        require(len(deleted) == len({r['path'] for r in deleted}) == 16, '16 unique deletions')
        require(len(kept) == len({r['path'] for r in kept}) == 36, '36 unique retained assets')
        require(sum(int(r['bytes']) for r in deleted) == done['deleted_bytes'] == 25041197776,
                'Exact deleted bytes')
        for r in deleted:
            require(r['status'] == 'DELETED_EXACT_WHITELIST' and not resolve(r['path']).exists(),
                    'Deleted whitelist asset unexpectedly present: ' + r['path'])
        for r in kept:
            require(r['sha256_before'] == r['sha256_after'] and r['status'] == 'PROTECTED_SHA256_UNCHANGED',
                    'Cleanup retained hash mismatch')
            require(resolve(r['path']).stat().st_size == int(r['bytes']), 'Retained bytes changed')
        self.hashes({r['path']: r['sha256_after'] for r in kept}, '36 retained protected weights')
        self.hashes({str(OUT/'cleanup/deleted_weights.csv'): done['deleted_csv_sha256'],
                     str(OUT/'cleanup/retained_verified.csv'): done['retained_csv_sha256']}, 'Cleanup tables')
        require(len(list((OUT/'cleanup/retired').glob('*.json'))) == 16, '16 retirement sidecars')
        require(done['model_loads'] == done['model_forwards'] == done['training_updates'] == 0, 'Cleanup zero models')
        return dict(status='PASS', deleted_files=16, retained_files=36, bytes=done['deleted_bytes'],
                    original_reports_unmodified_scope='Protected by whitelist; unrelated files not exhaustively hashed')

    def exports(self):
        r = self.receipt(OUT/'wave1_exports.json', 'Wave1 exported weights')
        require(r['status'] == 'COMPLETE' and r['model_calls'] == r['optimizer_updates'] == 0, 'Zero-update exports')
        recs = {x['arm']: x for x in r['records']}
        require(set(recs) == set(WAVES['wave1']), 'Exactly three Wave1 exports')
        reuse_path = AUDIT/'exports_tensor_reuse.json'
        if reuse_path.exists():
            reuse = read(reuse_path); prior = read(reuse['original_audit_path'])
            require(reuse['status']=='PASS_REUSE_PREVIOUS_ACTUAL_CPU_TENSOR_AUDIT'
                    and prior['checks']['exports']['status']=='PASS' and reuse['files']==r['files']
                    and reuse['records']==r['records'], 'Independent prior tensor result/input binding')
            self.hashes({reuse['original_audit_path']:reuse['original_audit_sha256'],
                         reuse['exports_manifest']:reuse['exports_manifest_sha256'],
                         str(reuse_path):sha(reuse_path)}, 'Previous actual CPU tensor audit')
            return dict(**prior['checks']['exports'], verification='REUSED_ACTUAL_INDEPENDENT_CPU_TENSOR_CHECK',
                        receipt_path=str(reuse_path), receipt_sha256=sha(reuse_path),
                        actual_weight_file_hashes='Checked in final unique-file SHA phase; no repeated tensor arithmetic')
        if not self.tensors:
            return dict(status='INCOMPLETE', reason='--tensors required for CPU weight-level identity and WISE verification')
        import torch
        torch.set_num_threads(1)
        require(not torch.cuda.is_initialized(), 'Audit CUDA must be uninitialized')
        for arm, record in recs.items():
            payload = torch.load(resolve(record['path']), map_location='cpu', weights_only=True, mmap=True)
            state = payload['generator_state']
            require(len(state) == 194 and all(t.dtype == torch.float32 for t in state.values()), '194 FP32 tensors')
            require(state_hash(state) == record['generator_state_sha256'] == payload['generator_state_sha256'], 'Export tensor hash')
            require(payload['EMA'] is False and payload['arm'] == arm, 'Live export identity')
            require(not any(k in payload for k in ('optimizer', 'rng', 'head_optimizer')), 'Generator-only export')
            source = [torch.load(resolve(v['path']), map_location='cpu', weights_only=True, mmap=True)
                      for v in record['provenance']]
            self.hashes({v['path']: v['sha256'] for v in record['provenance']}, 'Wave1 source weights')
            for name, value in state.items():
                require(all(tuple(x['generator_state']) == tuple(state) for x in source), 'WISE/source key order')
                target = source[0]['generator_state'][name]
                if arm == 'WISE_B_PLAIN_050':
                    require(payload['interpolation_alpha'] == .5 and len(source) == 2, 'Fixed WISE alpha .5')
                    target = .5 * target + .5 * source[1]['generator_state'][name]
                require(torch.equal(value.view(torch.uint8), target.view(torch.uint8)), 'Export exact values: ' + arm + ':' + name)
            del source, payload, state
        require(not torch.cuda.is_initialized(), 'CPU checkpoint audit initialized CUDA')
        return dict(status='PASS', exports=3, tensors_per_export=194, WISE_alpha=.5,
                    bitwise_source_or_midpoint=True, model_constructions=0, model_calls=0)

    def training(self):
        reg = self.registered(OUT/'training_registration.json', ('protected_hashes', 'code_hashes'))
        go = self.registered(OUT/'training_dispatch_go.json', ('code_hashes',))
        require(go['authorized'] and reg['arms'] == list(TRAIN) and reg['updates_per_arm'] == 1000,
                'Two released1000-update arms')
        require((reg['source_global_step'], reg['final_global_step'], reg['batch'], reg['microbatch']) == (13000,14000,8,1), 'Training settings')
        require(reg['dirty_bulk_weight'] == .2 and reg['owner_third_weight'] == 1
                and reg['frozen_layer_indices'] == list(range(8))
                and reg['weighted_denominator'] == 'WHOLE_EFFECTIVE_BATCH_WEIGHTED_FREE_SCALARS', 'Science contract')
        require(len(reg['records']) == 47 and all(r['split'] == 'TRAIN' for r in reg['records']), 'TRAIN47 only')
        self.hashes({str(OUT/'training_registration.json'): go['training_registration_sha256'],
                     reg['shared_data_manifest']: go['shared_data_manifest_sha256']}, 'Training release')
        manifest = read(reg['shared_data_manifest'])
        require(manifest['samples'] == 8000 and manifest['new_OT_calls'] == 0 and manifest['original_examples_exact'], 'Existing exact8000')
        old = read(manifest['source_manifest'])
        require(manifest['steps'] == old['steps'][:1000], 'Exact original first1000 updates')
        sigs, budgets, firsts = {}, {}, {}
        for arm in TRAIN:
            folder = OUT/'training'/arm
            status = read(folder/'status.json')
            require((status['status'],status['local_step'],status['global_step']) == ('COMPLETE',1000,14000), 'Unfinished: '+arm)
            started, finished, active, seen, updates = Counter(), Counter(), {}, set(), []
            for event in jsonl(folder/'calls.jsonl'):
                k, cid = event['kind'], event['call_id']
                require(event['arm'] == arm, 'Mixed training arm')
                if event['event'] == 'CALL_BEGIN':
                    require(cid not in seen, 'Repeated call ID'); seen.add(cid); active[cid] = k; started[k] += 1
                    if k == 'optimizer_update': updates.append(event['context']['global_step'])
                else:
                    require(event['event'] == 'CALL_DONE' and active.pop(cid,None) == k, 'Failed/unpaired training call')
                    finished[k] += 1
                self.latest_work = max(self.latest_work, float(event['time_unix']))
            expected = dict(train_forward=8000,train_backward=8000,optimizer_update=1000)
            require(dict(started) == dict(finished) == expected and not active, 'Exact F/B/update counts: '+arm)
            require(updates == list(range(13001,14001)), 'Update replay/gap')
            first = [read(p) for p in (folder/'attempts').glob('*/initialization.json') if read(p)['global_step'] == 13000]
            require(len(first) == 1, 'Unique source initialization'); first = first[0]; firsts[arm] = first
            for field, source in (('generator_state_sha256','source_generator_sha256'),('source_optimizer_sha256','source_optimizer_sha256'),('rng_sha256','source_rng_sha256')):
                require(first[field] == reg[source], 'Restore identity: '+field)
            require(first['head_loaded'] is False and first['head_forward_calls'] == 0 and first['scientific_probe_calls'] == 0, 'No head/probes')
            require(first['private_seam_code_object_identical'] and first['private_native_code_object_identical']
                    and first['frozen_block_forward_in_autograd'], 'Forward arithmetic/gradient graph')
            opt = first['optimizer_restore']
            require(opt['status'] == 'PASS' and opt['exact_retained_step_and_moments'], 'Optimizer restore receipt')
            expos = jsonl(folder/'exposure.jsonl'); require(len(expos) == 8000, 'Exposure count')
            signature = hashlib.sha256(); counts = Counter()
            for i, event in enumerate(expos):
                step, micro = divmod(i,8); cache = manifest['steps'][step]['samples'][micro]
                require(event['local_step'] == step+1 and event['microbatch'] == micro and event['global_step'] == 13001+step, 'Exposure order')
                for key in ('uid','alpha','sample_index'):
                    require(event[key] == cache[key], 'Exposure/cache identity: '+key)
                require(resolve(event['path']) == resolve(cache['path']) and event['input_sha256'] == cache['sha256'], 'Exact cached input')
                self.hashes({cache['path']: cache['sha256']}, '8000 shared cached inputs')
                stable = {k:event[k] for k in ('local_step','microbatch','uid','alpha','sample_index','N','K','t','free_coordinate_count','weighted_free_coordinate_count','input_sha256')}
                signature.update(json.dumps(stable,sort_keys=True).encode()); counts[event['uid']] += 1
            require(dict(counts) == manifest['parent_exposures'], 'Parent exposure count')
            sigs[arm] = signature.hexdigest(); budgets[arm] = dict(finished)
            logs = jsonl(folder/'training_log.jsonl'); diag = jsonl(folder/'geometry_diagnostics.jsonl')
            require([r['local_step'] for r in logs] == list(range(1,1001)) and len(diag) == 8000, 'Training logs/diagnostics')
            for index, log in enumerate(logs):
                require(log['head_status'] == 'FROZEN_NOT_LOADED_OR_CALLED' and log['occupancy_BCE'] is None, 'Unexpected head update')
                group = diag[index*8:(index+1)*8]
                denominator = sum(r['sample_weighted_free_coordinate_denominator'] if arm == 'CLEAN_BULK_FM'
                                  else r['sample_free_coordinate_denominator'] for r in group)
                require(math.isclose(log['loss_denominator'],denominator,rel_tol=1e-12), 'Whole effective batch denominator')
                num = sum(r['quality_weighted_velocity_sse'] if arm == 'CLEAN_BULK_FM' else r['all_free_velocity_sse'] for r in group)
                require(math.isclose(log['FM'],num/denominator,rel_tol=4e-6,abs_tol=1e-7), 'Actual weighted FM algebra')
                require(all(v is not None and math.isfinite(v) for v in log['preprocessing_gradient_norms'].values()), 'Detached preprocessing gradients')
            frozen_checks = jsonl(folder/'frozen_state_checks.jsonl')
            require([r['local_step'] for r in frozen_checks] == list(range(100,1001,100)), 'Required checkpoint/frozen checks')
            require(all(r['status'] == 'PASS' and r['actual'] == r['initial'] for r in frozen_checks), 'Frozen block changed')
            for key in ('checkpoint','inference_export'):
                self.hashes({status[key]['path']:status[key]['sha256']}, 'Wave2 '+arm+' endpoint')
        require(len(set(sigs.values())) == 1, 'Actual input streams not paired')
        # Completed Wave1 scoring is a required predecessor, not a score-based choice.
        wave1done = OUT/'results/wave1/completion.json'
        require(read(wave1done)['rows'] == 504, 'Wave1 full six-model matrix required before training')
        first_calls = [jsonl(OUT/'training'/a/'calls.jsonl')[0]['time_unix'] for a in TRAIN]
        require(wave1done.stat().st_mtime <= min(first_calls), 'Wave2 began before Wave1 scoring completion')
        if self.tensors:
            self.training_tensors(reg, firsts)
        else:
            return dict(status='INCOMPLETE', budgets=budgets, reason='--tensors required for actual optimizer/frozen endpoint verification')
        return dict(status='PASS', budgets=budgets, exposure_signature=sigs,
                    exact_common_8000=True, actual_full_batch_loss_verified=True, partial_optimizer_and_frozen_tensors=True)

    def training_tensors(self, reg, firsts):
        import torch
        torch.set_num_threads(1)
        require(not torch.cuda.is_initialized(), 'Audit must not initialize CUDA')
        source = torch.load(resolve(reg['source_checkpoint']),map_location='cpu',weights_only=True,mmap=True)
        require(state_hash(source['generator_state']) == reg['source_generator_sha256']
                and tree_hash(source['optimizer']) == reg['source_optimizer_sha256'], 'Actual source tensor state')
        names = source['optimizer_parameter_names'][0]
        ids = source['optimizer']['param_groups'][0]['params']
        require(len(names) == len(ids) == len(set(names)) == len(set(ids)) == 194, '194 exact source parameter identities')
        byname = dict(zip(names,ids))
        for arm in TRAIN:
            init = firsts[arm]
            expected_names = [n for n in names if not (arm == 'SURGICAL_LATE4' and frozen(n.removeprefix('generator.')))]
            require(init['trainable_parameter_names'] == expected_names, 'Trainable optimizer names')
            actual_names = init['optimizer_restore']['retained_moment_hashes']
            require(set(actual_names) == set(expected_names), 'Retained moments name set')
            for name in expected_names:
                entry = source['optimizer']['state'][byname[name]]
                require(float(entry['step']) == 13000 and tree_hash(entry) == actual_names[name], 'Retained moment/step equality: '+name)
            status = read(OUT/'training'/arm/'status.json')
            endpoint = torch.load(resolve(status['checkpoint']['path']),map_location='cpu',weights_only=True,mmap=True)
            require(endpoint['optimizer_parameter_names'] == [expected_names] and endpoint['local_step'] == 1000
                    and endpoint['global_step'] == 14000 and endpoint['EMA'] is False, 'Actual endpoint identity')
            require(state_hash(endpoint['generator_state']) == endpoint['generator_state_sha256']
                    and tree_hash(endpoint['optimizer']) == endpoint['optimizer_state_sha256'], 'Endpoint internal hashes')
            require(all(float(v['step']) == 14000 for v in endpoint['optimizer']['state'].values()), 'Final AdamW steps')
            require(len(endpoint['optimizer']['state']) == len(expected_names), 'No optimizer state for frozen parameters')
            if arm == 'SURGICAL_LATE4':
                ff = {n:v for n,v in source['generator_state'].items() if frozen(n)}
                require(state_hash(ff) == init['initial_frozen_state_sha256'] == endpoint['frozen_state_sha256'], 'Frozen subset state hash')
                for name, value in ff.items():
                    require(torch.equal(value.view(torch.uint8),endpoint['generator_state'][name].view(torch.uint8)), 'Frozen tensor bitwise changed: '+name)
            exported = torch.load(resolve(status['inference_export']['path']),map_location='cpu',weights_only=True,mmap=True)
            require(state_hash(exported['generator_state']) == endpoint['generator_state_sha256'], 'Export/final state equality')
            del endpoint, exported
        del source
        require(not torch.cuda.is_initialized(), 'CPU tensor checks initialized CUDA')

    def generation(self, wave):
        import numpy as np
        arms = WAVES[wave]; total = 28*len(arms)
        reg = self.registered(OUT/('generation_'+wave+'_registration.json'), ('protected_files',))
        go = self.registered(OUT/('generation_'+wave+'_dispatch_go.json'), ('code_hashes',))
        require(go['authorized'] and reg['arms'] == list(arms) and reg['requests'] == self.qq, 'Generation registration identity')
        self.hashes({str(OUT/('generation_'+wave+'_registration.json')):go['registration_sha256']}, 'Generation release')
        require(reg['budget']['RAW_outputs'] == total and reg['budget']['generator_forwards'] == 50*total
                and reg['budget']['D0_forwards'] == total and reg['budget']['head_forwards'] == 0, 'Wave generation budget')
        raw_maps, barriers, raw_times, d0_times, anchors = {}, [], [], [], []
        for arm in arms:
            root = OUT/'evaluation'/arm
            rr = self.receipt(root/'raw_complete.json','RAW aggregate')
            pp = self.receipt(root/'post_complete.json','POST aggregate')
            require(rr['outputs'] == pp['outputs'] == 28 and rr['generator_forwards'] == 1400 and pp['D0_forwards'] == 28, 'Arm budget')
            saved = [r for r in jsonl(root/'ledger.jsonl') if r['event'] == 'RAW_SAVED']
            require(Counter(r['request_id'] for r in saved) == Counter(self.requests.keys()), 'Unique28 RAW save ledger')
            raw_times.extend(float(r['unix']) for r in saved)
            for q in self.qq:
                base = root/q['request_id']; rp=base/'raw/receipt.json'; pp=base/'post/receipt.json'
                raw = self.receipt(rp,'RAW files'); post = self.receipt(pp,'POST files')
                require(raw['status'] == post['status'] == 'COMPLETE' and raw['arm'] == post['arm'] == arm
                        and raw['request_id'] == post['request_id'] == q['request_id'], 'Generation receipt identity')
                require((raw['generator_forwards'],raw['head_forwards'],raw['D0_forwards']) == (50,0,0), 'RAW counters')
                require((post['D0_forwards'],post['GEO_calls'],post['orientation_solves']) == (1,1,1), 'Single official D0/GEO/orientation')
                require(raw['source_labels_read'] is False and post['source_labels_used'] is False, 'Inference source labels forbidden')
                require(raw['checkpoint']['sha256'] == reg['models'][arm]['sha256'], 'Frozen generator source')
                require(resolve(raw['layout_path']) == resolve(reg['layouts'][q['condition_id']]['path']), 'Exact LONG layout')
                self.hashes({q['noise_npz']:raw['noise_sha256'],q['inference_npz']:raw['input_bundle_sha256'],
                             raw['layout_path']:raw['layout_sha256']}, 'Frozen noise/C/layout')
                calls=jsonl(base/'raw/calls.jsonl')
                require(len(calls)==100,'50 forward begin/end pairs')
                for i in range(50):
                    a,b=calls[i*2:i*2+2]
                    require(a['event']=='FORWARD_STARTED' and b['event']=='FORWARD_RETURNED'
                            and a['euler_step']==b['euler_step']==i, 'Ordered Euler50 calls')
                dc=jsonl(base/'post/calls.jsonl')
                require([r['event'] for r in dc]==['D0_FORWARD_STARTED','D0_FORWARD_RETURNED'], 'One actual D0 call')
                d0_times.append(float(dc[0]['unix'])); self.latest_work=max(self.latest_work,float(dc[-1]['unix']))
                raw_maps[str(rp.resolve())]=sha(rp)
                a=arrays(base/'raw/raw.npz'); tr=arrays(base/'raw/trajectory.npz')['state']
                fixed=a['raw_anchor_ids']>=0; target=a['C_vertices'][a['raw_anchor_ids'][fixed]]
                require(tr.shape==(51,q['N'],3,3) and tr.dtype==np.float32 and np.isfinite(tr).all(), '51 finite FP32 states')
                require(same(tr[-1],a['raw']) and all(same(t[fixed],target) for t in tr), 'Actual51 anchor coordinate bytes')
                layout=arrays(raw['layout_path']); require(np.array_equal(layout['anchor_ids'],a['raw_anchor_ids']), 'Anchor identity vs layout')
                noise=arrays(q['noise_npz']); z=noise.get('noise',noise.get('z'))
                require(z is not None, 'Recognized frozen Gaussian key')
                initial=np.zeros_like(a['raw']); initial[q['K']:]=z; initial[fixed]=target
                require(same(initial,tr[0]), 'Actual standard Gaussian starting state')
                pre=arrays(base/'post/pre_geo.npz')
                require(same(pre['output'][fixed],target), 'D0 exact anchor restoration')
                po=arrays(base/'post/post.npz'); self.post_identity(a,po,q['K'])
                require(post['anchors']['final_anchor_vertex_identity_verified'], 'Anchor identity receipt')
                source=read(base/'post/source_restoration.json')
                require(source['status']=='PASS' and source['source_C_order_fixed_after_restoration']
                        and source['source_C_vertex_ID_topology']['status']=='PASS', 'Source C winding/topology')
                anchors.append(dict(arm=arm,request_id=q['request_id'],states=51,fixed_corners=int(fixed.sum()),status='PASS'))
            bp=list(root.glob('post_attempt_*/all_RAW_barrier.json')); require(bp,'Recorded allRAW barrier')
            for p in bp:
                b=read(p); require(b['status']=='PASS' and b['RAW']==total and len(b['files'])==total,'Wave allRAW barrier size')
                self.hashes(b['files'],'AllRAW barrier'); barriers.append({str(resolve(k)):v for k,v in b['files'].items()})
        require(all(b==raw_maps for b in barriers),'Barrier exact same wave RAW receipts')
        require(max(raw_times)<min(d0_times),'D0 began before wave RAW completed')
        actual={str(p.resolve()) for arm in arms for s in ('raw','post') for p in (OUT/'evaluation'/arm).glob('*/'+s+'/calls.jsonl')}
        expected={str((OUT/'evaluation'/a/q/s/'calls.jsonl').resolve()) for a in arms for q in self.requests for s in ('raw','post')}
        require(actual==expected,'Extra/missing calls files')
        return dict(status='PASS',RAW=total,generator_forwards=50*total,D0_forwards=total,head_forwards=0,
                    all_RAW_before_any_D0=True,last_RAW_saved=max(raw_times),first_D0_started=min(d0_times),
                    actual_anchor_state_checks=51*total,anchor_checks=anchors)

    @staticmethod
    def post_identity(raw, post, k):
        import numpy as np
        require(same(post['vertices'][post['faces'][:k]],raw['C']), 'POST source-ordered C exact bytes')
        mapping={}
        for src,dst in zip(raw['C_faces_original'].reshape(-1),post['faces'][:k].reshape(-1)):
            require(int(src) not in mapping or mapping[int(src)]==int(dst), 'Source C vertex split')
            mapping[int(src)]=int(dst)
        require(len(set(mapping.values()))==len(mapping),'C vertex IDs merged')
        ids=raw['raw_anchor_ids'][post['source_face_ids'][:,None],post['source_corner_ids']]
        fixed=ids>=0
        expected=np.full(ids.shape,-1,np.int64)
        for src,dst in mapping.items(): expected[ids==src]=dst
        require(np.array_equal(post['faces'][fixed],expected[fixed]), 'Shared retained seam/C anchor IDs')

    def scores(self,wave):
        arms=WAVES[wave]; n=28*len(arms)
        reg=self.registered(OUT/'evaluation'/(wave+'_score_registration.json'),('protected','source_hashes'))
        done=read(OUT/'evaluation'/(wave+'_score_completion.json'))
        summary=self.receipt(OUT/'evaluation'/(wave+'_summary_completion.json'),'124-field score summaries')
        require(done['status']=='COMPLETE' and done['cells']==2*n and len(done['receipts'])==2*n,'RAW/POST score count')
        require(summary['rows']==2*n and summary['numeric_field_count']==124 and len(set(summary['numeric_fields']))==124,'124 metrics')
        identities=[]
        for item in done['receipts']:
            self.hashes({item['path']:item['sha256']},'Score receipt')
            r=self.receipt(item['path'],'Scored input/files'); identities.append((r['label'],r['request_id'],r['stage']))
        require(Counter(identities)==Counter({(a,q,s):1 for a in arms for q in self.requests for s in ('RAW','POST')}),'Complete score identities')
        br=self.registered(OUT/'binding'/wave/'registration.json',('protected','code'))
        bd=self.receipt(OUT/'binding'/wave/'completion.json','Binding tables')
        require(br['arms']==list(arms) and br['deltas']==[.015] and bd['cells']==n and bd['status']=='COMPLETE','Binding cells/threshold')
        require(bd['model_calls']==bd['GPU_calls']==bd['new_queries']==0,'No extra binding model/query calls')
        cells=list((OUT/'binding'/wave/'cells').glob('*/*/complete.json')); require(len(cells)==n,'No extra binding cells')
        identities=[]
        for p in cells:
            cell=self.receipt(p,'Binding cell');require(cell['delta']==.015,'Binding threshold')
            identities.append((cell['arm'],cell['request_id']))
        require(Counter(identities)==Counter({(a,q):1 for a in arms for q in self.requests}),'Binding identity matrix')
        return dict(status='PASS',RAW_POST_cells=2*n,BOUND015_cells=n,numeric_fields=124,extra_model_calls=0)

    def results(self):
        r=self.receipt(OUT/'results/final/completion.json','Final result tables')
        require(r['rows']==672 and r['numeric_fields']==124 and tuple(r['labels'])==LABELS,'672x124 final matrix')
        self.hashes(r['source_scores'],'Reused original/new scores')
        data=rows(OUT/'results/final/per_output.csv'); require(len(data)==672,'Final row count')
        expected={(a,q,s):1 for a in LABELS for q in self.requests for s in STAGES}
        require(Counter((x['label'],x['request_id'],x['stage']) for x in data)==Counter(expected),'Unique final identity matrix')
        fields=r['numeric_field_names'];require(len(fields)==len(set(fields))==124,'Metric columns')
        missing=Counter()
        for row in data:
            q=self.requests[row['request_id']]
            require(row['split']==q['split'] and row['parent_id']==q['parent_id'] and int(row['noise_index'])==q['noise_index'],'Final request metadata')
            for field in fields:
                require(field in row,'Missing metric column: '+field)
                v=row[field]
                if v in ('','None','null','NA'):missing[field]+=1
                elif v not in ('True','False','true','false'):require(math.isfinite(float(v)),'Unexplained nonfinite metric: '+field)
        new_calls={str(p.resolve()) for s in ('raw','post') for p in (OUT/'evaluation').glob('*/*/'+s+'/calls.jsonl')}
        expected_calls={str((OUT/'evaluation'/a/q/s/'calls.jsonl').resolve()) for a in ARMS for q in self.requests for s in ('raw','post')}
        require(new_calls==expected_calls,'No extra new generation arms/requests')
        require({p.name for p in (OUT/'training').iterdir() if p.is_dir()}==set(TRAIN),'No extra trained arms')
        lookup={(r['label'],r['request_id'],r['stage']):r for r in data}
        paired=rows(OUT/'results/final/paired_per_noise.csv')
        primary=[]
        for row in paired:
            a=lookup[row['baseline'],row['request_id'],row['stage']][row['metric']]
            b=lookup[row['candidate'],row['request_id'],row['stage']][row['metric']]
            if row['status']=='PASS':
                def number(v):
                    return float(truth(v)) if v in ('True','False','true','false') else float(v)
                x,y=number(a),number(b)
                require(math.isclose(number(row['baseline_value']),x,rel_tol=1e-12,abs_tol=1e-12)
                        and math.isclose(number(row['candidate_value']),y,rel_tol=1e-12,abs_tol=1e-12)
                        and math.isclose(float(row['delta']),y-x,rel_tol=1e-10,abs_tol=1e-12), 'Paired delta not original cell difference')
            else:
                require(row['delta'] in ('','None','null'),'NA paired metric must not invent delta')
            if row['purpose']=='PRIMARY_EQUAL_1000_UPDATES':primary.append(row)
        require(len(primary)==2*28*3*124
                and all(r['baseline']=='PLAIN_S1000' and r['candidate'] in WAVES['wave2'] for r in primary),
                'Fixed equal1000-update primary comparison')
        return dict(status='PASS',rows=672,numeric_fields=124,models=8,requests=28,stages=3,
                    paired_deltas_verified=len(paired),equal_budget_primary_pairs=len(primary),
                    NA_cells_by_metric=dict(missing),interpretation='NA preserved; original metric/status definitions retained, no re-scoring')

    def visuals(self):
        pages={}
        for wave in WAVES:
            kind=wave+'_POST';manifest=read(OUT/'gallery_manifests'/(kind+'.json'))
            complete=read(OUT/'gallery'/kind/'completion.json')
            require(len(manifest['pages'])==len(complete['images'])==28,'28 all-request POST pages per wave')
            byid={r['page_id']:r for r in manifest['pages']}
            for image in complete['images']:
                p=str(resolve(image['path'])); page=byid[image['page_id']]
                pages[p]=dict(image=image,page=page,wave=wave)
                self.hashes({p:image['image_sha256']},'Actually reviewed POST gallery')
        rr=rows(self.visual_file)
        selected=[r for r in rr if r['label'] in ARMS and r['stage']=='POST']
        expected={(a,q):1 for a in ARMS for q in self.requests}
        require(Counter((r['label'],r['request_id']) for r in selected)==Counter(expected),'Actual all140 new POST review identities')
        for row in selected:
            p=str(resolve(row['page_path']));require(p in pages,'Review page not registered')
            page=pages[p]
            require(row['page_sha256']==page['image']['image_sha256'] and row['request_id']==page['page']['request_id'],'Review image hash/identity')
            require(row['label'] in {c['label'] for c in page['page']['cells']},'Reviewed label absent from image')
            require(truth(row['actually_viewed']) and row['reviewer']=='AI_PROXY_NONBLIND' and row['note'].strip(),'Actual nonblind review declaration')
            for key in ('body_complete','C_adopted_as_seat','obvious_fragments_or_folded_sheets','visible_cracks'):
                require(row[key] in ('YES','NO','UNCERTAIN'),'Visual categorical field: '+key)
        self.hashes({str(self.visual_file):sha(self.visual_file)},'Canonical actual visual records')
        return dict(status='PASS',new_POST_reviews=140,POST_pages=56,extra_review_rows=len(rr)-140,
                    reviewer='AI_PROXY_NONBLIND',limit='Verifies declarations, image identity and coverage, not the truth of subjective judgements')

    def verify_hashes(self):
        checked, deferred, failures=[],[],[];bytes_read=0;reused=0
        for path,expected in sorted(self.expected.items()):
            p=Path(path)
            if not p.exists():failures.append(dict(path=path,reason='MISSING'));continue
            stat=p.stat()
            old=self.retained.get(path)
            if old and path not in self.core_large and old['sha256']==expected:
                # Windows file timestamps are integer100ns ticks from year1.
                old_ns=(int(old['last_write_utc_ticks'])-621355968000000000)*100
                if stat.st_size==int(old['bytes']) and abs(stat.st_mtime_ns-old_ns)<100:
                    checked.append(dict(path=path,sha256=expected,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                        verification='STAGE0_TWO_ACTUAL_SHA_CHECKS_PLUS_UNCHANGED_STAT',
                        scopes=sorted(self.scopes[path]),limitation='No third complete byte read of unrelated protected history'))
                    reused+=1;continue
            if stat.st_size>64<<20 and not self.verify_large:
                deferred.append(dict(path=path,bytes=stat.st_size,reason='--verify-large required for final actual SHA'));continue
            actual=sha(p);bytes_read+=stat.st_size
            if actual!=expected:failures.append(dict(path=path,expected=expected,actual=actual));continue
            checked.append(dict(path=path,sha256=actual,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                                verified_unix=time.time(),verification='DIRECT_CURRENT_SHA256',scopes=sorted(self.scopes[path])))
        return dict(status='FAIL' if failures else 'INCOMPLETE' if deferred else 'PASS',
                    unique_paths=len(self.expected),verified=len(checked),files=checked,
                    failed=failures,deferred=deferred,bytes_read=bytes_read,duplicate_hash_scans=0,
                    retained_history_double_hash_receipts_reused=reused)


def finalize_nonvisual(report_path, destination, visual_file=None, repair_results=False):
    """Add actual visual coverage without repeating completed tensor/disk scans.

    The immutable earlier report must contain every nonvisual section and pass.
    Every previously verified file is checked for unchanged size/mtime; a changed
    timestamp requires a new actual SHA. New visual assets are directly hashed.
    This reuses completed checks, never interprets a pending section as a pass.
    """
    source_path = resolve(report_path)
    previous = read(source_path)
    expected_sections = set(SECTIONS) - {'visuals'}
    require(set(previous['checks']) == expected_sections | {'hashes'},
            'All nonvisual sections must have been attempted')
    if repair_results:
        require(previous['status']=='FAIL'
                and previous['checks']['results']==dict(status='FAIL',error='ValueError("could not convert string to float: \'True\'")')
                and all(v['status']=='PASS' for k,v in previous['checks'].items() if k!='results'),
                'Only the exact paired-table boolean-parser error may be repaired')
    else:
        require(previous['status']=='PASS_SELECTED_STAGES'
                and all(v['status']=='PASS' for v in previous['checks'].values()),
                'Complete passing nonvisual evidence required')
    unchanged, rehashed = 0, []
    for item in previous['checks']['hashes']['files']:
        # These paths were already canonicalized by the immediately preceding
        # audit on this same host. Avoid thousands of redundant WSL realpath
        # traversals; the actual file stat is still checked for every entry.
        path = Path(item['path'])
        if not path.is_absolute():
            path = resolve(item['path'])
        stat = path.stat()
        if stat.st_size == item['bytes'] and stat.st_mtime_ns == item['mtime_ns']:
            unchanged += 1
        else:
            actual = sha(path)
            require(actual == item['sha256'], 'Evidence changed since nonvisual audit: '+str(path))
            rehashed.append(dict(path=str(path),sha256=actual,bytes=stat.st_size,
                                 mtime_ns=stat.st_mtime_ns))
    audit = Audit(verify_large=False,tensors=False,visual_file=visual_file)
    repaired_results = audit.results() if repair_results else None
    # Protocol and protected weights were already directly verified in the
    # completed nonvisual check; only new review/gallery paths are registered.
    prior = {r['path']:r['sha256'] for r in previous['checks']['hashes']['files']}
    require(all(prior.get(p) == h for p,h in audit.expected.items()), 'Protected manifest differs')
    audit.expected.clear(); audit.scopes.clear()
    visual = audit.visuals(); visual_hashes = audit.verify_hashes()
    require(visual['status'] == visual_hashes['status'] == 'PASS', 'Visual verification incomplete')
    result = dict(previous)
    result.update(status='PASS',scope='FULL',utc=datetime.now().astimezone().isoformat(),
                  source_sha256=sha(__file__))
    result['checks'] = dict(previous['checks'])
    if repaired_results is not None:
        result['checks']['results'] = repaired_results
    result['checks']['visuals'] = visual
    result['checks']['visual_hashes'] = visual_hashes
    result['nonvisual_reuse'] = dict(path=str(source_path),sha256=sha(source_path),
        source_sha256=previous['source_sha256'],unchanged_file_stats=unchanged,
        rehashed_after_stat_change=rehashed,
        repaired_results_boolean_parser=repair_results,
        preserved_original_error=previous['checks']['results'].get('error') if repair_results else None,
        reason='Avoid duplicate large tensor and disk reads immediately after actual terminal audit; no new model/scorer calls')
    with destination.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status='PASS',scope='FULL',path=str(destination),
                         nonvisual_report=str(source_path),new_POST_reviews=140,
                         model_calls=0,scorer_calls=0),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--sections',nargs='+',choices=SECTIONS,default=list(SECTIONS))
    p.add_argument('--verify-large',action='store_true')
    p.add_argument('--tensors',action='store_true',help='CPU mmap actual state/optimizer/WISE checks; zero networks')
    p.add_argument('--visual-file')
    p.add_argument('--finalize-nonvisual',help='Complete visual check and hash-bound reuse of a passing immutable nonvisual report')
    p.add_argument('--repair-results-boolean-parser',action='store_true',help='Rerun only result algebra after the exact recorded True/False conversion failure')
    p.add_argument('--output')
    args=p.parse_args()
    if not args.run:
        print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',sections=args.sections,
            intended=dict(RAW=140,generator_forwards=7000,D0=140,updates_each=1000,training_F_B_each=8000,
                          final_rows=672,metrics=124,new_POST_actual_reviews=140,cleanup_deleted=16,retained=36),
            model_constructions=0,model_calls=0),indent=2));return
    destination=resolve(args.output or AUDIT/('audit_'+str(time.time_ns())+'.json'))
    require(destination.is_relative_to(AUDIT.resolve()) and not destination.exists(),'New immutable audit destination only')
    if args.finalize_nonvisual:
        finalize_nonvisual(args.finalize_nonvisual,destination,args.visual_file,args.repair_results_boolean_parser)
        return
    result=dict(status='INCOMPLETE',scope='FULL' if set(args.sections)==set(SECTIONS) else 'SELECTED_STAGES',
                utc=datetime.now().astimezone().isoformat(),source_sha256=sha(__file__),
                model_constructions=0,model_calls=0,scorer_calls=0,geometry_calls=0,new_OT_calls=0,checks={})
    audit=None
    try:
        audit=Audit(args.verify_large,args.tensors,args.visual_file)
        for section in args.sections:
            print('AUDIT_SECTION',section,flush=True)
            try:
                if section.startswith('generation_'):value=audit.generation(section.removeprefix('generation_'))
                elif section.startswith('scores_'):value=audit.scores(section.removeprefix('scores_'))
                else:value=getattr(audit,section)()
            except FileNotFoundError as exc:value=dict(status='INCOMPLETE',missing=str(exc))
            except (OSError,ValueError,KeyError,IndexError,TypeError,AssertionError) as exc:value=dict(status='FAIL',error=repr(exc))
            result['checks'][section]=value
        result['checks']['hashes']=audit.verify_hashes()
        statuses=[v['status'] for v in result['checks'].values()]
        result['status']='FAIL' if 'FAIL' in statuses else 'INCOMPLETE' if any(s!='PASS' for s in statuses) else 'PASS' if result['scope']=='FULL' else 'PASS_SELECTED_STAGES'
    except (OSError,ValueError,KeyError,IndexError,TypeError) as exc:
        result.update(status='FAIL',error=repr(exc))
    result['interpretation']='Execution and evidence closure only; no superiority, promotion, extension or new action inferred.'
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    if not result['status'].startswith('PASS'):raise SystemExit(2)


if __name__=='__main__':main()
