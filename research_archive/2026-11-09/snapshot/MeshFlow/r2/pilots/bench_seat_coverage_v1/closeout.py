"""Read-only, CPU-only final evidence audit. Never marks the experiment complete.

Run after training, sampling, scoring, rendering AND actual review finish:
    python -m r2.pilots.bench_seat_coverage_v1.closeout --write
Missing final artifacts return PENDING before hashing large files. No torch,
model loading, regeneration, rating generation, rescoring or automatic retries.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
import statistics
import time

from .common import OUT, ROOT, SOURCE_ROOT, read, resolve, sha, write_new

LABELS = ('NATIVE_S6000', 'COVERAGE_FINAL', 'SEAM_S3000')
STAGES = ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')
JOBS = ('train', 'raw-native', 'raw-candidate', 'd0-native', 'd0-candidate',
        'score-raw', 'score-post', 'collect', 'gallery', 'pipeline')
REVIEW = OUT / 'visual_review/summary_v1'


class AuditFailure(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def lines(path):
    with resolve(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def table(path):
    with resolve(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def truth(value):
    return str(value).lower() in ('true', '1')


def event_time(row):
    if 'time_unix' in row:
        return float(row['time_unix'])
    if 'unix' in row:
        return float(row['unix'])
    return datetime.fromisoformat(row['utc'].replace('Z', '+00:00')).timestamp()


def stats(rows):
    fm = [float(r['FM']) for r in rows]
    den = sum(int(r['all_free_scalar_count']) for r in rows)
    return dict(updates=len(rows), FM_mean=statistics.mean(fm),
                FM_median=statistics.median(fm), FM_min=min(fm), FM_max=max(fm),
                coordinate_weighted_velocity_MSE=sum(float(r['all_free_velocity_sse']) for r in rows)/den,
                coordinate_weighted_endpoint_MSE=sum(float(r['all_free_endpoint_sse']) for r in rows)/den,
                input_seconds=sum(float(r['input_seconds']) for r in rows),
                update_seconds=sum(float(r['update_seconds']) for r in rows))


class Audit:
    def __init__(self):
        self.hashes = {}
        self.checks = []

    def digest(self, path):
        path = resolve(path).resolve()
        if str(path) not in self.hashes:
            self.hashes[str(path)] = sha(path)
        return self.hashes[str(path)]

    def verify(self, values, base=None):
        for path, expected in values.items():
            p = base / path if base is not None else resolve(path)
            require(self.digest(p) == expected, 'SHA256 mismatch: ' + str(p))

    def passed(self, name):
        self.checks.append(name)

    def prerequisites(self):
        expected = [(OUT/'training/NATIVE/status.json', 'COMPLETE'),
                    (OUT/'results/completion.json', 'COMPLETE'),
                    (OUT/'gallery/completion.json', 'RENDER_COMPLETE_PENDING_ACTUAL_REVIEW'),
                    (REVIEW/'completion.json', 'COMPLETE')]
        expected += [(OUT/'workflow'/job/'status.json', 'COMPLETE') for job in JOBS]
        expected += [(OUT/'evaluation'/label/name, 'COMPLETE') for label in LABELS[:2]
                     for name in ('raw_complete_full.json', 'post_complete_full.json')]
        pending = []
        for path, status in expected:
            if not path.exists():
                pending.append(dict(path=str(path), reason='MISSING'))
            else:
                actual = read(path).get('status')
                if actual != status:
                    pending.append(dict(path=str(path), reason='NOT_FINAL', status=actual))
        for path in (REVIEW/'actual_observations.csv', OUT/'gallery/manifest.json',
                     OUT/'gallery/definition.json', OUT/'results/per_output.csv'):
            if not path.exists():
                pending.append(dict(path=str(path), reason='MISSING'))
        return pending

    def training(self, reg):
        folder = OUT/'training/NATIVE'
        status = read(folder/'status.json')
        n = reg['updates']
        require(n == 1500 and reg['batch'] == 8, 'This audit is frozen to1500updates/batch8')
        require((status['stage_step'], status['global_step']) == (n, 19500), 'Training endpoint differs')
        require(status['registration_sha256'] == self.digest(OUT/'training_registration.json'), 'Training registration changed')
        require(all(status[k] == 0 for k in ('head_step', 'head_calls', 'probe_calls')), 'Nonzero head/probe activity')
        log = lines(folder/'training_log.jsonl')
        require([r['stage_step'] for r in log] == list(range(1, n+1)), 'Training steps missing/repeated')
        require(all(r['global_step'] == 18000+r['stage_step'] and r['head_calls'] == r['head_step'] == 0
                    and math.isfinite(float(r['FM'])) for r in log), 'Training cursor/FM/head fields differ')
        events = lines(folder/'calls.jsonl')
        begin, done = {}, {}
        for r in events:
            require(r['event'] in ('CALL_BEGIN', 'CALL_DONE'), 'Training failed/unknown call event')
            key = (r['attempt_id'], r['call_id'])
            dest = begin if r['event'] == 'CALL_BEGIN' else done
            require(key not in dest, 'Duplicate training call ID/event')
            dest[key] = r
        require(begin.keys() == done.keys(), 'Unfinished training calls')
        for k, b in begin.items():
            d = done[k]
            require(b['kind'] == d['kind'] and b['context'] == d['context'], 'Call begin/done mismatch')
        counts = Counter(r['kind'] for r in done.values())
        require(counts == Counter(train_forward=12000, train_backward=12000, optimizer_update=1500),
                'Actual training ledger budget differs: '+str(counts))
        for kind in ('train_forward', 'train_backward', 'optimizer_update'):
            observed = Counter((r['context']['stage_step'], r['context'].get('microbatch'))
                               for r in done.values() if r['kind'] == kind)
            target = Counter((s, m) for s in range(1, n+1)
                             for m in (range(8) if kind != 'optimizer_update' else (None,)))
            require(observed == target, 'Per-update call distribution differs: '+kind)
        recorded = status['calls']
        require(recorded['successful'] == dict(counts) and not recorded['pending_call_ids']
                and not any(recorded['failed'].values()), 'Status and call ledger disagree')
        schedule = read(reg['input_schedule'])
        exposure = lines(folder/'exposure.jsonl')
        expected = [(s['stage_step'], m, r) for s in schedule['steps'] for m, r in enumerate(s['samples'])]
        require(len(exposure) == len(expected) == 12000, 'Exposure count differs')
        for actual, (step, micro, row) in zip(exposure, expected):
            require(actual['stage_step'] == step and actual['microbatch'] == micro
                    and all(actual[k] == v for k, v in row.items()), 'Frozen exposure stream mismatch')
            require(actual['head_forward_called'] is False and actual['mechanism']['active_seam_slots'] == 0,
                    'Unexpected head/SEAM training activity')
        group = Counter(r['replay_group'] for r in exposure)
        require(group == Counter(OLD=6000, NEW=6000), 'Old/new exposure differs')
        require(Counter(r['condition_id'] for r in exposure) == Counter(schedule['condition_exposure']),
                'Condition exposure differs')
        loss_groups = {}
        for g in ('OLD', 'NEW'):
            rr = [r for r in exposure if r['replay_group'] == g]
            den = sum(r['all_free_scalar_count'] for r in rr)
            loss_groups[g] = dict(samples=len(rr), parents=len({r['parent_id'] for r in rr}),
                by_parent=dict(sorted(Counter(r['parent_id'] for r in rr).items())),
                by_augmentation=dict(Counter(r['augmentation'] for r in rr)),
                coordinate_weighted_velocity_MSE=sum(r['all_free_velocity_sse'] for r in rr)/den,
                coordinate_weighted_endpoint_MSE=sum(r['all_free_endpoint_sse'] for r in rr)/den)
        # Tensor identities were authenticated by the trainer; this audit never loads weights.
        step0 = read(folder/'checkpoint_step0.json')
        for suffix in ('generator', 'optimizer', 'rng'):
            key = suffix+'_state_sha256' if suffix != 'rng' else 'rng_sha256'
            require(step0[key] == reg['source_'+suffix+'_sha256'], 'Initial tensor identity receipt differs')
        weights = []
        for key in ('checkpoint', 'inference_export'):
            r = status[key]
            require(r['stage_step'] == n and r['global_step'] == 19500, 'Final weight cursor differs')
            require(resolve(r['path']).stat().st_size == r['bytes'], 'Final weight size differs')
            self.verify({r['path']: r['sha256']})
            side = read(resolve(r['path']).with_suffix('.json'))
            require(side == r, 'Final sidecar/status differs')
            weights.append(dict(**r, verification_scope='ACTUAL_FULL_FILE_SHA256_NO_TENSOR_LOAD'))
        self.passed('training1500_forward12000_backward12000_head0_probe0_and_frozen_stream')
        return dict(updates=n, global_step=19500, calls=dict(counts), attempts=sorted({k[0] for k in begin}),
                    first25=stats(log[:25]), last100=stats(log[-100:]), whole_run=stats(log), exposure=loss_groups,
                    loss_scope='Observed random-t training loss, not held-out generation quality or convergence proof',
                    allocator_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in log),
                    allocator_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in log),
                    process_resource_observation=status.get('resources'),
                    final_weights=weights, initialized_from_receipt=step0)

    def evaluation(self, panel):
        require(panel['new_RAW'] == panel['new_D0'] == 36 and panel['generator_forwards'] == 1800
                and panel['total_scoring_rows'] == 252, 'Registered evaluation budget differs')
        expected, new, reused, raw_ends, d0_starts = set(), 0, 0, [], []
        for q in panel['requests']:
            old = q['panel_origin'] == 'INHERITED24'
            labels = LABELS if old else LABELS[:2]
            for label in labels:
                reuse = old and label in (LABELS[0], LABELS[2])
                root = SOURCE_ROOT if reuse else OUT
                base = root/'evaluation'/label/q['request_id']
                raw, post = read(base/'raw/receipt.json'), read(base/'post/receipt.json')
                require(raw['status'] == post['status'] == 'COMPLETE', 'Incomplete output receipt')
                require(raw['label'] == post['label'] == label and raw['request_id'] == post['request_id'] == q['request_id'],
                        'Output identity mismatch')
                require(raw['generator_forwards'] == 50 and raw['D0_forwards'] == 0 and post['D0_forwards'] == 1,
                        'Per-output model call receipt differs')
                require(raw['noise_sha256'] == q['noise_sha256'] and post['raw_sha256'] == raw['raw_sha256'],
                        'Noise/RAW-D0 pairing mismatch')
                self.verify(raw['files']); self.verify(post['files'])
                self.verify({base/'raw/raw.npz': raw['raw_sha256'], base/'post/post.npz': post['post_sha256']})
                for stage in STAGES:
                    expected.add((label, q['request_id'], stage))
                if reuse:
                    reused += 1
                    continue
                new += 1
                require(raw.get('head_forwards', 0) == 0 and raw.get('source_labels_read') is False,
                        'Unexpected head/source-label inference')
                rr = lines(base/'raw/calls.jsonl')
                starts = [r for r in rr if r['event'] == 'FORWARD_STARTED']
                ends = [r for r in rr if r['event'] == 'FORWARD_COMPLETE']
                require(len(rr) == 100 and len(starts) == len(ends) == 50, 'RAW actual ledger budget differs')
                require([r['step'] for r in starts] == [r['step'] for r in ends] == list(range(50)), 'Euler step order differs')
                raw_ends.extend(event_time(r) for r in ends)
                dd = lines(base/'post/calls.jsonl')
                require([r['event'] for r in dd] == ['D0_FORWARD_STARTED', 'D0_FORWARD_RETURNED'], 'D0 actual ledger differs')
                d0_starts.append(event_time(dd[0]))
        require(new == 36 and reused == 48 and len(expected) == 252, 'Logical/reused output counts differ')
        require(max(raw_ends) <= min(d0_starts), 'All-new-RAW-before-D0 time barrier violated')
        for label in LABELS[:2]:
            for name in ('raw_complete_full.json', 'post_complete_full.json'):
                self.verify(read(OUT/'evaluation'/label/name)['files'])
            registration = read(OUT/'evaluation/inference_registrations'/(label+'__D0.json'))
            self.verify(registration['RAW_barrier']); self.verify(registration['code'])
            require(len(registration['RAW_barrier']) == 84, 'Saved D0 RAW barrier omits logical outputs')
        result = read(OUT/'results/completion.json')
        require((result['rows'], result['new_score_rows'], result['reused_score_rows']) == (252, 108, 144), 'Score accounting differs')
        self.verify(result['files']); self.verify(result['source_rows'])
        rows = table(OUT/'results/per_output.csv')
        keys = [(r['label'], r['request_id'], r['stage']) for r in rows]
        require(len(keys) == len(set(keys)) == 252 and set(keys) == expected, 'Score identities differ')
        require(Counter(r['result_origin'] for r in rows) == Counter(REUSED_READ_ONLY=144, NEW_REGISTERED=108), 'Score reuse origin differs')
        self.verify(read(OUT/'evaluation/score_registration.json')['source_hashes'])
        self.passed('36newRAW1800forwards_36newD0_48reused_252scores_allRAWbarrier')
        return expected, dict(new_RAW=new, new_sampling_forwards=1800, new_D0=36,
                              reused_outputs=reused, logical_outputs=84, score_rows=252,
                              new_score_rows=108, reused_score_rows=144, head_forwards=0,
                              latest_new_raw_forward_end=max(raw_ends), earliest_new_D0_start=min(d0_starts))

    def visuals(self, expected):
        manifest = read(OUT/'gallery/manifest.json')
        definition = read(OUT/'gallery/definition.json')
        rendered = read(OUT/'gallery/completion.json')
        require(len(manifest['pages']) == len(rendered['images']) == 90, 'Gallery page count differs')
        require(definition['manifest_sha256'] == self.digest(OUT/'gallery/manifest.json'), 'Gallery manifest changed')
        self.verify(definition['sources']); self.verify(definition['protected'])
        images = {r['page_id']: r for r in rendered['images']}
        cells = {}
        for page in manifest['pages']:
            im = images[page['page_id']]
            self.verify({im['path']: im['image_sha256']})
            receipt_path = resolve(im['path']).with_suffix('.json')
            require(read(receipt_path) == im, 'Image receipt/completion differs')
            require(im['definition_sha256'] == self.digest(OUT/'gallery/definition.json'), 'Image definition differs')
            require(im['cells'] == page['cells'], 'Rendered model columns differ')
            for c in page['cells']:
                key = (c['model'], c['request_id'], c['stage'])
                require(key not in cells, 'Repeated gallery cell')
                cells[key] = (page['page_id'], im, receipt_path)
        require(set(cells) == expected, 'Gallery cell identities differ from scores')
        rows = table(REVIEW/'actual_observations.csv')
        observed = set()
        for r in rows:
            k = (r['label'], r['request_id'], r['stage'])
            require(k not in observed and k in cells, 'Unknown/duplicate actual-review cell')
            observed.add(k)
            page, im, receipt = cells[k]
            require(r['page_id'] == page and r['image_sha256'] == im['image_sha256'], 'Review points to different image')
            require(r['image_receipt_sha256'] == self.digest(receipt)
                    and r['gallery_definition_sha256'] == self.digest(OUT/'gallery/definition.json'), 'Review provenance differs')
            require(r['reviewer_kind'] == 'AI_PROXY' and not truth(r['blind']) and truth(r['actually_viewed'])
                    and truth(r['views_and_sections_checked']), 'Missing actual AI_PROXY view attestation')
            require(all(r[k].strip() for k in ('note', 'viewed_at_utc', 'actual_view_source', 'reviewer')), 'Empty review evidence')
        require(observed == expected and len(rows) == 252, 'Actual review incomplete')
        completion = read(REVIEW/'completion.json')
        v = completion['validation']
        require(v['status'] == 'COMPLETE_ACTUAL_AI_PROXY_REVIEW'
                and v['actual_reviewed_pages'] == 90 and v['actual_reviewed_cells'] == 252
                and v['missing_cells'] == v['generated_ratings'] == 0, 'Review validator differs')
        if 'artifacts' in completion:
            self.verify(completion['artifacts'], REVIEW)
        self.passed('90renderedpages_252actualAI_PROXYcells_with_image_provenance')
        return dict(pages=90, actual_review_cells=252, reviewer_kind='AI_PROXY', blind=False,
                    scope='Verifies recorded actual-view attestations/provenance; hashes alone do not establish visual judgement or independent human review')

    def protection(self, reg, panel):
        self.verify(reg['training_code_hashes']); self.verify(reg['gate_hashes'])
        for key in ('manifest', 'input_schedule', 'eval_manifest', 'source_checkpoint'):
            self.verify({reg[key]: reg[key+'_sha256']})
        self.verify(panel['protected'])
        manifest = read(reg['manifest'])
        for key in ('protected', 'protected_inputs', 'protected_hashes'):
            if isinstance(manifest.get(key), dict): self.verify(manifest[key])
        for row in manifest['train']+manifest.get('dev', []):
            for path_key, hash_key in (('case_path','case_sha256'), ('input_path','input_sha256'),
                                       ('paired_path','paired_sha256'), ('quality_file','quality_sha256')):
                if path_key in row and hash_key in row: self.verify({row[path_key]: row[hash_key]})
        retention_path = ROOT/'r2/maintenance/bench_reference_20261023/retained_assets.json'
        retention = read(retention_path)
        self.verify({retention['source_receipt']: retention['source_receipt_sha256']})
        old_complete = read(retention['experiment_completion'])
        require(old_complete['status'] == 'COMPLETE', 'Old reference is no longer complete')
        self.verify(old_complete['artifacts_sha256'], SOURCE_ROOT)
        protected = []
        for r in retention['preserved_assets']:
            path = resolve(r['path'])
            require(path.stat().st_size == r['bytes'], 'Retained asset size changed: '+str(path))
            scope = 'CURRENT_STAT_PLUS_RETAINED_RECEIPT_AND_ORIGINAL_SIDECAR_NO_LARGE_FILE_REHASH'
            if path == resolve(reg['source_checkpoint']) or r['kind'] == 'head':
                self.verify({path: r['sha256']}); scope = 'ACTUAL_FULL_FILE_SHA256'
            if 'identity_sidecar' in r:
                self.verify({r['identity_sidecar']: r['identity_sidecar_sha256']})
                require(read(r['identity_sidecar'])['sha256'] == r['sha256'], 'Retained sidecar/receipt mismatch')
            protected.append(dict(path=str(path), bytes=path.stat().st_size, recorded_sha256=r['sha256'], verification_scope=scope))
        self.passed('source_checkpoint_fullhash_oldreports_data_codehashes_retainedweights_stat_and_sidecars')
        return dict(retention_receipt=str(retention_path), retention_receipt_sha256=self.digest(retention_path),
                    assets=protected, historical_completion_sha256=self.digest(retention['experiment_completion']),
                    original_report_hashes=old_complete['artifacts_sha256'])

    def resources(self):
        phases = []
        for job in JOBS:
            p = OUT/'workflow'/job/'status.json'
            r = read(p)
            phases.append(dict(job=job, status=r['status'], started=r.get('started'), finished=r.get('finished'),
                               seconds=r.get('seconds'), dispatch=r.get('dispatch'), evidence_path=str(p)))
        observations = []
        # Only fixed sampling observer directories, no recursive experiment scan.
        for label in LABELS[:2]:
            base = OUT/'evaluation'/label
            for pattern in ('sampling_attempt_*/resource_observation.json', 'D0_resource_observer_*/resource_observation.json'):
                for p in sorted(base.glob(pattern)):
                    r = read(p)
                    observations.append(dict(path=str(p), sha256=self.digest(p), summary=r.get('summary'),
                                             sample_count=len(r.get('samples', []))))
        overlap_path = OUT/'controller/training_overlap_decision.json'
        # Root overrides are additive dispatch evidence; preserve observed decisions verbatim.
        dispatch_evidence = []
        for base in (OUT/'dispatch', OUT/'controller'):
            for p in sorted(base.glob('*override*.json')):
                dispatch_evidence.append(dict(path=str(p), sha256=self.digest(p), value=read(p)))
        return dict(workflow_phases=phases, sampler_resource_observations=observations,
                    initial_overlap_decision=read(overlap_path) if overlap_path.exists() else None,
                    additive_dispatch_overrides=dispatch_evidence,
                    timing_scope='Actual phase wall times may overlap; their sum is not elapsed pipeline time or an isolated compute benchmark',
                    memory_scope='Allocator peaks are not process VRAM; WDDM unavailable process peaks remain NA. Total-device peaks include user/background load.')


def run():
    start = time.perf_counter()
    audit = Audit()
    result = dict(schema='BENCH_SEAT_COVERAGE_CLOSEOUT_AUDIT_V1',
                  audited_utc=datetime.now(timezone.utc).isoformat(), status='PENDING',
                  model_calls=0, scoring_calls=0, generated_ratings=0, writes_COMPLETE=False,
                  changes_model_or_default=False)
    try:
        pending = audit.prerequisites()
        if pending:
            result['pending'] = pending
            return result
        reg = read(OUT/'training_registration.json')
        panel = read(OUT/'eval_manifest.json')
        result['training'] = audit.training(reg)
        expected, result['evaluation'] = audit.evaluation(panel)
        result['visual_review'] = audit.visuals(expected)
        result['preservation'] = audit.protection(reg, panel)
        result['resources_and_timings'] = audit.resources()
        result['status'] = 'AUDIT_PASS'
    except FileNotFoundError as exc:
        result.update(status='PENDING', pending=[dict(path=str(exc.filename), reason='MISSING_FINAL_EVIDENCE')])
    except (AuditFailure, KeyError, ValueError, TypeError) as exc:
        result.update(status='AUDIT_FAIL', error=str(exc), error_type=type(exc).__name__)
    finally:
        result.update(checks_passed=audit.checks, verified_file_sha256=audit.hashes,
                      seconds=time.perf_counter()-start, audit_code_sha256=sha(__file__))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true', help='Write an independent audit receipt; never COMPLETE.json')
    a = p.parse_args()
    result = run()
    if a.write:
        path = OUT/'audits'/('closeout_check_'+str(time.time_ns())+'.json')
        write_new(path, result)
        result['audit_receipt'] = str(path)
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('verified_file_sha256','training','evaluation','visual_review','preservation','resources_and_timings')},
        ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['status'] in ('AUDIT_PASS', 'PENDING') else 1)


if __name__ == '__main__':
    main()
