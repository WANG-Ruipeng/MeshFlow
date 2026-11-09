"""Original 124-field scoring; one frozen finite score matrix per wave."""
from __future__ import annotations
import argparse
import json
import types
from pathlib import Path
from r2.pilots.seam_abc_v1 import evaluate as E
from r2.pilots.seam_pool47_couple_v1.endpoint_scoring import metric_row
from .evaluation import C, OUT, ARMS, WAVES, private_functions, requests, dispatch_cpu, source_paths

VERSION = 'seam_quality_wave12_original_124_fields_v1'
_common = types.SimpleNamespace(**vars(C)); _common.ARMS = ARMS
API = private_functions(E, C=_common)


def files(wave):
    if wave not in WAVES:
        raise ValueError('Unknown wave')
    prefix = OUT / 'evaluation'
    return {name: prefix / (wave + '_' + name + '.json')
            for name in ('score_registration', 'score_completion', 'summary_completion')}


def prepare(wave):
    paths = files(wave)
    if paths['score_registration'].exists():
        reg = C.read(paths['score_registration'])
        E.verify(reg['protected']); E.verify(reg['source_hashes'])
        return reg
    qq = requests()
    generation = OUT / ('generation_' + wave + '_registration.json')
    reg = C.read(generation)
    if reg['arms'] != list(WAVES[wave]):
        raise ValueError('Wave generation contract mismatch')
    protected = {str(generation): C.sha(generation)}
    for q in qq:
        for key in ('inference_npz', 'metadata_npz', 'noise_npz'):
            protected[q[key]] = C.sha(q[key])
        desc = q['query_descriptor']
        if C.sha(desc['path']) != desc['sha256']:
            raise ValueError('Queries changed')
        protected[desc['path']] = desc['sha256']
        for arm in WAVES[wave]:
            for stage in ('RAW', 'POST'):
                mesh, receipt = E.artifact(OUT, arm, q['request_id'], stage)
                protected[str(mesh)] = C.sha(mesh); protected[str(receipt)] = C.sha(receipt)
    code = {**E.source_hashes(), str(Path(metric_row.__code__.co_filename)): C.sha(metric_row.__code__.co_filename)}
    code.update({str(p): C.sha(p) for p in (*source_paths(), Path(__file__))})
    result = dict(status='FROZEN', version=VERSION, wave=wave, arms=list(WAVES[wave]), requests=qq,
        cells=len(WAVES[wave])*28*2, stages=['RAW', 'POST'], protected=protected, source_hashes=code,
        scorer='Unmodified evaluate.score_one in private namespace; only applicable arm labels extended',
        quality_definition=E.helpers()[0].cpu_math()['control'].frozen_definition(),
        model_calls=0, new_queries=0, baseline_scans=0)
    C.write(paths['score_registration'], result)
    return result


def worker(job):
    return API['score_one'](job)


def run(wave, workers=2):
    reg = prepare(wave); paths = files(wave)
    if paths['score_completion'].exists():
        done = C.read(paths['score_completion'])
        for r in done['receipts']:
            E.verify({r['path']: r['sha256']}); E.verify(C.read(r['path'])['files'])
        return done
    registration_sha = C.sha(paths['score_registration'])
    jobs = [(str(OUT), arm, q, stage, registration_sha)
            for q in reg['requests'] for arm in WAVES[wave] for stage in ('RAW', 'POST')]
    receipts, count, seconds = dispatch_cpu(jobs, worker, OUT/'evaluation'/wave, workers)
    E.verify(reg['protected']); E.verify(reg['source_hashes'])
    receipts.sort(key=lambda r: r['path'])
    if len(receipts) != reg['cells']:
        raise ValueError('Incomplete score matrix')
    done = dict(status='COMPLETE', wave=wave, cells=len(receipts), receipts=receipts,
        workers=count, seconds=seconds, registration_sha256=registration_sha,
        full_quality_scans=sum(not r['reused'] for r in receipts), model_calls=0, new_queries=0)
    C.write(paths['score_completion'], done)
    return done


def summarize(wave):
    paths = files(wave); complete = C.read(paths['score_completion'])
    if complete['cells'] != len(WAVES[wave])*56:
        raise ValueError('Wave not fully scored')
    rows, roles, deleted = [], [], []
    for q in requests():
        for arm in WAVES[wave]:
            for stage in ('RAW', 'POST'):
                folder = OUT/'evaluation/scores'/arm/q['request_id']/stage
                score = C.read(folder/'score.json'); E.verify(score['files'])
                row, role, drop = metric_row(score, C.arrays(score['identity']['mesh_path']), C.read(folder/'pairs.json'))
                rows.append(row); roles.extend(role); deleted.extend(drop)
    numeric = sorted(k for k in set().union(*(r.keys() for r in rows))
        if k not in (*E.META, 'label', 'stage') and any(isinstance(r.get(k), (float, int, bool)) for r in rows))
    if len(numeric) != 124:
        raise ValueError('Original 124 numeric field contract changed')
    outputs = {}
    for name, rr in (('per_output', rows), ('illegal_pairs_by_role', roles), ('deleted_faces', deleted)):
        path = OUT/'evaluation'/(wave+'_'+name+'.csv'); C.csvwrite(path, rr); outputs[str(path)] = C.sha(path)
    result = dict(status='COMPLETE', wave=wave, rows=len(rows), numeric_fields=numeric,
        numeric_field_count=124, source_completion_sha256=C.sha(paths['score_completion']),
        model_calls=0, new_scans=0, files=outputs)
    C.write(paths['summary_completion'], result)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare','score','summarize'))
    parser.add_argument('--wave', choices=tuple(WAVES), required=True)
    parser.add_argument('--workers', type=int, choices=(1,2), default=2); args=parser.parse_args()
    result=prepare(args.wave) if args.stage=='prepare' else run(args.wave,args.workers) if args.stage=='score' else summarize(args.wave)
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('protected','source_hashes','requests','receipts','numeric_fields','files')}),flush=True)


if __name__ == '__main__':
    main()
