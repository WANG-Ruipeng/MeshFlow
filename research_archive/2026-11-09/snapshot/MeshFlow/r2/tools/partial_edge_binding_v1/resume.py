"""Read-only status and bounded recovery for the completed CPU experiment.

A completed 144-cell experiment is only verified, never decoded or rescored.
Unfinished attempts require explicit engineering recovery, just as run.py does.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
from . import assets as A


def status():
    out=A.OUT
    reg=A.read(out/'registration.json')
    manifest=A.read(out/'input_manifest.json')
    if A.sha(out/'input_manifest.json')!=reg['input_manifest_sha256']:
        raise ValueError('Frozen input manifest changed')
    code=dict(reg['source_sha256']); amendments=[]
    amendment_path=out/'engineering_amendments.json'
    if amendment_path.exists():
        amendments=A.read(amendment_path)['items']
        for item in amendments:
            if code.get(item['path'])!=item['old_sha256']:
                raise ValueError('Engineering amendment chain mismatch')
            code[item['path']]=item['new_sha256']
    A.verify(reg['protected_files']); A.verify(code)
    labels=[f'{mode}_{int(round(delta*1000)):03d}' for mode in reg['modes'] for delta in reg['delta_fractions']]
    expected={(label,row['short_id']) for label in labels for row in manifest['rows']}
    actual={}; failures=[]; unfinished=[]
    for parent in sorted((out/'cells').glob('*/*')):
        if not parent.is_dir():continue
        key=(parent.parent.name,parent.name)
        if key not in expected:raise ValueError('Unregistered cell directory: '+str(parent))
        completion=parent/'complete.json'
        if completion.exists():
            saved=A.read(completion)
            if saved['status']!='COMPLETE' or saved['model_calls']!=0 or saved['CPU_only'] is not True:
                raise ValueError('Invalid zero-model CPU completion: '+str(completion))
            A.verify(saved['files']); actual[key]=saved
        else:
            unfinished.append('/'.join(key))
            if (parent/'failure.json').exists():failures.append('/'.join(key))
    done=len(actual)==len(expected)==144
    independent=out/'independent_verification.json'
    verification=None
    if independent.exists():
        audit=A.read(independent)
        if audit['status']!='PASS' or audit['verified_cells']!=144:
            raise ValueError('Independent verification failed or incomplete')
        if audit['input_manifest_sha256']!=A.sha(out/'input_manifest.json'):
            raise ValueError('Independent verification input identity changed')
        if audit['verifier_sha256']!=A.sha(Path(__file__).with_name('verify.py')):
            raise ValueError('Independent verifier source changed')
        if audit['final_scientific_code_sha256']!=code:
            raise ValueError('Independent verification code identity changed')
        verification=dict(status='PASS',path=str(independent),sha256=A.sha(independent))
    if done:
        execution=A.read(out/'execution_complete.json')
        if execution['status']!='COMPLETE' or execution['cells']!=144 or execution['model_calls']!=0 or execution['GPU_calls']!=0:
            raise ValueError('Execution completion identity differs')
    return dict(status='COMPLETE_VERIFIED' if done else 'BLOCKED_UNFINISHED_ATTEMPT' if unfinished else 'INCOMPLETE_RESUMABLE',
                cells_complete=len(actual),cells_registered=len(expected),requests=len(manifest['rows']),
                protected_files_verified=len(reg['protected_files']),source_hash_chain_amendments=len(amendments),
                unfinished_cells=unfinished,failed_cells=failures,independent_verification=verification,
                action='VERIFY_ONLY_NO_DECODER_OR_SCORER',model_calls=0,GPU_calls=0,decoder_calls=0,quality_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group()
    choice.add_argument('--status',action='store_true',help='Verify and report saved outputs; default')
    choice.add_argument('--resume',action='store_true',help='Only dispatch missing cells if no unfinished attempt exists')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2,help='Used only for unfinished authorized work; inspect current resources first')
    args=parser.parse_args()
    result=status()
    if not args.resume or result['status']=='COMPLETE_VERIFIED':
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if result['unfinished_cells']:
        print(json.dumps(result,ensure_ascii=False,indent=2))
        raise SystemExit('Existing unfinished/failed cells preserved. Explicit engineering recovery is required; no automatic retry.')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    # Delegate only to the frozen runner. Its immutable input/code checks and
    # own existing-attempt refusal remain authoritative; no scientific settings
    # or outputs are changed here. No direct worker/decoder implementation.
    completed=subprocess.run([sys.executable,'-B','-m','r2.tools.partial_edge_binding_v1.run','--stage','run','--workers',str(args.workers)],cwd=A.ROOT,check=False)
    raise SystemExit(completed.returncode)


if __name__=='__main__':main()
