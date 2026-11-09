"""Recompute the same 24 parameter-gradient probes with frozen metric weights.

No optimizer, model update or extraction; never recalibrates the weights.
"""
import argparse,json,traceback
from pathlib import Path
import numpy as np
from common import ROOT,dump,sha
from metric_calibration import evaluate_checkpoint,save_evaluation,FIELD_NAMES,SHARED_PARAMETER_NAMES

LABEL_STEPS={'parent3000':3000,'M0_5000':5000,'M0_7000':7000,'M0_9000':9000,'B0_9000':9000}


def main(checkpoint,label,root=ROOT,device='cuda',weights_path=None,out=None):
    root=Path(root);checkpoint=Path(checkpoint)
    weights_path=Path(weights_path) if weights_path is not None else root/'calibration/WEIGHTS.json'
    out=Path(out) if out is not None else root/'probes'/label
    out.mkdir(parents=True,exist_ok=False)
    frozen=json.loads(weights_path.read_text());weights_sha=sha(weights_path)
    protocol=dict(version='SINGLE_STATE_METRIC0_FROZEN_PARAMETER_PROBE',label=label,
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),expected_completed_steps=LABEL_STEPS[label],
        weights_file=str(weights_path),weights_sha256=weights_sha,weights=frozen.get('weights'),
        recalibration_allowed=False,shared_parameter_names=list(SHARED_PARAMETER_NAMES),
        source_files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in [Path(__file__),Path(__file__).with_name('metric_calibration.py')]])
    dump(out/'PROBE_PROTOCOL.json',protocol);dump(out/'PROBE_PROTOCOL_SHA256.json',dict(sha256=sha(out/'PROBE_PROTOCOL.json')))
    try:
        if frozen.get('passed') is not True or frozen.get('status')!='PASS':raise ValueError('Frozen calibration did not pass')
        coefficients=np.array([frozen['weights'][k] for k in FIELD_NAMES])
        if not np.isfinite(coefficients).all() or np.any(coefficients<=0):raise ValueError('Frozen weights must be finite and positive')
        evaluation=evaluate_checkpoint(checkpoint,root,device,weights=frozen['weights'])
        if evaluation['checkpoint_completed_steps']!=LABEL_STEPS[label]:raise ValueError('Checkpoint step does not match fixed label')
        if evaluation['data_sha256']!=frozen['data_sha256'] or evaluation['probe_noise_sha256']!=frozen['probe_noise_sha256']:
            raise ValueError('Probe data/bank differ from frozen calibration')
        if sha(weights_path)!=weights_sha:raise ValueError('Weights file changed during probe')
        evaluation.update(label=label,weights_sha256=weights_sha,frozen_weights=frozen['weights'],recalibrated=False)
        save_evaluation(out,evaluation,existing=True)
        result={k:v for k,v in evaluation.items() if k not in ['records','matching']}
    except Exception as exc:
        result=dict(passed=False,label=label,error=repr(exc),traceback=traceback.format_exc(),weights_sha256=weights_sha,
            checkpoint_sha256=sha(checkpoint),optimizer_updates=0,mesh_extractions=0,recalibrated=False)
        dump(out/'RESULT.json',result)
    print(json.dumps(dict(label=label,passed=result['passed'],summary=result.get('summary'),error=result.get('error')),indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--label',required=True,choices=LABEL_STEPS)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--device',default='cuda');p.add_argument('--weights',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();r=main(a.checkpoint,a.label,a.root,a.device,a.weights,a.out);raise SystemExit(0 if r['passed'] else 1)