"""Transparent off-grid resume adapter for the frozen Bench training worker.

The original worker probes immediately after loading any checkpoint. Recovery
from e.g. step5250 must not introduce an unregistered96-forward probe. This
entrypoint delegates the original CLI/training unchanged, except that exactly
the verified off-grid resume cursor logs NOT_SCHEDULED with zero calls. Later
registered1000-step probes delegate verbatim. No source trainer/registration is
edited and this module does not authorize a replay.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from . import train as T

VERSION='bench_parts_exact_offgrid_resume_probe_adapter_v1'


@dataclass(frozen=True)
class ResumeIdentity:
    root: Path
    out: Path
    source: Path
    arm: str
    model_mode: str
    cursor: int
    target: int
    source_sha256: str
    registration_sha256: str
    generator_state_sha256: str
    trainer_sha256: str
    wrapper_sha256: str
    registration: dict


def resolved(path):
    return T.resolve(path).resolve()


def validate_resume(root,arm,target,resume,fork=None):
    """Read-only exact source/owner/cursor checks before the original worker."""
    if fork is not None or resume is None:
        raise ValueError('This adapter requires an explicit --resume and forbids --fork')
    if arm not in ('COMMON','PLAIN','SEAM'):
        raise ValueError('Unknown arm')
    root=resolved(root);out=root/'training'/arm;source=resolved(resume)
    if source.parent!=out or source.suffix!='.pt':
        raise ValueError('Resume checkpoint must belong to this exact root/training/arm')
    reg=T.read(root/'registration.json');regsha=T.sha(root/'registration.json')
    if reg.get('status')!='FROZEN' or reg['stopping']['probe_interval']!=1000:
        raise ValueError('Expected frozen1000-step probe schedule')
    trainer=Path(T.__file__).resolve();trainer_hash=T.sha(trainer)
    registered={resolved(p):digest for p,digest in reg['training_code_hashes'].items()}
    if registered.get(trainer)!=trainer_hash:
        raise ValueError('Frozen original trainer identity differs')
    status=T.read(out/'status.json');sidecar=T.read(source.with_suffix('.json'))
    if status.get('arm')!=arm or status.get('status') not in ('READY','INTERRUPTED_SAFE'):
        raise ValueError('Exact arm must be in an explicitly recoverable state')
    cursor=int(sidecar['step'])
    if cursor!=sidecar['step'] or status['step']!=cursor or cursor<=0 or cursor%1000==0:
        raise ValueError('This adapter is only for the exact saved non1000 resume cursor')
    if source.name not in ('rolling.pt',f'step{cursor}.pt'):
        raise ValueError('Unrecognized checkpoint source identity')
    if target<=cursor or target%1000 or (arm=='COMMON' and target>12000) or (arm!='COMMON' and not (12000<cursor<target<=21000)):
        raise ValueError('Target must advance to a registered milestone within the existing arm budget')
    if (out/'fixed_loss'/f'step{cursor}.json').exists():
        raise ValueError('Unexpected off-grid probe artifact exists; explicit audit required')
    source_hash=T.sha(source)
    if source_hash!=sidecar['sha256']:
        raise ValueError('Checkpoint file SHA differs from its saved receipt')
    payload=T.torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    expected_stage=cursor if arm=='COMMON' else cursor-12000
    if (payload.get('schema')!='bench_parts_training_v1' or payload.get('arm')!=arm
            or payload.get('step')!=cursor or payload.get('global_step')!=cursor
            or payload.get('stage_step')!=expected_stage or payload.get('EMA') is not False
            or payload.get('registration_sha256')!=regsha
            or payload.get('generator_state_sha256')!=sidecar['generator_state_sha256']):
        raise ValueError('Loaded checkpoint owner/cursor/registration differs')
    identity=ResumeIdentity(root,out,source,arm,'SEAM' if arm=='SEAM' else 'PLAIN',cursor,target,
        source_hash,regsha,payload['generator_state_sha256'],trainer_hash,T.sha(__file__),reg)
    del payload
    return identity


class ResumeProbeGuard:
    """Suppress exactly one known unscheduled call, never a scheduled probe."""
    def __init__(self,identity,original_probe):
        self.identity=identity;self.original=original_probe;self.skipped=0

    def __call__(self,root,out,step,model,head,arm,reg,prepared):
        identity=self.identity
        if (resolved(root)!=identity.root or resolved(out)!=identity.out or arm!=identity.model_mode
                or reg!=identity.registration or T.sha(identity.root/'registration.json')!=identity.registration_sha256):
            raise ValueError('Probe call root/arm/registration differs from verified resume')
        if step==identity.cursor:
            if self.skipped or step%1000==0 or (identity.out/'fixed_loss'/f'step{step}.json').exists():
                raise ValueError('Duplicate or ambiguous off-grid initial probe')
            self.skipped=1
            record=dict(status='NOT_SCHEDULED',reason='EXACT_OFF_GRID_RESUME_CURSOR',step=step,
                generator_forward_budget=0,generator_forwards=0,head_forwards=0,
                schema=VERSION,checkpoint=str(identity.source),checkpoint_sha256=identity.source_sha256,
                registration_sha256=identity.registration_sha256,trainer_sha256=identity.trainer_sha256,
                wrapper_sha256=identity.wrapper_sha256,next_registered_probe=((step//1000)+1)*1000,
                scientific_probe_schedule_changed=False)
            T.event(identity.out/'ledger.jsonl','PROBE_NOT_SCHEDULED',**record)
            return record
        if step<=identity.cursor or step>identity.target or step%1000:
            raise ValueError('Unexpected probe cursor outside the resumed registered schedule')
        if self.skipped!=1:
            raise ValueError('Scheduled probe arrived before verified initial resume call')
        return self.original(root,out,step,model,head,arm,reg,prepared)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(T.DEFAULT_ROOT))
    parser.add_argument('--arm',choices=('COMMON','PLAIN','SEAM'),required=True)
    parser.add_argument('--target',type=int,required=True)
    parser.add_argument('--resume');parser.add_argument('--fork')
    args=parser.parse_args()
    identity=validate_resume(args.root,args.arm,args.target,args.resume,args.fork)
    original=T.fixed_probe;guard=ResumeProbeGuard(identity,original)
    T.fixed_probe=guard
    try:
        # Reuse the original CLI and scientific train function without forking
        # or copying its model, optimizer, input stream, or update loop.
        T.main()
        if guard.skipped!=1:
            raise RuntimeError('Expected exactly one explicitly logged off-grid skip')
    finally:
        T.fixed_probe=original
        if T.sha(Path(T.__file__))!=identity.trainer_sha256 or T.sha(identity.root/'registration.json')!=identity.registration_sha256:
            raise RuntimeError('Protected trainer or registration changed during adapter run')


if __name__=='__main__':main()
