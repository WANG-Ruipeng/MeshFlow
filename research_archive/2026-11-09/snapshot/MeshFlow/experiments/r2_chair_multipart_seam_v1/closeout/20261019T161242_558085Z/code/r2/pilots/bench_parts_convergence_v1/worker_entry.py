"""IO-only entrypoint for the unchanged registered Bench training worker.

Retry only an already-prepared file replacement (never its tensor serialization,
model call or optimizer update) when DrvFs reports transient sharing errors.
The scope is the exact experiment root. Missing source and unrelated IO errors
fail immediately; persistent sharing errors fail within30 seconds per operation.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from contextlib import contextmanager
import errno
import json
from pathlib import Path
import sys
import time

VERSION='bench_parts_worker_scoped_io_guard_v1'
SHARING_ERRNOS=frozenset((errno.EPERM,errno.EACCES,errno.EBUSY))


def within(path,root):
    try:Path(os.fsdecode(path)).resolve().relative_to(root);return True
    except (ValueError,TypeError,OSError):return False


class ScopedIO:
    def __init__(self,root,replace,*,log_path=None,replace_timeout=30.,metadata_timeout=2.,
                 clock=time.monotonic,sleep=time.sleep):
        self.root=Path(root).resolve();self.replace_original=replace;self.log_path=Path(log_path) if log_path else None
        self.replace_timeout=min(30.,max(0.,float(replace_timeout)))
        self.metadata_timeout=min(2.,max(0.,float(metadata_timeout)))
        self.clock=clock;self.sleep=sleep

    def record(self,event,**fields):
        value=dict(event=event,schema=VERSION,unix=time.time(),pid=os.getpid(),**fields)
        text=json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n'
        # Diagnostic log failure must not turn a successful atomic replacement
        # into another failed training operation. STDERR is the durable child log.
        if self.log_path is not None:
            try:
                with self.log_path.open('a',encoding='utf-8') as f:f.write(text);f.flush()
                return
            except OSError:pass
        print(text,end='',file=sys.stderr,flush=True)

    def replace(self,src,dst,*,src_dir_fd=None,dst_dir_fd=None):
        options={}
        if src_dir_fd is not None:options['src_dir_fd']=src_dir_fd
        if dst_dir_fd is not None:options['dst_dir_fd']=dst_dir_fd
        # Directory-FD and external paths are deliberately outside this guard.
        if options or not within(src,self.root) or not within(dst,self.root):
            return self.replace_original(src,dst,**options)
        start=self.clock();attempt=0;delay=.02
        while True:
            attempt+=1
            try:
                result=self.replace_original(src,dst)
                if attempt>1:self.record('ATOMIC_REPLACE_RETRY_SUCCEEDED',source=os.fsdecode(src),destination=os.fsdecode(dst),
                    attempts=attempt,wait_seconds=self.clock()-start,model_calls_replayed=0,serialization_calls_replayed=0)
                return result
            except OSError as exc:
                if exc.errno not in SHARING_ERRNOS:raise
                # Never fabricate success if the prepared source is absent or
                # its identity cannot be inspected. Do not rerun torch.save.
                try:source_present=Path(os.fsdecode(src)).is_file()
                except OSError:source_present=False
                remaining=self.replace_timeout-(self.clock()-start)
                if not source_present or remaining<=0:
                    self.record('ATOMIC_REPLACE_RETRY_EXHAUSTED',source=os.fsdecode(src),destination=os.fsdecode(dst),
                        attempts=attempt,wait_seconds=self.clock()-start,source_present=source_present,errno=exc.errno)
                    raise
                pause=min(delay,remaining)
                self.record('ATOMIC_REPLACE_TRANSIENT_SHARING',source=os.fsdecode(src),destination=os.fsdecode(dst),
                    attempts=attempt,errno=exc.errno,next_wait_seconds=pause,timeout_seconds=self.replace_timeout)
                self.sleep(pause);delay=min(delay*2,1.)

    def metadata_reader(self,original):
        def read(path):
            if not within(path,self.root) or Path(path).suffix.lower()!='.json':return original(path)
            start=self.clock();attempt=0;delay=.01
            while True:
                attempt+=1
                try:
                    result=original(path)
                    if attempt>1:self.record('METADATA_READ_RETRY_SUCCEEDED',path=str(path),attempts=attempt,wait_seconds=self.clock()-start)
                    return result
                except OSError as exc:
                    if exc.errno not in SHARING_ERRNOS|{errno.ENOENT}:raise
                    remaining=self.metadata_timeout-(self.clock()-start)
                    if remaining<=0:
                        self.record('METADATA_READ_RETRY_EXHAUSTED',path=str(path),attempts=attempt,errno=exc.errno,wait_seconds=self.clock()-start)
                        raise
                    self.sleep(min(delay,remaining));delay=min(delay*2,.2)
                # JSONDecodeError is intentionally not caught: corrupted
                # metadata must not be mistaken for a temporary file handle.
        return read


@contextmanager
def installed(root,trainer,base,*,log_path=None,guard_factory=ScopedIO):
    original_replace=os.replace;original_train_read=trainer.read;original_base_read=base.read
    guard=guard_factory(root,original_replace,log_path=log_path)
    os.replace=guard.replace
    trainer.read=guard.metadata_reader(original_train_read)
    base.read=guard.metadata_reader(original_base_read)
    try:yield guard
    finally:
        os.replace=original_replace;trainer.read=original_train_read;base.read=original_base_read


def main():
    # Original train import sets its frozen CPU threads/runtime environment;
    # this wrapper does not change precision, batch, data, model or RNG settings.
    from . import train as T
    from r2.pilots.bench_control_transfer_v1 import train as BASE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(T.DEFAULT_ROOT));parser.add_argument('--arm',choices=('COMMON','PLAIN','SEAM'),required=True)
    parser.add_argument('--target',type=int,required=True);parser.add_argument('--resume');parser.add_argument('--fork')
    args=parser.parse_args();root=T.resolve(args.root).resolve()
    if root!=T.DEFAULT_ROOT.resolve():raise ValueError('IO wrapper is restricted to the exact registered experiment root')
    if args.resume and args.fork:raise ValueError('Resume and fork are exclusive')
    out=root/'training'/args.arm;out.mkdir(parents=True,exist_ok=True)
    log=out/f'io_guard_{args.target}_{os.getpid()}.jsonl'
    with installed(root,T,BASE,log_path=log) as guard:
        offgrid=False;cursor=None
        if args.resume:
            source=T.resolve(args.resume).resolve()
            if source.parent!=out:raise ValueError('Resume source must belong to exact root/arm')
            cursor=int(T.read(source.with_suffix('.json'))['step']);offgrid=bool(cursor%1000)
        guard.record('WORKER_IO_GUARD_INSTALLED',root=str(root),arm=args.arm,target=args.target,resume_cursor=cursor,
            replace_timeout_seconds=30.,metadata_timeout_seconds=2.,determinism_env=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            delegate='resume_train' if offgrid else 'train',model_calls_replayed=0)
        if offgrid:
            from . import resume_train as R
            return R.main()
        return T.main()


if __name__=='__main__':main()
