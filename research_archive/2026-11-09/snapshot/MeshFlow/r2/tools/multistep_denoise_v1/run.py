"""Recoverable fixed-budget entrypoint; scientific stages run in separate processes."""
import argparse,subprocess,sys
from pathlib import Path
from common import prepare,OUT,write,read
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','denoise','score','gallery','all'],default='prepare');args=p.parse_args()
 reg=prepare()
 if args.stage=='prepare':print('REGISTERED 12 USER_MIX; 48 new denoiser forwards; endpoints D1/D3/D5');raise SystemExit(0)
 stages=['denoise','score','gallery'] if args.stage=='all' else [args.stage]
 for stage in stages:
  target=Path(__file__).with_name(stage+'.py')
  if not target.exists():raise FileNotFoundError('Stage implementation not ready: '+str(target))
  command=[sys.executable,'-B',str(target)]
  subprocess.run(command,check=True)