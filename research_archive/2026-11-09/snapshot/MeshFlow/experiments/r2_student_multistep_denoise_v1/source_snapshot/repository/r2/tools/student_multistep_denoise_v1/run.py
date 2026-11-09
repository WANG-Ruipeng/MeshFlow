"""Resumable stage runner; closed per-forward receipts never run twice."""
import argparse,subprocess,sys
from pathlib import Path
import common as C
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','denoise','score','gallery','audit','all'],default='all');a=p.parse_args()
 C.prepare()
 if a.stage=='prepare':print('PREPARED')
 for stage in ('denoise','score','gallery','audit'):
  if a.stage in ('all',stage):subprocess.run([sys.executable,'-B',str(Path(__file__).with_name(stage+'.py'))],check=True)
