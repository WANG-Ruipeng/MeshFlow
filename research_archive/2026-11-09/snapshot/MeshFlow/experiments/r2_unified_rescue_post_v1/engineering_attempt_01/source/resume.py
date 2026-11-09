"""Finite CPU140 entry; skips completed cells, refuses unexplained partials."""
import argparse
from . import run,summarize,displacement,finalize
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,choices=(1,2,3,4,5,6),default=4);a=p.parse_args()
    run.run(a.workers);summarize.aggregate();displacement.run();finalize.finish()
