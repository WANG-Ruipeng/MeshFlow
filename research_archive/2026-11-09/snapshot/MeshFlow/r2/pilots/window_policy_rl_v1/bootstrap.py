from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_window_policy_rl_v1'
OLD = ROOT / 'experiments/r2_terminal_reward_nft_v1'

def bind():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from r2.pilots.terminal_reward_nft_v1 import data as D, evaluation as E
    D.OUT = OUT
    E.OUT = OUT
    return D, E
