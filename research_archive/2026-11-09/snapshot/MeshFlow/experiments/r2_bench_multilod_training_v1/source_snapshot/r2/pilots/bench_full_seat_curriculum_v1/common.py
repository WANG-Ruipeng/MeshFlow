from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import resolve, read, sha, write_new, atomic_write, resources

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_full_seat_curriculum_v1'
EXPERIMENT = 'R2_BENCH_FULL_SEAT_CURRICULUM_V1'
STAGES = ('NATIVE', 'SEAM')
SOURCE = ROOT / 'experiments/r2_bench_parts_convergence_v1/training/COMMON/step12000.pt'
HEAD = ROOT / 'experiments/r2_bench_three_family_v1/production_contract/old_s15000_head_only.pt'
SOURCE_SHA256 = '40aecccd1838a5b04dadeac13e72b9cf7315f3721c16e180b97844a04c2f8032'
HEAD_SHA256 = '9802355787ff2abf9c02f983f9ca8547bec2182c32162a309cb5048927d51ce1'
MODULE = 'r2.pilots.bench_full_seat_curriculum_v1.'
