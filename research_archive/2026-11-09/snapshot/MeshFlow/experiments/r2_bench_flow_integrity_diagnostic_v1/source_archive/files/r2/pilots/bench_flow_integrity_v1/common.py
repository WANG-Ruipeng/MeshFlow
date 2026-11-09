from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import read, resolve, sha, write_new, atomic_write, resources
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'experiments/r2_bench_flow_integrity_diagnostic_v1'
MODELS = {
 'COMMON_S12000': ROOT/'experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/generator_step0.pt',
 'NATIVE_S6000': ROOT/'experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/generator_step6000.pt',
 'COVERAGE_S1500': ROOT/'experiments/r2_bench_seat_coverage_v1/training/NATIVE/generator_step1500.pt',
 'SEAM_S3000': ROOT/'experiments/r2_bench_full_seat_curriculum_v1/training/SEAM/generator_step3000.pt',
}
