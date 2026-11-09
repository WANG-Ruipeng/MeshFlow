"""CPU-safe identities and immutable I/O; does not hide CUDA from later training."""
from pathlib import Path
from r2.pilots.bench_c_tessellation_v1.common import resolve,read,sha,write,arrays,npz,verify,csvwrite

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_multilod_training_v1'
EXPERIMENT='R2_BENCH_MULTILOD_TRAINING_V1'
ARMS=('PLAIN1500','MULTILOD1500')
UPDATES=1500
BASE_STEP=25500
NATIVE_BASE=13500
OLD=ROOT/'experiments/r2_bench_four_routes_v1/exposure'
DATA_ROOT=ROOT/'experiments/r2_bench_multilod_data_v1'
MANIFEST=DATA_ROOT/'data/manifest.json'
SOURCE=OLD/'training/M_PLUS3000/checkpoint_step3000.pt'
SOURCE_INFERENCE=OLD/'training/M_PLUS3000/generator_step3000.pt'
SOURCE_IDENTITIES=dict(
    source_checkpoint_sha256='3c1e9e1db0f9a462632bd70a91b5acaa4bef1dd40ad20ad90a194ab17bbd49fd',
    source_generator_sha256='76c42158b9016e58cc235b859f6cd5bc68ff1f4e19f3c70e4f94f676447170f4',
    source_optimizer_sha256='52d3145ba8012c556e4c627fe88c3b7b6c2ae7b9c6f7c177207cd7dec3640beb',
    source_rng_sha256='824e3e71a9ca6cd66d3f042a4c5b0746b63968c8f185d377af10f044e5de66e8')
LOSS_NORMALIZATION='narrow_reference_group_MSE_coefficients_v1_multilod_actual_counts'
REGISTRATION=OUT/'training_registration.json'
write_new=write
