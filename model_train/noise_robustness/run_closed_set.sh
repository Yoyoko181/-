#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOOT_DB_ROOT="${FOOT_DB_ROOT:-/root/毕设/Foot_database}"
DATASET="${DATASET:-A3}"
TRAIN_SUBSETS="${TRAIN_SUBSETS:-A3_1,A3_3}"
TEST_SUBSET="${TEST_SUBSET:-A3_2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-200}"
USE_PRETRAINED="${USE_PRETRAINED:-0}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
NOISE_SNR_DB="${NOISE_SNR_DB:-10}"
STRUCT_NOISE_ROOT="${STRUCT_NOISE_ROOT:-}"
NONSTRUCT_NOISE="${NONSTRUCT_NOISE:-gaussian}"

ARGS=(
  --foot_db_root "$FOOT_DB_ROOT"
  --dataset "$DATASET"
  --train_subsets "$TRAIN_SUBSETS"
  --test_subset "$TEST_SUBSET"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --seed "$SEED"
  --num_workers "$NUM_WORKERS"
  --epochs "$EPOCHS"
  --noise_snr_db "$NOISE_SNR_DB"
  --nonstruct_noise "$NONSTRUCT_NOISE"
)

if [[ -n "${STRUCT_NOISE_ROOT}" ]]; then
  ARGS+=(--struct_noise_root "$STRUCT_NOISE_ROOT")
fi

if [[ "$USE_PRETRAINED" == "1" ]]; then
  ARGS+=(--use_pretrained)
fi

if [[ "$SKIP_GENERATE" == "1" ]]; then
  ARGS+=(--skip_generate)
fi

python3 "$SCRIPT_DIR/train_noise_robustness.py" "${ARGS[@]}"

