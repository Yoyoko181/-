#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/毕设/Terra-main}"
FOOT_DB_ROOT="${FOOT_DB_ROOT:-/root/毕设/Foot_database}"
INTERIM_ROOT="${INTERIM_ROOT:-$ROOT/people_database/interim_matlab}"
CWT_ROOT="${CWT_ROOT:-$ROOT/people_database}"

EPOCHS="${EPOCHS:-150}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
DROPOUT="${DROPOUT:-0.25}"

WGAN_EPOCHS="${WGAN_EPOCHS:-50}"
SYN_PER_CLASS="${SYN_PER_CLASS:-200}"
SYN_RATIO="${SYN_RATIO:-0.5}"

DATASET_A3="${DATASET_A3:-A3}"
DATASET_A2="${DATASET_A2:-A2}"
DATASET_A5="${DATASET_A5:-A5}"

TRAIN_SUBSETS_A3="${TRAIN_SUBSETS_A3:-A3_1,A3_3}"
TEST_SUBSET_A3="${TEST_SUBSET_A3:-A3_2}"

TRAIN_SUBSETS_A2="${TRAIN_SUBSETS_A2:-A2_1}"
TEST_SUBSET_A2="${TEST_SUBSET_A2:-A2_3}"

TRAIN_SUBSETS_A5="${TRAIN_SUBSETS_A5:-A5_2}"
TEST_SUBSET_A5_FAST="${TEST_SUBSET_A5_FAST:-A5_1}"
TEST_SUBSET_A5_SLOW="${TEST_SUBSET_A5_SLOW:-A5_3}"

TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$ROOT/model_train/results_archive/$TS}"
mkdir -p "$ARCHIVE_ROOT"

cd "$ROOT"
UPDATE_CODE="${UPDATE_CODE:-0}"
if [ "$UPDATE_CODE" = "1" ]; then
  git pull || true
fi

python3 -V

mkdir -p "$ARCHIVE_ROOT/logs"

fail_report() {
  local code="$?"
  echo "FAILED (exit=$code) at: ${CURRENT_STEP:-unknown}" >&2
  if [ -n "${CURRENT_LOG:-}" ] && [ -f "${CURRENT_LOG:-}" ]; then
    echo "Last 80 lines of log: $CURRENT_LOG" >&2
    tail -n 80 "$CURRENT_LOG" >&2 || true
  fi
  echo "Archive: $ARCHIVE_ROOT" >&2
  exit "$code"
}
trap fail_report ERR

require_dir() {
  local p="$1"
  if [ ! -d "$p" ]; then
    echo "Missing dir: $p" >&2
    exit 2
  fi
}

require_file() {
  local p="$1"
  if [ ! -f "$p" ]; then
    echo "Missing file: $p" >&2
    exit 2
  fi
}

preflight_python() {
  python3 -c "import torch; import scipy; import numpy; import matplotlib; print('torch', torch.__version__)" >/dev/null
}

preflight_data() {
  require_dir "$ROOT/model_train"
  require_dir "$INTERIM_ROOT"
  require_dir "$CWT_ROOT"
  if ! find "$INTERIM_ROOT/$DATASET_A3/A3_1" -maxdepth 1 -type f -name "P*.mat" | head -n 1 | grep -q .; then
    echo "No P*.mat found under $INTERIM_ROOT/$DATASET_A3/A3_1" >&2
    exit 2
  fi
  if ! find "$CWT_ROOT/$DATASET_A3/A3_1" -maxdepth 2 -type f \( -name "*.jpg" -o -name "*.png" \) | head -n 1 | grep -q .; then
    echo "No images found under $CWT_ROOT/$DATASET_A3/A3_1" >&2
    exit 2
  fi
}

run_step() {
  local name="$1"
  shift
  CURRENT_STEP="$name"
  CURRENT_LOG="$ARCHIVE_ROOT/logs/$name.log"
  echo "=== $name ==="
  echo "LOG: $CURRENT_LOG"
  "$@" 2>&1 | tee "$CURRENT_LOG"
}

run_and_archive_dir() {
  local src="$1"
  local dst="$2"
  if [ -d "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    rm -rf "$dst"
    cp -r "$src" "$dst"
  fi
}

preflight_python
preflight_data

mkdir -p "$ROOT/model_train/results"

cd "$ROOT/model_train/cross_material"
run_step baseline_cross_material python3 train_cross_material.py \
  --foot_db_root "$FOOT_DB_ROOT" \
  --cwt_out_root "$CWT_ROOT" \
  --dataset "$DATASET_A3" \
  --train_subsets "$TRAIN_SUBSETS_A3" \
  --test_subset "$TEST_SUBSET_A3" \
  --skip_generate \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR"
run_and_archive_dir "$ROOT/model_train/results/material" "$ARCHIVE_ROOT/baseline/material"

cd "$ROOT/model_train/cross_distance"
run_step baseline_cross_distance python3 train_cross_distance.py \
  --foot_db_root "$FOOT_DB_ROOT" \
  --cwt_out_root "$CWT_ROOT" \
  --dataset "$DATASET_A2" \
  --train_subsets "$TRAIN_SUBSETS_A2" \
  --test_subset "$TEST_SUBSET_A2" \
  --skip_generate \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR"
run_and_archive_dir "$ROOT/model_train/results/distance" "$ARCHIVE_ROOT/baseline/distance"

cd "$ROOT/model_train/cross_speed"
run_step baseline_cross_speed_fast python3 train_cross_speed.py \
  --foot_db_root "$FOOT_DB_ROOT" \
  --cwt_out_root "$CWT_ROOT" \
  --dataset "$DATASET_A5" \
  --train_subsets "$TRAIN_SUBSETS_A5" \
  --test_subset "$TEST_SUBSET_A5_FAST" \
  --skip_generate \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR"
run_and_archive_dir "$ROOT/model_train/results/speed" "$ARCHIVE_ROOT/baseline/speed_${TEST_SUBSET_A5_FAST}"

run_step baseline_cross_speed_slow python3 train_cross_speed.py \
  --foot_db_root "$FOOT_DB_ROOT" \
  --cwt_out_root "$CWT_ROOT" \
  --dataset "$DATASET_A5" \
  --train_subsets "$TRAIN_SUBSETS_A5" \
  --test_subset "$TEST_SUBSET_A5_SLOW" \
  --skip_generate \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR"
run_and_archive_dir "$ROOT/model_train/results/speed" "$ARCHIVE_ROOT/baseline/speed_${TEST_SUBSET_A5_SLOW}"

cd "$ROOT/model_train/new_models"

for v in base ms ms_se ms_se_lstm full; do
  run_step "new_models_hybrid_${v}" python3 train_classifier.py \
    --interim_root "$INTERIM_ROOT" \
    --dataset "$DATASET_A3" \
    --train_subsets "$TRAIN_SUBSETS_A3" \
    --test_subset "$TEST_SUBSET_A3" \
    --model hybrid \
    --hybrid_variant "$v" \
    --dropout "$DROPOUT" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --results_dir "$ARCHIVE_ROOT/new_models/hybrid_$v"
done

run_step new_models_resnet1d_se python3 train_classifier.py \
  --interim_root "$INTERIM_ROOT" \
  --dataset "$DATASET_A3" \
  --train_subsets "$TRAIN_SUBSETS_A3" \
  --test_subset "$TEST_SUBSET_A3" \
  --model resnet1d_se \
  --dropout "$DROPOUT" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --results_dir "$ARCHIVE_ROOT/new_models/resnet1d_se"

run_step new_models_bilstm_attn python3 train_classifier.py \
  --interim_root "$INTERIM_ROOT" \
  --dataset "$DATASET_A3" \
  --train_subsets "$TRAIN_SUBSETS_A3" \
  --test_subset "$TEST_SUBSET_A3" \
  --model bilstm_attn \
  --dropout "$DROPOUT" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --results_dir "$ARCHIVE_ROOT/new_models/bilstm_attn"

run_step wgan_gp_A3_1 python3 train_wgan_gp.py \
  --interim_root "$INTERIM_ROOT" \
  --dataset "$DATASET_A3" \
  --subset A3_1 \
  --epochs "$WGAN_EPOCHS" \
  --seed "$SEED" \
  --out_dir "$ARCHIVE_ROOT/wgan/wgan_gp_${DATASET_A3}_A3_1"

run_step wgan_gp_A3_3 python3 train_wgan_gp.py \
  --interim_root "$INTERIM_ROOT" \
  --dataset "$DATASET_A3" \
  --subset A3_3 \
  --epochs "$WGAN_EPOCHS" \
  --seed "$SEED" \
  --out_dir "$ARCHIVE_ROOT/wgan/wgan_gp_${DATASET_A3}_A3_3"

run_step synth_A3_1 python3 generate_synthetic.py \
  --wgan_dir "$ARCHIVE_ROOT/wgan/wgan_gp_${DATASET_A3}_A3_1" \
  --out_path "$ARCHIVE_ROOT/wgan/syn_A3_1.pt" \
  --num_per_class "$SYN_PER_CLASS" \
  --seed "$SEED"

run_step synth_A3_3 python3 generate_synthetic.py \
  --wgan_dir "$ARCHIVE_ROOT/wgan/wgan_gp_${DATASET_A3}_A3_3" \
  --out_path "$ARCHIVE_ROOT/wgan/syn_A3_3.pt" \
  --num_per_class "$SYN_PER_CLASS" \
  --seed "$SEED"

run_step hybrid_full_syn_A3_1 python3 train_classifier.py \
  --interim_root "$INTERIM_ROOT" \
  --dataset "$DATASET_A3" \
  --train_subsets "$TRAIN_SUBSETS_A3" \
  --test_subset "$TEST_SUBSET_A3" \
  --model hybrid \
  --hybrid_variant full \
  --dropout "$DROPOUT" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --synthetic_pt "$ARCHIVE_ROOT/wgan/syn_A3_1.pt" \
  --synthetic_ratio "$SYN_RATIO" \
  --results_dir "$ARCHIVE_ROOT/new_models/hybrid_full__syn_A3_1"

run_step hybrid_full_syn_A3_3 python3 train_classifier.py \
  --interim_root "$INTERIM_ROOT" \
  --dataset "$DATASET_A3" \
  --train_subsets "$TRAIN_SUBSETS_A3" \
  --test_subset "$TEST_SUBSET_A3" \
  --model hybrid \
  --hybrid_variant full \
  --dropout "$DROPOUT" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --synthetic_pt "$ARCHIVE_ROOT/wgan/syn_A3_3.pt" \
  --synthetic_ratio "$SYN_RATIO" \
  --results_dir "$ARCHIVE_ROOT/new_models/hybrid_full__syn_A3_3"

echo "$ARCHIVE_ROOT"
