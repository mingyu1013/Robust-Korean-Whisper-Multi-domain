#!/usr/bin/env bash
set -e

PYTHON=python

DATA_DIR="/path/to/dialect_only_arrow"   # 방언 필터링된 arrow (같은 거 쓰면 내부에서 필터해도 됨)
SINGLE_LORA_DIR="/path/to/stage1_lora_runs/best"  # Stage1 best LoRA 디렉터리
OUTPUT_DIR="/path/to/stage2_mas_lora_runs"

$PYTHON train/MAS_LoRA_train.py \
  --data_dir "$DATA_DIR" \
  --single_lora_dir "$SINGLE_LORA_DIR" \
  --output_dir "$OUTPUT_DIR"
