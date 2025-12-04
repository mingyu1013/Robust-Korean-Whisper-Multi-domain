#!/usr/bin/env bash
set -e

PYTHON=python

DATA_DIR="/path/to/arrow_dataset"      # 위에서 만든 전처리 결과
OUTPUT_DIR="/path/to/stage1_lora_runs"

$PYTHON train/LoRA_train.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR"
