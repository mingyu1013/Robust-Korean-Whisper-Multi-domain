#!/usr/bin/env bash
set -e

PYTHON=python

MODEL_DIR="/path/to/stage2_mas_lora_runs/best"   # 또는 Stage1 모델
DATA_DIR="/path/to/test_arrow"
OUT_JSON="/path/to/results/mas_lora_eval.json"

$PYTHON test/Model_eval.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$DATA_DIR" \
  --output_json "$OUT_JSON"
