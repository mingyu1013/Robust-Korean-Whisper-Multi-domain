#!/usr/bin/env bash
set -e

# 프로젝트 루트 기준에서 실행된다고 가정
PYTHON=python

DATA_ROOT="/path/to/raw_data"          # 원본 wav + transcript 위치
OUTPUT_DIR="/path/to/arrow_dataset"    # Preprocessing.py에서 저장할 경로

$PYTHON preprocessing/Preprocessing.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR"
