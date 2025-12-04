#!/usr/bin/env bash
set -e

PYTHON=python

IN_JSON="/path/to/results/mas_lora_eval.json"

$PYTHON test/eval_from_json_macro.py \
  --input_json "$IN_JSON"
