#!/bin/bash
# Count eval exactly as in the reports. Usage: bash run_eval.sh MODEL [test|dev]
#   MODEL = HF id (Qwen/Qwen3.5-4B) or a local checkpoint dir. Needs 1 GPU (A100/L40S/A40/H200 all fine).
set -euo pipefail
MODEL=${1:?usage: bash run_eval.sh MODEL [test|dev]}; SPLIT=${2:-test}
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
python -m twvp.eval.count --model "$MODEL" --data data/$SPLIT.jsonl --out results/$(basename "$MODEL")_$SPLIT.jsonl
