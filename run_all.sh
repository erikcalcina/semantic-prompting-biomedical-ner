#!/usr/bin/env bash
# Runs all experiments of the paper: 5 models x 2 datasets x 3 settings, each with the baseline
# and the semantically enhanced prompt, followed by the evaluation and the McNemar test.
# Extra options are passed on to run_pipeline.sh, e.g. ./run_all.sh --out_dir runs --gen_batch_size 8
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=(
  meta-llama/Llama-3.1-8B-Instruct
  meta-llama/Llama-3.1-8B
  meta-llama/Llama-3.2-3B-Instruct
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  google/txgemma-9b-chat
)

for model in "${MODELS[@]}"; do
  for dataset in maccrobat ncbi; do
    for regime in zero-shot few-shot fine-tuned; do
      "$ROOT/run_pipeline.sh" --model "$model" --dataset "$dataset" --regime "$regime" --setting both "$@"
    done
  done
done
