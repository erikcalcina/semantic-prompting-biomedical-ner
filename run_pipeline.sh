#!/usr/bin/env bash
# End-to-end pipeline for one model on one dataset. For each setting (baseline / semantic):
#   [fine-tune] -> predict -> post-process -> evaluate
# and, when both settings run, a McNemar test between them.
#
# Usage:
#   ./run_pipeline.sh --model meta-llama/Llama-3.1-8B-Instruct --dataset maccrobat --regime fine-tuned
#   ./run_pipeline.sh --model google/txgemma-9b-chat --dataset ncbi --regime few-shot --setting semantic
#
# Options [defaults]:
#   --model NAME          Hugging Face model id (required)
#   --dataset NAME        maccrobat | ncbi                                         [maccrobat]
#   --regime NAME         zero-shot | few-shot | fine-tuned                        [fine-tuned]
#   --setting NAME        baseline | semantic | both                               [both]
#   --out_dir DIR         adapters, predictions and scores go here                 [./runs]
#   --train_file FILE     training data  [data/train.llm.json | data/NCBI/ncbi_sentences_train.json]
#   --test_file FILE      test data      [data/test.llm.json  | data/NCBI/ncbi_sentences_test.json]
#   --adapter DIR         skip training and use this LoRA adapter (needs a single --setting)
#   --temperature T       [0.2 for zero-/few-shot, 0.1 for fine-tuned]
#   --max_new_tokens N    [2000]
#   --batch_size N        training batch size                                      [4]
#   --max_seq_length N    [4000 baseline, 2500 semantic]
#   --epochs N            [3]
#   --max_steps N         stop training after N steps, for smoke tests             [-1 = off]
#   --seed N              seed for fine-tuning and sampling  [unset: fine-tuning uses seed 42, sampling is unseeded]
#   --stop_at_turn_end    stop generation at the chat turn-end token (TxGemma fix)      [off, as in the paper]
#   --gen_batch_size N    prompts generated together; 1 = original one-by-one loop [1]
#   --delete_adapter      delete the adapter this run trained once its predictions are written
#   --skip_existing       reuse finished steps (trained adapter, predictions) from an earlier run
#   --normalize_label_case  match predicted labels case-insensitively to the dataset's
#                         labels (e.g. "Disease" -> "DISEASE"); default is the paper's
#                         single "Biological Structure" fix
#
# The defaults are the settings reported in the paper. Set PYTHON to pick the interpreter
# (default: python).
#
# Output: <out_dir>/<dataset>/<regime>/<setting>/<model>/{adapter/,predictions.raw.json,
#         responses.jsonl,predictions.json,scores.json} and <out_dir>/<dataset>/<regime>/mcnemar/<model>.{txt,json}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

MODEL=""; DATASET="maccrobat"; REGIME="fine-tuned"; SETTING="both"; OUT_DIR="$ROOT/runs"
TRAIN_FILE=""; TEST_FILE=""; ADAPTER=""; POSTPROCESS_ARGS=""
TEMPERATURE=""; MAX_NEW_TOKENS=""; BATCH_SIZE=""; MAX_SEQ_LENGTH=""; EPOCHS=3; MAX_STEPS=-1
GEN_BATCH_SIZE=1; DELETE_ADAPTER=0; SKIP_EXISTING=0; SEED_ARGS=(); STOP_ARGS=()

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die() { echo "Error: $*" >&2; exit 1; }
log() { echo; echo "=== $(date '+%F %T') $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --regime) REGIME="$2"; shift 2 ;;
    --setting) SETTING="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --train_file) TRAIN_FILE="$2"; shift 2 ;;
    --test_file) TEST_FILE="$2"; shift 2 ;;
    --adapter) ADAPTER="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --max_seq_length) MAX_SEQ_LENGTH="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --gen_batch_size) GEN_BATCH_SIZE="$2"; shift 2 ;;
    --seed) SEED_ARGS=(--seed "$2"); shift 2 ;;
    --stop_at_turn_end) STOP_ARGS=(--stop_at_turn_end); shift ;;
    --delete_adapter) DELETE_ADAPTER=1; shift ;;
    --skip_existing) SKIP_EXISTING=1; shift ;;
    --normalize_label_case) POSTPROCESS_ARGS="--normalize_label_case"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

[[ -n "$MODEL" ]] || die "--model is required"
case "$DATASET" in
  maccrobat) TRAIN_FILE="${TRAIN_FILE:-$ROOT/data/train.llm.json}"; TEST_FILE="${TEST_FILE:-$ROOT/data/test.llm.json}" ;;
  ncbi) TRAIN_FILE="${TRAIN_FILE:-$ROOT/data/NCBI/ncbi_sentences_train.json}"; TEST_FILE="${TEST_FILE:-$ROOT/data/NCBI/ncbi_sentences_test.json}" ;;
  *) die "--dataset must be maccrobat or ncbi" ;;
esac
case "$REGIME" in
  zero-shot) PROMPT_TYPE="prompt_only" ;;
  few-shot) PROMPT_TYPE="few_shot_prompting" ;;
  fine-tuned) PROMPT_TYPE="instruction_prompt" ;;
  *) die "--regime must be zero-shot, few-shot or fine-tuned" ;;
esac
case "$SETTING" in
  both) SETTINGS=(baseline semantic) ;;
  baseline|semantic) SETTINGS=("$SETTING") ;;
  *) die "--setting must be baseline, semantic or both" ;;
esac
if [[ -n "$ADAPTER" ]]; then
  [[ "$REGIME" == fine-tuned ]] || die "--adapter only applies to --regime fine-tuned"
  [[ "$SETTING" != both ]] || die "--adapter needs --setting baseline or semantic"
fi
[[ -f "$TEST_FILE" ]] || die "test file not found: $TEST_FILE"

run_setting() {
  local setting="$1" semantic_arg="" dir adapter raw
  local d_tokens=2000 d_batch=4 d_seq=4000 d_ft_temperature=0.1
  if [[ "$setting" == semantic ]]; then
    semantic_arg="--semantic"; d_seq=2500
  fi
  local temperature="$TEMPERATURE"
  if [[ -z "$temperature" ]]; then
    if [[ "$REGIME" == fine-tuned ]]; then temperature="$d_ft_temperature"; else temperature=0.2; fi
  fi

  dir="$OUT_DIR/$DATASET/$REGIME/$setting/$MODEL"
  raw="$dir/predictions.raw.json"
  mkdir -p "$dir"

  if [[ "$SKIP_EXISTING" == 1 && -f "$raw" ]]; then
    log "[$setting] reusing existing predictions $raw"
  else
    local adapter_args=()
    if [[ "$REGIME" == fine-tuned ]]; then
      adapter="${ADAPTER:-$dir/adapter}"
      if [[ -n "$ADAPTER" ]]; then
        :
      elif [[ "$SKIP_EXISTING" == 1 && -f "$adapter/adapter_config.json" ]]; then
        log "[$setting] reusing trained adapter $adapter"
      else
        log "[$setting] fine-tuning $MODEL on $TRAIN_FILE"
        "$PYTHON" "$ROOT/scripts/train.py" --model_name "$MODEL" \
          --dataset "$TRAIN_FILE" --dataset_name "$DATASET" $semantic_arg --output_dir "$adapter" \
          --epochs "$EPOCHS" --batch_size "${BATCH_SIZE:-$d_batch}" \
          --max_seq_length "${MAX_SEQ_LENGTH:-$d_seq}" --max_steps "$MAX_STEPS" "${SEED_ARGS[@]}"
      fi
      adapter_args=(--adapter_name "$adapter")
    fi

    log "[$setting] predicting ($REGIME, temperature $temperature) on $TEST_FILE"
    "$PYTHON" "$ROOT/scripts/predict.py" --model_name "$MODEL" "${adapter_args[@]}" \
      --dataset "$TEST_FILE" --dataset_name "$DATASET" $semantic_arg --prompt_type "$PROMPT_TYPE" \
      --temperature "$temperature" --max_new_tokens "${MAX_NEW_TOKENS:-$d_tokens}" \
      --batch_size "$GEN_BATCH_SIZE" --output_file "$raw" --responses_file "$dir/responses.jsonl" "${SEED_ARGS[@]}" "${STOP_ARGS[@]}"

    if [[ "$DELETE_ADAPTER" == 1 && "$REGIME" == fine-tuned && -z "$ADAPTER" ]]; then
      rm -rf "$adapter"
      log "[$setting] deleted adapter $adapter"
    fi
  fi

  log "[$setting] post-processing and evaluating"
  "$PYTHON" "$ROOT/scripts/postprocess.py" --input "$raw" \
    --output "$dir/predictions.json" --dataset_name "$DATASET" $POSTPROCESS_ARGS
  "$PYTHON" "$ROOT/evaluation/score.py" --dataset_true "$TEST_FILE" \
    --dataset_pred "$dir/predictions.json" --dataset_name "$DATASET" --per_label --output "$dir/scores.json"
}

for setting in "${SETTINGS[@]}"; do
  run_setting "$setting"
done

if [[ "$SETTING" == both ]]; then
  log "McNemar test: baseline vs semantic"
  mc="$OUT_DIR/$DATASET/$REGIME/mcnemar/$MODEL"
  mkdir -p "$(dirname "$mc")"
  "$PYTHON" "$ROOT/evaluation/mcnemar.py" --gold "$TEST_FILE" \
    --model1 "$OUT_DIR/$DATASET/$REGIME/baseline/$MODEL/predictions.json" \
    --model2 "$OUT_DIR/$DATASET/$REGIME/semantic/$MODEL/predictions.json" \
    --dataset_name "$DATASET" --over-union --output "$mc.json" | tee "$mc.txt"
fi

log "Summary: $MODEL | $DATASET | $REGIME"
for setting in "${SETTINGS[@]}"; do
  "$PYTHON" -c 'import json, sys; o = json.load(open(sys.argv[1]))["overall"]; print(f"{sys.argv[2]:9s} exact F1 = {o["exact"]["f1"]:.4f}   relaxed F1 = {o["relaxed"]["f1"]:.4f}")' \
    "$OUT_DIR/$DATASET/$REGIME/$setting/$MODEL/scores.json" "$setting"
done
