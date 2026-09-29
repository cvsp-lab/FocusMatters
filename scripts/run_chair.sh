#!/bin/bash

set -e

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
DEPS=$BASE_DIR/dependencies
PY_LLAVA="$DEPS/transformers-4.37.2/src:$DEPS"
PY_QWEN="$DEPS/Qwen_dependency/transformers-4.49.0:$DEPS/Qwen_dependency:$BASE_DIR/qwen"

GPU_ID=${GPU_ID:-0}
SEED=${SEED:-42}
NUM_SAMPLES=${NUM_SAMPLES:-500}
SPLIT_RATIO=${SPLIT_RATIO:-0.25}
INTERVENTION=${INTERVENTION:-low_mean}
if [ "$INTERVENTION" = "original" ]; then
    RUN_TAG=original
else
    RUN_TAG=${INTERVENTION}_sr${SPLIT_RATIO}
fi
MODELS=${MODELS:-"llava-1.5-7b qwen2.5-vl-7b internvl2.5-8b"}

IMAGE_FOLDER=${IMAGE_FOLDER:-$BASE_DIR/dataset/val2014}
CAPTION_FILE=${CAPTION_FILE:-$BASE_DIR/dataset/annotations/captions_val2014.json}
CHAIR_CACHE=${CHAIR_CACHE:-$BASE_DIR/dataset/chair.pkl}
COCO_PATH=${COCO_PATH:-$BASE_DIR/dataset/annotations}

LLAVA_7B_CKPT=${LLAVA_7B_CKPT:-liuhaotian/llava-v1.5-7b}
LLAVA_13B_CKPT=${LLAVA_13B_CKPT:-liuhaotian/llava-v1.5-13b}
QWEN_CKPT=${QWEN_CKPT:-Qwen/Qwen2.5-VL-7B-Instruct}
INTERNVL_CKPT=${INTERNVL_CKPT:-OpenGVLab/InternVL2_5-8B}

LLAVA_TARGET_LAYERS=${LLAVA_TARGET_LAYERS:-"11 12 13 14 15 16 17"}
QWEN_TARGET_LAYERS=${QWEN_TARGET_LAYERS:-"19 20 21 22 23 24 25 26 27"}
INTERNVL_TARGET_LAYERS=${INTERNVL_TARGET_LAYERS:-"9 10 11 12 13 14 15"}

OUT_BASE=$BASE_DIR/results/chair
mkdir -p "$OUT_BASE"

CHAIR_SCRIPT=$BASE_DIR/eval_utils/chair.py

echo "============================================================"
echo "  CHAIR | intervention=$INTERVENTION  split_ratio=$SPLIT_RATIO"
echo "  models : $MODELS"
echo "  output : $OUT_BASE"
echo "============================================================"

run_llava() {
    local TAG=$1 CKPT=$2
    local OUT_DIR=$OUT_BASE/$TAG/${RUN_TAG}
    mkdir -p "$OUT_DIR"
    echo "[$TAG] generating captions..."
    cd "$BASE_DIR/llava"
    PYTHONPATH="$PY_LLAVA:$BASE_DIR/llava:$PYTHONPATH" \
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_intervention_caption.py \
        --model            $TAG \
        --merged_ckpt      "$CKPT" \
        --image_folder     "$IMAGE_FOLDER" \
        --caption_file_path "$CAPTION_FILE" \
        --num_samples      $NUM_SAMPLES \
        --seed             $SEED \
        --max_new_tokens   512 \
        --output_dir       "$OUT_DIR" \
        --intervention     $INTERVENTION \
        --split_ratio      $SPLIT_RATIO \
        --target_layers    $LLAVA_TARGET_LAYERS \
        --save_time        "$OUT_DIR/time.json"
    cd "$BASE_DIR"
    score_chair "$OUT_DIR"
}

run_qwen() {
    local OUT_DIR=$OUT_BASE/qwen2.5-vl-7b/${RUN_TAG}
    mkdir -p "$OUT_DIR"
    echo "[Qwen2.5-VL-7b] generating captions..."
    cd "$BASE_DIR/qwen"
    PYTHONPATH="$PY_QWEN:$PYTHONPATH" \
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_intervention_caption_qwen.py \
        --model_path        "$QWEN_CKPT" \
        --image_folder      "$IMAGE_FOLDER" \
        --caption_file_path "$CAPTION_FILE" \
        --num_samples       $NUM_SAMPLES \
        --seed              $SEED \
        --max_new_tokens    512 \
        --output_dir        "$OUT_DIR" \
        --intervention      $INTERVENTION \
        --split_ratio       $SPLIT_RATIO \
        --target_layers     $QWEN_TARGET_LAYERS \
        --save_time         "$OUT_DIR/time.json"
    cd "$BASE_DIR"
    score_chair "$OUT_DIR"
}

run_internvl() {
    local OUT_DIR=$OUT_BASE/internvl2.5-8b/${RUN_TAG}
    mkdir -p "$OUT_DIR"
    echo "[InternVL2.5-8b] generating captions..."
    cd "$BASE_DIR/internvl"
    PYTHONPATH="$BASE_DIR/internvl:$PYTHONPATH" \
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_intervention_caption_internvl.py \
        --model_path        "$INTERNVL_CKPT" \
        --image_folder      "$IMAGE_FOLDER" \
        --caption_file_path "$CAPTION_FILE" \
        --num_samples       $NUM_SAMPLES \
        --seed              $SEED \
        --max_new_tokens    512 \
        --output_dir        "$OUT_DIR" \
        --intervention      $INTERVENTION \
        --split_ratio       $SPLIT_RATIO \
        --target_layers     $INTERNVL_TARGET_LAYERS \
        --save_time         "$OUT_DIR/time.json"
    cd "$BASE_DIR"
    score_chair "$OUT_DIR"
}

score_chair() {
    local OUT_DIR=$1
    if [ -f "$OUT_DIR/captions.jsonl" ]; then
        echo "[CHAIR] scoring → $OUT_DIR/chair_eval_full.json"
        python "$CHAIR_SCRIPT" \
            --cap_file       "$OUT_DIR/captions.jsonl" \
            --image_id_key   question_id \
            --caption_key    text \
            --cache          "$CHAIR_CACHE" \
            --coco_path      "$COCO_PATH" \
            --save_path      "$OUT_DIR/chair_eval_full.json"
    fi
}

for MODEL in $MODELS; do
    case "$MODEL" in
        llava-1.5-7b)   run_llava    llava-1.5-7b  "$LLAVA_7B_CKPT"  ;;
        llava-1.5-13b)  run_llava    llava-1.5-13b "$LLAVA_13B_CKPT" ;;
        qwen2.5-vl-7b)  run_qwen     ;;
        internvl2.5-8b) run_internvl ;;
        *) echo "[ERROR] Unknown model: $MODEL" >&2; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Done. Results under: $OUT_BASE"
echo "============================================================"
