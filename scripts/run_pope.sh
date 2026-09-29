#!/bin/bash

set -e

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
DEPS=$BASE_DIR/dependencies
PY_LLAVA="$DEPS/transformers-4.37.2/src:$DEPS"
PY_QWEN="$DEPS/Qwen_dependency/transformers-4.49.0:$DEPS/Qwen_dependency:$BASE_DIR/qwen"

GPU_ID=${GPU_ID:-1}
SEED=${SEED:-42}
NUM_SAMPLES=${NUM_SAMPLES:-0}
SPLIT_RATIO=${SPLIT_RATIO:-0.25}
INTERVENTION=${INTERVENTION:-low_mean}
if [ "$INTERVENTION" = "original" ]; then
    RUN_TAG=original
else
    RUN_TAG=${INTERVENTION}_sr${SPLIT_RATIO}
fi
MODELS=${MODELS:-"llava-1.5-7b"}

IMAGE_FOLDER=${IMAGE_FOLDER:-$BASE_DIR/dataset/val2014}
POPE_DIR=${POPE_DIR:-$BASE_DIR/dataset/pope/coco}
POPE_RANDOM=$POPE_DIR/coco_pope_random.json
POPE_POPULAR=$POPE_DIR/coco_pope_popular.json
POPE_ADVERSARIAL=$POPE_DIR/coco_pope_adversarial.json
POPE_FILES=("$POPE_RANDOM" "$POPE_POPULAR" "$POPE_ADVERSARIAL")
POPE_NAMES=("random" "popular" "adversarial")

LLAVA_7B_CKPT=${LLAVA_7B_CKPT:-liuhaotian/llava-v1.5-7b}
LLAVA_13B_CKPT=${LLAVA_13B_CKPT:-liuhaotian/llava-v1.5-13b}
QWEN_CKPT=${QWEN_CKPT:-Qwen/Qwen2.5-VL-7B-Instruct}
INTERNVL_CKPT=${INTERNVL_CKPT:-OpenGVLab/InternVL2_5-8B}

LLAVA_TARGET_LAYERS=${LLAVA_TARGET_LAYERS:-"11 12 13 14 15 16 17"}
QWEN_TARGET_LAYERS=${QWEN_TARGET_LAYERS:-"19 20 21 22 23 24 25 26 27"}
INTERNVL_TARGET_LAYERS=${INTERNVL_TARGET_LAYERS:-"9 10 11 12 13 14 15"}

OUT_BASE=$BASE_DIR/results/pope
mkdir -p "$OUT_BASE"

POPE_EVAL=$BASE_DIR/eval_utils/pope_for_store.py

echo "============================================================"
echo "  POPE | intervention=$INTERVENTION  split_ratio=$SPLIT_RATIO"
echo "  models : $MODELS"
echo "  output : $OUT_BASE"
echo "============================================================"

run_llava_pope() {
    local TAG=$1 CKPT=$2 POPE_FILE=$3 POPE_NAME=$4 OUT_DIR=$5
    cd "$BASE_DIR/llava"
    PYTHONPATH="$PY_LLAVA:$BASE_DIR/llava:$PYTHONPATH" \
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_intervention_pope.py \
        --model            $TAG \
        --merged_ckpt      "$CKPT" \
        --image_folder     "$IMAGE_FOLDER" \
        --pope_file        "$POPE_FILE" \
        --pope_name        "$POPE_NAME" \
        --output_dir       "$OUT_DIR" \
        --num_samples      $NUM_SAMPLES \
        --seed             $SEED \
        --max_new_tokens   64 \
        --intervention     $INTERVENTION \
        --split_ratio      $SPLIT_RATIO \
        --target_layers    $LLAVA_TARGET_LAYERS
    cd "$BASE_DIR"
}

run_qwen_pope() {
    local POPE_FILE=$1 POPE_NAME=$2 OUT_DIR=$3
    cd "$BASE_DIR/qwen"
    PYTHONPATH="$PY_QWEN:$PYTHONPATH" \
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_intervention_pope_qwen.py \
        --model_path       "$QWEN_CKPT" \
        --image_folder     "$IMAGE_FOLDER" \
        --pope_file        "$POPE_FILE" \
        --pope_name        "$POPE_NAME" \
        --output_dir       "$OUT_DIR" \
        --num_samples      $NUM_SAMPLES \
        --seed             $SEED \
        --max_new_tokens   64 \
        --intervention     $INTERVENTION \
        --split_ratio      $SPLIT_RATIO \
        --target_layers    $QWEN_TARGET_LAYERS
    cd "$BASE_DIR"
}

run_internvl_pope() {
    local POPE_FILE=$1 POPE_NAME=$2 OUT_DIR=$3
    cd "$BASE_DIR/internvl"
    PYTHONPATH="$BASE_DIR/internvl:$PYTHONPATH" \
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_intervention_pope_internvl.py \
        --model_path       "$INTERNVL_CKPT" \
        --image_folder     "$IMAGE_FOLDER" \
        --pope_file        "$POPE_FILE" \
        --pope_name        "$POPE_NAME" \
        --output_dir       "$OUT_DIR" \
        --num_samples      $NUM_SAMPLES \
        --seed             $SEED \
        --max_new_tokens   64 \
        --intervention     $INTERVENTION \
        --split_ratio      $SPLIT_RATIO \
        --target_layers    $INTERNVL_TARGET_LAYERS
    cd "$BASE_DIR"
}

score_pope() {
    local OUT_DIR=$1
    echo "[POPE-eval] $OUT_DIR"
    python "$POPE_EVAL" \
        --ans_file "$OUT_DIR/output_random" "$OUT_DIR/output_popular" "$OUT_DIR/output_adversarial" \
        --pope_type random popular adversarial \
        --pope_result_file "$OUT_DIR/pope_metric_results.json"
}

for MODEL in $MODELS; do
    OUT_DIR=$OUT_BASE/$MODEL/${RUN_TAG}
    mkdir -p "$OUT_DIR"

    for i in "${!POPE_NAMES[@]}"; do
        PFILE="${POPE_FILES[$i]}"
        PNAME="${POPE_NAMES[$i]}"
        if [ -f "$OUT_DIR/output_${PNAME}.json" ] && [ -f "$OUT_DIR/output_${PNAME}_label.json" ]; then
            echo "[SKIP] $MODEL/$PNAME already done"
            continue
        fi
        case "$MODEL" in
            llava-1.5-7b)   run_llava_pope    llava-1.5-7b  "$LLAVA_7B_CKPT"  "$PFILE" "$PNAME" "$OUT_DIR" ;;
            llava-1.5-13b)  run_llava_pope    llava-1.5-13b "$LLAVA_13B_CKPT" "$PFILE" "$PNAME" "$OUT_DIR" ;;
            qwen2.5-vl-7b)  run_qwen_pope     "$PFILE" "$PNAME" "$OUT_DIR" ;;
            internvl2.5-8b) run_internvl_pope "$PFILE" "$PNAME" "$OUT_DIR" ;;
            *) echo "[ERROR] Unknown model: $MODEL" >&2; exit 1 ;;
        esac
    done

    score_pope "$OUT_DIR"
done

echo "============================================================"
echo "  Done. Results under: $OUT_BASE"
echo "============================================================"
