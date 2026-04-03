#!/bin/bash
# =============================================================
# CHAIR (Caption Hallucination Assessment with Image Relevance)
# =============================================================
#
# Usage:
#   cd code_for_submit/
#   bash evaluation/run_chair.sh --masking origin
#   bash evaluation/run_chair.sh --masking ours
#
# Prerequisites:
#   - COCO val2014 images in dataset/val2014/
#   - COCO annotations in dataset/annotations/captions_val2014.json
#   - CHAIR cache in dataset/chair.pkl
#   - Model weights auto-downloaded from HuggingFace
#
# Configuration:
#   - GPU_ID: GPU device index to use
#   - NUM_SAMPLES: number of COCO images to evaluate (500 for full eval)
#   - methods: list of decoding strategies to evaluate
#   - MASKING: "origin" (no masking) or "ours" (DPP masking)
#
# Output structure:
#   results/chair/<DATE>/<EXP_NAME>/
#     ├── config.json              # experiment configuration
#     ├── captions.jsonl           # generated captions (one per line)
#     ├── chair_metric_results.json  # CHAIR-i, CHAIR-s metrics
#     └── captions_eval_results.json # per-caption evaluation details
#
# =============================================================

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Custom transformers path (required for OPERA and Devils decoding)
export PYTHONPATH="${ROOT}/transformers-4.37.2/src:${ROOT}:$PYTHONPATH"

GPU_ID=0
SEED=42
DATE=$(date "+%m%d_%H%M")

MAX_NEW_TOKENS=512
NUM_SAMPLES=500

# ---- Decoding strategies (greedy / opera / vcd / pai / devils) ----
methods=("greedy" "opera" "vcd" "pai" "devils")

# ---- Argument parsing ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --masking) MASKING="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ---- Masking mode: origin (baseline) / ours (DPP masking) ----
MASKING=${MASKING:-ours}

# ---- Dataset paths ----
IMAGE_FOLDER="${ROOT}/dataset/val2014"
CAPTION_FILE="${ROOT}/dataset/annotations/captions_val2014.json"

for METHOD in "${methods[@]}"; do
    if [ "$MASKING" = "ours" ]; then
        EXP=llava-1.5-7b_${METHOD}_token${MAX_NEW_TOKENS}_${SEED}_n${NUM_SAMPLES}_dppmask
    else
        EXP=llava-1.5-7b_${METHOD}_token${MAX_NEW_TOKENS}_${SEED}_n${NUM_SAMPLES}
    fi

    RESULT_PATH="${ROOT}/results/chair/${DATE}/${EXP}"
    mkdir -p ${RESULT_PATH}

    echo "=============================="
    echo " CHAIR: llava-1.5-7b / ${METHOD} / ${MASKING}"
    echo " Output: ${RESULT_PATH}"
    echo "=============================="

    # Step 1: Generate captions
    CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/eval_chair.py \
        --decoder ${METHOD} \
        --masking_method ${MASKING} \
        --image_folder ${IMAGE_FOLDER} \
        --caption_file_path ${CAPTION_FILE} \
        --num_samples ${NUM_SAMPLES} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --seed ${SEED} \
        --output_dir ${RESULT_PATH}

    # Step 2: Compute CHAIR metrics from generated captions
    python eval_utils/chair.py \
        --cap_file ${RESULT_PATH}/captions.jsonl \
        --chair_result_file ${RESULT_PATH}/chair_metric_results.json \
        --image_id_key question_id \
        --caption_key text \
        --cache ${ROOT}/dataset/chair.pkl \
        --coco_path ${ROOT}/dataset/annotations \
        --save_path ${RESULT_PATH}/captions_eval_results.json

    echo "[DONE] ${EXP}: ${RESULT_PATH}/chair_metric_results.json"
done

echo "=== All CHAIR evaluations completed! ==="
