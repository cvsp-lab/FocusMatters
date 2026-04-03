#!/bin/bash
# =============================================================
# POPE (Polling-based Object Probing Evaluation)
# =============================================================
#
# Usage:
#   cd code_for_submit/
#   bash evaluation/run_pope.sh --masking origin
#   bash evaluation/run_pope.sh --masking ours
#
# Prerequisites:
#   - COCO val2014 images in dataset/val2014/
#   - POPE question files in dataset/pope/coco/coco_pope_{random,popular,adversarial}.json
#   - Model weights auto-downloaded from HuggingFace
#
# Configuration:
#   - GPU_ID: GPU device index to use
#   - NUM_SAMPLES: number of POPE questions to evaluate (0 = all)
#   - methods: list of decoding strategies to evaluate
#   - MASKING: "origin" (no masking) or "ours" (DPP masking)
#
# Output structure (per decoding method):
#   results/pope/<DATE>/<EXP_NAME>/
#     ├── config.json                       # experiment configuration
#     ├── captions_pope_random.jsonl         # generated answers for random split
#     ├── captions_pope_popular.jsonl        # generated answers for popular split
#     ├── captions_pope_adversarial.jsonl    # generated answers for adversarial split
#     ├── output_random.json                # post-processed answers (random)
#     ├── output_random_label.json          # ground truth labels (random)
#     ├── output_popular.json               # post-processed answers (popular)
#     ├── output_popular_label.json         # ground truth labels (popular)
#     ├── output_adversarial.json           # post-processed answers (adversarial)
#     ├── output_adversarial_label.json     # ground truth labels (adversarial)
#     └── pope_metric_results.json          # unified metrics for all 3 splits
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
NUM_SAMPLES=0

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

POPE_TYPES=("random" "popular" "adversarial")

for METHOD in "${methods[@]}"; do
    if [ "$MASKING" = "ours" ]; then
        EXP=llava-1.5-7b_${METHOD}_token${MAX_NEW_TOKENS}_${SEED}_n${NUM_SAMPLES}_dppmask
    else
        EXP=llava-1.5-7b_${METHOD}_token${MAX_NEW_TOKENS}_${SEED}_n${NUM_SAMPLES}
    fi

    RESULT_PATH="${ROOT}/results/pope/${DATE}/${EXP}"
    mkdir -p ${RESULT_PATH}

    # Step 1: Generate answers for each POPE split (random, popular, adversarial)
    for POPE_TYPE in "${POPE_TYPES[@]}"; do
        POPE_FILE="${ROOT}/dataset/pope/coco/coco_pope_${POPE_TYPE}.json"

        echo "=============================="
        echo " POPE: llava-1.5-7b / ${METHOD} / ${MASKING} / ${POPE_TYPE}"
        echo " Output: ${RESULT_PATH}"
        echo "=============================="

        CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/eval_pope.py \
            --decoder ${METHOD} \
            --masking_method ${MASKING} \
            --image_folder ${IMAGE_FOLDER} \
            --caption_file_path ${POPE_FILE} \
            --pope_type ${POPE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --seed ${SEED} \
            --output_dir ${RESULT_PATH}
    done

    # Step 2: Compute unified POPE metrics across all 3 splits
    python eval_utils/pope.py \
        --ans_file ${RESULT_PATH}/output_random ${RESULT_PATH}/output_popular ${RESULT_PATH}/output_adversarial \
        --pope_type random popular adversarial \
        --pope_result_file ${RESULT_PATH}/pope_metric_results.json

    echo "[DONE] ${EXP}: ${RESULT_PATH}/pope_metric_results.json"
done

echo "=== All POPE evaluations completed! ==="
