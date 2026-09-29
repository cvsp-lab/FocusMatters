# Focus Matters: Attention-Value Dynamics for Hallucination Mitigation in Vision-Language Models

> This package provides code for reproducing the results in Table 2 of the main paper for LLaVA-1.5 (7B and 13B), Qwen-2.5-7B and InternVL-2.5-8B, covering both the origin baseline and ours. It includes evaluation scripts for CHAIR (CHAIR_I / CHAIR_S) and POPE (Random, Popular, and Adversarial settings) on the COCO val2014 dataset.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate focusmatters
pip install --no-deps --no-build-isolation git+https://github.com/clips/pattern.git
pip install future "backports.csv"
```

## Data Preparation

Place the following under `dataset/`:

```
dataset/
├── val2014/                                # COCO val2014 images (.jpg)
├── annotations/captions_val2014.json       # COCO caption annotations
├── pope/coco/coco_pope_random.json
├── pope/coco/coco_pope_popular.json
├── pope/coco/coco_pope_adversarial.json
└── chair.pkl                               # CHAIR vocabulary cache
```

- COCO val2014 + annotations: <https://cocodataset.org/>
- POPE JSON files: <https://github.com/RUCAIBox/POPE>
- `chair.pkl`: built from COCO synonym lists; see <https://github.com/LisaAnne/Hallucination>

## Model Checkpoints

The shell scripts download models via HuggingFace by default. To use a local
copy, override the corresponding env variable:

| Model            | Default ID                          | Override              |
|------------------|-------------------------------------|-----------------------|
| LLaVA-1.5-7b     | `liuhaotian/llava-v1.5-7b`          | `LLAVA_7B_CKPT=...`   |
| LLaVA-1.5-13b    | `liuhaotian/llava-v1.5-13b`         | `LLAVA_13B_CKPT=...`  |
| Qwen2.5-VL-7B    | `Qwen/Qwen2.5-VL-7B-Instruct`       | `QWEN_CKPT=...`       |
| InternVL2.5-8B   | `OpenGVLab/InternVL2_5-8B`          | `INTERNVL_CKPT=...`   |


## Running Evaluations

Common knobs (env vars):
- `MODELS` — which model(s) to evaluate. Subset of `"llava-1.5-7b llava-1.5-13b qwen2.5-vl-7b internvl2.5-8b"`.
- `NUM_SAMPLES` — number of COCO images to sample. 500 by default; set small for a smoke test.
- `INTERVENTION` — `original` (baseline, no intervention) or `low_mean` (ours). `low_mean` by default.
- `SPLIT_RATIO` — fraction of low-attention vision tokens to flatten when `INTERVENTION=low_mean`. 0.25 by default.
- `GPU_ID` — CUDA device index to run on. 0 by default.

### CHAIR
- From the 'FocusMatters/'

```bash
bash scripts/run_chair.sh
```

Example smoke test:
```bash
MODELS="llava-1.5-13b" NUM_SAMPLES=4 INTERVENTION=low_mean SPLIT_RATIO=0.25 GPU_ID=0 bash scripts/run_chair.sh
```

### POPE
- From the 'FocusMatters/'

```bash
bash scripts/run_pope.sh
```

Example smoke test:
```bash
MODELS="llava-1.5-13b" NUM_SAMPLES=4 INTERVENTION=low_mean SPLIT_RATIO=0.25 GPU_ID=0 bash scripts/run_pope.sh
```

The script iterates over all three POPE splits (`random`, `popular`,
`adversarial`) and writes per-split outputs plus a unified
`pope_metric_results.json`.

## Output Structure

```
results/
├── chair/<model>/<run_tag>/        # run_tag = original | low_mean_sr0.25
│   ├── captions.jsonl          # one JSON per image (question_id, image, prompt, text)
│   ├── config.json             # full args snapshot
│   ├── time.json               # avg_decode_time_sec = per-image generation time (Table 2 "Time")
│   └── chair_eval_full.json    # overall_metrics: CHAIRs, CHAIRi, Recall, Precision, F1
└── pope/<model>/<run_tag>/
    ├── captions_pope_<split>.jsonl
    ├── output_<split>.json     + output_<split>_label.json
    ├── config_pope_<split>.json
    └── pope_metric_results.json    # {Accuracy,Precision,Recall,F1,Yes_ratio}_<split>
```

## Project Structure

```
FocusMatters/
├── README.md
├── environment.yml
├── scripts/
│   ├── run_chair.sh
│   └── run_pope.sh
├── llava/                  # LLaVA-1.5-7b/13b entry + intervention manager
│   └── eval_configs/               # Model YAML config (LLaVA)
├── qwen/                   # Qwen2.5-VL-7B entry + intervention manager
├── internvl/               # InternVL2.5-8B entry + intervention manager
├── eval_utils/             # CHAIR / POPE metric utilities
├── compile/                # Result-table aggregator
├── dependencies/           # Patched transformers + LLaVA framework code
│   ├── transformers-4.37.2/        # used by LLaVA
│   ├── Qwen_dependency/            # used by Qwen2.5-VL
│   └── llava_dependency/           # LLaVA framework (registry, configs, processors)
├── dataset/                # User-provided (see Data Preparation)
└── results/                # Auto-generated
```
