"""
CHAIR (Caption Hallucination Assessment with Image Relevance) evaluation for LLaVA-1.5.

Generates image captions using LLaVA with configurable decoding strategies and
optional DPP-based masking, then saves results for CHAIR metric computation.

Supported configurations:
  - Models: LLaVA-1.5-7B
  - Decoding: greedy, OPERA, VCD, PAI, Devils
  - Masking: origin (no masking), ours (DPP masking)

Usage:
  python eval_chair.py --decoder greedy \
      --masking_method origin --output_dir results/chair/test
"""

import os
import sys
import random
import argparse
import json
import gc

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from pycocotools.coco import COCO

# ========================================
#  Project root path setup
# ========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Custom transformers (required for OPERA / Devils decoding modifications)
CUSTOM_TRANSFORMERS = os.path.join(PROJECT_ROOT, "transformers-4.37.2", "src")
if os.path.exists(CUSTOM_TRANSFORMERS) and CUSTOM_TRANSFORMERS not in sys.path:
    sys.path.insert(0, CUSTOM_TRANSFORMERS)

# Model framework imports (config, registry, processors)
from model_utils.models import load_preprocess
from model_utils.common.config import Config
from model_utils.common.registry import registry
from model_utils.datasets.builders import *
from model_utils.models import *
from model_utils.processors import *
from model_utils.tasks import *

# VCD decoding requires adding diffusion noise to create contrastive image
from decoder_zoo.VCD.vcd_utils.vcd_add_noise import add_diffusion_noise
from transformers import StoppingCriteriaList, MaxLengthCriteria

# DPP masking (proposed method)
try:
    from model_utils.models.dpp_utils import DPPMaskManager
except ImportError:
    print("[WARN] DPPMaskManager import failed. --masking_method ours is unavailable.")

# ========================================
#  Model configuration
# ========================================
# Two-level config system:
#   1. Eval config (configs/*.yaml): model arch, merged_ckpt, datasets
#   2. Model default config (model_utils/configs/models/*.yaml): vit_model, preprocess
# The Config class merges both, with eval config overriding defaults.
MODEL_EVAL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "llava-1.5_7b_eval.yaml")

# LLaVA-1.5 chat template: image token placeholder + user question
INSTRUCTION_TEMPLATE = "USER: <ImageHere>\n<question> ASSISTANT:"


def setup_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def save_args_to_json(args, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA CHAIR Evaluation")

    # Decoding
    parser.add_argument("--decoder", type=str, default="greedy",
                        choices=["greedy", "opera", "vcd", "pai", "devils"])
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of COCO images to evaluate. 0 = all images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--sample", action="store_true")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="chair")
    parser.add_argument("--image_folder", type=str,
                        default=os.path.join(PROJECT_ROOT, "dataset", "val2014"))
    parser.add_argument("--caption_file_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "dataset", "annotations", "captions_val2014.json"))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbosity", action="store_true", default=True)

    # Masking mode: "origin" = no masking (baseline), "ours" = DPP masking
    parser.add_argument("--masking_method", type=str, default="origin",
                        choices=["origin", "ours"])

    # Decoder-specific parameters
    parser.add_argument("--scale_factor", type=float, default=50)      # OPERA
    parser.add_argument("--threshold", type=int, default=15)           # OPERA
    parser.add_argument("--num_attn_candidates", type=int, default=5)  # OPERA
    parser.add_argument("--penalty_weights", type=float, default=1.0)  # OPERA
    parser.add_argument("--cd_alpha", type=float, default=1)           # VCD
    parser.add_argument("--cd_beta", type=float, default=0.1)         # VCD
    parser.add_argument("--noise_step", type=int, default=500)        # VCD

    # Config system compatibility
    parser.add_argument("--options", nargs="+")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0")

    setup_seeds(args.seed)

    # ---- Load model via two-level Config system ----
    # Config loads eval YAML -> merges with model default YAML -> builds model
    args.cfg_path = MODEL_EVAL_CONFIG_PATH
    cfg = Config(args)
    model_config = cfg.model_cfg

    # If merged_ckpt is a HuggingFace repo ID (e.g. "liuhaotian/llava-v1.5-7b"),
    # keep it as-is for auto-download. Only convert to absolute path if local.
    if hasattr(model_config, 'merged_ckpt') and not os.path.isabs(model_config.merged_ckpt):
        local_path = os.path.join(PROJECT_ROOT, model_config.merged_ckpt)
        if os.path.exists(local_path):
            model_config.merged_ckpt = local_path

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device).to(torch.bfloat16)
    model.eval()

    # Load image/text preprocessors from model default config
    processor_cfg = cfg.get_config().preprocess
    vis_processors, txt_processors = load_preprocess(processor_cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    save_args_to_json(args, os.path.join(args.output_dir, 'config.json'))

    # ---- Decoding strategy flags ----
    opera_decoding = (args.decoder == "opera")
    vcd_decoding = (args.decoder == "vcd")
    pai_decoding = (args.decoder == "pai")
    devils_decoding = (args.decoder == "devils")
    beam_search = opera_decoding
    output_attentions = opera_decoding  # OPERA needs attention maps for penalty
    num_beams = 5 if opera_decoding else args.beam

    # ---- Initialize DPP masking (proposed method) ----
    # Hooks into ViT encoder layers to:
    #   1. Capture attention maps from source_layers (6-10)
    #   2. Capture features from feature_layer (10)
    #   3. Build DPP kernel from attention * feature similarity
    #   4. Greedy MAP inference to select diverse visual tokens
    #   5. Inject mask (-inf for unselected tokens) into target_layers (11-17)
    dpp_managers = []
    if args.masking_method == "ours":
        dpp_managers.append(DPPMaskManager(
            model, visual_token_num=230,
            source_layers=[6, 7, 8, 9, 10], feature_layer=10,
            target_layers=[11, 12, 13, 14, 15, 16, 17],
        ))
        print(f"[MODEL] DPP Masking initialized: {len(dpp_managers)} managers")

    # ---- COCO dataset setup ----
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    norm = transforms.Normalize(mean, std)

    coco = COCO(args.caption_file_path)
    img_ids = coco.getImgIds()
    sampled_ids = sorted(img_ids)[:args.num_samples] if args.num_samples > 0 else sorted(img_ids)
    questions = [{"question_id": iid, "image": coco.loadImgs(iid)[0]["file_name"],
                  "text": "Please describe this image in detail."} for iid in sampled_ids]

    # ---- Caption generation loop ----
    answers_file = os.path.join(args.output_dir, 'captions.jsonl')
    ans_file = open(answers_file, "w")

    print(f"\n[GEN] Total: {len(questions)}, model: llava-1.5-7b, "
          f"decoder: {args.decoder}, masking: {args.masking_method}")

    for line in tqdm(questions, total=len(questions)):
        question_id, cur_prompt, image_file = line["question_id"], line["text"], line["image"]
        image_path = os.path.normpath(os.path.join(args.image_folder, image_file))

        raw_image = Image.open(image_path).convert('RGB')
        template = INSTRUCTION_TEMPLATE
        prompt = template.replace("<question>", cur_prompt)

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device, torch.bfloat16)

        # VCD: create noisy contrastive image via diffusion noise
        image_cd = None
        if vcd_decoding:
            image_cd = add_diffusion_noise(image, args.noise_step)
            image_cd = image_cd.unsqueeze(0).to(torch.bfloat16).cuda() if image_cd is not None else None

        # Clear DPP mask state before each image (mask is recomputed per image)
        if args.masking_method == "ours":
            for mgr in dpp_managers:
                mgr.clear()

        # Generate caption with selected decoding strategy
        with torch.inference_mode():
            out = model.generate(
                {"image": norm(image), "prompt": prompt, "img_path": image_path},
                output_attentions=output_attentions,
                num_beams=num_beams,
                max_new_tokens=args.max_new_tokens,
                use_nucleus_sampling=args.sample,
                beam_search=beam_search,
                opera_decoding=opera_decoding,
                vcd_decoding=vcd_decoding,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
                images_cd=image_cd,
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
                pai_decoding=pai_decoding,
                devils_decoding=devils_decoding,
            )

        if args.masking_method == "ours":
            gc.collect()
            torch.cuda.empty_cache()

        # Filter out sentences containing "unk" tokens
        output_text = out[0]
        output_text = ".".join(s for s in output_text.split(".") if "unk" not in s)

        ans_file.write(json.dumps({
            "question_id": question_id, "image": image_file,
            "prompt": cur_prompt, "text": output_text, "model_id": "llava-1.5-7b",
        }) + "\n")
        ans_file.flush()

    # ---- Cleanup: remove hooks to restore original model ----
    if args.masking_method == "ours":
        for mgr in dpp_managers:
            mgr.remove_hooks()

    ans_file.close()

    print("[DONE] CHAIR caption generation finished.")


if __name__ == "__main__":
    main()
