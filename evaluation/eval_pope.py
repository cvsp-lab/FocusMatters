"""
POPE (Polling-based Object Probing Evaluation) for LLaVA-1.5.

Generates yes/no answers to object-existence questions using LLaVA with
configurable decoding strategies and optional DPP masking.
Post-processes answers into evaluation format (output_{type}.json + label file).

Supported configurations:
  - Models: LLaVA-1.5-7B
  - Decoding: greedy, OPERA, VCD, PAI, Devils
  - Masking: origin (no masking), ours (DPP masking)
  - POPE splits: random, popular, adversarial

Usage:
  python eval_pope.py --decoder greedy --masking_method origin \
      --caption_file_path dataset/pope/coco/coco_pope_random.json \
      --pope_type random --output_dir results/pope/test
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

# Model framework imports
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
MODEL_EVAL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "llava-1.5_7b_eval.yaml")

INSTRUCTION_TEMPLATE = "USER: <ImageHere>\n<question> ASSISTANT:"


def setup_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False; cudnn.deterministic = True


def save_args_to_json(args, fp):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, 'w') as f: json.dump(vars(args), f, indent=4, ensure_ascii=False)


def pope_postprocess(answers_file, caption_file, output_dir, num_samples, pope_type):
    """
    Post-process POPE answers: extract yes/no from model output,
    then save as output_{pope_type}.json + output_{pope_type}_label.json
    for metric computation by eval_utils/pope.py.
    """
    with open(caption_file, 'r') as f:
        txt_labels = [json.loads(l.strip()) for l in f]
    if num_samples > 0: txt_labels = txt_labels[:num_samples]

    with open(answers_file, 'r') as f:
        generated = [json.loads(l) for l in f]

    results, labels = [], []
    for lbl, gen in zip(txt_labels, generated):
        ans = gen["text"].lower()
        if "assistant:" in ans: ans = ans.split("assistant:")[-1]
        ans = ans.replace("\n", " ").strip()
        real = "yes" if "yes" in ans else ("no" if "no" in ans else None)
        results.append({"image_path": lbl['image'], "question": lbl['text'], "answer": real, "model_answer": ans})
        labels.append({"label": lbl['label']})

    prefix = os.path.join(output_dir, f'output_{pope_type}')
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{prefix}.json', 'w') as f:
        for e in results: f.write(json.dumps(e) + '\n')
    with open(f'{prefix}_label.json', 'w') as f:
        for e in labels: f.write(json.dumps(e) + '\n')
    print(f"[POPE] Saved: {prefix}.json / {prefix}_label.json")


def parse_args():
    p = argparse.ArgumentParser(description="LLaVA POPE Evaluation")

    # Decoding
    p.add_argument("--decoder", default="greedy", choices=["greedy", "opera", "vcd", "pai", "devils"])
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--num_samples", type=int, default=0,
                   help="Number of POPE questions to evaluate. 0 = all questions.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--sample", action="store_true")

    # Dataset
    p.add_argument("--dataset_name", default="pope")
    p.add_argument("--image_folder", default=os.path.join(PROJECT_ROOT, "dataset", "val2014"))
    p.add_argument("--caption_file_path", required=True,
                   help="Path to POPE question file (e.g. coco_pope_random.json)")
    p.add_argument("--pope_type", required=True, choices=["random", "popular", "adversarial"],
                   help="POPE split type, used for output filename")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--verbosity", action="store_true", default=True)

    # Masking mode: "origin" = no masking (baseline), "ours" = DPP masking
    p.add_argument("--masking_method", default="origin", choices=["origin", "ours"])

    # Decoder-specific parameters
    p.add_argument("--scale_factor", type=float, default=50)      # OPERA
    p.add_argument("--threshold", type=int, default=15)           # OPERA
    p.add_argument("--num_attn_candidates", type=int, default=5)  # OPERA
    p.add_argument("--penalty_weights", type=float, default=1.0)  # OPERA
    p.add_argument("--cd_alpha", type=float, default=1)           # VCD
    p.add_argument("--cd_beta", type=float, default=0.1)         # VCD
    p.add_argument("--noise_step", type=int, default=500)        # VCD

    p.add_argument("--options", nargs="+")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0")
    setup_seeds(args.seed)

    # ---- Load model via two-level Config system ----
    args.cfg_path = MODEL_EVAL_CONFIG_PATH
    cfg = Config(args)
    model_config = cfg.model_cfg

    # Keep HuggingFace repo IDs as-is; only convert local relative paths
    if hasattr(model_config, 'merged_ckpt') and not os.path.isabs(model_config.merged_ckpt):
        local_path = os.path.join(PROJECT_ROOT, model_config.merged_ckpt)
        if os.path.exists(local_path):
            model_config.merged_ckpt = local_path

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device).to(torch.bfloat16)
    model.eval()

    processor_cfg = cfg.get_config().preprocess
    vis_processors, _ = load_preprocess(processor_cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    save_args_to_json(args, os.path.join(args.output_dir, 'config.json'))

    # ---- Decoding strategy flags ----
    opera_dec = (args.decoder == "opera")
    vcd_dec = (args.decoder == "vcd")
    pai_dec = (args.decoder == "pai")
    devils_dec = (args.decoder == "devils")
    beam_search = opera_dec
    num_beams = 5 if opera_dec else args.beam

    # ---- Initialize DPP masking (proposed method) ----
    dpp_managers = []
    if args.masking_method == "ours":
        dpp_managers.append(DPPMaskManager(
            model, visual_token_num=230,
            source_layers=[6,7,8,9,10], feature_layer=10,
            target_layers=[11,12,13,14,15,16,17],
        ))
        print(f"[MODEL] DPP Masking: {len(dpp_managers)} managers")

    # ---- CLIP normalization ----
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    norm = transforms.Normalize(mean, std)

    # ---- Load POPE questions ----
    with open(args.caption_file_path, 'r') as f:
        pope_data = [json.loads(l.strip()) for l in f]
    if args.num_samples > 0: pope_data = pope_data[:args.num_samples]

    questions = [{"question_id": e["question_id"], "image": e["image"], "text": e["text"]} for e in pope_data]

    # Caption file named per pope_type for separate storage
    answers_file = os.path.join(args.output_dir, f'captions_pope_{args.pope_type}.jsonl')
    af = open(answers_file, "w")

    print(f"\n[GEN] Total: {len(questions)}, model: llava-1.5-7b, decoder: {args.decoder}, masking: {args.masking_method}")

    # ---- Answer generation loop ----
    for line in tqdm(questions):
        qid, prompt_text, imf = line["question_id"], line["text"], line["image"]
        ip = os.path.normpath(os.path.join(args.image_folder, imf))
        raw = Image.open(ip).convert('RGB')
        tmpl = INSTRUCTION_TEMPLATE
        prompt = tmpl.replace("<question>", prompt_text)
        image = vis_processors["eval"](raw).unsqueeze(0).to(device, torch.bfloat16)

        # VCD: create noisy contrastive image
        image_cd = None
        if vcd_dec:
            image_cd = add_diffusion_noise(image, args.noise_step)
            image_cd = image_cd.unsqueeze(0).to(torch.bfloat16).cuda() if image_cd is not None else None

        # Clear DPP mask state before each image
        if args.masking_method == "ours":
            for m in dpp_managers: m.clear()

        with torch.inference_mode():
            out = model.generate(
                {"image": norm(image), "prompt": prompt, "img_path": ip},
                output_attentions=opera_dec, num_beams=num_beams,
                max_new_tokens=args.max_new_tokens, use_nucleus_sampling=args.sample,
                beam_search=beam_search, opera_decoding=opera_dec, vcd_decoding=vcd_dec,
                scale_factor=args.scale_factor, threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates, penalty_weights=args.penalty_weights,
                images_cd=image_cd, cd_alpha=args.cd_alpha, cd_beta=args.cd_beta,
                pai_decoding=pai_dec, devils_decoding=devils_dec,
            )

        if args.masking_method == "ours":
            gc.collect(); torch.cuda.empty_cache()

        ot = ".".join(s for s in out[0].split(".") if "unk" not in s)
        af.write(json.dumps({"question_id": qid, "image": imf, "prompt": prompt_text,
                              "text": ot, "model_id": "llava-1.5-7b"}) + "\n")
        af.flush()

    # ---- Cleanup ----
    if args.masking_method == "ours":
        for m in dpp_managers: m.remove_hooks()

    af.close()

    # Post-process: extract yes/no answers and save evaluation files
    pope_postprocess(answers_file, args.caption_file_path, args.output_dir, args.num_samples, args.pope_type)

    print("[DONE] POPE generation finished.")


if __name__ == "__main__":
    main()
