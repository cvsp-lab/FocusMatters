import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from llava_dependency.models import load_preprocess
from llava_dependency.common.config import Config
from llava_dependency.common.registry import registry
from llava_dependency.datasets.builders import *
from llava_dependency.models import *
from llava_dependency.processors import *
from llava_dependency.runners import *
from llava_dependency.tasks import *

from intervention_manager_llava import (
    InterventionManagerLLaVA, VALID_INTERVENTIONS,
)


MODEL_EVAL_CONFIG_PATH = {
    "llava-1.5-7b":  "eval_configs/llava-1.5_7b_eval.yaml",
    "llava-1.5-13b": "eval_configs/llava-1.5_13b_eval.yaml",
}
INSTRUCTION_TEMPLATE = {
    "llava-1.5-7b":  "USER: <ImageHere>\n<question> ASSISTANT:",
    "llava-1.5-13b": "USER: <ImageHere>\n<question> ASSISTANT:",
}


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument("--model",       type=str, default="llava-1.5-13b",
    choices=["llava-1.5-7b", "llava-1.5-13b"])
parser.add_argument("--merged_ckpt", type=str, default=None)
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--max_new_tokens", type=int, default=64)
parser.add_argument("--num_samples", type=int, default=0,
    help="0 =  ")
parser.add_argument("--options",     nargs="+")

parser.add_argument("--image_folder", type=str, required=True)
parser.add_argument("--pope_file", type=str, required=True,
    help="POPE jsonl (: coco_pope_random.json)")
parser.add_argument("--pope_name", type=str, required=True,
    help="random/popular/adversarial   — output  prefix")
parser.add_argument("--output_dir",   type=str, required=True)

parser.add_argument("--intervention", type=str, default="low_mean", choices=VALID_INTERVENTIONS)
parser.add_argument("--split_ratio",  type=float, default=0.25)
parser.add_argument("--target_layers", type=int, nargs="+",
    default=[11, 12, 13, 14, 15, 16, 17])

args = parser.parse_known_args()[0]


device = torch.device("cuda:0")
setup_seeds(args.seed)

args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device).to(torch.bfloat16)
model.eval()

processor_cfg = cfg.get_config().preprocess
vis_processors, txt_processors = load_preprocess(processor_cfg)
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, f"config_pope_{args.pope_name}.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

manager = InterventionManagerLLaVA(
    model=model,
    target_layers=args.target_layers,
    split_ratio=args.split_ratio,
    intervention=args.intervention,
)

mean_norm = (0.48145466, 0.4578275, 0.40821073)
std_norm  = (0.26862954, 0.26130258, 0.27577711)
norm_transform = transforms.Normalize(mean_norm, std_norm)


with open(args.pope_file, "r") as f:
    pope_data = [json.loads(l) for l in f if l.strip()]
if args.num_samples > 0:
    pope_data = pope_data[:args.num_samples]
print(f"[POPE] loaded {len(pope_data)} from {args.pope_file}")

caption_file = os.path.join(args.output_dir, f"captions_pope_{args.pope_name}.jsonl")
ans_file = open(caption_file, "w")


def _load_image(image_name):
    image_path = os.path.normpath(os.path.join(args.image_folder, image_name))
    raw = Image.open(image_path).convert("RGB")
    img = vis_processors["eval"](raw).unsqueeze(0).to(device, torch.bfloat16)
    return img, image_path


def _generate(image, prompt, image_path):
    return model.generate(
        {"image": norm_transform(image), "prompt": prompt, "img_path": image_path},
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
        use_nucleus_sampling=False,
        beam_search=False,
        dola_decoding=False,
        opera_decoding=False,
        vcd_decoding=False,
        scale_factor=50,
        threshold=15,
        num_attn_candidates=5,
        penalty_weights=1.0,
        images_cd=None,
        cd_alpha=1, cd_beta=0.1,
        use_ours_cd=False, ours_mask_cd=None,
        pai_decoding=False, devils_decoding=False,
        output_attentions=False,
    )


template = INSTRUCTION_TEMPLATE[args.model]
print(f"[Intervention] {args.intervention}, sr={args.split_ratio}, pope={args.pope_name}")

generated = []
for d in tqdm(pope_data, total=len(pope_data)):
    manager.clear()
    img, image_path = _load_image(d["image"])
    prompt = template.replace("<question>", d["text"])
    with torch.inference_mode():
        out = _generate(img, prompt, image_path)
    text = out[0]
    rec = {
        "question_id": d["question_id"],
        "image":       d["image"],
        "prompt":      d["text"],
        "text":        text,
        "model_id":    args.model,
    }
    ans_file.write(json.dumps(rec) + "\n")
    ans_file.flush()
    generated.append(rec)
    import gc; gc.collect(); torch.cuda.empty_cache()

manager.remove_hooks()
ans_file.close()


out_prefix = os.path.join(args.output_dir, f"output_{args.pope_name}")
eval_results = []
for g in generated:
    text = g["text"].lower().replace("\n", " ").strip()
    if "assistant:" in text:
        text = text.split("assistant:")[-1].strip()
    real_answer = "yes" if "yes" in text else ("no" if "no" in text else None)
    eval_results.append({
        "image_path":   g["image"],
        "question":     g["prompt"],
        "answer":       real_answer,
        "model_answer": text,
    })
with open(f"{out_prefix}.json", "w") as fp:
    for e in eval_results: fp.write(json.dumps(e) + "\n")
with open(f"{out_prefix}_label.json", "w") as fp:
    for d in pope_data: fp.write(json.dumps({"label": d["label"]}) + "\n")
print(f"[POPE] {out_prefix}.json + _label.json")
