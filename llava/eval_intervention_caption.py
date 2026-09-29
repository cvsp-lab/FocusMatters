import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import json
import time
from PIL import Image
import torch
import torch.nn.functional as F
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

from pycocotools.coco import COCO

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
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--num_samples", type=int, default=500)
parser.add_argument("--options",     nargs="+")

parser.add_argument("--dataset_name", type=str, default="chair")
parser.add_argument("--image_folder", type=str, required=True)
parser.add_argument("--caption_file_path", type=str, required=True)
parser.add_argument("--output_dir",   type=str, required=True)

parser.add_argument("--intervention", type=str, default="low_mean", choices=VALID_INTERVENTIONS)
parser.add_argument("--split_ratio",  type=float, default=0.25)
parser.add_argument("--target_layers", type=int, nargs="+",
    default=[11, 12, 13, 14, 15, 16, 17])
parser.add_argument("--save_time", type=str, default=None,
    help="Path to JSON file to save inference time")

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
with open(os.path.join(args.output_dir, "config.json"), "w") as f:
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


questions = []
if args.dataset_name == 'chair':
    coco = COCO(args.caption_file_path)
    img_ids = coco.getImgIds()
    sampled = sorted(img_ids)[:args.num_samples] if args.num_samples > 0 else sorted(img_ids)
    for img_id in sampled:
        fname = coco.loadImgs(img_id)[0]["file_name"]
        questions.append({"question_id": img_id, "image": fname,
                          "text": "Please describe this image in detail."})
else:
    raise ValueError(f"Unsupported dataset: {args.dataset_name}")

answers_file = os.path.join(args.output_dir, "captions.jsonl")
ans_file = open(answers_file, "w")
print(f"[Intervention] {args.intervention}, split_ratio={args.split_ratio}")
print(f"Start generation... Total: {len(questions)}")
start_time = time.time()


def _load_image(line):
    image_path = os.path.normpath(os.path.join(args.image_folder, line["image"]))
    raw = Image.open(image_path).convert("RGB")
    img = vis_processors["eval"](raw).unsqueeze(0).to(device, torch.bfloat16)
    return img, image_path


decode_events = []


def _generate(image, prompt, image_path):
    _de_start = torch.cuda.Event(enable_timing=True)
    _de_end   = torch.cuda.Event(enable_timing=True)
    _de_start.record()
    out = model.generate(
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
        cd_alpha=1,
        cd_beta=0.1,
        use_ours_cd=False,
        ours_mask_cd=None,
        pai_decoding=False,
        devils_decoding=False,
        output_attentions=False,
    )
    _de_end.record()
    decode_events.append((_de_start, _de_end))
    return out


template = INSTRUCTION_TEMPLATE[args.model]
for line in tqdm(questions, total=len(questions)):
    manager.clear()
    img, image_path = _load_image(line)
    prompt = template.replace("<question>", line["text"])

    with torch.inference_mode():
        out = _generate(img, prompt, image_path)

    text = out[0]
    text = ".".join([s for s in text.split(".") if "unk" not in s])
    ans_file.write(json.dumps({
        "question_id": line["question_id"],
        "image":       line["image"],
        "prompt":      line["text"],
        "text":        text,
        "model_id":    args.model,
    }) + "\n")
    ans_file.flush()

    import gc; gc.collect(); torch.cuda.empty_cache()

manager.remove_hooks()
ans_file.close()
print(f"[DONE] {answers_file}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"[TIME] Total elapsed: {elapsed_time:.2f}s")

torch.cuda.synchronize()

events_list = []
if hasattr(model, 'vision_events'):
    events_list = model.vision_events
    print("Found vision_events on model directly")
elif hasattr(model, 'llama_model') and hasattr(model.llama_model, 'vision_events'):
    events_list = model.llama_model.vision_events
    print("Found vision_events on model.llama_model")
elif hasattr(model, 'llama_model') and hasattr(model.llama_model, 'model') and hasattr(model.llama_model.model, 'vision_events'):
    events_list = model.llama_model.model.vision_events
    print("Found vision_events on model.llama_model.model")

avg_vision_time = 0.0
total_vision_time = 0.0
if events_list:
    vision_times = [start.elapsed_time(end) / 1000.0 for start, end in events_list]
    avg_vision_time = sum(vision_times) / len(vision_times)
    total_vision_time = sum(vision_times)
    print("\n" + "="*50)
    print(f"Vision Encoder Timing Stats")
    print(f"Total Samples: {len(vision_times)}")
    print(f"Average Time per sample: {avg_vision_time:.4f} sec")
    print(f"Total Time (Vision only): {total_vision_time:.4f} sec")
    print("="*50 + "\n")

decode_times = [s.elapsed_time(e) / 1000.0 for s, e in decode_events]
total_decode_time = sum(decode_times)
avg_decode_time = total_decode_time / len(decode_times) if decode_times else 0.0
if decode_times:
    print("\n" + "="*50)
    print(f"Decoding (model.generate) Timing Stats")
    print(f"Total Samples: {len(decode_times)}")
    print(f"Average Time per sample: {avg_decode_time:.4f} sec")
    print(f"Total Time (Decode only): {total_decode_time:.4f} sec")
    print("="*50 + "\n")

if args.save_time:
    time_data = {
        "elapsed_time_sec": elapsed_time
    }
    if events_list:
        time_data["avg_vision_time_sec"] = avg_vision_time
        time_data["total_vision_time_sec"] = total_vision_time
        time_data["num_vision_calls"] = len(events_list)
    if decode_times:
        time_data["avg_decode_time_sec"] = avg_decode_time
        time_data["total_decode_time_sec"] = total_decode_time
        time_data["per_sample_decode_times_sec"] = decode_times

    os.makedirs(os.path.dirname(args.save_time), exist_ok=True)
    with open(args.save_time, 'w') as f:
        json.dump(time_data, f, indent=2, ensure_ascii=False)
    print(f"[TIME] Saved timing stats to: {args.save_time}")
