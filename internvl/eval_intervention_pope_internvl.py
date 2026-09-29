import os, sys, random, argparse, json, time
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from intervention_manager_internvl import (
    InterventionManagerInternVL, VALID_INTERVENTIONS,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff, best_ratio = float("inf"), (1, 1)
    area = width * height
    for r in target_ratios:
        diff = abs(aspect_ratio - r[0] / r[1])
        if diff < best_diff:
            best_diff, best_ratio = diff, r
        elif diff == best_diff:
            if area > 0.5 * image_size * image_size * r[0] * r[1]:
                best_ratio = r
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    w, h = image.size
    aspect_ratio = w / h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
                for i in range(1, n + 1) for j in range(1, n + 1)
                if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    tr = find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size)
    tw, th = image_size * tr[0], image_size * tr[1]
    blocks = tr[0] * tr[1]
    resized = image.resize((tw, th))
    out = []
    for i in range(blocks):
        box = (
            (i % (tw // image_size)) * image_size,
            (i // (tw // image_size)) * image_size,
            ((i % (tw // image_size)) + 1) * image_size,
            ((i // (tw // image_size)) + 1) * image_size,
        )
        out.append(resized.crop(box))
    if use_thumbnail and len(out) != 1:
        out.append(image.resize((image_size, image_size)))
    return out

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--max_num_patches", type=int, default=6)

    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--pope_file", type=str, required=True)
    p.add_argument("--pope_name", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--intervention", type=str, default="low_mean", choices=VALID_INTERVENTIONS)
    p.add_argument("--split_ratio",  type=float, default=0.25)
    p.add_argument("--target_layers", type=int, nargs="+",
        default=[9, 10, 11, 12, 13, 14, 15])
    return p.parse_args()


def setup_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False; cudnn.deterministic = True


def main():
    args = parse_args()
    setup_seeds(args.seed)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    model_dtype = dtype_map[args.dtype]
    device = torch.device("cuda:0")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"config_pope_{args.pope_name}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[MODEL] Loading InternVL2.5 from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=model_dtype,
        low_cpu_mem_usage=True, use_flash_attn=False,
        trust_remote_code=True,
    ).eval().to(device)

    manager = InterventionManagerInternVL(
        model=model,
        target_layers=args.target_layers,
        split_ratio=args.split_ratio,
        intervention=args.intervention,
    )

    with open(args.pope_file, "r") as f:
        pope_data = [json.loads(l) for l in f if l.strip()]
    if args.num_samples > 0:
        pope_data = pope_data[:args.num_samples]
    print(f"[POPE] loaded {len(pope_data)} from {args.pope_file}")

    cap_file = os.path.join(args.output_dir, f"captions_pope_{args.pope_name}.jsonl")
    ans_file = open(cap_file, "w")
    print(f"[Intervention] {args.intervention}, sr={args.split_ratio}, pope={args.pope_name}")
    start = time.time()
    gen_cfg = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    generated = []
    for d in tqdm(pope_data, total=len(pope_data)):
        manager.clear()
        image_path = os.path.normpath(os.path.join(args.image_folder, d["image"]))
        prompt = "<image>\n" + d["text"]
        pixel_values = load_image(image_path,
                                  max_num=args.max_num_patches).to(model_dtype).cuda()
        with torch.inference_mode():
            output_text = model.chat(tokenizer, pixel_values, prompt, gen_cfg)
        rec = {"question_id": d["question_id"], "image": d["image"],
               "prompt": d["text"], "text": output_text,
               "model_id": "InternVL2_5-8B"}
        ans_file.write(json.dumps(rec) + "\n"); ans_file.flush()
        generated.append(rec)
        import gc; gc.collect(); torch.cuda.empty_cache()

    ans_file.close()
    manager.remove_hooks()
    print(f"[DONE] elapsed={time.time()-start:.1f}s → {cap_file}")

    out_prefix = os.path.join(args.output_dir, f"output_{args.pope_name}")
    eval_results = []
    for g in generated:
        text = g["text"].lower().replace("\n", " ").strip()
        if "assistant:" in text:
            text = text.split("assistant:")[-1].strip()
        real_answer = "yes" if "yes" in text else ("no" if "no" in text else None)
        eval_results.append({"image_path": g["image"], "question": g["prompt"],
                             "answer": real_answer, "model_answer": text})
    with open(f"{out_prefix}.json", "w") as fp:
        for e in eval_results: fp.write(json.dumps(e) + "\n")
    with open(f"{out_prefix}_label.json", "w") as fp:
        for d in pope_data: fp.write(json.dumps({"label": d["label"]}) + "\n")
    print(f"[POPE] {out_prefix}.json + _label.json")


if __name__ == "__main__":
    main()
