import os
import sys
import random
import argparse
import json
import time

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from pycocotools.coco import COCO
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from intervention_manager_internvl import (
    InterventionManagerInternVL, VALID_INTERVENTIONS,
)


vision_events = []


def patch_vision_forward(model):
    if hasattr(model, "extract_feature"):
        original = model.extract_feature

        def timed(*args, **kwargs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            result = original(*args, **kwargs)
            e.record()
            vision_events.append((s, e))
            return result

        model.extract_feature = timed
        return "extract_feature"

    if hasattr(model, "vision_model"):
        original = model.vision_model.forward

        def timed_forward(*args, **kwargs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            result = original(*args, **kwargs)
            e.record()
            vision_events.append((s, e))
            return result

        model.vision_model.forward = timed_forward
        return "vision_model.forward"

    return None


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
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--max_num_patches", type=int, default=6,
        help="max_num for dynamic_preprocess (1=single 448x448 patch)")

    p.add_argument("--dataset_name", type=str, default="chair")
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--caption_file_path", type=str, required=True)
    p.add_argument("--custom_prompt", type=str,
        default="<image>\nPlease describe this image in detail.")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--intervention", type=str, default="low_mean",
        choices=VALID_INTERVENTIONS)
    p.add_argument("--split_ratio", type=float, default=0.25)
    p.add_argument("--target_layers", type=int, nargs="+",
        default=[9, 10, 11, 12, 13, 14, 15])
    p.add_argument("--save_time", type=str, default=None,
        help="Path to JSON file to save inference time")
    return p.parse_args()


def setup_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False; cudnn.deterministic = True


def main():
    args = parse_args()
    setup_seeds(args.seed)

    dtype_map = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    model_dtype = dtype_map[args.dtype]
    device = torch.device("cuda:0")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[MODEL] Loading InternVL2.5 from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().to(device)

    patched = patch_vision_forward(model)
    print(f"[TIME] Vision patched on: {patched}")

    manager = InterventionManagerInternVL(
        model=model,
        target_layers=args.target_layers,
        split_ratio=args.split_ratio,
        intervention=args.intervention,
    )

    if args.dataset_name == "chair":
        coco = COCO(args.caption_file_path)
        img_ids = sorted(coco.getImgIds())
        if args.num_samples > 0:
            img_ids = img_ids[:args.num_samples]
        questions = [{"question_id": iid,
                      "image": coco.loadImgs(iid)[0]["file_name"],
                      "text": args.custom_prompt}
                     for iid in img_ids]
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    cap_file = os.path.join(args.output_dir, "captions.jsonl")
    ans_file = open(cap_file, "w")
    print(f"[Intervention] {args.intervention}, split_ratio={args.split_ratio}")
    print(f"Start generation... Total: {len(questions)}")
    start_time = time.time()
    decode_events = []

    gen_cfg = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    for line in tqdm(questions, total=len(questions)):
        qid = line["question_id"]
        prompt = line["text"]
        image_path = os.path.normpath(os.path.join(args.image_folder, line["image"]))

        manager.clear()

        pixel_values = load_image(
            image_path, max_num=args.max_num_patches).to(model_dtype).cuda()

        with torch.inference_mode():
            _de_start = torch.cuda.Event(enable_timing=True)
            _de_end   = torch.cuda.Event(enable_timing=True)
            _de_start.record()
            output_text = model.chat(tokenizer, pixel_values, prompt, gen_cfg)
            _de_end.record()
            decode_events.append((_de_start, _de_end))

        output_text = ".".join(
            [s for s in output_text.split(".") if "unk" not in s]).strip()

        ans_file.write(json.dumps({
            "question_id": qid,
            "image": line["image"],
            "prompt": prompt,
            "text": output_text,
            "model_id": "InternVL2_5-8B",
        }) + "\n")
        ans_file.flush()

        import gc; gc.collect(); torch.cuda.empty_cache()

    ans_file.close()
    manager.remove_hooks()
    elapsed_time = time.time() - start_time
    print(f"[DONE] elapsed={elapsed_time:.1f}s  → {cap_file}")

    torch.cuda.synchronize()

    avg_vision_time = 0.0
    total_vision_time = 0.0
    if vision_events:
        vision_times = [s.elapsed_time(e) / 1000.0 for s, e in vision_events]
        avg_vision_time = sum(vision_times) / len(vision_times)
        total_vision_time = sum(vision_times)
        print("\n" + "="*50)
        print(f"Vision Encoder Timing Stats")
        print(f"Total Calls: {len(vision_times)}")
        print(f"Average Time per call: {avg_vision_time:.4f} sec")
        print(f"Total Time (Vision only): {total_vision_time:.4f} sec")
        print("="*50 + "\n")
    else:
        print("[WARNING] No vision_events recorded!")

    decode_times = [s.elapsed_time(e) / 1000.0 for s, e in decode_events]
    total_decode_time = sum(decode_times)
    avg_decode_time = total_decode_time / len(decode_times) if decode_times else 0.0
    if decode_times:
        print("\n" + "="*50)
        print(f"Decoding (model.chat) Timing Stats")
        print(f"Total Samples: {len(decode_times)}")
        print(f"Average Time per sample: {avg_decode_time:.4f} sec")
        print(f"Total Time (Decode only): {total_decode_time:.4f} sec")
        print("="*50 + "\n")

    if args.save_time:
        time_data = {
            "elapsed_time_sec": elapsed_time
        }
        if vision_events:
            time_data["avg_vision_time_sec"] = avg_vision_time
            time_data["total_vision_time_sec"] = total_vision_time
            time_data["num_vision_calls"] = len(vision_events)
        if decode_times:
            time_data["avg_decode_time_sec"] = avg_decode_time
            time_data["total_decode_time_sec"] = total_decode_time
            time_data["per_sample_decode_times_sec"] = decode_times

        os.makedirs(os.path.dirname(args.save_time), exist_ok=True)
        with open(args.save_time, 'w') as f:
            json.dump(time_data, f, indent=2, ensure_ascii=False)
        print(f"[TIME] Saved timing stats to: {args.save_time}")


if __name__ == "__main__":
    main()
