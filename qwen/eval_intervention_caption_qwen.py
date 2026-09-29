import os
import random
import argparse
import json
import time

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.backends.cudnn as cudnn

from pycocotools.coco import COCO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False

from intervention_manager_qwen import (
    InterventionManagerQwen, VALID_INTERVENTIONS,
)


vision_events = []


def patch_visual_forward(model):
    original_forward = model.visual.forward

    def timed_forward(*args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = original_forward(*args, **kwargs)
        end_event.record()
        vision_events.append((start_event, end_event))
        return result

    model.visual.forward = timed_forward
    return original_forward


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_samples", type=int, default=500)

    p.add_argument("--dataset_name", type=str, default="chair")
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--caption_file_path", type=str, required=True)
    p.add_argument("--custom_prompt", type=str,
        default="Please describe this image in detail.")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--intervention", type=str, default="low_mean", choices=VALID_INTERVENTIONS)
    p.add_argument("--split_ratio",  type=float, default=0.25)
    p.add_argument("--target_layers", type=int, nargs="+",
        default=[19, 20, 21, 22, 23, 24, 25, 26, 27])
    p.add_argument("--save_time", type=str, default=None,
        help="Path to JSON file to save inference time")
    return p.parse_args()


def main():
    args = parse_args()
    setup_seeds(args.seed)
    device = torch.device("cuda:0")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[MODEL] Loading Qwen2.5-VL from: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    patch_visual_forward(model)

    manager = InterventionManagerQwen(
        model=model,
        target_layers=args.target_layers,
        split_ratio=args.split_ratio,
        intervention=args.intervention,
    )

    if args.dataset_name == 'chair':
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

    caption_file = os.path.join(args.output_dir, "captions.jsonl")
    ans_file = open(caption_file, "w")
    print(f"[Intervention] {args.intervention}, split_ratio={args.split_ratio}")
    print(f"Start generation... Total: {len(questions)}")
    start_time = time.time()
    decode_events = []

    def _build_inputs(image_path, prompt_text):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text",  "text": prompt_text},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if HAS_QWEN_VL_UTILS:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs = [Image.open(image_path).convert("RGB")]
            video_inputs = None
        return processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt").to(device)

    for line in tqdm(questions, total=len(questions)):
        qid = line["question_id"]
        image_path = os.path.normpath(os.path.join(args.image_folder, line["image"]))
        manager.clear()

        inputs = _build_inputs(image_path, line["text"])
        with torch.inference_mode():
            _de_start = torch.cuda.Event(enable_timing=True)
            _de_end   = torch.cuda.Event(enable_timing=True)
            _de_start.record()
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, num_beams=1)
            _de_end.record()
            decode_events.append((_de_start, _de_end))

        generated_ids = [
            output_ids[i][inputs.input_ids.shape[1]:]
            for i in range(len(output_ids))
        ]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        output_text = ".".join([s for s in output_text.split(".") if "unk" not in s])

        ans_file.write(json.dumps({
            "question_id": qid,
            "image":       line["image"],
            "prompt":      line["text"],
            "text":        output_text,
            "model_id":    "qwen2.5-vl-7b",
        }) + "\n")
        ans_file.flush()

        import gc; gc.collect(); torch.cuda.empty_cache()

    ans_file.close()
    manager.remove_hooks()
    elapsed_time = time.time() - start_time
    print(f"[DONE] elapsed={elapsed_time:.2f}s  → {caption_file}")

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
        print(f"Decoding (model.generate) Timing Stats")
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
