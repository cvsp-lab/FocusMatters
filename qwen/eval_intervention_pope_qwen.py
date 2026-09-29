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

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False

from intervention_manager_qwen import (
    InterventionManagerQwen, VALID_INTERVENTIONS,
)


def setup_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False; cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_samples", type=int, default=0)

    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--pope_file", type=str, required=True)
    p.add_argument("--pope_name", type=str, required=True,
        help="random/popular/adversarial")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--intervention", type=str, default="low_mean", choices=VALID_INTERVENTIONS)
    p.add_argument("--split_ratio",  type=float, default=0.25)
    p.add_argument("--target_layers", type=int, nargs="+",
        default=[19, 20, 21, 22, 23, 24, 25, 26, 27])
    return p.parse_args()


def main():
    args = parse_args()
    setup_seeds(args.seed)
    device = torch.device("cuda:0")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"config_pope_{args.pope_name}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[MODEL] Loading Qwen2.5-VL from: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    manager = InterventionManagerQwen(
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
        return processor(text=[text], images=image_inputs, videos=video_inputs,
                         padding=True, return_tensors="pt").to(device)

    generated = []
    for d in tqdm(pope_data, total=len(pope_data)):
        manager.clear()
        image_path = os.path.normpath(os.path.join(args.image_folder, d["image"]))
        inputs = _build_inputs(image_path, d["text"])
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, num_beams=1)
        gen_ids = [output_ids[i][inputs.input_ids.shape[1]:]
                   for i in range(len(output_ids))]
        output_text = processor.batch_decode(
            gen_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        rec = {"question_id": d["question_id"], "image": d["image"],
               "prompt": d["text"], "text": output_text,
               "model_id": "qwen2.5-vl-7b"}
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
