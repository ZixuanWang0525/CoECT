#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IKS stage-1 keyframe generation from PNR outputs.

What this script does:
1) Generate the first keyframe from the ORIGINAL prompt if available.
   - It first looks for one of: initial_prompt / input_text / raw_prompt on each PNR item.
   - If none exists, it falls back to positive_prompt.
2) Generate subsequent keyframes by iteratively editing the previous keyframe
   using each PNR event_description.
3) Save ONLY keyframe images. No guide images, no paired images, no extra visual artifacts.
4) Add a strong "no visible text" negative constraint for both generation and editing.

This script is designed to be usable offline with local Qwen-Image and
Qwen-Image-Edit checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch


def patch_scaled_dot_product_attention_for_legacy_torch() -> None:
    sdp = torch.nn.functional.scaled_dot_product_attention
    if getattr(sdp, "_coect_legacy_enable_gqa_patch", False):
        return

    q = torch.randn(1, 1, 2, 4)
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 4)
    try:
        sdp(q, k, v, enable_gqa=True)
        return
    except TypeError as e:
        if "enable_gqa" not in str(e):
            return

    original_sdp = sdp

    def patched_scaled_dot_product_attention(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return original_sdp(*args, **kwargs)

    patched_scaled_dot_product_attention._coect_legacy_enable_gqa_patch = True
    torch.nn.functional.scaled_dot_product_attention = patched_scaled_dot_product_attention
    warnings.warn(
        "Patched torch.nn.functional.scaled_dot_product_attention to ignore "
        "`enable_gqa` for legacy torch compatibility.",
        RuntimeWarning,
    )


patch_scaled_dot_product_attention_for_legacy_torch()

from diffusers import DiffusionPipeline, QwenImageEditPipeline


DEFAULT_PNR_JSON = "/mnt/43t/data/pnr_outputs_globalpath_v3_nonumeric.json"
DEFAULT_QWEN_IMAGE = "/mnt/43t/hyx/model_cache/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6"
DEFAULT_QWEN_IMAGE_EDIT = "/mnt/43t/hyx/model_cache/models--Qwen--Qwen-Image-Edit/snapshots/ac7f9318f633fc4b5778c59367c8128225f1e3de"
DEFAULT_OUT_DIR = "/mnt/43t/data/iks_keyframes_v1"
DEFAULT_RATIO = "16:9"
DEFAULT_INIT_STEPS = 28
DEFAULT_EDIT_STEPS = 24
DEFAULT_INIT_CFG = 3.5
DEFAULT_EDIT_CFG = 3.5
DEFAULT_MAX_SIDE = 768


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", s).strip()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def slugify(text: str, maxlen: int = 80) -> str:
    text = norm_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "sample"
    return text[:maxlen]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_size(ratio: str) -> Tuple[int, int]:
    ratios = {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
    }
    if ratio not in ratios:
        raise ValueError(f"Unsupported ratio: {ratio}")
    return ratios[ratio]


def load_and_downscale(image_path: str, max_side: int) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    if max_side and max_side > 0:
        w, h = img.size
        s = max(w, h)
        if s > max_side:
            scale = max_side / float(s)
            new_w = max(8, (int(w * scale) // 8) * 8)
            new_h = max(8, (int(h * scale) // 8) * 8)
            img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def add_no_text_negative(base_negative: str) -> str:
    no_text = (
        "text, words, letters, numbers, captions, subtitle, subtitles, logo, watermark, "
        "signature, label, labels, typographic overlay, characters, calligraphy, handwriting"
    )
    base_negative = norm_text(base_negative)
    if not base_negative:
        return no_text
    low = base_negative.lower()
    if "text" in low or "letters" in low or "watermark" in low:
        return base_negative
    return f"{base_negative}, {no_text}"


def make_generator(device: str, seed: Optional[int]):
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(int(seed))



def choose_initial_prompt(item: Dict[str, Any]) -> Tuple[str, str]:
    for k in ("initial_prompt", "input_text", "raw_prompt"):
        v = norm_text(item.get(k))
        if v:
            return v, k
    v = norm_text(item.get("positive_prompt"))
    if v:
        return v, "positive_prompt_fallback"
    raise ValueError("No usable initial prompt found in item")


def build_init_prompt(initial_prompt: str) -> str:
    """Use original prompt directly, with a light cleanup and explicit no-text preference."""
    initial_prompt = norm_text(initial_prompt)
    suffix = " cinematic, realistic scene, no visible text in the image"
    if "no visible text" in initial_prompt.lower():
        return initial_prompt
    return initial_prompt + suffix


def build_edit_prompt(global_positive: str, prev_event_desc: Optional[str], curr_event_desc: str) -> str:
    gp = norm_text(global_positive)
    prev_d = norm_text(prev_event_desc)
    curr_d = norm_text(curr_event_desc)
    parts = [
        "Edit the image so the scene progresses naturally to the current physical event.",
        f"Global scene context: {gp}" if gp else "",
        f"Previous event: {prev_d}" if prev_d else "",
        f"Current event to depict: {curr_d}",
        "Keep the same objects, scene layout, viewpoint, and identity unless the current event requires a visible change.",
        "Preserve background consistency and realism.",
        "Do not add any visible text, words, letters, numbers, logo, subtitle, caption, label, or watermark into the image.",
    ]
    return " ".join([p for p in parts if p])


def build_qwen_image_pipe(model_path: str, device: str):
    print(f"[{now()}] [Init ] Loading Qwen-Image from: {model_path}", flush=True)
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        pipe = pipe.to(device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
    print(f"[{now()}] [Init ] Qwen-Image ready on {device} (dtype={dtype})", flush=True)
    return pipe


def build_qwen_edit_pipe(model_path: str):
    print(f"[{now()}] [Init ] Loading Qwen-Image-Edit from: {model_path}", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = QwenImageEditPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    try:
        pipe.enable_model_cpu_offload()
        print(f"[{now()}] [Init ] Qwen-Image-Edit using CPU offload", flush=True)
    except Exception as e:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        print(f"[{now()}] [Warn ] CPU offload unavailable ({e}); moved edit pipe to {device}", flush=True)
    return pipe



def infer_edit_exec_device(pipe) -> str:
    try:
        if hasattr(pipe, "_execution_device") and pipe._execution_device is not None:
            return str(pipe._execution_device)
        for name in ("transformer", "vae", "text_encoder"):
            module = getattr(pipe, name, None)
            if module is not None:
                p = next(module.parameters(), None)
                if p is not None:
                    return str(p.device)
    except Exception:
        pass
    return "cpu"



def generate_initial_keyframe(
    pipe,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    device: str,
    seed: Optional[int],
) -> Image.Image:
    gen = make_generator(device, seed)
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            true_cfg_scale=float(cfg),
            generator=gen,
        )
    return out.images[0]


def edit_next_keyframe(
    pipe,
    source_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    max_side: int,
    seed: Optional[int],
) -> Image.Image:
    exec_device = infer_edit_exec_device(pipe)
    gen = make_generator(exec_device, seed)
    if max_side and max_side > 0:
        w, h = source_image.size
        s = max(w, h)
        if s > max_side:
            scale = max_side / float(s)
            new_w = max(8, (int(w * scale) // 8) * 8)
            new_h = max(8, (int(h * scale) // 8) * 8)
            source_image = source_image.resize((new_w, new_h), Image.LANCZOS)
    with torch.inference_mode():
        out = pipe(
            image=source_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            true_cfg_scale=float(cfg),
            generator=gen,
        )
    return out.images[0]



def process_one_item(
    item: Dict[str, Any],
    image_pipe,
    edit_pipe,
    out_root: str,
    width: int,
    height: int,
    init_steps: int,
    edit_steps: int,
    init_cfg: float,
    edit_cfg: float,
    image_device: str,
    init_seed: Optional[int],
    edit_seed: Optional[int],
    max_side: int,
    resume: bool,
) -> None:
    idx = item.get("index")
    global_positive = norm_text(item.get("positive_prompt", ""))
    global_negative = add_no_text_negative(item.get("negative_prompt", ""))
    events = item.get("events", []) or []
    if not events:
        raise ValueError(f"index={idx}: no events found")

    initial_prompt, initial_source = choose_initial_prompt(item)
    folder_name = f"{int(idx):04d}_{slugify(initial_prompt or global_positive)}"
    sample_dir = os.path.join(out_root, folder_name)
    ensure_dir(sample_dir)

    print(f"[{now()}] [Item ] index={idx} | init_source={initial_source}", flush=True)
    print(f"[{now()}] [Item ] out_dir={sample_dir}", flush=True)

    init_path = os.path.join(sample_dir, "keyframe_00_init.png")
    if resume and os.path.exists(init_path):
        current_img = Image.open(init_path).convert("RGB")
        print(f"[{now()}] [Skip ] existing init frame -> {Path(init_path).name}", flush=True)
    else:
        init_prompt = build_init_prompt(initial_prompt)
        current_img = generate_initial_keyframe(
            pipe=image_pipe,
            prompt=init_prompt,
            negative_prompt=global_negative,
            width=width,
            height=height,
            steps=init_steps,
            cfg=init_cfg,
            device=image_device,
            seed=init_seed,
        )
        current_img.save(init_path)
        print(f"[{now()}] [Save ] {Path(init_path).name}", flush=True)

    prev_event_desc: Optional[str] = None
    for i, ev in enumerate(events, start=1):
        ev_id = ev.get("event_id", i)
        ev_desc = norm_text(ev.get("event_description", ""))
        if not ev_desc:
            print(f"[{now()}] [Warn ] index={idx} event_id={ev_id}: empty event_description, skipped", flush=True)
            continue
        out_name = f"keyframe_{i:02d}_event{int(ev_id):02d}.png"
        out_path = os.path.join(sample_dir, out_name)
        if resume and os.path.exists(out_path):
            current_img = Image.open(out_path).convert("RGB")
            prev_event_desc = ev_desc
            print(f"[{now()}] [Skip ] existing event frame -> {out_name}", flush=True)
            continue

        edit_prompt = build_edit_prompt(global_positive, prev_event_desc, ev_desc)
        current_img = edit_next_keyframe(
            pipe=edit_pipe,
            source_image=current_img,
            prompt=edit_prompt,
            negative_prompt=global_negative,
            steps=edit_steps,
            cfg=edit_cfg,
            max_side=max_side,
            seed=edit_seed,
        )
        current_img.save(out_path)
        prev_event_desc = ev_desc
        print(f"[{now()}] [Save ] {out_name}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IKS stage-1 keyframes from PNR results")
    parser.add_argument("--pnr_json", type=str, default=DEFAULT_PNR_JSON)
    parser.add_argument("--qwen_image_model", type=str, default=DEFAULT_QWEN_IMAGE)
    parser.add_argument("--qwen_edit_model", type=str, default=DEFAULT_QWEN_IMAGE_EDIT)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ratio", type=str, default=DEFAULT_RATIO, choices=["1:1", "16:9", "9:16", "4:3", "3:4"])
    parser.add_argument("--init_steps", type=int, default=DEFAULT_INIT_STEPS)
    parser.add_argument("--edit_steps", type=int, default=DEFAULT_EDIT_STEPS)
    parser.add_argument("--init_cfg", type=float, default=DEFAULT_INIT_CFG)
    parser.add_argument("--edit_cfg", type=float, default=DEFAULT_EDIT_CFG)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument("--init_seed", type=int, default=None)
    parser.add_argument("--edit_seed", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.pnr_json):
        raise FileNotFoundError(f"PNR json not found: {args.pnr_json}")
    if not os.path.exists(args.qwen_image_model):
        raise FileNotFoundError(f"Qwen-Image model path not found: {args.qwen_image_model}")
    if not os.path.exists(args.qwen_edit_model):
        raise FileNotFoundError(f"Qwen-Image-Edit model path not found: {args.qwen_edit_model}")

    resume = True
    if args.no_resume:
        resume = False
    elif args.resume:
        resume = True

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    width, height = choose_size(args.ratio)
    ensure_dir(args.out_dir)

    data = load_json(args.pnr_json)
    items = data.get("items", []) if isinstance(data, dict) else []
    if not items:
        raise RuntimeError("No items found in PNR json")

    if args.sample_index is not None:
        items = [x for x in items if int(x.get("index", -1)) == int(args.sample_index)]
        if not items:
            raise KeyError(f"No sample found for index={args.sample_index}")

    start = max(0, int(args.start))
    end = len(items) if args.limit <= 0 else min(len(items), start + int(args.limit))
    if start >= end:
        print(f"[{now()}] [Exit ] Nothing to do: start={start}, end={end}", flush=True)
        return
    items = items[start:end]

    print(f"[{now()}] [Init ] PNR items: {len(items)}", flush=True)
    print(f"[{now()}] [Init ] image_device={device} ratio={args.ratio} size={width}x{height}", flush=True)
    print(f"[{now()}] [Init ] output={args.out_dir}", flush=True)

    image_pipe = build_qwen_image_pipe(args.qwen_image_model, device)
    edit_pipe = build_qwen_edit_pipe(args.qwen_edit_model)

    errors = 0
    for n, item in enumerate(items, start=1):
        try:
            print(f"\n[{now()}] [Run  ] {n}/{len(items)}", flush=True)
            process_one_item(
                item=item,
                image_pipe=image_pipe,
                edit_pipe=edit_pipe,
                out_root=args.out_dir,
                width=width,
                height=height,
                init_steps=args.init_steps,
                edit_steps=args.edit_steps,
                init_cfg=args.init_cfg,
                edit_cfg=args.edit_cfg,
                image_device=device,
                init_seed=args.init_seed,
                edit_seed=args.edit_seed,
                max_side=args.max_side,
                resume=resume,
            )
        except KeyboardInterrupt:
            print(f"\n[{now()}] [Exit ] Interrupted by user", flush=True)
            return
        except Exception as e:
            errors += 1
            idx = item.get("index")
            print(f"[{now()}] [Error] index={idx} -> {e}", flush=True)

    print(f"\n[{now()}] [Done ] processed={len(items)} errors={errors} out_dir={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()