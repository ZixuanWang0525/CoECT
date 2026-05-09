#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example:
  python run_pipeline.py \
    --run_name test_run \
    --cogvideo_cli \
    --cogvideo_model_path \
    --gpu_id \
    --num_frames 
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence



def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def norm_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def slugify(text: str, maxlen: int = 96) -> str:
    text = norm_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "sample"
    return text[:maxlen]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")


def require_path(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> None:
    """Run a subprocess and stream stdout/stderr to the terminal only."""
    cmd_str = " ".join([str(x) for x in cmd])
    header = f"\n{'=' * 100}\n[CMD] {cmd_str}\n[CWD] {cwd or Path.cwd()}\n{'=' * 100}\n"
    print(header, flush=True)

    if dry_run:
        print("[DRY_RUN] command not executed", flush=True)
        return

    proc = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with code {ret}: {cmd_str}")


def link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(str(src), str(dst))
    except Exception:
        shutil.copy2(src, dst)


def find_sample_dir(keyframes_root: Path, index: int) -> Path:
    prefix = f"{int(index):04d}_"
    candidates = sorted([p for p in keyframes_root.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not candidates:
        raise FileNotFoundError(f"No keyframe sample directory starts with {prefix} under {keyframes_root}")
    if len(candidates) > 1:
        print(f"[WARN] Multiple sample dirs for index={index}; using {candidates[0]}", flush=True)
    return candidates[0]


def get_prompt_from_pnr_item(item: Dict[str, Any]) -> str:
    for key in ("positive_prompt", "prompt", "input_text", "raw_prompt", "initial_prompt"):
        val = norm_text(item.get(key))
        if val:
            return val
    events = item.get("events", []) or []
    event_descs = [norm_text(e.get("event_description")) for e in events if isinstance(e, dict)]
    event_descs = [x for x in event_descs if x]
    if event_descs:
        return " ".join(event_descs)
    raise ValueError(f"No usable prompt found in PNR item index={item.get('index')}")


def run_pfg(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    cmd = [
        sys.executable, str(paths["pfg_py"]),
        "--mode", "all",
        "--prompts_json", str(paths["prompt_json"]),
        "--out_dir", str(paths["pfg_dir"]),
        "--final_json_name", paths["pfg_json"].name,
        "--languages", args.languages,
    ]
    if args.pfg_save_intermediate:
        cmd.append("--save_intermediate")
    if args.include_failed_items:
        cmd.append("--include_failed_items")

    env = os.environ.copy()
    if args.pfg_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.pfg_cuda_visible_devices

    run_command(cmd, cwd=paths["project_dir"], env=env, dry_run=args.dry_run)


def run_ppd(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    device = args.ppd_device or f"cuda:{args.gpu_id}"
    cmd = [
        sys.executable, str(paths["ppd_py"]),
        "--pfg_json", str(paths["pfg_json"]),
        "--out_file", str(paths["ppd_json"]),
        "--model_path", args.llm_model_path,
        "--device", device,
    ]
    if args.quiet:
        cmd.append("--quiet")

    run_command(cmd, cwd=paths["project_dir"], dry_run=args.dry_run)


def run_pnr(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    cmd = [
        sys.executable, str(paths["pnr_py"]),
        "--pfg_path", str(paths["pfg_json"]),
        "--ppd_path", str(paths["ppd_json"]),
        "--output_path", str(paths["pnr_json"]),
        "--model_path", args.llm_model_path,
        "--gpu_id", str(args.gpu_id),
        "--dtype", args.llm_dtype,
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--max_new_tokens", str(args.pnr_max_new_tokens),
        "--repetition_penalty", str(args.repetition_penalty),
    ]
    if args.sample_index is not None:
        cmd += ["--sample_index", str(args.sample_index)]

    run_command(cmd, cwd=paths["project_dir"], dry_run=args.dry_run)


def run_iks(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    cmd = [
        sys.executable, str(paths["iks_py"]),
        "--pnr_json", str(paths["pnr_json"]),
        "--qwen_image_model", args.qwen_image_model,
        "--qwen_edit_model", args.qwen_edit_model,
        "--out_dir", str(paths["keyframes_dir"]),
        "--ratio", args.ratio,
        "--init_steps", str(args.iks_init_steps),
        "--edit_steps", str(args.iks_edit_steps),
        "--init_cfg", str(args.iks_init_cfg),
        "--edit_cfg", str(args.iks_edit_cfg),
        "--gpu_id", str(args.gpu_id),
        "--max_side", str(args.iks_max_side),
    ]
    if args.init_seed is not None:
        cmd += ["--init_seed", str(args.init_seed)]
    if args.edit_seed is not None:
        cmd += ["--edit_seed", str(args.edit_seed)]
    if args.sample_index is not None:
        cmd += ["--sample_index", str(args.sample_index)]
    cmd.append("--no_resume" if args.no_resume else "--resume")

    run_command(cmd, cwd=paths["project_dir"], dry_run=args.dry_run)


def run_frame_interpolation(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    cmd = [
        sys.executable, str(paths["frame_interpolation_py"]),
        "--keyframes_root", str(paths["keyframes_dir"]),
        "--output_root", str(paths["latents_dir"]),
        "--total_frames", str(args.num_frames),
        "--width", str(args.width),
        "--height", str(args.height),
        "--gpu_id", str(args.gpu_id),
        "--vae_model_path", args.vae_model_path,
        "--timestep", str(args.noise_timestep),
    ]
    if args.save_latent_preview_frames:
        cmd.append("--save_preview_frames")
    if args.sample_name:
        cmd += ["--sample", args.sample_name]

    run_command(cmd, cwd=paths["project_dir"], dry_run=args.dry_run)


def run_cogvideo(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    require_file(paths["pnr_json"], "PNR json")
    require_path(Path(args.cogvideo_model_path), "CogVideoX model path")
    require_file(Path(args.cogvideo_cli), "CogVideoX cli_demo.py")

    pnr_data = load_json(paths["pnr_json"])
    items = pnr_data.get("items", []) if isinstance(pnr_data, dict) else []
    if not items:
        raise RuntimeError(f"No items found in PNR json: {paths['pnr_json']}")

    if args.sample_index is not None:
        items = [x for x in items if int(x.get("index", -1)) == int(args.sample_index)]
        if not items:
            raise KeyError(f"No PNR item found for index={args.sample_index}")

    ensure_dir(paths["videos_dir"])
    ensure_dir(paths["prior_compat_dir"])

    for item in items:
        idx = int(item["index"])
        prompt = get_prompt_from_pnr_item(item)
        sample_dir = find_sample_dir(paths["keyframes_dir"], idx)

        init_image = sample_dir / "keyframe_00_init.png"
        require_file(init_image, "initial keyframe")

        latent_path = paths["latents_dir"] / f"{sample_dir.name}.pt"
        require_file(latent_path, "interpolated latent prior")


        compat_dir = ensure_dir(paths["prior_compat_dir"] / sample_dir.name)
        link_or_copy(latent_path, compat_dir / "keyframe_00_init.pt")
        link_or_copy(latent_path, compat_dir / f"{sample_dir.name}.pt")

        video_name = f"{idx:04d}_{slugify(prompt, maxlen=80)}.mp4"
        video_path = paths["videos_dir"] / video_name

        env = os.environ.copy()
        env["ZT_DIR"] = str(compat_dir)

        cmd = [
            sys.executable, str(args.cogvideo_cli),
            "--generate_type", "i2v",
            "--model_path", str(args.cogvideo_model_path),
            "--prompt", prompt,
            "--image_or_video_path", str(init_image),
            "--output_path", str(video_path),
            "--num_frames", str(args.num_frames),
            "--fps", str(args.fps),
            "--seed", str(args.seed),
            "--guidance_scale", str(args.guidance_scale),
            "--num_inference_steps", str(args.num_inference_steps),
            "--dtype", args.cogvideo_dtype,
        ]
        if args.cogvideo_width is not None:
            cmd += ["--width", str(args.cogvideo_width)]
        if args.cogvideo_height is not None:
            cmd += ["--height", str(args.cogvideo_height)]

        run_command(
            cmd,
            cwd=Path(args.cogvideo_cli).resolve().parent,
            env=env,
            dry_run=args.dry_run,
        )



def build_paths(args: argparse.Namespace) -> Dict[str, Path]:
    project_dir = Path(args.project_dir).resolve() if args.project_dir else Path(__file__).resolve().parent
    run_name = args.run_name or f"run_{now_tag()}"
    output_root = Path(args.output_root).resolve() if args.output_root else project_dir / "outputs"
    run_dir = output_root / run_name

    return {
        "project_dir": project_dir,
        "prompt_json": Path(args.prompt_json).resolve() if args.prompt_json else project_dir / "prompt.json",

        "pfg_py": project_dir / "pfg.py",
        "ppd_py": project_dir / "ppd.py",
        "pnr_py": project_dir / "pnr.py",
        "iks_py": project_dir / "iks.py",
        "frame_interpolation_py": project_dir / "frame_interpolation.py",

        "run_dir": run_dir,
        "pfg_dir": run_dir / "01_pfg",
        "ppd_dir": run_dir / "02_ppd",
        "pnr_dir": run_dir / "03_pnr",
        "iks_dir": run_dir / "04_iks",
        "interp_dir": run_dir / "05_interpolation",
        "videos_dir": run_dir / "06_videos",

        "pfg_json": run_dir / "01_pfg" / "pfg_for_ppd.json",
        "ppd_json": run_dir / "02_ppd" / "ppd_for_pnr.json",
        "pnr_json": run_dir / "03_pnr" / "pnr_for_iks.json",
        "keyframes_dir": run_dir / "04_iks" / "keyframes",
        "latents_dir": run_dir / "05_interpolation" / "noisy_latents",
        "prior_compat_dir": run_dir / "05_interpolation" / "prior_compat_for_cogvideo",
    }


def validate_inputs(paths: Dict[str, Path], args: argparse.Namespace) -> None:
    require_file(paths["prompt_json"], "prompt.json")
    for key in ("pfg_py", "ppd_py", "pnr_py", "iks_py", "frame_interpolation_py"):
        require_file(paths[key], key)
    if not args.skip_cogvideo:
        require_file(Path(args.cogvideo_cli), "CogVideoX cli_demo.py")
        require_path(Path(args.cogvideo_model_path), "CogVideoX model path")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the PECR-TCP video generation pipeline from prompt.json to videos.")

    p.add_argument("--project_dir", type=str, default="", help="Directory containing pfg.py, ppd.py, pnr.py, iks.py, frame_interpolation.py, and prompt.json. Default: this script directory.")
    p.add_argument("--prompt_json", type=str, default="", help="Input prompt.json. Default: <project_dir>/prompt.json")
    p.add_argument("--output_root", type=str, default="", help="Output root. Default: <project_dir>/outputs")
    p.add_argument("--run_name", type=str, default="", help="Run name under output_root. Default: run_YYYYmmdd_HHMMSS")

    p.add_argument("--skip_pfg", action="store_true")
    p.add_argument("--skip_ppd", action="store_true")
    p.add_argument("--skip_pnr", action="store_true")
    p.add_argument("--skip_iks", action="store_true")
    p.add_argument("--skip_frame_interpolation", action="store_true")
    p.add_argument("--skip_cogvideo", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no_resume", action="store_true")

    p.add_argument("--sample_index", type=int, default=None, help="Process a single sample index when supported.")
    p.add_argument("--sample_name", type=str, default="", help="Process a single IKS sample folder for interpolation.")

    p.add_argument("--llm_model_path", type=str, default="/mnt/43t/hyx/model_cache/gpt-oss/models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb")
    p.add_argument("--qwen_image_model", type=str, default="/mnt/43t/hyx/model_cache/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6")
    p.add_argument("--qwen_edit_model", type=str, default="/mnt/43t/hyx/model_cache/models--Qwen--Qwen-Image-Edit/snapshots/ac7f9318f633fc4b5778c59367c8128225f1e3de")
    p.add_argument("--vae_model_path", type=str, default="/mnt/43t/hyx/model_cache/CogVideoX1.5-5B-I2V")
    p.add_argument("--cogvideo_cli", type=str, default="/mnt/43t/YixinHu/CVPR2026_code/CogVideo/inference/cli_demo.py")
    p.add_argument("--cogvideo_model_path", type=str, default="/mnt/43t/hyx/model_cache/CogVideoX1.5-5B-I2V")

    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--ppd_device", type=str, default="", help="Override PPD device, e.g. cuda:1. Default: cuda:<gpu_id>")
    p.add_argument("--pfg_cuda_visible_devices", type=str, default="", help="Optional CUDA_VISIBLE_DEVICES only for pfg.py")

    # PFG
    p.add_argument("--languages", type=str, default="en,zh")
    p.add_argument("--pfg_save_intermediate", action="store_true")
    p.add_argument("--include_failed_items", action="store_true")

    # PNR
    p.add_argument("--llm_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--pnr_max_new_tokens", type=int, default=512)
    p.add_argument("--repetition_penalty", type=float, default=1.0)

    # IKS
    p.add_argument("--ratio", type=str, default="16:9", choices=["1:1", "16:9", "9:16", "4:3", "3:4"])
    p.add_argument("--iks_init_steps", type=int, default=28)
    p.add_argument("--iks_edit_steps", type=int, default=24)
    p.add_argument("--iks_init_cfg", type=float, default=3.5)
    p.add_argument("--iks_edit_cfg", type=float, default=3.5)
    p.add_argument("--iks_max_side", type=int, default=768)
    p.add_argument("--init_seed", type=int, default=None)
    p.add_argument("--edit_seed", type=int, default=None)

    # Interpolation
    p.add_argument("--num_frames", type=int, default=85)
    p.add_argument("--width", type=int, default=1360)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--noise_timestep", type=int, default=699)
    p.add_argument("--save_latent_preview_frames", action="store_true")

    # Video generation
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cogvideo_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--cogvideo_width", type=int, default=None)
    p.add_argument("--cogvideo_height", type=int, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_paths(args)
    validate_inputs(paths, args)

    for key in ("run_dir", "pfg_dir", "ppd_dir", "pnr_dir", "iks_dir", "interp_dir", "videos_dir"):
        ensure_dir(paths[key])

    if not args.skip_pfg:
        run_pfg(args, paths)
    if not args.skip_ppd:
        run_ppd(args, paths)
    if not args.skip_pnr:
        run_pnr(args, paths)
    if not args.skip_iks:
        run_iks(args, paths)
    if not args.skip_frame_interpolation:
        run_frame_interpolation(args, paths)
    if not args.skip_cogvideo:
        run_cogvideo(args, paths)

    print(f"\n[DONE] All requested stages finished. Outputs: {paths['run_dir']}", flush=True)


if __name__ == "__main__":
    main()
