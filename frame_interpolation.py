import cv2
import numpy as np
import os
import re
import argparse
from pathlib import Path

import torch
from diffusers.models import AutoencoderKLCogVideoX


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def add_noise(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.IntTensor = torch.tensor([50 - 1], dtype=torch.long),
    beta_start: float = 0.00085,
    beta_end: float = 0.0120,
) -> torch.Tensor:
    num_train_timesteps = 1000
    betas = torch.linspace(
        beta_start,
        beta_end,
        num_train_timesteps,
        device=original_samples.device,
        dtype=original_samples.dtype,
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    snr_shift_scale = 3.0
    alphas_cumprod = alphas_cumprod / (
        snr_shift_scale + (1 - snr_shift_scale) * alphas_cumprod
    )

    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples


def compute_flow(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow


def warp_image(img, flow):
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def interpolate_frames_bidirectional(img1, img2, n_frames):
    flow12 = compute_flow(img1, img2)
    flow21 = compute_flow(img2, img1)

    frames = [img1]
    for i in range(1, n_frames + 1):
        alpha = i / (n_frames + 1)

        warp1 = warp_image(img1, flow12 * alpha)
        warp2 = warp_image(img2, flow21 * (1 - alpha))

        blended = cv2.addWeighted(warp1, 1 - alpha, warp2, alpha, 0)
        frames.append(blended)

    frames.append(img2)
    return frames


def load_keyframes(sample_dir):
    sample_dir = Path(sample_dir)
    image_paths = sorted(sample_dir.glob("keyframe_*.png"), key=natural_key)

    if len(image_paths) < 2:
        raise ValueError(f"[ERROR] {sample_dir} contains fewer than 2 keyframes.")

    images = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"[ERROR] Failed to read image: {p}")
        images.append(img)

    return images, image_paths


def interpolate_to_frames(keyframes, total_frames=45, resize_hw=(1360, 768)):
    if len(keyframes) < 2:
        raise ValueError("At least 2 keyframes are required for interpolation.")

    if total_frames < len(keyframes):
        raise ValueError(
            f"total_frames={total_frames} is smaller than num_keyframes={len(keyframes)}."
        )

    num_segments = len(keyframes) - 1
    frames_per_segment = (total_frames - len(keyframes)) // num_segments
    remainder = (total_frames - len(keyframes)) % num_segments

    all_frames = []
    for i in range(num_segments):
        img1 = cv2.resize(keyframes[i], resize_hw)
        img2 = cv2.resize(keyframes[i + 1], resize_hw)

        n_interp = frames_per_segment + (1 if i < remainder else 0)
        segment = interpolate_frames_bidirectional(img1, img2, n_interp)

        if i > 0:
            segment = segment[1:] 
        all_frames.extend(segment)

    if len(all_frames) != total_frames:
        raise RuntimeError(
            f"[ERROR] Interpolated frame count mismatch: expected {total_frames}, got {len(all_frames)}"
        )

    return all_frames


def frames_to_vae_latents(all_frames, vae, device):
    frame_tensors = []
    for img in all_frames:
        x = torch.from_numpy(img).float() / 255.0  
        x = x.permute(2, 0, 1)                     
        x = x * 2.0 - 1.0                         
        frame_tensors.append(x)


    x0 = torch.stack(frame_tensors, dim=0).to(device)


    x0 = x0.permute(1, 0, 2, 3).unsqueeze(0)

    with torch.no_grad():
        latent_dist = vae.encode(x0).latent_dist
        z_0 = (1.0 / vae.config.scaling_factor) * latent_dist.sample()

    return z_0


def latents_add_noise(z_0, timestep=49):
    _, _, F, _, _ = z_0.shape
    all_noise_frames = []

    for f in range(F):
        frame_latent = z_0[0, :, f]  # [C, H, W]
        noise = torch.randn_like(frame_latent)
        noisy_frame_latent = add_noise(
            original_samples=frame_latent,
            noise=noise,
            timesteps=torch.tensor([timestep], dtype=torch.long, device=frame_latent.device)
        )
        all_noise_frames.append(noisy_frame_latent.unsqueeze(1))

    z_t = torch.cat(all_noise_frames, dim=1).unsqueeze(0)  
    print(timestep)
    return z_t


def main_frame_interpolation(keyframes, vae, total_frames=45, resize_hw=(1360, 768), timestep=49):
    all_frames = interpolate_to_frames(
        keyframes=keyframes,
        total_frames=total_frames,
        resize_hw=resize_hw
    )

    device = next(vae.parameters()).device
    z_0 = frames_to_vae_latents(all_frames, vae=vae, device=device)
    z_t = latents_add_noise(z_0, timestep=timestep)
    return z_t, all_frames


def process_one_sample(
    sample_dir,
    output_root,
    vae,
    total_frames=45,
    resize_hw=(1360, 768),
    timestep=49,
    save_preview_frames=False,
):
    sample_dir = Path(sample_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    keyframes, image_paths = load_keyframes(sample_dir)

    print(f"\n[PROCESS] {sample_dir.name}")
    print(f"[INFO] num_keyframes = {len(keyframes)}")
    for p in image_paths:
        print(f"  - {p.name}")

    z_t, all_frames = main_frame_interpolation(
        keyframes=keyframes,
        vae=vae,
        total_frames=total_frames,
        resize_hw=resize_hw,
        timestep=timestep,
    )

    out_pt = output_root / f"{sample_dir.name}.pt"
    torch.save(z_t, out_pt)
    print(f"[DONE] Saved z_t to {out_pt}")
    print(f"[SHAPE] z_t = {tuple(z_t.shape)}")

    if save_preview_frames:
        preview_dir = output_root / f"{sample_dir.name}_preview_frames"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(all_frames):
            cv2.imwrite(str(preview_dir / f"frame_{idx:03d}.png"), frame)
        print(f"[DONE] Saved preview frames to {preview_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate IKS keyframes to 45 frames, encode with CogVideoX VAE, and save noisy latent z_t."
    )
    parser.add_argument(
        "--keyframes_root",
        type=str,
        default="",
        help="Root directory of IKS keyframe folders."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Directory to save z_t .pt files."
    )
    parser.add_argument(
        "--total_frames",
        type=int,
        default=45,
        help="Total number of frames after interpolation."
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Optional sample folder name to process only one sample."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1360,
        help="Resize width."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Resize height."
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU id."
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="THUDM/CogVideoX1.5-5b",
        help="CogVideoX model path or HF repo id. VAE is loaded from subfolder='vae'."
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=49,
        help="Noise timestep. Original code used 50-1 = 49."
    )
    parser.add_argument(
        "--save_preview_frames",
        action="store_true",
        help="If set, also save interpolated RGB frames for preview."
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    resize_hw = (args.width, args.height)

    print(f"[INFO] device         = {device}")
    print(f"[INFO] keyframes_root = {args.keyframes_root}")
    print(f"[INFO] output_root    = {args.output_root}")
    print(f"[INFO] total_frames   = {args.total_frames}")
    print(f"[INFO] resize         = {args.width}x{args.height}")
    print(f"[INFO] vae_model_path = {args.vae_model_path}")
    print(f"[INFO] timestep       = {args.timestep}")

    print("[INFO] Loading CogVideoX VAE ...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.vae_model_path,
        subfolder="vae"
    ).to(device)
    vae.eval()
    print("[INFO] VAE loaded.")

    keyframes_root = Path(args.keyframes_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.sample is not None:
        sample_dirs = [keyframes_root / args.sample]
    else:
        sample_dirs = sorted(
            [p for p in keyframes_root.iterdir() if p.is_dir()],
            key=natural_key
        )

    if len(sample_dirs) == 0:
        raise RuntimeError(f"[ERROR] No sample directories found in {keyframes_root}")

    print(f"[INFO] num_samples    = {len(sample_dirs)}")

    for sample_dir in sample_dirs:
        process_one_sample(
            sample_dir=sample_dir,
            output_root=output_root,
            vae=vae,
            total_frames=args.total_frames,
            resize_hw=resize_hw,
            timestep=args.timestep,
            save_preview_frames=args.save_preview_frames,
        )

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
