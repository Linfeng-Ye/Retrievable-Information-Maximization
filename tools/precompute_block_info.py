#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2_000_000_000
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rim.block_info import compute_block_info_batch
from shuffle import space_to_depth_patch2chan


def _parse_pair_or_int(values: list[int] | None):
    if values is None:
        return None
    if len(values) == 1:
        return int(values[0])
    if len(values) == 2:
        return [int(values[0]), int(values[1])]
    raise ValueError("Expected one or two integers")


def _parse_patch_hw(values: list[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    if len(values) == 1:
        size = int(values[0])
        if size <= 0:
            raise ValueError(f"--patch-size must be positive, got {size}")
        return (size, size)
    if len(values) == 2:
        h, w = int(values[0]), int(values[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"--patch-size entries must be positive, got {(h, w)}")
        return (h, w)
    raise ValueError("--patch-size must be one integer or two integers: H W")


def _load_image(path: Path, patch_hw: tuple[int, int] | None) -> torch.Tensor:
    pil = Image.open(path).convert("RGB")
    if patch_hw is not None and (pil.height, pil.width) != patch_hw:
        try:
            resample = Image.Resampling.BICUBIC
        except AttributeError:
            resample = Image.BICUBIC
        target_h, target_w = int(patch_hw[0]), int(patch_hw[1])
        pil = pil.resize((target_w, target_h), resample)
    arr = np.asarray(pil, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _prepare_training_grid(img_chw: torch.Tensor, space_to_depth_k: int) -> torch.Tensor:
    k = int(space_to_depth_k)
    if k <= 1:
        return img_chw
    y, _meta = space_to_depth_patch2chan(img_chw, k)
    return y.squeeze(0).contiguous()


def _load_prepared_image_once(path: Path, patch_hw: tuple[int, int] | None, space_to_depth_k: int) -> torch.Tensor:
    """Load one image from disk once, then prepare the training-grid tensor once."""
    img_chw = _load_image(path, patch_hw)
    return _prepare_training_grid(img_chw, int(space_to_depth_k))


def _iter_inputs(input_path: str, patch_hw: tuple[int, int] | None) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(input_path)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    files = sorted(x for x in p.iterdir() if x.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No image files found in {input_path}")
    if patch_hw is None:
        raise ValueError("--patch-size is required when --input is a folder so outputs share one block layout")
    return files


def _build_candidate_resolutions(
    *,
    explicit: list[int] | None,
    num_candidate_levels: int | None,
    base_resolution: int,
    max_resolution: int | None,
    image_hw: tuple[int, int],
) -> list[int]:
    if explicit:
        cand = [int(v) for v in explicit]
    else:
        if num_candidate_levels is None:
            raise ValueError("Provide either --candidate-resolutions or --num-candidate-levels")
        n = int(num_candidate_levels)
        if n <= 0:
            raise ValueError(f"--num-candidate-levels must be positive, got {n}")
        base = int(base_resolution)
        max_res = int(max_resolution) if max_resolution is not None else min(int(image_hw[0]), int(image_hw[1]))
        if base <= 0:
            raise ValueError(f"--base-resolution must be positive, got {base}")
        if max_res < base:
            raise ValueError(f"--max-resolution must be >= base resolution, got max={max_res}, base={base}")
        if n == 1:
            cand = [base]
        else:
            cand = np.rint(np.linspace(base, max_res, n, dtype=np.float64)).astype(np.int64).tolist()
    if any(v <= 0 for v in cand):
        raise ValueError(f"candidate resolutions must be positive, got {cand}")
    if any(cand[i] >= cand[i + 1] for i in range(len(cand) - 1)):
        raise ValueError(
            "candidate resolutions must be strictly increasing. "
            f"Generated/got {len(cand)} values from {cand[0]} to {cand[-1]}, but rounding produced duplicates or disorder. "
            "Use fewer --num-candidate-levels, a larger --max-resolution, or explicit --candidate-resolutions."
        )
    return [int(v) for v in cand]


@torch.no_grad()
def _precompute_one_image(
    *,
    img_chw: torch.Tensor,
    image_path: Path,
    image_idx: int,
    num_images: int,
    candidate_resolutions: list[int],
    block_size: int | list[int],
    block_stride: int | list[int] | None,
    include_highpass_tail: bool,
    max_downsample_factor: float,
    lowpass_channelwise: bool,
    entropy_block_chunk_size: int | None,
    show_progress: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict, int, tuple[int, int]]:
    """Compute one image's block info from an already-loaded CPU tensor."""
    x = img_chw.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
    print(
        f"Precompute image {image_idx}/{num_images}: "
        f"path={image_path} tensor={tuple(x.shape)} device={device}",
        flush=True,
    )
    A_batch, meta = compute_block_info_batch(
        x,
        candidate_resolutions,
        block_size=block_size,
        block_stride=block_stride,
        include_highpass_tail=include_highpass_tail,
        max_downsample_factor=float(max_downsample_factor),
        lowpass_channelwise=bool(lowpass_channelwise),
        entropy_block_chunk_size=entropy_block_chunk_size,
        show_progress=show_progress,
        progress_desc=f"image {image_idx} bands",
    )
    A_cpu = A_batch.detach().cpu()
    channels = int(x.shape[1])
    hw = (int(x.shape[2]), int(x.shape[3]))
    del x, A_batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return A_cpu, meta, channels, hw


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute reusable B-RIM block-wise information tensor.")
    ap.add_argument("--input", required=True, help="Image file or folder of image patches.")
    ap.add_argument("--output", required=True, help="Output .pt path.")
    ap.add_argument(
        "--patch-size",
        nargs="+",
        type=int,
        default=None,
        help="Optional resize size before preprocessing: one int for square, or H W for non-square.",
    )
    ap.add_argument("--block-size", nargs="+", type=int, default=[32], help="Block size as int or 'H W'.")
    ap.add_argument("--block-stride", nargs="+", type=int, default=None, help="Optional block stride as int or 'H W'.")
    ap.add_argument("--candidate-resolutions", nargs="+", type=int, default=None)
    ap.add_argument(
        "--num-candidate-levels",
        type=int,
        default=None,
        help="Generate this many equally spaced candidate resolutions from --base-resolution to --max-resolution.",
    )
    ap.add_argument("--base-resolution", type=int, default=16, help="First generated candidate resolution.")
    ap.add_argument(
        "--max-resolution",
        type=int,
        default=None,
        help="Last generated candidate resolution. Defaults to min(H,W) after optional space-to-depth packing.",
    )
    ap.add_argument("--include-highpass-tail", action="store_true")
    ap.add_argument(
        "--space-to-depth-k",
        type=int,
        default=1,
        help="Apply the same space-to-depth patch packing as train_giga_1.py --K before computing block info.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Deprecated compatibility option; images are always processed one by one.",
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument(
        "--max-downsample-factor",
        type=float,
        default=8.0,
        help="Maximum per-stage antialiased downsample factor. Lower values use more stages and less CUDA shared memory.",
    )
    ap.add_argument(
        "--lowpass-channelwise",
        action="store_true",
        help="Compute low-pass bands one channel at a time and sum per-channel block masses to reduce peak memory.",
    )
    ap.add_argument(
        "--entropy-block-chunk-size",
        type=int,
        default=32,
        help="Number of blocks to histogram at once. Lower values reduce peak memory during block entropy.",
    )
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars during candidate-band preprocessing.")
    ap.add_argument(
        "--allow-large-image",
        action="store_true",
        help="Deprecated compatibility option; large images now use chunked block entropy.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    block_size = _parse_pair_or_int(args.block_size)
    block_stride = _parse_pair_or_int(args.block_stride)
    patch_hw = _parse_patch_hw(args.patch_size)
    files = _iter_inputs(args.input, patch_hw)

    if int(args.batch_size) != 1:
        print(
            f"[info] --batch-size={args.batch_size} is ignored; precompute now processes images one by one.",
            flush=True,
        )

    results: list[torch.Tensor] = []
    layout_meta = None
    channels = None
    hw = None
    expected_chw = None
    candidate_resolutions = None
    for image_idx, image_path in enumerate(files, start=1):
        img = _load_prepared_image_once(image_path, patch_hw, int(args.space_to_depth_k))
        print("load image: {}".format(image_path), flush=True)
        if expected_chw is None:
            expected_chw = tuple(int(v) for v in img.shape)
        elif tuple(int(v) for v in img.shape) != expected_chw:
            raise ValueError(
                f"Image {image_path} has training-grid shape {tuple(img.shape)}, "
                f"but the first image has {expected_chw}. Use --patch-size or separate output files."
            )
        image_hw = (int(img.shape[1]), int(img.shape[2]))
        if candidate_resolutions is None:
            candidate_resolutions = _build_candidate_resolutions(
                explicit=args.candidate_resolutions,
                num_candidate_levels=args.num_candidate_levels,
                base_resolution=int(args.base_resolution),
                max_resolution=args.max_resolution,
                image_hw=image_hw,
            )
            print(f"Candidate resolutions ({len(candidate_resolutions)}): {candidate_resolutions}", flush=True)
        A_cpu, meta, image_channels, image_hw = _precompute_one_image(
            img_chw=img,
            image_path=image_path,
            image_idx=image_idx,
            num_images=len(files),
            candidate_resolutions=candidate_resolutions,
            block_size=block_size,
            block_stride=block_stride,
            include_highpass_tail=bool(args.include_highpass_tail),
            max_downsample_factor=float(args.max_downsample_factor),
            lowpass_channelwise=bool(args.lowpass_channelwise),
            entropy_block_chunk_size=int(args.entropy_block_chunk_size),
            show_progress=not bool(args.no_progress),
            device=device,
            dtype=dtype,
        )
        results.append(A_cpu)
        if layout_meta is None:
            layout_meta = meta
            channels = image_channels
            hw = image_hw
        elif (
            int(meta["num_blocks_h"]) != int(layout_meta["num_blocks_h"])
            or int(meta["num_blocks_w"]) != int(layout_meta["num_blocks_w"])
            or int(meta["num_blocks"]) != int(layout_meta["num_blocks"])
        ):
            raise ValueError(
                f"Image {image_path} produced block layout {meta}, but the first image produced {layout_meta}."
            )
        print(f"[{image_idx}/{len(files)}] computed {tuple(A_cpu.shape)}", flush=True)
        del img, A_cpu

    A = torch.cat(results, dim=0)
    if len(files) == 1:
        A_to_save = A[0].contiguous()
    else:
        A_to_save = A.contiguous()
    num_bands = int(A_to_save.shape[-1])
    if candidate_resolutions is None:
        raise RuntimeError("No input batches were processed")
    cand = [int(v) for v in candidate_resolutions]
    if args.include_highpass_tail:
        if hw is None:
            raise RuntimeError("Cannot infer highpass-tail resolution without an image shape")
        tail_res = min(int(hw[0]), int(hw[1]))
        if tail_res <= cand[-1]:
            raise ValueError(
                "--include-highpass-tail requires the largest candidate resolution to be below "
                f"the training grid size; got max candidate {cand[-1]} and grid {hw}."
            )
        cand_saved = cand + [tail_res]
    else:
        cand_saved = cand
    out = {
        "A": A_to_save,
        "candidate_resolutions": cand_saved,
        "patch_size": list(hw) if hw is not None else (None if patch_hw is None else list(patch_hw)),
        "block_size": layout_meta["block_size"],
        "block_stride": layout_meta["block_stride"],
        "num_blocks_h": layout_meta["num_blocks_h"],
        "num_blocks_w": layout_meta["num_blocks_w"],
        "num_blocks": layout_meta["num_blocks"],
        "num_bands": num_bands,
        "channels": channels,
        "info_type": "entropy",
        "lowpass_type": "anti_aliased_resize_channelwise" if args.lowpass_channelwise else "anti_aliased_resize",
        "band_definition": "L_1, L_j - L_{j-1}" + (", Y - L_M" if args.include_highpass_tail else ""),
        "quantization": {
            "image_bits": 8,
            "residual_signed_bins": 511,
            "residual_min": -255,
            "residual_max": 255,
        },
        "source": {
            "input": os.path.abspath(args.input),
            "num_patches": len(files),
            "space_to_depth_k": int(args.space_to_depth_k),
            "max_downsample_factor": float(args.max_downsample_factor),
            "lowpass_channelwise": bool(args.lowpass_channelwise),
            "entropy_block_chunk_size": int(args.entropy_block_chunk_size),
        },
        "candidate_generation": {
            "explicit": args.candidate_resolutions is not None,
            "num_candidate_levels": None if args.candidate_resolutions is not None else int(args.num_candidate_levels),
            "base_resolution": int(args.base_resolution),
            "max_resolution": args.max_resolution,
            "distribution": "linear",
        },
    }
    if not bool(torch.isfinite(A_to_save).all()):
        raise RuntimeError("Computed non-finite A")
    if float(A_to_save.min().item()) < -1e-8:
        raise RuntimeError("Computed negative A")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(out, args.output)
    print(f"Saved {args.output}: A={tuple(A_to_save.shape)} device_used={device}", flush=True)


if __name__ == "__main__":
    main()
