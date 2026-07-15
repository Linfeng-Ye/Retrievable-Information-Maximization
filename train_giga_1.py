import argparse
import math
import os
import time
from typing import Tuple
import json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2_000_000_000
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from shuffle import space_to_depth_patch2chan, depth_to_space_chan2patch

from utils.rim_exact import get_entropy_grid
from utils.rim_blockwise import solve_blockwise_masked_rim_exact
from utils.metrics import psnr, compute_metrics_tiled
from hash_encoder_2d import MultiResolutionHashEncoder2D, _parse_resolutions_arg
from rim.initializer import initialize_rim_iterative, load_block_info_for_training
# from thirdparty.torch_ngp.gridencoder import GridEncoder

# -----------------------------
# Utilities
# -----------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_hashtable(resolutions, log_hash_size, features):

    hashsize = 2**log_hash_size
    cnt=0
    for r in resolutions:
        alpha = r**2/hashsize
        cnt +=  min(((1-math.exp(-alpha)))*hashsize, r**2 , hashsize)
    return int(cnt*features)

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

# -----------------------------
# Tiny MLP Head
# -----------------------------
class ColorMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        # Kaiming init for ReLU nets
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output in [0,1] via sigmoid for stability
        return self.net(x)


# -----------------------------
# Samplers that keep image in uint8 and only convert batches
# -----------------------------
from typing import Union, Optional, Callable
ArrayLikeUInt8 = Union[np.ndarray, torch.Tensor]

def make_samplers(
    image_uint8: ArrayLikeUInt8,
    device: torch.device,
    val_samples: int = 100_000,
    index_buffer: int = 4_000_000,
) -> tuple[Callable[[int], tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, tuple[int, int]]:
    """
    Efficient random pixel sampler for ultra-high-res fitting.

    Key optimizations vs a naive per-step sampler:
      - Keep GT image on CPU as uint8 (optionally pinned) and only move sampled colors.
      - Sample *linear indices* from a large pre-generated buffer to amortize RNG overhead.
      - Use a flattened (HW, C) view for fast gather.
      - Precompute coordinate lookup tables (x_lut, y_lut) in [-1,1] to avoid per-sample division.

    Returns:
      sample_batch(B): -> (coords [B,2] float in [-1,1], colors [B,C] float in [0,1]) on device
      val_coords, val_colors: fixed validation subset on device
      (H, W): image size
    """
    if not (torch.cuda.is_available() and device.type == "cuda"):
        # If user asked for cuda but it's not available, keep things consistent.
        device = torch.device("cpu")

    # ---- Normalize input type and ensure CPU uint8 contiguous ----
    if isinstance(image_uint8, np.ndarray):
        img_cpu = torch.from_numpy(image_uint8)  # uint8, (H,W,C)
    elif torch.is_tensor(image_uint8):
        img_cpu = image_uint8
    else:
        raise TypeError(f"image_uint8 must be a numpy.ndarray or torch.Tensor, got {type(image_uint8)}")

    if img_cpu.device.type != "cpu":
        img_cpu = img_cpu.cpu()
    if img_cpu.dtype != torch.uint8:
        raise TypeError(f"image_uint8 must be uint8, got {img_cpu.dtype}")

    img_cpu = img_cpu.contiguous()
    H, W, C = img_cpu.shape
    HW = H * W

    # Pin memory for faster H2D (only meaningful on CUDA)
    use_pin = (device.type == "cuda")
    if use_pin and not img_cpu.is_pinned():
        try:
            img_cpu = img_cpu.pin_memory()
        except RuntimeError:
            # Some builds disallow pinning; fall back gracefully.
            pass

    img_flat = img_cpu.view(HW, C)  # (HW, C) uint8

    # ---- Precompute coordinate LUTs in [-1,1] ----
    # NOTE: align with your encoder input convention (you currently use *2-1).
    x_lut = torch.linspace(-1.0, 1.0, W, dtype=torch.float32)
    y_lut = torch.linspace(-1.0, 1.0, H, dtype=torch.float32)
    if use_pin:
        try:
            x_lut = x_lut.pin_memory()
            y_lut = y_lut.pin_memory()
        except RuntimeError:
            pass

    # ---- Validation subset (fixed) ----
    Nval = min(int(val_samples), HW)
    # Sample linear indices
    v_idx = torch.randint(low=0, high=HW, size=(Nval,), dtype=torch.int64)
    vx = v_idx.remainder(W)
    vy = v_idx.div(W, rounding_mode="floor")

    vcoords_cpu = torch.empty((Nval, 2), dtype=torch.float32, pin_memory=use_pin)
    vcoords_cpu[:, 0] = x_lut[vx]
    vcoords_cpu[:, 1] = y_lut[vy]
    vcoords = vcoords_cpu.to(device, non_blocking=use_pin)

    vcolors = img_flat[v_idx].to(device=device, dtype=torch.float32, non_blocking=use_pin) / 255.0

    # ---- RNG index buffer for training batches ----
    index_buffer = int(index_buffer)
    if index_buffer <= 0:
        index_buffer = 1_000_000
    # Cap to HW to avoid wasting RAM on very small images
    index_buffer = min(index_buffer, max(1, HW))

    idx_buf = torch.empty((index_buffer,), dtype=torch.int64, pin_memory=use_pin)
    # We refill by random_ (fast) rather than randint each step
    idx_buf.random_(0, HW)
    buf_ptr = 0

    def _refill():
        nonlocal idx_buf, buf_ptr
        idx_buf.random_(0, HW)
        buf_ptr = 0

    def sample_batch(B: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a random batch on *device*:
          coords: (B,2) float32 in [-1,1]
          colors: (B,C) float32 in [0,1]
        """
        nonlocal buf_ptr
        B = int(B)
        if B <= 0:
            raise ValueError("B must be positive")
        if B > idx_buf.numel():
            # Very large batch requested; fall back to on-the-fly sampling.
            idx = torch.randint(low=0, high=HW, size=(B,), dtype=torch.int64)
        else:
            if buf_ptr + B > idx_buf.numel():
                _refill()
            idx = idx_buf[buf_ptr:buf_ptr + B]
            buf_ptr += B

        xs = idx.remainder(W)
        ys = idx.div(W, rounding_mode="floor")

        coords_cpu = torch.empty((B, 2), dtype=torch.float32, pin_memory=use_pin)
        coords_cpu[:, 0] = x_lut[xs]
        coords_cpu[:, 1] = y_lut[ys]
        coords = coords_cpu.to(device, non_blocking=use_pin)

        colors = img_flat[idx].to(device=device, dtype=torch.float32, non_blocking=use_pin) / 255.0
        return coords, colors

    return sample_batch, vcoords, vcolors, (H, W)



@torch.inference_mode()
def reconstruct_patched_rows_streamed(
    encoder: torch.nn.Module,
    mlp: torch.nn.Module,
    meta: dict,
    out_npy: str,
    rows_per_chunk: int = 64,   # in shuffled rows (Hp)
    device: str = "cuda",
    half: bool = False,
    write_bigtiff: str | None = None,
    tiff_tile: int = 256,
):
    """
    Reconstruction for patch-rearranged training (space-to-depth).
    Predict in shuffled space (Hp,Wp,C*K*K) and unshuffle back to (H,W,C).
    Streams over shuffled rows to keep VRAM low.
    """
    K = int(meta["K"])
    C = int(meta["C"])
    H, W = map(int, meta["orig_hw"])
    Hp, Wp = map(int, meta["grid_hw"])
    CK2 = C * K * K
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(out_npy) or ".", exist_ok=True)
    x_vals = torch.linspace(0.0, 1.0, Wp, dtype=torch.float32)  # CPU
    if device_t.type == "cuda":
        x_vals = x_vals.pin_memory()
    mm = np.memmap(out_npy, mode="w+", dtype=np.uint8, shape=(H, W, C))
    encoder.eval(); mlp.eval()
    total = 0
    t0 = time.time()
    for r0 in range(0, Hp, rows_per_chunk):
        r1 = min(Hp, r0 + rows_per_chunk)
        rows = r1 - r0
        if Hp > 1:
            y_vals = torch.linspace(r0, r1 - 1, steps=rows, dtype=torch.float32) / (Hp - 1)
        else:
            y_vals = torch.zeros((rows,), dtype=torch.float32)
        x = x_vals.unsqueeze(0).expand(rows, Wp)
        y = y_vals.unsqueeze(1).expand(rows, Wp)
        coords = torch.stack((x, y), dim=-1).flatten(0, 1) * 2.0 - 1.0
        if device_t.type == "cuda":
            coords = coords.pin_memory().to(device_t, non_blocking=True)
        else:
            coords = coords.to(device_t)
        use_amp = (device_t.type == "cuda") and half
        with torch.cuda.amp.autocast(enabled=use_amp):
            feats = encoder(coords)
            preds = mlp(feats)
        preds = preds.float().clamp_(0.0, 1.0).reshape(rows, Wp, CK2)
        preds_u8 = (preds * 255.0 + 0.5).to(torch.uint8).cpu()
        # unshuffle chunk: (1, CK2, rows, Wp) -> (1, C, rows*K, Wp*K)
        y_chunk = preds_u8.permute(2, 0, 1).unsqueeze(0).contiguous()
        x_chunk = (
            y_chunk.view(1, C, K, K, rows, Wp)
                  .permute(0, 1, 4, 2, 5, 3)
                  .contiguous()
                  .view(1, C, rows * K, Wp * K)
        )
        out_r0 = r0 * K
        out_r1 = min(out_r0 + rows * K, H)
        if out_r0 >= H:
            break
        chunk_h = out_r1 - out_r0
        chunk_np = x_chunk[0, :, :chunk_h, :W].permute(1, 2, 0).contiguous().numpy()
        mm[out_r0:out_r1, :, :] = chunk_np
        total += rows * Wp
        if device_t.type == "cuda":
            del coords, feats, preds
            torch.cuda.empty_cache()
        done = total / (Hp * Wp)
        print(f"[shuf {r1:>6d}/{Hp}]  {done:6.2%}", end="\r", flush=True)
    if hasattr(mm, "_mmap") and mm._mmap is not None:
        mm._mmap.flush()
    del mm
    print(f"\nWrote reconstructed memmap: {out_npy}  ({W}x{H})  in {time.time()-t0:.1f}s")
    if write_bigtiff is not None:
        try:
            import tifffile as tiff
            print("Writing BigTIFF (this will take a while)...")
            arr = np.memmap(out_npy, mode="r", dtype=np.uint8, shape=(H, W, C))
            tiff.imwrite(
                write_bigtiff,
                arr,
                bigtiff=True,
                photometric="rgb" if C == 3 else None,
                compression="zlib",
                tile=(tiff_tile, tiff_tile),
                dtype=arr.dtype,
            )
            print(f"BigTIFF saved to: {write_bigtiff}")
            del arr
        except Exception as e:
            print(f"[warn] BigTIFF export failed: {e}. Memmap output is still available.")



# -----------------------------
# In-memory evaluation (no reconstruction memmap)
# -----------------------------

def _gaussian_1d(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    g1d = _gaussian_1d(window_size, sigma, device, dtype)
    g2d = (g1d[:, None] * g1d[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)
    return g2d.expand(channels, 1, window_size, window_size).contiguous()  # (C,1,ws,ws)

def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
              data_range: float = 1.0, K1: float = 0.01, K2: float = 0.03) -> torch.Tensor:
    """SSIM for (N,C,H,W) tensors in [0,1]. Returns scalar tensor."""
    assert img1.shape == img2.shape and img1.ndim == 4
    N, C, H, W = img1.shape
    if min(H, W) < window_size:
        raise ValueError(f"SSIM window_size={window_size} larger than image min(H,W)={min(H,W)}")
    window = _gaussian_window(window_size, sigma, C, img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu12

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=(1, 2, 3)).mean()

def ms_ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
                 data_range: float = 1.0, weights=None) -> torch.Tensor:
    """MS-SSIM for (N,C,H,W) tensors in [0,1]. Returns scalar tensor."""
    assert img1.shape == img2.shape and img1.ndim == 4
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.tensor(weights, device=img1.device, dtype=img1.dtype)

    # Determine usable number of levels such that smallest scale still supports the window.
    max_levels = int(weights.numel())
    H, W = img1.shape[-2:]
    levels = 0
    h, w = H, W
    for _ in range(max_levels):
        if min(h, w) < window_size:
            break
        levels += 1
        h = h // 2
        w = w // 2
    if levels <= 0:
        raise ValueError(f"MS-SSIM window_size={window_size} larger than image min(H,W)={min(H,W)}")
    weights = weights[:levels]
    weights = weights / weights.sum()

    mcs = []
    for i in range(levels):
        N, C, H, W = img1.shape
        window = _gaussian_window(window_size, sigma, C, img1.device, img1.dtype)
        pad = window_size // 2

        mu1 = F.conv2d(img1, window, padding=pad, groups=C)
        mu2 = F.conv2d(img2, window, padding=pad, groups=C)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu12

        K1, K2 = 0.01, 0.03
        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2

        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        cs = cs_map.mean(dim=(1, 2, 3))      # (N,)
        ssim = ssim_map.mean(dim=(1, 2, 3))  # (N,)

        if i < levels - 1:
            mcs.append(cs)
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
        else:
            ssim_final = ssim

    if len(mcs) == 0:
        out = ssim_final
    else:
        mcs = torch.stack(mcs, dim=0)  # (levels-1, N)
        out = torch.prod(mcs ** weights[:-1].unsqueeze(1), dim=0) * (ssim_final ** weights[-1])
    return out.mean()


@torch.inference_mode()
def predict_tile_patched_uint8(
    encoder: torch.nn.Module,
    mlp: torch.nn.Module,
    meta: dict,
    y0: int,
    x0: int,
    tile_size: int,
    device: str = "cuda",
    half: bool = False,
) -> torch.Tensor:
    """Predict an output-space tile (C,tile,tile) uint8 without writing reconstruction to disk."""
    K = int(meta["K"])
    C = int(meta["C"])
    H, W = map(int, meta["orig_hw"])
    Hp, Wp = map(int, meta["grid_hw"])
    CK2 = C * K * K

    if not (0 <= y0 <= H - tile_size and 0 <= x0 <= W - tile_size):
        raise ValueError(f"Tile origin out of bounds: (y0,x0)=({y0},{x0}), tile_size={tile_size}, HxW={H}x{W}")

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    encoder.eval(); mlp.eval()

    # Shuffled-space cell range covering this output tile
    r0 = y0 // K
    c0 = x0 // K
    r1 = (y0 + tile_size - 1) // K + 1
    c1 = (x0 + tile_size - 1) // K + 1
    r1 = min(r1, Hp)
    c1 = min(c1, Wp)

    rows = r1 - r0
    cols = c1 - c0

    if Wp > 1:
        x_vals = torch.arange(c0, c1, device=device_t, dtype=torch.float32) / (Wp - 1)
    else:
        x_vals = torch.zeros((cols,), device=device_t, dtype=torch.float32)
    if Hp > 1:
        y_vals = torch.arange(r0, r1, device=device_t, dtype=torch.float32) / (Hp - 1)
    else:
        y_vals = torch.zeros((rows,), device=device_t, dtype=torch.float32)

    x = x_vals.unsqueeze(0).expand(rows, cols)
    y = y_vals.unsqueeze(1).expand(rows, cols)
    coords = torch.stack((x, y), dim=-1).flatten(0, 1) * 2.0 - 1.0

    use_amp = (device_t.type == "cuda") and half
    with torch.cuda.amp.autocast(enabled=use_amp):
        feats = encoder(coords)
        preds = mlp(feats)  # (rows*cols, CK2)

    preds = preds.float().clamp_(0.0, 1.0).reshape(rows, cols, CK2)
    preds_u8 = (preds * 255.0 + 0.5).to(torch.uint8)  # (rows, cols, CK2)

    # unshuffle: (1, CK2, rows, cols) -> (1, C, rows*K, cols*K)
    y_chunk = preds_u8.permute(2, 0, 1).unsqueeze(0).contiguous()
    x_chunk = (
        y_chunk.view(1, C, K, K, rows, cols)
              .permute(0, 1, 4, 2, 5, 3)
              .contiguous()
              .view(1, C, rows * K, cols * K)
    )

    off_y = y0 - r0 * K
    off_x = x0 - c0 * K
    tile_u8 = x_chunk[0, :, off_y:off_y + tile_size, off_x:off_x + tile_size].contiguous()
    return tile_u8


@torch.inference_mode()
def eval_metrics_random_tiles_patched(
    encoder: torch.nn.Module,
    mlp: torch.nn.Module,
    meta: dict,
    gt_rgb_u8: torch.Tensor,  # (C,H,W) uint8 on CPU
    num_tiles: int = 32,
    tile_size: int = 256,
    device: str = "cuda",
    half: bool = False,
    seed: int = 0,
) -> dict:
    """Estimate PSNR/SSIM/MS-SSIM over random tiles without writing reconstruction memmap."""
    C = int(meta["C"])
    H, W = map(int, meta["orig_hw"])
    if gt_rgb_u8.dtype != torch.uint8 or gt_rgb_u8.shape != (C, H, W):
        raise ValueError(f"gt_rgb_u8 must be uint8 with shape (C,H,W)=({C},{H},{W}), got {gt_rgb_u8.dtype} {tuple(gt_rgb_u8.shape)}")

    if tile_size < 11:
        raise ValueError("tile_size must be >= 11 for SSIM/MS-SSIM with window_size=11")

    rng = np.random.default_rng(int(seed))
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    psnr_vals = []
    ssim_vals = []
    msssim_vals = []

    for _ in range(int(num_tiles)):
        y0 = int(rng.integers(0, H - tile_size + 1))
        x0 = int(rng.integers(0, W - tile_size + 1))

        pred_u8 = predict_tile_patched_uint8(
            encoder=encoder, mlp=mlp, meta=meta,
            y0=y0, x0=x0, tile_size=tile_size,
            device=str(dev), half=half,
        )
        gt_u8 = gt_rgb_u8[:, y0:y0 + tile_size, x0:x0 + tile_size].to(dev, non_blocking=True)

        pred = pred_u8.float() / 255.0
        gt = gt_u8.float() / 255.0

        mse = (pred - gt).pow(2).mean().clamp_min(1e-12)
        psnr_vals.append(float(-10.0 * torch.log10(mse)))

        pred4 = pred.unsqueeze(0)
        gt4 = gt.unsqueeze(0)
        ssim_vals.append(float(ssim_torch(pred4, gt4)))
        msssim_vals.append(float(ms_ssim_torch(pred4, gt4)))

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "ms_ssim": float(np.mean(msssim_vals)),
    }


@torch.inference_mode()
def eval_full_psnr_patched_streamed(
    encoder: torch.nn.Module,
    mlp: torch.nn.Module,
    meta: dict,
    gt_rgb_u8: torch.Tensor,  # (C,H,W) uint8 on CPU
    rows_per_chunk: int = 64,  # in shuffled rows (Hp)
    device: str = "cuda",
    half: bool = False,
) -> dict:
    """Exact full-image MSE/PSNR without writing reconstruction to disk (streams over shuffled rows)."""
    K = int(meta["K"])
    C = int(meta["C"])
    H, W = map(int, meta["orig_hw"])
    Hp, Wp = map(int, meta["grid_hw"])
    CK2 = C * K * K

    if gt_rgb_u8.dtype != torch.uint8 or gt_rgb_u8.shape != (C, H, W):
        raise ValueError(f"gt_rgb_u8 must be uint8 with shape (C,H,W)=({C},{H},{W}), got {gt_rgb_u8.dtype} {tuple(gt_rgb_u8.shape)}")

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    encoder.eval(); mlp.eval()

    x_vals = torch.linspace(0.0, 1.0, Wp, dtype=torch.float32, device=device_t)

    se_sum = 0.0
    n_sum = 0

    for r0 in range(0, Hp, rows_per_chunk):
        r1 = min(Hp, r0 + rows_per_chunk)
        rows = r1 - r0

        if Hp > 1:
            y_vals = torch.linspace(r0, r1 - 1, steps=rows, dtype=torch.float32, device=device_t) / (Hp - 1)
        else:
            y_vals = torch.zeros((rows,), dtype=torch.float32, device=device_t)

        x = x_vals.unsqueeze(0).expand(rows, Wp)
        y = y_vals.unsqueeze(1).expand(rows, Wp)
        coords = torch.stack((x, y), dim=-1).flatten(0, 1) * 2.0 - 1.0

        use_amp = (device_t.type == "cuda") and half
        with torch.cuda.amp.autocast(enabled=use_amp):
            feats = encoder(coords)
            preds = mlp(feats)

        preds = preds.float().clamp_(0.0, 1.0).reshape(rows, Wp, CK2)
        preds_u8 = (preds * 255.0 + 0.5).to(torch.uint8)

        y_chunk = preds_u8.permute(2, 0, 1).unsqueeze(0).contiguous()
        x_chunk = (
            y_chunk.view(1, C, K, K, rows, Wp)
                  .permute(0, 1, 4, 2, 5, 3)
                  .contiguous()
                  .view(1, C, rows * K, Wp * K)
        )

        out_r0 = r0 * K
        out_r1 = min(out_r0 + rows * K, H)
        if out_r0 >= H:
            break
        chunk_h = out_r1 - out_r0

        pred_chunk_u8 = x_chunk[0, :, :chunk_h, :W]
        gt_chunk_u8 = gt_rgb_u8[:, out_r0:out_r1, :W].to(device_t, non_blocking=True)

        diff = (pred_chunk_u8.float() - gt_chunk_u8.float()) / 255.0
        se_sum += diff.square().sum().item()
        n_sum += diff.numel()

        if device_t.type == "cuda":
            del coords, feats, preds, preds_u8, y_chunk, x_chunk, pred_chunk_u8, gt_chunk_u8, diff
            torch.cuda.empty_cache()

    mse = se_sum / max(1, n_sum)
    psnr_db = float("inf") if mse == 0.0 else (-10.0 * math.log10(mse))
    return {"mse_full": float(mse), "psnr_full": float(psnr_db)}


def train(
    input_image: str,
    width: int,
    height: int,
    K: int,
    steps: int,
    batch_size: int,
    eval_every: int,
    levels: int,
    features: int,
    base_resolution: int,
    growth_factor: float,
    scale_step: float,
    resolutions: list[int] | None,
    log_hash_size: int,
    lr: float,
    d_hidden: int,
    device_str: str,
    out_prefix: str,
    val_samples: int,
    tile: int,
    save_png: bool,
    recon_chunk_pixels: int,
    half_recon: bool,
    entropy_grid: bool,
    resolution_scale: float,
    rim_blockwise_selector: bool,
    rim_block_size: int,
    rim_mask_iters: int,
    rim_max_candidates: int | None,
    rim_accept_tol: float,
    rim_fallback_init_scale: float,
    rim_mask_init_logit: float,
    rim_mask_temperature: float,
    rim_save_details: bool,
    rim_preprocess_tiled: bool,
    rim_preprocess_tile_blocks: int,
    rim_preprocess_device: str,
    rim_preprocess_stream_stats: bool,
    rim_enabled: bool,
    rim_info_tensor_path: str | None,
    rim_candidate_resolutions: list[int] | None,
    rim_max_init_iters: int,
    rim_init_tol: float,
    rim_init_scheduler: str,
    rim_gate_solver: str,
    rim_save_init_state: bool,
    rim_gate_mode: str,
    rim_gate_temperature: float,
    rim_gate_init_logit: float | None,
    rim_gate_regularization_weight: float,
    rim_fixed_gate_threshold: float,
    rim_fallback_mode: str,
):  
    device = torch.device(device_str if torch.cuda.is_available() or "cuda" in device_str else "cpu")
    print(f"Using device: {device}")

    # ---- Load & resize image as uint8 ----
    assert os.path.isfile(input_image), f"Input image not found: {input_image}"
    pil = Image.open(input_image).convert("RGB")
    if (pil.width, pil.height) != (width, height):
        try:
            resample = Image.Resampling.BICUBIC
        except AttributeError:
            resample = Image.BICUBIC
        pil = pil.resize((width, height), resample)
    image_uint8 = np.asarray(pil, dtype=np.uint8).copy()  # (H,W,3) uint8
    image_uint8 = torch.from_numpy(image_uint8).permute(2, 0, 1).contiguous().to(torch.uint8)
    # gt_rgb_u8 = image_uint8.clone()[0]
    gt_rgb_u8 = image_uint8.clone()
    image_uint8, ImageMeta = space_to_depth_patch2chan(image_uint8, K)
    image_uint8 = image_uint8[0] # remove the batch dim
    image_uint8 = image_uint8.permute(1, 2, 0).contiguous()

    
    rim_details = None
    level_block_mask = None
    level_block_gates = None
    rim_init_state = None
    if rim_enabled:
        if not rim_info_tensor_path:
            raise ValueError("--rim_info_tensor_path is required when --rim_enabled is set")
        H_tmp, W_tmp, _C_tmp = image_uint8.shape
        block_info = load_block_info_for_training(
            rim_info_tensor_path,
            expected_block_size=int(rim_block_size),
            expected_hw=(int(H_tmp), int(W_tmp)),
        )
        cand = rim_candidate_resolutions or [int(r) for r in block_info["candidate_resolutions"]]
        A = block_info["A"]
        table_sizes = [int(1 << int(log_hash_size)) for _ in range(int(levels))]
        print("[B-RIM] iterative initialization", flush=True)
        print(
            f"[B-RIM] info tensor: A={tuple(A.shape)} blocks={int(block_info['num_blocks'])} "
            f"layout={int(block_info['num_blocks_h'])}x{int(block_info['num_blocks_w'])} "
            f"candidates={len(cand)} levels={int(levels)}",
            flush=True,
        )
        rim_result = initialize_rim_iterative(
            A,
            cand,
            num_levels=int(levels),
            table_sizes=table_sizes,
            image_hw=(int(H_tmp), int(W_tmp)),
            block_size=int(rim_block_size),
            num_blocks_h=int(block_info["num_blocks_h"]),
            num_blocks_w=int(block_info["num_blocks_w"]),
            max_iters=int(rim_max_init_iters),
            tol=float(rim_init_tol),
            init_scheduler=str(rim_init_scheduler),
            gate_solver=str(rim_gate_solver),
            device=str(device),
            verbose=True,
        )
        resolutions = np.asarray(rim_result.resolutions, dtype=np.int64)
        level_block_gates = rim_result.gates
        rim_init_state = {
            "partition": rim_result.partition,
            "resolutions": rim_result.resolutions,
            "gates_init": rim_result.gates.cpu(),
            "gate_mode": str(rim_gate_mode),
            "fixed_gate_threshold": float(rim_fixed_gate_threshold),
            "gate_temperature": float(rim_gate_temperature),
            "gate_init_logit": None if rim_gate_init_logit is None else float(rim_gate_init_logit),
            "gate_regularization_weight": float(rim_gate_regularization_weight),
            "fallback_mode": str(rim_fallback_mode),
            "block_size": int(rim_block_size),
            "block_stride": block_info.get("block_stride"),
            "num_blocks_h": int(block_info["num_blocks_h"]),
            "num_blocks_w": int(block_info["num_blocks_w"]),
            "num_blocks": int(block_info["num_blocks"]),
            "objective_history": rim_result.objective_history,
            "candidate_resolutions": [int(r) for r in cand],
            "table_sizes": table_sizes,
            "info_tensor_path": rim_info_tensor_path,
            "init_method": "iterative_fixed_gate_dp",
            "details": rim_result.details,
        }
        print(f"[B-RIM] selected partition: {rim_result.partition}", flush=True)
        print(f"[B-RIM] selected resolutions: {rim_result.resolutions}", flush=True)
        print(f"[B-RIM] objective history: {rim_result.objective_history}", flush=True)
        gate_means = rim_result.gates.float().mean(dim=1)
        active = (rim_result.gates >= float(rim_fixed_gate_threshold)).sum(dim=1)
        print(f"[B-RIM] gate_mode={rim_gate_mode} fallback_mode={rim_fallback_mode}", flush=True)
        print(f"[B-RIM] avg gate per level: {[round(float(v), 4) for v in gate_means]}", flush=True)
        print(f"[B-RIM] active blocks per level@threshold: {[int(v) for v in active]}", flush=True)
        if rim_save_init_state:
            os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
            torch.save(rim_init_state, f"{out_prefix}_rim_init.pt")
            print(f"[B-RIM] saved init state: {out_prefix}_rim_init.pt", flush=True)
    elif rim_blockwise_selector:
        print("Apply blockwise masked RIM", flush=True)
        rim_result = solve_blockwise_masked_rim_exact(
            img=image_uint8.detach().cpu().numpy(),
            num_levels=levels,
            block_size=int(rim_block_size),
            hash_log2=log_hash_size,
            res_min=base_resolution,
            scale_step=scale_step,
            max_candidates=rim_max_candidates,
            max_iters=int(rim_mask_iters),
            accept_tol=float(rim_accept_tol),
            preprocess_tiled=bool(rim_preprocess_tiled),
            preprocess_tile_blocks=int(rim_preprocess_tile_blocks),
            preprocess_device=str(rim_preprocess_device),
            preprocess_stream_stats=bool(rim_preprocess_stream_stats),
            verbose=True,
        )
        resolutions = np.asarray(rim_result.selected_resolutions, dtype=np.int64)
        level_block_mask = torch.from_numpy(rim_result.level_block_mask.copy())
        rim_details = rim_result.details
        print(f"[masked-RIM] selected resolutions: {resolutions.tolist()}", flush=True)
        print(
            f"[masked-RIM] block_size={rim_result.block_size} num_blocks="
            f"{rim_result.num_blocks_y}x{rim_result.num_blocks_x} objective={rim_result.objective_value:.6e}",
            flush=True,
        )
        if rim_save_details:
            os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
            np.savez_compressed(
                f"{out_prefix}_masked_rim.npz",
                selected_resolutions=np.asarray(rim_result.selected_resolutions, dtype=np.int64),
                level_block_mask=np.asarray(rim_result.level_block_mask, dtype=np.uint8),
                candidate_resolutions=np.asarray(rim_result.candidate_resolutions, dtype=np.float64),
                block_band_mass=np.asarray(rim_result.block_band_mass, dtype=np.float64),
                mask_init_logit=np.asarray([float(rim_mask_init_logit)], dtype=np.float64),
                mask_temperature=np.asarray([float(rim_mask_temperature)], dtype=np.float64),
            )
            with open(f"{out_prefix}_masked_rim.json", "w", encoding="utf-8") as f:
                json.dump(rim_details, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    elif entropy_grid:
        print("Apply entropy grid", flush=True)
        resolutions, _ = get_entropy_grid(img=image_uint8.detach().cpu().numpy(),
                                          num_levels= levels, 
                                          res_min=base_resolution, 
                                          scale_step = scale_step, 
                                          hash_log2=log_hash_size)
        print(resolutions, flush=True)
        
        resolutions = resolutions * resolution_scale
        print(resolutions, flush=True)
    
    H, W, C = image_uint8.shape
    print(f"Loaded shuffled image (uint8): {W}x{H}")

    # ---- Build model ----
    selector_enabled = bool(rim_blockwise_selector or rim_enabled)
    selector_block_size = int(rim_block_size) if selector_enabled else 0
    selector_image_hw = (H, W) if selector_enabled else None
    encoder = MultiResolutionHashEncoder2D(
        levels=levels,
        features=features,
        base_resolution=base_resolution,
        growth_factor=growth_factor,
        log_hash_size=log_hash_size,
        resolutions=resolutions,
        strict_increasing=True,
        image_hw=selector_image_hw,
        block_size=selector_block_size,
        level_block_mask=level_block_mask,
        level_block_gates=level_block_gates,
        gate_mode=str(rim_gate_mode) if rim_enabled else "trainable_sigmoid",
        fallback_mode=str(rim_fallback_mode) if rim_enabled else "blockwise",
        fallback_init_scale=float(rim_fallback_init_scale),
        trainable_mask_init_logit=float(rim_mask_init_logit),
        trainable_gate_init_logit=rim_gate_init_logit if rim_enabled else None,
        trainable_mask_temperature=float(rim_gate_temperature if rim_enabled else rim_mask_temperature),
        fixed_gate_threshold=float(rim_fixed_gate_threshold),
    ).to(device)
    # Honor user-provided resolutions / entropy-grid resolutions.
    # If none are provided, fall back to a geometric progression.
    if resolutions is None:
        resolutions = [int(base_resolution * (growth_factor ** i)) for i in range(levels)]
    else:
        resolutions = [int(r) for r in resolutions]
        if len(resolutions) != levels:
            print(f"[info] overriding levels={levels} with len(resolutions)={len(resolutions)}")
            levels = len(resolutions)

    # encoder = MultiResolutionHashEncoder2D(
    #     input_dim=2,
    #     num_levels=levels,
    #     level_dim=features,
    #     per_level_scale=1,
    #     base_resolution=int(resolutions[0]),
    #     log2_hashmap_size=log_hash_size,
    #     desired_resolution=None,
    #     gridtype='hash',
    #     align_corners=False,
    #     allocate_params=True,
    #     resolutions=resolutions,
    # ).to(device)

    in_dim = levels * features
    mlp = ColorMLP(in_dim=in_dim, hidden=d_hidden, out_dim=C).to(device)
    encoder_params = count_trainable_params(encoder)
    mlp_params = count_trainable_params(mlp)
    est_hash_params = count_hashtable(resolutions, log_hash_size, features)
    print(f"Estimated active hash parameters: {est_hash_params:,}")
    print(f"Encoder trainable parameters: {encoder_params:,}")
    print(f"MLP trainable parameters: {mlp_params:,}")
    print(f"Total trainable parameters: {encoder_params + mlp_params:,}")
    # ---- Optimizer & scheduler ----
    # opt = torch.optim.AdamW(list(encoder.parameters()) + list(mlp.parameters()), lr=lr, betas=(0.9, 0.99), weight_decay=1e-6)
    opt = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": lr, "weight_decay": 0.0, "betas": (0.9, 0.99)},
            {"params": mlp.parameters(),     "lr": lr, "weight_decay": 1e-6, "betas": (0.9, 0.99)},
        ],
        eps=1e-15,
    )
    sched = CosineAnnealingLR(opt, T_max=max(1, steps))
    # breakpoint()

    # ---- Samplers ----
    sampler, val_coords, val_colors, _ = make_samplers(image_uint8, device, val_samples=val_samples)

    # ---- Training loop ----
    t0 = time.time()
    for step in range(1, steps + 1):
        coords_batch, colors_batch = sampler(batch_size)
        feats = encoder(coords_batch)
        preds = mlp(feats)
        loss = F.mse_loss(preds, colors_batch)
        if rim_enabled and float(rim_gate_regularization_weight) > 0.0 and getattr(encoder, "gate_mode", "") == "trainable_sigmoid":
            gamma = torch.sigmoid(encoder.mask_logits / float(rim_gate_temperature))
            reg = gamma.mean() + (gamma * (1.0 - gamma)).mean()
            loss = loss + float(rim_gate_regularization_weight) * reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == 1:
            with torch.no_grad():
                val_feats = encoder(val_coords)
                val_preds = mlp(val_feats)
                val_ps = psnr(val_preds, val_colors)
                tr_ps = psnr(preds, colors_batch)
                gate_stats = encoder.get_mask_gate_stats() if hasattr(encoder, "get_mask_gate_stats") else None
            extra = ""
            if gate_stats is not None:
                extra = (
                    f"  gate_mean={gate_stats['gate_mean']:.4f}"
                    f" gate_on={gate_stats.get('gate_mean_init_on', float('nan')):.4f}"
                    f" gate_off={gate_stats.get('gate_mean_init_off', float('nan')):.4f}"
                )
            print(f"[{step:6d}/{steps}] loss={loss.item():.6f}  train_PSNR={tr_ps:5.2f} dB  val_PSNR={val_ps:5.2f} dB{extra}", flush=True)

    t1 = time.time()
    print(f"Training time: {t1 - t0:.2f}s")
    print("Done.")
    return mlp, encoder, ImageMeta, gt_rgb_u8

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Memory-safe ultra-high-res image fitting with hash encoding.")
    p.add_argument("--input_image", type=str, required=True, help="Path to input image (any format Pillow supports).")
    p.add_argument("--width", type=int, required=True, help="Training width (image will be resized).")
    p.add_argument("--height", type=int, required=True, help="Training height (image will be resized).")
    p.add_argument("--K", type=int, default=3, help="Patch size")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=32768)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--levels", type=int, default=16)
    p.add_argument("--features", type=int, default=2)
    p.add_argument("--d_hidden", type=int, default=128)
    p.add_argument("--base_resolution", type=int, default=16)
    p.add_argument("--growth_factor", type=float, default=1.5)
    p.add_argument(
        "--resolutions",
        type=str,
        default=None,
        help=(
            "Optional per-level grid resolutions as comma-separated ints (overrides --levels/--base_resolution/--growth_factor). "
            'Example: --resolutions "16,32,64,128"'
        ))
    p.add_argument("--entropy_grid", action="store_true", help="Use entropy based grid resolution.")
    p.add_argument("--log_hash_size", type=int, default=19, help="Hash table size T per level.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scale_step", type=float, default=1.05)
    p.add_argument("--resolution_scale", type=float, default=1)
    p.add_argument("--rim_blockwise_selector", action="store_true", help="Use the blockwise masked RIM solver and encoder selector.")
    p.add_argument("--rim_block_size", type=int, default=32, help="Spatial block size on the shuffled training grid for the blockwise selector.")
    p.add_argument("--rim_mask_iters", type=int, default=4, help="Maximum number of alternating masked-RIM iterations.")
    p.add_argument("--rim_max_candidates", type=int, default=64, help="Maximum number of candidate bands for the masked-RIM solver.")
    p.add_argument("--rim_accept_tol", type=float, default=1e-9, help="Objective improvement tolerance for accepting a masked-RIM iteration.")
    p.add_argument("--rim_fallback_init_scale", type=float, default=1e-4, help="Initialization scale for the learned fallback vectors T used by masked blocks.")
    p.add_argument("--rim_mask_init_logit", type=float, default=4.0, help="Absolute initialization logit for the trainable sigmoid mask gates. Initial active entries use +value and inactive entries use -value.")
    p.add_argument("--rim_mask_temperature", type=float, default=1.0, help="Temperature used in sigmoid(mask_logits / temperature).")
    p.add_argument("--rim_save_details", action="store_true", help="Save the masked-RIM masks/details next to out_prefix.")
    p.add_argument("--rim_preprocess_tiled", action="store_true", help="Memory-safe RIM preprocessing: resize/quantize candidate bands patch by patch instead of materializing full residual_up/q tensors.")
    p.add_argument("--rim_preprocess_tile_blocks", type=int, default=8, help="Number of RIM block rows/cols per preprocessing tile when --rim_preprocess_tiled is enabled. Peak memory is roughly O((tile_blocks*rim_block_size)^2*C).")
    p.add_argument("--rim_preprocess_device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for RIM preprocessing. Use cpu for maximum safety; cuda is faster but can still OOM if tile size is too large.")
    p.add_argument("--rim_preprocess_stream_stats", action="store_true", help="Lowest-memory RIM preprocessing: compute only streamed per-block statistics for each low-pass difference band; never materialize g_low_up, residual, residual_up, or a full quantized image. Requires --rim_preprocess_tiled.")
    p.add_argument("--rim_enabled", action="store_true", help="Enable iterative B-RIM initialization from a precomputed block-info .pt file.")
    p.add_argument("--rim_info_tensor_path", type=str, default=None, help="Path to block_info.pt produced by tools/precompute_block_info.py.")
    p.add_argument(
        "--rim_candidate_resolutions",
        type=str,
        default=None,
        help='Optional candidate resolutions for B-RIM as comma-separated ints. Defaults to the .pt metadata.',
    )
    p.add_argument("--rim_max_init_iters", type=int, default=20)
    p.add_argument("--rim_init_tol", type=float, default=1e-6)
    p.add_argument("--rim_init_scheduler", type=str, default="geometric", choices=["geometric", "linear"])
    p.add_argument("--rim_gate_solver", type=str, default="exact_fractional", choices=["exact_fractional", "prefix_fallback"])
    p.add_argument("--rim_save_init_state", action="store_true")
    p.add_argument("--rim_gate_mode", type=str, default="trainable_sigmoid", choices=["trainable_sigmoid", "fixed_binary"])
    p.add_argument("--rim_gate_temperature", type=float, default=1.0)
    p.add_argument(
        "--rim_gate_init_logit",
        type=float,
        default=None,
        help="Optional absolute initial logit for trainable B-RIM block gates; e.g. 2 gives sigmoid +/-2.",
    )
    p.add_argument("--rim_gate_regularization_weight", type=float, default=0.0)
    p.add_argument("--rim_fixed_gate_threshold", type=float, default=0.5)
    p.add_argument("--rim_fallback_mode", type=str, default="blockwise", choices=["blockwise", "global_shared"])
    
    p.add_argument("--device", type=str, default="cuda", help='Device string, e.g., "cuda" or "cpu".')
    p.add_argument("--out_prefix", type=str, default="fit", help="Prefix for outputs.")
    p.add_argument("--val_samples", type=int, default=100_000, help="Validation sample count.")
    p.add_argument("--tile", type=int, default=1024, help="Tile height for reconstruction.")
    p.add_argument("--recon_chunk_pixels", type=int, default=1_000_000, help="Max pixels per reconstruction sub-chunk (controls VRAM use).")
    p.add_argument("--half_recon", action="store_true", help="Use FP16 for reconstruction to reduce VRAM.")
    p.add_argument("--save_png", action="store_true", help="Attempt to write final PNG (skips if image is huge).")
    p.add_argument("--metrics", action="store_true", help="Compute PSNR/SSIM/MS-SSIM on random tiles (after reconstruction, or in-memory with --eval_only).")
    p.add_argument("--metrics_tiles", type=int, default=32, help="Number of random tiles for metric estimation.")
    p.add_argument("--metrics_tile_size", type=int, default=256, help="Tile size for metric estimation.")
    p.add_argument("--metrics_device", type=str, default="cuda", help="Device for metric computation (cuda/cpu).")
    p.add_argument("--metrics_seed", type=int, default=0, help="RNG seed for metric tile sampling.")
    p.add_argument("--eval_only", action="store_true", help="Evaluate without writing reconstruction memmap (no reconst.npy).")
    p.add_argument("--eval_full_psnr", action="store_true", help="(With --eval_only) also compute exact full-image PSNR by streaming over all pixels.")
    p.add_argument(
       "--reconst_dir",
        type=str,
        default=".",
        help="Directory to save reconstruction artifacts (reconst.npy, optional BigTIFF).")
    p.add_argument(
         "--reconst_name",
         type=str,
         default="reconst.npy",
         help='Reconstruction memmap filename (default: "reconst.npy").',
     )
    p.add_argument(
         "--write_bigtiff",
         nargs="?",
         const="recon.tif",
         default=None,
         help=(
             "If provided, export BigTIFF (tiled). "
             "Usage: --write_bigtiff [path]. If you pass the flag without a path, defaults to recon.tif. "
             "If omitted, no BigTIFF will be written."
         ),
     )
    p.add_argument("--bigtiff_tile", type=int, default=256, help="Tile size for BigTIFF export.")
    p.add_argument(
         "--delete_reconst",
         action="store_true",
         help="Delete reconst.npy after metrics/BigTIFF are done.",
     )
    return p.parse_args()


def main():
    args = parse_args()
    resolutions=None
    if not args.entropy_grid:
        resolutions = _parse_resolutions_arg(args.resolutions)
        if resolutions is not None and args.levels != len(resolutions):
            print(f"[info] overriding --levels={args.levels} with len(--resolutions)={len(resolutions)}")
    rim_candidate_resolutions = _parse_resolutions_arg(args.rim_candidate_resolutions)
    
    mlp, encoder, meta, gt_rgb_u8 = train(
        input_image=args.input_image,
        width=args.width,
        height=args.height,
        K = args.K,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        levels=(len(resolutions) if resolutions is not None else args.levels),
        features=args.features,
        base_resolution=args.base_resolution,
        growth_factor=args.growth_factor,
        scale_step=args.scale_step,
        resolutions=resolutions,
        log_hash_size=args.log_hash_size,
        lr=args.lr,
        d_hidden = args.d_hidden,
        device_str=args.device,
        out_prefix=args.out_prefix,
        val_samples=args.val_samples,
        tile=args.tile,
        save_png=bool(args.save_png),
        recon_chunk_pixels=int(args.recon_chunk_pixels),
        half_recon=bool(args.half_recon),
        entropy_grid=bool(args.entropy_grid),
        resolution_scale = args.resolution_scale,
        rim_blockwise_selector=bool(args.rim_blockwise_selector),
        rim_block_size=int(args.rim_block_size),
        rim_mask_iters=int(args.rim_mask_iters),
        rim_max_candidates=(None if int(args.rim_max_candidates) <= 0 else int(args.rim_max_candidates)),
        rim_accept_tol=float(args.rim_accept_tol),
        rim_fallback_init_scale=float(args.rim_fallback_init_scale),
        rim_mask_init_logit=float(args.rim_mask_init_logit),
        rim_mask_temperature=float(args.rim_mask_temperature),
        rim_save_details=bool(args.rim_save_details),
        rim_preprocess_tiled=bool(args.rim_preprocess_tiled),
        rim_preprocess_tile_blocks=int(args.rim_preprocess_tile_blocks),
        rim_preprocess_device=str(args.rim_preprocess_device),
        rim_preprocess_stream_stats=bool(args.rim_preprocess_stream_stats),
        rim_enabled=bool(args.rim_enabled),
        rim_info_tensor_path=args.rim_info_tensor_path,
        rim_candidate_resolutions=rim_candidate_resolutions,
        rim_max_init_iters=int(args.rim_max_init_iters),
        rim_init_tol=float(args.rim_init_tol),
        rim_init_scheduler=str(args.rim_init_scheduler),
        rim_gate_solver=str(args.rim_gate_solver),
        rim_save_init_state=bool(args.rim_save_init_state),
        rim_gate_mode=str(args.rim_gate_mode),
        rim_gate_temperature=float(args.rim_gate_temperature),
        rim_gate_init_logit=args.rim_gate_init_logit,
        rim_gate_regularization_weight=float(args.rim_gate_regularization_weight),
        rim_fixed_gate_threshold=float(args.rim_fixed_gate_threshold),
        rim_fallback_mode=str(args.rim_fallback_mode),
    )
    os.makedirs(args.reconst_dir, exist_ok=True)
    reconst_path = os.path.join(args.reconst_dir, args.reconst_name)
    def _resolve_out_path(base_dir: str, maybe_rel_path: str | None) -> str | None:
        """If maybe_rel_path is a bare filename (no directory), place it in base_dir."""
        if maybe_rel_path is None:
            return None
        if os.path.isabs(maybe_rel_path):
            return maybe_rel_path
        # if user provided a subdir like "out/recon.tif", respect it
        if os.path.dirname(maybe_rel_path):
            return maybe_rel_path
        return os.path.join(base_dir, maybe_rel_path)
    bigtiff_path = _resolve_out_path(args.reconst_dir, args.write_bigtiff)

    if args.eval_only:
        if bigtiff_path is not None:
            print("[warn] --eval_only set; skipping BigTIFF export because it requires full reconstruction.")
        metrics_out = {}
        if args.metrics:
            metrics_out.update(
                eval_metrics_random_tiles_patched(
                    encoder=encoder,
                    mlp=mlp,
                    meta=meta,
                    gt_rgb_u8=gt_rgb_u8,
                    num_tiles=int(args.metrics_tiles),
                    tile_size=int(args.metrics_tile_size),
                    device=str(args.metrics_device),
                    half=bool(args.half_recon),
                    seed=int(args.metrics_seed),
                )
            )
        # Exact full-image PSNR (streams over all pixels). If no tile-metrics requested, compute this by default.
        if args.eval_full_psnr or (not args.metrics):
            metrics_out.update(
                eval_full_psnr_patched_streamed(
                    encoder=encoder,
                    mlp=mlp,
                    meta=meta,
                    gt_rgb_u8=gt_rgb_u8,
                    rows_per_chunk=64,
                    device=args.device,
                    half=bool(args.half_recon),
                )
            )
        print(metrics_out)
        return

    reconstruct_patched_rows_streamed(
        encoder=encoder,
        mlp=mlp,
        meta=meta,
        out_npy=reconst_path,
        rows_per_chunk=64,
        device=args.device,
        half=bool(args.half_recon),
        write_bigtiff=bigtiff_path,
        tiff_tile=int(args.bigtiff_tile),
    )

    if args.metrics:
        metrics = compute_metrics_tiled(
            pred_memmap_path=reconst_path,
            gt_rgb_u8=gt_rgb_u8,
            num_tiles=int(args.metrics_tiles),
            tile_size=int(args.metrics_tile_size),
            device_str=str(args.metrics_device),
            seed=int(args.metrics_seed),
        )
        print(metrics)

    if args.delete_reconst:
        try:
            os.remove(reconst_path)
            print(f"Deleted: {reconst_path}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[warn] Could not delete {reconst_path}: {e}")
if __name__ == "__main__":
    main()
