import numpy as np
import torch
import torch.nn.functional as F

def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    """
    pred, target: float tensors in [0,1], shape (N,3) or (H,W,3) flattened.
    Returns PSNR in dB (float).
    """
    mse = F.mse_loss(pred, target, reduction="mean").clamp_min(eps)
    return float(10.0 * torch.log10(1.0 / mse).item())


def _try_import_metrics():
    ssim_fn = None
    ms_ssim_fn = None
    lpips_model = None
    try:
        from pytorch_msssim import ssim as _ssim, ms_ssim as _ms_ssim  # type: ignore
        ssim_fn = _ssim
        ms_ssim_fn = _ms_ssim
    except Exception:
        pass
    try:
        import lpips  # type: ignore
        lpips_model = lpips.LPIPS(net="alex")
    except Exception:
        pass
    return ssim_fn, ms_ssim_fn, lpips_model

@torch.inference_mode()
def compute_metrics_tiled(
    pred_memmap_path: str,
    gt_rgb_u8: torch.Tensor,
    *,
    num_tiles: int = 32,
    tile_size: int = 256,
    device_str: str = "cuda",
    seed: int = 0,
) -> dict:
    """Compute PSNR/SSIM/MS-SSIM/LPIPS on random tiles."""
    device = torch.device(device_str if (torch.cuda.is_available() and "cuda" in device_str) else "cpu")
    C, H, W = gt_rgb_u8.shape
    assert C == 3, "Metrics code assumes RGB ground truth."
    pred = np.memmap(pred_memmap_path, mode="r", dtype=np.uint8, shape=(H, W, C))
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, max(1, W - tile_size + 1), size=num_tiles, endpoint=False)
    ys = rng.integers(0, max(1, H - tile_size + 1), size=num_tiles, endpoint=False)
    ssim_fn, ms_ssim_fn, lpips_model = _try_import_metrics()
    if lpips_model is not None:
        lpips_model = lpips_model.to(device).eval()
    psnrs, ssims, msssims, lpipss = [], [], [], []
    for x0, y0 in zip(xs.tolist(), ys.tolist()):
        gt_tile = gt_rgb_u8[:, y0:y0+tile_size, x0:x0+tile_size].to(device=device, dtype=torch.float32) / 255.0
        pred_tile_np = np.asarray(pred[y0:y0+tile_size, x0:x0+tile_size, :])
        pred_tile = torch.from_numpy(pred_tile_np).permute(2, 0, 1).to(device=device, dtype=torch.float32) / 255.0
        gt_tile = gt_tile.unsqueeze(0)
        pred_tile = pred_tile.unsqueeze(0)
        psnrs.append(psnr(pred_tile, gt_tile))
        if ssim_fn is not None:
            try:
                ssims.append(float(ssim_fn(pred_tile, gt_tile, data_range=1.0, size_average=True).item()))
            except Exception:
                pass
        if ms_ssim_fn is not None:
            try:
                msssims.append(float(ms_ssim_fn(pred_tile, gt_tile, data_range=1.0, size_average=True).item()))
            except Exception:
                pass
        if lpips_model is not None:
            try:
                lp = lpips_model(pred_tile * 2 - 1, gt_tile * 2 - 1)
                lpipss.append(float(lp.mean().item()))
            except Exception:
                pass
    out = {
        "PSNR(dB)_mean": float(np.mean(psnrs)) if psnrs else None,
        "PSNR(dB)_std": float(np.std(psnrs)) if psnrs else None,
        "SSIM_mean": float(np.mean(ssims)) if ssims else None,
        "MS_SSIM_mean": float(np.mean(msssims)) if msssims else None,
        "LPIPS_mean": float(np.mean(lpipss)) if lpipss else None,
        "tiles_used": int(num_tiles),
        "tile_size": int(tile_size),
    }
    missing = []
    if out["SSIM_mean"] is None or out["MS_SSIM_mean"] is None:
        missing.append("pytorch-msssim")
    if out["LPIPS_mean"] is None:
        missing.append("lpips")
    if missing:
        print(f"[warn] Some metrics unavailable (missing or failed): {missing}. Install via: pip install {' '.join(missing)}")
    return out


@torch.inference_mode()
def compute_metrics_tiled_array(
    pred_rgb_u8: np.ndarray,
    gt_rgb_u8: torch.Tensor,
    *,
    num_tiles: int = 32,
    tile_size: int = 256,
    device_str: str = "cuda",
    seed: int = 0,
) -> dict:
    device = torch.device(device_str if (torch.cuda.is_available() and "cuda" in device_str) else "cpu")
    C, H, W = gt_rgb_u8.shape
    assert C == 3
    assert pred_rgb_u8.shape == (H, W, 3)

    rng = np.random.default_rng(seed)
    xs = rng.integers(0, max(1, W - tile_size + 1), size=num_tiles, endpoint=False)
    ys = rng.integers(0, max(1, H - tile_size + 1), size=num_tiles, endpoint=False)

    ssim_fn, ms_ssim_fn, lpips_model = _try_import_metrics()
    if lpips_model is not None:
        lpips_model = lpips_model.to(device).eval()

    psnrs, ssims, msssims, lpipss = [], [], [], []
    for x0, y0 in zip(xs.tolist(), ys.tolist()):
        gt_tile = gt_rgb_u8[:, y0:y0+tile_size, x0:x0+tile_size].to(device=device, dtype=torch.float32) / 255.0
        pred_tile = torch.from_numpy(pred_rgb_u8[y0:y0+tile_size, x0:x0+tile_size, :]).permute(2,0,1)
        pred_tile = pred_tile.to(device=device, dtype=torch.float32) / 255.0

        gt_tile = gt_tile.unsqueeze(0)
        pred_tile = pred_tile.unsqueeze(0)

        psnrs.append(psnr(pred_tile, gt_tile))
        if ssim_fn is not None:
            try: ssims.append(float(ssim_fn(pred_tile, gt_tile, data_range=1.0, size_average=True).item()))
            except: pass
        if ms_ssim_fn is not None:
            try: msssims.append(float(ms_ssim_fn(pred_tile, gt_tile, data_range=1.0, size_average=True).item()))
            except: pass
        if lpips_model is not None:
            try:
                lp = lpips_model(pred_tile * 2 - 1, gt_tile * 2 - 1)
                lpipss.append(float(lp.mean().item()))
            except: pass

    return {
        "PSNR(dB)_mean": float(np.mean(psnrs)) if psnrs else None,
        "PSNR(dB)_std": float(np.std(psnrs)) if psnrs else None,
        "SSIM_mean": float(np.mean(ssims)) if ssims else None,
        "MS_SSIM_mean": float(np.mean(msssims)) if msssims else None,
        "LPIPS_mean": float(np.mean(lpipss)) if lpipss else None,
        "tiles_used": int(num_tiles),
        "tile_size": int(tile_size),
    }