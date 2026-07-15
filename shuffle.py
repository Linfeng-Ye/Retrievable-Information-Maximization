import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2_000_000_000

def space_to_depth_patch2chan(
    x: torch.Tensor,
    K: int,
) -> Tuple[torch.Tensor, Dict]:
    """
    Forward: pack non-overlapping KxK blocks into channels.

    Input:
        x: (B,C,H,W) or (C,H,W)
    Output:
        y: (B, C*K*K, H', W') where H'=ceil(H/K), W'=ceil(W/K)
        meta: info needed for exact inversion (incl. crop)
    Padding:
        replicate/edge padding to make H,W multiples of K.
    """
    if K <= 0:
        raise ValueError("K must be a positive integer.")

    squeezed = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeezed = True
    if x.dim() != 4:
        raise ValueError("x must be (B,C,H,W) or (C,H,W).")

    B, C, H, W = x.shape

    pad_h = (K - (H % K)) % K
    pad_w = (K - (W % K)) % K
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")  # (L,R,T,B) = (0,pad_w,0,pad_h)

    Hpad, Wpad = H + pad_h, W + pad_w
    Hp, Wp = Hpad // K, Wpad // K

    # (B,C,Hpad,Wpad) -> (B,C,Hp,K,Wp,K) -> (B,C,K,K,Hp,Wp) -> (B,C*K*K,Hp,Wp)
    y = (
        x_pad.view(B, C, Hp, K, Wp, K)
             .permute(0, 1, 3, 5, 2, 4)   # B, C, K, K, Hp, Wp
             .contiguous()
             .view(B, C * K * K, Hp, Wp)
    )

    meta = {
        "K": K,
        "orig_hw": (H, W),
        "pad_hw": (Hpad, Wpad),
        "grid_hw": (Hp, Wp),
        "C": C,
        "squeezed": squeezed,
    }
    return y, meta


def depth_to_space_chan2patch(
    y: torch.Tensor,
    meta: Dict,
    *,
    crop_to_original: bool = True,
) -> torch.Tensor:
    """
    Inverse: unpack channels back into KxK blocks (depth-to-space).

    Input:
        y: (B, C*K*K, H', W') from space_to_depth_patch2chan
    Output:
        x: (B,C,H,W) (cropped to original if requested), or (C,H,W) if original was 3D.
    """
    if y.dim() != 4:
        raise ValueError("y must be (B, C*K*K, H', W').")

    K = int(meta["K"])
    C = int(meta["C"])
    H, W = meta["orig_hw"]
    Hpad, Wpad = meta["pad_hw"]
    Hp, Wp = meta["grid_hw"]

    B, CK2, Hp_y, Wp_y = y.shape
    if Hp_y != Hp or Wp_y != Wp:
        raise ValueError(f"Grid mismatch: expected (H',W')=({Hp},{Wp}), got ({Hp_y},{Wp_y}).")
    if CK2 != C * K * K:
        raise ValueError(f"Channel mismatch: expected {C*K*K}, got {CK2}.")

    # (B, C*K*K, Hp, Wp) -> (B, C, K, K, Hp, Wp) -> (B, C, Hp, K, Wp, K) -> (B,C,Hpad,Wpad)
    x_pad = (
        y.view(B, C, K, K, Hp, Wp)
         .permute(0, 1, 4, 2, 5, 3)       # B, C, Hp, K, Wp, K
         .contiguous()
         .view(B, C, Hpad, Wpad)
    )

    x = x_pad[:, :, :H, :W] if crop_to_original else x_pad

    if meta.get("squeezed", False):
        x = x.squeeze(0)  # back to (C,H,W)

    return x
