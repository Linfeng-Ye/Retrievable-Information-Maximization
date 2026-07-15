from __future__ import annotations

import torch

from rim.block_info import compute_block_info_batch
from rim.initializer import initialize_rim_iterative
from hash_encoder_2d import MultiResolutionHashEncoder2D


def test_block_info_and_initializer_smoke():
    x = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    candidate_resolutions = [4, 8, 16, 24]
    A, meta = compute_block_info_batch(x, candidate_resolutions, block_size=8)
    assert A.shape == (2, 16, 4)
    assert torch.isfinite(A).all()
    assert float(A.min()) >= 0.0

    out = initialize_rim_iterative(
        A,
        candidate_resolutions,
        num_levels=2,
        table_sizes=[128, 128],
        image_hw=(32, 32),
        block_size=8,
        num_blocks_h=int(meta["num_blocks_h"]),
        num_blocks_w=int(meta["num_blocks_w"]),
        max_iters=4,
        device="cpu",
    )
    assert len(out.partition) == 2
    assert out.partition[0][0] == 0
    assert out.partition[-1][1] == len(candidate_resolutions) - 1
    assert out.gates.shape == (2, 16)
    assert torch.logical_and(out.gates >= 0, out.gates <= 1).all()
    assert all(torch.isfinite(torch.tensor(out.objective_history)))


def test_block_info_channelwise_lowpass_matches_default():
    x = torch.randint(0, 256, (1, 3, 24, 24), dtype=torch.uint8)
    candidate_resolutions = [4, 8, 12]
    A_default, meta_default = compute_block_info_batch(x, candidate_resolutions, block_size=6)
    A_channelwise, meta_channelwise = compute_block_info_batch(
        x,
        candidate_resolutions,
        block_size=6,
        lowpass_channelwise=True,
    )
    assert meta_channelwise == meta_default
    assert torch.allclose(A_channelwise, A_default)


def test_block_info_tiled_entropy_matches_default():
    x = torch.randint(0, 256, (1, 3, 30, 28), dtype=torch.uint8)
    candidate_resolutions = [4, 8, 12]
    A_default, meta_default = compute_block_info_batch(
        x,
        candidate_resolutions,
        block_size=[7, 6],
        block_stride=[5, 4],
    )
    A_tiled, meta_tiled = compute_block_info_batch(
        x,
        candidate_resolutions,
        block_size=[7, 6],
        block_stride=[5, 4],
        entropy_block_chunk_size=3,
    )
    assert meta_tiled == meta_default
    assert torch.allclose(A_tiled, A_default)


def test_block_info_tiled_channelwise_lowpass_matches_default():
    x = torch.randint(0, 256, (1, 3, 24, 24), dtype=torch.uint8)
    candidate_resolutions = [4, 8, 12]
    A_default, meta_default = compute_block_info_batch(
        x,
        candidate_resolutions,
        block_size=6,
        lowpass_channelwise=True,
    )
    A_tiled, meta_tiled = compute_block_info_batch(
        x,
        candidate_resolutions,
        block_size=6,
        lowpass_channelwise=True,
        entropy_block_chunk_size=2,
    )
    assert meta_tiled == meta_default
    assert torch.allclose(A_tiled, A_default)


def test_encoder_gate_modes_smoke():
    gates = torch.rand(2, 16)
    coords = torch.rand(64, 2) * 2.0 - 1.0
    enc_trainable = MultiResolutionHashEncoder2D(
        levels=2,
        features=2,
        log_hash_size=4,
        resolutions=[4, 8],
        image_hw=(32, 32),
        block_size=8,
        level_block_gates=gates,
        gate_mode="trainable_sigmoid",
        fallback_mode="global_shared",
    )
    y = enc_trainable(coords)
    assert y.shape == (64, 4)
    assert enc_trainable.mask_logits.requires_grad

    enc_fixed = MultiResolutionHashEncoder2D(
        levels=2,
        features=2,
        log_hash_size=4,
        resolutions=[4, 8],
        image_hw=(32, 32),
        block_size=8,
        level_block_gates=gates,
        gate_mode="fixed_binary",
        fallback_mode="blockwise",
    )
    y2 = enc_fixed(coords)
    assert y2.shape == (64, 4)
    assert "gates" in dict(enc_fixed.named_buffers())
