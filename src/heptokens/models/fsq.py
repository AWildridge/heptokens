"""Finite Scalar Quantization (FSQ) — codebook-free discrete bottleneck.

Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple"
(arXiv:2309.15505). Each latent dimension ``i`` is bounded with a tanh and
rounded to one of ``levels[i]`` uniformly spaced values, so the token lives on
a fixed grid of ``prod(levels)`` points (``sum(log2(levels))`` bits/particle).
There are **no learned parameters**: no codebook to collapse, no EMA dynamics,
no commitment loss — the mechanisms behind the Campaign-4/5 contrastive-VQ
negative are absent by construction. Gradients pass the rounding via the
straight-through estimator.
"""

from typing import Sequence, Tuple

import torch
import torch.nn as nn


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient."""
    return z + (z.round() - z).detach()


class FSQ(nn.Module):
    """Quantize ``[..., len(levels)]`` latents to a fixed per-dimension grid.

    ``forward`` returns ``(z_q, z_e)``, both normalized to roughly ``[-1, 1]``:
    ``z_q`` is the released grid token (rounded, STE gradient) and ``z_e`` the
    bounded pre-rounding latent on the same scale — the smooth surface the
    physics kernel can shape with a clean gradient (the Campaign-5 placement
    result), with the rounding error bounded by half a grid step.
    """

    def __init__(self, levels: Sequence[int], eps: float = 1e-3):
        super().__init__()
        if any(int(l) < 2 for l in levels):
            raise ValueError(f"FSQ levels must all be >= 2, got {levels!r}")
        levels_t = torch.tensor([int(l) for l in levels], dtype=torch.float32)
        self.register_buffer("levels", levels_t)
        # Even level counts have no code at 0; shift the tanh so the codes
        # land on half-integers (the reference implementation's offset).
        half_l = (levels_t - 1) * (1 - eps) / 2
        offset = torch.where(levels_t % 2 == 0, 0.5, 0.0)
        self.register_buffer("_half_l", half_l)
        self.register_buffer("_offset", offset)
        self.register_buffer("_shift", (offset / half_l).atanh())
        self.register_buffer("_half_width", levels_t // 2)

    @property
    def num_dims(self) -> int:
        return int(self.levels.numel())

    @property
    def bits(self) -> float:
        return float(self.levels.log2().sum())

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Map unconstrained latents into the (unnormalized) grid range."""
        return (z + self._shift).tanh() * self._half_l - self._offset

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_e = self.bound(z)
        z_q = round_ste(z_e)
        return z_q / self._half_width, z_e / self._half_width

    @torch.no_grad()
    def codes_to_indices(self, z_q_normalized: torch.Tensor) -> torch.Tensor:
        """Mixed-radix integer index of each grid token (for release/diagnostics).

        The rounded (unnormalized) codes span ``L//2`` below zero for both
        parities, so ``digit = code + L//2`` lands in ``0..L-1``.
        """
        digits = (z_q_normalized * self._half_width + self._half_width).round()
        basis = torch.cumprod(
            torch.cat([torch.ones_like(self.levels[:1]), self.levels[:-1]]), dim=0
        )
        return (digits * basis).sum(dim=-1).long()
