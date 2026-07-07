"""Context-free per-particle encoder — the particle-library Φ.

``ParticleEncoder`` maps each constituent *independently* to a token:
no attention, no jet context. The same (PID, charge, kinematics) always
produces the same token, anywhere in any event — the property required for a
universal particle vocabulary and for cross-experiment alignment (CMS/ATLAS
particles with the same properties land in the same place by construction).

Categorical inputs enter through embeddings, not raw code values: PID via a
PDG-id lookup table (unknown codes fall into an "other" bucket) and charge via
a 3-way embedding, concatenated with the continuous kinematic columns and
passed through the repo MLP (``CoderModel``).
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

from heptokens.models.coders import BaseEncoder, CoderModel

log = logging.getLogger(__name__)

# CMS particle-flow candidate PDG ids (NanoAOD convention) as the default
# vocabulary; anything else lands in the trailing "other" bucket.
DEFAULT_PDG_IDS = (22, 211, -211, 130, 11, -11, 13, -13, 1, 2)


class ParticleEncoder(BaseEncoder):
    """Per-particle MLP with embedded categorical (PID, charge) inputs.

    Args:
        input_dim: Total number of feature columns (validated against the
            configured indices; injected by ``LitContrastiveTokenizer``).
        output_dim: Token dimension.
        continuous_indices: Columns holding continuous (scaled) kinematics.
        pid_index: Column holding the PDG id (raw code values, not scaled) —
            ``None`` disables the PID embedding.
        charge_index: Column holding charge in {-1, 0, +1} — ``None`` disables.
        pdg_ids: Known PDG codes; unknown codes map to an "other" bucket.
        pid_emb_dim / charge_emb_dim: Embedding widths.
        hidden_dims: MLP hidden layers (``CoderModel``).
    """

    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int = 16,
        continuous_indices: tuple[int, ...] = (0, 1, 2),
        pid_index: int | None = None,
        charge_index: int | None = None,
        pdg_ids: tuple[int, ...] = DEFAULT_PDG_IDS,
        pid_emb_dim: int = 8,
        charge_emb_dim: int = 2,
        hidden_dims: tuple[int, ...] = (128, 128, 64),
        input_key: str = "csts",
        mask_key: str = "mask",
    ):
        super().__init__()
        self.continuous_indices = list(continuous_indices)
        self.pid_index = pid_index
        self.charge_index = charge_index
        self.input_key = input_key
        self.mask_key = mask_key

        used = [*self.continuous_indices]
        if pid_index is not None:
            used.append(pid_index)
        if charge_index is not None:
            used.append(charge_index)
        if input_dim is not None and max(used) >= input_dim:
            raise ValueError(
                f"Feature index {max(used)} out of range for input_dim={input_dim}"
            )

        fused_dim = len(self.continuous_indices)
        if pid_index is not None:
            self.register_buffer("pdg_ids", torch.tensor(pdg_ids, dtype=torch.long))
            self.pid_embedding = nn.Embedding(len(pdg_ids) + 1, pid_emb_dim)
            fused_dim += pid_emb_dim
        if charge_index is not None:
            self.charge_embedding = nn.Embedding(3, charge_emb_dim)
            fused_dim += charge_emb_dim

        self.coder = CoderModel(
            input_dim=fused_dim, output_dim=output_dim, hidden_dims=list(hidden_dims)
        )

    def _map_pid(self, pid: torch.Tensor) -> torch.Tensor:
        """Map raw PDG codes to contiguous ids; unknown -> the 'other' bucket."""
        matches = pid.long().unsqueeze(-1) == self.pdg_ids  # [..., n_pdg]
        idx = matches.float().argmax(dim=-1)
        other = torch.full_like(idx, self.pdg_ids.shape[0])
        return torch.where(matches.any(dim=-1), idx, other)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch[self.input_key]
        mask = batch.get(self.mask_key)

        parts = [x[..., self.continuous_indices]]
        if self.pid_index is not None:
            parts.append(self.pid_embedding(self._map_pid(x[..., self.pid_index])))
        if self.charge_index is not None:
            charge = (x[..., self.charge_index].round().long() + 1).clamp(0, 2)
            parts.append(self.charge_embedding(charge))
        fused = torch.cat(parts, dim=-1)

        return self.coder(fused, mask)
