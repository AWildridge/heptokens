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
import torch.nn.functional as F

from heptokens.models.coders import BaseDecoder, BaseEncoder, CoderModel
from heptokens.models.transformer import Transformer

log = logging.getLogger(__name__)

# CMS particle-flow candidate PDG ids (NanoAOD convention) as the default
# vocabulary; anything else lands in the trailing "other" bucket.
DEFAULT_PDG_IDS = (22, 211, -211, 130, 11, -11, 13, -13, 1, 2)


def map_pdg(pid: torch.Tensor, pdg_ids: torch.Tensor) -> torch.Tensor:
    """Map raw PDG codes to contiguous class ids; unknown -> the 'other' bucket."""
    matches = pid.long().unsqueeze(-1) == pdg_ids
    idx = matches.float().argmax(dim=-1)
    other = torch.full_like(idx, pdg_ids.shape[0])
    return torch.where(matches.any(dim=-1), idx, other)


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
        hidden_dims: MLP hidden layers (``CoderModel``), used when
            ``backbone="mlp"``.
        backbone: ``"mlp"`` (context-free per-particle ``CoderModel`` — the
            same (PID, charge, kinematics) always maps to the same token) or
            ``"attention"`` (a ``Transformer`` over the jet's constituents,
            so a constituent's token depends on the rest of the jet —
            context-aware, richer but no longer a universal per-particle map).
        attn_layers / d_model / n_heads: Transformer config for
            ``backbone="attention"``.
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
        backbone: str = "mlp",
        attn_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
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

        if backbone == "mlp":
            # Context-free: each constituent is embedded independently.
            self.coder = CoderModel(
                input_dim=fused_dim, output_dim=output_dim, hidden_dims=list(hidden_dims)
            )
        elif backbone == "attention":
            # Context-aware: attention over the jet's constituents. Same
            # ``(x, mask)`` call as CoderModel, so ``forward`` is unchanged —
            # but a constituent's token now depends on the rest of the jet.
            self.coder = Transformer(
                input_dim=fused_dim,
                output_dim=output_dim,
                d_model=d_model,
                n_heads=n_heads,
                num_layers=attn_layers,
                dim_feedforward=2 * d_model,
            )
        else:
            raise ValueError(f"backbone must be 'mlp' or 'attention', got {backbone!r}")
        self.backbone = backbone

    def _map_pid(self, pid: torch.Tensor) -> torch.Tensor:
        """Map raw PDG codes to contiguous ids; unknown -> the 'other' bucket."""
        return map_pdg(pid, self.pdg_ids)

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


class ParticleDecoder(BaseDecoder):
    """Full-particle reconstruction decoder for VQ baselines on mixed features.

    Reconstructs the *whole* particle from the quantized token: L1 on the
    continuous (scaled) kinematics plus cross-entropy on the PID class and on
    the charge — so categorical identity information is forced through the
    bottleneck instead of being silently dropped (a kinematics-only target
    would strawman VQ on species metrics).

    ``forward`` returns a tensor in the original feature layout (continuous
    columns + argmax-decoded PDG code + argmax charge) so reconstruction
    monitors and exports keep working; ``compute_loss`` uses the raw heads.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        continuous_indices: tuple[int, ...] = (0, 1, 2),
        pid_index: int | None = 3,
        charge_index: int | None = 4,
        pdg_ids: tuple[int, ...] = DEFAULT_PDG_IDS,
        hidden_dims: tuple[int, ...] = (128, 128, 64),
        kin_weight: float = 1.0,
        pid_weight: float = 1.0,
        charge_weight: float = 1.0,
        input_key: str = "csts",
        mask_key: str = "mask",
    ):
        super().__init__()
        self.continuous_indices = list(continuous_indices)
        self.pid_index = pid_index
        self.charge_index = charge_index
        self.kin_weight = kin_weight
        self.pid_weight = pid_weight
        self.charge_weight = charge_weight
        self.input_key = input_key
        self.mask_key = mask_key
        self.n_features = output_dim

        self.register_buffer("pdg_ids", torch.tensor(pdg_ids, dtype=torch.long))
        # Inverse lookup for forward(): class id -> PDG code (other -> 0).
        self.register_buffer(
            "pdg_values", torch.tensor([*pdg_ids, 0], dtype=torch.float32)
        )

        self.n_pid = len(pdg_ids) + 1
        head_dim = len(self.continuous_indices)
        if pid_index is not None:
            head_dim += self.n_pid
        if charge_index is not None:
            head_dim += 3
        self.coder = CoderModel(
            input_dim=input_dim, output_dim=head_dim, hidden_dims=list(hidden_dims)
        )

    def _split_heads(self, head: torch.Tensor):
        n_cont = len(self.continuous_indices)
        cont = head[..., :n_cont]
        offset = n_cont
        pid_logits = charge_logits = None
        if self.pid_index is not None:
            pid_logits = head[..., offset : offset + self.n_pid]
            offset += self.n_pid
        if self.charge_index is not None:
            charge_logits = head[..., offset : offset + 3]
        return cont, pid_logits, charge_logits

    def forward(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch.get(self.mask_key)
        head = self.coder(z_q, mask)
        cont, pid_logits, charge_logits = self._split_heads(head)

        out = torch.zeros((*head.shape[:-1], self.n_features), device=head.device)
        out[..., self.continuous_indices] = cont
        if pid_logits is not None:
            out[..., self.pid_index] = self.pdg_values[pid_logits.argmax(dim=-1)]
        if charge_logits is not None:
            out[..., self.charge_index] = (charge_logits.argmax(dim=-1) - 1).float()
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out

    def compute_loss(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        targets = batch[self.input_key]
        mask = batch.get(self.mask_key)
        if mask is None:
            mask = torch.ones(targets.shape[:-1], dtype=torch.bool, device=targets.device)

        head = self.coder(z_q, mask)
        cont, pid_logits, charge_logits = self._split_heads(head)

        loss = self.kin_weight * F.l1_loss(
            cont[mask], targets[mask][:, self.continuous_indices]
        )
        if pid_logits is not None:
            pid_target = map_pdg(targets[mask][:, self.pid_index], self.pdg_ids)
            loss = loss + self.pid_weight * F.cross_entropy(pid_logits[mask], pid_target)
        if charge_logits is not None:
            charge_target = (
                targets[mask][:, self.charge_index].round().long() + 1
            ).clamp(0, 2)
            loss = loss + self.charge_weight * F.cross_entropy(
                charge_logits[mask], charge_target
            )
        return loss
