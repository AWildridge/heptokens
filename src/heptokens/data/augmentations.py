"""Physics-motivated augmentations for contrastive per-constituent training.

All augmentations are *correspondence-preserving*: they never add, drop, split
or reorder constituents, so position ``[b, n]`` in the augmented tensor is the
same physical particle as in the original. This is required by two-view
objectives (VICReg, InfoNCE) that match embeddings elementwise.

Intended usage: apply on *physical-unit* features, before any quantile/standard
preprocessing transform. If applied after preprocessing, the smearing sigmas
act in the scaled space instead — legal, but calibrate accordingly.

NOTE: check the augmentation set against the coordinate choices. E.g. a global
azimuthal rotation is a no-op when ``phi`` is already stored relative to the
jet axis, so ``rotate_phi`` is off by default.
"""

import torch as T


def wrap_angle(x: T.Tensor) -> T.Tensor:
    """Wrap angles to the interval [-pi, pi)."""
    return T.remainder(x + T.pi, 2 * T.pi) - T.pi


class ConstituentAugmentation:
    """Stochastic augmentation of a constituent tensor ``[B, N, D]``.

    Args:
        pt_indices: Feature columns holding transverse momenta. Smeared
            multiplicatively with a log-normal factor ``exp(sigma * eps)`` so
            positivity is preserved (detector momentum resolution).
        eta_indices: Feature columns holding pseudorapidity-like angles.
            Smeared additively (angular resolution).
        phi_indices: Feature columns holding azimuthal angles. Smeared
            additively and optionally wrapped to [-pi, pi).
        categorical_indices: Columns that are categorical (PID, charge, ...).
            Never modified by any augmentation.
        pt_smear_sigma: Relative pT smearing width. 0 disables.
        angle_smear_sigma: Additive angular smearing width. 0 disables.
        noise_sigma: Additive Gaussian noise on all non-categorical columns
            (soft-noise / pileup proxy). 0 disables.
        rotate_phi: Apply one global random phi shift per jet (U(1) rotation).
            Only meaningful for absolute-coordinate phi columns.
        wrap_phi: Wrap phi columns back to [-pi, pi) after augmentation.
    """

    def __init__(
        self,
        pt_indices: tuple[int, ...] = (0,),
        eta_indices: tuple[int, ...] = (),
        phi_indices: tuple[int, ...] = (),
        categorical_indices: tuple[int, ...] = (),
        pt_smear_sigma: float = 0.05,
        angle_smear_sigma: float = 0.0,
        noise_sigma: float = 0.0,
        rotate_phi: bool = False,
        wrap_phi: bool = True,
    ) -> None:
        self.pt_indices = tuple(pt_indices)
        self.eta_indices = tuple(eta_indices)
        self.phi_indices = tuple(phi_indices)
        self.categorical_indices = tuple(categorical_indices)
        self.pt_smear_sigma = pt_smear_sigma
        self.angle_smear_sigma = angle_smear_sigma
        self.noise_sigma = noise_sigma
        self.rotate_phi = rotate_phi
        self.wrap_phi = wrap_phi

    def __call__(self, csts: T.Tensor, mask: T.Tensor | None = None) -> T.Tensor:
        out = csts.clone()

        if self.pt_smear_sigma > 0:
            for i in self.pt_indices:
                factor = T.exp(self.pt_smear_sigma * T.randn_like(out[..., i]))
                out[..., i] = out[..., i] * factor

        if self.angle_smear_sigma > 0:
            for i in (*self.eta_indices, *self.phi_indices):
                out[..., i] = out[..., i] + self.angle_smear_sigma * T.randn_like(out[..., i])

        if self.rotate_phi and self.phi_indices:
            # One shift per jet, broadcast over constituents.
            dphi = (T.rand(out.shape[0], 1, device=out.device) * 2 - 1) * T.pi
            for i in self.phi_indices:
                out[..., i] = out[..., i] + dphi

        if self.noise_sigma > 0:
            noise = self.noise_sigma * T.randn_like(out)
            out = out + noise

        if self.wrap_phi:
            for i in self.phi_indices:
                out[..., i] = wrap_angle(out[..., i])

        # Categorical columns are never augmented.
        for i in self.categorical_indices:
            out[..., i] = csts[..., i]

        # Padded positions keep their original (zeroed) values.
        if mask is not None:
            out = T.where(mask.unsqueeze(-1), out, csts)

        return out


class TwoViewTransform:
    """Batch transform adding two augmented views of the constituent tensor.

    Plugs into the datamodule ``transforms`` hook (``collate_and_transform``).
    Adds ``view_keys`` to the batch dict while leaving the original input
    untouched, so monitors and probes can still read the clean tensor.
    """

    def __init__(
        self,
        augmentation: ConstituentAugmentation,
        input_key: str = "csts",
        mask_key: str = "mask",
        view_keys: tuple[str, str] = ("csts_view0", "csts_view1"),
    ) -> None:
        self.augmentation = augmentation
        self.input_key = input_key
        self.mask_key = mask_key
        self.view_keys = tuple(view_keys)

    def __call__(self, batch: dict) -> dict:
        csts = batch[self.input_key]
        mask = batch.get(self.mask_key)
        batch[self.view_keys[0]] = self.augmentation(csts, mask)
        batch[self.view_keys[1]] = self.augmentation(csts, mask)
        return batch
