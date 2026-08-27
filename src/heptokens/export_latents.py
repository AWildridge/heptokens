"""Export continuous per-constituent tokens — `python -m heptokens.export_latents`.

Continuous analog of ``export_tokens.py`` for ``LitContrastiveTokenizer``
checkpoints: the tokens are float latent vectors, so there is no codebook
table and no index array.

Usage (from an experiment repo with local configs):
    python -m heptokens.export_latents -cd configs \\
        ckpt_path=/path/to/tokenizer/best.ckpt \\
        output_dir=results/latents/model_name \\
        datamodule.data_path=/path/to/file.h5

Output npz contains:
    latents     : [N, num_elements, latent_dim]  float32/float16 (0 = masked)
    labels      : [N]  int8    (only if the dataset provides labels)
    eventNumber : [N]  int64   (only if present in the dataset)
    weights     : [N]  float32 (only if the dataset provides per-event weights)

Config extras: ``model_class`` selects the checkpoint class ("contrastive"
default, or "vae" for ``LitVae`` posterior-mean tokens); ``latent_dtype``
("fp32" default, or "fp16" to halve the release size).

A sidecar ``<stem>_meta.json`` records the interface contract (latent dim,
checkpoint path, dataset stem) so released token datasets are versioned.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from omegaconf import DictConfig

from heptokens.models.contrastive import LitContrastiveTokenizer

log = logging.getLogger(__name__)


class MemmapLatentWriter(BasePredictionWriter):
    """Write float latent predictions to memory-mapped files batch-by-batch."""

    def __init__(
        self,
        tmp_dir: str,
        total_jets: int,
        num_elements: int,
        latent_dim: int,
        latent_dtype=np.float32,
    ):
        super().__init__(write_interval="batch")
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.total_jets = total_jets
        self.latent_dtype = latent_dtype
        self.pos = 0

        self.latents_path = self.tmp_dir / "latents.npy"
        self.labels_path = self.tmp_dir / "labels.npy"
        self.events_path = self.tmp_dir / "eventNumber.npy"
        self.weights_path = self.tmp_dir / "weights.npy"

        self.latents_mmap = np.lib.format.open_memmap(
            str(self.latents_path),
            mode="w+",
            dtype=latent_dtype,
            shape=(total_jets, num_elements, latent_dim),
        )
        self.labels_mmap = np.lib.format.open_memmap(
            str(self.labels_path),
            mode="w+",
            dtype=np.int8,
            shape=(total_jets,),
        )
        self.events_mmap = np.lib.format.open_memmap(
            str(self.events_path),
            mode="w+",
            dtype=np.int64,
            shape=(total_jets,),
        )
        self.weights_mmap = np.lib.format.open_memmap(
            str(self.weights_path),
            mode="w+",
            dtype=np.float32,
            shape=(total_jets,),
        )
        self.has_labels = False
        self.has_events = False
        self.has_weights = False

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        latents = prediction["latents"].cpu().numpy().astype(self.latent_dtype)
        n = len(latents)

        self.latents_mmap[self.pos : self.pos + n] = latents

        if "labels" in prediction:
            self.has_labels = True
            self.labels_mmap[self.pos : self.pos + n] = (
                prediction["labels"].cpu().numpy().astype(np.int8)
            )

        if "eventNumber" in prediction:
            self.has_events = True
            self.events_mmap[self.pos : self.pos + n] = prediction["eventNumber"].cpu().numpy()

        if "weights" in prediction:
            self.has_weights = True
            self.weights_mmap[self.pos : self.pos + n] = (
                prediction["weights"].cpu().numpy().astype(np.float32)
            )

        self.pos += n

    def finalize(self) -> int:
        """Flush mmaps and return actual number of jets written."""
        del self.latents_mmap, self.labels_mmap, self.events_mmap, self.weights_mmap
        return self.pos


def export(cfg: DictConfig) -> None:
    """Export logic — importable by experiment repos with their own entrypoints."""
    log.info("Instantiating data module from config")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="predict")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_class_name = str(cfg.get("model_class", "contrastive"))
    if model_class_name == "vae":
        from heptokens.models.vae import LitVae

        model_cls = LitVae
    elif model_class_name == "contrastive":
        model_cls = LitContrastiveTokenizer
    else:
        raise ValueError(f"Unknown model_class: {model_class_name!r} (contrastive|vae)")
    log.info(
        f"Loading tokenizer checkpoint: {cfg.ckpt_path} "
        f"(class={model_class_name}, device={device})"
    )
    model = model_cls.load_from_checkpoint(cfg.ckpt_path, map_location=device, weights_only=False)
    model.eval()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_jets = len(datamodule.test_set)
    num_elements = getattr(datamodule.test_set, "num_elements", 40)
    latent_dim = model.latent_dim
    latent_dtype = {"fp32": np.float32, "fp16": np.float16}[str(cfg.get("latent_dtype", "fp32"))]

    tmp_base = os.environ.get("LOCAL_TMPDIR", os.environ.get("TMPDIR", None))
    if tmp_base is None or not Path(tmp_base).exists():
        tmp_base = str(output_dir)

    with tempfile.TemporaryDirectory(dir=tmp_base) as tmp_dir:
        writer = MemmapLatentWriter(
            tmp_dir, total_jets, num_elements, latent_dim, latent_dtype=latent_dtype
        )

        log.info(
            f"Exporting latents via trainer.predict() ({total_jets:,} jets, memmap to {tmp_dir})"
        )
        trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[writer], logger=False)
        trainer.predict(model, datamodule=datamodule, return_predictions=False)

        actual_jets = writer.finalize()
        log.info(f"Wrote {actual_jets:,} jets to memmap")

        latents = np.lib.format.open_memmap(str(writer.latents_path), mode="r")[:actual_jets]

        stem = Path(cfg.datamodule.data_path).stem
        out_path = output_dir / f"{stem}.npz"

        save_dict = dict(latents=latents)
        if writer.has_labels:
            labels = np.lib.format.open_memmap(str(writer.labels_path), mode="r")[:actual_jets]
            save_dict["labels"] = labels
        if writer.has_events:
            event_numbers = np.lib.format.open_memmap(str(writer.events_path), mode="r")[
                :actual_jets
            ]
            save_dict["eventNumber"] = event_numbers
        if writer.has_weights:
            weights = np.lib.format.open_memmap(str(writer.weights_path), mode="r")[:actual_jets]
            save_dict["weights"] = weights

        log.info(f"Exported {actual_jets:,} jets -> latents {latents.shape}")
        log.info(f"Saving compressed {out_path} ...")
        np.savez_compressed(out_path, **save_dict)
        log.info(f"Saved {out_path}: {out_path.stat().st_size / 1e9:.2f} GB")

        # Version the interface: released token datasets must be reproducible.
        meta = {
            "latent_dim": int(latent_dim),
            "num_elements": int(num_elements),
            "n_jets": int(actual_jets),
            "ckpt_path": str(cfg.ckpt_path),
            "data_path": str(cfg.datamodule.data_path),
            "model_class": model_class_name,
            "latent_dtype": np.dtype(latent_dtype).name,
            "has_weights": bool(writer.has_weights),
            "created": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = output_dir / f"{stem}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        log.info(f"Saved interface metadata to {meta_path}")


@hydra.main(version_base=None, config_path="pkg://heptokens.conf", config_name="tokenize.yaml")
def main(cfg: DictConfig) -> None:
    export(cfg)


if __name__ == "__main__":
    main()
