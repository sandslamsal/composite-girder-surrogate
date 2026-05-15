"""Load a trained PINN checkpoint and run predictions on new sections.

Typical use:

    from src.models.inference import PINNPredictor
    pred = PINNPredictor.load("checkpoints/full_run/best.pt")
    out = pred.predict(df)        # df has the same columns as the parquet input

Returns a dataframe with predicted (y_na, curvature, moment, slip) in
physical units, optionally with MC-Dropout uncertainty bands.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.models.pinn import CompositeGirderPINN
from src.utils.normalize import FeatureNormalizer, TARGET_COLUMNS


class PINNPredictor:
    """Inference wrapper that bundles the trained PINN and its normaliser.

    Uses MC-Dropout for epistemic uncertainty: each forward pass with
    dropout active gives a slightly different prediction; the ensemble of
    ``T`` passes gives mean + std-dev per target.
    """

    def __init__(
        self,
        model: CompositeGirderPINN,
        normalizer: FeatureNormalizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str | Path,
             device: Optional[torch.device] = None) -> "PINNPredictor":
        if device is None:
            device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        normalizer = FeatureNormalizer()
        normalizer.load_state_dict(ckpt["normalizer_state"])
        cfg = ckpt.get("config", {})
        model_cfg = cfg.get("model", {})
        model = CompositeGirderPINN(
            input_dim=normalizer.input_dim,
            output_dim=normalizer.output_dim,
            width=model_cfg.get("width", 256),
            n_blocks=model_cfg.get("n_blocks", 5),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        return cls(model=model, normalizer=normalizer, device=device)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Single deterministic forward pass (dropout off)."""
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(
                self.normalizer.transform_features(df),
                dtype=torch.float32, device=self.device,
            )
            pred = self.model(X).cpu().numpy()
        mp = df["mp_estimate_kip_in"].to_numpy()
        phys = self.normalizer.inverse_transform_targets(pred, mp)
        return pd.DataFrame(phys, columns=TARGET_COLUMNS, index=df.index)

    def predict_with_uncertainty(
        self, df: pd.DataFrame, n_samples: int = 50
    ) -> pd.DataFrame:
        """MC-Dropout: run ``n_samples`` forward passes with dropout active,
        return mean and std-dev per target in physical units.

        Output dataframe has columns
        ``{target}_mean`` and ``{target}_std`` for each of the four targets.
        """
        self.model.train()  # enable dropout
        X = torch.tensor(
            self.normalizer.transform_features(df),
            dtype=torch.float32, device=self.device,
        )
        mp = df["mp_estimate_kip_in"].to_numpy()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X).cpu().numpy()
                samples.append(self.normalizer.inverse_transform_targets(pred, mp))
        self.model.eval()
        stacked = np.stack(samples, axis=0)        # (T, N, 4)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        out = {}
        for j, name in enumerate(TARGET_COLUMNS):
            out[f"{name}_mean"] = mean[:, j]
            out[f"{name}_std"] = std[:, j]
        return pd.DataFrame(out, index=df.index)
