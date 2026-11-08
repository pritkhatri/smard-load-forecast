"""Small MLP in PyTorch on the same feature frame (optional dependency: torch)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES


class MLPModel:
    """Two-hidden-layer MLP. Inputs are standardised with training statistics; missing
    values (e.g. lag_24 for late target hours) are set to 0 after standardising and a
    missing-indicator column is added. Predicts y - lag_168 like the residual LightGBM."""

    def __init__(self, hidden=(128, 64), epochs=40, lr=1e-3, batch=512, seed=0):
        self.hidden, self.epochs, self.lr, self.batch, self.seed = hidden, epochs, lr, batch, seed
        self.features = list(FEATURES)
        self.net = None

    def _base(self, X):  # noqa: N803
        return X["lag_168"].fillna(X["same_hour_mean_4w"]).to_numpy(float)

    def _matrix(self, X, fit=False):  # noqa: N803
        a = X[self.features].to_numpy(float)
        if fit:
            self.mu = np.nanmean(a, axis=0)
            self.sd = np.nanstd(a, axis=0) + 1e-9
        miss = np.isnan(a)
        z = np.where(miss, 0.0, (a - self.mu) / self.sd)
        if fit:
            self.miss_cols = np.where(miss.any(axis=0))[0]
        return np.hstack([z, miss[:, self.miss_cols].astype(float)]).astype(np.float32)

    def fit(self, X: pd.DataFrame, y):  # noqa: N803
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        y = np.asarray(y, float) - self._base(X)
        ok = np.isfinite(y)
        X = X.loc[ok]  # noqa: N806
        y = y[ok]
        self.y_mu, self.y_sd = float(y.mean()), float(y.std() + 1e-9)
        xm = torch.tensor(self._matrix(X, fit=True))
        yt = torch.tensor(((y - self.y_mu) / self.y_sd).astype(np.float32))[:, None]
        d = xm.shape[1]
        layers, prev = [], d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net = nn.Sequential(*layers, nn.Linear(prev, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        g = torch.Generator().manual_seed(self.seed)
        n = len(xm)
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch):
                idx = perm[i : i + self.batch]
                opt.zero_grad()
                loss = nn.functional.l1_loss(self.net(xm[idx]), yt[idx])
                loss.backward()
                opt.step()
            sched.step()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        import torch

        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(self._matrix(X))).numpy()[:, 0]
        return out * self.y_sd + self.y_mu + self._base(X)
