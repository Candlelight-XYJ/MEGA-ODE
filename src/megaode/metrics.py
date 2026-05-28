"""Metrics used by MEGA-ODE training and evaluation."""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error, r2_score

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(values: ArrayLike) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def wasserstein(A: ArrayLike, B: ArrayLike) -> float:
    return float(wasserstein_distance(_to_numpy(A).ravel(), _to_numpy(B).ravel()))


def kl_divergence(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_hat_prob = F.softmax(y_hat, dim=-1)
    y_prob = F.softmax(y, dim=-1)
    return F.kl_div(y_prob.log(), y_hat_prob, reduction="batchmean")


def cosine_similarity(A: ArrayLike, B: ArrayLike) -> float:
    a = _to_numpy(A).ravel()
    b = _to_numpy(B).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def pearson_correlation(A: ArrayLike, B: ArrayLike) -> float:
    a = _to_numpy(A).ravel()
    b = _to_numpy(B).ravel()
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def r2(A: ArrayLike, B: ArrayLike) -> float:
    return float(r2_score(_to_numpy(B).ravel(), _to_numpy(A).ravel()))


def rmse(A: ArrayLike, B: ArrayLike) -> float:
    return float(np.sqrt(mean_squared_error(_to_numpy(B).ravel(), _to_numpy(A).ravel())))


def pearson_corrcoef(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = y_pred - torch.mean(y_pred)
    y = y_true - torch.mean(y_true)
    denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y)) + eps
    return torch.sum(x * y) / denom


def corr_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return 1 - pearson_corrcoef(y_pred, y_true)


def combined_loss(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    mse = F.mse_loss(y_pred, y_true)
    return alpha * mse + (1 - alpha) * corr_loss(y_pred, y_true)


def evaluate_prediction(y_pred: ArrayLike, y_true: ArrayLike) -> Dict[str, float]:
    """Return common regression and distribution metrics."""
    return {
        "mse": float(mean_squared_error(_to_numpy(y_true).ravel(), _to_numpy(y_pred).ravel())),
        "rmse": rmse(y_pred, y_true),
        "r2": r2(y_pred, y_true),
        "pearson": pearson_correlation(y_pred, y_true),
        "cosine_similarity": cosine_similarity(y_pred, y_true),
        "wasserstein": wasserstein(y_pred, y_true),
    }
