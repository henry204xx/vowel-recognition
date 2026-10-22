"""Frame-independent full-covariance Gaussian vowel models."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class GaussianModel:
    mean: np.ndarray
    covariance: np.ndarray
    chol: np.ndarray
    logdet: float


def estimate_gaussian(frames: np.ndarray, regularization: float = 1e-3) -> GaussianModel:
    frames = np.asarray(frames, float)
    if frames.ndim != 2 or not len(frames): raise ValueError("Gaussian estimation needs frames")
    mean = frames.mean(axis=0)
    covariance = np.cov(frames, rowvar=False, bias=True) if len(frames) > 1 else np.zeros((frames.shape[1], frames.shape[1]))
    covariance = np.atleast_2d(covariance) + regularization * np.eye(frames.shape[1])
    chol = np.linalg.cholesky(covariance)
    return GaussianModel(mean, covariance, chol, float(2 * np.log(np.diag(chol)).sum()))


def negative_log_likelihood(frames: np.ndarray, model: GaussianModel) -> np.ndarray:
    frames = np.asarray(frames, float)
    centered = frames - model.mean
    solved = np.linalg.solve(model.chol, centered.T)
    return .5 * (frames.shape[1] * np.log(2 * np.pi) + model.logdet + np.sum(solved ** 2, axis=0))


def build_models(features_by_vowel: dict[str, list[np.ndarray]], regularization: float) -> dict[str, GaussianModel]:
    return {vowel: estimate_gaussian(np.vstack(frames), regularization) for vowel, frames in features_by_vowel.items()}
