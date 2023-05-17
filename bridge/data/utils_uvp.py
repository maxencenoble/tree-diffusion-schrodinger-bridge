import numpy as np
import torch
import scipy.linalg as ln
from numpy import linalg as LA


# This code is inspired by
# https://github.com/sbyebss/Scalable-Wasserstein-Barycenter/

def mean_cov_from_samples(b_samples):
    return mean_real(b_samples), cov_real(b_samples)


def cov_real(b_samples):
    if type(b_samples) is torch.Tensor:
        return np.cov(b_samples.detach().numpy().T)
    return np.cov(b_samples.T)


def mean_real(b_samples):
    if type(b_samples) is torch.Tensor:
        return np.mean(b_samples.detach().numpy(), 0)
    return np.mean(b_samples, 0)


def BW2_distance(mean_ideal, mean_ours, cov_ideal, cov_ours):
    under_squre = ln.sqrtm(cov_ours) @ cov_ideal @ ln.sqrtm(cov_ours)
    return 0.5 * LA.norm(mean_ideal - mean_ours) ** 2 + 0.5 * np.trace(cov_ideal) + 0.5 * np.trace(cov_ours) - np.trace(
        ln.sqrtm(under_squre))


def BW2_UVP(b_samples, mean_ideal, cov_ideal):
    b_samples = b_samples if isinstance(b_samples, np.ndarray) else b_samples.cpu()
    mean_ours, cov_ours = mean_cov_from_samples(b_samples)
    BW2_UVP = 100 * BW2_distance(mean_ideal, mean_ours, cov_ideal, cov_ours) * 2 / np.trace(cov_ideal)
    return BW2_UVP
