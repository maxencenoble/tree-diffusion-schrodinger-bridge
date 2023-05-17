import numpy as np
import torch
import sklearn
import scipy.linalg as ln
from numpy import linalg as LA
from torch.utils.data import TensorDataset


# This code is inspired by
# https://github.com/sbyebss/Scalable-Wasserstein-Barycenter/

def centered_mean_and_covariance(dim, seed):
    centered_mean = np.zeros(dim)
    random_cov = given_singular_spd_cov(dim, seed)
    return centered_mean, random_cov


def given_singular_spd_cov(dim, random_state=None, range_sing=[0.5, 5.]):
    generator = sklearn.utils.check_random_state(random_state)
    A = generator.rand(dim, dim)
    U, _, V = np.linalg.svd(np.dot(A.T, A))
    X = np.dot(np.dot(U, np.diag(range_sing[0] + generator.rand(dim) * (range_sing[1] - range_sing[0]))), V)
    return X


def data_distrib(npar, dim, seed):
    centered_mean, random_cov = centered_mean_and_covariance(dim, seed)
    return torch.from_numpy(np.random.multivariate_normal(centered_mean, random_cov, npar)), random_cov


def gaussian_ds(npar, dim, seed):
    init_sample, random_cov = data_distrib(npar, dim, seed)
    init_sample = init_sample.float()
    mean_per_dim = init_sample.mean(axis=0)
    var_per_dim = init_sample.var(axis=0)
    init_ds = TensorDataset(init_sample)
    return init_ds, mean_per_dim, var_per_dim, random_cov


def cov_ideal(list_cov, weights, dim):
    n_datasets = len(list_cov)
    Sn = np.asmatrix(np.eye(dim))
    # si represents the -1/2 power, the s means the 1/2 power and S represents S matrix itself.
    num_itr = 0
    while True:
        num_itr += 1
        s = ln.sqrtm(Sn)
        si = LA.inv(s)
        ans_medium = np.asmatrix(np.zeros_like(Sn))
        for i in range(n_datasets):
            ans_medium += weights[i] * ln.sqrtm(np.matmul(np.matmul(s, np.asmatrix(list_cov[i])), s))
        Sn_1 = np.matmul(ans_medium, ans_medium)
        Sn_1 = np.matmul(np.matmul(si, Sn_1), si)

        if np.power(Sn_1 - Sn, 2).sum() <= 1e-10:
            break
        Sn = Sn_1
    cov_ideal = Sn_1

    return cov_ideal
