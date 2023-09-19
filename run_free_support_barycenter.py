import ot
import os
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from bridge.data.two_dim import data_distrib as two_dim_distrib
from bridge.data.gaussian import data_distrib as gaussian_distrib
from bridge.data.posterior import data_distrib as posterior_distrib
from bridge.data.gaussian import cov_ideal
from bridge.data.utils_uvp import BW2_UVP, mean_cov_from_samples

parser = argparse.ArgumentParser(description='Compute Free Support Sinkhorn Barycenter.')
parser.add_argument('--data', type=str, help='2d, gaussian or posterior')

# OT PARAMETERS
OT_REG = 0.5
NPAR = 1500
NUM_ITER = 100

# GENERAL SETTING
REG = 1e-8
B_WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3]).astype(np.float32)

# 1. 2D SETTING
# OT_REG = 0.5, NPAR = 1500, NUM_ITER = 100

Datasets_2D = ['moon', 'circle', 'swiss']
SCALING_FACTOR = 7.
COMPUTE_exact_2d = True
OT_REG_convolutional_2d = 5e-4
COMPUTE_convolutional_2d = True

# 2. GAUSSIAN SETTING
# DIM = 2 : OT_REG = 0.1
# DIM = 16 : OT_REG = 0.2
# DIM = 64 : OT_REG = 0.5
# DIM = 128 : OT_REG = 1.0
# DIM = 256 : OT_REG = 2.0

DIM = 2
Datasets_G = [1, 2, 3]
# Datasets_G = [11, 22, 33]
# Datasets_G = [111, 222, 333]

# 3. POSTERIOR SETTING
# DATA_NAME_P = 'wine-homogeneous': OT_REG = 0.5
# DATA_NAME_P = 'wine-heterogeneous': OT_REG = 0.5

SEED = 1
Datasets_P = [1, 2, 3]  # DO NOT CHANGE
DATA_NAME_P = 'wine-homogeneous'
SEED_SUBSAMPLE = 0


def get_init(list_mean, list_inv_var):
    gaussian_mean = torch.mean(torch.stack(list_mean), dim=0)
    gaussian_var = 1 / torch.mean(torch.stack(list_inv_var), dim=0)
    return gaussian_mean + torch.sqrt(gaussian_var) * torch.randn((NPAR, *gaussian_mean.shape))


def main():
    args = parser.parse_args()
    list_mean, list_inv_var = [], []
    measures_locations, measures_weights = [], []

    print('---Loading or generating data---')
    if args.data == '2d':
        for i in range(3):
            X = two_dim_distrib(NPAR, Datasets_2D[i], SCALING_FACTOR)
            measures_locations.append(X)
            measures_weights.append(ot.unif(X.shape[0]))
            list_mean.append(X.mean(axis=0))
            list_inv_var.append(1 / (REG + X.var(axis=0)))

    list_cov = []
    if args.data == 'gaussian':
        for i in range(3):
            X, cov = gaussian_distrib(NPAR, DIM, Datasets_G[i])
            list_cov.append(cov)
            measures_locations.append(X)
            measures_weights.append(ot.unif(X.shape[0]))
            list_mean.append(X.mean(axis=0))
            list_inv_var.append(1 / (REG + X.var(axis=0)))
        ideal_cov = cov_ideal(list_cov, B_WEIGHTS, DIM)

    if args.data == 'posterior':
        for idx in Datasets_P:
            # load samples and subsample
            rng = np.random.default_rng(SEED_SUBSAMPLE + idx)
            data_root = os.path.join('./data/posterior', DATA_NAME_P, f'dataset_{idx}_seed_{SEED}.npy')
            X = posterior_distrib(data_root)
            id_subset = rng.choice(X.shape[0], NPAR, replace=False)
            X = X[id_subset, :]

            measures_locations.append(X)
            measures_weights.append(ot.unif(X.shape[0]))
            list_mean.append(X.mean(axis=0))
            list_inv_var.append(1 / (REG + X.var(axis=0)))

        full_data_root = os.path.join('./data/posterior', DATA_NAME_P, f'dataset_full_seed_{SEED}.npy')
        full_samples = posterior_distrib(full_data_root)
        rng = np.random.default_rng(SEED_SUBSAMPLE)
        id_subset = rng.choice(full_samples.shape[0], NPAR, replace=False)
        full_samples = full_samples[id_subset, :]
        ideal_mean, ideal_cov = mean_cov_from_samples(full_samples)

    print('---Initialisation---')
    X_init = get_init(list_mean, list_inv_var).detach().cpu().numpy()

    for i in range(3):
        measures_locations[i] = measures_locations[i].detach().cpu().numpy()

    if args.data == '2d' and COMPUTE_exact_2d:
        print('---Running Free Support Exact OT algorithm (2D)---')
        X = ot.lp.free_support_barycenter(measures_locations=measures_locations,
                                          measures_weights=measures_weights,
                                          X_init=X_init,
                                          weights=B_WEIGHTS,
                                          numItermax=NUM_ITER,
                                          verbose=True)
        lim = 2.5 * SCALING_FACTOR
        plt.clf()
        plt.title('Free Support exact barycenter in 2D')
        plt.hist2d(X[:, 0], X[:, 1], range=[[-lim, lim], [-lim, lim]], bins=100)
        plt.savefig('images/2d_free_support_exact_barycenter.png', bbox_inches='tight', transparent=True, dpi=200)

    if args.data == '2d' and COMPUTE_convolutional_2d:
        print('---Running Convolutional Sinkhorn algorithm (2D)---')
        lim = 2.5 * SCALING_FACTOR
        all_hist_list = []
        for i in range(3):
            X_i = measures_locations[i]
            hist_i = plt.hist2d(X_i[:, 1], X_i[:, 0], range=[[-lim, lim], [-lim, lim]], bins=[100,134])[0]
            hist_i = hist_i / np.sum(hist_i)
            all_hist_list.append(hist_i)
        A = np.array(all_hist_list)
        hist_barycenter = ot.bregman.convolutional_barycenter2d(A=A,
                                                                reg=OT_REG_convolutional_2d,
                                                                weights=B_WEIGHTS,
                                                                verbose=True,
                                                                warn=False)
        hist_barycenter = hist_barycenter[::-1]
        plt.clf()
        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(hist_barycenter)
        ax1.set_title('Convolutional Sinkhorn barycenter in 2D')
        fig.savefig('images/2d_convolutional_sinkhorn_barycenter.png', bbox_inches='tight', transparent=True, dpi=200)

    print('---Running Free Support Sinkhorn algorithm---')
    X = ot.bregman.free_support_sinkhorn_barycenter(measures_locations=measures_locations,
                                                    measures_weights=measures_weights,
                                                    X_init=X_init,
                                                    reg=OT_REG,
                                                    weights=B_WEIGHTS,
                                                    numItermax=NUM_ITER,
                                                    warn=False,
                                                    verbose=True)

    if args.data == '2d':
        lim = 2.5 * SCALING_FACTOR
        plt.clf()
        plt.title('Free Support Sinkhorn barycenter in 2D')
        plt.hist2d(X[:, 0], X[:, 1], range=[[-lim, lim], [-lim, lim]], bins=100)
        plt.savefig('images/2d_free_support_sinkhorn_barycenter.png', bbox_inches='tight', transparent=True, dpi=200)

    if args.data == 'gaussian':
        print('BW2_UVP criterion :{:.5f}'.format(BW2_UVP(X, np.zeros(DIM), ideal_cov)))

    if args.data == 'posterior':
        print('BW2_UVP criterion :{:.5f}'.format(BW2_UVP(X, ideal_mean, ideal_cov)))


if __name__ == '__main__':
    main()
