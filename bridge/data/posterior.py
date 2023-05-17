import numpy as np
import torch
from torch.utils.data import TensorDataset


def data_distrib(data_root):
    return torch.from_numpy(np.load(data_root))


def posterior_ds(data_root):
    init_sample = data_distrib(data_root)
    init_sample = init_sample.float()
    mean_per_dim = init_sample.mean(axis=0)
    var_per_dim = init_sample.var(axis=0)
    init_ds = TensorDataset(init_sample)
    return init_ds, mean_per_dim, var_per_dim, init_sample.shape[0]
