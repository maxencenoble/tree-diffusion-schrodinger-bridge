import os
import torch
import numpy as np
import torchvision.datasets

from ..models import *
from ..data.two_dim import two_dim_ds
from ..data.gaussian import gaussian_ds
from ..data.posterior import posterior_ds
from ..data.stackedmnist import Stacked_MNIST, mnist_transforms
from ..data.celeba import CelebA, celeba_transforms

from ..trees.class_tree import BridgeTree, BarycenterTree

from .plotters import TwoDPlotter, ImPlotter
from .logger import CSVLogger, Logger

# Tree
# --------------------------------------------------------------------------------


TREE = 'Tree'
BARYCENTER_TREE = 'Barycenter'
BRIDGE_TREE = 'Bridge'


def get_tree(args):
    tree_tag = getattr(args, TREE)
    try:
        root_index_in_data = args.datasets.index(args.data_root)
    except ValueError:
        root_index_in_data = None

    if tree_tag == BRIDGE_TREE:
        assert len(
            args.datasets) == 2, 'More than 2 datasets, cannot build a bridge tree.'

        n_vertices = 2

        kwargs = {
            "n_vertices": n_vertices,
            "datasets": args.datasets,
            "root_idx": root_index_in_data,
            "weight": args.edge_weight
        }

        tree = BridgeTree(**kwargs)

    if tree_tag == BARYCENTER_TREE:
        assert len(args.barycenter_weights) == len(
            args.datasets), 'Incompatible number of edges and vertices, cannot build a barycenter tree.'

        n_vertices = len(args.datasets) + 1

        kwargs = {
            "n_vertices": n_vertices,
            "datasets": args.datasets,
            "barycenter_weights": args.barycenter_weights,
            "root_idx": root_index_in_data
        }

        tree = BarycenterTree(**kwargs)

    return tree, n_vertices


# Dataset
# --------------------------------------------------------------------------------

DATASET = 'Dataset'
DATASET_2D = '2d'
DATASET_GAUSSIAN = 'gaussian'
DATASET_POSTERIOR = 'posterior'
DATASET_CELEBA = 'celeba'
DATASET_STACKEDMNIST = 'stackedmnist'


def get_dataset(args, data_label):
    dataset_tag = getattr(args, DATASET)

    # 2D DATASET

    if dataset_tag == DATASET_2D:
        data_tag = data_label
        npar = max(args.npar, args.cache_npar)
        ds, mean_per_dim, var_per_dim = two_dim_ds(npar, data_tag, args.scaling_factor)
        nb_samples_total = npar

    # GAUSSIAN DATASET

    if dataset_tag == DATASET_GAUSSIAN:
        super_root = os.path.join(args.data_dir, 'gaussian')
        if not os.path.isdir(super_root):
            os.mkdir(super_root)
        cov_root = os.path.join(super_root, "dim_" + str(args.dim) + "_cov_" + str(data_label) + ".npy")
        npar = max(args.npar, args.cache_npar)
        ds, mean_per_dim, var_per_dim, cov = gaussian_ds(npar, int(args.dim), int(data_label))
        nb_samples_total = npar
        # saving the covariance matrix
        np.save(cov_root, cov)

    # POSTERIOR DATASET

    if dataset_tag == DATASET_POSTERIOR:
        super_root = os.path.join(args.data_dir, 'posterior', args.name_data)
        data_path = f'dataset_{data_label}_seed_{args.seed_data}.npy'
        root = os.path.join(super_root, data_path)
        ds, mean_per_dim, var_per_dim, nb_samples_total = posterior_ds(root)

    # MNIST DATASET

    if dataset_tag == DATASET_STACKEDMNIST:
        super_root = os.path.join(args.data_dir, 'mnist')
        source_data = torchvision.datasets.MNIST(super_root,
                                                 train=True,
                                                 transform=mnist_transforms(args.data.image_size),
                                                 download=False)

        root = os.path.join(super_root, data_label)
        ds = Stacked_MNIST(source_data=source_data,
                           root=root,
                           load=args.load_data,
                           label=data_label,
                           imageSize=args.data.image_size,
                           num_channels=args.data.channels
                           )
        mean_per_dim = ds.mean_per_dim
        var_per_dim = ds.var_per_dim
        nb_samples_total = ds.nb_samples_total

    # CELEBA DATASET

    if dataset_tag == DATASET_CELEBA:
        root = os.path.join(args.data_dir, 'celeba')
        ds = CelebA(root,
                    label=data_label,
                    split='train',
                    transform=celeba_transforms(args.data.image_size, args.data.random_flip),
                    download=False)

        # choice coherent with celeba_transforms
        mean_per_dim = 0.5 * torch.ones([args.data.channels, args.data.image_size, args.data.image_size])
        var_per_dim = 0.5 * torch.ones([args.data.channels, args.data.image_size, args.data.image_size])
        nb_samples_total = len(ds.filename)

    return ds, mean_per_dim, var_per_dim, nb_samples_total


# Model
# --------------------------------------------------------------------------------

MODEL = 'Model'
BASIC_MODEL = 'Basic'
UNET_MODEL = 'UNET'


def get_models(args):
    model_tag = getattr(args, MODEL)

    if model_tag == BASIC_MODEL:
        kwargs = {
            "x_dim": args.dim,
            "time_emb_dim": args.model.time_emb_dim,
            "encoder_layers": [args.model.encoder_dim_1, args.model.encoder_dim_2],
            "decoder_layers": [args.model.decoder_dim_1, args.model.decoder_dim_2]
        }

        net_f, net_b = ScoreNetwork(**kwargs), ScoreNetwork(**kwargs)

    if model_tag == UNET_MODEL:
        image_size = args.data.image_size

        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        elif image_size == 28:
            channel_mult = (1, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")

        attention_ds = []
        for res in args.model.attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        kwargs = {
            "in_channels": args.data.channels,
            "model_channels": args.model.num_channels,
            "out_channels": args.data.channels,
            "num_res_blocks": args.model.num_res_blocks,
            "attention_resolutions": tuple(attention_ds),
            "dropout": args.model.dropout,
            "channel_mult": channel_mult,
            "num_classes": None,
            "use_checkpoint": args.model.use_checkpoint,
            "num_heads": args.model.num_heads,
            "num_heads_upsample": args.model.num_heads_upsample,
            "use_scale_shift_norm": args.model.use_scale_shift_norm
        }

        net_f, net_b = UNetModel(**kwargs), UNetModel(**kwargs)

    return net_f, net_b


# Optimizer
# --------------------------------------------------------------------------------
def get_optimizers(net_f, net_b, lr):
    return torch.optim.Adam(net_f.parameters(), lr=lr), torch.optim.Adam(net_b.parameters(), lr=lr)


# Logger
# --------------------------------------------------------------------------------
LOGGER = 'LOGGER'
LOGGER_PARAMS = 'LOGGER_PARAMS'

CSV_TAG = 'CSV'
NOLOG_TAG = 'NONE'


def get_logger(ipf, args, name):
    logger_tag = getattr(args, LOGGER)

    if logger_tag == CSV_TAG:
        kwargs = {'source': ipf.source_vertex.idx,
                  'destination': ipf.destination_vertex.idx,
                  'directory': args.CSV_log_dir,
                  'name': name}
        return CSVLogger(**kwargs)

    if logger_tag == NOLOG_TAG:
        return Logger()


def get_plotter(ipf, args):
    dataset_tag = getattr(args, DATASET)
    if dataset_tag == DATASET_GAUSSIAN or dataset_tag == DATASET_POSTERIOR:
        return None
    elif dataset_tag == DATASET_2D:
        return TwoDPlotter(num_steps=ipf.num_steps,
                           source=ipf.source_vertex.idx,
                           destination=ipf.destination_vertex.idx,
                           scaling_factor=args.scaling_factor,
                           plot_freq=args.plot_freq,
                           plot_particle=args.plot_particle,
                           plot_trajectory=args.plot_trajectory,
                           plot_registration=args.plot_registration,
                           plot_density=args.plot_density,
                           plot_density_smooth=args.plot_density_smooth)
    else:
        return ImPlotter(plot_level=args.plot_level,
                         source=ipf.source_vertex.idx,
                         destination=ipf.destination_vertex.idx,
                         dataset=dataset_tag,
                         plot_npar=args.plot_npar)
