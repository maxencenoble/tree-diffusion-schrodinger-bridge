import torch
import os
import torch.nn.functional as F
import numpy as np
from ..langevin import Langevin
from torch.utils.data import DataLoader
from .config_getters import get_models, get_optimizers, get_dataset, get_plotter, get_logger, get_tree
import datetime
from tqdm import tqdm
from .ema import EMAHelper
from . import repeater
import random
from ..data import CacheLoader
from ..data.gaussian import cov_ideal
from ..data.utils_uvp import mean_cov_from_samples, BW2_UVP
from ..data.utils import add_module_state_dict
from accelerate import Accelerator
from torch.utils.data import TensorDataset
from collections import deque
import copy
import math

DATASET = 'Dataset'
DATASET_2D = '2d'
DATASET_GAUSSIAN = 'gaussian'
DATASET_POSTERIOR = 'posterior'
DATASET_CELEBA = 'celeba'
DATASET_STACKEDMNIST = 'stackedmnist'

TREE = 'Tree'
BARYCENTER_TREE = 'Barycenter'


class IPFTree(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset_tag = getattr(self.args, DATASET)
        if self.dataset_tag == DATASET_GAUSSIAN:
            self.barycenter_weights = copy.deepcopy(self.args.barycenter_weights)
        self.datasets = copy.deepcopy(self.args.datasets)
        self.nb_datasets = len(self.args.datasets)
        self.tree, self.n_vertices = self.tree_structure()
        self.shape = None

        # for Posterior and Gaussian setting
        self.mean_ideal = None
        self.cov_ideal = None

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator(fp16=False, cpu=args.device == 'cpu')
        self.device = self.accelerator.device  # torch.device(args.device)

        # build general models
        self.build_models()
        self.build_ema()
        self.save_init_model('f')
        self.save_init_model('b')

    def tree_structure(self):
        return get_tree(self.args)

    def build_models(self):
        """Builds 2 neural networks (net_f and net_b) to approximate the SDE drifts on the edges of the tree:
        - net_f is used for the forward direction w.r.t. to the orientation of the tree.
        - net_b is used for the backward direction w.r.t. to the orientation of the tree."""
        net_f, net_b = get_models(self.args)

        if self.args.dataparallel:
            net_f = torch.nn.DataParallel(net_f)
            net_b = torch.nn.DataParallel(net_b)

        net_f = net_f.to(self.device)
        net_b = net_b.to(self.device)
        self.net = torch.nn.ModuleDict({'f': net_f, 'b': net_b})

        del net_f
        del net_b
        self.clear()

    def update_ema(self, forward_or_backward):
        self.ema_helpers[forward_or_backward] = EMAHelper(mu=self.args.ema_rate, device=self.device)
        self.ema_helpers[forward_or_backward].register(self.net[forward_or_backward])

    def build_ema(self):
        """Builds 2 EMA instances (forward and backward) related to the models."""
        self.ema_helpers = {}
        self.sample_net = {}
        if self.args.ema:
            self.update_ema('f')
            self.update_ema('b')
            sample_net_f, sample_net_b = get_models(self.args)
            if self.args.dataparallel:
                sample_net_f = torch.nn.DataParallel(sample_net_f)
                sample_net_b = torch.nn.DataParallel(sample_net_b)
            sample_net_f = sample_net_f.to(self.device)
            sample_net_b = sample_net_b.to(self.device)

            self.sample_net = torch.nn.ModuleDict({'f': sample_net_f, 'b': sample_net_b})

            del sample_net_f
            del sample_net_b
            self.clear()

    def save_init_model(self, direction_to_save):
        """Saves the initial NN models as initialization for the edges."""
        dir = './'
        name_net = 'net' + '_' + direction_to_save + '_init.ckpt'
        name_net_ckpt = dir + name_net
        if self.args.dataparallel:
            torch.save(self.net[direction_to_save].module.state_dict(), name_net_ckpt)
        else:
            torch.save(self.net[direction_to_save].state_dict(), name_net_ckpt)

        if self.args.ema:
            name_sample_net = 'sample_net' + '_' + direction_to_save + '_init.ckpt'
            name_sample_net_ckpt = dir + name_sample_net
            sample_net = self.ema_helpers[direction_to_save].ema_copy(self.net[direction_to_save])
            if self.args.dataparallel:
                torch.save(sample_net.module.state_dict(), name_sample_net_ckpt)
            else:
                torch.save(sample_net.state_dict(), name_sample_net_ckpt)

            del sample_net
            self.clear()

    def populate_tree(self):
        """Builds IPF classes on the edges and fills the dataloaders for the original datasets."""
        for vertex, vertex_node in enumerate(self.tree.graph):
            for j in vertex_node.edges.keys():
                weight = self.tree.graph[vertex].edges[j].weight
                # get the ipf
                self.tree.graph[vertex].edges[j].ipf = IPFSequential(self.args,
                                                                     weight,
                                                                     self.tree.graph[vertex].vertex,
                                                                     self.tree.graph[j].vertex,
                                                                     self.net,
                                                                     self.sample_net,
                                                                     self.ema_helpers)

                # get the cache_dl, save_dl (not None iff we have a dataset)
                dic_v, dic_j = self.tree.graph[vertex].edges[j].ipf.build_dataloader_from_ipf()

                if dic_v:
                    self.tree.graph[vertex].vertex.cache_dl = dic_v['cache_dl']
                    self.tree.graph[vertex].vertex.save_dl_ode = dic_v['save_dl']
                    self.tree.graph[vertex].vertex.save_dl_sde = dic_v['save_dl']
                    self.tree.graph[vertex].vertex.mean = dic_v['mean']
                    self.tree.graph[vertex].vertex.var = dic_v['var']
                    self.tree.graph[vertex].vertex.nb_samples = dic_v['nb_samples']

                if dic_j:
                    self.tree.graph[j].vertex.cache_dl = dic_j['cache_dl']
                    self.tree.graph[j].vertex.save_dl_ode = dic_j['save_dl']
                    self.tree.graph[j].vertex.save_dl_sde = dic_j['save_dl']
                    self.tree.graph[j].vertex.mean = dic_j['mean']
                    self.tree.graph[j].vertex.var = dic_j['var']
                    self.tree.graph[j].vertex.nb_samples = dic_j['nb_samples']

                del dic_v
                del dic_j
                self.clear()
        tree_tag = getattr(self.args, TREE)
        if tree_tag == BARYCENTER_TREE:
            self.fill_mean_and_cov_ideal()

    def fill_mean_and_cov_ideal(self):
        """Assuming that we have a barycenter Tree, we compute:
        - the mean and the covariance of the full samples (posterior setting)
        - the mean and the covariance of the true non-reg barycenter (gaussian setting)."""
        if self.dataset_tag == DATASET_POSTERIOR:
            super_root = os.path.join(self.args.data_dir, 'posterior', self.args.name_data)
            data_path = f'dataset_full_seed_{self.args.seed_data}.npy'
            data_root = os.path.join(super_root, data_path)
            true_samples = np.load(data_root)
            self.mean_ideal, self.cov_ideal = mean_cov_from_samples(true_samples)

        elif self.dataset_tag == DATASET_GAUSSIAN:
            list_cov = []
            for data_label in self.datasets:
                cov_root = os.path.join(self.args.data_dir, 'gaussian',
                                        "dim_" + str(self.args.dim) + "_cov_" + str(data_label) + ".npy")
                cov = np.load(cov_root)
                list_cov.append(cov)
            cov = cov_ideal(list_cov, self.barycenter_weights, self.args.dim)
            self.mean_ideal, self.cov_ideal = np.zeros(self.args.dim), cov
            np.save('cov_ideal.npy', cov)

    def build_root_dataset(self, init_samples):
        """When the root is not an existing dataset (init_samples==None), we sample from a normal distribution with
        - its mean: mean of the means of the original data distributions
        - its variance: PARAM x (mean of the (variances)^-1 of the original data distributions)^-1
        - its nb of samples: max of the nbs of samples in the original data distributions
        where PARAM=self.args.init_var_rate.
        We initialize self.data_root, a tensor which will contain the data of the cacheloader of the root.
        We build a cacheloader for training and dataloaders for saving and plotting."""
        reg = 1e-8
        if init_samples is None:
            stack_mean, stack_inv_var, stack_nb_samples = [], [], []
            for vertex in self.tree.graph:
                if vertex.vertex.data is not None:
                    stack_mean.append(vertex.vertex.mean)
                    stack_inv_var.append(1 / (vertex.vertex.var + reg))
                    stack_nb_samples.append(vertex.vertex.nb_samples)
            gaussian_mean = torch.mean(torch.stack(stack_mean), dim=0)
            gaussian_var = 1 / torch.mean(torch.stack(stack_inv_var), dim=0)
            nb_samples = max(stack_nb_samples)

            init_samples = gaussian_mean + torch.sqrt(self.args.init_var_rate * gaussian_var) * torch.randn(
                (nb_samples, *gaussian_mean.shape))

        self.data_root = torch.zeros((0, *init_samples[0].shape)).to(self.device)

        if self.dataset_tag == DATASET_POSTERIOR or self.dataset_tag == DATASET_GAUSSIAN:
            print('BW2_UVP criterion (initialisation) :{:.5f}'.format(
                BW2_UVP(init_samples, self.mean_ideal, self.cov_ideal)))
            print('----------------------------------------------------------------------------------------')

        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id + self.accelerator.process_index)

        self.kwargs = {"num_workers": self.args.num_workers,
                       "pin_memory": self.args.pin_memory,
                       "worker_init_fn": worker_init_fn,
                       "drop_last": True}
        save_dl = DataLoader(
            TensorDataset(init_samples), batch_size=self.args.plot_npar, shuffle=True, **self.kwargs)
        cache_dl = DataLoader(
            TensorDataset(init_samples), batch_size=self.args.cache_npar, shuffle=True, **self.kwargs)
        (cache_dl, save_dl) = self.accelerator.prepare(cache_dl, save_dl)
        cache_dl = repeater(cache_dl)
        save_dl = repeater(save_dl)

        # for training
        self.tree.graph[0].vertex.cache_dl = cache_dl

        # for saving as a dataset (size of the cacheloader)
        self.tree.graph[0].vertex.cache_dl_to_save_sde = cache_dl
        self.tree.graph[0].vertex.cache_dl_to_save_ode = cache_dl

        # for plotting
        self.tree.graph[0].vertex.save_dl_ode = save_dl
        self.tree.graph[0].vertex.save_dl_sde = save_dl

        self.tree.graph[0].vertex.mean = torch.mean(init_samples, 0)
        self.tree.graph[0].vertex.var = torch.var(init_samples, 0)
        self.tree.graph[0].vertex.nb_samples = init_samples.size(dim=0)

        del init_samples
        del save_dl
        del cache_dl
        self.clear()

    def initial_forward_pass(self):
        """Initial forward pass on the tree."""
        queue = deque([0])
        while queue:
            vertex = queue.popleft()
            set_neighbors = self.tree.graph[vertex].edges.keys()
            for j in set_neighbors:
                queue.append(j)
                shape, dic = self.tree.graph[vertex].edges[j].ipf.build_dataloader_from_forward()
                if not self.shape:
                    self.shape = shape
                if not self.tree.graph[j].vertex.data:
                    self.tree.graph[j].vertex.save_dl_ode = dic['save_dl']
                    self.tree.graph[j].vertex.save_dl_sde = dic['save_dl']
                del dic
                self.clear()

    def save_data_root(self, epsilon, dynamics):
        """Saves in self.data_root the samples that are located in the ODE/SDE cacheloader of the root."""
        if dynamics == 'sde':
            root_cache_dl = self.tree.graph[0].vertex.cache_dl_to_save_sde
        elif dynamics == 'ode':
            root_cache_dl = self.tree.graph[0].vertex.cache_dl_to_save_ode

        len_dataloader = self.tree.graph[0].vertex.nb_samples // self.args.cache_npar
        for _ in range(len_dataloader):
            batch = next(root_cache_dl)[0]
            self.data_root = torch.cat((self.data_root, batch), dim=0)
            del batch
        torch.save(self.data_root.cpu(), 'data_root_epsilon={:.3f}_{}.pt'.format(epsilon, dynamics))

        # save BW2-UVP in case of Gaussian or Posterior setting
        if self.dataset_tag == DATASET_POSTERIOR or self.dataset_tag == DATASET_GAUSSIAN:
            print('BW2_UVP criterion with {}:{:.5f}'.format(dynamics, BW2_UVP(self.data_root.cpu(), self.mean_ideal,
                                                                              self.cov_ideal)))

        batch = next(self.tree.graph[0].vertex.cache_dl)[0]
        self.data_root = torch.zeros((0, *batch[0].shape)).to(self.device)
        del batch
        self.clear()

    def save_along_reversed_path(self, rev_path_queue, n, i, test_with_corrector=False, save_root=False):
        """Saves data on the vertices of the edges of rev_path_queue from SDE backward sampling (for plotting)."""
        rev_path_queue = deque(rev_path_queue)
        source_sample_index = rev_path_queue.popleft()
        while rev_path_queue:
            dest_sample_index = rev_path_queue.popleft()
            current_rev_edge = self.tree.graph[dest_sample_index].edges[source_sample_index]
            sample_direction = 'b' if current_rev_edge.direction == 'f' else 'f'
            current_rev_edge.ipf.ipf_save_with_sde(sample_direction, n, i, test_with_corrector)

            if not self.tree.graph[dest_sample_index].vertex.data:
                self.tree.graph[dest_sample_index].vertex.save_dl_sde = current_rev_edge.ipf.next_dic_sde['save_dl']
                # saving a full cache dl for the root after the training phase
                if dest_sample_index == 0 and save_root:
                    corrector = n > self.args.start_corrector if not test_with_corrector else True
                    init_cache_dl = current_rev_edge.ipf.destination_vertex.cache_dl
                    current_rev_edge.ipf.update_cacheloaders(init_cache_dl,
                                                             sample_direction,
                                                             True,
                                                             self.args.ema,
                                                             'sde',
                                                             corrector,
                                                             self.args.schedule_SDE,
                                                             self.args.coeff_schedule_SDE,
                                                             sample=True)
                    self.tree.graph[0].vertex.cache_dl_to_save_sde = current_rev_edge.ipf.next_vertex_dic['cache_dl']
                    del init_cache_dl
            del current_rev_edge
            self.clear()
            source_sample_index = dest_sample_index

    def save_along_forward_path(self, path_queue, n, i, test_with_corrector=False, save_root=False):
        """Saves data on the vertices of the edges of path_queue from ODE forward sampling (for plotting)."""
        path_queue = deque(path_queue)
        source_sample_index = path_queue.popleft()
        while path_queue:
            dest_sample_index = path_queue.popleft()
            current_edge = self.tree.graph[source_sample_index].edges[dest_sample_index]
            sample_direction = current_edge.direction
            current_edge.ipf.ipf_save_with_ode(sample_direction, n, i, test_with_corrector)
            if not self.tree.graph[dest_sample_index].vertex.data:
                self.tree.graph[dest_sample_index].vertex.save_dl_ode = current_edge.ipf.next_dic_ode['save_dl']
                # saving a full cache dl for the root after the training phase
                if dest_sample_index == 0 and save_root:
                    corrector = n > self.args.start_corrector if not test_with_corrector else True
                    init_cache_dl = current_edge.ipf.source_vertex.cache_dl
                    current_edge.ipf.update_cacheloaders(init_cache_dl,
                                                         sample_direction,
                                                         True,
                                                         self.args.ema,
                                                         'ode',
                                                         corrector,
                                                         self.args.schedule_ODE,
                                                         self.args.coeff_schedule_ODE,
                                                         sample=True)
                    self.tree.graph[0].vertex.cache_dl_to_save_ode = current_edge.ipf.next_vertex_dic['cache_dl']
                    del init_cache_dl
            del current_edge
            self.clear()
            source_sample_index = dest_sample_index

    def clear(self):
        torch.cuda.empty_cache()


class IPFTreeSequential(IPFTree):

    def __init__(self, args):
        super().__init__(args)
        self.n_ipf = self.args.n_ipf
        self.start_n_ipf = self.args.start_n_ipf

    def train_along_path(self, path, n):
        """Sequentially trains along the backward direction of the edges of path."""
        queue_vertices = deque(path)
        source_index = queue_vertices.popleft()
        while queue_vertices:
            dest_index = queue_vertices.popleft()
            print('Training the backward direction on the edge ' + str(source_index) + ' -> ' + str(dest_index) + '...')
            current_edge = self.tree.graph[source_index].edges[dest_index]
            forward_direction = current_edge.direction
            # forward_direction: direction of sampling for training
            backward_direction = 'b' if forward_direction == 'f' else 'f'
            # backward_direction: direction of training
            current_edge.ipf.ipf_train(forward_direction, backward_direction, n)
            print('Training the backward direction on the edge ' + str(source_index) + ' -> ' + str(
                dest_index) + ' DONE !')
            # updating the edge status after training
            if forward_direction == 'f':
                current_edge.ipf.flag_b = max(current_edge.ipf.flag_b + 1, n)
            else:
                current_edge.ipf.flag_f = max(current_edge.ipf.flag_f + 1, n)
            current_edge.ipf.save_model(backward_direction, n, self.args.num_iter)
            print('Saving the model: DONE !')
            self.tree.graph[dest_index].vertex.first_forward = True
            if not self.tree.graph[dest_index].vertex.data:
                self.tree.graph[dest_index].vertex.cache_dl = current_edge.ipf.next_vertex_dic['cache_dl']
                self.tree.graph[dest_index].vertex.nb_samples = current_edge.ipf.next_vertex_dic['nb_samples']
            del current_edge
            self.clear()
            source_index = dest_index

    def train_ipf_tree(self, epsilon, n_ipf, start_n_ipf=0, k_epsilon=1):
        """Performs n_ipf mIPF cycles, starting at start_n_ipf+1 (unless start_n_ipf==0), for a given epsilon."""
        # Prepare the IPF edges
        for vertex, vertex_node in enumerate(self.tree.graph):
            for j in vertex_node.edges.keys():
                print('Edge from {source} to {dest}'.format(source=vertex, dest=j))
                self.tree.graph[vertex].edges[j].ipf.set_time_horizon(epsilon, k_epsilon)
                print('*****************************************************************')
        print('\n\n')

        # Initial forward pass in the tree
        self.initial_forward_pass()
        self.tree.graph[0].vertex.first_forward = True

        # Shuffle the leaves
        leaves = list(self.tree.get_leaves())
        random.seed(self.args.starting_leaf_seed + start_n_ipf)
        random.shuffle(leaves)
        print('Current root: ', self.tree.get_root())
        print('Current order of the leaves: ', leaves)
        print('------------------------------------')

        # in the case where the root is not a leaf
        if not self.tree.graph[0].vertex.data:

            leaf_queue = deque(leaves)
            root_index = self.tree.get_root()
            next_leaf_index = leaf_queue.popleft()

            if start_n_ipf == 0:
                print('IPF ITERATION 0: the chosen root is not a dataset.')
                path = self.tree.find_path(root_index, next_leaf_index)
                reversed_path = path.copy()
                reversed_path.reverse()
                print('SDE: Sampling from {leaf} to {root} (backward, BEFORE training)'.format(root=root_index,
                                                                                               leaf=next_leaf_index))
                self.save_along_reversed_path(reversed_path, 0, 0)
                print('---------------------------------------------------------------------------------')
                print('Training from {root} to {leaf}...'.format(root=root_index, leaf=next_leaf_index))
                self.train_along_path(path, 0)
                print('Training from {root} to {leaf} DONE !'.format(root=root_index, leaf=next_leaf_index))
                print('---------------------------------------------------------------------------------')
                print('SDE: Sampling from {leaf} to {root} (backward, AFTER training)'.format(root=root_index,
                                                                                              leaf=next_leaf_index))
                self.save_along_reversed_path(reversed_path, 0, self.args.num_iter)
            self.tree.change_root(next_leaf_index)
            print('**********************************************************************************')

        # At this point, the root of the tree is also a leaf
        for n in range(start_n_ipf + 1, start_n_ipf + n_ipf + 1):
            # Shuffle the leaves
            leaves = list(self.tree.get_leaves())
            random.seed(self.args.starting_leaf_seed + 2 * n)
            random.shuffle(leaves)
            print('Current root: ', self.tree.get_root())
            print('Current order of the leaves: ', leaves)
            print('------------------------------------')

            print('IPF ITERATION: ' + str(n) + '/' + str(start_n_ipf + n_ipf))
            count_leaves = 0
            count_back_returns = 0
            leaf_queue = deque(leaves)
            while count_leaves < self.nb_datasets:
                current_leaf_index = self.tree.get_root()
                print('Current root:' + str(current_leaf_index))
                next_leaf_index = leaf_queue.popleft()
                print('Current leaf:' + str(next_leaf_index))
                path_between_leaves = self.tree.find_path(current_leaf_index, next_leaf_index)

                reversed_path = path_between_leaves.copy()
                reversed_path.reverse()

                if n % self.args.save_stride == 0 or n == start_n_ipf + n_ipf:
                    if self.args.plot_SDE:
                        print('SDE: Sampling from {leaf} to {root} (backward, BEFORE training)'.format(
                            root=current_leaf_index,
                            leaf=next_leaf_index))
                        self.save_along_reversed_path(reversed_path, n, 0)
                    print('---------------------------------------------------------------------------------')
                    if self.args.plot_ODE:
                        print('ODE: Sampling from {root} to {leaf} (forward, BEFORE training)'.format(
                            root=current_leaf_index, leaf=next_leaf_index))
                        self.save_along_forward_path(path_between_leaves, n, 0)
                    print('---------------------------------------------------------------------------------')

                print('Training from {root} to {leaf}...'.format(root=current_leaf_index, leaf=next_leaf_index))
                self.train_along_path(path_between_leaves, n)
                print('Training from {root} to {leaf} DONE !'.format(root=current_leaf_index, leaf=next_leaf_index))

                if n % self.args.save_stride == 0 or n == start_n_ipf + n_ipf:
                    print('---------------------------------------------------------------------------------')
                    if self.args.plot_SDE:
                        print('SDE: Sampling from {leaf} to {root} (backward, AFTER training)'.format(
                            root=current_leaf_index,
                            leaf=next_leaf_index))
                        self.save_along_reversed_path(reversed_path, n, self.args.num_iter, save_root=True)
                    print('---------------------------------------------------------------------------------')
                    if self.args.plot_ODE:
                        print('ODE: Sampling from {root} to {leaf} (forward, AFTER training)'.format(
                            root=current_leaf_index,
                            leaf=next_leaf_index))
                        self.save_along_forward_path(path_between_leaves, n, self.args.num_iter, save_root=True)

                print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                if count_back_returns < 2 * self.args.n_back_and_return:
                    leaf_queue.appendleft(current_leaf_index)
                    count_back_returns += 1
                else:
                    leaf_queue.append(current_leaf_index)
                    count_leaves += 1
                self.tree.change_root(next_leaf_index)

                if not self.tree.graph[0].vertex.data:
                    self.save_data_root(epsilon, 'sde')
                    self.save_data_root(epsilon, 'ode')
                    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

            all_leaves = [self.tree.get_root()] + list(self.tree.get_leaves())
            random.seed(self.args.starting_leaf_seed + 2 * n + 1)
            random.shuffle(all_leaves)
            self.tree.change_root(all_leaves[0])
            print('**********************************************************************************')

    def test_ipf_tree(self, epsilon, k_epsilon=1):
        """Tests the model along all edges for a given epsilon."""
        # Prepare the IPF edges
        for vertex, vertex_node in enumerate(self.tree.graph):
            for j in vertex_node.edges.keys():
                print('Edge from {source} to {dest}'.format(source=vertex, dest=j))
                self.tree.graph[vertex].edges[j].ipf.set_time_horizon(epsilon, k_epsilon)
                _, _ = self.tree.graph[vertex].edges[j].ipf.build_dataloader_from_forward()
                print('*****************************************************************')
        print('\n\n')

        leaves = list(self.tree.get_leaves())
        if not self.tree.graph[0].vertex.data:
            leaf_queue = deque(leaves)
            next_leaf_index = leaf_queue.popleft()
            self.tree.change_root(next_leaf_index)

        # At this point, the root of the tree is also a leaf
        leaves = list(self.tree.get_leaves())
        leaf_queue = deque(leaves)
        count_leaves = 0
        while count_leaves < self.nb_datasets:
            current_leaf_index = self.tree.get_root()
            next_leaf_index = leaf_queue.popleft()
            path_between_leaves = self.tree.find_path(current_leaf_index, next_leaf_index)

            reversed_path = path_between_leaves.copy()
            reversed_path.reverse()

            if self.args.plot_SDE:
                print('SDE: Sampling from {leaf} to {root}.'.format(root=current_leaf_index,
                                                                    leaf=next_leaf_index))
                self.save_along_reversed_path(reversed_path, self.args.start_n_ipf, 0,
                                              test_with_corrector=self.args.test_with_corrector,
                                              save_root=True)
            print('---------------------------------------------------------------------------------')
            if self.args.plot_ODE:
                print('ODE: Sampling from {root} to {leaf}.'.format(root=current_leaf_index,
                                                                    leaf=next_leaf_index))
                self.save_along_forward_path(path_between_leaves, self.args.start_n_ipf, 0,
                                             test_with_corrector=self.args.test_with_corrector,
                                             save_root=True)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            leaf_queue.append(current_leaf_index)
            count_leaves += 1
            self.tree.change_root(next_leaf_index)

            # for Gaussian and posterior settings
            if not self.tree.graph[0].vertex.data:
                self.save_data_root(epsilon, 'sde')
                self.save_data_root(epsilon, 'ode')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    def train_model(self):
        # Prepare the root dataset
        init_samples = None
        if self.args.data_root_load:
            init_samples = torch.load(self.args.data_root_path)
        if not self.tree.graph[0].vertex.data:
            self.build_root_dataset(init_samples)
        self.train_ipf_tree(epsilon=self.args.epsilon,
                            n_ipf=self.n_ipf,
                            start_n_ipf=self.start_n_ipf,
                            k_epsilon=1)

    def test_model(self):
        assert self.args.checkpoint_run, 'Specify checkpoint models.'
        # Prepare the root dataset
        if not self.tree.graph[0].vertex.data:
            self.build_root_dataset(None)
        self.test_ipf_tree(epsilon=self.args.epsilon,
                           k_epsilon=1)


class IPFBase(torch.nn.Module):

    def __init__(self, args, edge_weight, source_vertex, destination_vertex, net, sample_net, ema_helpers):
        super().__init__()
        self.args = args
        self.dataset_tag = getattr(self.args, DATASET)
        self.edge_weight = edge_weight
        self.source_vertex = source_vertex
        self.destination_vertex = destination_vertex

        self.accelerator = Accelerator(fp16=False, cpu=self.args.device == 'cpu')
        self.device = self.accelerator.device

        self.shape = None

        self.mean_per_dim = []
        self.var_per_dim = []
        self.nb_samples_total = []

        # training params
        self.num_steps = self.args.num_steps
        self.batch_size = self.args.batch_size
        self.num_iter = self.args.num_iter
        self.grad_clipping = self.args.grad_clipping
        self.lr = self.args.lr

        # get models
        self.net = net
        self.sample_net = sample_net
        self.ema_helpers = ema_helpers

        # initialization of the flags to update the models
        if self.args.checkpoint_run and self.args.start_n_ipf > 0:
            self.flag_f = self.args.start_n_ipf
            self.flag_b = self.args.start_n_ipf
        else:
            self.flag_f = -1
            self.flag_b = -1

        # get optims
        self.build_optimizers()

        # get loggers
        self.logger = self.get_logger()
        self.save_logger = self.get_logger('plot_logs')

        # Langevin
        gammas = torch.zeros(self.num_steps).to(self.device)
        self.langevin = Langevin(num_steps=self.num_steps,
                                 num_corrector_steps=self.args.num_corrector_steps,
                                 gammas=gammas,
                                 device=self.device,
                                 mean_match=self.args.mean_match,
                                 snr=self.args.snr)

        # Training datasets
        self.training_ds = None
        self.training_dl = None
        self.next_vertex_dic = None

        # Plotting
        self.plotter = self.get_plotter()
        self.next_dic_sde = None
        self.next_dic_ode = None

        if self.accelerator.process_index == 0:
            path = './source={source},dest={destination}'.format(
                source=self.source_vertex.idx, destination=self.destination_vertex.idx)
            if not os.path.exists(path):
                os.mkdir(path)
            path_im = './source={source},dest={destination}/im'.format(
                source=self.source_vertex.idx, destination=self.destination_vertex.idx)
            if not os.path.exists(path_im):
                os.mkdir(path_im)
            path_gif = './source={source},dest={destination}/gif'.format(source=self.source_vertex.idx,
                                                                         destination=self.destination_vertex.idx)
            if not os.path.exists(path_gif):
                os.mkdir(path_gif)
            path_checkpoints = './source={source},dest={destination}/checkpoints_model'.format(
                source=self.source_vertex.idx,
                destination=self.destination_vertex.idx)
            if not os.path.exists(path_checkpoints):
                os.mkdir(path_checkpoints)

        self.stride = self.args.gif_stride if self.plotter is not None else None
        self.stride_log = self.args.log_stride

    def get_logger(self, name='logs'):
        return get_logger(self, self.args, name)

    def get_plotter(self):
        return get_plotter(self, self.args)

    def accelerate(self, forward_or_backward):
        (self.net[forward_or_backward], self.optimizer[forward_or_backward]) = self.accelerator.prepare(
            self.net[forward_or_backward], self.optimizer[forward_or_backward])

    def build_optimizers(self):
        optimizer_f, optimizer_b = get_optimizers(self.net['f'], self.net['b'], self.lr)
        self.optimizer = {'f': optimizer_f, 'b': optimizer_b}
        del optimizer_f
        del optimizer_b
        self.clear()

    def build_dataloader(self, data_label=0, dataset=None, drop_last=True):
        if not dataset:
            dataset, mean_per_dim, var_per_dim, nb_samples_total = get_dataset(self.args, data_label)
        else:
            mean_per_dim = var_per_dim = None
            nb_samples_total = len(dataset)

        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[
                               1][0] + worker_id + self.accelerator.process_index)

        self.kwargs = {"num_workers": self.args.num_workers,
                       "pin_memory": self.args.pin_memory,
                       "worker_init_fn": worker_init_fn,
                       "drop_last": drop_last}
        save_dl = DataLoader(
            dataset, batch_size=self.args.plot_npar, shuffle=True, **self.kwargs)
        cache_dl = DataLoader(
            dataset, batch_size=self.args.cache_npar, shuffle=True, **self.kwargs)
        (cache_dl, save_dl) = self.accelerator.prepare(cache_dl, save_dl)
        cache_dl = repeater(cache_dl)
        save_dl = repeater(save_dl)
        dic = {'cache_dl': cache_dl,
               'save_dl': save_dl,
               'mean': mean_per_dim,
               'var': var_per_dim,
               'nb_samples': nb_samples_total}
        del save_dl
        del cache_dl
        self.clear()
        return dic

    def build_dataloader_from_ipf(self):
        """Used at the initialisation of the tree."""
        dic_source, dic_destination = None, None
        if self.source_vertex.data:
            dic_source = self.build_dataloader(data_label=self.source_vertex.data)
        if self.destination_vertex.data:
            dic_destination = self.build_dataloader(data_label=self.destination_vertex.data)
        return dic_source, dic_destination

    def build_dataloader_from_forward(self):
        """Used in the initial forward pass on the tree."""
        assert self.source_vertex.save_dl_ode, 'Source data not loaded for ODE'
        assert self.source_vertex.save_dl_sde, 'Source data not loaded for SDE'

        if self.accelerator.is_local_main_process:

            source_samples = next(self.source_vertex.save_dl_sde)[0]  # same for ODE
            shape = source_samples[0].shape
            if not self.shape:
                self.shape = shape
            source_samples = source_samples.to(self.device)
            x_tot, _, _ = self.langevin.record_init_langevin(source_samples, shape)
            shape_len = len(x_tot.shape)
            new_batch = copy.deepcopy(x_tot[:, -1, :]).cpu()
            x_tot_plot = x_tot.permute(1, 0, *list(range(2, shape_len))).detach()  # .cpu().numpy()

            if self.args.plot:
                self.plotter(source_samples, x_tot_plot, 0, 0, 'f', 'sde', self.epsilon, self.k_epsilon)

            del source_samples
            del x_tot_plot
            del x_tot

            # return the last iterate batch
            ds_init = TensorDataset(new_batch)
            del new_batch

            self.training_ds = CacheLoader(self.args.num_cache_batches,
                                           self.langevin,
                                           self.shape,
                                           batch_size=self.args.cache_npar,
                                           start_corrector=self.args.start_corrector,
                                           device=self.device,
                                           plot_cache_time=self.args.plot_cache_time)
            self.clear()

        return shape, self.build_dataloader(dataset=ds_init)

    def set_time_horizon(self, epsilon, k_epsilon):
        self.epsilon = epsilon
        self.k_epsilon = k_epsilon

        n = self.num_steps // 2
        self.T = self.epsilon / (4 * self.edge_weight)
        # we expect to have sum_k gamma_k = self.T

        gamma_min = (0.99 * self.T) / (2 * n)
        gamma_min = min(self.args.gamma_min, gamma_min)
        # if the user prefers to set a value for gamma_min

        reweighting = self.T / 2 - n * gamma_min
        # we are now sure that this is positive

        if self.args.gamma_space == 'linspace':
            gamma_half = np.linspace(0, 1, n)
        elif self.args.gamma_space == 'geomspace':
            gamma_half = np.geomspace(0, 1, n)

        gamma_half *= reweighting / np.sum(gamma_half)
        gamma_half += gamma_min
        print("Value for gamma_min: {:.6f}".format(gamma_min))
        print("Value for gamma_max: {:.6f}".format(gamma_half[-1]))

        gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        gammas = torch.tensor(gammas).to(self.device)
        # Result: sum gamma_k = self.T & gammas begins and finishes at gamma_min
        print('Theoretical value for T: {:.6f}'.format(2 * self.T))
        self.T = torch.sum(gammas)  # to avoid float errors
        print('Empirical value for T: {:.6f}'.format(2 * self.T))

        self.langevin.gammas = gammas.float()
        self.langevin.time = torch.cumsum(self.langevin.gammas, 0).to(self.langevin.device).float()

    def update_model_from_init(self, direction_to_load):
        """Given a direction (f or b), updates the model according to the initial tree model."""
        dir = './'
        name_net = 'net' + '_' + direction_to_load + '_init.ckpt'
        name_net_ckpt = dir + name_net
        state_dict = torch.load(name_net_ckpt, map_location=self.device)
        if self.args.dataparallel:
            state_dict = add_module_state_dict(state_dict)
        self.net[direction_to_load].load_state_dict(state_dict)

        del state_dict

        if self.args.ema:
            name_sample_net = 'sample_net' + '_' + direction_to_load + '_init.ckpt'
            name_sample_net_ckpt = edge_dir + name_sample_net
            state_dict = torch.load(name_sample_net_ckpt, map_location=self.device)
            if self.args.dataparallel:
                state_dict = add_module_state_dict(state_dict)
            self.sample_net[direction_to_load].load_state_dict(state_dict)
            self.ema_helpers[direction_to_load].register(self.sample_net[direction_to_load])

            del state_dict
            self.clear()

    def update_model_from_checkpoint(self, direction_to_load):
        """Given a direction (f or b), updates the model according to a checkpoint."""
        edge_dir = os.path.join(self.args.checkpoints_dir,
                                "source={source},dest={destination}".format(source=self.source_vertex.idx,
                                                                            destination=self.destination_vertex.idx))
        path = edge_dir + "/" + self.args.checkpoint_f if direction_to_load == 'f' else edge_dir + "/" + self.args.checkpoint_b
        state_dict = torch.load(path, map_location=self.device)
        if self.args.dataparallel:
            state_dict = add_module_state_dict(state_dict)
        self.net[direction_to_load].load_state_dict(state_dict)
        del state_dict

        if self.args.ema:
            sample_path = edge_dir + "/" + self.args.sample_checkpoint_f if direction_to_load == 'f' else edge_dir + "/" + self.args.sample_checkpoint_b
            state_dict = torch.load(sample_path, map_location=self.device)
            if self.args.dataparallel:
                state_dict = add_module_state_dict(state_dict)
            self.sample_net[direction_to_load].load_state_dict(state_dict)
            self.ema_helpers[direction_to_load].register(self.sample_net[direction_to_load])

            del state_dict
            self.clear()

    def update_model_from_previous_it(self, direction_to_load, n, i):
        """Given a direction (f or b), updates the model according to the previous iteration."""
        edge_dir = './source={source},dest={destination}/'.format(
            source=self.source_vertex.idx, destination=self.destination_vertex.idx)

        name_net = 'net' + '_' + direction_to_load + '_' + str(n) + "_" + str(i) + '.ckpt'
        name_net_ckpt = edge_dir + 'checkpoints_model/' + name_net
        state_dict = torch.load(name_net_ckpt, map_location=self.device)
        if self.args.dataparallel:
            state_dict = add_module_state_dict(state_dict)
        self.net[direction_to_load].load_state_dict(state_dict)
        del state_dict

        if self.args.ema:
            name_sample_net = 'sample_net' + '_' + direction_to_load + '_' + str(n) + "_" + str(i) + '.ckpt'
            name_sample_net_ckpt = edge_dir + 'checkpoints_model/' + name_sample_net
            state_dict = torch.load(name_sample_net_ckpt, map_location=self.device)
            if self.args.dataparallel:
                state_dict = add_module_state_dict(state_dict)
            self.sample_net[direction_to_load].load_state_dict(state_dict)
            self.ema_helpers[direction_to_load].register(self.sample_net[direction_to_load])

            del state_dict
            self.clear()

    def update_model(self, direction_to_load, n, i):
        """Given a direction, updates the model of the edge."""
        if self.args.checkpoint_run and self.args.start_n_ipf > 0 and n == self.args.start_n_ipf:
            # first time the edge is ever visited, with initialisation
            self.update_model_from_checkpoint(direction_to_load)
        elif self.args.checkpoint_run and self.args.start_n_ipf == 0 and n < 0:
            # first time the edge is ever visited, with initialisation
            self.update_model_from_checkpoint(direction_to_load)
        elif n == -1:
            # first time the edge is ever visited, but no given initialisation
            self.update_model_from_init(direction_to_load)
        else:
            self.update_model_from_previous_it(direction_to_load, n, i)

    def save_model(self, direction_to_save, n, i):
        edge_dir = './source={source},dest={destination}/'.format(
            source=self.source_vertex.idx, destination=self.destination_vertex.idx)

        name_net = 'net' + '_' + direction_to_save + '_' + str(n) + "_" + str(i) + '.ckpt'
        name_net_ckpt = edge_dir + 'checkpoints_model/' + name_net
        if self.args.dataparallel:
            torch.save(self.net[direction_to_save].module.state_dict(), name_net_ckpt)
        else:
            torch.save(self.net[direction_to_save].state_dict(), name_net_ckpt)

        if self.args.ema:
            name_sample_net = 'sample_net' + '_' + direction_to_save + '_' + str(n) + "_" + str(i) + '.ckpt'
            name_sample_net_ckpt = edge_dir + 'checkpoints_model/' + name_sample_net
            sample_net = self.ema_helpers[direction_to_save].ema_copy(self.net[direction_to_save])
            if self.args.dataparallel:
                torch.save(sample_net.module.state_dict(), name_sample_net_ckpt)
            else:
                torch.save(sample_net.state_dict(), name_sample_net_ckpt)

            del sample_net
            self.clear()

    def update_cacheloaders(self, init_cache_dl, sample_direction, first_pass_on_edge, use_ema, dynamics='sde',
                            corrector=False, schedule='zero', coeff_schedule=2, sample=False):
        """Returns the cache training dataloader and the cache dataloader of the next vertex to visit."""
        backward_direction = 'b' if sample_direction == 'f' else 'f'
        if use_ema:
            sample_net = self.ema_helpers[sample_direction].ema_copy(self.net[sample_direction])
            backward_net = self.ema_helpers[backward_direction].ema_copy(self.net[backward_direction])
        else:
            sample_net = self.net[sample_direction]
            backward_net = self.net[backward_direction]

        sample_net = self.accelerator.prepare(sample_net)
        backward_net = self.accelerator.prepare(backward_net)
        self.training_ds.update_data(sample_net, backward_net, init_cache_dl, first_pass_on_edge, dynamics,
                                     corrector, schedule, coeff_schedule, sample)
        del sample_net
        del backward_net

        next_cache_ds = TensorDataset(self.training_ds.next_data.cpu())
        self.next_vertex_dic = self.build_dataloader(dataset=next_cache_ds)
        del next_cache_ds

        self.training_dl = DataLoader(self.training_ds, batch_size=self.batch_size)
        self.training_dl = self.accelerator.prepare(self.training_dl)
        self.training_dl = repeater(self.training_dl)

        self.clear()

    def ipf_save_with_sde(self, sample_direction, n, i, test_with_corrector=False):
        """Saves the samples obtained on sample_direction by discretizing the SDE."""
        backward_direction = 'b' if sample_direction == 'f' else 'f'
        corrector = n > self.args.start_corrector if not test_with_corrector else True
        self.update_model('f', self.flag_f, self.num_iter)
        self.update_model('b', self.flag_b, self.num_iter)

        if self.accelerator.is_local_main_process:

            if self.args.ema:
                sample_net = self.ema_helpers[sample_direction].ema_copy(self.net[sample_direction])
                backward_net = self.ema_helpers[backward_direction].ema_copy(self.net[backward_direction])
            else:
                sample_net = self.net[sample_direction]
                backward_net = self.net[backward_direction]

            if sample_direction == 'b':
                save_dl_to_sample_from = self.destination_vertex.save_dl_sde
            else:
                save_dl_to_sample_from = self.source_vertex.save_dl_sde

            with torch.no_grad():
                self.set_seed(seed=0 + self.accelerator.process_index)
                batch = next(save_dl_to_sample_from)[0]
                batch = batch.to(self.device)
                shape = batch[0].shape
                x_tot, _, _ = self.langevin.record_langevin_seq(sample_net,
                                                                backward_net,
                                                                batch,
                                                                shape,
                                                                corrector,
                                                                self.args.schedule_SDE,
                                                                self.args.coeff_schedule_SDE,
                                                                sample=True)
                shape_len = len(x_tot.shape)
                new_batch = copy.deepcopy(x_tot[:, -1, :]).cpu()
                x_tot_plot = x_tot.permute(1, 0, *list(range(2, shape_len))).detach()  # .cpu().numpy()

            init_x = batch.detach().cpu().numpy()
            final_x = x_tot_plot[-1].detach().cpu().numpy()
            std_final = np.std(final_x)
            std_init = np.std(init_x)
            mean_final = np.mean(final_x)
            mean_init = np.mean(init_x)
            init_vertex = self.source_vertex.idx if sample_direction == 'f' else self.destination_vertex.idx
            fin_vertex = self.destination_vertex.idx if sample_direction == 'f' else self.source_vertex.idx
            print('Sampling on the training direction: ' + sample_direction + ' from {init} to {fin}'.format(
                init=init_vertex, fin=fin_vertex))
            print('Initial variance on vertex ' + str(init_vertex) + ' : ' + str(std_init ** 2))
            print('Final variance on vertex ' + str(fin_vertex) + ' : ' + str(std_final ** 2))

            self.save_logger.log_metrics({'sampling_direction': sample_direction,
                                          'init_var': std_init ** 2, 'final_var': std_final ** 2,
                                          'mean_init': mean_init, 'mean_final': mean_final,
                                          'T': self.T})

            if self.args.plot:
                self.plotter(batch, x_tot_plot, i, n, sample_direction, 'sde', self.epsilon, self.k_epsilon)

            del batch
            del x_tot_plot
            del x_tot
            self.clear()

            # get the last iterate batch
            self.next_dic_sde = self.build_dataloader(dataset=TensorDataset(new_batch))
            del new_batch

    def ipf_save_with_ode(self, sample_direction, n, i, test_with_corrector=False):
        """Saves the samples obtained on sample_direction by discretizing the ODE."""
        backward_direction = 'b' if sample_direction == 'f' else 'f'
        corrector = n > self.args.start_corrector if not test_with_corrector else True
        self.update_model('f', self.flag_f, self.num_iter)
        self.update_model('b', self.flag_b, self.num_iter)

        if self.accelerator.is_local_main_process:

            if self.args.ema:
                forward_net = self.ema_helpers[sample_direction].ema_copy(self.net[sample_direction])
                backward_net = self.ema_helpers[backward_direction].ema_copy(self.net[backward_direction])
            else:
                forward_net = self.net[sample_direction]
                backward_net = self.net[backward_direction]

            if sample_direction == 'b':
                save_dl_to_sample_from = self.destination_vertex.save_dl_ode
            else:
                save_dl_to_sample_from = self.source_vertex.save_dl_ode

            with torch.no_grad():
                self.set_seed(seed=0 + self.accelerator.process_index)
                batch = next(save_dl_to_sample_from)[0]
                batch = batch.to(self.device)
                shape = batch[0].shape
                x_tot, _, _ = self.langevin.record_ode_seq(forward_net,
                                                           backward_net,
                                                           batch,
                                                           shape,
                                                           corrector,
                                                           self.args.schedule_ODE,
                                                           self.args.coeff_schedule_ODE,
                                                           sample=True)
                shape_len = len(x_tot.shape)
                new_batch = copy.deepcopy(x_tot[:, -1, :]).cpu()
                x_tot_plot = x_tot.permute(1, 0, *list(range(2, shape_len))).detach()  # .cpu().numpy()

            init_x = batch.detach().cpu().numpy()
            final_x = x_tot_plot[-1].detach().cpu().numpy()
            std_final = np.std(final_x)
            std_init = np.std(init_x)
            mean_final = np.mean(final_x)
            mean_init = np.mean(init_x)
            init_vertex = self.source_vertex.idx if sample_direction == 'f' else self.destination_vertex.idx
            fin_vertex = self.destination_vertex.idx if sample_direction == 'f' else self.source_vertex.idx
            print('Sampling on the direction: ' + sample_direction + ' from {init} to {fin}'.format(
                init=init_vertex, fin=fin_vertex))
            print('Initial variance on vertex ' + str(init_vertex) + ' : ' + str(std_init ** 2))
            print('Final variance on vertex ' + str(fin_vertex) + ' : ' + str(std_final ** 2))

            self.save_logger.log_metrics({'sampling_direction': sample_direction,
                                          'init_var': std_init ** 2, 'final_var': std_final ** 2,
                                          'mean_init': mean_init, 'mean_final': mean_final,
                                          'T': self.T})

            if self.args.plot:
                self.plotter(batch, x_tot_plot, i, n, sample_direction, 'ode', self.epsilon, self.k_epsilon)

            del batch
            del x_tot_plot
            del x_tot
            self.clear()

            # get the last iterate batch
            self.next_dic_ode = self.build_dataloader(dataset=TensorDataset(new_batch))
            del new_batch

    def set_seed(self, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def clear(self):
        torch.cuda.empty_cache()


class IPFSequential(IPFBase):

    def __init__(self, args, edge_weight, source_vertex, destination_vertex, net, sample_net, ema_helpers):
        super().__init__(args, edge_weight, source_vertex, destination_vertex, net, sample_net, ema_helpers)

    def ipf_train(self, forward_direction, backward_direction, n):
        self.update_model('f', self.flag_f, self.num_iter)
        self.update_model('b', self.flag_b, self.num_iter)

        if forward_direction == 'f':
            init_cache_dl = self.source_vertex.cache_dl
            first_pass_on_edge = (n > 1) or self.destination_vertex.first_forward
        else:
            init_cache_dl = self.destination_vertex.cache_dl
            first_pass_on_edge = (n > 1) or self.source_vertex.first_forward
        self.update_cacheloaders(init_cache_dl,
                                 forward_direction,
                                 first_pass_on_edge,
                                 self.args.ema)

        self.build_optimizers()
        self.accelerate(backward_direction)

        for i in tqdm(range(self.num_iter + 1)):
            self.set_seed(seed=n * self.num_iter + i)

            x, out, steps_expanded = next(self.training_dl)
            x = x.to(self.device)
            out = out.to(self.device)
            steps_expanded = steps_expanded.to(self.device)
            eval_steps = self.T - steps_expanded

            if self.args.mean_match:
                pred = self.net[backward_direction](x, eval_steps) - x
            else:
                pred = self.net[backward_direction](x, eval_steps)

            loss = F.mse_loss(pred, out) + self.args.loss_weight * F.mse_loss(pred, torch.zeros_like(pred))

            # loss.backward()
            self.accelerator.backward(loss)

            if self.grad_clipping:
                clipping_param = self.args.grad_clip
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.net[backward_direction].parameters(), clipping_param)
            else:
                total_norm = 0.

            if (i % self.stride_log == 0) and (i > 0):
                self.logger.log_metrics({'training_direction': backward_direction,
                                         'loss': loss,
                                         'grad_norm': total_norm}, step=i + self.num_iter * n)

            self.optimizer[backward_direction].step()
            self.optimizer[backward_direction].zero_grad(set_to_none=True)
            self.net[backward_direction].zero_grad(set_to_none=True)
            del x
            del out
            del steps_expanded
            del pred
            del loss
            self.clear()

            if self.args.ema:
                self.ema_helpers[backward_direction].update(self.net[backward_direction])

            if (i % self.args.cache_refresh_stride == 0) and (i > 0):
                self.update_cacheloaders(init_cache_dl,
                                         forward_direction,
                                         first_pass_on_edge,
                                         self.args.ema)
                self.clear()

        del init_cache_dl
        self.clear()
