import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torchvision.utils as vutils
from PIL import Image
import seaborn as sns
import os

from ..data.stackedmnist import mnist_inv_transforms
from ..data.celeba import celeba_inv_transforms

DATASET_CELEBA = 'celeba'
DATASET_STACKEDMNIST = 'stackedmnist'

matplotlib.use('Agg')

DPI = 200


def make_gif(plot_paths, output_directory='./gif', gif_name='gif'):
    frames = [Image.open(fn) for fn in plot_paths]

    frames[0].save(os.path.join(output_directory, f'{gif_name}.gif'),
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100,
                   loop=0)


class ImPlotter(object):

    def __init__(self, source, destination, plot_level, dataset, plot_npar):
        path_edge = './source={source},dest={destination}'.format(source=source, destination=destination)
        im_dir = os.path.join(path_edge, 'im')
        gif_dir = os.path.join(path_edge, 'gif')

        if not os.path.isdir(path_edge):
            os.mkdir(path_edge)
        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)
        self.im_dir = im_dir
        self.gif_dir = gif_dir
        self.plot_npar = plot_npar
        self.plot_level = plot_level
        self.dataset = dataset

    def plot(self, initial_sample, x_tot_plot, i, n, forward_or_backward, dynamics_type, epsilon, k_eps):
        if self.plot_level > 0:
            x_tot_plot = x_tot_plot[:, :self.plot_npar]
            x_plot = x_tot_plot[-1]

            if self.dataset == DATASET_STACKEDMNIST:
                initial_sample = mnist_inv_transforms()(initial_sample)
                x_plot = mnist_inv_transforms()(x_plot)
            if self.dataset == DATASET_CELEBA:
                initial_sample = celeba_inv_transforms()(initial_sample)
                x_plot = celeba_inv_transforms()(x_plot)

            name = '{0}_{1}_{2}'.format(forward_or_backward, n, i)
            im_dir = os.path.join(self.im_dir, name)

            if not os.path.isdir(im_dir):
                os.mkdir(im_dir)

            torch.save(initial_sample, os.path.join(im_dir, 'im_grid_first_{}_tensor.pt'.format(dynamics_type)))
            torch.save(x_plot, os.path.join(im_dir, 'im_grid_final_{}_tensor.pt'.format(dynamics_type)))

            # check that the tensors are nearly between 0 and 1 before saving image
            if self.plot_level == 1:
                plt.clf()

                filename_grid_png = os.path.join(im_dir,
                                                 'im_grid_first_{}_epsilon={:.3f}_eps{}.png'.format(dynamics_type,
                                                                                                    epsilon, k_eps))
                vutils.save_image(initial_sample, filename_grid_png, nrow=10)
                filename_grid_png = os.path.join(im_dir,
                                                 'im_grid_final_{}_epsilon={:.3f}_eps{}.png'.format(dynamics_type,
                                                                                                    epsilon, k_eps))
                vutils.save_image(x_plot, filename_grid_png, nrow=10)

            if self.plot_level >= 2:
                plt.clf()
                plot_paths = []
                num_steps, num_particles, channels, H, W = x_tot_plot.shape
                plot_steps = np.linspace(0, num_steps - 1, self.plot_npar, dtype=int)

                for k in plot_steps:
                    # save png
                    filename_grid_png = os.path.join(im_dir, 'im_grid_{}_epsilon={:.3f}_eps{}.png'.format(k,
                                                                                                          dynamics_type,
                                                                                                          epsilon,
                                                                                                          k_eps))
                    plot_paths.append(filename_grid_png)
                    vutils.save_image(x_tot_plot[k], filename_grid_png, nrow=10)

                make_gif(plot_paths, output_directory=self.gif_dir, gif_name=name)

    def __call__(self, initial_sample, x_tot_plot, i, n, forward_or_backward, dynamics_type, epsilon, k_eps):
        self.plot(initial_sample, x_tot_plot, i, n, forward_or_backward, dynamics_type, epsilon, k_eps)


class TwoDPlotter(object):

    def __init__(self, num_steps, source, destination, scaling_factor, plot_freq, plot_particle, plot_trajectory,
                 plot_registration, plot_density, plot_density_smooth):
        path_edge = './source={source},dest={destination}'.format(source=source, destination=destination)
        im_dir = os.path.join(path_edge, 'im')
        gif_dir = os.path.join(path_edge, 'gif')

        if not os.path.isdir(path_edge):
            os.mkdir(path_edge)
        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)

        self.im_dir = im_dir
        self.gif_dir = gif_dir

        self.num_steps = num_steps
        self.source = source
        self.destination = destination
        self.scaling_factor = scaling_factor

        self.plot_freq = plot_freq
        self.plot_particle = plot_particle
        self.plot_trajectory = plot_trajectory
        self.plot_registration = plot_registration
        self.plot_density = plot_density
        self.plot_density_smooth = plot_density_smooth

    def plot(self, initial_sample, x_tot_plot, i, n, forward_or_backward, dynamics_type, epsilon, k_eps):
        fb = forward_or_backward
        ipf_it = n
        x_tot_plot = x_tot_plot.cpu().numpy()
        name = str(i) + '_' + fb + '_' + str(n) + '_' + dynamics_type + '_' + 'epsilon={:.3f}'.format(
            epsilon) + '_' + 'eps' + str(k_eps) + '_'
        if forward_or_backward == 'f':
            init = self.source
            fin = self.destination
        else:
            init = self.destination
            fin = self.source
        lim = 2.5 * self.scaling_factor
        self.save_sequence(x=x_tot_plot, name=name, xlim=(-lim, lim), ylim=(-lim, lim), ipf_it=ipf_it, init=init,
                           fin=fin)

    def save_sequence(self, x, name, xlim, ylim, ipf_it, init, fin):
        freq = self.plot_freq - 1
        if not os.path.isdir(self.im_dir):
            os.mkdir(self.im_dir)
        if not os.path.isdir(self.gif_dir):
            os.mkdir(self.gif_dir)

        # PARTICLES WITH INITIAL & FINAL DISTRIBUTIONS
        if self.plot_particle:
            plot_paths = []
            for k in range(self.num_steps):
                if k % freq == 0:
                    filename = name + 'particle_' + str(k) + '.png'
                    filename = os.path.join(self.im_dir, filename)
                    plt.clf()
                    if (xlim is not None) and (ylim is not None):
                        plt.xlim(*xlim)
                        plt.ylim(*ylim)
                    else:
                        xlim = [-15, 15]
                        ylim = [-15, 15]
                    plt.scatter(x[k, :, 0], x[k, :, 1], alpha=0.2, s=10, label='current')
                    plt.scatter(x[0, :, 0], x[0, :, 1], zorder=2, alpha=0.2, s=10, label='t=0')
                    plt.scatter(x[-1, :, 0], x[-1, :, 1], zorder=2, alpha=0.2, s=10, label='t=T')
                    plt.legend()
                    if ipf_it is not None:
                        str_title = 'IPFP iteration: ' + str(ipf_it) + ', from vertex ' + str(init) + ' to ' + str(fin)
                        plt.title(str_title)

                    # plt.axis('equal')
                    plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)
                    plot_paths.append(filename)
            make_gif(plot_paths, output_directory=self.gif_dir, gif_name=name)

        # TRAJECTORIES
        if self.plot_trajectory:
            N_part = 500
            filename = name + 'trajectory.png'
            filename = os.path.join(self.im_dir, filename)
            plt.clf()
            plt.plot(x[-1, :, 0], x[-1, :, 1], '*')
            plt.plot(x[0, :, 0], x[0, :, 1], '*')
            for j in range(N_part):
                xj = x[:, j, :]
                plt.plot(xj[:, 0], xj[:, 1], 'g', linewidth=2)
                plt.plot(xj[0, 0], xj[0, 1], 'rx')
                plt.plot(xj[-1, 0], xj[-1, 1], 'rx')
            plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)

        # REGISTRATION

        colors = np.cos(0.1 * x[0, :, 0]) * np.cos(0.1 * x[0, :, 1])

        if self.plot_registration:
            name_gif = name + 'registration'
            plot_paths_reg = []
            for k in range(self.num_steps):
                if k % freq == 0:
                    filename = name + 'registration_' + str(k) + '.png'
                    filename = os.path.join(self.im_dir, filename)
                    plt.clf()
                    if (xlim is not None) and (ylim is not None):
                        plt.xlim(*xlim)
                        plt.ylim(*ylim)
                    plt.plot(x[-1, :, 0], x[-1, :, 1], '*', alpha=0)
                    plt.plot(x[0, :, 0], x[0, :, 1], '*', alpha=0)
                    plt.scatter(x[k, :, 0], x[k, :, 1], c=colors)
                    if ipf_it is not None:
                        str_title = 'IPFP iteration: ' + str(ipf_it) + ', from vertex ' + str(init) + ' to ' + str(fin)
                        plt.title(str_title)
                    plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)
                    plot_paths_reg.append(filename)

            make_gif(plot_paths_reg, output_directory=self.gif_dir, gif_name=name_gif)

        # DENSITY

        if self.plot_density:
            name_gif = name + 'density'

            plot_paths_reg = []

            npts = 100
            for k in range(self.num_steps):
                if k % freq == 0:
                    filename = name + 'density_' + str(k) + '.png'
                    filename = os.path.join(self.im_dir, filename)
                    plt.clf()
                    if (xlim is not None) and (ylim is not None):
                        plt.xlim(*xlim)
                        plt.ylim(*ylim)
                    else:
                        xlim = [-15, 15]
                        ylim = [-15, 15]
                    if ipf_it is not None:
                        str_title = 'IPFP iteration: ' + str(ipf_it) + ', from vertex ' + str(init) + ' to ' + str(fin)
                        plt.title(str_title)
                    plt.hist2d(x[k, :, 0], x[k, :, 1], range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]], bins=npts)
                    plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)
                    plot_paths_reg.append(filename)
            make_gif(plot_paths_reg, output_directory=self.gif_dir, gif_name=name_gif)

        if self.plot_density_smooth:
            name_gif_smooth = name + 'density_smooth'
            plot_paths_reg_smooth = []
            for k in range(self.num_steps):
                if k % freq == 0:
                    filename_smooth = name + 'density_' + str(k) + '_smooth.png'
                    filename_smooth = os.path.join(self.im_dir, filename_smooth)
                    plt.clf()

                    if (xlim is not None) and (ylim is not None):
                        plt.xlim(*xlim)
                        plt.ylim(*ylim)
                    else:
                        xlim = [-15, 15]
                        ylim = [-15, 15]
                    if ipf_it is not None:
                        str_title = 'IPFP iteration: ' + str(ipf_it) + ', from vertex ' + str(init) + ' to ' + str(fin)
                        plt.title(str_title)

                    plot = sns.kdeplot(x=x[k, :, 0], y=x[k, :, 1], clip=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
                                       fill=True, thresh=0., bw_adjust=0.45, levels=250,
                                       cmap="viridis", cut=50)
                    fig = plot.get_figure()
                    fig.savefig(filename_smooth, bbox_inches='tight', transparent=True, dpi=DPI)
                    plot_paths_reg_smooth.append(filename_smooth)

                    make_gif(plot_paths_reg_smooth, output_directory=self.gif_dir, gif_name=name_gif_smooth)

    def __call__(self, initial_sample, x_tot_plot, i, n, forward_or_backward, dynamics_type, epsilon, k_eps):
        self.plot(initial_sample, x_tot_plot, i, n, forward_or_backward, dynamics_type, epsilon, k_eps)
