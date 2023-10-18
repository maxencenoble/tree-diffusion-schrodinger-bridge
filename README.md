# Tree-based Diffusion Schr&ouml;dinger Bridge

This is the official code for the paper 'Tree-Based Diffusion Schr&ouml;dinger Bridge with Applications to Wasserstein
Barycenters'. It extends the framework of [Diffusion Schr&ouml;dinger Bridge](https://arxiv.org/abs/2106.01357) [1] to
any
tree-structured joint distribution with known marginals on the leaves (thus including the classical Schr&ouml;dinger
Bridge problem). By considering star-shaped trees, it enables to compute regularized Wasserstein-2 barycenters for
high-dimensional empirical probability
distributions, which is of main interest in Optimal Transport (OT). Our method is competitive with respect to state-of-the-art *regularized*
algorithms from [2] and [3] in high-dimensional settings.

![drawing](images/drawing_tree.png)

In our setting, **each edge of the tree is parameterized by two neural networks**, which model the forward and backward
drifts of the diffusion processes. In theory, this requires to consider 2M neural networks, where M stands for the
number of edges in the tree. To avoid any memory issue in practice, *our code only requires to consider 2 active neural
networks*
at each stage of the training process. 

Illustration (2D)
------------

The following plots were obtained by considering the dataset 'Swiss Roll' as the first root in the training process. The
corresponding models are saved in the directory `./checkpoints_model`.

| Swiss Roll                                                                                                         | Circle                                                                                                              | Moons                                                                                                              | *Setting*                                                       |
|--------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| ![swiss_b](images/2d_3datasets_epsilon=0.1/source=0,dest=1/0_b_50_sde_epsilon=0.100_eps1_density_49_smooth.png)    | ![circle_f](images/2d_3datasets_epsilon=0.1/source=1,dest=2/0_f_50_sde_epsilon=0.100_eps1_density_49_smooth.png)    | ![moons_f](images/2d_3datasets_epsilon=0.1/source=1,dest=3/0_f_50_sde_epsilon=0.100_eps1_density_49_smooth.png)    | Estimation of the **leaves** (OT reg=0.1, 50 mIPF cycles).      |
| ![swiss_f_1](images/2d_3datasets_epsilon=0.05/source=0,dest=1/0_f_60_sde_epsilon=0.050_eps1_density_49_smooth.png) | ![circle_b_1](images/2d_3datasets_epsilon=0.05/source=1,dest=2/0_b_60_sde_epsilon=0.050_eps1_density_49_smooth.png) | ![moons_b_1](images/2d_3datasets_epsilon=0.05/source=1,dest=3/0_b_60_sde_epsilon=0.050_eps1_density_49_smooth.png) | Estimation of the **barycenter** (OT reg=0.05, 60 mIPF cycles). |
| ![swiss_f_2](images/2d_3datasets_epsilon=0.1/source=0,dest=1/0_f_50_sde_epsilon=0.100_eps1_density_49_smooth.png)  | ![circle_b_2](images/2d_3datasets_epsilon=0.1/source=1,dest=2/0_b_50_sde_epsilon=0.100_eps1_density_49_smooth.png)  | ![moons_b_2](images/2d_3datasets_epsilon=0.1/source=1,dest=3/0_b_50_sde_epsilon=0.100_eps1_density_49_smooth.png)  | Estimation of the **barycenter** (OT reg=0.1, 50 mIPF cycles).  |
| ![swiss_f_3](images/2d_3datasets_epsilon=0.2/source=0,dest=1/0_f_50_sde_epsilon=0.200_eps1_density_49_smooth.png)  | ![circle_b_3](images/2d_3datasets_epsilon=0.2/source=1,dest=2/0_b_50_sde_epsilon=0.200_eps1_density_49_smooth.png)  | ![moons_b_3](images/2d_3datasets_epsilon=0.2/source=1,dest=3/0_b_50_sde_epsilon=0.200_eps1_density_49_smooth.png)  | Estimation of the **barycenter** (OT reg=0.2, 50 mIPF cycles).  |

We provide below barycenter plots obtained by other methods.

| Free-support exact barycenter [2]                       | Free-support regularized barycenter [2]                     | Convolutional regularized barycenter [4]                    |
|---------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------|
| ![swiss_b](images/2d_free_support_exact_barycenter.png) | ![circle_f](images/2d_free_support_sinkhorn_barycenter.png) | ![moons_f](images/2d_convolutional_sinkhorn_barycenter.png) |

Contributors
------------

* Maxence Noble
* Valentin De Bortoli

Installation
------------

This project can be installed from its git repository.

1. Obtain the sources by:

   `git clone git@github.com:maxencenoble/tree-diffusion-schrodinger-bridge.git`

You may modify `requirements.txt` according to your CUDA version.

2. Install the packages via an Anaconda environment:

- `conda create -n tree_dsb python=3.8`
- `conda activate tree_dsb`
- `pip install -r requirements.txt`

How to use this code?
---------------------

For CELEBA, make sure that you already have in the path `./data/celeba` the 6 required files:

- `list_landmarks_align_celeba.txt`
- `list_eval_partition.txt`,
- `list_bbox_celeba.txt`,
- `list_attr_celeba.txt`,
- `img_align_celeba.zip`,
- `identity_CelebA.txt`.

You can find them
at https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg.

1. Download datasets:

- MNIST: `python data.py --data mnist`
- CELEBA: `python data.py --data celeba`
- Posterior aggregation (**already available**): `python data_posterior.py --data wine --splitting hom/het`

2. Change the configuration files:

- `./config/config.yaml`: SDE/ODE settings for plots, initialisation setting, corrector setting
- `./config/dataset/`: specific settings for each dataset (OT regularization, starting root, training parameters,
  checkpoints...)
- `./config/model/`: specific setting for each model (fully connected neural network 'Basic' or UNET)

The size of the cache dataset used to obtain samples in the training stage is given by : `cache_npar`
x `num_cache_batches` x `num_steps` x `SHAPE`,
where `SHAPE` is the shape of the samples. Make sure that the parameter `num_workers` fits on your machine.
If GPU has insufficient memory, then reduce the cache size.

3. Train models and save plots:

- 2d, 2 datasets - Bridge (CPU):  `python train_model.py dataset=2d_bridge model=Basic tree=Bridge`
- 2d, 3 datasets - Barycenter (CPU):  `python train_model.py dataset=2d_3datasets model=Basic tree=Barycenter`
- Gaussian, 3 datasets - Barycenter (CPU, no
  plot):  `python train_model.py dataset=gaussian_3datasets model=Basic tree=Barycenter`
- Posterior, 3 datasets - Barycenter (CPU, no
  plot):  `python train_model.py dataset=posterior_3datasets model=Basic tree=Barycenter`
- MNIST, 2 datasets - Barycenter (GPU):`python train_model.py dataset=stackedmnist_2datasets model=UNET tree=Barycenter`
- MNIST, 3 datasets - Barycenter (GPU):`python train_model.py dataset=stackedmnist_3datasets model=UNET tree=Barycenter`
- CELEBA, 2 datasets - Barycenter (GPU):`python train_model.py dataset=celeba_2datasets model=UNET tree=Barycenter`

Checkpoints and sampled images will be saved to a newly created directory named `experiments`.

4. Use checkpoint models:

- Make sure that the pretrained models are saved according to the structure of the tree you are considering (ie, the
  directory of checkpoints for this experiment has local directories `source=...,dest=.../`, each one containing
  networks for the forward and the backward sampling directions that match `datasets`).
- Set `checkpoint_run` to True in the dataset configuration file.

In this repository, there are 3 sets of pretrained models for the setting `2d_3datasets`, staring from the root `swiss`,
with equal barycenter weights,
each one corresponding to a certain level of OT regularization (`epsilon=0.2, 0.1, 0.05`). To use them, make sure that
you modify the following
parameters in the dataset configuration file:

- `epsilon`
- `checkpoints_dir`
- `checkpoints_f`, `checkpoint_b`

5. Train models from pretrained models:

- Follow Step 4.
- Set `start_n_ipf` to the mIPF cycle corresponding to the pretrained models.

Checkpoints and sampled images will be saved to a newly created directory named `experiments`.

6. Test pretrained models:

- Follow Step 4.
- 2d, 2 datasets - Bridge (CPU):  `python test_model.py dataset=2d_bridge model=Basic tree=Bridge`
- 2d, 3 datasets - Barycenter (CPU):  `python test_model.py dataset=2d_3datasets model=Basic tree=Barycenter`
- Gaussian, 3 datasets - Barycenter (CPU, no
  plot):  `python test_model.py dataset=gaussian_3datasets model=Basic tree=Barycenter`
- Posterior, 3 datasets - Barycenter (CPU, no
  plot):  `python test_model.py dataset=posterior_3datasets model=Basic tree=Barycenter`
- MNIST, 2 datasets - Barycenter (GPU):`python test_model.py dataset=stackedmnist_2datasets model=UNET tree=Barycenter`
- MNIST, 3 datasets - Barycenter (GPU):`python test_model.py dataset=stackedmnist_3datasets model=UNET tree=Barycenter`
- CELEBA, 2 datasets - Barycenter (GPU):`python test_model.py dataset=celeba_2datasets model=UNET tree=Barycenter`

Checkpoints and sampled images will be saved to a newly created directory named `experiments`.

7. Check the setting in `run_free_support_barycenter.py` and compare with the method from [2]:

- 2d, 3 datasets - Barycenter (CPU):  `python run_free_support_barycenter.py --data 2d`
- Gaussian, 3 datasets - Barycenter (CPU, no plot):  `python run_free_support_barycenter.py --data gaussian`
- Posterior, 3 datasets - Barycenter (CPU, no plot):  `python run_free_support_barycenter.py --data posterior`

References
------------

[1] V. De Bortoli, J. Thornton, J. Heng & A. Doucet, *Diffusion Schrödinger bridge with applications
to score-based generative modeling*, Advances in Neural Information Processing Systems, 2021.

[2] M. Cuturi & A. Doucet, *Fast computation of Wasserstein barycenters*, International conference on machine
learning, 2014.

[3] L. Li, A. Genevay, M. Yurochkin & J. Solomon, *Continuous regularized Wasserstein
barycenters*, Advances in Neural Information Processing Systems, 2020.

[4] J. Solomon, F. De Goes, G. Peyré, M. Cuturi, A. Butscher, A. Nguyen, & L. Guibas *Convolutional wasserstein
distances: Efficient optimal transportation on geometric domains*, ACM Transactions on Graphics, 2015.


