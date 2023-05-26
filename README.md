# Tree Diffusion Schr&ouml;dinger Bridge

This is the official code for the paper 'Tree-Based Diffusion Schr&ouml;dinger Bridge with Applications to Wasserstein Barycenters'

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
- Posterior aggregation: `python data_posterior.py --data wine --splitting hom/het`

2. Change the configuration files:

- `./config/config.yaml`: SDE and/or ODE to save plots, starting leaf seed
- `./config/dataset/`: specific settings for each dataset
- `./config/model/`: specific setting for each model (fully connected neural network: Basic or UNET)

The size of the cache dataset used to obtain samples in the training stage is given by : `cache_npar`
x `num_cache_batches` x `num_steps` x `SHAPE`,
where `SHAPE` is the shape of the samples. Make sure that the parameter `num_workers` fits on your machine.

3. Train Networks and save plots:

- 2d, 2 datasets - Bridge (CPU):  `python main.py dataset=2d_bridge model=Basic tree=Bridge`
- 2d, 3 datasets - Barycenter (CPU):  `python main.py dataset=2d_3datasets model=Basic tree=Barycenter`
- Gaussian, 3 datasets - Barycenter (CPU):  `python main.py dataset=gaussian_3datasets model=Basic tree=Barycenter`
- Posterior, 3 datasets - Barycenter (CPU):  `python main.py dataset=posterior_3datasets model=Basic tree=Barycenter`
- MNIST, 2 datasets - Barycenter (GPU):`python main.py dataset=stackedmnist_2datasets model=UNET tree=Barycenter`
- MNIST, 3 datasets - Barycenter (GPU):`python main.py dataset=stackedmnist_3datasets model=UNET tree=Barycenter`
- CELEBA, 2 datasets - Barycenter (GPU):`python main.py dataset=celeba_2datasets model=UNET tree=Barycenter`

Checkpoints and sampled images will be saved to a newly created directory named `experiments`.

If GPU has insufficient memory, then reduce the cache size.

4. Train Networks from a checkpoint:
- make sure that the pretrained models are saved according to the structure of the tree you are considering (ie, the
  directory of checkpoints for this experiment has local directories `source=...,dest=.../`, each one containing
  networks for the forward and the backward sampling directions that match `datasets`)
- set `checkpoint_run` to True in the dataset configuration file
- set `start_n_ipf` to the iteration corresponding to the saved models

5. Check the setting in `run_free_support_barycenter.py` and compare with the method from Cuturi and Doucet (2014):

- 2d, 3 datasets - Barycenter (CPU):  `python run_free_support_barycenter.py --data 2d`
- Gaussian, 3 datasets - Barycenter (CPU):  `python run_free_support_barycenter.py --data gaussian`
- Posterior, 3 datasets - Barycenter (CPU):  `python run_free_support_barycenter.py --data posterior`


