import argparse

from posterior_utils.utils.toy_tools_data import *
from posterior_utils.utils.toy_tools_func import *

DATASETS_CLASSIFICATION = ['wine', 'breast-cancer', 'titanic', 'mushroom', 'adult', 'covertype']

parser = argparse.ArgumentParser(description='Compute samples from posterior distributions for Logistic Regression.')
parser.add_argument('--data', type=str, default='wine', help='see DATASETS_CLASSIFICATION')
parser.add_argument('--splitting', type=str, default='hom', help='het for heterogeneous, hom for homogeneous')
parser.add_argument('--seed', type=int, default=1, help='seed to generate data')

# MCMC PARAMETERS
NB_DATASETS = 3
MC_ITER = 5500000
THINNING = 500
BURN_IN_RATIO = 0.1

print('Number of samples: ', int((1 - BURN_IN_RATIO) * MC_ITER / THINNING))

# Saving
SAMPLES_PATH = './data/posterior/'
NLAGS = 500

# Randomness
RANDOM_STATE = 42

# Regularization of the model
L2 = 1.0
TEMPERATURE = 1.0
REG_MINSKER = float(NB_DATASETS)  # only for subdatasets

# Algorithm settings
MAX_DATA_SIZE = 20000


def main():
    args = parser.parse_args()
    assert args.data in DATASETS_CLASSIFICATION, 'Choose another dataset.'
    assert args.splitting in ['het', 'hom'], 'Splitting of data not recognized.'

    path_h = '-heterogeneous' if args.splitting == 'het' else '-homogeneous'
    save_path = SAMPLES_PATH + args.data + path_h + '/'
    os.makedirs(save_path, exist_ok=True)

    SEED = int(args.seed)
    RNG = np.random.default_rng(SEED)

    print('--- Preparing the datasets ---')
    datasets_t, datasets, oe, Xtrain, Xtest, Ytrain, Ytrain_oe, Ytest, Ytest_oe, output_dim = prepare_data(
        args.data,
        nb_datasets=NB_DATASETS,
        max_data_size=MAX_DATA_SIZE,
        heterogeneity=args.splitting,
        path_data=SAMPLES_PATH,
        rng=RNG,
        random_state=RANDOM_STATE)

    print('--- Preparing the models ---')
    logistic_full, lr_full, param_best, all_sub_logistic, all_lr, all_param_best = prepare_models(
        datasets_t, datasets, oe, Xtrain, Xtest, Ytrain,
        Ytrain_oe, Ytest,
        output_dim,
        temperature=TEMPERATURE,
        l2=L2,
        reg_minsker=REG_MINSKER,
        random_state=RANDOM_STATE)

    print('--- Running the ULA sampler on the whole dataset ---')
    loss_train_full, sample_dim, nb_samples, acf_full = run_ULA_sampler('full', save_path,
                                                                        logistic_full,
                                                                        param_best,
                                                                        lr=lr_full,
                                                                        mc_iter=MC_ITER,
                                                                        thinning=THINNING,
                                                                        burn_in=BURN_IN_RATIO,
                                                                        seed=SEED,
                                                                        nlags=NLAGS)

    print('--- Running the ULA sampler on the subdatasets ---')
    all_loss_train_sub = []
    all_acf_sub = []
    for i in range(NB_DATASETS):
        print(f'Sampling for dataset {i + 1}')
        loss_train_sub, _, _, autocorr_sub = run_ULA_sampler(i + 1, save_path, all_sub_logistic[i],
                                                             all_param_best[i],
                                                             lr=all_lr[i],
                                                             mc_iter=MC_ITER,
                                                             thinning=THINNING,
                                                             burn_in=BURN_IN_RATIO,
                                                             seed=SEED,
                                                             nlags=NLAGS)
        all_loss_train_sub.append(loss_train_sub)
        all_acf_sub.append(autocorr_sub)

    print('Sample dimension: ', sample_dim)
    print('Number of samples: ', nb_samples)

    print('---Evaluating the quality of the sampling---')

    path = os.path.join(save_path, f'loss_train_seed_{SEED}.png')
    save_figure(path, 'NLL evaluated on each train dataset', 'MC iteration', 'NLL',
                all_q_sub=all_loss_train_sub, q_full=loss_train_full)

    path = os.path.join(save_path, f'acf_seed_{SEED}.png')
    save_figure(path, 'ACF-dim=0 (including burn-in & thinning)', 'Lag', 'Autocorrelation',
                all_q_sub=all_acf_sub, q_full=acf_full, log_x=False, log_y=False)


if __name__ == '__main__':
    main()
