import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import fetch_covtype

from ..bayesian_benchmarks.data import (_ALL_CLASSIFICATION_DATASETS,
                                        _ALL_REGRESSION_DATASETS,
                                        get_classification_data,
                                        get_regression_data)


def generate_homogeneous_classification(inputs, targets, num_clients, transform=None, rng=np.random.default_rng(0)):
    datasets_dict = {}
    for label in np.unique(targets):
        idx_label = np.where(targets == label)[0]
        idx_label_permute = rng.permutation(idx_label)
        size_label = len(idx_label) // num_clients
        print(f'Number of samples with label {label} per subdataset: ', size_label)
        num_total = 0
        for client in range(num_clients):
            idx_client = idx_label_permute[num_total:num_total + size_label]
            if client not in datasets_dict.keys():
                datasets_dict[client] = [inputs[idx_client], targets[idx_client]]
            else:
                datasets_dict[client][0] = np.vstack((datasets_dict[client][0], inputs[idx_client]))
                datasets_dict[client][1] = np.hstack((datasets_dict[client][1], targets[idx_client]))
            num_total += size_label
    datasets_t, datasets = [], []
    for x, y in datasets_dict.values():
        datasets.append([x, y])
        if transform == torch.from_numpy:
            datasets_t.append([torch.from_numpy(x), torch.from_numpy(y)])
        elif transform is not None:
            datasets_t.append([x, transform(y.reshape(-1, 1))])
        else:
            datasets_t.append([x, y])
    return datasets_t, datasets


def generate_heterogeneous_classification(inputs, targets, num_clients, transform=None, rng=np.random.default_rng(0)):
    datasets_dict = {}
    for label in np.unique(targets):
        idx_label = np.where(targets == label)[0]
        p = rng.uniform(0, 1, size=num_clients)
        num_data = (1 + np.round((len(idx_label) - 1 * num_clients) * rng.dirichlet(p))).astype('int')
        print(f'Partition for label {label} : ', num_data)
        num_total = 0
        for client, num in enumerate(num_data):
            if client == num_clients - 1:
                num = len(idx_label) - num_total
            idx_client = idx_label[num_total: num_total + num]
            if num == 0:
                continue
            if client not in datasets_dict.keys():
                datasets_dict[client] = [inputs[idx_client], targets[idx_client]]
            else:
                datasets_dict[client][0] = np.vstack((datasets_dict[client][0], inputs[idx_client]))
                datasets_dict[client][1] = np.hstack((datasets_dict[client][1], targets[idx_client]))
            num_total += num
    datasets_t, datasets = [], []
    for x, y in datasets_dict.values():
        datasets.append([x, y])
        if transform == torch.from_numpy:
            datasets_t.append([torch.from_numpy(x), torch.from_numpy(y)])
        elif transform is not None:
            datasets_t.append([x, transform(y.reshape(-1, 1))])
        else:
            datasets_t.append([x, y])
    return datasets_t, datasets


def load_UCI_dataset(dataset_name, rng=None, prop=.9, random_state=0):
    print('Preparing dataset %s' % dataset_name)

    if dataset_name in _ALL_CLASSIFICATION_DATASETS:
        dataset = get_classification_data(dataset_name, rng, prop)
    elif dataset_name in _ALL_REGRESSION_DATASETS:
        dataset = get_regression_data(dataset_name, rng, prop)
    else:
        raise NameError('Invalid dataset_name.')

    print(f'Statistics: N={dataset.N}, D={dataset.D}, Xtrain={dataset.X_train.shape}')

    if dataset_name in _ALL_CLASSIFICATION_DATASETS:
        Xtrain = dataset.X_train
        Ytrain = dataset.Y_train.ravel()
        le = LabelEncoder()
        le.fit(Ytrain)
        Ytrain = le.transform(Ytrain)
        Ytest = le.transform(dataset.Y_test.ravel())

    elif dataset_name in _ALL_REGRESSION_DATASETS:
        Xtrain, Ytrain = shuffle(dataset.X_train, dataset.Y_train, random_state=random_state)
        scalerY = StandardScaler()
        scalerY.fit(np.concatenate((dataset.Y_train, dataset.Y_test)))
        Ytrain = scalerY.transform(Ytrain)
        Ytest = scalerY.transform(dataset.Y_test)

    scalerX = StandardScaler()
    scalerX.fit(np.concatenate((Xtrain, dataset.X_test)))
    Xtrain = scalerX.transform(Xtrain)
    Xtest = scalerX.transform(dataset.X_test) if len(dataset.Y_test) > 0 else dataset.X_test

    return Xtrain, Xtest, Ytrain, Ytest


def prepare_covertype_data(path_data, max_data_size, rng, random_state, prop_train=0.8):
    """Preprocessing of covertype data."""
    inputs, targets = fetch_covtype(data_home=path_data, download_if_missing=True, return_X_y=True,
                                    random_state=random_state)

    # Transform to binary dataset
    idx = np.where(targets <= 2)[0]
    inputs = inputs[idx]
    targets = targets[idx]

    inputs = StandardScaler().fit_transform(inputs)
    targets = LabelEncoder().fit_transform(targets)

    # Subsample
    if len(targets) > max_data_size:
        id_subset = rng.choice(len(targets), max_data_size, replace=False)
        inputs = inputs[id_subset]
        targets = targets[id_subset]
        print(f'inputs = {np.shape(inputs)}, targets = {np.shape(targets)}')

    return train_test_split(inputs, targets, test_size=1. - prop_train, random_state=random_state)


def prepare_data(data, nb_datasets, max_data_size, heterogeneity, path_data, rng, random_state, prop_train=0.8):
    """Prepare the dataset and splits it between subdatasets according to heterogeneity or not."""
    if data == 'covertype':
        Xtrain, Xtest, Ytrain, Ytest = prepare_covertype_data(path_data, max_data_size, rng, random_state, prop_train)
    else:
        Xtrain, Xtest, Ytrain, Ytest = load_UCI_dataset(data, rng, prop=prop_train)

    output_dim = len(np.unique(Ytrain))
    if output_dim == 2:  # for binary logistic regression
        output_dim = 1

    oe = OneHotEncoder(sparse=False).fit(Ytrain.reshape(-1, 1))
    transform = oe.transform if output_dim > 2 else None

    # splitting the dataset
    if heterogeneity == 'het':
        datasets_t, datasets = generate_heterogeneous_classification(Xtrain, Ytrain, nb_datasets, transform, rng)
    elif heterogeneity == 'hom':
        datasets_t, datasets = generate_homogeneous_classification(Xtrain, Ytrain, nb_datasets, transform, rng)

    Ytrain_oe = Ytrain
    Ytest_oe = Ytest

    if output_dim > 2:
        Ytrain_oe = oe.transform(Ytrain.reshape(-1, 1))
        Ytest_oe = oe.transform(Ytest.reshape(-1, 1))

    return datasets_t, datasets, oe, Xtrain, Xtest, Ytrain, Ytrain_oe, Ytest, Ytest_oe, output_dim


def save_figure(save_path, title, x_label, y_label, all_q_sub, q_full, log_x=True, log_y=True):
    plt.figure(2)
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
    plt.grid('True')
    plt.plot(q_full, label='data: full')
    for i in range(len(all_q_sub)):
        plt.plot(all_q_sub[i], label=f'data: {i + 1}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(save_path), bbox_inches='tight')
    plt.clf()
