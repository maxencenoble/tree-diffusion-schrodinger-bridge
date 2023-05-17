import numpy as np
import os
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from statsmodels.tsa.stattools import acf


def run_ULA_sampler(name_data, save_path, logistic_train, param_init, lr, mc_iter, thinning, burn_in, seed, nlags):
    """Run the ULA sampler on the model logistic_train. Evaluates the performance along the model logistic_test."""
    ula = ULA(logistic_train.grad, lr)
    param_ula = np.copy(param_init)
    loss_train_init = logistic_train.loss(param_init)
    loss_train = np.zeros(mc_iter + 1)
    loss_train[0] = np.copy(loss_train_init)
    for i in tqdm(range(mc_iter)):
        param_ula = ula.step(np.copy(param_ula))
        loss_train[i + 1] = logistic_train.loss(np.copy(param_ula))

    # Save the subsamples
    length = len(ula.get_saved_params())
    idx = np.arange(int(burn_in * length), length, step=thinning, dtype=int)
    samples = np.take(ula.get_saved_params(), idx, axis=0)
    samples = samples.reshape(samples.shape[0], -1)
    np.save(os.path.join(save_path, f'dataset_{name_data}_seed_{seed}'), samples)

    # compute acf on the first dimension
    autocorr = acf(samples[:, 0], nlags=nlags)

    return loss_train, samples.shape[1], samples.shape[0], autocorr


def get_learning_rate(Xtrain, l2, base_lr=0.01):
    """Heuristic to get an appropriate learning rate for MALA."""
    w, v = np.linalg.eig(np.dot(Xtrain.T, Xtrain))
    w = np.max(w.real)
    M = w * (1 / 4) + l2
    return base_lr / M


def prepare_models(datasets_t, datasets, oe, Xtrain, Xtest, Ytrain, Ytrain_oe, Ytest, output_dim,
                   temperature, l2, reg_minsker, random_state, max_iter=1000):
    """Define the Logistic model for the full dataset and teh subdatasets."""
    # Define the logistic regression model
    Logistic = LogisticGradSto if output_dim > 2 else BinaryLogisticGradSto

    # Define the model for the whole dataset
    batch_size_full = len(Ytrain_oe)
    logistic_full = Logistic(Xtrain, Ytrain_oe, batch_size_full, temperature, l2, oe)

    # Calculate the learning rate with heuristic for the whole dataset
    lr_full = get_learning_rate(Xtrain, l2)
    print('Learning rate on the full dataset: ', lr_full)

    # Define the deterministic model
    clf = LogisticRegression(fit_intercept=False, random_state=random_state, max_iter=max_iter).fit(Xtrain, Ytrain)
    print(f'Accuracy of the full deterministic model = {(10 ** 2 * clf.score(Xtest, Ytest)).round(1)}%')

    # Give the coefficient of the model
    param_best = np.squeeze(clf.coef_.T)

    # Define the model for each subdataset
    all_sub_logistic = list()
    all_lr = list()
    all_param_best = list()
    for i in range(len(datasets)):
        Xi, Yi_oe, Yi = datasets_t[i][0], datasets_t[i][1], datasets[i][1]
        batch_size = len(Yi)
        all_sub_logistic.append(Logistic(Xi, Yi_oe, batch_size, temperature, l2, oe, reg_minsker))
        lr = get_learning_rate(Xi, l2)
        print(f'Learning rate on the subdataset {i + 1}: ', lr)
        all_lr.append(lr)
        clf = LogisticRegression(fit_intercept=False, random_state=random_state, max_iter=max_iter).fit(Xi, Yi)
        all_param_best.append(np.squeeze(clf.coef_.T))

    return logistic_full, lr_full, param_best, all_sub_logistic, all_lr, all_param_best


class GradStoU:

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.num_clients = len(self.dataset)
        self.datasize = len(dataset)
        self.batch_size = min(batch_size, self.datasize)

    def update_minibatch(self):
        ind = np.random.choice(self.datasize, size=self.batch_size, replace=False)
        self.batch_i = self.dataset[ind]

    def gradsto(self, theta):
        # batch_i = np.squeeze(random.sample(list(self.dataset), self.batch_size))
        return (theta - self.batch_i.mean(axis=0)) / self.num_clients

    def grad(self, theta):
        return (theta - self.dataset.mean(axis=0)) / self.num_clients


class BinaryLogisticGradSto:

    def __init__(self, Xtrain, Ytrain, batch_size, temperature=1., l2=.1, oe=None, reg_minsker=1.):
        """ We need y in {0,1}. """
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.datasize = len(Ytrain)
        self.batch_size = min(batch_size, self.datasize)
        self.temperature = temperature
        self.l2 = l2 / temperature
        self.reg_minsker = reg_minsker

    def loss(self, theta):
        Xtheta = np.dot(self.Xtrain, theta)
        Xtheta_pos = np.maximum(0, Xtheta)
        Xtheta_neg = Xtheta - Xtheta_pos
        neglogit1 = np.log(1 + np.exp(- Xtheta_pos)) + np.log(1 + np.exp(Xtheta_neg)) - Xtheta_neg
        penalization = (self.temperature * self.l2 / 2) * np.linalg.norm(theta) ** 2
        return self.reg_minsker * np.sum(neglogit1 + (1 - self.Ytrain) * Xtheta) + penalization

    def grad_loss(self, theta, X, Y):
        return (self.reg_minsker * self.datasize / self.temperature) * np.mean(
            X.T * (1 / (1 + np.exp(- np.dot(X, theta))) - Y),
            axis=1) + self.l2 * theta

    def update_minibatch(self):
        ind = np.random.choice(self.datasize, size=self.batch_size, replace=False)
        self.Xbatch_i, self.Ybatch_i = self.Xtrain[ind], self.Ytrain[ind]

    def gradsto(self, theta):
        return self.grad_loss(theta, self.Xbatch_i, self.Ybatch_i)

    def grad(self, theta):
        return self.grad_loss(theta, self.Xtrain, self.Ytrain)

    def proba(self, X, theta):
        Xtheta = np.dot(X, theta)
        return 1 / (1 + np.exp(- Xtheta))


class LogisticGradSto:

    def __init__(self, Xtrain, Ytrain, batch_size, temperature=1., l2=.1, oe=None, reg_minsker=1.):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        if oe is None:
            pass  # todo
        else:
            self.idy = np.expand_dims(oe.inverse_transform(self.Ytrain).squeeze(), axis=1)
        self.datasize = len(Ytrain)
        self.batch_size = min(batch_size, self.datasize)
        self.temperature = temperature
        self.l2 = l2 / temperature
        self.reg_minsker = reg_minsker

    def loss(self, theta):
        """
        Y: onehot encoded
        """
        Xtheta = np.dot(self.Xtrain, theta)
        penalization = (self.temperature * self.l2 / 2) * np.linalg.norm(theta) ** 2
        min_Xtheta = np.min(Xtheta, axis=1)
        # loss_ = np.take_along_axis(Xtheta, self.idy, axis=1).mean() + np.mean(
        #     np.log(np.sum(np.exp(np.expand_dims(min_Xtheta, axis=1) - Xtheta), axis=1)) - min_Xtheta)
        # return loss_ / self.temperature + penalization
        return self.reg_minsker * np.take_along_axis(Xtheta, self.idy, axis=1).sum() + self.reg_minsker * np.sum(
            np.log(np.sum(np.exp(np.expand_dims(min_Xtheta, axis=1) - Xtheta), axis=1)) - min_Xtheta) + penalization

    def grad_loss(self, theta, X, Y):
        Xtheta = - np.dot(X, theta)
        P = softmax(Xtheta, axis=1)
        return self.reg_minsker * self.datasize * X.T.dot(Y - P) / (
                len(Y) * self.temperature) + self.l2 * theta  # Normalized

    def update_minibatch(self):
        ind = np.random.choice(self.datasize, size=self.batch_size, replace=False)
        self.Xbatch_i, self.Ybatch_i = self.Xtrain[ind], self.Ytrain[ind]

    def gradsto(self, theta):
        return self.grad_loss(theta, self.Xbatch_i, self.Ybatch_i)

    def grad(self, theta):
        return self.grad_loss(theta, self.Xtrain, self.Ytrain)

    def proba(self, X, theta):
        return softmax(- np.dot(X, theta), axis=1)


Logistic = [BinaryLogisticGradSto, LogisticGradSto][0]  # Todo: change for multi-logistic regression.


class Optimization(Logistic):

    def __init__(self, Xtrain, Ytrain, batch_size=64, temperature=1, l2=.1):
        super().__init__(Xtrain, Ytrain, batch_size, temperature, l2)
        self.log = []

    def SGD(self, theta, lr, niter=10, thinning=1, verbose=True):
        iterator = range(niter)
        if verbose == True:
            iterator = tqdm(iterator)
        for epoch in iterator:
            self.update_minibatch()
            grad = self.gradsto(theta)
            theta -= lr * grad
            if epoch % thinning == 0:
                self.log.append(np.linalg.norm(grad))
        return theta


class GaussianToy:

    def __init__(self, mean, cov_inv):
        self.mean = mean
        self.cov_inv = cov_inv

    def update_minibatch(self):
        pass

    def gradsto(self, theta):
        return np.dot(self.cov_inv, theta - self.mean)

    def grad(self, theta):
        return np.dot(self.cov_inv, theta - self.mean)

    def loss(self, theta):
        vector = theta - self.mean
        return vector.dot(self.cov_inv).dot(vector) / 2


class GaussianToyProduct:

    def __init__(self, means, cov):
        self.means = means
        self.cov_inv = [np.linalg.inv(sigma) for sigma in cov]

    # Define the log posterior
    def log_pi(self, theta):  # TODO: remove the loop
        logpi_theta = 0
        for (mu, sigma_inv) in zip(self.means, self.cov_inv):
            vector = theta - mu
            logpi_theta += vector.dot(self.cov_inv).dot(vector) / 2
        return logpi_theta

    # Define the gradient of the log posterior
    def log_grad(self, theta):  # TODO: remove the loop
        loggrad_theta = 0
        for (mu, sigma_inv) in zip(self.means, self.cov_inv):
            loggrad_theta += np.dot(self.cov_inv, theta - mu)
        return loggrad_theta


class GaussianU:

    def __init__(self, dataset, batch_size, cov_inv):
        self.dataset = dataset
        self.datasize = len(dataset)
        self.batch_size = min(batch_size, self.datasize)
        self.cov_inv = cov_inv

    def update_minibatch(self):
        ind = np.random.choice(self.datasize, size=self.batch_size, replace=False)
        self.batch_i = self.dataset[ind]

    def gradsto(self, theta):
        return self.cov_inv.dot(theta - self.batch_i.mean(axis=0)) / self.datasize

    def grad(self, theta):
        return self.cov_inv.dot(theta - self.dataset.mean(axis=0)) / self.datasize


class GaussianProduct:

    def __init__(self, datasets, mean_list, cov_list):
        self.datasets = datasets
        self.mean_list = mean_list
        self.cov_inv_list = [np.linalg.inv(cov) for cov in cov_list]

    # Define the log posterior
    def log_pi(self, x):
        Ux = 0
        for (Xi, cov_inv) in zip(self.datasets, self.cov_inv_list):
            vector = Xi - np.tile(x, (len(Xi), 1))
            Ux += np.trace(vector.dot(cov_inv).dot(vector.T)) / (2 * len(Xi))
        return Ux

    # Define the gradient of the log posterior
    def log_grad(self, x):
        Ux = 0
        for (Xi, cov_inv) in zip(self.datasets, self.cov_inv_list):
            vector = Xi - np.tile(x, (len(Xi), 1))
            Ux += np.mean(np.dot(vector, cov_inv), axis=0)
        return Ux


def mse_calculation(func, Sampler, args, niter, mc_iter, burn_in, param_true, verbose=False, path_save=None):
    iterator0 = range(niter) if verbose else tqdm(range(niter))
    # stores the estimate mse
    mse = np.zeros((niter, mc_iter - burn_in + 1))
    for mc in iterator0:  # todo: this loop should be performed in parallel
        iterator1 = tqdm(range(1, mc_iter + 1)) if verbose else range(1, mc_iter + 1)
        # initialize the sampler
        sampler = Sampler(*args)
        # initialize the list containing the func(theta) where the theta are sampled by the algorithm
        ftheta = np.zeros(mc_iter + 1)
        ftheta[0] = func(sampler.get_server_param())  # sampler.theta
        for it in iterator1:
            sampler.step()
            ftheta[it] = func(sampler.get_server_param())  # sampler.theta
        # remove the burn in period
        ftheta = ftheta[burn_in:]
        # update the mse
        param_new = np.cumsum(ftheta, axis=0) / np.arange(1, mc_iter - burn_in + 2)
        mse_new = (param_new - param_true) ** 2
        mse[mc] = np.copy(mse_new)  # (mse_new + mc * mse) / (mc + 1)
        if path_save is not None:
            np.save(path_save + f'-{mc + 1}', sampler.get_saved_params())
    return mse


class Mala:

    def __init__(self, tau=.1, log_pi=lambda x: - np.linalg.norm(x) ** 2 / 2, log_grad=lambda x: - x):
        self.tau = tau
        self.log_pi = log_pi
        self.log_grad = log_grad

    def __call__(self, x):
        return np.exp(self.log_pi(x))

    def ratio(self, x, y):
        return np.exp(- np.linalg.norm(y - x - self.tau * self.log_grad(x)) ** 2 / (4 * self.tau))

    def step(self, x, maxiter=5 * 1e3, niter=0):
        xi = np.random.randn(len(x))
        y = x + self.tau * self.log_grad(x) + np.sqrt(2 * self.tau) * xi
        alpha = min(1, self(y) * self.ratio(y, x) / (self(x) * self.ratio(x, y)))
        u = np.random.rand()
        if u <= alpha:
            return y
        elif niter > maxiter:
            print('Exceed the recursion limit setting.')
            return x
        return self.step(x, maxiter, niter + 1)


class ULA:

    def __init__(self, minus_log_grad, stepsize=.1):
        self.stepsize = stepsize
        self.minus_log_grad = minus_log_grad
        self.saved_params = []

    def step(self, x):
        gaussian = np.random.randn(*np.shape(x))
        xnew = x - self.stepsize * self.minus_log_grad(x) + np.sqrt(2 * self.stepsize) * gaussian
        self.saved_params.append(xnew)
        return xnew

    def get_saved_params(self):
        return np.squeeze(self.saved_params)


def hpd(samples=None, loss_func=None, loss_list=None, alpha=.01):
    if loss_list is not None:
        idx = int(alpha * len(loss_list))
        return np.sort(loss_list)[::-1][idx]
    if len(samples) < 1 / alpha:
        return None
    idx = int(alpha * len(samples))
    return np.sort([loss_func(theta) for theta in samples])[::-1][idx]


class Dglmc:

    def __init__(self, gradU, pc=1, stepsize=.1, rho=.1, param_clients=None):
        self.gradU = gradU
        self.dim = param_clients.shape[1:]
        self.param_server, self.param_clients = param_clients.mean(axis=0), param_clients
        self.stepsize, self.rho, self.pc = stepsize, rho, pc
        self.saved_params = []

    def z_update(self):
        for i, gradu in enumerate(self.gradU):
            gradu.update_minibatch()
            gradient = (self.param_server - self.param_clients[i]) / self.rho - gradu.gradsto(self.param_clients[i])
            noise = np.sqrt(2 * self.stepsize) * np.random.randn(*self.dim)
            self.param_clients[i] = np.copy(self.param_clients[i] + self.stepsize * gradient + noise)

    def param_server_update(self):
        mu = np.mean(self.param_clients, axis=0)
        self.param_server = np.copy(mu + np.sqrt(self.rho / len(self.param_clients)) * np.random.randn(*self.dim))
        self.saved_params.append(self.param_server)

    def step(self):
        self.z_update()
        if np.random.binomial(1, self.pc, 1):
            self.param_server_update()
            return True

    def get_server_param(self):
        return np.copy(self.param_server)

    def get_saved_params(self):
        return np.squeeze(self.saved_params)


class StochasticQuantization:

    def __init__(self, s=1):
        if s == 0:
            print("There will be no compression")
            # raise ValueError(("There will be no compression")
        self.s = s

    def quantize(self, v):
        if self.s == 0:
            return v
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return v
        r = self.s * np.abs(v) / v_norm
        l = np.floor(r)
        l += np.ceil(r - l) - np.ones_like(l)
        b = np.random.binomial(n=1, p=r - l)
        xi = (l + b) / self.s
        return v_norm * np.sign(v) * xi

    def communication_eval(self, v):
        s = self.s
        dim_v = len(np.flatten(v))
        if s == 0:
            return 32 * dim_v  # there is no quantization
        elif s < np.sqrt(dim_v / 2 - np.sqrt(dim_v)):
            t = s * (s + np.sqrt(dim_v))
            return 32 + 3 * t * (1 + np.log(2 * (s ** 2 + dim_v) / t) / 2)
        else:
            t = s ** 2 + min(dim_v, s * np.sqrt(dim_v))
            return 32 + dim_v * (2 + (1 + np.log(1 + t / dim_v)) / 2)


class Qlsd:

    def __init__(self, gradU, quantization, stepsize=.1, param0=None):
        self.gradU = gradU
        self.quantization = quantization
        self.stepsize = stepsize
        self.param = np.copy(param0)
        self.dim = np.shape(param0)
        self.saved_params = []

    def step(self):
        gradient = 0
        for grad in self.gradU:
            gradient += self.quantization(grad.gradsto(np.copy(self.param)))
        self.param -= self.stepsize * gradient + np.sqrt(2 * self.stepsize) * np.random.randn(*self.dim)
        self.saved_params.append(self.param)

    def get_server_param(self):
        return np.copy(self.param)

    def get_saved_params(self):
        return np.squeeze(self.saved_params)


class QlsdPp:  # Todo: add a Base class

    def __init__(self, gradU, quantization, stepsize=.1, param0=None, memory_coef=.1):
        self.gradU = gradU
        self.quantization = quantization
        self.stepsize = stepsize
        self.param = np.copy(param0)
        self.dim = np.shape(param0)
        self.memory_coef = memory_coef
        self.memory_term = np.zeros((len(gradU), *self.dim))
        self.saved_params = []

    def step(self):
        gradient = 0
        memory_sum = self.memory_term.sum(axis=0)
        for i, grad in enumerate(self.gradU):
            grad_i = self.quantization(grad.gradsto(np.copy(self.param)) - self.memory_term[i])
            gradient += np.copy(grad_i)
            self.memory_term[i] += self.memory_coef * np.copy(grad_i)
        gradient_ = gradient + memory_sum
        self.param -= self.stepsize * gradient_ + np.sqrt(2 * self.stepsize) * np.random.randn(*self.dim)
        self.saved_params.append(self.param)

    def get_server_param(self):
        return np.copy(self.param)

    def get_saved_params(self):
        return np.squeeze(self.saved_params)


def average(it, x0, x1):
    if it == 0:
        return np.copy(x1)
    return (it * np.copy(x0) + np.copy(x1)) / (it + 1)


def heterogeneity(gamma, pc, means, cov, mean_true, cov_true):
    cov_eig = np.linalg.eig(cov_true)[0]
    mult_coeff = (gamma * max(cov_eig) / (pc * min(cov_eig))) ** 2
    sum_norm = 0
    for (mu, sigma) in zip(means, cov):
        sum_norm += np.linalg.norm(np.linalg.inv(sigma).dot(mean_true - mu))
    return mult_coeff * sum_norm / len(means)
