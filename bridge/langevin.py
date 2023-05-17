import torch
import numpy as np


class Langevin(torch.nn.Module):
    def __init__(self, num_steps, num_corrector_steps, gammas, device=None, mean_match=True, snr=0.01):
        super().__init__()

        self.mean_match = mean_match

        self.num_steps = num_steps  # num diffusion steps
        self.num_corrector_steps = num_corrector_steps
        self.gammas = gammas.float()  # schedule
        self.snr = snr

        if device is not None:
            self.device = device
        else:
            self.device = gammas.device

        self.steps = torch.arange(self.num_steps).to(self.device)
        self.time = torch.cumsum(self.gammas, 0).to(self.device).float()  # T on the edge

    def record_init_langevin(self, source_samples, shape):

        x = source_samples
        N = x.shape[0]
        time = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))

        x_tot = torch.Tensor(N, self.num_steps, *shape).to(x.device)
        out = torch.Tensor(N, self.num_steps, *shape).to(x.device)

        num_iter = self.num_steps
        steps_expanded = time

        for k in range(num_iter):
            gamma = self.gammas[k]

            # Discretized Brownian motion
            t_old = x
            z = torch.randn(x.shape, device=x.device)
            x = t_old + torch.sqrt(2 * gamma) * z
            t_new = x

            x_tot[:, k, :] = x
            out[:, k, :] = (t_old - t_new)  # / (2 * gamma)

            del t_old
            del t_new
            torch.cuda.empty_cache()

        return x_tot, out, steps_expanded

    def record_langevin_seq(self, forward_net, backward_net, source_samples, shape, corrector, schedule='zero',
                            coeff_schedule=2, sample=False):

        x = source_samples
        N = x.shape[0]
        time = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        steps = time

        schedule_rate = np.linspace(0, 1, self.num_steps)
        if schedule == 'zero':
            schedule_rate = np.zeros(self.num_steps)
        if schedule == 'cosine':
            schedule_rate = 0.5 * (1 + np.cos(np.pi * (1 - schedule_rate) ** coeff_schedule))
        if schedule == 'binary':
            schedule_rate = np.zeros(self.num_steps)
            schedule_rate[int(coeff_schedule * self.num_steps):] = 1.

        x_tot = torch.Tensor(N, self.num_steps, *shape).to(x.device)
        out = torch.Tensor(N, self.num_steps, *shape).to(x.device)

        num_iter = self.num_steps
        num_corrector_steps = self.num_corrector_steps if corrector else 0
        steps_expanded = steps

        if self.mean_match:
            for k in range(num_iter):
                gamma = self.gammas[k]
                t_old = forward_net(x, steps[:, k, :])

                if sample & (k == num_iter - 1) & (num_corrector_steps == 0):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma * (1 - schedule_rate[k])) * z

                k_ = k if k == num_iter - 1 else k + 1
                gamma_ = self.gammas[k_]

                for _ in range(num_corrector_steps):
                    z = torch.randn(x.shape, device=x.device)
                    z_norm = z.reshape(N, -1).norm(dim=1).mean()

                    score = 0.5 * (forward_net(x, steps[:, k_, :]) + backward_net(x, self.time[-1] - steps[:, k_, :]))
                    score_norm = score.reshape(N, -1).norm(dim=1).mean()

                    eps = 2 * (self.snr / score_norm) ** 2
                    eps = eps * (z_norm ** 2)

                    x = (1 - eps) * x + eps * score + torch.sqrt(2 * eps * gamma_) * z

                t_new = forward_net(x, steps[:, k, :])
                x_tot[:, k, :] = x
                out[:, k, :] = (t_old - t_new)

                del t_old
                del t_new
                torch.cuda.empty_cache()
        else:
            for k in range(num_iter):
                gamma = self.gammas[k]
                t_old = x + forward_net(x, steps[:, k, :])

                if sample & (k == num_iter - 1) & (num_corrector_steps == 0):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma * (1 - schedule_rate[k])) * z

                k_ = k if k == num_iter - 1 else k + 1
                gamma_ = self.gammas[k_]

                for _ in range(num_corrector_steps):
                    z = torch.randn(x.shape, device=x.device)
                    z_norm = z.reshape(N, -1).norm(dim=1).mean()

                    score = 0.5 * (forward_net(x, steps[:, k_, :]) + backward_net(x, self.time[-1] - steps[:, k_, :]))
                    score_norm = score.reshape(N, -1).norm(dim=1).mean()

                    eps = 2 * (self.snr / score_norm) ** 2
                    eps = eps * (z_norm ** 2)

                    x = x + eps * score + torch.sqrt(2 * eps * gamma_) * z

                t_new = x + forward_net(x, steps[:, k, :])
                x_tot[:, k, :] = x
                out[:, k, :] = (t_old - t_new)

                del t_old
                del t_new
                torch.cuda.empty_cache()

        return x_tot, out, steps_expanded

    def record_ode_seq(self, forward_net, backward_net, source_samples, shape, corrector, schedule='cosine',
                       coeff_schedule=2, sample=False):

        x = source_samples
        N = x.shape[0]
        time = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        steps = time

        schedule_rate = np.linspace(0, 1, self.num_steps)
        if schedule == 'cosine':
            schedule_rate = 0.5 * (1 + np.cos(np.pi * (1 - schedule_rate) ** coeff_schedule))
        if schedule == 'one':
            schedule_rate = np.ones(self.num_steps)

        x_tot = torch.Tensor(N, self.num_steps, *shape).to(x.device)

        num_iter = self.num_steps
        num_corrector_steps = self.num_corrector_steps if corrector else 0
        steps_expanded = steps

        if self.mean_match:
            for k in range(num_iter):
                gamma = self.gammas[k]
                t_old = schedule_rate[k] * x + (1 - 0.5 * schedule_rate[k]) * forward_net(x, steps[:, k, :]) - 0.5 * \
                        schedule_rate[k] * backward_net(x, self.time[-1] - steps[:, k, :])

                if sample & (k == num_iter - 1) & (num_corrector_steps == 0):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma * (1 - schedule_rate[k])) * z

                k_ = k if k == num_iter - 1 else k + 1
                gamma_ = self.gammas[k_]

                for _ in range(num_corrector_steps):
                    z = torch.randn(x.shape, device=x.device)
                    z_norm = z.reshape(N, -1).norm(dim=1).mean()

                    score = 0.5 * (forward_net(x, steps[:, k_, :]) + backward_net(x, self.time[-1] - steps[:, k_, :]))
                    score_norm = score.reshape(N, -1).norm(dim=1).mean()

                    eps = 2 * (self.snr / score_norm) ** 2
                    eps = eps * (z_norm ** 2)

                    x = (1 - eps) * x + eps * score + torch.sqrt(2 * eps * gamma_) * z

                x_tot[:, k, :] = x

                del t_old
                torch.cuda.empty_cache()
        else:
            for k in range(num_iter):
                gamma = self.gammas[k]
                t_old = x + (1 - 0.5 * schedule_rate[k]) * forward_net(x, steps[:, k, :]) - 0.5 * \
                        schedule_rate[k] * backward_net(x, self.time[-1] - steps[:, k, :])

                if sample & (k == num_iter - 1) & (num_corrector_steps == 0):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma * (1 - schedule_rate[k])) * z

                k_ = k if k == num_iter - 1 else k + 1
                gamma_ = self.gammas[k_]

                for _ in range(num_corrector_steps):
                    z = torch.randn(x.shape, device=x.device)
                    z_norm = z.reshape(N, -1).norm(dim=1).mean()

                    score = 0.5 * (forward_net(x, steps[:, k_, :]) + backward_net(x, self.time[-1] - steps[:, k_, :]))
                    score_norm = score.reshape(N, -1).norm(dim=1).mean()

                    eps = 2 * (self.snr / score_norm) ** 2
                    eps = eps * (z_norm ** 2)

                    x = x + eps * score + torch.sqrt(2 * eps * gamma_) * z

                x_tot[:, k, :] = x

                del t_old
                torch.cuda.empty_cache()

        return x_tot, None, steps_expanded
