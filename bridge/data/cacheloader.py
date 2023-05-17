import torch
from torch.utils.data import Dataset
import time

DATASET_2D = '2d'
DATASET_CELEBA = 'celeba'
DATASET_STACKEDMNIST = 'stackedmnist'


class CacheLoader(Dataset):
    def __init__(self,
                 num_batches,
                 langevin,
                 shape,
                 batch_size,
                 start_corrector,
                 device,
                 plot_cache_time):

        super().__init__()
        self.num_batches = num_batches
        self.shape = shape
        self.device = device
        self.langevin = langevin
        self.batch_size = batch_size
        self.num_steps = langevin.num_steps
        self.start_corrector = start_corrector
        self.plot_cache_time = plot_cache_time
        self.clear_data()

    def clear_data(self):
        self.data = torch.zeros(
            (self.num_batches, self.batch_size * self.num_steps, 2, *self.shape)).to(self.device)  # .cpu()
        self.next_data = torch.zeros((self.num_batches, self.batch_size, *self.shape)).to(self.device)  # .cpu()
        self.steps_data = torch.zeros(
            (self.num_batches, self.batch_size * self.num_steps, 1)).to(self.device)  # .cpu() # steps

    def update_data(self, sample_net, backward_net, init_cache_dl, n, first_pass_on_edge):
        self.clear_data()
        torch.cuda.empty_cache()

        start = time.time()

        with torch.no_grad():
            for b in range(self.num_batches):
                batch = next(init_cache_dl)[0]
                batch = batch.to(self.device)

                if n < 2 and (not first_pass_on_edge):
                    x, out, steps_expanded = self.langevin.record_init_langevin(batch, self.shape)
                else:
                    x, out, steps_expanded = self.langevin.record_langevin_seq(sample_net,
                                                                               backward_net,
                                                                               batch,
                                                                               self.shape,
                                                                               corrector=False)

                # store the last iterate
                self.next_data[b] = x[:, -1, :]

                # store x, out
                x = x.unsqueeze(2)
                out = out.unsqueeze(2)
                flat_data = torch.cat((x, out), dim=2).flatten(start_dim=0, end_dim=1)
                self.data[b] = flat_data

                # store steps
                flat_steps = steps_expanded.flatten(start_dim=0, end_dim=1)
                self.steps_data[b] = flat_steps

                del batch
                del x
                del out
                del steps_expanded
                del flat_data
                del flat_steps
                torch.cuda.empty_cache()

        self.next_data = self.next_data.flatten(start_dim=0, end_dim=1)
        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)

        stop = time.time()
        if self.plot_cache_time:
            print('Cache size: {0}'.format(self.data.shape))
            print("Load time: {0}".format(stop - start))

        torch.cuda.empty_cache()

    def __getitem__(self, index):
        item = self.data[index]
        x = item[0]
        out = item[1]
        steps = self.steps_data[index]

        return x, out, steps

    def __len__(self):
        return self.data.shape[0]
