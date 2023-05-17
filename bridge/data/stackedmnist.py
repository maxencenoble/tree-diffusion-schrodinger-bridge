import os, shutil
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.utils import save_image
import torchvision.transforms as transforms


def mnist_transforms(size):
    return transforms.Compose([transforms.Resize(size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])


def mnist_inv_transforms():
    return transforms.Compose([transforms.Normalize((0.,), (1 / 0.5,)),
                               transforms.Normalize((-0.5,), (1.,))])


class Stacked_MNIST(Dataset):
    def __init__(self, source_data, root, load, label, imageSize, num_channels):
        super(Stacked_MNIST, self).__init__()
        self.digit_class = int(label[-1])
        self.num_channels = min(3, num_channels)
        if load:
            self.data = torch.load(os.path.join(root, "data.pt"))
            self.targets = torch.load(os.path.join(root, "targets.pt"))
        else:
            source_data = list(filter(lambda i: i[1] == self.digit_class, source_data))
            self.data = torch.zeros((0, self.num_channels, imageSize, imageSize))
            self.targets = torch.zeros((0), dtype=torch.int64)
            # around 6000 images in total per digit class 
            dataloader_R = DataLoader(source_data, batch_size=100, shuffle=True, drop_last=True)
            dataloader_G = DataLoader(source_data, batch_size=100, shuffle=True, drop_last=True)
            dataloader_B = DataLoader(source_data, batch_size=100, shuffle=True, drop_last=True)

            im_dir = root + '/im'
            if os.path.exists(im_dir):
                shutil.rmtree(im_dir)
            os.makedirs(im_dir)

            idx = 0
            for (xR, yR), (xG, yG), (xB, yB) in tqdm(zip(dataloader_R, dataloader_G, dataloader_B)):
                x = torch.cat([xR, xG, xB][-self.num_channels:], dim=1)
                y = (100 * yR + 10 * yG + yB) % 10 ** self.num_channels
                self.data = torch.cat((self.data, x), dim=0)
                self.targets = torch.cat((self.targets, y), dim=0)

                for k in range(100):
                    if idx < 10000:
                        im = x[k]
                        filename = root + '/im/{:05}.jpg'.format(idx)
                        save_image(im, filename)
                    idx += 1

            if not os.path.isdir(root):
                os.makedirs(root)

            torch.save(self.data, os.path.join(root, "data.pt"))
            torch.save(self.targets, os.path.join(root, "targets.pt"))
        self.mean_per_dim = self.data.mean(axis=0)
        self.var_per_dim = self.data.var(axis=0)
        self.nb_samples_total = self.data.shape[0]

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        return img, targets

    def __len__(self):
        return len(self.targets)
