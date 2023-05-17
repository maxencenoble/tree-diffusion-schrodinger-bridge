import os
import PIL
import torch
import numpy as np
import torchvision.transforms as transforms

from .vision import VisionDataset

CELEBA_ATTRIBUTES = ['5_o_Clock_Shadow', 'Arched_Eyebrows',
                     'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                     'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
                     'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                     'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                     'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                     'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                     'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                     'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


def celeba_transforms(size, random_flip):
    transform = [transforms.CenterCrop(148),
                 transforms.Resize(size),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])]
    if random_flip:
        transform.insert(0, transforms.RandomHorizontalFlip())
    return transforms.Compose(transform)


def celeba_inv_transforms():
    return transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                               transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.])])


class CelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "./"
    # There currently does not appear to be an easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = ['list_landmarks_align_celeba.txt',
                 'list_eval_partition.txt',
                 'list_bbox_celeba.txt',
                 'list_attr_celeba.txt',
                 'img_align_celeba.zip',
                 'identity_CelebA.txt']

    def __init__(self, root,
                 label=None,
                 split="train",
                 target_type="attr",
                 transform=None,
                 target_transform=None,
                 download=False):
        import pandas
        super(CelebA, self).__init__(root)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self.transform = transform
        self.target_transform = target_transform

        if split.lower() == "train":
            split = 0
        elif split.lower() == "valid":
            split = 1
        elif split.lower() == "test":
            split = 2
        else:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="valid" or split="test"')

        with open(os.path.join(self.root, self.base_folder, "list_eval_partition.txt"), "r") as f:
            splits = pandas.read_csv(f, delim_whitespace=True, header=None, index_col=0)

        with open(os.path.join(self.root, self.base_folder, "identity_CelebA.txt"), "r") as f:
            self.identity = pandas.read_csv(f, delim_whitespace=True, header=None, index_col=0)

        with open(os.path.join(self.root, self.base_folder, "list_bbox_celeba.txt"), "r") as f:
            self.bbox = pandas.read_csv(f, delim_whitespace=True, header=1, index_col=0)

        with open(os.path.join(self.root, self.base_folder, "list_landmarks_align_celeba.txt"), "r") as f:
            self.landmarks_align = pandas.read_csv(f, delim_whitespace=True, header=1)

        with open(os.path.join(self.root, self.base_folder, "list_attr_celeba.txt"), "r") as f:
            self.attr = pandas.read_csv(f, delim_whitespace=True, header=1)

        mask = (splits[1] == split)
        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(self.identity[mask].values)
        self.bbox = torch.as_tensor(self.bbox[mask].values)
        self.landmarks_align = torch.as_tensor(self.landmarks_align[mask].values)
        self.attr = torch.as_tensor(self.attr[mask].values)
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor')  # map from {-1, 1} to {0, 1}

        if label is not None:

            # select the correct attr
            assert label in CELEBA_ATTRIBUTES, 'The required label is not in Celeba attributes.'

            index_label = CELEBA_ATTRIBUTES.index(label)
            indices = np.where(self.attr[:, index_label] == 1)
            self.filename = self.filename[indices]
            self.identity = self.identity[indices]
            self.bbox = self.bbox[indices]
            self.landmarks_align = self.landmarks_align[indices]
            self.attr = self.attr[indices]

    def download(self):
        import zipfile

        for filename in self.file_list:
            fp = os.path.join(self.root, self.base_folder, filename)
            if not os.path.exists(fp):
                raise RuntimeError(f'File is missing: {fp}')
        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target

    def __len__(self):
        return len(self.attr)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
