import os, sys
import torchvision.datasets
import argparse

from bridge.data.celeba import CelebA

DEFAULT_DATA_PATH = './data/'

parser = argparse.ArgumentParser(description='Download data.')
parser.add_argument('--data', type=str, help='mnist or celeba')
parser.add_argument('--data_dir', type=str, help='downloading location', default=DEFAULT_DATA_PATH)

sys.path.append('..')


# SETTING PARAMETERS

def main():
    args = parser.parse_args()

    if args.data == 'mnist':
        super_root = os.path.join(args.data_dir, 'mnist')
        torchvision.datasets.MNIST(super_root, train=True, download=True)

    if args.data == 'celeba':
        super_root = os.path.join(args.data_dir, 'celeba')
        CelebA(super_root, split='train', download=True)


if __name__ == '__main__':
    main()
