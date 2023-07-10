from bridge.runners.ipf import IPFTreeSequential
import hydra
import os
import sys
import torch

sys.path.append('..')


# SETTING PARAMETERS

@hydra.main(config_path="./conf", config_name="config")
def main(args):
    print('Directory: ' + os.getcwd())
    print('Cuda version: ', torch.version.cuda)
    ipfTree = IPFTreeSequential(args)
    tree = ipfTree.tree
    print(tree)
    ipfTree.populate_tree()
    ipfTree.test_model()


if __name__ == '__main__':
    main()
