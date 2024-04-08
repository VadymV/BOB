"""
Miscellaneous functions.
"""

import pickle
import random

import numpy as np
import torch
from torch.backends import cudnn


def set_seed(seed: int):
    """
    Sets the seed for reproducibility.
    Args:
        seed (int): The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(path_to_file: str) -> object:
    """
    Loads a serialized representation of a Python object.
    :param path_to_file: Path to a file.
    :return: Loaded object.
    """
    with open(path_to_file, "rb") as f:
        return pickle.load(f)
