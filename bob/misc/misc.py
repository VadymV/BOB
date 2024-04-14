"""
Miscellaneous functions.
"""
import logging
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


def set_logging(log_dir: str) -> None:
    """
    Creates a logging file.
    :param log_dir: a logging directory
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f'{log_dir}/logs.log'),
            logging.StreamHandler()
        ]
    )

    logging.info('Logging directory is %s', log_dir)


def load_pickle(path_to_file: str) -> object:
    """
    Loads a serialized representation of a Python object.
    :param path_to_file: Path to a file.
    :return: Loaded object.
    """
    with open(path_to_file, "rb") as f:
        return pickle.load(f)
