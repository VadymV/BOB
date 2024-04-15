"""
Miscellaneous functions.
"""
import argparse
import logging
import pickle
import random

import numpy as np
import torch
from torch.backends import cudnn

DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'



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
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/logs.log'),
            logging.StreamHandler()
        ]
    )

    logging.info('Logging directory is %s', log_dir)


def create_args() -> argparse.ArgumentParser:
    """
    Creates the argument parser.

    Returns:
        An argument parser.
    """
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--project_path',
                        type=str,
                        help='A path to the folder containing the DEAP data '
                             'called "data_preprocessed_python"')
    parser.add_argument('--seed',
                        type=int,
                        help='Seed for reproducible results.',
                        default=1)
    return parser


def load_pickle(path_to_file: str) -> object:
    """
    Loads a serialized representation of a Python object.
    :param path_to_file: Path to a file.
    :return: Loaded object.
    """
    with open(path_to_file, 'rb') as f:
        return pickle.load(f)
