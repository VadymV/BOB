"""
This module splits data into train, validation, and test sets.
"""

import argparse
import logging
import os
import shutil
from collections import OrderedDict

import torch.cuda
from torcheeg.model_selection import train_test_split_groupby_trial

from bob.datasets.deap import DEAP, PREPROCESSED_DATA_FOLDER_NAME, \
    TRAIN_TEST_SPLIT_FOLDER_NAME, TRAIN_VAL_SPLIT_FOLDER_NAME
from bob.misc.misc import set_seed

_DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
_NUM_WORKERS = 8 if torch.cuda.is_available() else 0
_PIN_MEMORY = True if torch.cuda.is_available() else False


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
    return parser


def main():
    """
    Splits data into train/validation/test sets.
    """
    parser = create_args()
    args = parser.parse_args()
    logging.info('Args: %args', args)
    data_path = os.path.join(args.project_path, 'data_preprocessed_python')
    set_seed(1)

    # .torcheeg folder is a working directory
    torcheeg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 '.torcheeg/')
    if os.path.exists(torcheeg_path):
        shutil.rmtree(torcheeg_path)

    # Create pre-processed data
    dataset = DEAP(root_path=data_path, io_path=None)

    # Create train and test indices
    train_val_dataset, _ = train_test_split_groupby_trial(
        dataset=dataset)

    # Create train and validation indices
    _, _ = train_test_split_groupby_trial(
        dataset=train_val_dataset)

    found_folders = [f for f in os.scandir(torcheeg_path) if f.is_dir()]

    folders_by_creation_time = dict()
    for folder in found_folders:
        folders_by_creation_time[os.path.getctime(folder.path)] = folder.path

    sorted_folders = OrderedDict(sorted(folders_by_creation_time.items()))

    for i, v in enumerate(sorted_folders.values(), 1):
        if i == 1:
            # Pre-processed data were created at first
            output = PREPROCESSED_DATA_FOLDER_NAME
        elif i == 2:
            # Train test split was created at first
            output = TRAIN_TEST_SPLIT_FOLDER_NAME
        elif i == 3:
            # Train validation split was created at first
            output = TRAIN_VAL_SPLIT_FOLDER_NAME
        else:
            raise ValueError('Check the version of the torcheeg library')

        output_path = os.path.join(args.project_path, output)
        destination = shutil.copytree(v, output_path)
        logging.info('Copied data to %s', destination)


if __name__ == '__main__':
    main()
