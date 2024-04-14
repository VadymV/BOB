"""
This module trains a model.
"""

import argparse
import logging
import os

import torch.cuda
import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torcheeg.model_selection import train_test_split_groupby_trial
from torcheeg.trainers import BYOLTrainer
from yaml import SafeLoader

from bob.datasets.deap import DEAP, PREPROCESSED_DATA_FOLDER_NAME, \
    TRAIN_TEST_SPLIT_FOLDER_NAME, TRAIN_VAL_SPLIT_FOLDER_NAME
from bob.misc.misc import set_seed, set_logging
from bob.models.atcnet_encoder import ATCNetEncoder

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
    parser.add_argument('--seed',
                        type=int,
                        help='Seed for reproducible results.',
                        default=1)
    return parser


def main():
    """
    Trains a model.
    """
    parser = create_args()
    args = parser.parse_args()

    set_logging(args.project_path)
    set_seed(args.seed)
    logging.info('Args: %s', args)

    current_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(current_path, 'config.yaml')
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    data_path = os.path.join(args.project_path, 'data_preprocessed_python')

    features_path = os.path.join(args.project_path,
                                 PREPROCESSED_DATA_FOLDER_NAME)
    split_path_test = os.path.join(args.project_path,
                                   TRAIN_TEST_SPLIT_FOLDER_NAME)
    split_path_val = os.path.join(args.project_path,
                                  TRAIN_VAL_SPLIT_FOLDER_NAME)

    dataset = DEAP(root_path=data_path, io_path=features_path)

    train_val_dataset, test_dataset = train_test_split_groupby_trial(
        dataset=dataset, split_path=split_path_test)
    train_dataset, val_dataset = train_test_split_groupby_trial(
        dataset=train_val_dataset, split_path=split_path_val)

    train_loader = DataLoader(train_dataset,
                              num_workers=_NUM_WORKERS,
                              pin_memory=_PIN_MEMORY,
                              batch_size=512,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            num_workers=_NUM_WORKERS,
                            pin_memory=_PIN_MEMORY,
                            batch_size=512,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             num_workers=_NUM_WORKERS,
                             pin_memory=_PIN_MEMORY,
                             batch_size=512,
                             shuffle=False)

    wandb.init(project='BOB')
    wandb_logger = WandbLogger(project='BOB')

    encoder = ATCNetEncoder(num_electrodes=28, chunk_size=1125)

    trainer = BYOLTrainer(extractor=encoder,
                          extract_channels=32,
                          proj_channels=32,
                          proj_hid_channels=64,
                          accelerator=_DEVICE)
    trainer.fit(train_loader, val_loader, logger=wandb_logger, max_epochs=500)

    checkpoint = f'{args.project_path}/model.ckpt'
    torch.save(trainer.student_model.state_dict(), checkpoint)

    wandb.finish()


if __name__ == '__main__':
    main()
