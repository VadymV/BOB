"""
This module trains a model.
"""

import argparse
import logging
import os

import torch.cuda
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torcheeg.model_selection import train_test_split_groupby_trial
from torcheeg.trainers import BYOLTrainer

import wandb
from bob.datasets.deap import DEAP
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
    parser.add_argument('--read_from_cache',
                        action='store_true',
                        help='If selected, assure that the preprocessed data '
                             'are available in folders: "features", '
                             '"split_test" and "split_val".')
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
    logging.info('Args: %args', args)
    data_path = os.path.join(args.project_path, 'data_preprocessed_python')

    if args.read_from_cache:
        features_path = os.path.join(args.project_path, 'features')
        split_path_test = os.path.join(args.project_path, 'split_test')
        split_path_val = os.path.join(args.project_path, 'split_val')
        logging.info('Using existing features from %s', features_path)
        logging.info('Using existing test split from %s', split_path_test)
        logging.info('Using existing val split from %s', split_path_val)
    else:
        features_path = None
        split_path_test = None
        split_path_val = None

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
