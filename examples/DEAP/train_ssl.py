"""
This module trains a model.
"""

import logging
import os

import torch.cuda
import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger
from torcheeg.trainers import BYOLTrainer
from yaml import SafeLoader

from bob.data.deap import DEAP, PREPROCESSED_DATA_FOLDER_NAME, \
    ONLINE_TRANSFORM_WITH_CONTRAST
from bob.misc.misc import set_seed, set_logging, create_args, DEVICE
from bob.models.atcnet_encoder import ATCNetEncoder

_PROJECT_NAME = 'BOB-SSL'


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
    config_path = os.path.join(current_path, '../../config.yaml')
    with open(config_path, encoding='utf8') as f:
        config = yaml.load(f, Loader=SafeLoader)

    wandb.init(project=_PROJECT_NAME)
    wandb_logger = WandbLogger(project=_PROJECT_NAME)

    encoder = ATCNetEncoder(num_electrodes=28, chunk_size=1125)

    trainer = BYOLTrainer(extractor=encoder,
                          extract_channels=32,
                          proj_channels=32,
                          proj_hid_channels=64,
                          accelerator=DEVICE)
    data_path = os.path.join(args.project_path, 'data_preprocessed_python')
    features_path = os.path.join(args.project_path,
                                 PREPROCESSED_DATA_FOLDER_NAME)
    dataset = DEAP(root_path=data_path, io_path=features_path,
                   online_transform=ONLINE_TRANSFORM_WITH_CONTRAST)
    train_loader, val_loader, _ = dataset.get_loaders(args.project_path,
                                                      config[
                                                          'batch_size_ssl'])
    trainer.fit(train_loader, val_loader, logger=wandb_logger, max_epochs=1)

    checkpoint = f'{args.project_path}/model-ssl.ckpt'
    torch.save(trainer.student_model.state_dict(), checkpoint)

    wandb.finish()


if __name__ == '__main__':
    main()
