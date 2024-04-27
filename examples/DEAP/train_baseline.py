"""
Trains a baseline model in a supervised way with binary labels
"""

import logging
import os

import torch.cuda
import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger
from torcheeg.models import ATCNet
from torcheeg.trainers import ClassifierTrainer
from yaml import SafeLoader

from bob.data.deap import PREPROCESSED_DATA_FOLDER_NAME, DEAP, \
    ONLINE_TRANSFORM_TENSOR
from bob.misc.misc import set_seed, set_logging, create_args, DEVICE

_PROJECT_NAME = 'BOB-Baseline'


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

    encoder = ATCNet(num_electrodes=28, num_classes=2, chunk_size=1125)

    trainer = ClassifierTrainer(model=encoder,
                                lr=config['learning_rate_baseline'],
                                weight_decay=config['weight_decay_baseline'],
                                num_classes=2,
                                accelerator=DEVICE)

    data_path = os.path.join(args.project_path, 'data_preprocessed_python')
    features_path = os.path.join(args.project_path,
                                 PREPROCESSED_DATA_FOLDER_NAME)
    dataset = DEAP(root_path=data_path, io_path=features_path,
                   online_transform=ONLINE_TRANSFORM_TENSOR)
    train_loader, val_loader, _ = \
        dataset.get_loaders(args.project_path,
                            config['batch_size_baseline'])
    trainer.fit(train_loader, val_loader, logger=wandb_logger,
                max_epochs=config['epochs_baseline'])

    checkpoint = f'{args.project_path}/model-baseline.ckpt'
    torch.save(trainer.model.state_dict(), checkpoint)

    wandb.finish()


if __name__ == '__main__':
    main()
