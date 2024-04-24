"""
Trains a linear model in a supervised way with binary labels (positivity or negativity) on top of the SSL model.

"""

import logging
import os

import torch.cuda
import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger
from torcheeg.trainers import ClassifierTrainer
from yaml import SafeLoader

from bob.data.deap import DEAP, PREPROCESSED_DATA_FOLDER_NAME, \
    ONLINE_TRANSFORM_TENSOR
from bob.misc.misc import set_seed, set_logging, create_args, DEVICE
from bob.models.atcnet_encoder import ATCNetEncoder
from bob.models.linear import Linear

_PROJECT_NAME = 'BOB-Linear'


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
    config_path = os.path.join(current_path, '../config.yaml')
    with open(config_path, encoding='utf8') as f:
        config = yaml.load(f, Loader=SafeLoader)

    wandb.init(project=_PROJECT_NAME)
    wandb_logger = WandbLogger(project=_PROJECT_NAME)

    encoder = ATCNetEncoder(num_electrodes=28, chunk_size=1125)
    checkpoint_encoder = f'{args.project_path}/model-ssl.ckpt'
    encoder.load_state_dict(torch.load(checkpoint_encoder))
    encoder.eval()

    model = Linear(encoder, num_features=32, num_classes=2)
    trainer = ClassifierTrainer(model=model,
                                lr=config['learning_rate_linear_model'],
                                weight_decay=config[
                                    'weight_decay_linear_model'],
                                num_classes=2,
                                accelerator=DEVICE)
    data_path = os.path.join(args.project_path, 'data_preprocessed_python')
    features_path = os.path.join(args.project_path,
                                 PREPROCESSED_DATA_FOLDER_NAME)
    dataset = DEAP(root_path=data_path, io_path=features_path,
                   online_transform=ONLINE_TRANSFORM_TENSOR)
    train_loader, val_loader, _ = \
        dataset.get_loaders(args.project_path,
                            config['batch_size_linear_model'])
    trainer.fit(train_loader, val_loader, logger=wandb_logger, max_epochs=1)

    checkpoint = f'{args.project_path}/model-linear.ckpt'
    torch.save(trainer.model.state_dict(), checkpoint)

    wandb.finish()


if __name__ == '__main__':
    main()
