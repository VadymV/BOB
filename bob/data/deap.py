"""
This module contains the DEAP dataset.
"""
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants import DEAP_CHANNEL_LIST
from torcheeg.model_selection import train_test_split_groupby_trial
from torcheeg.transforms import PickElectrode

from bob.data.transform import FeatureEngineering

PREPROCESSED_DATA_FOLDER_NAME = 'preprocessed_data'
TRAIN_TEST_SPLIT_FOLDER_NAME = 'train_test'
TRAIN_VAL_SPLIT_FOLDER_NAME = 'train_val'

ONLINE_TRANSFORM_WITH_CONTRAST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Contrastive(
        transforms.Compose([transforms.RandomMask(p=0.5, ratio=0.1),
                            transforms.RandomNoise(p=0.5)]),
        num_views=2)
])

ONLINE_TRANSFORM_FEATURE = transforms.Compose([
    FeatureEngineering(),
    transforms.ToTensor(),
])

ONLINE_TRANSFORM_TENSOR = transforms.Compose([
    transforms.ToTensor(),
])

_OFFLINE_TRANSFORM = transforms.Compose([
    PickElectrode(PickElectrode.to_index_list(
        ['FP1', 'AF3', 'F3', 'F7',
         'FC5', 'FC1', 'C3', 'T7',
         'CP5', 'CP1', 'P3', 'P7',
         'PO3', 'O1', 'FP2', 'AF4',
         'F4', 'F8', 'FC6', 'FC2',
         'C4', 'T8', 'CP6', 'CP2',
         'P4', 'P8', 'PO4', 'O2'], DEAP_CHANNEL_LIST)),
    transforms.To2d()
])

_LABEL_TRANSFORM = transforms.Compose([
    transforms.Select('valence'),
    transforms.Binary(5.0),
])

# The following variables could be set externally.
# However, within the scope of this project it is not necessary.
_NUM_WORKERS = 8 if torch.cuda.is_available() else 0
_PIN_MEMORY = True if torch.cuda.is_available() else False


class DEAP(DEAPDataset):
    """
    DEAP dataset.
    """

    def __init__(self, root_path: str, io_path: Optional[str],
                 online_transform: Optional[transforms.Compose]):
        """
        Initializes the instance based on specified paths.

        Args:
            root_path: A path to the folder containing the original data.
            io_path: A path to the folder containing the preprocessed data
        """
        super(DEAP, self).__init__(io_path=io_path,
                                   root_path=root_path,
                                   num_worker=6,
                                   chunk_size=1125,
                                   baseline_chunk_size=1125,
                                   num_baseline=1,
                                   label_transform=_LABEL_TRANSFORM,
                                   online_transform=online_transform,
                                   offline_transform=_OFFLINE_TRANSFORM)

    def get_loaders(self, project_path: str, batch_size: int):
        """
        Creates data loaders.
        Args:
            project_path: Path to the project folder.
            batch_size: A batch size.

        Returns:
            Train loader, validation loader, and test loader.
        """
        split_path_test = os.path.join(project_path,
                                       TRAIN_TEST_SPLIT_FOLDER_NAME)
        split_path_val = os.path.join(project_path,
                                      TRAIN_VAL_SPLIT_FOLDER_NAME)

        train_val_dataset, test_dataset = train_test_split_groupby_trial(
            dataset=self, split_path=split_path_test)
        train_dataset, val_dataset = train_test_split_groupby_trial(
            dataset=train_val_dataset, split_path=split_path_val)

        train_loader = DataLoader(train_dataset,
                                  num_workers=_NUM_WORKERS,
                                  pin_memory=_PIN_MEMORY,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset,
                                num_workers=_NUM_WORKERS,
                                pin_memory=_PIN_MEMORY,
                                batch_size=batch_size,
                                shuffle=False)
        test_loader = DataLoader(test_dataset,
                                 num_workers=_NUM_WORKERS,
                                 pin_memory=_PIN_MEMORY,
                                 batch_size=batch_size,
                                 shuffle=False)

        return train_loader, val_loader, test_loader
