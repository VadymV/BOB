"""
This module contains the DEAP dataset.
"""

from typing import Tuple, Optional

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants import DEAP_CHANNEL_LIST
from torcheeg.transforms import PickElectrode

PREPROCESSED_DATA_FOLDER_NAME = 'preprocessed_data'
TRAIN_TEST_SPLIT_FOLDER_NAME = 'train_test'
TRAIN_VAL_SPLIT_FOLDER_NAME = 'train_val'


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

_ONLINE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Contrastive(
        transforms.Compose([transforms.RandomMask(p=0.5, ratio=0.1),
                            transforms.RandomNoise(p=0.5)]),
        num_views=2)
])

_LABEL_TRANSFORM = transforms.Compose([
    transforms.Select('valence'),
    transforms.Binary(5.0),
])


class DEAP(DEAPDataset):
    """
    DEAP dataset.

    Attributes:
        root_path (str): A path to the folder containing the original data.
        io_path (str): A path to the folder containing the preprocessed data
    """

    def __init__(self, root_path: str, io_path: Optional[str]):
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
                                   offline_transform=_OFFLINE_TRANSFORM)

    def __getitem__(self, index: int) -> Tuple:
        """
        Return an item from the dataset.

        Args:
            index: The index of the item.

        Returns:
            A tuple of the item and its label.
        """
        info = self.read_info(index)
        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)

        baseline_index = str(info['baseline_id'])
        baseline = self.read_eeg(eeg_record, baseline_index)

        signal = _ONLINE_TRANSFORM(eeg=eeg, baseline=baseline)['eeg']
        label = _LABEL_TRANSFORM(y=info)['y']

        return signal, label
