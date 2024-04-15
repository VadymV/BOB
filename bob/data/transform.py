"""
Transforms the EEG signal into a feature vector.
"""
from typing import Dict, Optional

import numpy as np
import torch
from torcheeg.transforms import EEGTransform


class FeatureEngineering(EEGTransform):
    """
    Transforms the EEG signal into a feature vector.

    The  EEG signal is divided into 10 equally spaced windows and the mean of
    each window for each channel is calculated.
    Finally, the feature vector is the concatenation of the 10 windows
    for each channel.
    """

    def __init__(self, apply_to_baseline: bool = False, windows: int = 10):
        super(FeatureEngineering, self).__init__(
            apply_to_baseline=apply_to_baseline)
        self.windows = windows

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Optional[np.ndarray] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            eeg (np.ndarray): The input EEG signals.
            baseline (np.ndarray, optional) : The corresponding baseline signal.

        Returns:
            dict: If baseline is passed and apply_to_baseline is set to True,
            then {'eeg': ..., 'baseline': ...}, else {'eeg': ...}.
            The output is represented by :obj:`torch.Tensor`.
        """
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> torch.Tensor:
        print(
            f'Transforming an image of shape {eeg.shape} to'
            f'a tensor of shape {(eeg.shape[0], eeg.shape[1] * self.windows)}'
        )
        eeg = np.array_split(eeg, self.windows, axis=-1)
        eeg = [np.mean(i, axis=-1) for i in eeg]
        eeg = np.concatenate(eeg, axis=-1)
        return eeg
