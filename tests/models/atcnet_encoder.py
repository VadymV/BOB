"""
This module tests the ATCNetEncoder module.
"""

import pytest
import torch

from bob.models.atcnet_encoder import ATCNetEncoder


@pytest.fixture
def my_atcnet_encoder():
    """
    Creates an instance of the ATCNetEncoder class.
    Returns:
        An instance of the ATCNetEncoder class.
    """

    return ATCNetEncoder(num_electrodes=28, chunk_size=1125)


def test_atcnet_encoder(my_atcnet_encoder):
    assert my_atcnet_encoder(torch.randn(32, 1, 28, 1125)).shape == torch.Size(
        [32, 32])
