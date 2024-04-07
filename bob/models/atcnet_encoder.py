"""
This module contains the encoder for the ATCNet model.
"""

import torch
from torch import nn
from torcheeg.models.transformer import ATCNet


class ATCNetEncoder(ATCNet):
    """
    This module contains the encoder for the ATCNet model.

    Attributes:
        num_electrodes (int): The number of electrodes.
        chunk_size (int): The chunk size.
    """

    def __init__(self, num_electrodes: int = 28, chunk_size: int = 1125):
        """
        Initializes the ATCNetEncoder module.
        Args:
            num_electrodes: The number of electrodes.
            chunk_size: The chunk size.
        """
        super(ATCNetEncoder, self).__init__(num_electrodes=num_electrodes,
                                            chunk_size=chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the ATCNetEncoder module.

        Args:
            x: The input tensor of shape (batch_size, num_electrodes,
            chunk_size).

        Returns:
            The output tensor of shape (batch_size, features).
        """
        x = self.conv_block(x)
        x = x[:, :, -1, :]
        x = x.permute(0, 2, 1)

        sw_concat = None
        for i in range(self.num_windows):
            st = i
            end = x.shape[1] - self.num_windows + i + 1
            x2 = x[:, st:end, :]
            x2_ = self.get_submodule("msa_drop" + str(i))(x2)
            x2 = torch.add(x2, x2_)

            for j in range(self.tcn_depth):
                out = self.get_submodule("tcn" + str((i + 1) * j))(x2)
                if x2.shape[1] != out.shape[1]:
                    x2 = self.get_submodule("re" + str(i))(x2)
                x2 = torch.add(x2, out)
                x2 = nn.ELU()(x2)
            x2 = x2[:, -1, :]
            if i == 0:
                sw_concat = x2
            else:
                sw_concat = sw_concat.add(x2)

        x = sw_concat / self.num_windows
        return x
