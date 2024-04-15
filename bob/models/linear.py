"""
Linear model

"""

import torch


class Linear(torch.nn.Module):
    """
    Linear model.
    """
    def __init__(self, encoder: torch.nn.Module, num_features, num_classes):
        """
        Initialize.
        Args:
            encoder: The encoder model.
            num_features: The number of input features.
            num_classes: The number of classes.
        """
        super(Linear, self).__init__()
        self.encoder = encoder
        self.fc = torch.nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.encoder(x)
        return self.fc(x)
