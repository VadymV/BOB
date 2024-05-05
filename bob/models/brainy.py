"""
Definition of the Brainy model.
"""

import numpy as np
import torch
from torch import nn, Tensor
from transformers import CLIPProcessor, CLIPModel, BatchEncoding

from bob.models.atcnet_encoder import ATCNetEncoder


class ModalityEncoder(nn.Module):
    """
    Base class for modality encoders.

    Attributes:
        identifier (str): The identifier of the encoder as a string.
        encoder (nn.Module): The encoder module.
        feature_dim (int): The output dimension of the encoder.
    """

    def __init__(self) -> None:
        """
        Initializes a ModalityEncoder object.
        """
        super(ModalityEncoder, self).__init__()
        self.identifier = None
        self.encoder = None
        self.feature_dim = None

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class ATCNetModel(ModalityEncoder):
    """
    The ATCNet model.

    Attributes:
        encoder (nn.Module): The encoder module.
        feature_dim (int): The output dimension of the encoder.
        identifier (str): The identifier of the encoder as a string.
    """

    def __init__(self,
                 num_electrodes: int,
                 chunk_size: int,
                 identifier: str) -> None:
        """
        Initializes an ATCNetModel object.

        :param num_electrodes: Number of electrodes
        :param chunk_size: The number of recordings within one chunk
        :param identifier: The identifier of the encoder as a string
        """
        super(ATCNetModel, self).__init__()

        # Image encoder:
        self.encoder = ATCNetEncoder(num_electrodes=num_electrodes,
                                     chunk_size=chunk_size)
        self.feature_dim = 32
        self.identifier: str = identifier

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return x


class CLIPImageEncoder(ModalityEncoder):
    """
    CLIP image encoder that is based on a ViT.
    The encoder's parameters are "frozen".
    The output (feature) dimension is 1024.

    Attributes:
        identifier (str): The identifier of the encoder as a string
    """

    def __init__(self, identifier: str):
        """
        Initializes a CLIPImageEncoder object.

        Args:
            identifier: The identifier of the encoder.
        """
        super(CLIPImageEncoder, self).__init__()
        self.identifier: str = identifier
        self.encoder = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.freeze_params()
        self.encoder.eval()

        # Output from the 'pooler_output' layer
        self.feature_dim = 1024

    def __process_input(self, images: Tensor) -> BatchEncoding:
        return self.processor(text=[""], images=images, return_tensors="pt",
                              padding=True)

    def freeze_params(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        inputs = self.__process_input(x)
        x = self.encoder(**inputs).vision_model_output.pooler_output
        return x


class ProjectionHead(nn.Module):
    """
    Projection head maps the output of the encoder to the latent space.

    Attributes:
        projection: Linear layer.
    """

    def __init__(
            self,
            input_dim: int,
            projection_dim: int,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, projection_dim, bias=False)

    def forward(self, x):
        projected = self.projection(x)
        return projected


class ClipLoss(nn.Module):
    """
    CLIP loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits_per_image, logits_per_brain):
        device = logits_per_image.device
        labels = torch.arange(logits_per_image.shape[0], device=device,
                              dtype=torch.long)

        total_loss = (
                             nn.functional.cross_entropy(logits_per_image,
                                                         labels) +
                             nn.functional.cross_entropy(logits_per_brain,
                                                         labels)
                     ) / 2

        return total_loss


class BrainyModel(nn.Module):
    """
    Brainy model.

    Attributes:
        image_encoder: Image encoder.
        brain_encoder: Brain encoder.
        loss: Loss function.
        brain_projection: Brain projection head.
        image_projection: Image projection head.
        logit_scale: A trainable parameter to scale logits.
        identifier: Identifier of the model as a string.
    """

    def __init__(self,
                 image_encoder: ModalityEncoder,
                 brain_encoder: ModalityEncoder,
                 loss: ClipLoss,
                 image_embedding_dim: int,
                 brain_embedding_dim: int,
                 projection_embedding: int,
                 identifier: str,
                 ) -> None:
        """
        Initializes a BrainyModel object.
        Args:
            image_encoder: Image encoder.
            brain_encoder: Brain encoder.
            loss: Loss function.
            image_embedding_dim: Image embedding dimension.
            brain_embedding_dim: Brain embedding dimension.
            projection_embedding: Projection embedding dimension.
            identifier: Identifier of the model as a string.
        """
        super(BrainyModel, self).__init__()

        self.image_encoder = image_encoder
        self.brain_encoder = brain_encoder
        self.loss = loss,
        self.brain_projection = ProjectionHead(brain_embedding_dim,
                                               projection_embedding)
        self.image_projection = ProjectionHead(image_embedding_dim,
                                               projection_embedding)
        self.identifier = identifier
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image_input: Tensor) -> Tensor:
        """
        Encodes image data into a projection space.

        Args:
            image_input: Image representation.

        Returns:
            An encoded image in a projection space.
        """
        image_features = self.image_encoder(image_input)
        image_projection = self.image_projection(image_features)

        return image_projection

    def encode_brain(self, brain_input: Tensor) -> Tensor:
        """
        Encodes brain data into a projection space.

        Args:
            brain_input: Brain signals representation.

        Returns:
            Brain features in a projection space.
        """
        brain_features = self.brain_encoder(brain_input)
        brain_projection = self.brain_projection(brain_features)

        return brain_projection

    def forward(self, brain_signals: Tensor, images: Tensor) -> (
            Tensor, Tensor):
        """
        Forward pass of the model.

        Args:
            brain_signals: Brain signals.
            images: Image data.

        Returns:
            Image logits and brain logits.
        """
        brain_features = self.encode_brain(brain_signals)
        image_features = self.encode_image(images)

        # Normalized features.
        brain_features = brain_features / brain_features.norm(dim=1,
                                                              keepdim=True)
        image_features = image_features / image_features.norm(dim=1,
                                                              keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ brain_features.t()
        logits_per_brain = logits_per_image.t()

        return logits_per_image, logits_per_brain
