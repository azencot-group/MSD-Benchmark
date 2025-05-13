from abc import ABC, abstractmethod
from typing import Any, Iterable

import torch
from torch import nn

from msd.configurations.msd_component import MSDComponent


class AbstractModel(ABC, nn.Module, MSDComponent):
    """
    Abstract base class for all models used in the benchmark.

    Subclasses must implement encoding, decoding, latent manipulation, and
    forward pass behavior. Utility functions for preprocessing and postprocessing
    are optionally overridable.
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def to(self, device: Any) -> 'AbstractModel':
        """
        Move the model and internal buffers to the specified device.

        :param device: Torch device string or object.
        :return: The model on the new device.
        """
        self.device = torch.device(device)
        return super().to(device)

    @abstractmethod
    def latent_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the latent representation of the input.

        :param x: Input sample(s), typically shape [B, ...].
        :return: Latent vector(s), shape [B, D].
        """
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sample(s) into latent space.

        :param x: Input tensor.
        :return: Latent tensor.
        """
        pass

    @abstractmethod
    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector(s) back into observation space.

        :param Z: Latent tensor.
        :return: Reconstructed tensor.
        """
        pass


    @abstractmethod
    def sample(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Sample from the latent manifold.

        :param Z: Latent batch.
        :return: Sampled latent tensor.
        """
        pass

    @abstractmethod
    def swap_channels(self, Z1: torch.Tensor, Z2: torch.Tensor, C: Iterable[int]) -> torch.Tensor:
        """
        Perform latent channel swapping across two latent vectors.

        :param Z1: First latent batch [B, ...].
        :param Z2: Second latent batch [B, ...].
        :param C: List of indices to swap.
        :return: New tensor with channels C swapped between Z1 and Z2.
        """
        pass

    @abstractmethod
    def latent_dim(self) -> int:
        """
        Return the dimensionality of the latent space.

        :return: Integer dimension of latent space.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the full model forward pass.

        :param x: Input tensor.
        :return: Output tensor.
        """
        pass


    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optional preprocessing of input before passing to model.

        :param x: Raw input tensor.
        :return: Transformed input tensor.
        """
        return x

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optional postprocessing of model output.

        :param x: Model output tensor.
        :return: Transformed output tensor.
        """
        return x