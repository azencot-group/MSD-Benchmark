from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor


class Manipulator(ABC):
    """
    Abstract base class for latent space manipulation strategies.

    A manipulator defines how to generate a second latent vector for each original one,
    typically used in disentanglement evaluations via interventions.
    """

    @abstractmethod
    def get_samples(self, Z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Generate a second latent vector for each original in Z.

        :param Z: Latent representations of shape (N, D).
        :return: Tuple of:
            - indices used to select/manipulate the second sample (Tensor of shape (N,))
            - manipulated latent vectors (Tensor of shape (N, D))
        """
        pass

    @property
    @abstractmethod
    def manipulation_method(self) -> str:
        """
        Return the name of the manipulation strategy.

        :return: Name of the manipulation method.
        """
        pass

class SwapManipulator(Manipulator):
    """
    Randomly swaps latent vectors between samples to simulate interventions.
    """

    def get_samples(self, Z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Randomly permute the batch to create pairings for swapping.

        :param Z: Latent representations of shape (N, D).
        :return: Tuple of permutation indices and permuted latent vectors.
        """
        rindexes = torch.randperm(Z.shape[0])
        return rindexes, Z[rindexes]

    @property
    def manipulation_method(self) -> str:
        """
        :return: 'swap'
        """
        return 'swap'


class SampleManipulator(Manipulator):
    """
    Uses the model's sampling function to generate new latent vectors.
    """

    def __init__(self, model):
        """
        :param model: The model providing the `.sample(Z)` method.
        """
        self.model = model

    def get_samples(self, Z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Sample new latent vectors using the model.

        :param Z: Latent representations of shape (N, D).
        :return: Tuple of identity indices and sampled latent vectors.
        """
        S = self.model.sample(Z)
        return torch.arange(Z.shape[0]), S

    @property
    def manipulation_method(self) -> str:
        """
        :return: 'sample'
        """
        return 'sample'