from abc import ABC, abstractmethod
from itertools import combinations
from typing import List, Sequence

import numpy as np


class LatentDivider(ABC):
    """
    Abstract base class for latent space division strategies.

    A latent divider generates subsets of latent channels to be used
    in interventions for disentanglement evaluation.
    """

    @abstractmethod
    def divide(self, nF: int, nD: int) -> List[Sequence[int]]:
        """
        Divide the latent space into subsets.

        :param nF: Number of ground truth factors.
        :param nD: Dimension of the latent space.
        :return: List of latent index subsets (each subset is a sequence of integers).
        """
        pass

class ContiguousLatentDivider(LatentDivider):
    """
    Divides the latent space into contiguous, non-overlapping chunks of equal size.
    Assumes that the number of latent dimensions is divisible by the number of factors.
    """

    def divide(self, nF: int, nD: int) -> List[np.ndarray]:
        """
        Create contiguous chunks of size nD / nF.

        :param nF: Number of ground truth factors.
        :param nD: Latent dimension (must be divisible by nF).
        :return: List of np.ndarray, each containing the indices of a latent subset.
        """
        assert nD % nF == 0, "nD must be divisible by nF"
        nFD = int(nD / nF)
        return [np.arange(i * nFD, i * nFD + nFD) for i in range(nF)]


class DisjointLatentDivider(LatentDivider):
    """
    Generates all disjoint latent subsets within a depth range.

    Useful for exploring arbitrary combinations of latent indices.
    """

    def __init__(self, min_depth: int = None, max_depth: int = None):
        """
        :param min_depth: Minimum size of subsets (inclusive).
        :param max_depth: Maximum size of subsets (inclusive).
                          Defaults to all sizes up to nD - nF + 1.
        """
        self.min_depth = min_depth
        self.max_depth = max_depth or np.inf

    def divide(self, nF: int, nD: int) -> List[tuple[int, ...]]:
        """
        Generate all combinations of latent indices of sizes between min_depth and max_depth.

        :param nF: Number of ground truth factors.
        :param nD: Latent dimension.
        :return: List of tuples, each representing a subset of latent indices.
        """
        min_depth = self.min_depth or 1
        max_depth = min(self.max_depth, nD - nF + 1)
        subsets = [list(combinations(range(nD), i)) for i in range(min_depth, max_depth + 1)]
        subsets = [item for sublist in subsets for item in sublist]
        return subsets
