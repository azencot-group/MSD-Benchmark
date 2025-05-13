import itertools as it
from typing import List, Union, Dict

from msd.data.datasets.synthetic_video_generation.factor import Factor


class FactorSpace:
    """
    A class representing a space of factors, each with a set of labels.
    The class provides methods to generate all possible combinations of factor labels,
    and retrieve specific factors by name or index.
    """
    def __init__(self, factors: List[Factor]):
        """
        Initialize the FactorSpace with a list of factors.
        :param factors:
        """
        self.factors = factors
        self._name2index = {f.name: i for i, f in enumerate(self.factors)}
        self._index2name = {i: f.name for i, f in enumerate(self.factors)}

    @property
    def names(self) -> List[str]:
        """
        Return the names of the factors in the FactorSpace.
        :return: List of factor names.
        """
        return [f.name for f in self.factors]

    def combinations(self) -> List[Dict[str, int]]:
        """
        Return all possible combinations of factor label indices.
        Each combination is a dict mapping factor names to label indices.
        """
        names = [f.name for f in self.factors]
        label_lists = [f.labels for f in self.factors]  # each is a 1D np.ndarray
        combos = it.product(*label_lists)
        return [dict(zip(names, combo)) for combo in combos]

    def merge(self, factor_space):
        """
        Merge two FactorSpaces.
        :param factor_space: Another FactorSpace to merge with.
        :return: Merged FactorSpace.
        """
        merged_factors = self.factors + factor_space.factors
        return FactorSpace(merged_factors)

    def __len__(self):
        return len(self.factors)

    def __getitem__(self, idx: Union[int, str]):
        if isinstance(idx, int):
            return self.factors[idx]
        elif isinstance(idx, str):
            return self.factors[self._name2index[idx]]
        else:
            raise TypeError(f"Index must be int or str, not {type(idx)}")


    def __or__(self, factor_space):
        """
        Merge two FactorSpaces using the | operator.
        :param factor_space: Another FactorSpace to merge with.
        :return: Merged FactorSpace.
        """
        return self.merge(factor_space)

    def __contains__(self, item):
        """
        Check if a factor is in the FactorSpace.
        :param item: Factor or factor name to check.
        :return: True if the factor is in the FactorSpace, False otherwise.
        """
        return item in self.factors or item in self._name2index

    def __repr__(self):
        return f"FactorSpace({self.factors})"
