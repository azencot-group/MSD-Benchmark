from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Set, Tuple

import numpy as np
from torch.utils.data import Dataset

from msd.configurations.msd_component import MSDComponent


class AbstractReader(Dataset, ABC, MSDComponent):
    def __init__(self, classes: Dict[str, Dict[str, Any]], split: str):
        """
        Initialize the dataset reader.
        :param classes: Contains class information.
        :param split: Split to load ('train', 'val', or 'test').
        """
        self._classes = classes # {[index, type, n_classes, ignore, values] for class}
        self.split = split

    @property
    def classes(self) -> Dict[str, Dict[str, Any]]:
        return {k: v for k, v in self._classes.items() if not v['ignore']}

    @property
    def static_factors(self) -> Set[str]:
        """
        :return: Set of names of static factors.
        """
        return {k for k, v in self.classes.items() if v['type'] == 'static'}

    @property
    def dynamic_factors(self) -> Set[str]:
        """
        :return: Set of names of dynamic factors.
        """
        return {k for k, v in self.classes.items() if v['type'] == 'dynamic'}

    @property
    def class_dims(self) -> Dict[str, int]:
        """
        :return: Mapping of factor names to their number of unique classes.
        """
        return {k: v['n_classes'] for k, v in self.classes.items()}

    @property
    def names_map(self) -> Dict[str, Dict[int, str]]:
        """
        :return: Mapping of factor names to their class names.
        """
        return {k: {p: q for q, p in v['values'].items()} | {-1: 'unknown'} for k, v in self.classes.items()}

    @abstractmethod
    def __len__(self) -> int:
        """
        :return: Number of samples in the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: Union[int, slice, List]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Retrieve sample(s) from the dataset.

        :param index: Index of the sample to retrieve.
        :return: Tuple containing the data, static factors, and dynamic factors.
        """
        pass

    def __repr__(self) -> str:
        """
        :return: String representation of the dataset reader.
        """
        return f'{self.__class__.__name__}(split={self.split}, classes={set(self.classes.keys())})'

    def close(self):
        """
        Close the dataset if applicable.
        """
        pass
