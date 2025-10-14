from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from msd.configurations.msd_component import MSDComponent
from msd.data.hooks import AbstractPreprocessHook
from msd.data.readers.abstract_reader import AbstractReader


class DisentanglementDataset(Dataset, MSDComponent):
    """
    A dataset wrapper for disentanglement learning tasks.

    This class provides a flexible interface to load, preprocess, and access
    disentanglement datasets, separating static and dynamic factors of variation.

    It supports:
    - Supervised and unsupervised modes.
    - Preprocessing hooks to apply transformations (e.g., normalization, augmentation).
    - Optional conversion of factor labels to human-readable names.
    - Access to metadata, including factor names, dimensions, and mappings.

    Attributes:
    ----------
    reader : AbstractReader
        An instance of AbstractReader used to load data and labels.
    preprocess_hooks : List[AbstractPreprocessHook]
        A list of preprocessing hooks applied sequentially to the data.
    _supervised : bool
        Whether the dataset is operating in supervised mode.
    _return_names : bool
        Whether to return human-readable factor names instead of raw labels.

    Methods:
    -------
    supervised:
        Property returning the supervised mode status.
    apply_hooks(data):
        Applies all registered preprocessing hooks to the input data.
    _label2name(labels):
        Converts numeric labels to their corresponding string names.
    __getitem__(index):
        Retrieves sample(s) by index, applies preprocessing, and returns data
        along with labels if supervised.
    classes:
        Returns the factor-to-property mapping.
    class_dims:
        Returns the factor-to-class-dimension mapping.
    names_map:
        Returns the factor-to-name-mapping.
    static_factors:
        Returns the set of static factor names.
    dynamic_factors:
        Returns the set of dynamic factor names.
    """
    def __init__(self,
                 reader: AbstractReader,
                 preprocess_hooks: List[AbstractPreprocessHook] = None,
                 supervised: bool = False,
                 return_names: bool = False):
        """
        Initialize the dataset.

        :param reader: An instance of AbstractReader for reading data.
        :param preprocess_hooks: List of preprocessing hooks to apply.
        :param supervised: Whether the dataset is in supervised mode.
        :param return_names: Whether to return factor names along with their values.
        """
        self.reader = reader
        self.preprocess_hooks = preprocess_hooks if preprocess_hooks is not None else []
        self._supervised = supervised
        self._return_names = return_names

    @property
    def supervised(self) -> bool:
        """
        :return: Whether the dataset is in supervised mode.
        """
        return self._supervised

    def apply_hooks(self, data: Any) -> Any:
        """
        Apply all registered preprocessing hooks to the data.

        :param data: The data to preprocess.
        :return: Preprocessed data.
        """
        for hook in self.preprocess_hooks:
            data = hook(data)
        return data

    def _label2name(self, labels: Dict[str, np.ndarray]) -> Dict[str, str]:
        """
        Convert labels to their corresponding names.

        :param labels: Dictionary of labels.
        :return: Dictionary of labels with names.
        """
        return {
            k: self.names_map[k][v.item()] if np.isscalar(v) or v.shape == () else [self.names_map[k][vi] for vi in v]
            for k, v in labels.items()
        }

    def get(self, index: Union[int, slice, List[int]], supervised, return_names) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """
        Retrieves sample(s) by index, applies preprocessing, and returns data along with labels if supervised.
        :param index: Index of the sample(s) to retrieve.
        :param supervised: Whether to return labels (supervised) or not (unsupervised).
        :param return_names: Whether to return class names instead of labels.
        :return: Tuple of (data, static_factors, dynamic_factors) if supervised, otherwise just data.
        """
        x, ys, yd = self.reader[index]
        x = self.apply_hooks(x)
        if not supervised:
            return x
        if return_names:
            ys = self._label2name(ys)
            yd = self._label2name(yd)
        return x, ys, yd

    def __getitem__(self, index: Union[int, slice, List[int]]) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """
        Retrieves sample(s) by index, applies preprocessing, and returns data along with labels if supervised.
        :param index: Index of the sample(s) to retrieve.
        :return: Tuple of (data, static_factors, dynamic_factors) if supervised, otherwise just data.
        """
        return self.get(index, self._supervised, self._return_names)

    def __len__(self) -> int:
        """
        :return: Number of samples in the dataset.
        """
        return len(self.reader)

    @property
    def classes(self) -> Dict[str, Dict]:
        """
        :return: Mapping of factor names to their properties.
        """
        return self.reader.classes

    @property
    def class_dims(self) -> Dict[str, int]:
        """
        :return: Mapping of factor names to their number of unique classes.
        """
        return self.reader.class_dims

    @property
    def names_map(self) -> Dict[str, Dict[int, str]]:
        """
        :return: Mapping of factor names to their class names.
        """
        return self.reader.names_map

    @property
    def static_factors(self) -> Set[str]:
        """
        :return: Set of names of static factors.
        """
        return self.reader.static_factors

    @property
    def dynamic_factors(self) -> Set[str]:
        """
        :return: Set of names of dynamic factors.
        """
        return self.reader.dynamic_factors

    @property
    def sample_shape(self) -> Tuple[int, ...]:
        """
        :return: Shape of a single data sample (excluding batch size).
        """
        return self.get(0, supervised=False, return_names=False).shape

    def __repr__(self) -> str:
        """
        :return: String representation of the dataset.
        """
        return f'{self.__class__.__name__}(split={self.reader.split}, classes={set(self.classes.keys())}, supervised={self._supervised}, return_names={self._return_names})'
