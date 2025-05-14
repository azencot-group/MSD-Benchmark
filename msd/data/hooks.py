from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from msd.configurations.msd_component import MSDComponent


class AbstractPreprocessHook(ABC, MSDComponent):
    """
    Abstract base class for preprocessing hooks that modify data during retrieval.
    Each hook must implement the `apply` method to transform the data.
    """
    def __init__(self):
        pass

    def __call__(self, data: Any) -> Any:
        """
        Call the preprocessing hook to apply it to the data.

        :param data: The data to preprocess.
        :return: Preprocessed data.
        """
        return self.apply(data)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def apply(self, data: Any) -> Any:
        """
        Apply the preprocessing hook to the data.

        :param data: The data to preprocess.
        :return: Preprocessed data.
        """
        pass

class Transpose(AbstractPreprocessHook):
    """
    Transpose the data along the specified axes.
    """
    def __init__(self, axes: tuple):
        super().__init__()
        self.axes = np.array(axes)

    def apply(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) == len(self.axes) + 1:
            return data.transpose(np.concatenate(([0], self.axes+1)))
        else:
            return data.transpose(self.axes)

    def __repr__(self):
        return f"{self.__class__.__name__}(axes={self.axes})"

class Normalize(AbstractPreprocessHook):
    """
    Normalize the data to a specified range.
    """
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, data_min=None, data_max=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.data_min = data_min
        self.data_max = data_max

    def apply(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        data_min = self.data_min if self.data_min is not None else np.min(data)
        data_max = self.data_max if self.data_max is not None else np.max(data)
        return (data - data_min) / (data_max - data_min) * (self.max_val - self.min_val) + self.min_val

    def __repr__(self):
        return f"{self.__class__.__name__}(min_val={self.min_val}, max_val={self.max_val}, data_min={self.data_min}, data_max={self.data_max})"

class ToNumpy(AbstractPreprocessHook):
    def __init__(self, dtype: str = 'float32'):
        super().__init__()
        self.dtype = np.dtype(dtype)
    """
    Convert the data to a NumPy array.
    """
    def apply(self, data: Any) -> np.ndarray:
        return np.array(data, dtype=self.dtype).squeeze()
