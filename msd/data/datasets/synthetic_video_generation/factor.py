from abc import abstractmethod, ABC
from typing import Dict, Iterable, Union, List, Any, Optional

import numpy as np


class Factor(ABC):
    """
    Abstract base class for factors in a video generation model.
    Each factor has a name, a set of values, and a type (static or dynamic).
    The class provides methods to initialize the factor, retrieve its properties,
    and convert it to a dynamic factor, given dynamics.
    """
    def __init__(self, name: str, values: List[Any], factor_type: str, static_factor: Optional['StaticFactor'] = None):
        """
        Initialize the Factor with a name, values, factor type, and optional static factor.
        :param name: Name of the factor.
        :param values: List of values for the factor.
        :param factor_type: Type of the factor (e.g., 'static', 'dynamic').
        :param static_factor: Optional static factor for dynamic factors.
        """
        self._name = name
        self._values = values
        self._factor_type = factor_type
        self._static_factor = static_factor
        self._labels = np.arange(len(values))
        self._label2value = {l: v for l, v in enumerate(values)}
        self._value2label = {v: l for l, v in enumerate(values)}

    def encode(self, value: Any) -> int:
        """
        Encode a value to its corresponding label index.
        :param value: Value to be encoded.
        :return: Corresponding label index.
        """
        return self._label2value[value]

    def decode(self, label: int) -> Any:
        """
        Decode a label index to its corresponding value.
        :param label: Label index to be decoded.
        :return: Corresponding value.
        """
        return self._values[label]

    @property
    @abstractmethod
    def values_map(self):
        """
        Return a mapping of values to labels.
        :return: Mapping of values to labels.
        """
        pass

    @property
    def name(self):
        """
        Return the name of the factor.
        :return: Name of the factor.
        """
        return self._name

    @property
    def values(self):
        """
        Return the values of the factor.
        :return: Values of the factor.
        """
        return self._values

    @property
    def factor_type(self):
        """
        Return the type of the factor.
        :return: Type of the factor.
        """
        return self._factor_type

    @property
    def labels(self):
        """
        Return the labels of the factor.
        :return: Labels of the factor.
        """
        return self._labels

    @property
    def static_factor(self):
        """
        Return the static factor if it exists.
        :return: Static factor or None.
        """
        return self._static_factor

    def __len__(self):
        """
        Return the number of labels in the factor.
        :return: Number of labels.
        """
        return len(self.labels)

    def __repr__(self):
        """
        Return a string representation of the factor.
        :return: String representation of the factor.
        """
        return f"{self.factor_type.capitalize()}Factor(name={self.name}, values={self._label2value}, type={self.factor_type})"

    def __str__(self):
        """
        Return a string representation of the factor.
        :return: String representation of the factor.
        """
        return self.name

class StaticFactor(Factor):
    """
    Static factor with a fixed set of values.
    It can be converted to a dynamic factor with specified dynamics.
    """
    def __init__(self, name: str, values: List[Any], static_factor: Optional['StaticFactor'] = None):
        super().__init__(name, values, factor_type='static', static_factor=static_factor)

    @property
    def values_map(self):
        if self.static_factor is not None:
            label2value = self.static_factor._label2value
            return {label2value[v]: self._value2label[v] for v in self.values}
        else:
            return self._value2label

    def to_dynamic(self, dynamics: Dict[str, 'Sequence']):
        return DynamicFactor(f'{self.name}_dynamic', dynamics, self)

    def __getitem__(self, index):
        """
        Creates a new StaticFactor with the same name and a subset of values.
        :param index: Index or slice to select values.
        :return: A new StaticFactor with the selected values.
        """
        sliced_values = self.labels[index]
        return StaticFactor(self._name, sliced_values, self)


class DynamicFactor(Factor):
    """
    Dynamic factor with a set of values that can change over time.
    It is initialized with a dictionary of dynamics, where each key is a value
    and each value is a sequence of values that the factor can take over time.
    The class provides methods to retrieve the sequences and their corresponding values.
    """
    def __init__(self, name: str, dynamics: Dict[str, 'Sequence'], static_factor: Optional[StaticFactor] = None):
        """
        Initialize the DynamicFactor with a name, dynamics, and optional static factor.
        :param name: Name of the dynamic factor.
        :param dynamics: Dictionary of dynamics, where each key is a value and each value is a sequence.
        :param static_factor: Optional static factor for the dynamic factor.
        """
        values, sequences = zip(*dynamics.items())
        super().__init__(name, values, factor_type='dynamic', static_factor=static_factor)
        self.sequences = sequences

    @property
    def values_map(self):
        """
        Return a mapping of values to labels for the dynamic factor.
        :return: Mapping of values to labels.
        """
        return {v: self._value2label[v] for v in self.values}

    def __getitem__(self, index):
        """
        Retrieve the sequence at the given index.
        :param index: Index of the sequence to retrieve.
        :return: The sequence at the given index.
        """
        return self.sequences[index]

    def __repr__(self):
        return super().__repr__()[:-1] + f", dynamics={self.sequences})"

class Sequence(ABC):
    pass

class CyclicSequence(Sequence):
    """
    Repeats values in a fixed cycle over time.

    Example:
        CyclicSequence([1, 2, 3]) -> 1, 2, 3, 1, 2, 3, ...
    """
    def __init__(self, values: Union[np.ndarray, List]):
        """
        Initialize the CyclicSequence with a list of values.
        :param values: List of values to be repeated in a cycle.
        """
        self._values = values

    def __getitem__(self, index: Union[int, slice]):
        """
        Retrieve the value at the given index, wrapping around if necessary.
        :param index:
        :return:
        """
        if isinstance(index, slice):
            start = index.start or 0
            stop = index.stop
            step = index.step or 1
            if stop is None:
                raise ValueError("CyclicSequence slice must have a stop value")
            return [self._values[i % len(self._values)] for i in range(start, stop, step)]
        else:
            return self._values[index % len(self._values)]

    def __repr__(self):
        return f"Sequence({self._values})"

class HarmonicSequence(CyclicSequence):
    """
    Repeats a sequence forward and then backward (mirror loop).

    Example:
        HarmonicSequence([1, 2, 3]) -> 1, 2, 3, 2, 1, 2, 3, ...
    """
    def __init__(self, values: Union[np.ndarray, List]):
        """
        Initialize the HarmonicSequence with a list of values.
        :param values: List of values to be repeated in a harmonic manner.
        """
        values = np.concatenate((values, values[::-1][1:-1]))
        super().__init__(values)
