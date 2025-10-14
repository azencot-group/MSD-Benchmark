from typing import Any, Dict, Iterable, List, Union

from msd.data.datasets.synthetic_video_generation.factor import Factor
from msd.data.datasets.synthetic_video_generation.factor_space import FactorSpace


class StateMapper:
    """
    A class to map states to data points based on factors and their labels.
    This class provides methods to retrieve data points corresponding to given factor values.
    """
    def __init__(self, factors: List[Factor], data: Iterable[Any], labels: Iterable[Iterable[int]]):
        """
        Initialize the StateMapper with factors, data, and labels.
        :param factors: List of factors.
        :param data: Data to be mapped.
        :param labels: Labels corresponding to the data.
        """
        self.factors = FactorSpace(factors)
        self.data = data
        self.labels = labels
        self.index_map = {tuple(l): i for i, l in enumerate(labels)}
        self.state_space = self.factors.combinations()

    def __getitem__(self, factors: Dict[str, int]) -> Any:
        """
        Retrieve data point corresponding to given factor values.

        :param factors: A dictionary with keys as factor names and values as their corresponding values.
        :return: Corresponding data.
        """
        return self.get(factors)

    def get(self, factors: Union[int, Dict[str, int]]) -> Any:
        """
        Retrieve data point corresponding to given factor values.

        :param factors: A dictionary with keys as factor names and values as their corresponding values.
        :return: Corresponding data.
        """
        if isinstance(factors, int):
            factors = self.state_space[factors]
        label = tuple(factors[k.name] for k in self.factors)
        return self.data[self.index_map[label]]
