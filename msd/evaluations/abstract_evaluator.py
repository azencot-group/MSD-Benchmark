from abc import ABC, abstractmethod

import pandas as pd

from msd.configurations.config_initializer import ConfigInitializer
from msd.configurations.msd_component import MSDComponent
from msd.evaluations.evaluation_manager import EvaluationManager


class AbstractEvaluator(ABC, MSDComponent):
    """
    Abstract base class for evaluation modules.

    Subclasses should implement:
    - `eval(epoch)` for running the evaluation and returning results
    - `name()` for identifying the evaluator

    Evaluators typically access the model and logger via the EvaluationManager.
    """

    def __init__(self, initializer: ConfigInitializer, dataset_type: str, evaluation_manager: EvaluationManager):
        """
        :param initializer: ConfigInitializer instance.
        :param dataset_type: One of 'train', 'val', or 'test'.
        :param evaluation_manager: The managing EvaluationManager instance.
        """
        self.initializer = initializer
        self.config = self.initializer.config
        self.dataset_type = dataset_type
        self.evaluation_manager = evaluation_manager
        self.model = self.evaluation_manager.model
        self.logger = self.evaluation_manager.logger
        self.device = self.initializer.config.device

    @abstractmethod
    def eval(self, epoch: int) -> tuple[pd.Series, pd.DataFrame]:
        """
        Run the evaluation for a given epoch.

        :param epoch: The training epoch to evaluate.
        :return: A tuple of (summary Series, detailed DataFrame).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: The string identifier for this evaluator.
        """
        pass
