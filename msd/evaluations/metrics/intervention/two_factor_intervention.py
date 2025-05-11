"""
Two-factor intervention evaluators for disentanglement benchmarking.

These evaluators group factors into two categories — 'static' and 'dynamic' —
and measure disentanglement by applying latent space interventions to each group.
"""

from abc import ABC
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING, Iterable

from torch import Tensor

from msd.evaluations.metrics.intervention.intervention_evaluator import InterventionEvaluator
from msd.evaluations.latent_exploration.latent_manipulator import SampleManipulator, SwapManipulator

if TYPE_CHECKING:
    from msd.configurations.config_initializer import ConfigInitializer
    from msd.evaluations.evaluation_manager import EvaluationManager


class TwoFactorIntervention(InterventionEvaluator, ABC):
    """
    Base class for two-factor (static vs dynamic) intervention-based evaluation.

    This evaluator aggregates learned mappings into two super-factors: 'static' and 'dynamic'.
    Each group is evaluated by intervening on the corresponding subset of latent channels.

    :param initializer: ConfigInitializer object for dataset/model setup.
    :param dataset_type: Dataset split ('train', 'val', or 'test').
    :param evaluation_manager: The EvaluationManager coordinating the evaluation.
    :param n_samples: Number of samples to use for evaluation.
    """

    def __init__(
        self,
        initializer: 'ConfigInitializer',
        dataset_type: str,
        evaluation_manager: 'EvaluationManager',
        n_samples: Optional[int]):
        super().__init__(initializer, dataset_type, evaluation_manager, n_samples)

    def _get_mapping(
            self,
            epoch: int,
            batch: Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]
    ) -> Dict[str, Iterable]:
        """
        Combines learned factor-to-subset mapping into two-factor groupings.

        :param epoch: Current epoch number.
        :param batch: A batch of input tensors and label dictionaries.
        :return: Mapping from 'static' and 'dynamic' to lists of latent dimensions.
        """
        mapping = self.latent_explorer.get_map(epoch, batch)
        expected = self._expected_map()
        two_factor_map = {s: tuple(set().union(*[mapping[_v] for _v in v if _v in mapping])) for s, v in expected.items()}
        return two_factor_map

    def _expected_map(self) -> Dict[str, List[str]]:
        """
        Computes the canonical grouping of dataset factors into 'static' and 'dynamic'.

        :return: Dictionary with 'static' and 'dynamic' keys mapping to lists of factor names.
        """
        return {s: [f for f, v in self.dataset.classes.items() if v['type'] == s] for s in ['dynamic', 'static']}

    @property
    def factor_method(self) -> str:
        """
        :return: Identifier of the factor grouping method.
        """
        return 'two_factor'


class TwoFactorSwap(TwoFactorIntervention):
    """
    Two-factor evaluator using latent channel swapping between samples.

    Evaluates disentanglement by swapping latent subsets corresponding to
    'static' or 'dynamic' groupings and checking for prediction consistency.
    """

    def __init__(
        self,
        initializer: 'ConfigInitializer',
        dataset_type: str,
        evaluation_manager: 'EvaluationManager',
        n_samples: Optional[int] = None
    ) -> None:
        super().__init__(initializer, dataset_type, evaluation_manager, n_samples)

    def _init_manipulator(self) -> SwapManipulator:
        """
        :return: Instance of SwapManipulator.
        """
        return SwapManipulator()


class TwoFactorSample(TwoFactorIntervention):
    """
    Two-factor evaluator using resampling of latent channels.

    Evaluates disentanglement by sampling new latent values (from prior)
    for the static or dynamic channels and observing prediction changes.

    :param initializer: Configuration object for dataset/model setup.
    :param dataset_type: Dataset split ('train', 'val', or 'test').
    :param evaluation_manager: Manager coordinating the evaluation.
    :param n_samples: Number of samples to use for evaluation.
    """

    def __init__(
        self,
        initializer: 'ConfigInitializer',
        dataset_type: str,
        evaluation_manager: 'EvaluationManager',
        n_samples: Optional[int] = None
    ) -> None:
        super().__init__(initializer, dataset_type, evaluation_manager, n_samples)

    def _init_manipulator(self) -> SampleManipulator:
        """
        :return: Instance of SampleManipulator using the model's prior.
        """
        return SampleManipulator(self.model)