"""
Multi-factor intervention evaluators for disentanglement benchmarking.

These evaluators extend the abstract `InterventionEvaluator` to support
evaluation of models via manipulation (swap or sample) of latent variables
according to factor-wise mappings discovered by a latent explorer.
"""

from abc import ABC
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple

from torch import Tensor

from msd.evaluations.latent_exploration.latent_manipulator import SampleManipulator, SwapManipulator
from msd.evaluations.metrics.intervention.intervention_evaluator import InterventionEvaluator

if TYPE_CHECKING:
    from msd.configurations.config_initializer import ConfigInitializer
    from msd.evaluations.evaluation_manager import EvaluationManager


class MultiFactorIntervention(InterventionEvaluator, ABC):
    """
    Base class for multi-factor intervention-based evaluators.

    Uses a latent explorer to define a mapping from factors to subsets of latent
    channels, and assumes each factor influences only itself.

    Subclasses define how latent variables are manipulated (swap or sample).

    :param initializer: Configuration initializer for model, dataset, logger, etc.
    :param dataset_type: One of ['train', 'val', 'test'].
    :param evaluation_manager: Orchestrates evaluation components.
    :param n_samples: Number of samples to evaluate.
    """


    def __init__(
        self,
        initializer: "ConfigInitializer",
        dataset_type: str,
        evaluation_manager: "EvaluationManager",
        n_samples: Optional[int]
    ) -> None:
        super().__init__(initializer, dataset_type, evaluation_manager, n_samples)

    def _get_mapping(
        self,
        epoch: int,
        batch: Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]
    ) -> Dict[str, List[int]]:
        """
        Get mapping from factors to latent dimensions using the latent explorer.

        :param epoch: Current evaluation epoch.
        :param batch: A single batch of data (inputs, static labels, dynamic labels).
        :return: Mapping from factor name to list of latent channel indices.
        """
        return self.latent_explorer.get_map(epoch, batch)

    def _expected_map(self) -> Dict[str, List[str]]:
        """
        Define ground-truth mapping: each factor influences only itself.

        :return: Dictionary mapping each factor to a list containing itself.
        """
        return {k: [k] for k in self.dataset.classes}

    @property
    def factor_method(self) -> str:
        """
        :return: Identifier of factor mapping strategy.
        """
        return 'multi_factor'



class MultiFactorSwap(MultiFactorIntervention):
    """
    Multi-factor evaluator using latent dimension swapping between pairs of samples.

    Swapped latent subsets should cause only the corresponding factor(s) to change
    if the model is disentangled.
    """

    def __init__(
        self,
        initializer: "ConfigInitializer",
        dataset_type: str,
        evaluation_manager: "EvaluationManager",
        n_samples: Optional[int] = None
    ) -> None:
        super().__init__(initializer, dataset_type, evaluation_manager, n_samples)

    def _init_manipulator(self) -> SwapManipulator:
        """
        :return: An instance of SwapManipulator.
        """
        return SwapManipulator()


class MultiFactorSample(MultiFactorIntervention):
    """
    Multi-factor evaluator using latent resampling from the model's prior.

    For each sample, latent channels corresponding to a factor are resampled.
    Changes in predicted attributes are used to evaluate disentanglement.
    """

    def __init__(
        self,
        initializer: "ConfigInitializer",
        dataset_type: str,
        evaluation_manager: "EvaluationManager",
        n_samples: Optional[int] = None
    ) -> None:
        super().__init__(initializer, dataset_type, evaluation_manager, n_samples)

    def _init_manipulator(self) -> SampleManipulator:
        """
        :return: An instance of SampleManipulator using the model's prior.
        """
        return SampleManipulator(self.model)