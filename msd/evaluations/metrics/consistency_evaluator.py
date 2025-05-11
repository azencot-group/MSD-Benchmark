"""
Consistency-based evaluators for disentanglement benchmarking.

These evaluators measure temporal and representational consistency of static factor predictions
under various forms of latent space manipulation.

Classes:
    ConsistencyEvaluator: Base class for evaluating consistency of static factors.
    ConsistencySwap: Uses latent swapping to assess consistency.
    ConsistencyGlobalSample: Uses global resampling and compares prediction mode.
    ConsistencyLocalSample: Uses global resampling and compares temporal stability.
"""

from abc import ABC
from typing import Dict, List, Tuple, Set

import pandas as pd
import torch
from torch import Tensor

from msd.evaluations.latent_exploration.latent_manipulator import SampleManipulator, SwapManipulator
from msd.evaluations.metrics.intervention.intervention_evaluator import InterventionEvaluator


class ConsistencyEvaluator(InterventionEvaluator, ABC):
    """
    Base class for evaluating the consistency of static factors under latent manipulations.

    This evaluator assumes frame-level classification and is used to measure whether
    static predictions remain stable when the latent representation is manipulated.
    """

    def get_factors(self) -> List[str]:
        """
        Get the list of static factors in the dataset.

        :return: A list of factor names.
        """
        return list(self.dataset.static_factors)

    def _get_mapping(self, epoch: int, batch: Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]) -> Dict[str, List[int]]:
        """
        Return the latent subset mapping for static factors only.

        :param epoch: Current evaluation epoch.
        :param batch: A data batch (inputs and labels).
        :return: Dictionary mapping each static factor to a list of latent dimensions.
        """
        return {k: v for k, v in self.latent_explorer.get_map(epoch, batch).items() if k in self.get_factors()}

    def _expected_map(self) -> Dict[str, List[str]]:
        """
        Return the expected mapping between each static factor and itself.

        :return: A dictionary mapping each factor to itself.
        """
        return {k: [k] for k in self.dataset.static_factors}

    def factor_method(self) -> str:
        """
        :return: The factor selection method name.
        """
        return 'multi_factor'

    def classify(self, D: Tensor) -> Dict[str, Tensor]:
        """
        Run frame-level classification on decoded samples.

        :param D: Decoded model outputs.
        :return: Dictionary of predicted values per static factor and per frame.
        """
        return {k: v.detach().cpu() for k, v in self.judge(D, frame_level=True).items()}

    def get_label(self, factor: str, Y: Dict[str, Tensor], P: Dict[str, Tensor], idx: Tensor) -> Dict[str, Tensor]:
        """
        Get frame-wise labels for consistency evaluation.

        :param factor: Current factor under evaluation.
        :param Y: Ground truth labels.
        :param P: Model predictions.
        :param idx: Sample indices.
        :return: Dictionary of labels per factor.
        """
        return {k: v[idx] if k != factor else v for k, v in Y.items()}

    def accuracy(self, Y: Dict[str, Tensor], P: Dict[str, Tensor]) -> Dict[str, List[bool]]:
        """
        Compute accuracy of predictions relative to static ground truth.

        :param Y: Ground truth labels.
        :param P: Predictions per frame.
        :return: Dictionary of accuracy lists per factor.
        """
        accuracies = {}
        for k, p in P.items():
            t = p.shape[1]
            y = Y[k].repeat(t, 1).t()
            # noinspection PyUnresolvedReferences
            accuracies[k] = (y == p).detach().cpu().tolist()
        return accuracies


    def calculate_score(self, _df: pd.DataFrame, missing: Set[str]) -> Tuple[float, pd.DataFrame]:
        """
        Compute overall consistency score based on deviation from perfect agreement.

        :param _df: DataFrame of per-factor accuracies.
        :param missing: Set of missing factors.
        :return: Final consistency score and completed DataFrame.
        """
        expected = self._expected_map()
        random_floor_dict = {k: 0 for k in self.get_factors()}
        for k in missing:
            v = expected[k]
            _df.loc[k, v] = random_floor_dict[k]
        scores = _df[[c for c in _df.columns if c != 'subset']]
        scores_mask = pd.DataFrame(columns=scores.columns, index=scores.index, data=1.0)

        diff = (scores - scores_mask).abs()
        final_score = 1 - diff.mean().mean()
        return final_score, _df

    @property
    def name(self) -> str:
        """
        :return: Name of the evaluator.
        """
        return f'consistency_{self.manipulation_method}'


class ConsistencySwap(ConsistencyEvaluator):
    """
    Evaluator using latent channel swapping for consistency.
    """

    def _init_manipulator(self) -> SwapManipulator:
        return SwapManipulator()


class ConsistencyGlobalSample(ConsistencyEvaluator):
    """
    Evaluator using global latent sampling, comparing predictions to their dominant mode.
    """

    def _init_manipulator(self) -> SampleManipulator:
        return SampleManipulator(self.model)

    def get_label(self, factor: str, Y: Dict[str, Tensor], P: Dict[str, Tensor], idx: Tensor) -> Dict[str, Tensor]:
        """
        Compute global label from predicted mode across frames.

        :param factor: Current factor.
        :param Y: Unused (predictions are used as proxy).
        :param P: Frame-level predictions.
        :param idx: Unused.
        :return: Dictionary with mode label for each factor.
        """
        return {k: torch.mode(v)[0] for k, v in P.items()}

    @property
    def name(self) -> str:
        """
        :return: Name of the evaluator.
        """
        return f'local_consistency_{self.manipulation_method}'


class ConsistencyLocalSample(ConsistencyGlobalSample):
    """
    Evaluator that checks whether predictions are temporally stable (i.e., frame-to-frame consistency).
    """

    def accuracy(self, Y: Dict[str, Tensor], P: Dict[str, Tensor]) -> Dict[str, List[bool]]:
        """
        Check if predictions remain unchanged between consecutive frames.

        :param Y: Ignored.
        :param P: Frame-level predictions.
        :return: Dictionary of binary temporal consistency per factor.
        """
        return {k: (p[:, :-1] == p[:, 1:]).detach().cpu().tolist() for k, p in P.items()}

    @property
    def name(self) -> str:
        """
        :return: Name of the evaluator.
        """
        return f'global_consistency_{self.manipulation_method}'