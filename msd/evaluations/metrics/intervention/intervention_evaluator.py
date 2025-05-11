from abc import abstractmethod
from typing import TYPE_CHECKING, List, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from msd.evaluations.latent_exploration.latent_manipulator import Manipulator
from msd.evaluations.abstract_evaluator import AbstractEvaluator

if TYPE_CHECKING:
    from msd.configurations.config_initializer import ConfigInitializer
    from msd.evaluations.evaluation_manager import EvaluationManager


class InterventionEvaluator(AbstractEvaluator):
    """
    Base class for evaluating disentanglement through latent space interventions.

    This evaluator assesses how well subsets of latent variables control different ground-truth factors.
    It operates by manipulating specific latent dimensions (e.g., swapping them) and measuring whether
    predictions from a classifier reflect the expected semantic changes.

    Subclasses must define:
        - The type of manipulation (e.g., swap, resample)
        - The mapping strategy (e.g., learned or predefined)
        - The expected ground-truth mapping (used for scoring)
    """

    def __init__(self, initializer: "ConfigInitializer", dataset_type: str, evaluation_manager: "EvaluationManager", n_samples: int = None):
        """
        Initialize the evaluator with configuration and model components.

        :param initializer: A ConfigInitializer instance for loading datasets and components.
        :param dataset_type: Dataset split to evaluate on ('train', 'val', or 'test').
        :param evaluation_manager: EvaluationManager handling judge and model loading.
        :param n_samples: Optional limit on the number of samples to evaluate.
        """
        super().__init__(initializer, dataset_type, evaluation_manager)
        self.judge = self.evaluation_manager.get_judge()
        self.latent_explorer = self.evaluation_manager.get_latent_explorer()
        self.dataset, self.data_loader = self.initializer.get_dataset(self.dataset_type, loaders=True, labels=True)
        self.manipulator = self._init_manipulator()
        self.manipulation_method = self.manipulator.manipulation_method
        self.n_samples = n_samples if n_samples else len(self.dataset)

    @abstractmethod
    def _init_manipulator(self) -> Manipulator:
        """
        Instantiate the latent manipulator used for interventions.

        :return: A Manipulator instance (e.g., SwapManipulator, SampleManipulator).
        """
        pass

    @abstractmethod
    def _get_mapping(self, epoch, batch: Tuple[Tensor, Dict[str, Tensor], Dict[str, Iterable]]) -> Dict[str, List[int]]:
        """
        Define or retrieve the mapping between ground-truth factors and latent subsets.

        :param epoch: Current evaluation epoch.
        :param batch: A tuple of samples (X), static labels (Ys) and dynamic labels (Yd) used for local mapping estimation (if applicable).
        :return: A dictionary mapping each factor name to a list of latent indices.
        """
        pass

    @property
    @abstractmethod
    def factor_method(self) -> str:
        """
        Describe the factor selection method (e.g., learned, predefined).

        :return: Name of the factor mapping strategy.
        """
        pass

    @abstractmethod
    def _expected_map(self) -> Dict[str, List[str]]:
        """
        Define the expected alignment between latent subsets and semantic factors.

        :return: A dictionary mapping factor names to their correct labels in the output.
        """
        pass

    def get_samples(self, Z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Generate a manipulated sample from the latent space.

        :param Z: Latent representations of shape (B, D).
        :return: A tuple (indices, manipulated_latents).
        """
        return self.manipulator.get_samples(Z)

    def get_label(self, factor: str, Y: Dict[str, Tensor], P: Dict[str, Tensor], idx: Tensor) -> Dict[str, Tensor]:
        """
        Return the correct label dictionary for comparison with predictions.

        :param factor: Name of the manipulated factor.
        :param Y: Ground-truth labels.
        :param P: Model predictions.
        :param idx: Sample indices.
        :return: Dictionary of labels for scoring.
        """
        return Y

    def get_factors(self) -> List[str]:
        """
        Retrieve the list of ground-truth factors.

        :return: List of factor names.
        """
        return list(self.dataset.classes.keys())


    def classify(self, D: Tensor) -> Dict[str, Tensor]:
        """
        Run the judge model on decoded data.

        :param D: Decoded output from the model.
        :return: Dictionary of predicted class labels.
        """
        return {k: v.detach().cpu() for k, v in self.judge(D, False).items()}

    def accuracy(self, Y: Dict[str, Tensor], P: Dict[str, Tensor]) -> Dict[str, List[bool]]:
        """
        Compute binary accuracy per factor.

        :param Y: Ground-truth labels.
        :param P: Predicted labels.
        :return: Per-factor binary accuracy as a list of bools.
        """
        # noinspection PyUnresolvedReferences
        return {k: (Y[k] == P[k]).reshape(-1).detach().cpu().tolist() for k in P.keys()}

    def eval(self, epoch: int) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Evaluate the model's disentanglement by latent intervention.

        :param epoch: Epoch index used for logging and model state.
        :return: A tuple containing:
            - A dictionary with the final evaluation score keyed by evaluator name.
            - A DataFrame with per-factor accuracy scores for each subset.
        """
        self.logger.info(f'Evaluating {self.name} at epoch {epoch}')
        self.model.eval()
        nD = self.model.latent_dim()
        factors = self.get_factors()
        accumulated_accs = {name: {k: [] for k in factors} for name in self._expected_map()}
        mapping = {}
        i = 0
        for X, Ys, Yd in self.data_loader:
            X, Y = X.to(self.device), Ys | Yd
            with torch.no_grad():
                Z = self.model.encode(self.model.preprocess(X))
                idx, S = self.get_samples(Z)
            mapping = self._get_mapping(epoch, (X, Ys, Yd))
            for name, subset in mapping.items():
                complement = np.array([i for i in range(nD) if i not in subset])
                _Z, _S = self.model.swap_channels(Z, S, complement) if len(subset) > 0 else (Z, S)
                with torch.no_grad():
                    _D = self.model.decode(_Z)
                    _D = self.model.postprocess(_D)
                    P = self.classify(_D)
                accs = self.accuracy(self.get_label(name, Y, P, idx), P)
                for f in factors:
                    accumulated_accs[name][f].append(accs[f])
            i += X.shape[0]
            if i >= self.n_samples:
                break
        keys = set(mapping.keys())
        expected_keys = set(self._expected_map().keys())
        missing = set.difference(expected_keys, keys)
        if len(missing) > 0:
            self.logger.warning(f'Could not find a valid mapping for all factors: {missing}')
            for k in missing:
                mapping[k] = np.nan
        _df = pd.DataFrame(columns=['subset'] + factors)
        for name, accs in accumulated_accs.items():
            mean_accs = {k: np.mean(v) for k, v in accs.items()}
            _df.loc[name, 'subset'] = mapping[name]
            _df.loc[name, factors] = [mean_accs[k] for k in factors]
        _df[factors] = _df[factors].astype(float)
        final_score, _df = self.calculate_score(_df, missing)
        self.logger.log_table(f'evaluate/{self.name}_accuracies', _df, step=epoch)
        self.logger.log(f'evaluate/{self.name}_score', final_score, step=epoch)
        return pd.Series({self.name: final_score}), _df

    def calculate_score(self, _df: pd.DataFrame, missing: set) -> Tuple[float, pd.DataFrame]:
        """
        Compute the intervention-based disentanglement score.

        :param _df: DataFrame with per-factor accuracy entries.
        :param missing: Factors missing from mapping.
        :return: Tuple of final score and the full results DataFrame.
        """
        expected = self._expected_map()
        random_floor_dict = {k: 1 / v for k, v in self.dataset.class_dims.items()}
        for k in missing:
            v = expected[k]
            _df.loc[k, v] = 0
            _df.loc[k, [c for c in _df.columns if c not in v
                        and c != 'subset']] = 1 + np.array([random_floor_dict[_v]
                                                            for _v in [c for c in _df.columns if c not in v and c != 'subset']])
        scores = _df[[c for c in _df.columns if c != 'subset']]
        scores_mask = pd.DataFrame(columns=scores.columns, index=scores.index, data=0.0)
        for k, v in expected.items():
            _v = [c for c in scores.columns if c not in v]
            scores_mask.loc[k, v] = 1
            scores_mask.loc[k, [c for c in scores_mask.columns if c not in v]] = [random_floor_dict[vv] for vv in _v]

        diff = (scores - scores_mask).abs()
        on, off = [], []
        for k, v in expected.items():
            on.append(diff.loc[k, v].values)
            off.append(diff.loc[k, [c for c in scores.columns if c not in v]].values)
        on = np.concatenate(on).mean()
        off = np.concatenate(off).mean()
        final_score = 1 - (on + off) / 2
        return final_score, _df

    @property
    def name(self) -> str:
        """
        :return: Unique name identifying this evaluator instance.
        """
        return f'intervention_{self.factor_method}_{self.manipulation_method}'
