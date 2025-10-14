"""
DCI Evaluator for Disentanglement Benchmarking.

This evaluator measures:
- Explicitness (predictability of factors from latents)
- Modularity (each latent subset aligns with one factor)
- Compactness (each factor depends on a small set of latents)

It relies on a latent explorer and a predictor to derive feature importances.
"""

from typing import Dict, List, TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from msd.evaluations.abstract_evaluator import AbstractEvaluator

if TYPE_CHECKING:
    from msd.configurations.config_initializer import ConfigInitializer
    from msd.evaluations.evaluation_manager import EvaluationManager


class DCIEvaluator(AbstractEvaluator):
    """
    Disentanglement evaluation using the DCI metric family.

    :param initializer: Configuration initializer.
    :param dataset_type: Dataset split name ('train', 'val', or 'test').
    :param evaluation_manager: Central evaluation controller.
    :param test_size: Proportion of samples to use for testing (float).
    :param random_state: Random seed for reproducibility (int).
    """
    def __init__(self,
                 initializer: "ConfigInitializer",
                 dataset_type: str,
                 evaluation_manager: "EvaluationManager",
                 test_size: float,
                 random_state: int):
        super().__init__(initializer, dataset_type, evaluation_manager)
        self.latent_explorer = self.evaluation_manager.get_latent_explorer()
        self.dataset, self.data_loader = self.initializer.get_dataset(self.dataset_type, loaders=True, labels=True)
        self.test_size = test_size
        self.random_state = random_state
        self.epsilon = 1e-7


    def eval(self, epoch: int) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Run evaluation and compute DCI scores.

        :param epoch: Evaluation epoch.
        :return: Dictionary of global DCI metrics and a table with detailed results.
        """
        factors = list(self.dataset.classes.keys())
        Z, Y = [], []
        mapping = None
        for X, Ys, Yd in self.data_loader:
            mapping = self.latent_explorer.get_map(epoch, (X, Ys, Yd))
            X, _Y = X.to(self.device), Ys | Yd
            with torch.no_grad():
                Z.append(self.model.latent_vector(self.model.preprocess(X)).detach().cpu().numpy())
            Y.append(_Y)
        Z, Y = np.concatenate(Z), {k: np.concatenate([y[k].detach().cpu().numpy() for y in Y]) for k in Y[0].keys()}
        idx = np.arange(len(Z))
        train_idx, test_idx = train_test_split(idx, test_size=self.test_size, random_state=self.random_state)
        Z_train, Z_test, Y_train, Y_test = (Z[train_idx], Z[test_idx],
                                            {k: Y[k][train_idx] for k in Y.keys()}, {k: Y[k][test_idx] for k in Y.keys()})
        explicitness = pd.DataFrame(columns=['explicitness_train', 'explicitness_test'])
        importance = pd.DataFrame(columns=factors)
        for f in factors:
            predictor = self.evaluation_manager.create_predictor()
            predictor.fit(Z_train, Y_train[f])
            train_acc = predictor.score(Z_train, Y_train[f])
            test_acc = predictor.score(Z_test, Y_test[f])
            importance_matrix = predictor.feature_importances_
            importance[f] = importance_matrix
            explicitness.loc[f] = [train_acc, test_acc]
        explicitness['explicitness_mean'] = explicitness.mean(axis=1)

        df = pd.DataFrame({'subset': {k: str(v) for k, v in mapping.items()}})
        modularity, modularity_score = self.modularity(factors, mapping, importance)
        compactness, compactness_score = self.compactness(factors, mapping, importance)
        df = pd.concat([df, modularity, compactness, explicitness], axis=1)
        scores = {
            'modularity': modularity_score,
            'compactness': compactness_score,
        } | explicitness.mean().to_dict()
        self.logger.log_table('evaluate/dci_scores', df, step=epoch)
        for k, v in scores.items():
            self.logger.log(f'evaluate/dci_{k}_score', v, step=epoch)
        return pd.Series(scores), df

    def modularity(
            self,
            factors: List[str],
            mapping: Dict[str, List[int]],
            importance: pd.DataFrame
    ) -> Tuple[pd.DataFrame, float]:
        """
        Compute modularity scores.

        :param factors: List of factor names.
        :param mapping: Latent subset mapping per factor.
        :param importance: Importance matrix.
        :return: Modularity DataFrame and modularity score.
        """
        modularity = pd.DataFrame(columns=['modularity'])
        modularity_score = 0
        total_weight = importance.values.sum()
        for k, v in mapping.items():
            v = np.array(v)
            modularity_j = 1
            code_weight = importance.loc[v].values.sum()
            for f in factors:
                importance_weight = importance[f][v].sum()
                p = (importance_weight / code_weight) if importance_weight != 0 else self.epsilon
                modularity_j += p * (np.log(p) / np.log(len(factors)))
            modularity.loc[k] = [modularity_j]
            rho = code_weight / total_weight
            modularity_score += rho * modularity_j
        return modularity, modularity_score

    def compactness(
            self,
            factors: List[str],
            mapping: Dict[str, List[int]],
            importance: pd.DataFrame
    ) -> Tuple[pd.DataFrame, float]:
        """
        Compute compactness scores.

        :param factors: List of factor names.
        :param mapping: Latent subset mapping per factor.
        :param importance: Importance matrix.
        :return: Compactness DataFrame and compactness score.
        """
        compactness = pd.DataFrame(columns=['compactness'])
        for f in factors:
            compactness_i = 1
            factor_weight = importance[f].sum()
            for k, v in mapping.items():
                v = np.array(v)
                importance_weight = importance[f][v].sum()
                p = (importance_weight / factor_weight) if importance_weight != 0 else self.epsilon
                compactness_i += p * (np.log(p) / np.log(self.model.latent_dim()))
            compactness.loc[f] = [compactness_i]
        compactness_score = compactness.values.mean()
        return compactness, compactness_score

    @property
    def name(self) -> str:
        """
        :return: Name identifier for the evaluator.
        """
        return 'dci'