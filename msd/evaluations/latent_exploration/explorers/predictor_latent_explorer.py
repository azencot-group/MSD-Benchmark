import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from msd.configurations.config_initializer import ConfigInitializer
from msd.evaluations.evaluation_manager import EvaluationManager
from msd.evaluations.latent_exploration.explorers.latent_explorer import LatentExplorer


class PredictorLatentExplorer(LatentExplorer):
    """
    A LatentExplorer that evaluates latent variable disentanglement by training
    supervised predictors to classify each ground-truth factor from the latent codes.

    For each factor:
    - Trains a simple supervised model (e.g., RandomForest, LinearClassifier)
    - Reports classification accuracy
    - Extracts feature importance to identify latent dimension alignment
    """

    def __init__(self,
                 initializer: ConfigInitializer,
                 dataset_type: str,
                 evaluation_manager: EvaluationManager,
                 batch_exploration: bool,
                 n_samples: int = None):
        """
        :param initializer: A ConfigInitializer instance.
        :param dataset_type: One of ['train', 'val', 'test'].
        :param evaluation_manager: The EvaluationManager coordinating this evaluation.
        :param batch_exploration: Whether to use only the current batch or the full dataset.
        :param n_samples: Max number of samples to explore; defaults to full dataset.
        """
        super().__init__(initializer, dataset_type, evaluation_manager, batch_exploration, n_samples)

    def eval(self, epoch: int, data_loader):
        """
        Runs the predictor-based latent analysis.

        - Computes latent vectors Z from input data.
        - Trains one predictor per factor.
        - Computes classification accuracy and feature importance.
        - Builds a factor-to-latent-dim mapping based on strongest importance scores.

        :param epoch: Current training epoch (used for caching and logging).
        :param data_loader: Data iterator (batch or full loader).
        :return: Tuple:
            - `accuracy`: DataFrame with accuracy per factor.
            - `mapping`: Dict of {factor_name: [latent_dim indices]} showing aligned dimensions.
        """
        n = 0
        factors = list(self.dataset.classes.keys())
        Z, Y = [], []
        for X, Ys, Yd in data_loader:
            X = X.to(self.device)
            X, _Y = self.model.preprocess(X), Ys | Yd
            with torch.no_grad():
                Z.append(self.model.latent_vector(X).detach().cpu().numpy())
            Y.append(_Y)
            n += X.shape[0]
            if n >= self.n_samples:
                break
        Z, Y = np.concatenate(Z), {k: np.concatenate([y[k].detach().cpu().numpy() for y in Y]) for k in Y[0].keys()}
        accuracy = pd.DataFrame(columns=['score'])
        importance = pd.DataFrame(columns=factors)
        for f in tqdm(factors, desc='Explore'):
            predictor = self.evaluation_manager.create_predictor()
            predictor.fit(Z, Y[f])
            accuracy.loc[f] = predictor.score(Z, Y[f])
            _importance = predictor.feature_importances_
            importance[f] = _importance / _importance.sum()

        _mapping = importance.idxmax(axis=1).to_dict()
        mapping = {f: [i for i, v in enumerate(_mapping.values()) if v == f] for f in factors if f in _mapping.values()}
        log_df = pd.merge(pd.DataFrame({'subset': {k: str(v) for k, v in mapping.items()}}), accuracy, left_index=True, right_index=True)
        self.logger.log_table('evaluate/mapping', log_df, step=epoch)
        return accuracy, mapping
