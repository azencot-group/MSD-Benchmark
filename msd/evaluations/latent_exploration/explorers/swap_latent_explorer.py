import itertools as it

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from msd.configurations.config_initializer import ConfigInitializer
from msd.evaluations.evaluation_manager import EvaluationManager
from msd.evaluations.latent_exploration.explorers.latent_explorer import LatentExplorer
from msd.evaluations.latent_exploration.latent_manipulator import SwapManipulator


class SwapLatentExplorer(LatentExplorer):
    """
    A latent explorer that uses controlled latent swaps to evaluate disentanglement.

    The model's latent dimensions are divided into candidate subsets using a latent divider.
    Then, latent channels are swapped between samples and decoded.
    A judge model is used to predict the factors from decoded outputs.
    Factor preservation is evaluated to determine alignment between latent dimensions and factors.
    """


    def __init__(
        self,
        initializer: ConfigInitializer,
        dataset_type: str,
        evaluation_manager: EvaluationManager,
        latent_divider_cfg: DictConfig,
        batch_exploration: bool,
        n_samples: int = None,
        size_penalty: float = None,
    ):
        """
        :param initializer: The ConfigInitializer object instantiator.
        :param dataset_type: The dataset split (e.g., 'train', 'test', etc.).
        :param evaluation_manager: The EvaluationManager coordinating this evaluation.
        :param latent_divider_cfg: Config object for latent divider (returns subsets of latent dims).
        :param batch_exploration: Whether to run on a single batch only.
        :param n_samples: Max number of samples to evaluate.
        :param size_penalty: Penalty factor for using larger latent subsets during mapping.
        """
        super().__init__(initializer, dataset_type, evaluation_manager, batch_exploration, n_samples)
        self.latent_divider = self.initializer.initialize(latent_divider_cfg, use_cache=False)
        self.judge = self.evaluation_manager.get_judge()
        self.size_penalty = 1 if size_penalty is None else size_penalty
        self.manipulator = SwapManipulator()
        self.mapping = {}

    def eval(self, epoch: int, data_loader):
        """
        Perform swap-based latent evaluation.

        - Encodes latent vectors for samples
        - Swaps selected latent subsets with random samples
        - Decodes the results and uses the judge to predict factor labels
        - Computes per-factor prediction accuracy per latent subset
        - Uses a mapping algorithm to find best subset-to-factor assignment

        :param epoch: Current training epoch.
        :param data_loader: Data iterator.
        :return: Tuple (mean accuracy, factor-to-latent-subset mapping)
        """
        self.model.eval()
        nF, nD = len(self.dataset.classes), self.model.latent_dim()
        subsets = self.latent_divider.divide(nF, nD)
        factors = list(self.dataset.classes.keys())

        df = pd.DataFrame(columns=['subset'] + factors)
        accumulated_accs = {subset: {k: [] for k in factors} for subset in subsets}
        n = 0
        for X, Ys, Yd in data_loader:
            X, Y = self.model.preprocess(X).to(self.device), Ys | Yd
            with torch.no_grad():
                Z = self.model.encode(X)
                _, S = self.manipulator.get_samples(Z)
            for subset in subsets:
                complement = np.array([i for i in range(nD) if i not in subset])
                with torch.no_grad():
                    _Z, _ = self.model.swap_channels(Z, S, complement)
                    _D = self.model.decode(_Z)
                    _D = self.model.postprocess(_D)
                    P = self.judge(_D)
                for i, k in enumerate(factors):
                    # noinspection PyUnresolvedReferences
                    accumulated_accs[tuple(subset)][k] += (Y[k].detach().cpu().numpy() == P[k].detach().cpu().numpy()).tolist()
            n += X.shape[0]
            if n >= self.n_samples:
                break
        for subset, accs in accumulated_accs.items():
            mean_accs = {k: np.mean(v) for k, v in accs.items()}
            df.loc[len(df)] = [subset] + [mean_accs[k] for k in factors]
        _df = df.copy()
        _df[factors] = (_df[factors] * 100).round(2).astype(str) + '%'
        _df['subset'] = _df['subset'].astype(str)

        mapping, accuracy = self.generate_map(df, factors)
        self.logger.log_dict('evaluate/mapping', {k: str(v) for k, v in mapping.items()}, step=epoch)
        return accuracy, mapping

    def generate_map(self, df: pd.DataFrame, factors: list):
        """
        Determine optimal mapping from factors to latent subsets based on accuracy.

        Penalizes subsets with more dimensions using an exponential penalty.

        :param df: DataFrame of scores per factor and subset.
        :param factors: List of all factor names.
        :return: (mapping: dict, mean_accuracy: float)
        """
        df_penalty = df.copy()
        for factor in factors:
            df_penalty[factor] = df_penalty.apply(lambda row: row[factor] *
                                                  self.size_penalty ** (len(row['subset']) - 1), axis=1)

        def is_unique(mapping, df_):
            used_channels_ = set()
            for subset in mapping.values():
                for channel in df_.loc[df_['subset'] == subset, 'subset'].values[0]:
                    if channel in used_channels_:
                        return False
                    used_channels_.add(channel)
            return True
        factor_permutations = list(it.permutations(factors))
        best_mapping = None
        max_accuracy = float('-inf')
        for perm in factor_permutations:
            temp_mapping = {}
            used_channels = set()
            total_accuracy = 0
            for factor in perm:
                df_filtered = df_penalty[~df_penalty['subset'].apply(lambda x: any(channel in used_channels for channel in x))]
                if not df_filtered.empty:
                    df_sorted = df_filtered.sort_values(by=[factor], ascending=False)
                    best_subset = df_sorted.iloc[0]['subset']
                    temp_mapping[factor] = best_subset
                    used_channels.update(best_subset)
                    total_accuracy += df_sorted.iloc[0][factor]
            if is_unique(temp_mapping, df_penalty) and total_accuracy > max_accuracy:
                max_accuracy = total_accuracy
                best_mapping = temp_mapping
        max_accuracy /= len(factors)
        return best_mapping, max_accuracy
