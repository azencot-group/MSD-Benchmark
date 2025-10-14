from os import path as osp
from typing import Any, List, Optional, TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from msd.configurations.msd_component import MSDComponent
from msd.evaluations.judge.judge import Judge
from msd.evaluations.latent_exploration.explorers.latent_explorer import LatentExplorer
from msd.methods.abstract_model import AbstractModel

if TYPE_CHECKING:
    from msd.configurations.config_initializer import ConfigInitializer

class EvaluationManager(MSDComponent):
    """
    Orchestrates evaluation by loading models, running evaluators, and managing evaluation tools like judges, latent explorers, and predictors.
    Supports repeated evaluations for robust scoring, logging metrics, and integration with visualization or analysis tools.
    """

    def __init__(
            self,
            initializer: 'ConfigInitializer',
            dataset_type: str,
            main: int = 0,
            repeat: int = 1,
            judge_cfg: Optional[DictConfig] = None,
            predictor_cfg: Optional[DictConfig] = None,
            latent_explorer_cfg: Optional[DictConfig] = None,
            model: Optional[AbstractModel] = None
    ):
        """
        :param initializer: ConfigInitializer used to build the pipeline.
        :param dataset_type: One of 'train', 'val', or 'test'.
        :param main: Primary evaluator index.
        :param repeat: Number of repeated evaluation runs.
        :param judge_cfg: Optional judge configuration.
        :param predictor_cfg: Optional predictor configuration.
        :param latent_explorer_cfg: Optional latent explorer configuration.
        :param model: Optional preloaded model. If None, loads from checkpoint.
        """
        self.initializer = initializer
        self.main = main
        self.repeat = repeat
        self.judge_cfg = judge_cfg
        self.predictor_cfg = predictor_cfg
        self.latent_explorer_cfg = latent_explorer_cfg
        self.config = self.initializer.config
        self.name = self.config.name
        self.checkpoint_dir = self.config.checkpoint_dir
        self.device = self.initializer.config.device
        self.logger = self.initializer.get_logger()
        self.testing = False

        # Load model (either passed directly or from checkpoint)
        if model is None:
            self.model = self.initializer.get_model().to(self.device)
            ckpt = osp.join(self.checkpoint_dir, f'{self.config.name}_{self.config.load_model}.pth')
            if osp.exists(ckpt):
                self.checkpoint = torch.load(ckpt, weights_only=False)
                self.model.load_state_dict(self.checkpoint['model'])
                self.epoch = self.checkpoint['epoch']
                self.logger.info(f'Loaded model from epoch {self.epoch}: {self.config.load_model}')
            else:
                self.logger.info(f'No model checkpoint found at {ckpt}. Starting from scratch.')
                self.epoch = 0
        else:
            self.model = model
            self.epoch = None

        if self.judge_cfg is not None:
            self.judge = self.initializer.initialize(self.judge_cfg, initializer=self.initializer).to(self.device)
        if self.latent_explorer_cfg is not None:
            self.latent_explorer = self.initializer.initialize(self.latent_explorer_cfg,
                                                               dataset_type=dataset_type,
                                                               initializer=self.initializer,
                                                               evaluation_manager=self)

        # Load all evaluators defined in config
        self.evaluators = [self.initializer.initialize(e,
                                                       identifier=f'{e.name}' + (f'({e.parameters["dataset_type"]})' if 'dataset_type' in e.parameters else ''),
                                                       dataset_type=dataset_type,
                                                       initializer=self.initializer,
                                                       evaluation_manager=self)
                           for e in self.config.evaluation.evaluators]

    def evaluate(self, epoch: Optional[int] = None) -> List[Tuple[pd.Series, pd.DataFrame]]:
        """
        Run evaluation once across all configured evaluators.

        Each evaluator returns:
          - A Series of high-level scores
          - A DataFrame of detailed breakdowns

        :param epoch: Optional epoch index. If not provided, uses the latest checkpoint.
        :return: List of (score_series, result_dataframe) tuples, one per evaluator.
        """
        if epoch is None:
            if self.epoch is None:
                checkpoint = torch.load(osp.join(self.checkpoint_dir, f'{self.config.name}_{self.config.load_model}.pth'))
                self.epoch = checkpoint['epoch']
            epoch = self.epoch
        evaluations = [e.eval(epoch) for e in self.evaluators]
        return evaluations

    def run_test(self) -> None:
        """
        Run evaluation `repeat` times and aggregate statistics across runs.
        Logs both mean and standard deviation for all tasks and scores.
        """
        self.testing = True

        results = [self.evaluate() for _ in range(self.repeat)]
        experiments = list(zip(*results))
        for i, e in enumerate(experiments):
            if not any(e):
                continue
            eval_name = self.evaluators[i].name
            scores, tables = zip(*e)
            if all(a is None for a in tables):
                continue
            sdf = pd.DataFrame(scores)
            s_mean, s_std = sdf.mean(axis=0), sdf.std(axis=0)
            s = pd.concat([s_mean, s_std], keys=['mean', 'std'], axis=1)
            df = pd.concat(tables).select_dtypes(include=[np.number])
            d_mean, d_std = df.groupby(df.index).mean(), df.groupby(df.index).std()
            d = pd.concat([d_mean, d_std], keys=['mean', 'std'], axis=1).loc[tables[0].index]
            # noinspection PyUnresolvedReferences
            d.columns = [f'{f}_{o}' for f, o in zip(d.columns.droplevel(0), d.columns.droplevel(1))]
            self.logger.log_table(f'evaluate/repeat_scores/{eval_name}', s)
            self.logger.log_table(f'evaluate/repeat_tables/{eval_name}', d)

        self.testing = False

    def get_judge(self) -> Judge:
        """
        :return: The initialized judge module (if available).
        """
        return self.judge

    def get_latent_explorer(self) -> LatentExplorer:
        """
        :return: The initialized latent explorer module (if available).
        """
        return self.latent_explorer

    def create_predictor(self) -> Any:
        """
        Instantiate a new predictor from configuration.

        :return: Predictor instance.
        """
        return self.initializer.initialize(self.predictor_cfg, use_cache=False)
