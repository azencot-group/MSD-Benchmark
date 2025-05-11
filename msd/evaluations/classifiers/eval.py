from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch

from msd.configurations.config_initializer import ConfigInitializer
from msd.evaluations.abstract_evaluator import AbstractEvaluator
from msd.evaluations.evaluation_manager import EvaluationManager
from msd.methods.abstract_model import AbstractModel


class ClassifierEvaluator(AbstractEvaluator):
    """
    Evaluator for trained classifiers on a labeled dataset.

    Computes sequence-level and frame-level loss and accuracy metrics, and returns
    a summary DataFrame with per-task performance.
    """

    def __init__(self, initializer: ConfigInitializer, dataset_type: str, evaluation_manager: EvaluationManager):
        """
        :param initializer: ConfigInitializer used to load model, data, and loss functions.
        :param dataset_type: One of 'train', 'val', or 'test'.
        :param evaluation_manager: Parent evaluation manager handling coordination and logging.
        """
        self.initializer = initializer
        self.dataset, self.data_loader = self.initializer.get_dataset(dataset_type, loaders=True, labels=True)
        super().__init__(initializer, dataset_type, evaluation_manager)
        train_cfg = self.config.trainer
        self.static_loss = self.initializer.initialize(train_cfg.static_loss)
        self.sequence_loss = self.initializer.initialize(train_cfg.sequence_loss)
        self.logger = self.initializer.get_logger()

    def init_model(self) -> AbstractModel:
        """
        Initialize the model and move it to the appropriate device.

        :return: The model ready for evaluation.
        """
        return self.initializer.get_model().to(self.device)

    def eval(self, epoch: int) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Evaluate the model over the dataset split and log losses and accuracies.

        :param epoch: Current epoch (used for logging).
        :return: A tuple of (accuracy Series, summary DataFrame).
        """
        self.model.eval()
        losses = {}
        accuracy = {}
        for X, Ys, Yd in self.data_loader:
            X = self.model.preprocess(X.to(self.device))
            Y = (Ys | Yd)
            with torch.no_grad():
                P_sequence, P_static = self.model(X)
            batch_losses = ({f'{k}_sequence': self.sequence_loss(P_sequence[k], Y[k].to(self.device))
                             for k in P_sequence.keys()} |
                            {f'{k}_static': self.static_loss(P_static[k], Y[k].to(self.device)) for k in P_static.keys()})
            batch_loss = sum(batch_losses.values())
            batch_losses['total'] = batch_loss
            batch_acc = ({f'{k}_sequence': (torch.argmax(P_sequence[k], dim=1) == Y[k].to(self.device))
                         .float().mean().detach().cpu().numpy() for k in P_sequence.keys()} |
                         {f'{k}_static': (torch.argmax(P_static[k], dim=2) == torch.stack([Y[k]] * P_static[k].shape[1], dim=1)
                                          .to(self.device)).float().mean().detach().cpu().numpy() for k in P_static.keys()})
            for k, v in batch_losses.items():
                v_cpu = v.detach().cpu().numpy()
                losses[k] = losses.get(k, []) + [v_cpu]
            for k, v in batch_acc.items():
                accuracy[k] = accuracy.get(k, []) + [v]

        losses = {k: np.mean(v) for k, v in losses.items()}
        accuracy = {k: np.mean(v) for k, v in accuracy.items()}
        self.logger.log_dict(f'{self.dataset_type}/eval_losses', losses, step=epoch)
        self.logger.log_dict(f'{self.dataset_type}/eval_accuracy', accuracy, step=epoch)
        df = pd.DataFrame({
            'task': list(accuracy.keys()),
            'accuracy': list(accuracy.values()),
            'loss': [losses[k] for k in accuracy.keys()]})

        return df['accuracy'], df

    @property
    def name(self) -> str:
        """
        :return: Identifier for this evaluator.
        """
        return 'accuracy_evaluator'
