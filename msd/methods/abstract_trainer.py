import time
from abc import ABC, abstractmethod
from os import path as osp
from typing import TYPE_CHECKING

import numpy as np
import torch

from msd.configurations.msd_component import MSDComponent
from msd.utils.loading_utils import init_directories

if TYPE_CHECKING:
    from msd.configurations.config_initializer import ConfigInitializer
    from msd.methods.abstract_model import AbstractModel


class AbstractTrainer(ABC, MSDComponent):
    """
    Abstract base class for all training workflows.

    Handles model setup, checkpointing, logging, and evaluation. Subclasses must implement `train_step()` to perform per-epoch updates.
    """

    def __init__(self, initializer: 'ConfigInitializer', return_labels: bool = False):
        """
        :param initializer: ConfigInitializer for datasets, models, loggers, etc.
        :param return_labels: Whether to return labels (for supervised training).
        """
        self.initializer = initializer
        self.cfg = self.initializer.config
        self.train_cfg = self.cfg.trainer
        self.name = self.cfg.name
        self.device = self.cfg.device
        self.epochs = self.train_cfg.epochs
        self.load_model = self.cfg.load_model
        self.save_model = self.train_cfg.save_model
        self.resume = self.train_cfg.resume
        self.verbose = self.train_cfg.verbose
        self.out_dir = self.cfg.checkpoint_dir
        self.model_path = osp.join(self.out_dir, f'{self.name}.pth')
        init_directories(self.out_dir)

        self.train_set, self.train_loader = self.initializer.get_dataset(split='train', loaders=True, labels=return_labels)
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.evaluator = self.initializer.get_evaluator(model=self.model)

        self.logger = self.initializer.get_logger()
        if self.resume:
            model_path = self.model_path.replace('.pth', '_last.pth')
            if osp.exists(model_path):
                state = torch.load(model_path)
                self.logger.info(f'Resuming from {model_path} at epoch {state["epoch"] + 1}...')
                self.epochs.start = state['epoch'] + 1
                self.model.load_state_dict(state['model'])
                self.optimizer.load_state_dict(state['optimizer'])
            else:
                self.logger.info(f'No model found at {model_path}. Starting from scratch...')
        else:
            self.logger.info('Starting from scratch...')

    def init_model(self) -> 'AbstractModel':
        """
        Initialize the model from configuration.

        :return: Instantiated model moved to device.
        """
        return self.initializer.get_model().to(self.device)

    def init_optimizer(self) -> torch.optim.Optimizer:
        """
        Initialize the optimizer using the trainer config.

        :return: Instantiated optimizer.
        """
        return self.initializer.initialize(self.train_cfg.optimizer, params=self.model.parameters())

    def save_state(self, epoch: int, ext: str, **kwargs) -> None:
        """
        Save training state including model, optimizer, and any additional metadata.

        :param epoch: Current training epoch.
        :param ext: Extension name (e.g., 'last', 'best').
        :param kwargs: Additional state to include in the checkpoint.
        """
        state = {
            'epoch': epoch,
            'classes': self.train_set.classes,
            'arguments': self.cfg.model.parameters
        } | kwargs
        if self.save_model:
            state |= {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        torch.save(state, self.model_path.replace('.pth', f'_{ext}.pth'))

    def train(self) -> None:
        """
        Run full training loop including evaluation and checkpointing.
        """
        best_epoch, best_score = 0, -np.inf
        for epoch in range(self.epochs.start, self.epochs.end + 1):
            self.logger.log('train/epoch', epoch, epoch)
            start_time = time.time()
            epoch_loss = self.train_step(epoch)
            end_time = time.time()
            if any([np.isnan(v) for v in epoch_loss.values()]):
                self.logger.error(f'Epoch {epoch} has NaN loss values. Stopping training...')
                break
            self.logger.log('train/epoch_time', end_time - start_time, step=epoch)
            self.logger.log_dict('train/loss', epoch_loss, step=epoch)
            self.save_state(epoch, 'last', train_loss=epoch_loss)
            if epoch % self.verbose == 0:
                self.logger.log('evaluate/epoch', epoch, epoch)
                start_time = time.time()
                evaluations = self.evaluator.evaluate(epoch)
                score, _ = evaluations[self.evaluator.main]
                score = score.mean()
                end_time = time.time()
                self.logger.log('evaluate/evaluation_time', end_time - start_time, step=epoch)
                self.logger.log('evaluate/score', score, step=epoch)
                if score >= best_score:
                    best_score = score
                    best_epoch = epoch
                    self.save_state(epoch, 'best', evaluations=evaluations)
                self.logger.info(f'Epoch {best_epoch} has the best score: {best_score}')

    @abstractmethod
    def train_step(self, epoch: int) -> dict:
        """
        Perform a single training epoch over the dataset.

        :param epoch: Current epoch index.
        :return: Dictionary of loss values for logging.
        """
        pass