from typing import Dict

import numpy as np

from msd.configurations.config_initializer import ConfigInitializer
from msd.methods.abstract_model import AbstractModel
from msd.methods.abstract_trainer import AbstractTrainer


class ClassifierTrainer(AbstractTrainer):
    """
    Trainer class for supervised classification models used in disentanglement evaluation.

    Supports both static and sequence-level losses. Expects a classifier that returns
    two prediction dictionaries (sequence-level and frame-level).
    """

    def __init__(self, initializer: ConfigInitializer):
        """
        :param initializer: ConfigInitializer instance providing model, loss, and data setup.
        """
        super().__init__(initializer, return_labels=True)
        self.static_loss = self.initializer.initialize(self.train_cfg.static_loss)
        self.sequence_loss = self.initializer.initialize(self.train_cfg.sequence_loss)

    def init_model(self) -> AbstractModel:
        """
        Initialize the model using the training dataset's class configuration.

        :return: Instantiated model on the correct device.
        """
        classes = self.train_set.classes
        return self.initializer.get_model(classes=classes).to(self.device)

    def train_step(self, epoch: int) -> Dict[str, float]:
        """
        Perform one full training epoch over the dataset.

        :param epoch: Current training epoch index (unused but may be logged externally).
        :return: Dictionary of average loss values across the epoch.
        """
        self.model.train()
        losses = {}
        for X, Ys, Yd in self.train_loader:
            X = self.model.preprocess(X.to(self.device))
            Y = (Ys | Yd) # merge static and dynamic labels
            self.optimizer.zero_grad()
            P_sequence, P_static = self.model(X)
            # Compute loss per head
            batch_losses = ({f'{k}_sequence': self.sequence_loss(P_sequence[k], Y[k].to(self.device))
                             for k in P_sequence.keys()} |
                            {f'{k}_static': self.static_loss(P_static[k], Y[k].to(self.device))
                             for k in P_static.keys()})
            batch_loss = sum(batch_losses.values())
            batch_losses['total'] = batch_loss

            batch_loss.backward()
            self.optimizer.step()

            for k, v in batch_losses.items():
                v_cpu = v.detach().cpu().numpy()
                losses[k] = losses.get(k, []) + [v_cpu]

        losses = {k: np.mean(v) for k, v in losses.items()}
        return losses
