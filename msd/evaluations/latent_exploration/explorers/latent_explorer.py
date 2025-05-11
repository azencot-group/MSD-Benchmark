from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

from msd.configurations.msd_component import MSDComponent

if TYPE_CHECKING:
    from msd.configurations.config_initializer import ConfigInitializer
    from msd.evaluations.evaluation_manager import EvaluationManager


class LatentExplorer(ABC, MSDComponent):
    """
    Abstract base class for latent space exploration modules.

    A LatentExplorer is responsible for extracting and analyzing latent representations
    of samples in a dataset, typically to evaluate disentanglement properties such as
    factor alignment, smooth transitions, or interpretability.

    It supports both full-dataset and batch-level exploration modes, and caches results
    for training and testing modes separately.
    """

    def __init__(
            self,
            initializer: 'ConfigInitializer',
            dataset_type: str,
            evaluation_manager: 'EvaluationManager',
            batch_exploration: bool,
            n_samples: int = None
    ):
        """
        :param initializer: ConfigInitializer object that instantiates components.
        :param dataset_type: Either 'train', 'val', or 'test' to specify which data to use.
        :param evaluation_manager: The parent EvaluationManager controlling this evaluator.
        :param batch_exploration: If True, exploration is run only on the current batch.
        :param n_samples: Optional number of samples to explore; defaults to full dataset.
        """
        self.initializer = initializer
        self.config = self.initializer.config
        self.dataset_type = dataset_type
        self.evaluation_manager = evaluation_manager
        self.model = self.evaluation_manager.model
        self.logger = self.evaluation_manager.logger
        self.device = self.initializer.config.device
        self.dataset, self.data_loader = self.initializer.get_dataset(self.dataset_type, loaders=True, labels=True)
        self.n_samples = len(self.dataset) if n_samples is None else n_samples
        self.batch_exploration = batch_exploration
        self.mapping_train, self.mapping_test = {}, {}

    @abstractmethod
    def eval(self, epoch: int, data_loader: DataLoader):
        """
        Run latent space exploration and return mappings.

        :param epoch: Current epoch number.
        :param data_loader: Iterable batch loader (or single batch if batch_exploration=True).
        :return: Tuple (Z: torch.Tensor, mapping: Dict[feature_name, torch.Tensor])
        """
        pass

    def get_map(self, epoch: int = None, batch=None) -> dict:
        """
        Retrieve or compute the latent space mapping for a given epoch.

        :param epoch: The epoch number (required unless batch_exploration is True).
        :param batch: A single batch of data, required if batch_exploration is enabled.
        :return: Dictionary mapping feature names to latent representations.
        """
        if self.batch_exploration:
            assert batch is not None, "Batch must be provided when batch_exploration is True"
            data_loader = [batch]
        else:
            data_loader = self.data_loader

        if not self.evaluation_manager.testing:
            if epoch not in self.mapping_train or self.batch_exploration:
                self.mapping_train[epoch] = self.eval(epoch, data_loader)
            _, mapping = self.mapping_train[epoch]
        else:
            if epoch not in self.mapping_test or self.batch_exploration:
                self.mapping_test[epoch] = self.eval(epoch, data_loader)
            _, mapping = self.mapping_test[epoch]

        return dict(mapping)
