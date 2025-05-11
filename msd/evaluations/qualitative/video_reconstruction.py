import numpy as np
import torch
from matplotlib import pyplot as plt

from msd.data.visualization import plot_sequences
from msd.evaluations.abstract_evaluator import AbstractEvaluator

class VideoReconstruction(AbstractEvaluator):
    def __init__(self, initializer, dataset_type, evaluation_manager, n_samples):
        super().__init__(initializer, dataset_type, evaluation_manager)
        self.dataset = self.initializer.get_dataset(self.dataset_type, loaders=False, labels=False)
        self.n_samples = n_samples
        self.idxs = np.random.choice(len(self.dataset), self.n_samples, replace=False)

    def eval(self, epoch):
        self.logger.info(f'Visualizing reconstruction at epoch {epoch}')
        X = torch.from_numpy(self.dataset[self.idxs]).to(self.device)
        assert X.shape[0] >= self.n_samples, f'Batch size {X.shape[0]} is less than n_samples {self.n_samples}'
        with torch.no_grad():
            Z = self.model.encode(X)
            D = self.model.decode(Z)
        _X, _D = X[:self.n_samples].detach().cpu().numpy(), D[:self.n_samples].detach().cpu().numpy()
        seq = np.empty((_X.shape[0] * 2,) + _X.shape[1:], dtype=_X.dtype)
        seq[0::2] = _X
        seq[1::2] = _D
        titles = [f'Original (epoch {epoch})', f'Reconstructed (epoch {epoch})'] * self.n_samples
        # noinspection PyTypeChecker
        fig = plot_sequences(seq, titles, suptitle=f'Reconstruction (epoch {epoch})', out_path=None, show=False)
        self.logger.plot('evaluate/reconstruction', fig, step=epoch)
        plt.close(fig)
        return {'score': 0}, None

    @property
    def name(self):
        return 'reconstruction_visualizer'
