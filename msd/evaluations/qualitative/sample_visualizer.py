import numpy as np
import torch
from matplotlib import pyplot as plt

from msd.data.visualization import plot_sequences
from msd.evaluations.abstract_evaluator import AbstractEvaluator


class SampleVisualizer(AbstractEvaluator):
    def __init__(self, initializer, dataset_type, evaluation_manager, n_samples):
        super().__init__(initializer, dataset_type, evaluation_manager)
        self.n_samples = n_samples
        self.dataset = self.initializer.get_dataset(dataset_type, loaders=False, labels=False)

    def eval(self, epoch):
        self.logger.info(f'Visualizing samples at epoch {epoch}')
        X = torch.from_numpy(self.dataset[:self.n_samples]).to(self.device)
        Z = self.model.sample(self.model.encode(X))
        with torch.no_grad():
            D = self.model.decode(Z).detach().cpu().numpy()
        X = X.detach().cpu().numpy()
        seq = np.empty((X.shape[0] * 2,) + X.shape[1:], dtype=X.dtype)
        seq[0::2] = X
        seq[1::2] = D
        titles = [f'Original (epoch {epoch})', f'Sampled (epoch {epoch})'] * self.n_samples
        # noinspection PyTypeChecker
        fig = plot_sequences(seq, titles, suptitle=f'Sample & Decode (epoch: {epoch}', out_path=None, show=False)
        self.logger.plot('evaluate/sample', fig, step=epoch)
        plt.close(fig)
        return {'score': 0}, None

    @property
    def name(self):
        return 'sample_visualizer'
