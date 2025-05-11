from abc import abstractmethod

import numpy as np
import torch
from matplotlib import pyplot as plt


from msd.data.visualization import compare_sequences, plot_sequence
from msd.evaluations.abstract_evaluator import AbstractEvaluator
from msd.evaluations.latent_exploration.latent_manipulator import SampleManipulator


class LatentVisualizer(AbstractEvaluator):
    def __init__(self, initializer, dataset_type, evaluation_manager, n_samples=1, swap_factors=None, skip_frames=1, plot_components=False):
        super().__init__(initializer, dataset_type, evaluation_manager)
        self.dataset, self.data_loader = self.initializer.get_dataset(dataset_type, loaders=True, labels=True)
        self.method = self._method()
        self.n_samples = n_samples
        self.skip_frames = skip_frames
        self.plot_components = plot_components
        self.swap_factors = swap_factors if swap_factors else set(self.dataset.classes.keys())

    @abstractmethod
    def get_samples(self, Z):
        pass

    @abstractmethod
    def _method(self):
        pass

    @property
    def name(self):
        return f'{self.method}_visualizer'

    def eval(self, epoch):
        X, Ys, Yd = next(iter(self.data_loader))
        if self.n_samples > len(X):
            self.logger.warning(f'Not enough samples in batch to visualize {self.n_samples} samples. Visualizing {len(X)} samples.')
            self.n_samples = len(X)
        X = X.to(self.device)
        mapping = self.evaluation_manager.latent_explorer.get_map(epoch, (X, Ys, Yd))
        with torch.no_grad():
            Z1 = self.model.encode(X)
            idxs, Z2 = self.get_samples(Z1)
            X2 = X[idxs]
            D1 = self.model.decode(Z1).detach().cpu().numpy()
            D2 = self.model.decode(Z2).detach().cpu().numpy()
        X = X.detach().cpu().numpy()
        X2 = X2.detach().cpu().numpy()

        swapped = {}
        for factor in self.swap_factors:
            if factor not in mapping:
                self.logger.error(f'Unable to swap factor {factor}. No latent mapping: {mapping}')
                continue
            subset = mapping[factor]
            _Z1, _Z2 = self.model.swap_channels(Z1, Z2, subset)
            with torch.no_grad():
                _D1, _D2 = self.model.decode(_Z1).detach().cpu().numpy(), self.model.decode(_Z2).detach().cpu().numpy()
            swapped[factor] = {
                'd1': _D1,
                'd2': _D2,
            }

        self.logger.info(f'Visualizing {self.method} at epoch {epoch}')
        for i in range(self.n_samples):
            samples1 = [('x1', X[i]), ('d1', D1[i])]
            samples2 = [('x2', X2[i]), ('d2', D2[i])]
            for factor, swap in swapped.items():
                d1swp, d2swp = swap['d1'][i], swap['d2'][i]
                samples1.append((f'd1_{factor}', d1swp))
                samples2.append((f'd2_{factor}', d2swp))

            if self.plot_components:
                for sample in [samples1, samples2]:
                    for t, s in sample:
                        fig = plot_sequence(s[::self.skip_frames], show=False)
                        self.logger.plot(f'evaluate/latent_{self.method}_{t}_{i}', fig, step=epoch)
                        plt.close(fig)
            else:
                titles1, samples1 = zip(*samples1)
                titles2, samples2 = zip(*samples2)
                fig = compare_sequences(np.array(samples1)[:, ::self.skip_frames], titles1,
                                        np.array(samples2)[:, ::self.skip_frames], titles2,
                                        suptitle=f'Latent {self.method} (epoch {epoch})', show=False, out_path=None)
                self.logger.plot(f'evaluate/latent_{self.method}_{i}', fig, step=epoch)
                plt.close(fig)
        return {'score': 0}, None

class LatentSwapVisualizer(LatentVisualizer):
    def __init__(self, initializer, dataset_type, evaluation_manager, n_samples=1, swap_factors=None, skip_frames=1, plot_components=False):
        super().__init__(initializer, dataset_type, evaluation_manager, n_samples, swap_factors, skip_frames, plot_components)

    def _method(self):
        return 'swap'

    def get_samples(self, Z):
        n = Z.shape[0] // 2
        idxs = np.concatenate([np.arange(n, 2*n), np.arange(n)])
        return idxs, Z[idxs]

class LatentSampleVisualizer(LatentVisualizer):
    def __init__(self, initializer, dataset_type, evaluation_manager, n_samples=1, swap_factors=None, skip_frames=1, plot_components=False):
        super().__init__(initializer, dataset_type, evaluation_manager, n_samples, swap_factors, skip_frames, plot_components)
        self._manipulator = SampleManipulator(self.model)

    def _method(self):
        return 'sample'

    def get_samples(self, Z):
        return self._manipulator.get_samples(Z)
