from abc import abstractmethod

import numpy as np
import torch

from msd.evaluations.abstract_evaluator import AbstractEvaluator
from msd.evaluations.latent_exploration.latent_manipulator import SampleManipulator


class LatentAudio(AbstractEvaluator):
    def __init__(self, initializer, dataset_type, evaluation_manager):
        super().__init__(initializer, dataset_type, evaluation_manager)
        self.dataset, self.data_loader = self.initializer.get_dataset(dataset_type, loaders=True, labels=True)
        self.method = self._method()

    @abstractmethod
    def get_samples(self, Z):
        pass

    @abstractmethod
    def _method(self):
        pass

    @property
    def name(self):
        return f'{self.method}_audio'

    def eval(self, epoch):
        X, Ys, Yd = next(iter(self.data_loader))
        X = X.to(self.device)
        mapping = self.evaluation_manager.latent_explorer.get_map(epoch, (X, Ys, Yd))
        with torch.no_grad():
            Z1 = self.model.encode(self.model.preprocess(X))
            idxs, Z2 = self.get_samples(Z1)
            X2 = X[idxs]
            D1 = self.model.postprocess(self.model.decode(Z1))
            D2 = self.model.postprocess(self.model.decode(Z2))

        samples1 = [('a1', X[0].detach().cpu().numpy()), ('d1', D1[0].detach().cpu().numpy())]
        samples2 = [('a2', X2[0].detach().cpu().numpy()), ('d2', D2[0].detach().cpu().numpy())]

        for factor, subset in mapping.items():
            _Z1, _Z2 = self.model.swap_channels(Z1, Z2, subset)
            with torch.no_grad():
                _D1, _D2 = self.model.postprocess(self.model.decode(_Z1)), self.model.postprocess(self.model.decode(_Z2))
            samples1.append((f'd1_swap_{factor}', _D1[0].detach().cpu().numpy()))
            samples2.append((f'd2_swap_{factor}', _D2[0].detach().cpu().numpy()))
        self.logger.info(f'Logging audio data ({self.method}) at epoch {epoch}')
        for (name, x) in samples1 + samples2:
            self.logger.log_audio(f'evaluate/reconstruction/{name}', x, 16000, epoch)
        return {'score': 0}, None

class LatentAudioSwap(LatentAudio):
    def __init__(self, initializer, dataset_type, evaluation_manager):
        super().__init__(initializer, dataset_type, evaluation_manager)

    def _method(self):
        return 'swap'

    def get_samples(self, Z):
        n = Z.shape[0] // 2
        idxs = np.concatenate([np.arange(n, 2*n), np.arange(n)])
        return idxs, Z[idxs]

class LatentAudioSample(LatentAudio):
    def __init__(self, initializer, dataset_type, evaluation_manager):
        super().__init__(initializer, dataset_type, evaluation_manager)
        self._manipulator = SampleManipulator(self.model)

    def _method(self):
        return 'sample'

    def get_samples(self, Z):
        return self._manipulator.get_samples(Z)
