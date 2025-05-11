import numpy as np
import torch

from msd.evaluations.abstract_evaluator import AbstractEvaluator


class AudioReconstruction(AbstractEvaluator):
    def __init__(self, initializer, dataset_type, evaluation_manager, n_samples):
        super().__init__(initializer, dataset_type, evaluation_manager)
        self.dataset = self.initializer.get_dataset(self.dataset_type, loaders=False, labels=False)
        self.n_samples = n_samples
        self.idxs = np.random.choice(len(self.dataset), self.n_samples, replace=False)

    def eval(self, epoch):
        self.logger.info(f'Reconstructing audio at epoch {epoch}')
        X = torch.from_numpy(self.dataset[self.idxs]).to(self.device)
        assert X.shape[0] >= self.n_samples, f'Batch size {X.shape[0]} is less than n_samples {self.n_samples}'
        with torch.no_grad():
            Z = self.model.encode(self.model.preprocess(X))
            D = self.model.postprocess(self.model.decode(Z))
        _X, _D = X[:self.n_samples].detach().cpu().numpy(), D[:self.n_samples].detach().cpu().numpy()
        for i, (x, d) in enumerate(zip(_X, _D)):
            self.logger.log_audio(f'evaluate/reconstruction/{i}_org', x, 16000, epoch)
            self.logger.log_audio(f'evaluate/reconstruction/{i}_rec', d, 16000, epoch)
        return {'score': 0}, None

    @property
    def name(self):
        return 'reconstruction_audio'
