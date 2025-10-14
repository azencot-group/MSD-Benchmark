import pandas as pd
import torch

from msd.evaluations.latent_exploration.explorers.latent_explorer import LatentExplorer


class PredictorLatentExplorerStaticDynamic(LatentExplorer):
    def __init__(self, initializer, dataset_type, evaluation_manager, batch_exploration, n_samples=None):
        super().__init__(initializer, dataset_type, evaluation_manager, batch_exploration, n_samples)

    def eval(self, epoch, data_loader):
        static_factors = [c[0] for c in self.dataset.classes.items() if c[1]['type'] == 'static']
        dynamic_factors = [c[0] for c in self.dataset.classes.items() if c[1]['type'] == 'dynamic']
        Y = []
        for X, Ys, Yd in data_loader:
            X, Y = X.to(self.device), Ys | Yd
            with torch.no_grad():
                Z_s, Z_d, Is, Id = self.model.latent_vector_splited_static_dynamic(X)
        accuracy_s, mapping_s = self.get_mapping_from_Z(Y, Z_s, epoch, static_factors)
        accuracy_d, mapping_d = self.get_mapping_from_Z(Y, Z_d, epoch, dynamic_factors)
        accuracy = pd.concat([accuracy_s, accuracy_d])
        # get the final indexes
        final_map = {}
        for k in mapping_s:
            if k not in list(final_map.keys()):
                final_map[k] = [Is[i] for i in mapping_s[k]]
        for k in mapping_d:
            if k not in list(final_map.keys()):
                final_map[k] = [Id[i] for i in mapping_d[k]]

        self.logger.log_table('evaluate/mapping', pd.DataFrame({'subset': {k: str(v) for k, v in final_map.items()}}),
                              step=epoch)
        return accuracy, final_map

    def get_mapping_from_Z(self, Y, Z, epoch, factors):
        accuracy = pd.DataFrame(columns=['score'])
        importance = pd.DataFrame(columns=factors)
        for f in factors:
            predictor = self.evaluation_manager.create_predictor()
            predictor.fit(Z.cpu(), Y[f].cpu())
            accuracy.loc[f] = predictor.score(Z.cpu(), Y[f].cpu())
            _importance = predictor.feature_importances_
            importance[f] = _importance / _importance.sum()
        self.logger.log_table('evaluate/latent_explorer', importance, step=epoch)
        self.logger.log_table('evaluate/latent_explorer_accuracy', accuracy, step=epoch)
        _mapping = importance.idxmax(axis=1).to_dict()
        mapping = {f: [i for i, v in enumerate(_mapping.values()) if v == f] for f in factors if f in _mapping.values()}
        return accuracy, mapping
