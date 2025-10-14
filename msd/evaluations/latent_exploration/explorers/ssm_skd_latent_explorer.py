import torch

from msd.evaluations.latent_exploration.explorers.latent_explorer import LatentExplorer


class SsmSkdLatentExplorer(LatentExplorer):
    def __init__(self, initializer, dataset_type, evaluation_manager, batch_exploration, n_samples=None):
        super().__init__(initializer, dataset_type, evaluation_manager, batch_exploration, n_samples)

    def eval(self, epoch, data_loader):
        return None, (self.eval_multifactor(data_loader, 'static') |
                      self.eval_multifactor(data_loader, 'dynamic'))

    def eval_duofactor(self):
        static_factor = [f[0] for f in self.dataset.classes.items() if f[1]['type'] == 'static'][0]
        dynamic_factor = [f[0] for f in self.dataset.classes.items() if f[1]['type'] == 'dynamic'][0]

        return {static_factor: list(range(self.model.k_dim)),
                dynamic_factor: list(range(self.model.k_dim, 2 * self.model.k_dim))}

    def eval_multifactor(self, data_loader, swap_type):
        factors = [f[0] for f in self.dataset.classes.items() if f[1]['type'] == swap_type]
        mapping = {f: [] for f in factors}
        no_swap_acc = self.evaluate_factorial_swap(data_loader, swap_type, factors, None)

        for c in range(self.model.k_dim):
            acc = self.evaluate_factorial_swap(data_loader, swap_type, factors, c)
            diff = acc - no_swap_acc
            if diff.max() > 0:
                mapping[factors[diff.argmax()]].append(c if swap_type == 'static' else c + self.model.k_dim)

        return mapping

    def evaluate_factorial_swap(self, data_loader, swap_type, factors, coordinate_to_retain):
        judge = self.evaluation_manager.get_judge()
        labels_original = {f: [] for f in factors}
        labels_after_swap = {f: [] for f in factors}

        coordinates_to_swap = torch.arange(self.model.k_dim).to(self.device)
        if coordinate_to_retain is not None:
            coordinates_to_swap = torch.cat([coordinates_to_swap[:coordinate_to_retain],
                                             coordinates_to_swap[coordinate_to_retain + 1:]])
        if swap_type == 'dynamic':
            coordinates_to_swap += self.model.k_dim

        for _ in range(10):
            for data, static_labels, dynamic_labels in data_loader:
                data, labels = data.to(self.device), static_labels | dynamic_labels

                with torch.no_grad():
                    z = self.model.encode(data)
                    s = self.model.sample(z)
                    swapped, _ = self.model.swap_channels(z, s, coordinates_to_swap)
                    swapped = self.model.decode(swapped)
                    swapped_labels = judge(swapped)

                for f in factors:
                    labels_original[f].append(labels[f].to(self.device))
                    labels_after_swap[f].append(swapped_labels[f].to(self.device))

        return torch.Tensor([(torch.cat(labels_original[f]) == torch.cat(labels_after_swap[f])).float().mean().item()
                             for f in factors])
