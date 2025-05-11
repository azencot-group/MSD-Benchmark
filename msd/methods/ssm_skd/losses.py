import typing as tp

import torch
from torch import nn
from torch.nn import functional as f


class LossModule(nn.Module):
    def __init__(self, name: str, weight: float = 1.0):
        super().__init__()

        self.name = name
        self.weight = weight

    def forward(self, info):
        raise NotImplementedError


class SSM_MSELoss(LossModule):
    def __init__(self, input_key: str, target_key: str, name: str, weight: float = 1):
        super().__init__(name, weight)

        self.input_key = input_key
        self.target_key = target_key

    def forward(self, info):
        return self.weight * f.mse_loss(info[self.input_key], info[self.target_key])


class EigenLoss(LossModule):
    def __init__(self, dynamic_threshold, k_mats_key: str, name: str, weight: float = 1):
        super().__init__(name, weight)

        self.static_loss = nn.MSELoss()
        self.dynamic_loss = nn.Threshold(dynamic_threshold, 0)

        self.k_mats_key = k_mats_key

    def forward(self, info):
        k_mats = info[self.k_mats_key]
        eigvals = torch.linalg.eigvals(k_mats)
        eigvals_real = torch.real(eigvals)
        eigvals_distance_from_one = torch.sqrt((eigvals_real - 1) ** 2 + torch.imag(eigvals) ** 2)

        indices = torch.argsort(eigvals_distance_from_one, dim=1)
        indices_static, indices_dynamic = indices[:, :1], indices[:, 1:]

        eigvals_distance_from_one_static = torch.gather(eigvals_distance_from_one, 1, indices_static)
        eigvals_real_dynamic = torch.gather(eigvals_real, 1, indices_dynamic)

        loss_static = self.static_loss(eigvals_distance_from_one_static,
                                       eigvals_distance_from_one_static.new_zeros(eigvals_distance_from_one_static.shape))
        loss_dynamic = self.dynamic_loss(eigvals_real_dynamic).mean()

        return self.weight * (loss_static + loss_dynamic)


class MultiLoss(nn.Module):
    def __init__(self, losses: tp.List[LossModule]):
        super().__init__()

        self.losses = nn.ModuleList(losses)

    def forward(self, info):
        total_loss, losses = 0, {}

        for loss_module in self.losses:
            module_loss = loss_module(info)
            total_loss += module_loss
            losses[loss_module.name] = module_loss

        return total_loss, losses
