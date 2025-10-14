from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

from msd.methods.abstract_model import AbstractModel
from msd.methods.msp.base_networks import ResNetDecoder, ResNetEncoder
from msd.methods.msp.dynamics_models import LinearTensorDynamicsLSTSQ


class SeqAELSTSQ(AbstractModel):
    def latent_vector(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return z.mean(dim=(1, 2))

    def sample(self, Z: torch.Tensor) -> torch.Tensor:
        batch_size = Z.shape[0]
        permutation1 = torch.randperm(batch_size, device=Z.device)
        permutation2 = torch.randperm(batch_size, device=Z.device)
        permutations = torch.stack((Z[permutation1], Z[permutation2]))
        coefficients = torch.rand((2, batch_size), device=Z.device)
        coefficients = coefficients / torch.sum(coefficients, dim=0, keepdim=True)
        coefficients = coefficients.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, Z.shape[1], Z.shape[2], Z.shape[3])
        return (coefficients * permutations).sum(dim=0)

    def swap_channels(self, Z1: torch.Tensor, Z2: torch.Tensor, C: Iterable[int]):
        C = torch.tensor(C, dtype=torch.long).to(Z1.device)
        mask_coordinates_to_retain = torch.ones(self.dim_a).cuda()
        mask_coordinates_to_retain[C] = 0
        mask_coordinates_to_retain = mask_coordinates_to_retain.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return (Z1 * mask_coordinates_to_retain + Z2 * (1 - mask_coordinates_to_retain),
                Z2 * mask_coordinates_to_retain + Z1 * (1 - mask_coordinates_to_retain))

    def latent_dim(self) -> int:
        return self.dim_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    def __init__(
            self,
            dim_a,
            dim_m,
            alignment=False,
            ch_x=3,
            k=1.0,
            kernel_size=3,
            change_of_basis=False,
            predictive=True,
            bottom_width=4,
            n_blocks=3,
            *args,
            **kwargs):
        super().__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive
        self.enc = ResNetEncoder(dim_a * dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.dec = ResNetDecoder(ch_x, k=k, kernel_size=kernel_size, bottom_width=bottom_width, n_blocks=n_blocks)
        self.dynamics_model = LinearTensorDynamicsLSTSQ(alignment=alignment)
        if change_of_basis:
            self.change_of_basis = nn.Parameter(torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)

    def _encode_base(self, xs, enc):
        shape = xs.shape
        x = torch.reshape(xs, (shape[0] * shape[1], *shape[2:]))
        H = enc(x)
        H = torch.reshape(H, (shape[0], shape[1], *H.shape[1:]))
        return H

    def encode(self, xs):
        H = self._encode_base(xs, self.enc)
        H = torch.reshape(H, (H.shape[0], H.shape[1], self.dim_m, self.dim_a))
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(self.change_of_basis, 'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        return H

    def phi(self, xs):
        return self._encode_base(xs, self.enc.phi)

    def get_M(self, xs):
        dyn_fn = self.dynamics_fn(xs)
        return dyn_fn.M

    def decode(self, H):
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(torch.linalg.inv(self.change_of_basis), 'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        n, t = H.shape[:2]
        if hasattr(self, "pidec"):
            H = rearrange(H, 'n t d_s d_a -> (n t) d_a d_s')
            H = self.pidec(H)
        else:
            H = rearrange(H, 'n t d_s d_a -> (n t) (d_s d_a)')
        x_next_preds = self.dec(H)
        x_next_preds = torch.reshape(x_next_preds, (n, t, *x_next_preds.shape[1:]))
        return x_next_preds

    def dynamics_fn(self, xs, return_loss=False, fix_indices=None):
        H = self.encode(xs)
        return self.dynamics_model(H, return_loss=return_loss, fix_indices=fix_indices)

    def loss(self, xs, return_reg_loss=True, T_cond=2, reconst=False):
        xs_cond = xs[:, :T_cond]
        xs_pred = self(xs_cond, return_reg_loss=return_reg_loss, n_rolls=xs.shape[1] - T_cond,
                       predictive=self.predictive, reconst=reconst)
        if return_reg_loss:
            xs_pred, reg_losses = xs_pred
        if reconst:
            xs_target = xs
        else:
            xs_target = xs[:, T_cond:] if self.predictive else xs[:, 1:]
        loss = torch.mean(torch.sum((xs_target - xs_pred) ** 2, dim=[2, 3, 4]))
        return (loss, reg_losses) if return_reg_loss else loss

    def __call__(self, xs_cond, return_reg_loss=False, n_rolls=1, fix_indices=None, predictive=True, reconst=False):
        H = self.encode(xs_cond)
        ret = self.dynamics_model(H, return_loss=return_reg_loss, fix_indices=fix_indices)
        if return_reg_loss:
            fn, losses = ret
        else:
            fn = ret

        if predictive:
            H_last = H[:, -1:]
            H_preds = [H] if reconst else []
            array = np.arange(n_rolls)
        else:
            H_last = H[:, :1]
            H_preds = [H[:, :1]] if reconst else []
            array = np.arange(xs_cond.shape[1] + n_rolls - 1)

        for _ in array:
            H_last = fn(H_last)
            H_preds.append(H_last)
        H_preds = torch.cat(H_preds, dim=1)
        x_preds = self.decode(H_preds)
        return (x_preds, losses) if return_reg_loss else x_preds

    def loss_equiv(self, xs, T_cond=2, reduce=False):
        bsize = len(xs)
        xs_cond = xs[:, :T_cond]
        xs_target = xs[:, T_cond:]
        H = self.encode(xs_cond[:, -1:])
        dyn_fn = self.dynamics_fn(xs_cond)

        H_last = H
        H_preds = []
        n_rolls = xs.shape[1] - T_cond
        for _ in np.arange(n_rolls):
            H_last = dyn_fn(H_last)
            H_preds.append(H_last)
        H_pred = torch.cat(H_preds, dim=1)

        dyn_fn.M = dyn_fn.M[torch.arange(-1, bsize - 1)]
        H_last = H
        H_preds_perm = []
        for _ in np.arange(n_rolls):
            H_last = dyn_fn(H_last)
            H_preds_perm.append(H_last)
        H_pred_perm = torch.cat(H_preds_perm, dim=1)

        xs_pred = self.decode(H_pred)
        xs_pred_perm = self.decode(H_pred_perm)
        reduce_dim = (1, 2, 3, 4, 5) if reduce else (2, 3, 4)
        loss = torch.sum((xs_target - xs_pred) ** 2, dim=reduce_dim).detach().cpu().numpy()
        loss_perm = torch.sum((xs_target - xs_pred_perm) ** 2, dim=reduce_dim).detach().cpu().numpy()
        return loss, loss_perm
