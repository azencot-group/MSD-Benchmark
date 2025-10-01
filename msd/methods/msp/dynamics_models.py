import einops
import numpy as np
import pytorch_pfn_extras as ppe
import torch
import torch.nn as nn


def make_identity(N, D, device):
    if N is None:
        return torch.Tensor(np.array(np.eye(D))).to(device)
    else:
        return torch.Tensor(np.array([np.eye(D)] * N)).to(device)


def make_identity_like(A):
    assert A.shape[-2] == A.shape[-1]  # Ensure A is a batch of squared matrices
    device = A.device
    shape = A.shape[:-2]
    eye = torch.eye(A.shape[-1], device=device)[(None,) * len(shape)]
    return eye.repeat(*shape, 1, 1)


def make_diagonal(vecs):
    vecs = vecs[..., None].repeat(*([1, ] * len(vecs.shape)), vecs.shape[-1])
    return vecs * make_identity_like(vecs)


# Calculate Normalized Laplacian
def tracenorm_of_normalized_laplacian(A):
    D_vec = torch.sum(A, dim=-1)
    D = make_diagonal(D_vec)
    L = D - A
    inv_A_diag = make_diagonal(1 / torch.sqrt(1e-10 + D_vec))
    L = torch.matmul(inv_A_diag, torch.matmul(L, inv_A_diag))
    sigmas = torch.linalg.svdvals(L)
    return torch.sum(sigmas, dim=-1)


def _rep_M(M, T):
    return einops.repeat(M, "n a1 a2 -> n t a1 a2", t=T)


def _loss(A, B):
    return torch.sum((A - B) ** 2)


def _solve(A, B):
    ATA = A.transpose(-2, -1) @ A
    ATB = A.transpose(-2, -1) @ B
    return torch.linalg.solve(ATA, ATB)


def loss_bd(M_star, alignment):
    # Block Diagonalization Loss
    S = torch.abs(M_star)
    STS = torch.matmul(S.transpose(-2, -1), S)
    if alignment:
        laploss_sts = tracenorm_of_normalized_laplacian(torch.mean(STS, 0))
    else:
        laploss_sts = torch.mean(tracenorm_of_normalized_laplacian(STS), 0)
    return laploss_sts


def loss_orth(M_star):
    # Orthogonalization of M
    I = make_identity_like(M_star)
    return torch.mean(torch.sum((I - M_star @ M_star.transpose(-2, -1)) ** 2, dim=(-2, -1)))


class LinearTensorDynamicsLSTSQ(nn.Module):
    class DynFn(nn.Module):
        def __init__(self, M):
            super().__init__()
            self.M = M

        def __call__(self, H):
            return H @ _rep_M(self.M, T=H.shape[1])

        def inverse(self, H):
            M = _rep_M(self.M, T=H.shape[1])
            return torch.linalg.solve(M, H.transpose(-2, -1)).transpose(-2, -1)

    def __init__(self, alignment=True):
        super().__init__()
        self.alignment = alignment

    def __call__(self, H, return_loss=False, fix_indices=None):
        # Regress M.
        # Note: backpropagation is disabled when fix_indices is not None.

        # H0.shape = H1.shape [n, t, s, a]
        H0, H1 = H[:, :-1], H[:, 1:]
        # The difference between the the time shifted components
        loss_internal_0 = _loss(H0, H1)
        ppe.reporting.report({'loss_internal_0': loss_internal_0.item()})
        _H0 = H0.reshape(H0.shape[0], -1, H0.shape[-1])
        _H1 = H1.reshape(H1.shape[0], -1, H1.shape[-1])
        if fix_indices is not None:
            # Note: backpropagation is disabled.
            dim_a = _H0.shape[-1]
            active_indices = np.array(list(set(np.arange(dim_a)) - set(fix_indices)))
            _M_star = _solve(_H0[:, :, active_indices], _H1[:, :, active_indices])
            M_star = make_identity(_H1.shape[0], _H1.shape[-1], _H1.device)
            M_star[:, active_indices[:, np.newaxis], active_indices] = _M_star
        else:
            M_star = _solve(_H0, _H1)
        dyn_fn = self.DynFn(M_star)
        loss_internal_T = _loss(dyn_fn(H0), H1)
        ppe.reporting.report({'loss_internal_T': loss_internal_T.item()})

        # M_star is returned in the form of module, not the matrix
        if return_loss:
            losses = (
                loss_bd(dyn_fn.M, self.alignment),
                loss_orth(dyn_fn.M),
                loss_internal_T
            )
            return dyn_fn, losses
        else:
            return dyn_fn
