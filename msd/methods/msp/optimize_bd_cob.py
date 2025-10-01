import torch
import torch.nn as nn
import torch.utils.data
from einops import repeat


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


def optimize_bd_cob(mats, batchsize=32, n_epochs=50, epochs_monitor=10):
    # Optimize change of basis matrix U by minimizing block diagonalization loss

    class ChangeOfBasis(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.U = nn.Parameter(torch.empty(d, d))
            torch.nn.init.orthogonal_(self.U)

        def __call__(self, mat):
            _U = repeat(self.U, "a1 a2 -> n a1 a2", n=mat.shape[0])
            n_mat = torch.linalg.solve(_U, mat) @ _U
            return n_mat

    change_of_basis = ChangeOfBasis(mats.shape[-1]).to(mats.device)
    dataloader = torch.utils.data.DataLoader(
        mats, batch_size=batchsize, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(change_of_basis.parameters(), lr=0.1)
    for ep in range(n_epochs):
        total_loss, total_N = 0, 0
        for mat in dataloader:
            n_mat = change_of_basis(mat)
            n_mat = torch.abs(n_mat)
            n_mat = torch.matmul(n_mat.transpose(-2, -1), n_mat)
            loss = torch.mean(tracenorm_of_normalized_laplacian(n_mat))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mat.shape[0]
            total_N += mat.shape[0]
        if ((ep + 1) % epochs_monitor) == 0:
            print('ep:{} loss:{}'.format(ep, total_loss / total_N))
    return change_of_basis
