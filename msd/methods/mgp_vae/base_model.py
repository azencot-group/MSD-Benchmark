import numpy as np
import torch
from torch import nn


class My_Tanh(nn.Module):
    def __init__(self):
        super(My_Tanh, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 0.5 * (self.tanh(x) + 1)

class MGP_Base(nn.Module):
    def __init__(self, NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W):
        super(MGP_Base, self).__init__()
        self.NUM_INPUT_CHANNELS = NUM_INPUT_CHANNELS
        self.NDIM = NDIM
        self.NUM_FRAMES = NUM_FRAMES
        self.H = H
        self.W = W

    def matrix_diag_4d(self, diagonal):
        bs = diagonal.shape[0]
        diagonal = diagonal.view(diagonal.size()[0] * diagonal.size()[1], diagonal.size()[2], diagonal.size()[3])
        result = torch.diagonal(diagonal, dim1=-2, dim2=-1)

        result = result.view(bs, self.NDIM, self.NUM_FRAMES)
        return result

    @staticmethod
    def matrix_diag_3d(diagonal):
        result = torch.diagonal(diagonal, dim1=-2, dim2=-1)
        return result

    def create_path(self, K_L1, mu_L1):
        BATCH_SIZE = K_L1.shape[0]
        inc_L1 = torch.randn(BATCH_SIZE, self.NUM_FRAMES, self.NDIM, device=K_L1.device)
        X1 = torch.einsum('ikj,ijlk->ilj', inc_L1, K_L1) + mu_L1
        return X1.contiguous()

    def create_path_rho(self, K_L1, mu_L1, rho=0.5):
        BATCH_SIZE = K_L1.shape[0]
        c11 = torch.randn(BATCH_SIZE, self.NUM_FRAMES, device=K_L1.device)
        c12 = (rho * c11) + (np.sqrt(max(1.0 - rho ** 2, 1e-6)) * torch.randn(BATCH_SIZE, self.NUM_FRAMES, device=K_L1.device))
        c21 = torch.randn(BATCH_SIZE, self.NUM_FRAMES, device=K_L1.device)
        c22 = (rho * c21) + (np.sqrt(max(1.0 - rho ** 2, 1e-6)) * torch.randn(BATCH_SIZE, self.NUM_FRAMES, device=K_L1.device))

        inc_L1 = torch.stack([c11, c12, c21, c22], dim=2)
        X1 = torch.einsum('ikj,ijlk->ilj', inc_L1, K_L1) + mu_L1

        return X1