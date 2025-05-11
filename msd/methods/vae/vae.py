from typing import List, TypeVar

import torch
from torch import nn
import torch.nn.functional as F

from msd.methods.abstract_model import AbstractModel
from msd.methods.vae.timeseries import TSEncoder, TSDecoder
from msd.methods.vae.video import VideoEncoder, VideoDecoder

Tensor = TypeVar('torch.tensor')

class VAE(AbstractModel):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List, lstm_hidden_dim: int, num_frames: int, width: int, height: int, beta: float=1., sparsity_weight: float=0, mode: str = 'video') -> None:
        super(VAE, self).__init__()
        self.C, self.T, self.W, self.H = in_channels, num_frames, width, height
        self._latent_dim = latent_dim
        self.beta = beta
        self.sparsity_weight = sparsity_weight
        self.mode = mode.lower()

        if self.mode == 'video':
            self._encoder = VideoEncoder(in_channels, latent_dim, hidden_dims, lstm_hidden_dim, width, height)
            self._decoder = VideoDecoder(in_channels, latent_dim, hidden_dims, lstm_hidden_dim, num_frames, width, height)
        else:
            self._encoder = TSEncoder(in_channels, latent_dim, hidden_dims, lstm_hidden_dim)
            self._decoder = TSDecoder(in_channels, latent_dim, hidden_dims, lstm_hidden_dim, num_frames)

        if self.mode == 'audio':
            from msd.utils.audio import MelSpecEncoder
            self.melspec = MelSpecEncoder()

    def preprocess(self, x):
        if self.mode == 'audio':
            return self.melspec.encode(x)
        return x

    def postprocess(self, x):
        if self.mode == 'audio':
            return self.melspec.decode(x)
        return x


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=mu.device)
        return mu + eps * std

    def forward(self, X):
        mu, logvar = self._encoder(X)
        Z = self.reparameterize(mu, logvar) # [batch X 128]
        S = self.decode(Z)
        return S, mu, logvar

    def encode(self, X):
        return self.reparameterize(*self._encoder(X))

    def decode(self, Z):
        return self._decoder(Z)

    def sample(self, Z):
        bs = Z.shape[0]
        Z = torch.randn(bs, self._latent_dim).to(self.device)
        return Z

    def swap_channels(self, Z1, Z2, C):
        Z1c = Z1[:, C]
        Z1, Z2 = Z1.clone(), Z2.clone()
        Z1[:, C] = Z2[:, C]
        Z2[:, C] = Z1c
        return Z1, Z2

    def latent_vector(self, X):
        return self.encode(X)

    def latent_dim(self):
        return self._latent_dim

    def loss_function(self, x_recon, x, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        sparsity_loss = self.sparsity_weight * torch.sum(torch.abs(mu)) / x.shape[0]
        loss = recon_loss + self.beta * kl_loss + sparsity_loss
        return {'loss': loss, 'reconstruction_loss': recon_loss, 'kl_loss': kl_loss, 'sparsity_loss': sparsity_loss}
