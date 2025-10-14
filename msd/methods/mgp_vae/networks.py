import numpy as np
import torch
import torch.nn as nn

from msd.methods.abstract_model import AbstractModel
from msd.methods.mgp_vae.base_model import MGP_Base
from msd.methods.mgp_vae.covariance_fns import covariance_function
from msd.methods.mgp_vae.timeseries import TSEncoder, TSDecoder
from msd.methods.mgp_vae.video import VideoEncoder, VideoDecoder


class MGP_VAE(MGP_Base, AbstractModel):
    def __init__(self, NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W, NUM_FEA, FEA, mean_start, mean_end, KEEP_RHO, fac, LSTM_HIDDEN=-1, mode='video'):
        super(MGP_VAE, self).__init__(NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W)
        self.mode = mode
        if self.mode == 'video':
            self.encoder = VideoEncoder(NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W, KEEP_RHO)
            self.decoder = VideoDecoder(NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W)
        else:
            self.encoder = TSEncoder(NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W, LSTM_HIDDEN, KEEP_RHO)
            self.decoder = TSDecoder(NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W, LSTM_HIDDEN)

        self.NUM_INPUT_CHANNELS = NUM_INPUT_CHANNELS
        self.NDIM = NDIM
        self.NUM_FRAMES = NUM_FRAMES
        self.H = H
        self.W = W
        self.fac = fac
        self.NUM_FEA = NUM_FEA
        self.FEA = FEA
        self.mean_start = mean_start
        self.mean_end = mean_end
        self.FEA_DIM = int(self.NDIM / self.NUM_FEA)

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

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, Z):
        return self.decoder(Z)

    def latent_vector(self, X):
        return self.encode(X).mean(axis=1)

    def sample(self, Z):
        batch_size = Z.shape[0]
        sigma0, sigma_p_inv, det_p = self.setup_pz()
        sigma0 = torch.stack([sigma0 for _ in range(batch_size)]).to(self.device)
        mu0 = self.get_prior_mean(batch_size=batch_size).permute(0, 2, 1)
        KL0 = torch.tril(sigma0)
        Z = self.create_path(KL0, mu0)
        return Z

    def swap_channels(self, Z1, Z2, C):
        Z1c = Z1[:, :, C]
        Z1, Z2 = Z1.clone(), Z2.clone()
        Z1[:, :, C] = Z2[:, :, C]
        Z2[:, :, C] = Z1c
        return Z1, Z2

    def latent_dim(self):
        return self.NDIM

    def forward(self, x):
        x1, KL1, muL1, det_q1 = self.encoder(x)
        dec = self.decoder(x1)

        return dec, x1, KL1, muL1, det_q1

    def setup_pz(self):
        sigmas, dets = [], []

        for f in range(self.NUM_FEA):
            prior = self.FEA[f]

            if len(prior.split('_')) == 2:
                p_type, H = prior.split('_')[0], float(prior.split('_')[1])

            else:
                p_type, H = prior, None

            for n in range(self.FEA_DIM):
                raw_sigma, det_ = covariance_function(p_type, self.NUM_FRAMES, H, self.fac)
                sigma = torch.from_numpy(raw_sigma).float()
                det = torch.tensor(det_)
                sigmas.append(sigma)
                dets.append(det)

        sigma_p = torch.stack(sigmas).to(self.device)
        sigma_p_inv = torch.inverse(sigma_p)
        det_p = torch.stack(dets).to(self.device)

        return sigma_p, sigma_p_inv, det_p

    def get_prior_mean(self, batch_size):
        mean = torch.zeros(batch_size, self.NDIM, self.NUM_FRAMES).to(self.device)

        for i in range(self.NUM_FRAMES):
            for f in range(self.NUM_FEA):
                if self.mean_start[f] is not None and self.mean_end[f] is not None:
                    mean[:, f * self.FEA_DIM:(f + 1) * self.FEA_DIM, i] = (self.mean_start[f] + i *
                                                                           ((self.mean_end[f] - self.mean_start[f]) /
                                                                            (self.NUM_FRAMES - 1)))
        return mean
