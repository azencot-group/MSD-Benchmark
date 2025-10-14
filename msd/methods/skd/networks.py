import functools

import numpy as np
import torch
import torchvision.transforms as T
from torch import nn

from msd.methods.abstract_model import AbstractModel
from msd.methods.skd.encoding import skd_decNet, skd_encNet
from msd.methods.skd.koopman_utils import get_sorted_indices, static_dynamic_split, t_to_np
from msd.methods.skd.modules import SKD_KoopmanLayer


class KoopmanCNN(AbstractModel):
    def __init__(self, dropout, noise, w_rec, w_pred, w_eigs, eigs_thresh, static_size, static_mode, dynamic_mode,
                 dynamic_thresh, ball_thresh, device, n_frames, n_channels, n_height, n_width, conv_dim, k_dim,
                 hidden_dim, rnn, lstm_dec_bi, data_type):
        super(KoopmanCNN, self).__init__()
        self.dropout = dropout
        self.noise = noise
        self.w_rec = w_rec
        self.w_pred = w_pred
        self.w_eigs = w_eigs
        self.eigs_thresh = eigs_thresh
        self.static_size = static_size
        self.static_mode = static_mode
        self.dynamic_mode = dynamic_mode
        self.dynamic_thresh = dynamic_thresh
        self.ball_thresh = ball_thresh
        self.device = device
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.n_height = n_height
        self.n_width = n_width
        self.conv_dim = conv_dim
        self.k_dim = k_dim
        self.hidden_dim = hidden_dim
        self.rnn = rnn
        self.lstm_dec_bi = lstm_dec_bi
        self.data_type = data_type

        self.encoder = skd_encNet(self.n_frames, self.n_channels, self.n_height, self.n_width, self.conv_dim,
                              self.k_dim, self.hidden_dim, self.rnn, self.data_type == 'video')
        self.drop = torch.nn.Dropout(self.dropout)
        self.dynamics = SKD_KoopmanLayer(self.n_frames, self.k_dim, self.static_size, self.static_mode, self.eigs_thresh,
                                     self.dynamic_mode, self.dynamic_thresh, self.ball_thresh, self.noise, self.device)
        self.decoder = skd_decNet(self.n_frames, self.n_channels, self.n_height, self.n_width, self.conv_dim,
                              self.k_dim, self.hidden_dim, self.rnn, self.lstm_dec_bi, self.data_type == 'video')

        self.loss_func = nn.MSELoss()

        self.names = ['total', 'rec', 'predict_ambient', 'predict_latent', 'eigs']

        if self.data_type == 'audio':
            from msd.utils.audio import MelSpecEncoder
            self.melspec = MelSpecEncoder()

    def preprocess(self, x):
        if self.data_type == 'audio':
            return self.melspec.encode(x)
        return x

    def postprocess(self, x):
        if self.data_type == 'audio':
            return self.melspec.decode(x)
        return x

    def encode(self, X):
        Z = self.encoder(X)
        Zr = Z.squeeze().reshape(-1, self.n_frames, self.k_dim)
        return Zr

    def latent_vector(self, X):
        outputs = self(X)
        _, Ct_te, Z = outputs[0], outputs[-1], outputs[2]
        Z = t_to_np(Z.reshape(X.shape[0], self.n_frames, self.k_dim))
        C = t_to_np(Ct_te)
        # eig
        D, V = np.linalg.eig(C)
        # project onto V
        ZL = torch.from_numpy(np.real(Z @ V).mean(axis=1)).to(self.device)  # create a single latent code scaler for each sample
        return ZL

    def latent_vector_splited_static_dynamic(self, X):
        outputs = self(X)
        _, Ct_te, Z = outputs[0], outputs[-1], outputs[2]
        Z = t_to_np(Z.reshape(X.shape[0], self.n_frames, self.k_dim))
        C = t_to_np(Ct_te)

        # eig
        D, V = np.linalg.eig(C)

        # static/dynamic split
        I = get_sorted_indices(D, self.static_mode)
        Id, Is = static_dynamic_split(D, I, self.static_mode, self.static_size)
        # project onto V
        # create a single latent code scaler for each sample
        ZL_s = torch.from_numpy(np.real(Z @ V[:, Is]).mean(axis=1)).to(self.device)
        ZL_d = torch.from_numpy(np.real(Z @ V[:, Id]).mean(axis=1)).to(self.device)

        return ZL_s, ZL_d, Is, Id

    def decode(self, Z):
        X_dec = self.decoder(Z)
        return X_dec

    # def sample(self, batch_size):
    def sample(self, Z):
        _Z = Z.reshape(-1, self.k_dim)
        Z2, Ct = self.dynamics(Z)
        batch_size, fsz = Z.shape[0], self.n_frames
        _Z = t_to_np(_Z.reshape(batch_size, fsz, -1))
        C = t_to_np(Ct)
        D, V = np.linalg.eig(C)
        U = np.linalg.inv(V)

        convex_size = 2
        Js = [np.random.permutation(batch_size) for _ in range(convex_size)]  # convex_size permutations
        A = np.random.rand(batch_size, convex_size)  # bsz x 2
        A = A / np.sum(A, axis=1)[:, None]

        Zp = _Z @ V
        Zpi = [np.array([a * z for a, z in zip(A[:, c], Zp[j])]) for c, j in enumerate(Js)]
        Zpc = functools.reduce(lambda a, b: a + b, Zpi)

        Z_sample = torch.from_numpy(np.real(Zpc @ U)).to(self.device)
        return Z_sample

    def swap_channels(self, Z1, Z2, C):
        _, Ct = self.dynamics(Z1.reshape(-1, self.k_dim))
        Ct = t_to_np(Ct)
        D, V = np.linalg.eig(Ct)
        U = np.linalg.inv(V)

        Z1p = t_to_np(Z1) @ V
        Z2p = t_to_np(Z2) @ V

        Z1pc = Z1p[:, :, C]
        Z1p[:, :, C] = Z2p[:, :, C]
        Z2p[:, :, C] = Z1pc

        _Z1p = torch.from_numpy(np.real(Z1p @ U)).to(self.device)
        _Z2p = torch.from_numpy(np.real(Z2p @ U)).to(self.device)

        return _Z1p, _Z2p

    def latent_dim(self):
        return self.k_dim


    def forward(self, X, train=True):
        # input noise added for stability of the Koopman matrix calculation
        if train and self.noise in ["input"]:
            blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3))
            X = torch.concat([torch.concat([blurrer(x).unsqueeze(0) for x in X], dim=0) for _ in range(1)])

        # ----- X.shape: b x t x c x w x h ------
        Z = self.encoder(X)

        # latent both noise - another option to stabilize the numeric calculation of the Koopman matrix
        if train and self.noise in ["latent_both"]:
            Z = Z + 0.25 * torch.rand(Z.shape).to(Z.device)

        Z2, Ct = self.dynamics(Z)
        Z = self.drop(Z)

        # latent reconstruction noise
        if train and self.noise in ["latent_rec"]:
            Z = Z + 0.25 * torch.rand(Z.shape).to(Z.device)

        X_dec = self.decoder(Z)
        X_dec2 = self.decoder(Z2)

        return X_dec, X_dec2, Z, Z2, Ct

    def loss(self, X, outputs):
        X_dec, X_dec2, Z, Z2, Ct = outputs

        # PENALTIES
        a1 = self.w_rec
        a2 = self.w_pred
        a3 = self.w_pred
        a4 = self.w_eigs

        # reconstruction
        E1 = self.loss_func(X, X_dec)

        # Koopman losses
        E2, E3, E4 = self.dynamics.loss(X_dec, X_dec2, Z, Z2, Ct)

        # LOSS
        loss = a1 * E1 + a2 * E2 + a3 * E3 + a4 * E4

        return loss, E1, E2, E3, E4
