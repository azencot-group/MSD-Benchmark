from typing import List

import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List, lstm_hidden_dim: int, width: int, height: int) -> None:
        super().__init__()
        self.C, self.H, self.W = in_channels, height, width
        self.down_factor = 2 ** len(hidden_dims)
        layers = []
        prev_dim = in_channels
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(prev_dim, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            prev_dim = h_dim
        self.cnn = nn.Sequential(*layers)
        self.flatten_dim = hidden_dims[-1] * (self.H // self.down_factor) * (self.W // self.down_factor)
        self.lstm = nn.LSTM(input_size=self.flatten_dim, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(lstm_hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(lstm_hidden_dim * 2, latent_dim)

    def forward(self, X):
        B, T, C, H, W = X.shape
        X = X.reshape(B * T, C, H, W)
        X = self.cnn(X)
        X = X.reshape(B, T, -1)

        lstm_out, _ = self.lstm(X)
        lstm_out = lstm_out[:, -1, :]

        mu = self.fc_mu(lstm_out)
        logvar = self.fc_logvar(lstm_out)
        return mu, logvar


class VideoDecoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List, lstm_hidden_dim: int, num_frames: int, width: int, height: int) -> None:
        super().__init__()
        self.C, self.T, self.W, self.H = in_channels, num_frames, width, height
        self.down_factor = 2 ** len(hidden_dims)
        self.fc_init = nn.Linear(latent_dim, latent_dim)
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc_lstm = nn.Linear(lstm_hidden_dim * 2, hidden_dims[-1] * (self.H // self.down_factor) * (self.W // self.down_factor))
        hidden_dims = hidden_dims[::-1]
        layers = []
        prev_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(nn.ConvTranspose2d(prev_dim, h_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.ConvTranspose2d(prev_dim, in_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*layers)

    def forward(self, Z):
        B = Z.shape[0]
        Z = self.fc_init(Z)
        Z = Z.unsqueeze(1).repeat(1, self.T, 1)

        lstm_out, _ = self.lstm(Z)
        lstm_out = self.fc_lstm(lstm_out)
        lstm_out = lstm_out.view(B * self.T, -1, self.H // self.down_factor, self.W // self.down_factor)

        S = self.deconv(lstm_out)
        S = S.view(B, self.T, self.C, self.H, self.W)
        return S
