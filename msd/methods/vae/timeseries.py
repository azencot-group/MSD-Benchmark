from typing import List

import torch.nn as nn

class TSEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int], lstm_hidden_dim: int) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ))
            prev_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.lstm = nn.LSTM(input_size=hidden_dims[-1], hidden_size=lstm_hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(lstm_hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(lstm_hidden_dim * 2, latent_dim)

    def forward(self, X):
        B, T, F = X.shape
        X = X.reshape(B * T, F)
        X = self.feature_extractor(X)
        X = X.view(B, T, -1)
        lstm_out, _ = self.lstm(X)
        lstm_out = lstm_out[:, -1, :]
        mu = self.fc_mu(lstm_out)
        logvar = self.fc_logvar(lstm_out)
        return mu, logvar

class TSDecoder(nn.Module):
    def __init__(self, output_dim: int, latent_dim: int, hidden_dims: List[int], lstm_hidden_dim: int, num_frames: int) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.fc_init = nn.Linear(latent_dim, latent_dim)
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        layers = []
        prev_dim = lstm_hidden_dim * 2
        for h_dim in hidden_dims[::-1]:
            layers.append(nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder_layers = nn.Sequential(*layers)

    def forward(self, Z):
        B = Z.shape[0]
        Z = self.fc_init(Z).unsqueeze(1).repeat(1, self.num_frames, 1)
        lstm_out, _ = self.lstm(Z)
        lstm_out = lstm_out.reshape(B * self.num_frames, -1)
        recon = self.decoder_layers(lstm_out)
        return recon.view(B, self.num_frames, -1)