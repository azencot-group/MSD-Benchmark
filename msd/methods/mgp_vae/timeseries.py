import torch
from torch import nn

from msd.methods.mgp_vae.base_model import MGP_Base


class TSEncoder(MGP_Base):
    def __init__(self, NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W, LSTM_HIDDEN, KEEP_RHO):
        super(TSEncoder, self).__init__(NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W)
        self.LSTM_HIDDEN = LSTM_HIDDEN
        self.KEEP_RHO = KEEP_RHO
        self.lstm = nn.LSTM(NUM_INPUT_CHANNELS, self.LSTM_HIDDEN, batch_first=True) # [BxTxF] -> 128*4*4*NUM_FRAME

        self.dense1 = nn.Linear(in_features=self.NUM_FRAMES * self.LSTM_HIDDEN, out_features=128)
        self.bn1_mlp = nn.BatchNorm1d(num_features=128)

        self.raw_kl1_size = self.NDIM * self.NUM_FRAMES * self.NUM_FRAMES
        self.dense2_1 = nn.Linear(in_features=128, out_features=self.raw_kl1_size)
        self.bn2_1 = nn.BatchNorm1d(num_features=self.raw_kl1_size)

        self.dense2_2 = nn.Linear(in_features=128, out_features=(self.NDIM * self.NDIM) * ((self.NDIM * self.NDIM) + 1) // 2)
        self.bn2_2 = nn.BatchNorm1d(num_features=(self.NDIM * self.NDIM) * ((self.NDIM * self.NDIM) + 1) // 2)
        self.dense2_3 = nn.Linear(in_features=128, out_features=self.NDIM * self.NUM_FRAMES)
        self.bn2_3 = nn.BatchNorm1d(num_features=self.NDIM * self.NUM_FRAMES)
        self.dense2_4 = nn.Linear(in_features=128, out_features=self.NDIM * self.NDIM)
        self.bn2_4 = nn.BatchNorm1d(num_features=self.NDIM * self.NDIM)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def MLP_1(self, x):
        x = self.bn1_mlp(self.elu(self.dense1(x)))
        x = self.bn2_1(self.tanh(self.dense2_1(x)))
        return x

    def MLP_3(self, x):
        x = self.bn1_mlp(self.elu(self.dense1(x)))
        x = self.bn2_3(self.tanh(self.dense2_3(x)))
        return x

    def forward(self, _x):
        BATCH_SIZE = _x.shape[0] # [B,T,C] - T=24, C=13
        x, _ = self.lstm(_x)
        # flatten
        x = x.reshape(x.size()[0], -1)

        # create path sample
        raw_KL1 = self.MLP_1(x).view(-1, self.NDIM, self.NUM_FRAMES, self.NUM_FRAMES)
        KL1 = torch.tril(raw_KL1)

        if BATCH_SIZE == 1:
            KL1_diag = self.matrix_diag_3d(KL1)
        else:
            KL1_diag = self.matrix_diag_4d(KL1)
        det_q1 = torch.prod(KL1_diag * KL1_diag + 1e-6, dim=2)

        muL1 = self.MLP_3(x).view(-1, self.NUM_FRAMES, self.NDIM)

        if self.KEEP_RHO:
            X1 = self.create_path_rho(KL1, muL1)
        else:
            X1 = self.create_path(KL1, muL1)

        return X1, KL1, muL1, det_q1


class TSDecoder(MGP_Base):
    def __init__(self, NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W, LSTM_HIDDEN):
        super(TSDecoder, self).__init__(NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W)
        self.LSTM_HIDDEN = LSTM_HIDDEN
        self.lstm = nn.LSTM(input_size=self.NDIM, hidden_size=self.LSTM_HIDDEN, batch_first=True)
        self.dense = nn.Linear(in_features=self.LSTM_HIDDEN, out_features=self.NUM_INPUT_CHANNELS)
        self.activation = nn.Tanh()

    def forward(self, x1):
        x = x1
        lstm_out, _ = self.lstm(x)  # [BATCH_SIZE, NUM_FRAMES, LSTM_HIDDEN]
        reconstructed_sequence = self.dense(lstm_out)  # [BATCH_SIZE, NUM_FRAMES, NUM_INPUT_CHANNELS]
        reconstructed_sequence = self.activation(reconstructed_sequence)

        return reconstructed_sequence