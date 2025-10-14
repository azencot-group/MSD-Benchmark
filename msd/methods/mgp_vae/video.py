import torch
from torch import nn

from msd.methods.mgp_vae.base_model import MGP_Base, My_Tanh


class VideoEncoder(MGP_Base):
    def __init__(self, NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W, KEEP_RHO):
        super(VideoEncoder, self).__init__(NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W)
        self.KEEP_RHO = KEEP_RHO

        self.conv1 = nn.Conv2d(in_channels=NUM_INPUT_CHANNELS, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=64)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=64)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=128)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(num_features=128)

        # layers for MLP
        if self.H == 64:
            self.dense1 = nn.Linear(in_features=self.NUM_FRAMES * 128 * 4 * 4, out_features=128)
        elif self.H == 32:
            self.dense1 = nn.Linear(in_features=self.NUM_FRAMES * 128 * 2 * 2, out_features=128)
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
        BATCH_SIZE = _x.shape[0] # [B,T,C,H,W]
        x = _x.reshape(BATCH_SIZE * self.NUM_FRAMES, self.NUM_INPUT_CHANNELS, self.H, self.W)
        x = self.bn1(self.elu(self.conv1(x)))
        x = self.bn2(self.elu(self.conv2(x)))
        x = self.bn3(self.elu(self.conv3(x)))
        x = self.bn4(self.elu(self.conv4(x)))
        x = self.bn5(self.elu(self.conv5(x)))
        x = self.bn6(self.elu(self.conv6(x)))
        x = self.bn7(self.elu(self.conv7(x)))
        x = self.bn8(self.elu(self.conv8(x)))

        x = x.reshape(BATCH_SIZE, self.NUM_FRAMES, 128, x.size()[2], x.size()[3]) # [BxT,128,4,4]

        # flatten
        x = x.reshape(x.size()[0], -1)

        # create path sample
        raw_KL1 = self.MLP_1(x).reshape(-1, self.NDIM, self.NUM_FRAMES, self.NUM_FRAMES)
        KL1 = torch.tril(raw_KL1)

        if BATCH_SIZE == 1:
            KL1_diag = self.matrix_diag_3d(KL1)
        else:
            KL1_diag = self.matrix_diag_4d(KL1)
        det_q1 = torch.prod(KL1_diag * KL1_diag, dim=2)

        muL1 = self.MLP_3(x).reshape(-1, self.NUM_FRAMES, self.NDIM)

        if self.KEEP_RHO:
            X1 = self.create_path_rho(KL1, muL1)
        else:
            X1 = self.create_path(KL1, muL1)

        return X1, KL1, muL1, det_q1


class VideoDecoder(MGP_Base):
    def __init__(self, NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W):
        super(VideoDecoder, self).__init__(NUM_INPUT_CHANNELS, NDIM, NUM_FRAMES, H, W)

        factor = self.NDIM

        self.dense1 = nn.Linear(in_features=self.NUM_FRAMES * factor, out_features=self.NUM_FRAMES * 8 * 8 * 16, bias=True)
        self.bn_dense1 = nn.BatchNorm1d(num_features=self.NUM_FRAMES * 8 * 8 * 16)
        self.dense2 = nn.Linear(in_features=self.NUM_FRAMES * 8 * 8 * 16, out_features=self.NUM_FRAMES * 8 * 8 * 64, bias=True)
        self.bn_dense2 = nn.BatchNorm1d(num_features=self.NUM_FRAMES * 8 * 8 * 64)

        self.conv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.conv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=16)
        self.conv6 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=16)
        self.conv7 = nn.ConvTranspose2d(in_channels=16, out_channels=self.NUM_INPUT_CHANNELS,
                                        kernel_size=3, stride=1, padding=1)

        self.my_tanh = My_Tanh()

        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def forward(self, x1):
        BATCH_SIZE = x1.shape[0]
        x = x1

        # flatten
        x = x.view(BATCH_SIZE, self.NUM_FRAMES * self.NDIM)
        x = self.bn_dense1(self.elu(self.dense1(x)))
        x = self.bn_dense2(self.elu(self.dense2(x)))

        x = x.view(BATCH_SIZE * self.NUM_FRAMES, 64, 8, 8)

        x = self.bn1(self.elu(self.conv1(x)))
        x = self.bn2(self.elu(self.conv2(x)))
        x = self.bn3(self.elu(self.conv3(x)))
        x = self.bn4(self.elu(self.conv4(x)))
        x = self.bn5(self.elu(self.conv5(x)))
        if self.H == 64:
            x = self.bn6(self.elu(self.conv6(x)))

        x = self.my_tanh(self.conv7(x))
        x = x.view(BATCH_SIZE, self.NUM_FRAMES, self.NUM_INPUT_CHANNELS, self.H, self.W)

        return x