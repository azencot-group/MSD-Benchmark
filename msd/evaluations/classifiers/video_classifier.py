import torch
import torch.nn as nn

from msd.evaluations.classifiers.abstract_classifier import AbstractClassifier


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_):
        return self.main(input_)


class ClassifierEncoder(nn.Module):
    def __init__(self, dim, nc, nf=64):
        super(ClassifierEncoder, self).__init__()
        self.dim = dim
        self.nc = nc
        self.nf = nf
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
            nn.Conv2d(nf * 8, dim, 4, 1, 0),
            nn.BatchNorm2d(dim),
            nn.Tanh()
        )

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]

class VideoClassifier(AbstractClassifier):
    def __init__(self, g_dim, channels, hidden_dim, frames, classes):
        super(VideoClassifier, self).__init__()
        self.g_dim = g_dim
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.frames = frames
        self.classes = {k: v for k, v in classes.items() if not v['ignore']}
        self.static_classes = {k: v for k, v in classes.items() if v['type'] == 'static'}
        self.encoder = ClassifierEncoder(self.g_dim, self.channels)
        self.bilstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)

        def create_heads(classes_):
            return nn.ModuleDict({feature_name: nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(True),
                nn.Linear(self.hidden_dim, v['n_classes']),
                nn.Softmax(dim=1)
            ) for feature_name, v in classes_.items()})

        self.static_heads = create_heads(self.static_classes)
        self.heads = create_heads(self.classes)

    def encode(self, x):
        x_shape = x.shape # [N, T, C, H, W]
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        return x_embed.view(x_shape[0], x_shape[1], -1) # [N, T, g_dim]

    def forward(self, x):
        conv_x = self.encode(x)
        lstm_out, _ = self.bilstm(conv_x)
        static_features = [torch.cat((lstm_out[:, t, :self.hidden_dim], lstm_out[:, t, self.hidden_dim:]), dim=1)
                           for t in range(self.frames)]
        static_predictions = {feature_name: torch.stack([head(f) for f in static_features], dim=1)
                              for feature_name, head in self.static_heads.items()}
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        sequence_predictions = {feature_name: head(lstm_out_f) for feature_name, head in self.heads.items()}
        return sequence_predictions, static_predictions
