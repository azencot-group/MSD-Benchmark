from torch import nn

from msd.methods.skd.modules import skd_conv, skd_upconv


class skd_encNet(nn.Module):
    def __init__(self, n_frames, n_channels, n_height, n_width, conv_dim, k_dim, hidden_dim, rnn, with_cnn):
        super(skd_encNet, self).__init__()
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.n_height = n_height
        self.n_width = n_width
        self.conv_dim = conv_dim
        self.k_dim = k_dim
        self.hidden_dim = hidden_dim
        self.rnn = rnn
        self.with_cnn = with_cnn
        in_dim = self.n_channels
        if self.with_cnn:
            self.c1 = skd_conv(self.n_channels, self.conv_dim)
            self.c2 = skd_conv(self.conv_dim, self.conv_dim * 2)
            self.c3 = skd_conv(self.conv_dim * 2, self.conv_dim * 4)
            self.c4 = skd_conv(self.conv_dim * 4, self.conv_dim * 8)
            self.c5 = nn.Sequential(
                nn.Conv2d(self.conv_dim * 8, self.k_dim, 4, 1, 0),
                nn.BatchNorm2d(self.k_dim),
                nn.Tanh()
            )
            in_dim = k_dim


        if self.rnn in ["encoder", "both"]:
            self.lstm = nn.LSTM(in_dim, self.k_dim, batch_first=True, bias=True,
                                bidirectional=False)

    def forward(self, _x):
        if self.with_cnn:
            x = _x.reshape(-1, self.n_channels, self.n_height, self.n_width)
            h1 = self.c1(x)
            h2 = self.c2(h1)
            h3 = self.c3(h2)
            h4 = self.c4(h3)
            h5 = self.c5(h4)
            x = h5
            x = x.reshape(-1, self.n_frames, self.k_dim)
        else:
            x = _x

        # lstm
        if self.rnn in ["encoder", "both"]:
            x = self.lstm(x)[0]
        if self.with_cnn:
            x = x.reshape(-1, self.k_dim, 1, 1)

        return x


class skd_decNet(nn.Module):
    def __init__(self, n_frames, n_channels, n_height, n_width, conv_dim, k_dim, hidden_dim, rnn, lstm_dec_bi, with_cnn):
        super(skd_decNet, self).__init__()

        self.n_frames = n_frames
        self.n_channels = n_channels
        self.n_height = n_height
        self.n_width = n_width
        self.conv_dim = conv_dim
        self.koopman_dim = k_dim
        self.lstm_hidden_size = hidden_dim
        self.lstm_dec_bi = lstm_dec_bi
        self.rnn = rnn
        self.with_cnn = with_cnn
        if rnn in ["decoder", "both"]:
            self.lstm = nn.LSTM(self.koopman_dim, self.lstm_hidden_size if self.with_cnn else self.n_channels, batch_first=True, bias=True,
                                bidirectional=lstm_dec_bi)
        if self.with_cnn:
            self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(self.lstm_hidden_size * (2 if self.lstm_dec_bi else 1),
                                   self.conv_dim * 8, 4, 1, 0),
                nn.BatchNorm2d(self.conv_dim * 8),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.upc2 = skd_upconv(self.conv_dim * 8, self.conv_dim * 4)
            self.upc3 = skd_upconv(self.conv_dim * 4, self.conv_dim * 2)
            self.upc4 = skd_upconv(self.conv_dim * 2, self.conv_dim)
            self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(self.conv_dim, self.n_channels, 4, 2, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        # lstm
        if self.rnn in ["decoder", "both"]:
            if self.with_cnn:
                x = (self.lstm(x.reshape(-1, self.n_frames, self.koopman_dim))[0]
                     .reshape(-1, self.lstm_hidden_size * (2 if self.lstm_dec_bi else 1), 1, 1))
            else:
                x = self.lstm(x)[0]
        if self.with_cnn:
            d1 = self.upc1(x)
            d2 = self.upc2(d1)
            d3 = self.upc3(d2)
            d4 = self.upc4(d3)
            output = self.upc5(d4)
            x = output.view(-1, self.n_frames, self.n_channels, self.n_height, self.n_width)

        return x