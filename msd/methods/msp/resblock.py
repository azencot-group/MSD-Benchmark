import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.lazy
import torch.nn.utils.parametrize as P


class Emb2D(nn.modules.lazy.LazyModuleMixin, nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.emb = torch.nn.parameter.UninitializedParameter()

    def __call__(self, x):
        if torch.nn.parameter.is_lazy(self.emb):
            _, h, w = x.shape[1:]
            self.emb.materialize((self.dim, h, w))
            self.emb.data = positionalencoding2d(self.dim, h, w)
        emb = torch.tile(self.emb[None].to(x.device), [x.shape[0], 1, 1, 1])
        x = torch.cat([x, emb], dim=1)
        return x


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class WeightStandarization(nn.Module):
    def forward(self, weight):
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return weight


class Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 kernel_size=3,
                 padding=None,
                 activation=F.relu,
                 resample=None,
                 group_norm=True,
                 skip_connection=True,
                 posemb=False):
        super(Block, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.pe = Emb2D() if posemb else lambda x: x

        in_ch_conv = in_channels + self.pe.dim if posemb else in_channels
        self.skip_connection = skip_connection
        self.activation = activation
        self.resample = resample
        initializer = torch.nn.init.xavier_uniform_
        if self.resample is None or self.resample == 'up':
            hidden_channels = out_channels if hidden_channels is None else hidden_channels
        else:
            hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_ch_conv, hidden_channels,
                            kernel_size=kernel_size, padding=padding)
        self.c2 = nn.Conv2d(hidden_channels, out_channels,
                            kernel_size=kernel_size, padding=padding)
        initializer(self.c1.weight, math.sqrt(2))
        initializer(self.c2.weight, math.sqrt(2))
        P.register_parametrization(
            self.c1, 'weight', WeightStandarization())
        P.register_parametrization(
            self.c2, 'weight', WeightStandarization())

        if group_norm:
            self.b1 = nn.GroupNorm(min(32, in_channels), in_channels)
            self.b2 = nn.GroupNorm(min(32, hidden_channels), hidden_channels)
        else:
            self.b1 = self.b2 = lambda x: x
        if self.skip_connection:
            self.c_sc = nn.Conv2d(in_ch_conv, out_channels,
                                  kernel_size=1, padding=0)
            initializer(self.c_sc.weight)

    def residual(self, x):
        x = self.b1(x)
        x = self.activation(x)
        if self.resample == 'up':
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.pe(x)
        x = self.c1(x)
        x = self.b2(x)
        x = self.activation(x)
        x = self.c2(x)
        if self.resample == 'down':
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        # Upsample -> Conv
        if self.resample == 'up':
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
            x = self.pe(x)
            x = self.c_sc(x)
        elif self.resample == 'down':
            x = self.pe(x)
            x = self.c_sc(x)
            x = F.avg_pool2d(x, 2)
        else:
            x = self.pe(x)
            x = self.c_sc(x)
        return x

    def __call__(self, x):
        if self.skip_connection:
            return self.residual(x) + self.shortcut(x)
        else:
            return self.residual(x)
