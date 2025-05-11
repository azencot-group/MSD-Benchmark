import torch
from torch import nn

from msd.methods.abstract_model import AbstractModel


class SsmSkd(AbstractModel):
    def __init__(self, in_dim, k_dim, hidden_dim, data_type):
        super().__init__()
        self.k_dim = k_dim
        self.CLIP_VALUE_MIN = 1e-5
        self.CLIP_VALUE_MAX = 1e8
        self.data_type = data_type.lower()
        self.encoder = Encoder(in_dim, k_dim, self.data_type)
        self.decoder = Decoder(k_dim, in_dim, self.data_type, hidden_dim)

        self.bottleneck = KoopmanBottleneck(k_dim)

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

    def latent_vector(self, x):
        z = self.encode(x)
        z_static, z_dynamic = self.extract_static_dynamic_latents(z)

        return torch.real(torch.cat((z_static, z_dynamic), dim=2).mean(dim=1)).type(torch.float32)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, z):
        latents, bottleneck_info = self.bottleneck(z)
        k_mats = bottleneck_info['k_mats']
        eigvals, eigvecs = torch.linalg.eig(k_mats)
        i_eigvecs = torch.linalg.inv(eigvecs)
        z = latents.type(torch.complex128) @ eigvecs @ i_eigvecs

        batch_size = z.shape[0]
        permutation1 = torch.randperm(batch_size, device=latents.device)
        permutation2 = torch.randperm(batch_size, device=latents.device)
        permutations = torch.stack((z[permutation1], z[permutation2]))
        coefficients = torch.rand((2, batch_size), device=latents.device)
        coefficients = coefficients / torch.sum(coefficients, dim=0, keepdim=True)
        coefficients = coefficients.unsqueeze(2).unsqueeze(3).expand(-1, -1, latents.shape[1], latents.shape[2])

        return torch.real((coefficients * permutations).sum(dim=0)).type(torch.float32)

    def swap_channels(self, z1, z2, c):
        c = torch.tensor(c, dtype=torch.long).to(z1.device)
        c_static, c_dynamic = c[c < self.k_dim], c[c >= self.k_dim] - self.k_dim
        mask_static_coordinates_to_retain = torch.ones(self.k_dim, dtype=torch.complex128).cuda()
        mask_static_coordinates_to_retain[c_static] = 0
        mask_static_coordinates_to_retain = mask_static_coordinates_to_retain.unsqueeze(0).unsqueeze(0)
        mask_dynamic_coordinates_to_retain = torch.ones(self.k_dim, dtype=torch.complex128).cuda()
        mask_dynamic_coordinates_to_retain[c_dynamic] = 0
        mask_dynamic_coordinates_to_retain = mask_dynamic_coordinates_to_retain.unsqueeze(0).unsqueeze(0)

        z1_static, z1_dynamic = self.extract_static_dynamic_latents(z1)
        z2_static, z2_dynamic = self.extract_static_dynamic_latents(z2)

        return (torch.real(z1_static * mask_static_coordinates_to_retain +
                           z2_static * (1 - mask_static_coordinates_to_retain) +
                           z1_dynamic * mask_dynamic_coordinates_to_retain +
                           z2_dynamic * (1 - mask_dynamic_coordinates_to_retain)).type(torch.float32),
                torch.real(z2_static * mask_static_coordinates_to_retain +
                           z1_static * (1 - mask_static_coordinates_to_retain) +
                           z2_dynamic * mask_dynamic_coordinates_to_retain +
                           z1_dynamic * (1 - mask_dynamic_coordinates_to_retain)).type(torch.float32))

    def extract_static_dynamic_latents(self, z):
        latents, bottleneck_info = self.bottleneck(z)
        k_mats = bottleneck_info['k_mats']
        eigvals, eigvecs = torch.linalg.eig(k_mats)
        i_eigvecs = torch.linalg.inv(eigvecs)
        eigvals_distance_from_one = torch.sqrt((torch.real(eigvals) - 1) ** 2 + torch.imag(eigvals) ** 2)
        indices = torch.argsort(eigvals_distance_from_one, dim=1)
        indices_static, indices_dynamic = indices[:, :1], indices[:, 1:]
        z_proj = latents.type(torch.complex128) @ eigvecs

        z_static = (z_proj.gather(2, indices_static.unsqueeze(1).expand(-1, latents.shape[1], -1)) @
                    i_eigvecs.gather(1, indices_static.unsqueeze(2).expand(-1, -1, latents.shape[2])))
        z_dynamic = (z_proj.gather(2, indices_dynamic.unsqueeze(1).expand(-1, latents.shape[1], -1)) @
                     i_eigvecs.gather(1, indices_dynamic.unsqueeze(2).expand(-1, -1, latents.shape[2])))

        return z_static, z_dynamic

    def latent_dim(self):
        return 2 * self.k_dim

    def forward(self, x):
        latents, encoder_info = self.forward_encode(x)
        latents_after_dropout = torch.nn.functional.dropout(latents, 0.2)
        pred_latents = encoder_info['pred_latents']
        decoded = self.forward_decode(latents_after_dropout)
        pred_decoded = self.forward_decode(pred_latents)

        return latents, encoder_info, decoded, pred_decoded

    def forward_encode(self, x):
        latents = self.encoder(x)
        latents, bottleneck_info = self.bottleneck(latents)

        return latents, bottleneck_info

    def forward_decode(self, latents):
        return self.decoder(latents)


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, dataset_type):
        super().__init__()

        self.dataset_type = dataset_type.lower()

        if self.dataset_type == 'video':
            self.cnn = nn.Sequential(Conv(in_dim, 32, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     Conv(32, 64, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     Conv(64, 128, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     Conv(128, 256, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     Conv(256, out_dim, 4, 1, 0, True, nn.Tanh()))
            in_dim = out_dim


        self.lstm = nn.LSTM(in_dim, out_dim, batch_first=True)

    def forward(self, x):
        if self.dataset_type == 'video':
            shape = x.shape
            x = x.reshape(shape[0] * shape[1], *shape[2:])
            x = self.cnn(x)
            x = x.reshape(shape[0], shape[1], -1)

        return self.lstm(x)[0]


class Conv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, bn, activation):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
                                  nn.BatchNorm2d(out_dim) if bn else nn.Identity(),
                                  activation)

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, dataset_type, hidden_dim=None):
        super().__init__()

        self.dataset_type = dataset_type.lower()

        self.lstm = nn.LSTM(in_dim, hidden_dim if self.dataset_type == 'video' else out_dim, batch_first=True)

        if self.dataset_type == 'video':
            self.cnn = nn.Sequential(ConvTranspose(hidden_dim, 256, 4, 1, 0, True, nn.LeakyReLU(0.2, inplace=True)),
                                     ConvTranspose(256, 128, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     ConvTranspose(128, 64, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     ConvTranspose(64, 32, 4, 2, 1, True, nn.LeakyReLU(0.2, inplace=True)),
                                     ConvTranspose(32, out_dim, 4, 2, 1, False, nn.Sigmoid()))

    def forward(self, x):
        x = self.lstm(x)[0]

        if self.dataset_type == 'video':
            shape = x.shape
            x = x.reshape(shape[0] * shape[1], shape[2], 1, 1)
            x = self.cnn(x)
            x = x.reshape(shape[0], shape[1], *x.shape[1:])

        return x


class ConvTranspose(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, bn, activation):
        super().__init__()

        self.conv_t = nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding),
                                    nn.BatchNorm2d(out_dim) if bn else nn.Identity(),
                                    activation)

    def forward(self, x):
        return self.conv_t(x)


class KoopmanBottleneck(nn.Module):
    def __init__(self, k_dim):
        super().__init__()

        self.k_dim = k_dim

    # noinspection PyMethodMayBeStatic
    def forward(self, z):
        x, y = z[:, :-1].type(torch.float64), z[:, 1:].type(torch.float64)

        k_mats = torch.linalg.lstsq(x, y).solution

        pred_y = x @ k_mats
        pred = torch.cat((x[:, 0].unsqueeze(dim=1).type(torch.float32), pred_y.type(torch.float32)), dim=1)

        return z, {'k_mats': k_mats, 'pred_latents': pred}
