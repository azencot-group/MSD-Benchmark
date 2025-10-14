import torch
from torch import nn

from msd.methods.skd.koopman_utils import get_unique_num


class skd_conv(nn.Module):
    def __init__(self, nin, nout):
        super(skd_conv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_):
        return self.net(input_)


class skd_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(skd_upconv, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_):
        return self.net(input_)



class SKD_KoopmanLayer(nn.Module):

    def __init__(self, n_frames, k_dim, static_size, static_mode, eigs_thresh,
                 dynamic_mode, dynamic_thresh, ball_thresh, noise, device):
        super(SKD_KoopmanLayer, self).__init__()

        self.run = None
        self.n_frames = n_frames
        self.k_dim = k_dim
        self.device = device
        self.noise = noise

        # eigen values arguments
        self.static = static_size
        self.mode = static_mode
        self.eigs_tresh = eigs_thresh ** 2
        self.dynamic_loss_mode = dynamic_mode
        self.ball_thresh = ball_thresh

        # loss functions
        self.loss_func = nn.MSELoss()
        self.dynamic_threshold_loss = nn.Threshold(dynamic_thresh, 0)

    def forward(self, Z):
        # Z is in b * t x c x 1 x 1
        Zr = Z.squeeze().reshape(-1, self.n_frames, self.k_dim)

        if self.training and self.noise in ["latent"]:
            Zr = Zr + 0.003 * torch.rand(Zr.shape).to(Zr.device)

        # split
        X, Y = Zr[:, :-1], Zr[:, 1:]

        # solve linear system (broadcast)
        Ct = torch.linalg.lstsq(X.reshape(-1, self.k_dim), Y.reshape(-1, self.k_dim)).solution

        # predict (broadcast)
        Y2 = X @ Ct
        Z2 = torch.cat((X[:, 0].unsqueeze(dim=1), Y2), dim=1)

        assert (torch.sum(torch.isnan(Y2)) == 0)

        return Z2.reshape(Z.shape), Ct

    def loss(self, X_dec, X_dec2, Z, Z2, Ct):
        # predict ambient
        E1 = self.loss_func(X_dec, X_dec2)

        # predict latent
        E2 = self.loss_func(Z, Z2)

        E3_static, E3_dynamic = 0, 0

        # Koopman operator constraints (disentanglement)
        D = torch.linalg.eigvals(Ct)

        Dn = torch.real(torch.conj(D) * D)
        Dr = torch.real(D)
        Db = torch.sqrt((Dr - torch.ones(len(Dr)).to(Dr.device)) ** 2 + torch.imag(D) ** 2)

        # ----- static loss ----- #
        Id, new_static_number = None, None
        if self.mode == 'norm':
            I = torch.argsort(Dn)
            new_static_number = get_unique_num(D, I, self.static)
            Is, Id = I[-new_static_number:], I[:-new_static_number]
            Dns = torch.index_select(Dn, 0, Is)
            E3_static = self.loss_func(Dns, torch.ones(len(Dns)).to(Dns.device))

        elif self.mode == 'real':
            I = torch.argsort(Dr)
            new_static_number = get_unique_num(D, I, self.static)
            Is, Id = I[-new_static_number:], I[:-new_static_number]
            Drs = torch.index_select(Dr, 0, Is)
            E3_static = self.loss_func(Drs, torch.ones(len(Drs)).to(Drs.device))

        elif self.mode == 'ball':
            I = torch.argsort(Db)
            # we need to pick the first indexes from I and not the last
            new_static_number = get_unique_num(D, torch.flip(I, dims=[0]), self.static)
            Is, Id = I[:new_static_number], I[new_static_number:]
            Dbs = torch.index_select(Db, 0, Is)
            E3_static = self.loss_func(Dbs, torch.zeros(len(Dbs)).to(Dbs.device))

        elif self.mode == 'space_ball':
            I = torch.argsort(Db)
            # we need to pick the first indexes from I and not the last
            new_static_number = get_unique_num(D, torch.flip(I, dims=[0]), self.static)
            Is, Id = I[:new_static_number], I[new_static_number:]
            Dbs = torch.index_select(Db, 0, Is)
            E3_static = torch.mean(self.sp_b_thresh(Dbs))

        elif self.mode == 'none':
            E3_static = torch.zeros(1).to(self.device)

        # report unique number
        if self.run:
            self.run['general/static_eigen_vals_number'].log(new_static_number)

        if self.dynamic_loss_mode == 'strict':
            Dnd = torch.index_select(Dn, 0, Id)
            E3_dynamic = self.loss_func(Dnd, self.eigs_tresh * torch.ones(len(Dnd)).to(Dnd.device))

        elif self.dynamic_loss_mode == 'thresh' and self.mode == 'none':
            I = torch.argsort(Dn)
            new_static_number = get_unique_num(D, I, self.static)
            Is, Id = I[-new_static_number:], I[:-new_static_number]
            Dnd = torch.index_select(Dn, 0, Id)
            E3_dynamic = torch.mean(self.dynamic_threshold_loss(Dnd))

        elif self.dynamic_loss_mode == 'thresh':
            Dnd = torch.index_select(Dn, 0, Id)
            E3_dynamic = torch.mean(self.dynamic_threshold_loss(Dnd))

        elif self.dynamic_loss_mode == 'ball':
            Dbd = torch.index_select(Db, 0, Id)
            E3_dynamic = torch.mean(
                (Dbd < self.ball_thresh).float() * ((torch.ones(len(Dbd))).to(Dbd.device) * 2 - Dbd))

        elif self.dynamic_loss_mode == 'real':
            Drd = torch.index_select(Dr, 0, Id)
            E3_dynamic = torch.mean(self.dynamic_threshold_loss(Drd))

        if self.dynamic_loss_mode == 'none':
            E3 = E3_static
        else:
            E3 = E3_static + E3_dynamic

        return E1, E2, E3
