import torch
from torch import nn

from msd.methods.abstract_trainer import AbstractTrainer


def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()


def KL_loss_L1_without_mean(sigma_p_inv, sigma_q, mu_q, det_p, det_q):
    # sigma_p_inv: (d, nlen, nlen), det_p: (d)
    # sigma_q: (batch_size, d, nlen, nlen), mu_q: (batch_size, d, nlen)

    l1 = torch.einsum('kij,mkji->mk', sigma_p_inv, sigma_q)      # tr(sigma_p_inv sigma_q)
    l2 = torch.einsum('mki,mki->mk', mu_q, torch.einsum('kij,mkj->mki', sigma_p_inv, mu_q))      # <mu_q, sigma_p_inv, mu_q>
    loss = torch.sum(l1 + l2 + torch.log(det_p) - torch.log(det_q), dim=1)  # KL divergence b/w two Gaussian distri
    return loss


def KL_loss_L1(sigma_p_inv, sigma_q, mu_q, mu_p, det_p, det_q):
    # sigma_p_inv: (n_dim, n_frames, n_frames), det_p: (d)
    # sigma_q: (batch_size, n_dim, n_frames, n_frames), mu_q: (batch_size, d, nlen)

    l1 = torch.einsum('kij,mkji->mk', sigma_p_inv, sigma_q)      # tr(sigma_p_inv sigma_q)
    l2 = torch.einsum('mki,mki->mk', mu_p - mu_q,
                      torch.einsum('kij,mkj->mki', sigma_p_inv, mu_p - mu_q))      # <mu_q, sigma_p_inv, mu_q>
    loss = torch.sum(l1 + l2 + torch.log(det_p) - torch.log(det_q), dim=1)
    return loss

class MGP_VAE_Trainer(AbstractTrainer):
    def __init__(self, initializer, kl_beta, zero_mean, clip_norm=1.0):
        super().__init__(initializer)
        self.BATCH_SIZE = self.train_loader.batch_size
        self.kl_beta = kl_beta
        self.zero_mean = zero_mean
        self.clip_norm = clip_norm
        self.mse_loss = self.initializer.initialize(self.train_cfg.mse_loss)
        self.mse_mean = torch.nn.MSELoss(reduction='mean')
        self.scheduler = self.initializer.initialize(self.train_cfg.scheduler, optimizer=self.optimizer)

        _, self.sigma_p_inv, self.det_p = self.model.setup_pz()

    def init_model(self):
        model = self.initializer.get_model().to(self.device)
        model.apply(weights_init)
        return model

    def train_step(self, epoch):
        self.model.train()
        losses = {'KL': 0, 'MSE(sum)': 0, 'MSE(mean)': 0}
        for X in self.train_loader:
            X = X.float().to(self.device)
            self.optimizer.zero_grad()
            Xp = self.model.preprocess(X)
            dec, Z, KL1, muL1, det_q1 = self.model(Xp)
            mse_loss = self.mse_loss(Xp, dec)
            mse_mean = self.mse_mean(Xp, dec)

            sigma_q1 = torch.einsum('ijkl,ijlm->ijkm', KL1, torch.einsum('ijkl->ijlk', KL1))
            mul1_transpose = torch.transpose(muL1, dim0=1, dim1=2)
            if self.zero_mean:
                mu_p_transpose = self.model.get_prior_mean(batch_size=Xp.shape[0])
                kl_loss1 = KL_loss_L1(self.sigma_p_inv, sigma_q1, mul1_transpose, mu_p_transpose, self.det_p, det_q1)
            else:
                kl_loss1 = KL_loss_L1_without_mean(self.sigma_p_inv, sigma_q1, mul1_transpose, self.det_p, det_q1)
            kl_loss = torch.mean(kl_loss1)
            kl_loss = kl_loss * self.kl_beta
            total_loss = mse_loss + kl_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), max_norm=self.clip_norm)
            self.optimizer.step()

            losses['KL'] += kl_loss1.sum().item()
            losses['MSE(sum)'] += mse_loss.item()
            losses['MSE(mean)'] += mse_mean.item()
        losses = {k: v / len(self.train_set) for k, v in losses.items()}
        losses['total'] = losses['KL'] + losses['MSE(sum)']
        self.scheduler.step()
        return losses
