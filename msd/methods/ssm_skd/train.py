import torch

from msd.methods.abstract_trainer import AbstractTrainer
from msd.methods.ssm_skd.losses import EigenLoss, MultiLoss, SSM_MSELoss


class SsmSkdTrainer(AbstractTrainer):
    def __init__(self, initializer, w_rec, w_pred, w_eigs, dynamic_thresh, gradient_clip_val):
        super().__init__(initializer)

        loss_modules = [SSM_MSELoss('reals', 'decoded', 'rec_loss', w_rec),
                        SSM_MSELoss('reals', 'pred_decoded', 'pred_ambient_loss', w_pred / 2),
                        SSM_MSELoss('latents', 'pred_latents', 'pred_latent_loss', w_pred / 2),
                        EigenLoss(dynamic_thresh, 'k_mats', 'eigen_loss', w_eigs)]
        self.losses = MultiLoss(loss_modules)
        self.gradient_clip_val = gradient_clip_val

    def train_step(self, epoch):
        self.model.train()

        batches = 0
        total_loss = 0
        losses = {'rec_loss': 0, 'pred_ambient_loss': 0, 'pred_latent_loss': 0, 'eigen_loss': 0}

        for X in self.train_loader:
            batches += 1
            X = X.float().to(self.device)

            self.optimizer.zero_grad()
            Xp = self.model.preprocess(X)
            latents, encoder_info, decoded, pred_decoded = self.model(Xp)

            loss_info = {'reals': Xp, 'latents': latents, 'decoded': decoded, 'pred_decoded': pred_decoded,
                         **encoder_info}
            mloss = self.losses(loss_info)
            mloss[0].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_val)
            self.optimizer.step()

            total_loss += mloss[0].item()
            losses = {k: losses[k] + mloss[1][k].item() for k in losses.keys()}

        total_loss /= batches
        losses = {'loss': total_loss, **{k: v / batches for k, v in losses.items()}}

        return losses
