import torch

from msd.methods.abstract_trainer import AbstractTrainer
from msd.methods.msp.optimize_bd_cob import optimize_bd_cob


class MspTrainer(AbstractTrainer):
    def __init__(self, initializer, T_cond=2):
        super().__init__(initializer)
        self.T_cond = T_cond

    def train(self) -> None:
        super().train()

    def train_step(self, epoch):
        if hasattr(self.model, 'change_of_basis'):
            delattr(self.model, 'change_of_basis')

        self.model.train()

        batches = 0
        losses = {'loss': 0, 'loss_bd': 0, 'loss_orth': 0}

        for X in self.train_loader:
            batches += 1
            _X = X.to(self.device)
            loss, (loss_bd, loss_orth, _) = self.model.loss(
                _X, T_cond=self.T_cond, return_reg_loss=True, reconst=False)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses = {
                'loss': losses['loss'] + loss.item(),
                'loss_bd': losses['loss_bd'] + loss_bd.item(),
                'loss_orth': losses['loss_orth'] + loss_orth.item()
            }

        if epoch % self.verbose == 0:
            Ms = []
            count = 0
            with torch.no_grad():
                for images in self.train_loader:
                    images = images.to(self.device)
                    Ms.append(self.model.get_M(images).detach())
                    count += 1
                    if count > 100:
                        break
            self.model.Ms = torch.cat(Ms, dim=0)
            self.model.CofB = optimize_bd_cob(self.model.Ms, self.logger, n_epochs=50)
            self.model.change_of_basis = torch.nn.Parameter(self.model.CofB.U)

        return {k: v / batches for k, v in losses.items()}
