import torch

from msd.methods.abstract_trainer import AbstractTrainer

class SKD_Trainer(AbstractTrainer):
    def __init__(self, initializer, clip_norm=1.0):
        super().__init__(initializer)
        self.clip_norm = clip_norm
        self.scheduler = self.initializer.initialize(self.train_cfg.scheduler, optimizer=self.optimizer)

    def init_model(self):
        model = self.initializer.get_model().to(self.device)
        return model

    def train_step(self, epoch):
        self.model.train()
        losses = {'loss': 0, 'rec': 0, 'rec_pred': 0, 'latent_pred': 0, 'eig': 0}
        for X in self.train_loader:
            X = X.float().to(self.device)
            Xp = self.model.preprocess(X)
            self.optimizer.zero_grad()
            outputs = self.model(Xp)
            model_loss = self.model.loss(Xp, outputs)

            model_loss[0].backward()
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), max_norm=self.clip_norm)
            self.optimizer.step()

            losses['loss'] += model_loss[0].item()
            losses['rec'] += model_loss[1].item()
            losses['rec_pred'] += model_loss[2].item()
            losses['latent_pred'] += model_loss[3].item()
            losses['eig'] += model_loss[4].item()
        losses = {k: v for k, v in losses.items()}
        self.scheduler.step()
        return losses

    @staticmethod
    def score(evaluations):
        return evaluations[0]
