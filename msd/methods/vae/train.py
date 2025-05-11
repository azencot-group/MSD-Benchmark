from msd.methods.abstract_trainer import AbstractTrainer

class VAE_Trainer(AbstractTrainer):
    def __init__(self, initializer):
        super().__init__(initializer)

    def train_step(self, epoch):
        self.model.train()
        epoch_loss = {'loss': 0, 'reconstruction_loss': 0, 'kl_loss': 0, 'sparsity_loss': 0}
        for X in self.train_loader:
            X = X.float().to(self.device)
            Xp = self.model.preprocess(X)
            self.optimizer.zero_grad()
            S, mu, logvar = self.model(Xp)

            loss = self.model.loss_function(S, Xp, mu, logvar)
            loss['loss'].backward()
            self.optimizer.step()

            for k in epoch_loss.keys():
                epoch_loss[k] += loss[k].detach().item()
        epoch_loss = {k: v / len(self.train_loader) for k, v in epoch_loss.items()}
        return epoch_loss
