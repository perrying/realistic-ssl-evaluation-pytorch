import torch
import torch.nn as nn
import torch.nn.functional as F

class MixMatch(nn.Module):
    def __init__(self, temperature, n_augment, alpha):
        super().__init__()
        self.T = temperature
        self.K = n_augment
        self.beta_distirb = torch.distributions.beta.Beta(alpha, alpha)

    def sharpen(self, y):
        y = y.pow(1/self.T)
        return y / y.sum(1,keepdim=True)

    def forward(self, x, y, model, mask):
        # NOTE: this implementaion uses mixup for only unlabeled data
        model.update_batch_stats(False)
        u_x = x[mask == 1]
        # K augmentation and make prediction labels
        u_x_hat = [u_x for _ in range(self.K)]
        y_hat = sum([model(u_x_hat[i]).softmax(1) for i in range(len(u_x_hat))]) / self.K
        y_hat = self.sharpen(y_hat)
        y_hat = y_hat.repeat(len(u_x_hat), 1)
        # mixup
        u_x_hat = torch.cat(u_x_hat, 0)
        index = torch.randperm(u_x_hat.shape[0])
        shuffled_u_x_hat, shuffled_y_hat = u_x_hat[index], y_hat[index]
        lam = self.beta_distirb.sample().item()
        # lam = max(lam, 1-lam)
        mixed_x = lam * u_x_hat + (1-lam) * shuffled_u_x_hat
        mixed_y = lam * y_hat + (1-lam) * shuffled_y_hat.softmax(1)
        # mean squared error
        loss = F.mse_loss(model(mixed_x), mixed_y)
        model.update_batch_stats(True)
        return loss
