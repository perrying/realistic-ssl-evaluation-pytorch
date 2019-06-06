import torch
import torch.nn as nn
import torch.nn.functional as F

class MixMatch(nn.Module):
    def __init__(self, temperature, n_augment, augment_func, alpha):
        super().__init__()
        self.T = temperature
        self.K = n_augment
        self.augment = augment_func
        self.beta_distirb = torch.distributions.beta.Beta(alpha, alpha)

    def sharpen(self, y):
        y = y.pow(1/self.T)
        return y / y.sum(1,keepdim=True)

    def onehot(self, y, n_classes=10):
        return torch.eye(n_classes).to(y.device)[y].float()

    def forward(self, x, y, model, mask):
        model.update_batch_stats(False)
        l_x, u_x = x[mask!=1], x[mask!=0]
        l_x_hat = self.augment(l_x)
        u_x_hat = []
        # K augmentation
        for _ in range(self.K):
            u_x_hat.append(self.augment(u_x))
        # make prediction label
        y_hat = sum([model(u_x_hat[i]).softmax(1) for i in range(len(u_x_hat))]) / self.K
        y_hat = self.sharpen(y_hat)
        y_ul = torch.cat([y, y_hat], 0)
        x_ul = torch.cat([l_x_hat, *u_x_hat], 0)
        # mixup
        index = torch.randperm(x_ul.shape[0])
        shuffled_x_ul, shuffled_y_ul = x_ul[index], y_ul[index]
        lam = self.beta_distirb.sample().item()
        lam = max(lam, 1-lam)
        mixed_x = lam * x_ul + (1-lam) * shuffled_x_ul
        mixed_y = lam * self.onehot(y_ul) + (1-lam) * shuffled_y_ul.softmax(1)
        # mean squared error
        loss = F.mse_loss(model(mixed_x), mixed_y, reduction="none").sum(1).mean()
        model.update_batch_stats(True)
        return loss
