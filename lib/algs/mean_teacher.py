import torch
import torch.nn as nn
import torch.nn.functional as F

class MT(nn.Module):
    def __init__(self, model, ema_factor):
        super().__init__()
        self.model = model
        self.ema_factor = ema_factor
        self.global_step = 0

    def forward(self, x, y, model, mask):
        self.global_step += 1
        y_hat = self.model(x)
        model.update_batch_stats(False)
        _y = model(x) # recompute y since y as input is detached
        model.update_batch_stats(True)
        return (F.mse_loss(y_hat.softmax(1).detach(), _y.softmax(1), reduction="none").mean(1) * mask).mean()

    def moving_average(self, parameters):
        ema_factor = min(1 - 1 / (self.global_step), self.ema_factor)
        for emp_p, p in zip(self.model.parameters(), parameters):
            emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data
