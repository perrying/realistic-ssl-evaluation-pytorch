import torch
import torch.nn as nn
import torch.nn.functional as F

class VAT(nn.Module):
    def __init__(self, eps=1.0, xi=1e-6, n_iteration=1):
        super().__init__()
        self.eps = eps
        self.xi = xi
        self.n_iteration = n_iteration

    def kld(self, q_logit, p_logit):
        q = q_logit.softmax(1)
        qlogp = (q * self.__logsoftmax(p_logit)).sum(1)
        qlogq = (q * self.__logsoftmax(q_logit)).sum(1)
        return qlogq - qlogp

    def normalize(self, v):
        v = v / (1e-12 + self.__reduce_max(v.abs(), range(1, len(v.shape))))
        v = v / (1e-6 + v.pow(2).sum((1,2,3),keepdim=True)).sqrt()
        return v

    def forward(self, x, y, model, mask):
        model.update_batch_stats(False)
        d = torch.randn_like(x)
        d = self.normalize(d)
        for _ in range(self.n_iteration):
            d.requires_grad = True
            x_hat = x + self.xi * d
            y_hat = model(x_hat)
            kld = self.kld(y.detach(), y_hat).mean()
            d = torch.autograd.grad(kld, d)[0]
            d = self.normalize(d).detach()
        x_hat = x + self.eps * d
        y_hat = model(x_hat)
        # NOTE:
        # Original implimentation of VAT defines KL(P(y|x)||P(x|x+r_adv)) as loss function
        # However, Avital Oliver's implimentation use KL(P(y|x+r_adv)||P(y|x)) as loss function of VAT
        # see issue https://github.com/brain-research/realistic-ssl-evaluation/issues/27
        loss = (self.kld(y_hat, y.detach()) * mask).mean()
        model.update_batch_stats(True)
        return loss

    def __reduce_max(self, v, idx_list):
        for i in idx_list:
            v = v.max(i, keepdim=True)[0]
        return v

    def __logsoftmax(self,x):
        xdev = x - x.max(1, keepdim=True)[0]
        lsm = xdev - xdev.exp().sum(1, keepdim=True).log()
        return lsm