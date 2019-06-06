import torch
import torch.nn.functional as F
import random

class transform:
    def __init__(self, flip=True, r_crop=True, g_noise=True):
        self.flip = flip
        self.r_crop = r_crop
        self.g_noise = g_noise
        print("holizontal flip : {}, random crop : {}, gaussian noise : {}".format(
            self.flip, self.r_crop, self.g_noise
        ))

    def __call__(self, x):
        if self.flip and random.random() > 0.5:
            x = x.flip(-1)
        if self.r_crop:
            h, w = x.shape[-2:]
            x = F.pad(x, [2,2,2,2], mode="reflect")
            l, t = random.randint(0, 4), random.randint(0,4)
            x = x[:,:,t:t+h,l:l+w]
        if self.g_noise:
            n = torch.randn_like(x) * 0.15
            x = n + x
        return x
