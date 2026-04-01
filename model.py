# model.py

import torch
import torch.nn as nn
from u2net_refactor import U2NET_full

class U2NETMultiClass(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.base = U2NET_full()
        self.n_classes = n_classes

        # update side outputs --> each should output n_classes
        for i in range(1, self.base.height + 1): 
            side_layer = getattr(self.base, f"side{i}", None)
            if side_layer is not None:
                in_channels = side_layer.in_channels
                setattr(self.base, f"side{i}", nn.Conv2d(in_channels, n_classes, 3, padding=1))

        # update outconv --> fuse all side maps
        height = self.base.height  # number of stages
       # self.base.outconv = nn.Conv2d(height * n_classes, n_classes, kernel_size=1)
        self.base.outconv = nn.Conv2d(height * n_classes, n_classes, kernel_size=1)

    def forward(self, x):
        return self.base(x)
