# Name: rfs_models.py
# Environment: Python 3.9
# Author: Katy Scott
# Last updated: July 5, 2021
# Contains main model class and other related functions for training

import torch.nn as nn


class KT6Model(nn.Module):

    def __init__(self):
        super(KT6Model, self).__init__()
        # L1 ImgIn shape=(?, 1, 256, 256)
        # Conv -> (?, 32, 84, 84)
        # Pool -> (?, 32, 42, 42)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=3),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.7)
        )
        # L2 ImgIn shape = (?, 32, 42, 42)
        # Conv -> (?, 32, 19, 19)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.SELU(),
            nn.Dropout(0.5),
        )
        # L3 ImgIn shape = (?, 32, 19, 19)
        # Conv -> (?, 32, 9, 9)
        # Pool -> (?, 32, 4, 4)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # L4 FC 32x4x4 inputs -> 32 outputs
        self.layer4 = nn.Sequential(
            nn.Linear(32*4*4, 32),
            nn.SELU()
        )

        # L5 final FC 32 inputs -> 1 output
        self.layer5 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) # Flatten for FC
        out = self.layer4(out)
        out = self.layer5(out)
        return out


model = KT6Model()
model
