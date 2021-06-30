# Name: rfs_models.py
# Environment: Python 3.9
# Author: Katy Scott
# Last updated: June 30, 2021
# Contains main model class and other related functions for training

import torch
import torch.nn as nn


class KT6Model(nn.Module):

    def __init__(self, activation):
        super(KT6Model, self).__init__()
        # L1 ImgIn shape=(?, 256, 256, 1)
        # Conv -> (?,
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=3),
            nn.SELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.7)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2),
            nn.SELU(),
            nn.Dropout(0.5),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.SELU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32) # TODO: need to figure out the math to know what the input is for this layer

        )
