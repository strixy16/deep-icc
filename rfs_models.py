# Name: rfs_models.py
# Environment: Python 3.9
# Author: Katy Scott
# Last updated: July 5, 2021
# Contains main model class and other related functions for training

import torch
import torch.nn as nn


class KT6Model(nn.Module):
    """ KT6Model - CNN developed in CISC867 course"""

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


class DeepConvSurv(nn.Module):
    """
    DeepConvSurv model - from Zhu, Yao, and Huang
    “Deep convolutional neural network for survival analysiswith pathological images”. In:2016 IEEE International
    Conference on Bioinformatics andBiomedicine (BIBM). 2016, pp. 544–547.DOI:10.1109/BIBM.2016.7822579.7

    Code adapted from: https://github.com/vanAmsterdam/deep-survival/blob/master/DeepConvSurv.py
    """

    def __init__(self):
        super(DeepConvSurv, self).__init__()
        # L1 ImgIn shape=(?, 1, 256, 256)
        # Conv -> (?, 32, 84, 84)
        # Pool -> (?, 32, 42, 42)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # L2 ImgIn shape = (?, 32, 42, 42)
        # Conv -> (?, 32, 19, 19)
        # Conv -> (?, 32, 9, 9)
        # Pool -> (?, 32, 4, 4)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # L3 FC 32x4x4 inputs -> 32 outputs
        self.layer3 = nn.Sequential(
            nn.Linear(32 * 4 * 4, 32),
            nn.ReLU()
        )

        # L4 final FC 32 inputs -> 1 output
        self.layer4 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # Flatten for FC
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class NegativeLogLikelihood(nn.Module):
    """Negative log likelihood loss function from Katzman et al. (2018) DeepSurv model (equation 4)"""

    def __init__(self, device):
        """Initialize NegativeLogLikelihood class

        Args:
            device: string, what kind of tensor to use for loss calculation
        """
        super(NegativeLogLikelihood, self).__init__()
        self.reg = Regularization(order=2, weight_decay=0)
        self.device = device

    def forward(self, risk_pred, y, e, model):
        """Calculate loss

        Args:
            risk_pred: torch.Tensor, risk prediction output from network
            y: torch.Tensor,
            e: torch.Tensor,
            model: nn.Module model,
        """
        # Think this is getting set of patients still at risk of failure at time t???
        mask = torch.ones(y.shape[0], y.shape[0], device=self.device)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss


class Regularization(object):
    def __init__(self, order, weight_decay):
        """Initialize Regularization class

        Args:
            order: int, norm order number
            weight_decay: float, weight decay rate
        """
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        """Calculates regularization(self.order) loss for model

        Args:
            model: torch.nn Module object

        Returns:
            reg_loss: torch.Tensor, regularization loss
        """
        reg_loss = 0
        # Getting weight and bias parameters from model
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


def reset_weights(model):
    """
    Resetting model weights to avoid weight leakage during cross-validation

    Args:
        model - nn.Module object, model to reset weights in
    """

    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

