# Name: rfs_models.py
# Environment: Python 3.9
# Author: Katy Scott
# Last updated: July 5, 2021
# Contains main model class and other related functions for training

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class LeNet5(nn.Module):
    """ Implementing Yann LeCuns LeNet 5"""

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


class LeNetCholangio(nn.Module):
    """ Implementing Yann LeCuns LeNet 5 adapted for Cholangio survival prediction"""

    def __init__(self):
        super(LeNetCholangio, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.SELU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.SELU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.SELU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.SELU(),
            nn.Linear(in_features=84, out_features=1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class KT6Model(nn.Module):
    """ KT6Model - CNN developed in CISC867 course"""
    # TODO: Make dropout values tuneable?

    def __init__(self, do1=0.7, do2=0.5):
        super(KT6Model, self).__init__()
        # L1 ImgIn shape=(?, 1, 256, 256)
        # Conv -> (?, 32, 84, 84)
        # Pool -> (?, 32, 42, 42)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=3),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(do1)
        )
        # L2 ImgIn shape = (?, 32, 42, 42)
        # Conv -> (?, 32, 19, 19)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.SELU(),
            nn.Dropout(do2),
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


class SimpleCholangio(nn.Module):
    def __init__(self):
        super(SimpleCholangio, self).__init__()
        # L1 ImgIn shape=(?, 1, 256, 256)
        # Conv -> (?, 32, 84, 84)
        # Pool -> (?, 32, 42, 42)
        # Conv -> (?, 8, 40, 40)
        # Pool -> (?, 8, 20, 20)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # L2 FC 8x20x20 inputs -> 32 outputs
        self.layer2 = nn.Sequential(
            nn.Linear(8*20*20, 256),
            nn.ReLU()
        )
        # L3 final FC 32 inputs -> 1 output
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)   # Flatten for FC
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, resnet_type, l2=400, l3=160, d1=0.25, d2=0.6):
        super(ResNet, self).__init__()
        res_model = ''
        if resnet_type == '18':
            res_model = models.resnet18(pretrained=True)
        elif resnet_type == '34':
            res_model = models.resnet34(pretrained=True)

        # Setup all resnet layers except final FC layer
        self.orig = nn.Sequential(*(list(res_model.children())[:-3]))
        for param in self.orig.parameters():
            param.requires_grad = False

        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )

        # CAM final linear layer
        self.layercam = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(14, 14))
        )

        self.layercph = nn.Sequential(
            nn.Linear(1024*14*14, 256),
            nn.Linear(256, 1)
        )
        # Replace final linear layer with this one
        # self.layercph = nn.Sequential(
        #     # This has to be 512 because that's the output from the resnet18 and 34 models
        #     nn.Linear(512, l2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(l2),
        #     nn.Dropout(d1),
        #     nn.Linear(l2, l3),
        #     nn.ReLU(),
        #     nn.Dropout(d2),
        #     nn.Linear(l3, 1)
        # )

    def forward(self, x):
        # Resnet model
        out = self.orig(x)
        out = self.layercam(out)
        # Flatten for FC
        out = out.view(out.size(0), -1)
        # CPH output
        out = self.layercph(out)
        return out


class DeepSurvGene(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, num_genes, activation='ReLU'):
        ''' Initialize DeepSurvGene class

        Args:
            activation: string, name of activation function to use
            num_genes: int, number of genes, needed for size of first layer

        Returns:
            torch.nn Module object, built sequential network
        '''
        super(DeepSurvGene, self).__init__()
        # parses parameters of network from configuration
        # Set some defaults for network arguments
        # Fraction of input units to drop in dropout layer
        self.drop = 0.375  # 0.401
        # Flag to in/exclude normalization layers
        self.norm = True
        # Default dimensions of fully connected layers
        self.dims = [num_genes, 4, 1]  # 10, 17, 17, 17, 1]
        # Activation type to use
        self.activation = activation
        # Build network using class function (below)
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        layers = []
        for i in range(len(self.dims) - 1):
            if i and self.drop is not None:
                # Add dropout layer
                layers.append(nn.Dropout(self.drop))

            # Add fully connected layer
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

            if self.norm:
                # Add batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i + 1]))

            # Adds activation layer
            # eval creates proper format of activation to get from NN
            layers.append(eval('nn.{}()'.format(self.activation)))

        # Build sequential network from list of layers created in for loop
        return nn.Sequential(*layers)

    def forward(self, X):
        ''' Forward propagation through network

        Args:
            X: data to pass through network

        Returns:
            Output of model (risk prediction)
        '''
        return self.model(X)


class CholClassifier(nn.Module):
    def __init__(self, resnet_type, num_genes, l2=256, l3=128, l4=4, l5=4, d1=0.2, d2=0.375):
        super(CholClassifier, self).__init__()
        res_model = ''
        if resnet_type == '18':
            res_model = models.resnet18(pretrained=True)
        elif resnet_type == '34':
            res_model = models.resnet34(pretrained=True)

        # Setup all resnet layers except final FC layer
        self.res = nn.Sequential(*(list(res_model.children())[:-1]))
        for param in self.res.parameters():
            param.requires_grad = False

        # Replace final linear layer with this one
        self.ct = nn.Sequential(
            # This has to be 512 because that's the output from the resnet18 and 34 models
            nn.Linear(512, l2),
            nn.ReLU(),
            nn.BatchNorm1d(l2),
            nn.Dropout(d1),
            nn.Linear(l2, l3),
            nn.ReLU(),
        )

        self.gene = nn.Sequential(
            nn.Linear(num_genes, l4),
            nn.BatchNorm1d(l4),
            nn.SELU(),
            nn.Dropout(d2),
            nn.Linear(l4, l5),
            nn.BatchNorm1d(l5),
            nn.SELU()
        )

        self.final = nn.Linear(l3 + l5, 1)

    def forward(self, img, genes):
        img = self.res(img)
        img = img.view(img.size(0), -1)
        img = self.ct(img)

        gene = self.gene(genes)

        x = torch.cat((img, gene), dim=1)
        # x = nn.SELU(x)

        return self.final(x)


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
        verbose - int, whether to print out what is being reset (true if 2)
    """

    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

