from lifelines.utils import concordance_index
import numpy as np
import torch
import torch.nn as nn
# from pyramidpooling import SpatialPyramidPooling


def select_model(modelname):
    if modelname == "HDFSModel1":
        return HDFSModel1()
    elif modelname == "HDFSModel2":
        return HDFSModel2()
    elif modelname == "LiCNN":
        return LiCNN()
    else:
        raise Exception('Invalid model name. Check spelling or hdfs_models.py for options')


class HDFSModel1(nn.Module):
    def __init__(self):
        super(HDFSModel1, self).__init__()
        # ImgIn shape = (?, 1, 221, 221)
        # Conv -> (?, 16, 72, 72)
        # Pool -> (?, 16, 36, 36)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=3),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        # Conv -> (?, 16, 16, 16)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=2),
            nn.SELU(),
            nn.Dropout(0.5)
        )
        # Conv -> (?, 8, 7, 7)
        # Pool -> (?, 8, 3, 3)
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # FC1 -> 8*3*3 inputs -> 8 outputs
        # FC2 -> 8 -> 3
        # Final -> 3 -> 1
        self.layer4 = nn.Sequential(
            nn.Linear(8*3*3, 8),
            nn.SELU(),
            nn.Linear(8, 3),
            nn.SELU(),
            nn.Linear(3, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        return x


class HDFSModel2(nn.Module):
    def __init__(self):
        super(HDFSModel2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),         # (N, 1, 221, 221) -> (N, 8, 215, 215)
            nn.BatchNorm2d(8),                      
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (N, 8, 215, 215) -> (N, 8, 107, 107)
            # nn.Dropout(0.3)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, stride=2), # (N, 8, 107, 107) -> (N, 8, 52, 52)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(16), # (N, 8, 52, 52) -> (N, 8, 16, 16)
            # nn.Dropout(0.3)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(8*16*16, 512), # (N, 2048) -> (N, 512)
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 32), # (N, 512) -> #(N, 32)
            nn.ReLU(),
            nn.Linear(32, 1) # (N, 32) -> #(N, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

class LiCNN(nn.Module):
    """
    Implementation of network from H. Li et al., "Deep Convolutional Neural Networks For Imaging Data Based Survival
    Analysis Of Rectal Cancer," 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), 2019,
    pp. 846-849, doi: 10.1109/ISBI.2019.8759301.
    """
    def __init__(self):
        super(LiCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 36, kernel_size=5),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(36, 64, kernel_size=5),
            nn.ReLU(),
        )

        # self.layer4 = SpatialPyramidPooling(levels=8)
        self.layer4 = nn.AdaptiveMaxPool2d(8)

        self.layer5 = nn.Sequential(
            nn.Linear(64*8*8, 500),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        return x


class ZhangCNN():
    """
    Implementation of network from Zhang, Liwen, et al. "A deep learning risk prediction model for 
    overall survival in patients with gastric cancer: a multicenter study." Radiotherapy and Oncology 
    150 (2020): 73-80.
    """
    def __init__(self):
        super(ZhangCNN, self).__init__()
    
    def forward(self, x):
        return x
    

#--------------------------- Evaluation Functions ---------------------------------------


def c_index(risk_pred, y, e):
    """ Calculate c-index

    Args:
        risk_pred: np.ndarray or torch.Tensor, model prediction
        y: np.ndarray or torch.Tensor, times of event e
        e: np.ndarray or torch.Tensor, event indicator

    Returns:
        c_index: float, concordance index
    """
    # Convert risk_pred, y, and e from torch.Tensor to np.ndarray if not already
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()

    return concordance_index(y, risk_pred, e)


class NegativeLogLikelihood(nn.Module):
    """Negative log likelihood loss function from Katzman et al. (2018) DeepSurv model (equation 4)"""

    def __init__(self, device, reg_weight_decay=0):
        """Initialize NegativeLogLikelihood class

        Args:
            device: string, what kind of tensor to use for loss calculation
            reg_weight_decay: weight decay value for Regularization
        """
        super(NegativeLogLikelihood, self).__init__()
        self.reg = Regularization(order=2, weight_decay=reg_weight_decay)
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