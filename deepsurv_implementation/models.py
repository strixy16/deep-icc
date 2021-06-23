import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModel(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, activation, covariates):
        ''' Initialize BasicModel class

        Args:
            activation: string, name of activation function to use
            covariates: int, number of covariates, needed for size of first layer

        Returns:
            torch.nn Module object, built sequential network
        '''
        super(BasicModel, self).__init__()
        # parses parameters of network from configuration
        # Set some defaults for network arguments
        # Fraction of input units to drop in dropout layer
        self.drop = 0.375#0.401
        # Flag to in/exclude normalization layers
        self.norm = True
        # Default dimensions of fully connected layers
        self.dims = [covariates, 4, 1]#10, 17, 17, 17, 1]
        # Activation type to use
        self.activation = activation
        # Build network using class function (below)
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None:
                # Add dropout layer
                layers.append(nn.Dropout(self.drop))

            # Add fully connected layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))

            if self.norm:
                # Add batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))

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


class NegativeLogLikelihood(nn.Module):
    '''Negative log likelihood loss function from Katzman et al. (2018) DeepSurv model (equation 4)'''
    def __init__(self, gpu):
        ''' Initialize NegativeLogLikelihood class

        Args:
            gpu: string, what kind of tensor to use for loss calculation
        '''
        super(NegativeLogLikelihood, self).__init__()
        # self.L2_reg = 0
        self.reg = Regularization(order=2, weight_decay=0)
        self.device = gpu

    def forward(self, risk_pred, y, e, model):
        # Think this is getting set of patients still at risk of failure at time t???
        mask = torch.ones(y.shape[0], y.shape[0], device=self.device)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss


class NegativeLogLikelihoodStrat(nn.Module):
    def __init__(self, gpu):
        super(NegativeLogLikelihoodStrat, self).__init__()
        self.device = gpu

    def forward(self, risk_pred, y, e, low, high):
        mask = torch.ones(y.shape[0], y.shape[0], device=self.device)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        strat_loss = 1 / (1 + torch.abs((high.mean() - low.mean())))
        strat_loss = F.smooth_l1_loss(strat_loss, torch.zeros(1).squeeze().to(self.device), reduction='none').to(self.device)
        return neg_log_loss, strat_loss


class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' Initialize Regularization class

        Args:
            order: int, norm order number
            weight_decay: float, weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Calculates regularization(self.order) loss for model

        Args:
            model: torch.nn Module object

        Returns:
            reg_loss: torch.Tensor, regularization loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss
