import torch
import numpy as np
from torch.utils.data import Dataset
from lifelines.utils import concordance_index


class SurvivalDataset(Dataset):
    def __init__(self, dataset, args):
        '''Initialize SurvivalDataset class

        Args:
            dataset: pandas.Dataframe, Contains covariates, time of event (T), and event indicator (E) values.
            T and E must be the final two columns
            args: Namespace,
        '''
        # Get covariates out of dataframe (args.covariates is num of columns containing covariates)
        self.X = dataset.iloc[:, 0:args.covariates].values
        # Get time and event indicator columns out of dataframe
        self.data = list(zip(dataset.time, dataset.event))
        self.len = len(dataset)
        print('=> load {} samples'.format(self.len))
        # Normalize covariate data with class function
        if args.normalize:
            self._normalize()

    def _normalize(self):
        '''Normalize X data (covariates) (transform values to range between 0 and 1)'''
        self.X = (self.X - self.X.min(axis=0)) / (self.X.max(axis=0) - self.X.min(axis=0))

    def __getitem__(self, item):
        '''Getter for single data piece

        Args:
            item: int, index of data to retrieve

        Returns:
            X_tensor: torch.Tensor, covariate values for data item
            y_tensor: torch.Tensor, time of event value for data item
            e_tensor: int torch.Tensor, event indicator value for data item
        '''
        y, e = self.data[item]
        X_tensor = torch.from_numpy(self.X[item])
        e_tensor = torch.Tensor([e]).int()
        y_tensor = torch.Tensor([y])
        return X_tensor, y_tensor, e_tensor

    def __len__(self):
        return self.len


def save_error(train_ci, val_ci, coxLoss, stratLoss, variance, epoch, slname):
    '''Save training and validation statistics to csv file

        Args:
            train_ci: float, training concordance index for this epoch
            val_ci: float, validation concordance index for this epoch
            coxLoss:
            stratLoss:
            variance:
            epoch: int, epoch these stats are from
            slname: string, filename
    '''
    if epoch == 0:
        # Create file for first epoch
        f = open(slname, 'w')
        f.write('epoch,coxLoss,stratLoss,trainCI,valCI,variance\n')
        f.write('{},{:.4f},{:.4f},{:.4f},{:.4f},{}\n'.format(epoch, coxLoss, stratLoss, train_ci, val_ci, variance))
        f.close()
    else:
        f = open(slname, 'a')
        f.write('{},{:.4f},{:.4f},{:.4f},{:.4f},{}\n'.format(epoch, coxLoss, stratLoss, train_ci, val_ci, variance))
        f.close()


def c_index(risk_pred, y, e):
    '''Calculate c-index

    Args:
        risk_pred: np.ndarray or torch.Tensor, model prediction
        y: np.ndarray or torch.Tensor, times of event e
        e: np.ndarray or torch.Tensor, event indicator

    Returns:
        c_index: float, concordance index
    '''
    # Convert risk_pred, y, and e from torch.Tensor to np.ndarray if not already
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate):
    '''Adjust learning rate according to (epoch, lr, and lr_decay_rate)

    Args:
        optimizer: torch.optim object,
        epoch: int, epoch number
        lr: float, initial learning rate
        lr_decay_rate: float, decay rate to apply to learning rate

    Returns:
        lr: float, updated learning rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1+epoch*lr_decay_rate)
    return optimizer.param_groups[0]['lr']

