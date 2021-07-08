# Name: rfs_train.py
# Environment: Python 3.9
# Author: Katy Scott
# Last updated: June 24, 2021
# Contains Dataset class and other functions used in training

# TODO: C-index function
# TODO: move patient_data_split here maybe?

from lifelines.utils import concordance_index
import numpy as np
import torch
from torch.utils.data import Dataset


class CTSurvDataset(Dataset):

    def __init__(self, info, img_path, idx, img_dim):
        """Initialize CTSurvDataset class
        Dataset for CT images used in survival prediction

        Args:
            info: pandas.Dataframe, read from CSV, contains image file names, patient ID, slice ID, and RFS time and event labels
                Column titles should be: File, Pat ID, Slice Num, RFS Code, RFS Time
            img_path: string, path to folder containing image files listed in info
            idx: list, indices to include in this dataset (ex. indices of training data)
            img_dim: int, dimension of images
        """

        self.info = info.iloc[idx, :]
        self.fname = np.asarray(self.info['File'])
        self.patid = np.asarray(self.info['Pat ID'])
        self.slice = np.asarray(self.info['Slice Num'])
        self.event = np.asarray(self.info['RFS Code'])
        self.time = np.asarray([self.info['RFS Time']])

        self.img_path = img_path
        self.dim = img_dim

        # TODO: introduce MinMaxScaler and/or Normalization

    def __getitem__(self, index):
        fname = self.fname[index]
        e_tensor = torch.Tensor([self.event[index]]).int()
        t_tensor = torch.Tensor([self.time[index]])

        # Load in CT bin image as numpy array
        img = np.fromfile(self.img_path + self.fname[index])
        # Reshape to a 2D array
        img_2D = np.reshape(img, (self.dim, self.dim))

        X_tensor = torch.from_numpy(img_2D)

        return X_tensor, t_tensor, e_tensor

    def __len__(self):
        return len(self.event)


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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def save_error(train_ci, val_ci, coxLoss, variance, epoch, slname):
    """Save training and validation statistics to csv file

        Args:
            train_ci: float, training concordance index for this epoch
            val_ci: float, validation concordance index for this epoch
            coxLoss: , training loss, negative log likelihood
            variance:
            epoch: int, epoch these stats are from
            slname: string, filename
    """
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

