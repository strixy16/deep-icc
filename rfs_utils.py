# Name: rfs_train.py
# Environment: Python 3.9
# Author: Katy Scott
# Last updated: June 24, 2021
# Contains misc. functions used in training


from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


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
            slname: string, filename (sl = save location)
    """
    if epoch == 0:
        # Create file for first epoch
        f = open(slname, 'w')
        f.write('epoch,coxLoss,trainCI,valCI,variance\n')
        f.write('{},{:.4f},{:.4f},{:.4f},{}\n'.format(epoch, coxLoss, train_ci, val_ci, variance))
        f.close()
    else:
        f = open(slname, 'a')
        f.write('{},{:.4f},{:.4f},{:.4f},{}\n'.format(epoch, coxLoss, train_ci, val_ci, variance))
        f.close()


def saveplot_coxloss(filename, model_name):
    """
    Function to save a plot of loss over model training.

    Args:
        filename: string, path and name of file that was output during training
        model_name: string, name of model for title of plots
    """

    evaluation_df = pd.read_csv(filename)

    fig = plt.figure()
    axLoss = plt.subplot(111)
    axLoss.plot(evaluation_df['epoch'], evaluation_df['coxLoss'], label='Training')
    plt.title("Training Loss - " + model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    axLoss.legend()

    out_path = os.path.dirname(filename)
    out_file = os.path.join(out_path, 'loss.png')

    plt.savefig(out_file)


def saveplot_concordance(filename, model_name):
    """
    Function to plot concordance index for training and validation of model training.

    Args:
        filename: string, path and name of file that was output during training
        model_name: string, name of model for title of plots
    """

    evaluation_df = pd.read_csv(filename)

    fig = plt.figure()
    axCI = plt.subplot(111)
    axCI.plot(evaluation_df['epoch'], evaluation_df['trainCI'], label="Training")
    axCI.plot(evaluation_df['epoch'], evaluation_df['valCI'], label="Validation")
    plt.title("Concordance Index - " + model_name)
    plt.xlabel("Epoch")
    plt.ylabel("C-index")
    axCI.legend()

    out_path = os.path.dirname(filename)
    out_file = os.path.join(out_path, 'c_index.png')

    plt.savefig(out_file)