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
import torch

from rfs_models import *


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


def save_error(train_ci=0, val_ci=0, coxLoss=0, valCoxLoss=0, variance=0, epoch=0, slname='convergence.csv'):
    """Save training and validation statistics to csv file for a given training epoch

        Args:
            train_ci: float, training concordance index for this epoch
            val_ci: float, validation concordance index for this epoch
            coxLoss: float, training loss, negative log likelihood
            valCoxLoss: float, validation loss, negative log likelihood
            variance:
            epoch: int, epoch these stats are from
            slname: string, filename with path (sl = save location)
    """
    if epoch == 0:
        # Create file for first epoch
        f = open(slname, 'w')
        f.write('epoch,coxLoss,trainCI,valCoxLoss,valCI,variance\n')
    else:
        f = open(slname, 'a')

    f.write('{},{:.4f},{:.4f},{:.4f},{:.4f},{}\n'.format(epoch, coxLoss, train_ci, valCoxLoss, val_ci, variance))
    f.close()


def save_final_result(train_ci=0, val_ci=0, test_ci=0, coxLoss=0, valCoxLoss=0, testCoxLoss=0,
                     slname='final_result.csv'):
    """
    Save final training, validation, test results to a csv file

    Args:
        train_ci: float, training concordance index from end of training
        val_ci: float, validation concordance index from end of training
        test_ci: float, test concordance index
        coxLoss: float, training loss from end of training
        valCoxLoss: float, validation loss from end of training
        testCoxLoss: float, test loss
        slname: string, filename with path (sl = save location)
    """
    f = open(slname, 'w')
    # Validation values are default, validation wasn't used for this training, only saving train and test vals
    if val_ci == 0 and valCoxLoss == 0:
        f.write('TrainLoss,TestLoss,TrainCI,TestCI\n')
        f.write('{:.4f},{:.4f},{:.4f},{:.4f}'.format(coxLoss, testCoxLoss, train_ci, test_ci))
    # Validation used, saving train, validation, and test values
    else:
        f.write('TrainLoss,ValLoss,TestLoss,TrainCI,ValCI,TestCI\n')
        f.write('{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(coxLoss, valCoxLoss, testCoxLoss,
                                                                   train_ci, val_ci, test_ci))
    f.close()

def plot_coxloss(filename, model_name, valid=False, save=True):
    """
    Function to save or display a plot of loss over model training.

    Args:
        filename: string, path and name of file that was output during training
        model_name: string, name of model for title of plots
        valid: boolean, if validation set was used, plot validation concordance
        save: boolean, if true, save the plot instead of displaying it
    """

    evaluation_df = pd.read_csv(filename)

    fig = plt.figure()
    axLoss = plt.subplot(111)
    axLoss.plot(evaluation_df['epoch'], evaluation_df['coxLoss'], label='Training')
    if valid:
        axLoss.plot(evaluation_df['epoch'], evaluation_df['valCoxLoss'], label='Validation')
    plt.title("Training Loss - " + model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    axLoss.legend()

    if save:
        out_path = os.path.dirname(filename)
        out_file = os.path.join(out_path, 'loss.png')

        plt.savefig(out_file)

    else:
        plt.show()


def plot_concordance(filename, model_name, valid=False, save=True):
    """
    Function to save or display a plot of concordance index over model training.

    Args:
        filename: string, path and name of file that was output during training
        model_name: string, name of model for title of plots
        valid: boolean, if validation set was used, plot validation concordance
        save: boolean, if true, save the plot instead of displaying it
    """

    evaluation_df = pd.read_csv(filename)

    fig = plt.figure()
    axCI = plt.subplot(111)
    axCI.plot(evaluation_df['epoch'], evaluation_df['trainCI'], label="Training")
    if valid:
        axCI.plot(evaluation_df['epoch'], evaluation_df['valCI'], label="Validation")
    plt.title("Concordance Index - " + model_name)
    plt.xlabel("Epoch")
    plt.ylabel("C-index")
    axCI.legend()

    if save:
        out_path = os.path.dirname(filename)
        out_file = os.path.join(out_path, 'c_index.png')

        plt.savefig(out_file)

    else:
        plt.show()


def savemodel(out_path, model):
    """
    Function to save out a trained PyTorch model

    Args:
        out_path: string, path to where to save out the model
        model: nn.Module object, model to save
    """
    save_model_fname = model._get_name() + '.pth'
    save_model_fname = os.path.join(out_path, save_model_fname)

    torch.save(model.state_dict(), save_model_fname)


def select_model(modelname, device, num_genes=0):
    """
    Function to set up a model for training.

    Args:
         modelname: string, desired model to build
         device: torch.device, device to save model to (for use with GPU)
         num_genes: int, for models using genetic data, need # of genes for input layer

    Returns:
        model: nn.Module, model to use for training
    """
    # Determining which model to build (models from rfs_models.py)
    if modelname == 'KT6Model':
        model = KT6Model().to(device)
    elif modelname == 'DeepConvSurv':
        model = DeepConvSurv().to(device)
    elif modelname == 'SimpleCholangio':
        model = SimpleCholangio().to(device)
    elif modelname == 'Resnet18':
        model = ResNet('18').to(device)
    elif modelname == 'Resnet34':
        model = ResNet('34').to(device)
    elif modelname == 'DeepSurvGene':
        model = DeepSurvGene(num_genes).to(device)
    elif modelname == 'CholClassifier18':
        model = CholClassifier('18', num_genes).to(device)
    elif modelname == 'CholClassifier34':
        model = CholClassifier('34', num_genes).to(device)
    else:
        raise Exception('Invalid model type name.')

    return model
