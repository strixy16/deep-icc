import argparse
from datetime import datetime
import json
import os
import optuna
from optuna.trial import TrialState
import pickle
from sklearn.model_selection import KFold
from skimage.color import gray2rgb
import torch.optim as optim
from torch.utils.data import DataLoader

from rfs_preprocessing import *
from rfs_utils import *
from rfs_models import *

parser = argparse.ArgumentParser(description='Training variables:')
parser.add_argument('--batchsize', default=32, type=int, help='Number of samples used for each training cycle')
parser.add_argument('--covariate', default=18, type=int, help='Number of genes flags in gene feature data')
parser.add_argument('--datadir', default='/media/katy/Data/ICC/Data', type=str, help='Full path to the Data directory '
                                                                                     'where images, labels, are stored'
                                                                                     'and output will be saved.')
parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs to run')
parser.add_argument('--imdim', default=256, type=int, help='Dimension of image to load')
# parser.add_argument('--learnrate', default=3e-3, type=float, help='Starting learning rate for training')
parser.add_argument('--modelname', default='Resnet18', type=str, help='Name of model type to build and train. '
                                                                       'Current options are KT6Model, DeepConvSurv'
                                                                        'Resnet18 and Resnet34')
parser.add_argument('--randseed', default=16, type=int, help='Random seed for reproducibility')
parser.add_argument('--scanthresh', default=300, type=int, help='Threshold for number of tumour pixels to filter images'
                                                                ' through')
parser.add_argument('--validation', default=0, type=int, help='Select validation method from list: '
                                                              '0: hold out, 1: k-fold')
parser.add_argument('--split', default=0.8, type=float, help='Fraction of data to use for training (ex. 0.8) with'
                                                             'hold-out validation')
parser.add_argument('--kfold_num', default=5, type=int, help='If using k-fold cross validation, supply k value')
parser.add_argument('--verbose', default=1, type=int, help='Levels of output: 0: none, 1: training output')
parser.add_argument('--plots', default=True, type=bool, help='Save plots of evaluation values over model training')


def main():
    global args, device

    # Utilize GPUs for Tensor computations if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    out_dir = 'Output/' + str(args.modelname) + '-' + datetime.now().strftime("%Y-%m-%d-%H%M")
    out_path = os.path.join(args.datadir, out_dir)

    # Make output folder for the training run
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Save out parameters use for the run
    save_param_fname = os.path.join(out_path, 'parameters.txt')
    with open(save_param_fname, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # DATA LOADING
    # Load in gene features
    genefeat_fname = os.path.join(args.datadir, 'MSK_Genomic_Data.csv')
    genefeat = pd.read_csv(genefeat_fname)
    # Removing space at the end of the Patient ID names
    genefeat['ScoutID'] = genefeat['ScoutID'].str.strip()
    # Fixing columns with illegal characters in the name
    genefeat.rename(columns={'CDKN2A.DEL': 'CDKN2A_DEL', 'TGF-Beta_Pathway': 'TGF_Beta_Pathway'}, inplace=True)

    # Get number of covariates = number of genetic columns
    args.covariates = genefeat.shape[1] - 1


