# Name: rfs_train.py
# Environment: Python 3.9
# Author: Katy Scott
# Last updated: June 24, 2021
# Main training file for RFS prediction model

import argparse
from datetime import date
import json
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from patient_data_split import *
from rfs_utils import *

parser = argparse.ArgumentParser(description='Training variables:')
parser.add_argument('--batchsize', default=64, type=int, help='Number of samples used for each training cycle')
parser.add_argument('--epochs', default=500, type=int, help='Number of training epochs to run')
parser.add_argument('--imdim', default=256, type=int, help='Dimension of image to load')
parser.add_argument('--lr', default=1e-3, type=float, help='Starting learning rate for training')
parser.add_argument('--randseed', default=16, type=int, help='Random seed for reproducibility')
parser.add_argument('--split', default=0.8, type=float, help='Fraction of data to use for training (ex. 0.8)')


def main():
    global args, device

    # Utilize GPUs for Tensor computations if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training variables
    args = parser.parse_args()

    ### Filepath Setup ###

    info_path = '/media/katy/Data/ICC/Data/Labels/RFS_all_tumors_zero.csv'
    img_path = '/media/katy/Data/ICC/Data/Images/Tumors/' + str(args.imdim) + '/Zero'
    save_path = '/media/katy/Data/ICC/Data/Output/' + str(date.today()) + '/'

    # Make output folder for today
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Saving out parameters for the run
    save_param_fname = save_path + 'parameters.txt'
    with open(save_param_fname, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ### Data Loading ###
    info = pd.read_csv(info_path)
    patnum = np.asarray(info['Pat ID'])
    event = np.asarray(info['RFS Code'])

    # Split data into train and validation sets
    train_idx, val_idx = pat_train_test_split(patnum, event, args.split, args.randseed)

    # TODO: removeSmallScans would go here or before train-val split

    # Set up data with custom Dataset class (in rfs_utils)
    train_dataset = CTSurvDataset(info, img_path, train_idx, args.imdim)
    val_dataset = CTSurvDataset(info, img_path, val_idx, args.imdim)

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize)

    # Now the model class stuff

    print("Stop here")

if __name__ == '__main__':
    main()
