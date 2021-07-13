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
import torch.optim
from torch.utils.data import DataLoader

from patient_data_split import *
from rfs_utils import *
from rfs_models import *

parser = argparse.ArgumentParser(description='Training variables:')
parser.add_argument('--batchsize', default=64, type=int, help='Number of samples used for each training cycle')
parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs to run')
parser.add_argument('--imdim', default=256, type=int, help='Dimension of image to load')
parser.add_argument('--lr', default=1e-3, type=float, help='Starting learning rate for training')
parser.add_argument('--randseed', default=16, type=int, help='Random seed for reproducibility')
parser.add_argument('--split', default=0.8, type=float, help='Fraction of data to use for training (ex. 0.8)')
# TODO: change this to False once testing is done
parser.add_argument('--plots', default=True, type=bool,
                    help='Save plots of evaluation values over model training')


def main():
    global args, device

    # Utilize GPUs for Tensor computations if available
    device = torch.device("cpu")

    # Training variables
    args = parser.parse_args()

    ### Filepath Setup ###

    info_path = '/media/katy/Data/ICC/Data/Labels/RFS_all_tumors_zero.csv'
    img_path = '/media/katy/Data/ICC/Data/Images/Tumors/' + str(args.imdim) + '/Zero/'
    save_path = '/media/katy/Data/ICC/Data/Output/' + str(date.today()) + '/'

    # Make output folder for today
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Saving out parameters for the run
    save_param_fname = os.path.join(save_path, 'parameters.txt')
    with open(save_param_fname, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Setting up file to save out evaluation values to/load them from
    save_eval_fname = os.path.join(save_path, 'convergence.csv')

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

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batchsize)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batchsize)

    # Now the model class stuff
    model = KT6Model().to(device)

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = NegativeLogLikelihood(device)

    for epoch in range(0, args.epochs):
        # Initialize value holders for loss, c-index, and var values
        coxLossMeter = AverageMeter()
        ciMeter = AverageMeter()
        varMeter = AverageMeter()

        # TRAINING
        model.train()
        for X, y, e in train_loader:
            risk_pred = model(X.float().to(device))
            # Calculate loss
            cox_loss = criterion(-risk_pred, y.to(device), e.to(device), model)
            train_loss = cox_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            coxLossMeter.update(cox_loss.item(), y.size(0))
            varMeter.update(risk_pred.var(), y.size(0))
            train_c = c_index(risk_pred, y, e)
            ciMeter.update(train_c.item(), y.size(0))

        # VALIDATION
        model.eval()
        ciValMeter = AverageMeter()
        for val_X, val_y, val_e in val_loader:
            val_riskpred = model(val_X.float().to(device))
            val_c = c_index(val_riskpred, val_y, val_e)
            ciValMeter.update(val_c.item(), val_y.size(0))

        print('Epoch: {} \t Train Loss: {:.4f} \t Train CI: {:.3f} \t Val CI: {:.3f}'.format(epoch, train_loss, train_c, val_c))
        # output average results for this epoch
        save_error(ciMeter.avg, ciValMeter.avg, coxLossMeter.avg, varMeter.avg, epoch,
                   save_eval_fname)

    if args.plots:
        saveplot_coxloss(save_eval_fname, model._get_name())
        saveplot_concordance(save_eval_fname, model._get_name())


if __name__ == '__main__':
    # convergence = "../Data/Output/2021-07-08/convergence.csv"
    # model_name = "KT6"
    #
    # saveplot_coxloss(convergence, model_name)
    # saveplot_concordance(convergence, model_name)

    main()
