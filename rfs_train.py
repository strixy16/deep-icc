# Name: rfs_train.py
# Environment: Python 3.9
# Author: Katy Scott
# Last updated: June 24, 2021
# Main training file for RFS prediction model

import argparse
from datetime import datetime
import json
from sklearn.model_selection import KFold
import torch.optim
from torch.utils.data import DataLoader

from rfs_preprocessing import *
from rfs_utils import *
from rfs_models import *

parser = argparse.ArgumentParser(description='Training variables:')
parser.add_argument('--batchsize', default=64, type=int, help='Number of samples used for each training cycle')
parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs to run')
parser.add_argument('--imdim', default=256, type=int, help='Dimension of image to load')
parser.add_argument('--lr', default=1e-3, type=float, help='Starting learning rate for training')
parser.add_argument('--modelname', default='DeepConvSurv', type=str, help='Name of model type to build and train. Current options are KT6Model and DeepConvSurv')
parser.add_argument('--randseed', default=16, type=int, help='Random seed for reproducibility')
parser.add_argument('--split', default=0.8, type=float, help='Fraction of data to use for training (ex. 0.8)')
parser.add_argument('--scanthresh', default=300, type=int, help='Threshold for number of tumour pixels to filter images through')
parser.add_argument('--validation', default=1, type=int, help='Select validation method from list: 0: hold out, 1: k-fold')
parser.add_argument('--kfold_num', default=5, type=int, help='If using k-fold cross validation, supply k value')
parser.add_argument('--plots', default=True, type=bool,
                    help='Save plots of evaluation values over model training')


def main():
    global args, device

    # Utilize GPUs for Tensor computations if available
    device = torch.device("cpu")

    # Training variables
    args = parser.parse_args()

    ### Filepath Setup ###
    info_path = '/media/katy/Data/ICC/Data/Labels/' + str(args.imdim) + '/RFS_all_tumors_zero.csv'
    z_img_path = '/media/katy/Data/ICC/Data/Images/Tumors/' + str(args.imdim) + '/Zero/'
    n_img_path = '/media/katy/Data/ICC/Data/Images/Tumors/' + str(args.imdim) + '/NaN/'
    save_path = '/media/katy/Data/ICC/Data/Output/' + str(args.modelname) + "-" + datetime.now().strftime("%Y-%m-%d-%H%M") + '/'

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

    # Filter scans with mostly background in the image
    filtered_indices = removeSmallScans(info, z_img_path, args.imdim, args.scanthresh)
    filtered_info = info.iloc[filtered_indices]

    patnum = np.asarray(filtered_info['Pat_ID'])
    event = np.asarray(filtered_info['RFS_Code'])

    ### Model creation ###
    model = DeepConvSurv().to(device)
    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = NegativeLogLikelihood(device)

    ### Setup data based on validation type ###
    # Hold-out validation
    if args.validation == 0:
        # Split data into train and validation sets
        train_idx, val_idx = pat_train_test_split(patnum, event, args.split, args.randseed)

        # Set up data with custom Dataset class (in rfs_utils)
        train_dataset = CTSurvDataset(filtered_info, z_img_path, train_idx, args.imdim)
        val_dataset = CTSurvDataset(filtered_info, z_img_path, val_idx, args.imdim)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batchsize)
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batchsize, drop_last=True)

        train(model, args.epochs, optimizer, criterion, train_loader, val_loader, save_eval_fname, args.plots)

    # K-fold cross validation
    elif args.validation == 1:
        kf = KFold(n_splits=args.kfold_num)

        # Get unique patient numbers
        u_pats = np.unique(patnum)

        for fold, (train_patidx, val_patidx) in enumerate(kf.split(u_pats)):
            print(f'FOLD {fold}')
            print('-------------------------------------')

            model.apply(reset_weights)

            # Get patient numbers from idx values for this fold
            train_pats = u_pats[train_patidx]
            val_pats = u_pats[val_patidx]

            # Get full info for the train and validation cohorts
            train_info = filtered_info.query('Pat_ID in @train_pats')
            val_info = filtered_info.query('Pat_ID in @val_pats')

            train_dataset = CTSurvDataset(train_info, z_img_path, len(train_info), args.imdim)
            val_dataset = CTSurvDataset(val_info, z_img_path, len(val_info), args.imdim)

            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batchsize)
            val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batchsize, drop_last=True)

            train(model, args.epochs, optimizer, criterion, train_loader, val_loader, save_eval_fname, args.plots)

            # TODO: figure out how the saving is gonna work for k-folds (5 versions of the training plots?)

    else:
        print("Invalid validation method selected.")
        return -1

    # # Split data into train and validation sets
    # train_idx, val_idx = pat_train_test_split(patnum, event, args.split, args.randseed)
    #
    # # Set up data with custom Dataset class (in rfs_utils)
    # train_dataset = CTSurvDataset(filtered_info, z_img_path, train_idx, args.imdim)
    # val_dataset = CTSurvDataset(filtered_info, z_img_path, val_idx, args.imdim)
    #
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batchsize)
    # val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batchsize, drop_last=True)

    # train(model, args.epochs, optimizer, criterion, train_loader, val_loader, save_eval_fname, args.plots)


def train(model, epochs, optimizer, criterion, train_loader, val_loader, save_eval_fname, plots=True):
    """
    Function to train a given neural network model

    Args:
         model - nn.Module, model to train
         epochs - int, number of epochs to train model for
         optimizer - torch.optim object, optimizer to use in model training
         criterion - , loss function to use in optimization
         save_eval_fname - string, filename + path to save training results
         plots - bool, whether to save out loss and c-index plots over training
    """
    for epoch in range(0, epochs):
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

    if plots:
        saveplot_coxloss(save_eval_fname, model._get_name())
        saveplot_concordance(save_eval_fname, model._get_name())


if __name__ == '__main__':
    main()
