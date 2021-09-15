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
# parser.add_argument('--covariate', default=18, type=int, help='Number of genes flags in gene feature data')
parser.add_argument('--datadir', default='/media/katy/Data/ICC/Data', type=str, help='Full path to the Data directory '
                                                                                     'where images, labels, are stored'
                                                                                     'and output will be saved.')
parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs to run')
parser.add_argument('--imdim', default=256, type=int, help='Dimension of image to load')
parser.add_argument('--learnrate', default=3e-3, type=float, help='Starting learning rate for training')
parser.add_argument('--modelname', default='Resnet18', type=str, help='Name of model type to use for CNN half of model.'
                                                                       'Current options are Resnet18 and Resnet34')
parser.add_argument('--randseed', default=16, type=int, help='Random seed for reproducibility')
parser.add_argument('--scanthresh', default=500, type=int, help='Threshold for number of tumour pixels to filter images'
                                                                ' through')
parser.add_argument('--validation', default=0, type=int, help='Select validation method from list: '
                                                              '0: hold out, 1: k-fold')
parser.add_argument('--split', default=0.8, type=float, help='Fraction of data to use for training (ex. 0.8)')
parser.add_argument('--valid_split', default=0.2, type=float, help='Fraction of training data to use for hold out '
                                                                   'validation (ex. 0.2)')
# parser.add_argument('--kfold_num', default=5, type=int, help='If using k-fold cross validation, supply k value')
parser.add_argument('--verbose', default=1, type=int, help='Levels of output: 0: none, 1: training output')
parser.add_argument('--plots', default=True, type=bool, help='Save plots of evaluation values over model training')


def main():
    global args, device

    # Utilize GPUs for Tensor computations if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get input arguments (either from defaults or input when this function is called from terminal)
    args = parser.parse_args()

    # Setup output directory to save parameters/results/etc.
    out_dir = 'Output/' + str(args.modelname) + '-' + datetime.now().strftime("%Y-%m-%d-%H%M")
    out_path = os.path.join(args.datadir, out_dir)

    # DATA LOADING
    # This does hold-out validation
    train_loader, valid_loader, test_loader = load_chol_tumor_w_gene(args.datadir, split=args.split, valid=True,
                                                                     valid_split=args.valid_split)

    # MODEL SETUP
    num_genes = train_loader.dataset.num_genes
    if args.modelname == 'Resnet18':
        model = CholClassifier('18', num_genes).to(device)
    elif args.modelname == 'Resnet34':
        model = CholClassifier('34', num_genes).to(device)
    else:
        print("Invalid model type name.")
        return -1

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate)
    criterion = NegativeLogLikelihood(device)

    # Make output folder for the training run
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Save out parameters used for the run
    save_param_fname = os.path.join(out_path, 'parameters.txt')
    with open(save_param_fname, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Setting up file to save out evaluation values to/load them from
    save_eval_fname = os.path.join(out_path, 'convergence.csv')

    # Model Training
    for epoch in range(0, args.epochs):
        # Initialize value holders for loss, c-index, and var values
        coxLossMeter = AverageMeter()
        ciMeter = AverageMeter()
        varMeter = AverageMeter()

        # Train
        model.train()
        for X, g, y, e in train_loader:
            # X = CT image
            # g = genetic markers
            # y = time to event
            # e = event indicator

            # Resnet models expect an RGB image - generate a 3 channel version of CT image here
            if args.modelname == 'Resnet18' or args.modelname == 'Resnet34':
                # Convert grayscale image to rgb to generate 3 channels
                rgb_X = gray2rgb(X)
                # Reshape so channels is second value
                rgb_X = torch.from_numpy(rgb_X)
                X = torch.reshape(rgb_X, (rgb_X.shape[0], rgb_X.shape[-1], rgb_X.shape[2], rgb_X.shape[3]))

            X, g, y, e = X.float().to(device), g.float().to(device), y.float().to(device), e.float().to(device)
            # Pass image and gene to network
            risk_pred = model(X, g)
            # Calculate loss
            cox_loss = criterion(-risk_pred, y, e, model)
            train_loss = cox_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Update loss and variance tracking
            coxLossMeter.update(cox_loss.item(), y.size(0))
            varMeter.update(risk_pred.var(), y.size(0))
            train_c = c_index(risk_pred, y, e)
            ciMeter.update(train_c.item(), y.size(0))

        # Validation
        model.eval()
        valLossMeter = AverageMeter()
        ciValMeter = AverageMeter()
        for val_X, val_g, val_y, val_e in valid_loader:
            # val_X = CT image
            # val_g = genetic markers
            # val_y = time to event
            # val_e = event indicator

            # Resnet models expect an RGB image - generate a 3 channel version of CT image here
            if args.modelname == 'Resnet18' or args.modelname == 'Resnet34':
                # Convert grayscale image to rgb to generate 3 channels
                rgb_valX = gray2rgb(val_X)
                # Reshape so channels is second value
                rgb_valX = torch.from_numpy(rgb_valX)
                val_X = torch.reshape(rgb_valX,
                                      (rgb_valX.shape[0], rgb_valX.shape[-1], rgb_valX.shape[2], rgb_valX.shape[3]))

            val_X, val_g, val_y, val_e = val_X.float().to(device), val_g.float().to(device), val_y.to(device), val_e.to(device)

            val_risk_pred = model(val_X, val_g)
            val_cox_loss = criterion(-val_risk_pred, val_y, val_e, model)
            val_c = c_index(val_risk_pred, val_y, val_e)
            valLossMeter.update(val_cox_loss.item(), y.size(0))
            ciValMeter.update(val_c.item(), val_y.size(0))

        # Printing average loss and c-index values for the epoch
        print('Epoch: {} \t Train Loss: {:.4f} \t Val Loss: {:.4f} \t Train CI: {:.3f} \t Val CI: {:.3f}'.format(epoch,
               coxLossMeter.avg, valLossMeter.avg, ciMeter.avg, ciValMeter.avg))
        # Saving average results for this epoch
        save_error(ciMeter.avg, ciValMeter.avg, coxLossMeter.avg, valLossMeter.avg, varMeter.avg, epoch, save_eval_fname)

    if args.plots:
        saveplot_coxloss(save_eval_fname, model._get_name())
        saveplot_concordance(save_eval_fname, model._get_name())


if __name__ == '__main__':
    main()
