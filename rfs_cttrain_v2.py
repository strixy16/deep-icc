import argparse
from datetime import datetime
import json
import os
# import optuna
# from optuna.trial import TrialState
# import pickle
# from sklearn.model_selection import KFold
from skimage.color import gray2rgb
# import torch.optim as optim
# from torch.utils.data import DataLoader

from rfs_preprocessing import *
from rfs_utils import *
from rfs_models import *

parser = argparse.ArgumentParser(description='Training variables:')
parser.add_argument('--batchsize', default=16, type=int, help='Number of samples used for each training cycle')
parser.add_argument('--datadir', default='/media/katy/Data/ICC/Data', type=str, help='Full path to the Data directory '
                                                                                     'where images, labels, are stored'
                                                                                     'and output will be saved.')
parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs to run')
parser.add_argument('--imdim', default=256, type=int, help='Dimension of image to load')
parser.add_argument('--learnrate', default=1e-5, type=float, help='Starting learning rate for training')
parser.add_argument('--modelname', default='Resnet34', type=str, help='Name of model type to use for CNN half of model.'
                                                                       'Current options are Resnet18 and Resnet34')
parser.add_argument('--randseed', default=16, type=int, help='Random seed for reproducibility')
parser.add_argument('--scanthresh', default=300, type=int, help='Threshold for number of tumour pixels to filter images'
                                                                ' through')
parser.add_argument('--validation', default=0, type=int, help='Whether to use a validation set with train and test')
parser.add_argument('--split', default=0.8, type=float, help='Fraction of data to use for training (ex. 0.8)')
parser.add_argument('--valid_split', default=0.2, type=float, help='Fraction of training data to use for hold out '
                                                                   'validation (ex. 0.2)')
parser.add_argument('--saveplots', default=True, type=bool, help='What to do with plots of evaluation values over model '
                                                              'training. If false, will display plots instead.')
parser.add_argument('--testing', default=True, type=bool, help='Set this to disable saving output (e.g. plots, '
                                                               ' parameters). For use while testing script.')


def train_ct():
    ## PRELIMINARY SETUP ##
    global args, device

    # Utilize GPUs for Tensor computations if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get input arguments (either from defaults or input when this function is called from terminal)
    args = parser.parse_args()

    ## OUTPUT SETUP ##
    # Check if testing mode to see if output should be saved or not
    if not args.testing:
        # Setup output directory to save parameters/results/etc.
        out_dir = 'Output/' + str(args.modelname) + '-' + datetime.now().strftime("%Y-%m-%d-%H%M")
        out_path = os.path.join(args.datadir, out_dir)

        # Make output folder for the training run
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Save out parameters used for the run
        save_param_fname = os.path.join(out_path, 'parameters.txt')
        with open(save_param_fname, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # Setting up file to save out evaluation values to/load them from
        save_eval_fname = os.path.join(out_path, 'convergence.csv')


    ## DATA LOADING ##
    if args.validation:
        # Loading the tumor images, splitting into train/valid/test based on event indicator, and setting up DataLoader
        # objects for each
        train_loader, valid_loader, test_loader = load_chol_tumor(args.datadir, imdim=args.imdim,
                                                                  scanthresh=args.scanthresh, split=args.split,
                                                                  batch_size=args.batchsize, valid=True,
                                                                  valid_split=args.valid_split, seed=args.randseed)
    else:
        # Loading the tumor images, splitting into train/test based on event indicator, and setting up DataLoader
        # objects for each
        train_loader, test_loader = load_chol_tumor(args.datadir, imdim=args.imdim, scanthresh=args.scanthresh,
                                                    split=args.split, batch_size=args.batchsize, valid=False,
                                                    seed=args.randseed)

    ## MODEL SETUP ##
    model = select_model(args.modelname, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate)
    criterion = NegativeLogLikelihood(device)

    for epoch in range(0, args.epochs):
        # Initialize value holders for training loss, c-index, and var values
        coxLossMeter = AverageMeter()
        ciMeter = AverageMeter()
        varMeter = AverageMeter()

        # Training phase
        model.train()
        for X, y, e in train_loader:
            # X = CT image
            # y = time to event
            # e = event indicator

            # ResNet models expect an RGB image, so a 3 channel version of the CT image is generated here
            # (CholClassifier contains a ResNet component, so included here as well)
            if type(model) == ResNet or type(model) == CholClassifier:
                # Convert grayscale image to rgb to generate 3 channels
                rgb_X = gray2rgb(X)
                # Reshape so channels is second value
                rgb_X = torch.from_numpy(rgb_X)
                X = torch.reshape(rgb_X, (rgb_X.shape[0], rgb_X.shape[-1], rgb_X.shape[2], rgb_X.shape[3]))

            # Convert all values to float for backprop and evaluation calculations
            X, y, e = X.float().to(device), y.float().to(device), e.float().to(device)

            # Forward pass through model
            risk_pred = model(X)

            # Calculate loss and evaluation metrics
            cox_loss = criterion(-risk_pred, y, e, model)
            coxLossMeter.update(cox_loss.item(), y.size(0))

            train_ci = c_index(risk_pred, y, e)
            ciMeter.update(train_ci.item(), y.size(0))

            varMeter.update(risk_pred.var(), y.size(0))

            # Updating parameters based on forward pass
            optimizer.zero_grad()
            cox_loss.backward()
            optimizer.step()

        # Validation phase
        if args.validation:
            model.eval()
            valLossMeter = AverageMeter()
            ciValMeter = AverageMeter()
            for val_X, val_y, val_e in valid_loader:
                # val_X = CT image
                # val_g = genetic markers
                # val_y = time to event
                # val_e = event indicator

                # ResNet models expect an RGB image, so a 3 channel version of the CT image is generated here
                # (CholClassifier contains a ResNet component, so included here as well)
                if type(model) == ResNet or type(model) == CholClassifier:
                    # Convert grayscale image to rgb to generate 3 channels
                    rgb_valX = gray2rgb(val_X)
                    # Reshape so channels is second value
                    rgb_valX = torch.from_numpy(rgb_valX)
                    val_X = torch.reshape(rgb_valX,
                                          (rgb_valX.shape[0], rgb_valX.shape[-1], rgb_valX.shape[2], rgb_valX.shape[3]))

                # Convert all values to float for backprop and evaluation calculations
                val_X, val_y, val_e = val_X.float().to(device), val_y.float().to(device), val_e.float().to(device)

                # Forward pass through the model
                val_risk_pred = model(val_X)

                # Calculate loss and evaluation metrics
                val_cox_loss = criterion(-risk_pred, y, e, model)
                valLossMeter.update(val_cox_loss.item(), y.size(0))

                val_ci = c_index(val_risk_pred, y, e)
                ciValMeter.update(val_ci.item(), y.size(0))

            print('Epoch: {} \t Train Loss: {} \t Val Loss: {:.4f} \t Train CI: {:.4f} \t Val CI: {:.3f}'.format(
                  epoch, coxLossMeter.val, valLossMeter.val, ciMeter.val, ciValMeter.val))

            if not args.testing:
                save_error(train_ci=ciMeter.val, val_ci=ciValMeter.val,
                           coxLoss=coxLossMeter.val, valCoxLoss=valLossMeter.val,
                           variance=varMeter.val, epoch=epoch, slname=save_eval_fname)

        else:
            print('Epoch: {} \t Train Loss: {} \t Train CI: {:.4f}'.format(
                epoch, coxLossMeter.val, valLossMeter.val, ciMeter.val, ciValMeter.val))

            if not args.testing:
                save_error(train_ci=ciMeter.val, coxLoss=coxLossMeter.val,
                           variance=varMeter.val, epoch=epoch, slname=save_eval_fname)

    if not args.testing:
        plot_coxloss(save_eval_fname, model._get_name(), valid=args.validation, save=args.saveplots)
        plot_concordance(save_eval_fname, model._get_name(), valid=args.validation, save=args.saveplots)

    # TODO: add final row of average values from the AverageMeters


if __name__ == '__main__':
    train_ct()