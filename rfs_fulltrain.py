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
parser.add_argument('--learnrate', default=3e-3, type=float, help='Starting learning rate for training')
parser.add_argument('--modelname', default='Resnet18', type=str, help='Name of model type to use for CNN half of model.'
                                                                       'Current options are Resnet18 and Resnet34')
parser.add_argument('--randseed', default=16, type=int, help='Random seed for reproducibility')
parser.add_argument('--scanthresh', default=300, type=int, help='Threshold for number of tumour pixels to filter images'
                                                                ' through')
parser.add_argument('--validation', default=0, type=int, help='Select validation method from list: '
                                                              '0: hold out, 1: k-fold')
parser.add_argument('--split', default=0.9, type=float, help='Fraction of data to use for training (ex. 0.8) with'
                                                             'hold-out validation')
parser.add_argument('--kfold_num', default=5, type=int, help='If using k-fold cross validation, supply k value')
parser.add_argument('--verbose', default=1, type=int, help='Levels of output: 0: none, 1: training output')
parser.add_argument('--plots', default=True, type=bool, help='Save plots of evaluation values over model training')


def main():
    global args, device

    # Utilize GPUs for Tensor computations if available
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    args = parser.parse_args()

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

    # DATA LOADING
    # This does hold-out validation
    train_loader, valid_loader, test_loader = load_chol_tumor_w_gene(args.datadir, split=args.split, valid=True)

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

    save_eval_fname = os.path.join(out_path, 'convergence.csv')


    # Training

    for epoch in range(0, args.epochs):
        # Initialize value holders for loss, c-index, and var values
        coxLossMeter = AverageMeter()
        ciMeter = AverageMeter()
        varMeter = AverageMeter()

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

            X, g, y, e = X.float().to(device), g.to(device), y.float().to(device), e.float().to(device)

            risk_pred = model(X, g)

            cox_loss = criterion(-risk_pred, y, e, model)
            train_loss = cox_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            coxLossMeter.update(cox_loss.item(), y.size(0))
            varMeter.update(risk_pred.var(), y.size(0))
            train_c = c_index(risk_pred, y, e)
            ciMeter.update(train_c.item, y.size(0))

            # Printing average loss and c-index values for the epoch
            print(
                'Epoch: {} \t Train Loss: {:.4f} \t Train CI: {:.3f}'.format(epoch, coxLossMeter.avg,
                                                                                               ciMeter.avg))


if __name__ == '__main__':
    main()