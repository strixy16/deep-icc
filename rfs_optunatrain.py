
import argparse
from datetime import datetime
import json
import os
import optuna
from optuna.trial import TrialState
from sklearn.model_selection import KFold
from skimage.color import gray2rgb
import torch.optim as optim
from torch.utils.data import DataLoader

from rfs_preprocessing import *
from rfs_utils import *
from rfs_models import *

parser = argparse.ArgumentParser(description='Training variables:')
parser.add_argument('--batchsize', default=32, type=int, help='Number of samples used for each training cycle')
parser.add_argument('--datadir', default='/media/katy/Data/ICC/Data', type=str, help='Full path to the Data directory '
                                                                                     'where images, labels, are stored'
                                                                                     'and output will be saved.')
parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs to run')
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


def define_resnet(trial, resnet_type):
    l2 = trial.suggest_int("l2", 4, 512)
    l3 = trial.suggest_int("l3", 4, 512)
    d1 = trial.suggest_float("d1", 0.2, 0.7)
    d2 = trial.suggest_float("d2", 0.2, 0.7)

    model = ResNet(resnet_type, l2, l3, d1)
    return model


def load_chol_tumor(data_dir="../Data/", imdim=256, scanthresh=300, split=0.8, batch_size=32, seed=16):

    info_path = os.path.join(data_dir, 'Labels', str(imdim),'RFS_all_tumors_zero.csv')
    z_img_path = os.path.join(data_dir, 'Images/Tumors', str(imdim), 'Zero/')

    ## Data Loading ###
    info = pd.read_csv(info_path)

    # Filter scans with mostly background in the image
    filtered_indices = removeSmallScans(info, z_img_path, imdim, scanthresh)
    filtered_info = info.iloc[filtered_indices]

    patnum = np.asarray(filtered_info['Pat_ID'])
    event = np.asarray(filtered_info['RFS_Code'])

    # Split data into train and validation sets
    train_idx, val_idx = pat_train_test_split(patnum, event, split, seed=seed)

    # Set up data with custom Dataset class (in rfs_utils)
    train_dataset = CTSurvDataset(filtered_info, z_img_path, train_idx, imdim)
    val_dataset = CTSurvDataset(filtered_info, z_img_path, val_idx, imdim)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    return train_loader, val_loader


# def train(model, epochs, optimizer, criterion, train_loader, val_loader, trial, save_eval_fname, plots=True, verbose=1):
#     """
#     Function to train a given neural network model
#
#     Args:
#          model - nn.Module, model to train
#          epochs - int, number of epochs to train model for
#          optimizer - torch.optim object, optimizer to use in model training
#          criterion - , loss function to use in optimization
#          save_eval_fname - string, filename + path to save training results
#          plots - bool, whether to save out loss and c-index plots over training
#
#     Returns:
#         ciMeter.avg:
#         ciValMeter.avg
#         coxLossMeter.avg
#
#     """
#     for epoch in range(0, epochs):
#         # Initialize value holders for loss, c-index, and var values
#         coxLossMeter = AverageMeter()
#         ciMeter = AverageMeter()
#         varMeter = AverageMeter()
#
#         # TRAINING
#         model.train()
#         for X, y, e in train_loader:
#             # Resnet models expect an RGB image - generate a 3 channel version of CT image here
#             if isinstance(model, ResNet):
#                 # Convert grayscale image to rgb to generate 3 channels
#                 rgb_X = gray2rgb(X)
#                 # Reshape so channels is second value
#                 rgb_X = torch.from_numpy(rgb_X)
#                 X = torch.reshape(rgb_X, (rgb_X.shape[0], rgb_X.shape[-1], rgb_X.shape[2], rgb_X.shape[3]))
#
#             risk_pred = model(X.float().to(device))
#             # Calculate loss
#             cox_loss = criterion(-risk_pred, y.to(device), e.to(device), model)
#             train_loss = cox_loss
#
#             optimizer.zero_grad()
#             train_loss.backward()
#             optimizer.step()
#             coxLossMeter.update(cox_loss.item(), y.size(0))
#             varMeter.update(risk_pred.var(), y.size(0))
#             train_c = c_index(risk_pred, y, e)
#             ciMeter.update(train_c.item(), y.size(0))
#
#         # VALIDATION
#         model.eval()
#         ciValMeter = AverageMeter()
#         for val_X, val_y, val_e in val_loader:
#             # Resnet models expect an RGB image - generate a 3 channel version of CT image here
#             if isinstance(model, ResNet):
#                 # Convert grayscale image to rgb to generate 3 channels
#                 rgb_valX = gray2rgb(val_X)
#                 # Reshape so channels is second value
#                 rgb_valX = torch.from_numpy(rgb_valX)
#                 val_X = torch.reshape(rgb_valX, (rgb_valX.shape[0], rgb_valX.shape[-1], rgb_valX.shape[2], rgb_valX.shape[3]))
#
#             val_riskpred = model(val_X.float().to(device))
#             val_c = c_index(val_riskpred, val_y, val_e)
#             ciValMeter.update(val_c.item(), val_y.size(0))
#
#         if verbose > 0:
#             # Printing average loss and c-index values for the epoch
#             print('Epoch: {} \t Train Loss: {:.4f} \t Train CI: {:.3f} \t Val CI: {:.3f}'.format(epoch, coxLossMeter.avg, ciMeter.avg, ciValMeter.avg))
#
#         # Saving average results for this epoch
#         # save_error(ciMeter.avg, ciValMeter.avg, coxLossMeter.avg, varMeter.avg, epoch, save_eval_fname)
#
#         # TODO: need to figure out how to return c-index/make it what is being maximized?

        
def objective(trial):
    global args, device

    # Utilize GPUs for Tensor computations if available
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training variables
    args = parser.parse_args()

    train_loader, val_loader = load_chol_tumor(args.datadir, imdim=args.imdim, scanthresh=args.scanthresh, split=args.split,
                                               batch_size=args.batchsize, seed=args.randseed)

    if args.modelname == "Resnet18":
        model = define_resnet(trial, resnet_type='18').to(device)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = NegativeLogLikelihood(device)

    for epoch in range(args.epochs):
        print(epoch)
        model.train()
        for X, y, e in train_loader:
            # Resnet models expect an RGB image - generate a 3 channel version of CT image here
            if isinstance(model, ResNet):
                # Convert grayscale image to rgb to generate 3 channels
                rgb_X = gray2rgb(X)
                # Reshape so channels is second value
                rgb_X = torch.from_numpy(rgb_X)
                X = torch.reshape(rgb_X, (rgb_X.shape[0], rgb_X.shape[-1], rgb_X.shape[2], rgb_X.shape[3]))

            X, y, e = X.float().to(device), y.to(device), e.to(device)

            risk_pred = model(X)
            # Calculate loss
            cox_loss = criterion(-risk_pred, y, e, model)
            train_loss = cox_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if torch.isnan(risk_pred).any() or torch.isnan(y).any() or torch.isnan(e).any():
                print(risk_pred)
                print(y)
                print(e)
                print("stop here")

            train_c = c_index(risk_pred, y, e)

        # VALIDATION
        model.eval()
        ciValMeter = AverageMeter()
        for val_X, val_y, val_e in val_loader:
            # Resnet models expect an RGB image - generate a 3 channel version of CT image here
            if isinstance(model, ResNet):
                # Convert grayscale image to rgb to generate 3 channels
                rgb_valX = gray2rgb(val_X)
                # Reshape so channels is second value
                rgb_valX = torch.from_numpy(rgb_valX)
                val_X = torch.reshape(rgb_valX, (rgb_valX.shape[0], rgb_valX.shape[-1], rgb_valX.shape[2], rgb_valX.shape[3]))

            val_X, val_y, val_e = val_X.float().to(device), val_y.to(device), val_e.to(device)

            val_riskpred = model(val_X)
            val_cox_loss = criterion(-val_riskpred, val_y, val_e, model)
            val_c = c_index(val_riskpred, val_y, val_e)
            ciValMeter.update(val_c.item(), val_y.size(0))

        trial.report(val_c, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_c


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("   Number of finished trials: ", len(study.trials))
    print("   Number of pruned trials:   ", len(pruned_trials))
    print("   Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("   Value: ", trial.value)

    print("   Params: ")
    for key, value in trial.params.items():
        print("     {}: {}".format(key, value))
