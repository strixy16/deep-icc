
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
parser.add_argument('--datadir', default='/media/katy/Data/ICC/Data', type=str, help='Full path to the Data directory '
                                                                                     'where images, labels, are stored'
                                                                                     'and output will be saved.')
parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs to run')
parser.add_argument('--imdim', default=256, type=int, help='Dimension of image to load')
# parser.add_argument('--learnrate', default=3e-3, type=float, help='Starting learning rate for training')
parser.add_argument('--modelname', default='Resnet34', type=str, help='Name of model type to build and train. '
                                                                          'Current options are KT6Model, DeepConvSurv'
                                                                          'Resnet18 and Resnet34')
parser.add_argument('--randseed', default=16, type=int, help='Random seed for reproducibility')
parser.add_argument('--scanthresh', default=300, type=int, help='Threshold for number of tumour pixels to filter images'
                                                                ' through')
parser.add_argument('--validation', default=1, type=int, help='Whether to use a validation set with train and test')
parser.add_argument('--split', default=0.9, type=float, help='Fraction of data to use for training (ex. 0.8)')
parser.add_argument('--valid_split', default=0.2, type=float, help='Fraction of training data to use for hold out '
                                                                   'validation (ex. 0.2)')
parser.add_argument('--verbose', default=1, type=int, help='Levels of output: 0: none, 1: training output')
parser.add_argument('--plots', default=True, type=bool, help='Save plots of evaluation values over model training')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def define_resnet(trial, resnet_type):
    """
    Setting up Resnet model with Optuna tuning for variables

    Args:
         trial: optuna.trial.Trial, interface for parameter suggestion
         resnet_type: string, size of Resnet to build

    Returns:
        model: ResNet nn.Module, model constructed with parameters from Optuna
    """
    # Number of nodes for last two fully connected layers before output
    l2 = trial.suggest_int("l2", 200, 512)
    l3 = trial.suggest_int("l3", 4, 512)

    # Dropout values
    d1 = trial.suggest_float("d1", 0.1, 0.9)
    d2 = trial.suggest_float("d2", 0.1, 0.9)

    model = ResNet(resnet_type, l2, l3, d1, d2)
    return model


        
def objective(trial):
    # torch.cuda.empty_cache()
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get DataLoader objects for train and validation sets
    train_loader, val_loader = load_chol_tumor(args.datadir, imdim=args.imdim, scanthresh=args.scanthresh,
                                               split=args.split, batch_size=args.batchsize, valid=False,
                                               valid_split=args.valid_split, seed=args.randseed)

    # Define model
    if args.modelname == "Resnet18":
        model = define_resnet(trial, resnet_type='18')
    elif args.modelname == "Resnet34":
        model = define_resnet(trial, resnet_type='34')

    model.to(device)

    # model = select_model(args.modelname, device)

    # Setting up training hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = NegativeLogLikelihood(device)

    print("Trial:", trial.number)
    print("Parameters:", trial.params)
    # Model training
    for epoch in range(args.epochs):
        # Initialize value holders for loss, c-index, and var values
        coxLossMeter = AverageMeter()
        ciMeter = AverageMeter()
        varMeter = AverageMeter()

        try:
            model.train()
            for X, y, e, _ in train_loader:
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

                coxLossMeter.update(cox_loss.item(), y.size(0))
                varMeter.update(risk_pred.var(), y.size(0))

                if torch.isnan(risk_pred).any() or torch.isnan(y).any() or torch.isnan(e).any():
                    print(risk_pred[0])
                    print("Got NaNs in risk_pred. Pruning this trial.")
                    raise optuna.exceptions.TrialPruned()

                train_c = c_index(risk_pred, y, e)
                ciMeter.update(train_c.item(), y.size(0))

            # torch.cuda.empty_cache()

            # VALIDATION
            model.eval()
            ciValMeter = AverageMeter()
            for val_X, val_y, val_e, _ in val_loader:
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

                if torch.isnan(val_riskpred).any() or torch.isnan(val_y).any() or torch.isnan(val_e).any():
                    print(val_riskpred[0])
                    print("Got NaNs in val_riskpred. Pruning this trial.")
                    raise optuna.exceptions.TrialPruned()

                val_c = c_index(val_riskpred, val_y, val_e)
                ciValMeter.update(val_c.item(), val_y.size(0))

        except RuntimeError:
            print("Illegal memory access was encountered")
        # print('Epoch: {} \t Train Loss: {:.4f} \t Train CI: {:.3f} \t Val CI: {:.3f}'.format(epoch, coxLossMeter.avg,
        #                                                                                      ciMeter.avg,
        #                                                                                      ciValMeter.avg))
        trial.report(val_c, epoch)

        # torch.cuda.empty_cache()
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_c


if __name__ == "__main__":
    global args

    # Utilize GPUs for Tensor computations if available
    # device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training variables
    args = parser.parse_args()

    out_dir = 'Output/' + str(args.modelname) + "-" + datetime.now().strftime("%Y-%m-%d-%H%M")
    out_path = os.path.join(args.datadir, out_dir)
    # Make output folder for this run
    if not os.path.exists(out_path):
        os.makedirs(out_path)



    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

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

    # Save out parameters from best trial
    param_fname = os.path.join(out_path, 'parameters.txt')
    with open(param_fname, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        f.write('\n')
        json.dump(trial.params, f, indent=2)


    # TODO: saving out trial
    trial_fname = os.path.join(out_path, 'best_trial.pkl')
    outfile = open(trial_fname, 'wb')
    pickle.dump(trial, outfile)
    outfile.close()
