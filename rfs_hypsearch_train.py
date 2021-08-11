# Name: rfs_hypsearch_train.py
# Environment: Python 3.9
# Author: Katy Scott
# Last updated: August 9, 2021
# rfs_train.py but rewritten for hyperparameter search


from datetime import datetime
import json
from sklearn.model_selection import KFold
from skimage.color import gray2rgb
import torch.optim
from torch.utils.data import DataLoader

import argparse
from functools import partial
import numpy as np
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from rfs_preprocessing import *
from rfs_utils import *
from rfs_models import *

parser = argparse.ArgumentParser(description='Training variables:')
parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs to run')
parser.add_argument('--imdim', default=256, type=int, help='Dimension of image to load')
parser.add_argument('--modelname', default='KT6Model', type=str, help='Name of model type to build and train. '
                                                                          'Current options are KT6Model, DeepConvSurv'
                                                                          'Resnet18 and Resnet34')
parser.add_argument('--randseed', default=16, type=int, help='Random seed for reproducibility')
parser.add_argument('--scanthresh', default=300, type=int, help='Threshold for number of tumour pixels to filter images'
                                                                ' through')
parser.add_argument('--validation', default=0, type=int, help='Select validation method from list: '
                                                              '0: hold out')
parser.add_argument('--split', default=0.8, type=float, help='Fraction of data to use for training (ex. 0.8) with'
                                                             'hold-out validation')
parser.add_argument('--verbose', default=1, type=int, help='Levels of output: 0: none, 1: training output')
parser.add_argument('--plots', default=True, type=bool, help='Save plots of evaluation values over model training')


# def main():
#     # global args, device
#
#     # Training variables
#     args = parser.parse_args()
#
#     ### Filepath Setup ###
#     info_path = '/media/katy/Data/ICC/Data/Labels/' + str(args.imdim) + '/RFS_all_tumors_zero.csv'
#     z_img_path = '/media/katy/Data/ICC/Data/Images/Tumors/' + str(args.imdim) + '/Zero/'
#     n_img_path = '/media/katy/Data/ICC/Data/Images/Tumors/' + str(args.imdim) + '/NaN/'
#     save_path = '/media/katy/Data/ICC/Data/Output/' + str(args.modelname) + "-" + datetime.now().strftime(
#         "%Y-%m-%d-%H%M") + '/'
#
#     # Make output folder for this run
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     # Saving out parameters for the run
#     # save_param_fname = os.path.join(save_path, 'parameters.txt')
#     # with open(save_param_fname, 'w') as f:
#     #     json.dump(args.__dict__, f, indent=2)
#
#     config = {
#         "batchsize": tune.choice([8, 16, 32, 64]),
#         "learnrate": tune.loguniform(1e-4, 1e-1),
#         "drop1": tune.loguniform(0.05, 0.9),
#         "drop2": tune.loguniform(0.05, 0.9)
#     }
#     scheduler = ASHAScheduler(
#         metric="loss",
#         mode="min",
#         grace_period=1,
#         reduction_factor=2
#     )
#     reporter = CLIReporter(metric_columns=["loss", "c_index", "training_iteration"])
#
#
#     ### Data Loading ###
#     info = pd.read_csv(info_path)
#
#     # Filter scans with mostly background in the image
#     filtered_indices = removeSmallScans(info, z_img_path, args.imdim, args.scanthresh)
#     filtered_info = info.iloc[filtered_indices]
#
#     patnum = np.asarray(filtered_info['Pat_ID'])
#     event = np.asarray(filtered_info['RFS_Code'])
#
#     # Split data into train and validation sets
#     train_idx, val_idx = pat_train_test_split(patnum, event, args.split, args.randseed)
#
#     # Set up data with custom Dataset class (in rfs_utils)
#     train_dataset = CTSurvDataset(filtered_info, z_img_path, train_idx, args.imdim)
#     val_dataset = CTSurvDataset(filtered_info, z_img_path, val_idx, args.imdim)
#
#     # train(config, args, train_dataset, val_dataset, )
#     result = tune.run(
#         partial(train, args=args, trainset=train_dataset, valset=val_dataset),
#         resources_per_trial={"cpu": 2, "gpu": 1},
#         config=config,
#         scheduler=scheduler,
#         progress_reporter=reporter
#     )
#
#     best_trial = result.get_best_trial("loss", "min", "last")
#     print("Best trial config: {}".format(best_trial.config))
#     print("Best trial final validation loss: {}".format(
#         best_trial.last_result["loss"]))
#     print("Best trial final validation accuracy: {}".format(
#         best_trial.last_result["accuracy"]))
#
# # def load_data(data_dir=):
#
#
# def train(config, args, trainset, valset, checkpoint_dir=None):
#     # Utilize GPUs for Tensor computations if available
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     ### Model creation ###
#     # Choosing the model type
#     if args.modelname == "KT6Model":
#         model = KT6Model(config["drop1"], config["drop2"]).to(device)
#     elif args.modelname == "DeepConvSurv":
#         model = DeepConvSurv().to(device)
#     elif args.modelname == "Resnet18":
#         model = ResNet('18').to(device)
#     elif args.modelname == "Resnet34":
#         model = ResNet('34').to(device)
#     else:
#         print("Invalid model type name.")
#         return -1
#
#     # Define loss function and optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=config["learnrate"])
#     criterion = NegativeLogLikelihood(device)
#
#     if checkpoint_dir:
#         model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
#         model.load_state_dict(model_state)
#         optimizer.load_state_dict(optimizer_state)
#
#     train_loader = DataLoader(trainset, shuffle=True, batch_size=config["batchsize"], drop_last=True)
#     val_loader = DataLoader(valset, shuffle=True, batch_size=config["batchsize"], drop_last=True)
#
#     for epoch in range(args.epochs):
#         # Initialize value holders for loss, c-index, and var values
#         coxLossMeter = AverageMeter()
#         ciMeter = AverageMeter()
#         varMeter = AverageMeter()
#
#         epoch_steps = 0
#         model.train()
#         for X, y, e in train_loader:
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
#
#             val_riskpred = model(val_X.float().to(device))
#             val_c = c_index(val_riskpred, val_y, val_e)
#             ciValMeter.update(val_c.item(), val_y.size(0))
#
#         with tune.checkpoint_dir(epoch) as checkpoint_dir:
#             path = os.path.join(checkpoint_dir, "checkpoint")
#             torch.save((model.state_dict(), optimizer.state_dict()), path)
#
#         tune.report
#
#         # Printing average loss and c-index values for the epoch
#         print('Epoch: {} \t Train Loss: {:.4f} \t Train CI: {:.3f} \t Val CI: {:.3f}'.format(epoch, coxLossMeter.avg,
#                                                                                             ciMeter.avg, ciValMeter.avg))
#
#

def load_data(data_dir="../Data/"):

    imdim = 256
    scanthresh = 300

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
    train_idx, val_idx = pat_train_test_split(patnum, event, 0.8, seed=16)

    # Set up data with custom Dataset class (in rfs_utils)
    train_dataset = CTSurvDataset(filtered_info, z_img_path, train_idx, imdim)
    val_dataset = CTSurvDataset(filtered_info, z_img_path, val_idx, imdim)

    return train_dataset, val_dataset


def train_hpo(config, checkpoint_dir="../Output/Checkpoints/", data_dir=None):
    net = KT6Model(config['drop1'], config['drop2'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    net.to(device)

    criterion = NegativeLogLikelihood(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

    # If starting from a saved checkpoint, load in details of that checkpoint
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, valset = load_data(data_dir)

    trainloader = DataLoader(trainset, batch_size=int(config["batch_size"]), shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=int(config["batch_size"]), shuffle=True, drop_last=True)

    epochs = 100
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_steps = 0

        for X, y, e in trainloader:
            count = 0
            X, y, e = X.float().to(device), y.to(device), e.to(device)

            optimizer.zero_grad()

            outputs = net(X)
            loss = criterion(-outputs, y, e, net)
            cind = c_index(outputs, y, e)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if count % 2000 == 1999:
                print("[%d, %5d] loss: %.3f, c-index: %.3f" % (epoch + 1, count + 1,
                                                running_loss / epoch_steps, cind))
                running_loss = 0.0
            count += 1

        val_loss = 0.0
        val_steps = 0
        total = 0
        cind_sum = 0
        for val_X, val_y, val_e in valloader:
            with torch.no_grad():
                vcount = 0
                val_X, val_y, val_e = val_X.float().to(device), val_y.to(device), val_e.to(device)

                outputs = net(val_X)
                loss = criterion(-outputs, val_y, val_e, net)
                v_cind = c_index(outputs, val_y, val_e)
                cind_sum += v_cind

                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), cindex=(cind_sum / val_steps))

    print("Finished Training")


def main():
    data_dir = '/media/katy/Data/ICC/Data'
    load_data(data_dir)

    # config = {
    #     "drop1": 0.7,
    #     "drop2": 0.5,
    #     "lr": 1e-3,
    #     "batch_size": 8
    # }
    #
    # train_hpo(config, checkpoint_dir=None, data_dir=data_dir)

    config = {
        "drop1": tune.loguniform(0.1, 0.9),
        "drop2": tune.loguniform(0.1, 0.9),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32])
    }

    scheduler = ASHAScheduler( metric="cindex",
                               mode="min",
                               grace_period=1,
                               reduction_factor=2)

    reporter = CLIReporter(metric_columns=["loss", "cindex", "training_iteration"])

    result = tune.run(
        partial(train_hpo, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        scheduler=scheduler,
        progress_reporter=reporter
    )


if __name__ == '__main__':
    main()
