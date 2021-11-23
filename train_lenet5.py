# Code adapted from https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320

import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary

import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from rfs_models import *
from rfs_preprocessing import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.003
BATCH_SIZE = 16
N_EPOCHS = 15

IMG_SIZE = 32

DATA_DIR = '/media/katy/Data/ICC/Data'
SPLIT = 0.8


def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''

    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.float().to(device)
            y_true = y_true.long().flatten().to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''

    # temporarily change the style of the plots to seaborn
    plt.style.use('seaborn')

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss')
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs",
           xlabel='Epoch',
           ylabel='Loss')
    ax.legend()
    fig.show()

    # change the plot style to default
    plt.style.use('default')

    plt.show()


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0

    for X, y_true in train_loader:

        # ROW_IMG = 4
        # N_ROWS = 4
        #
        # fig = plt.figure()
        # for index in range(1, ROW_IMG * N_ROWS + 1):
        #     plt.subplot(N_ROWS, ROW_IMG, index)
        #     plt.axis('off')
        #     plt.imshow(X[index-1][0], cmap='gray_r')
        #
        # fig.suptitle('Cholangio Dataset - preview')
        # plt.show()

        optimizer.zero_grad()

        X = X.float().to(device)
        y_true = y_true.long().flatten().to(device)

        # Forward pass
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''

    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:
        X = X.float().to(device)
        y_true = y_true.long().flatten().to(device)

        # Forward pass and record loss
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)


def train_cholangio():
    N_CLASSES = 2
    SCAN_THRESH = 3

    info_path = os.path.join(DATA_DIR, 'Labels', 'bin_RFS_all_tumors_zero.csv')
    z_img_path = os.path.join(DATA_DIR, 'Images/Tumors', str(IMG_SIZE), 'Zero/')

    info = pd.read_csv(info_path)

    # Filter scans with mostly background in the image
    filtered_indices = removeSmallScans(info, z_img_path, IMG_SIZE, SCAN_THRESH)
    filtered_info = info.iloc[filtered_indices]

    patnum = np.asarray(filtered_info['Pat_ID'])
    unique_patnum = np.unique(patnum)
    label = np.asarray(filtered_info['RFS_Binary'])

    train_idx, test_idx = pat_train_test_split(patnum, label, SPLIT, seed=RANDOM_SEED)

    # train_idx, test_idx = train_test_split(unique_patnum, train_size=SPLIT, random_state=RANDOM_SEED, shuffle=True)

    train_dataset = CTBinSurvDataset(filtered_info, z_img_path, train_idx, IMG_SIZE)
    test_dataset = CTBinSurvDataset(filtered_info, z_img_path, test_idx, IMG_SIZE)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

    torch.manual_seed(RANDOM_SEED)
    model = LeNet5(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    summary(model, input_size=(1, 32, 32), batch_size=BATCH_SIZE)

    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, test_loader, N_EPOCHS, DEVICE)




if __name__ == '__main__':
    train_cholangio()
    # N_CLASSES = 10
    # # define transforms
    # # transforms.ToTensor() automatically scales images to [0,1] range
    # transforms = transforms.Compose([transforms.Resize((32, 32)),
    #                                  transforms.ToTensor()])
    #
    # # download and create datasets
    # train_dataset = datasets.MNIST(root='/media/katy/Data/ICC/',
    #                                train=True,
    #                                transform=transforms,
    #                                download=True)
    #
    # valid_dataset = datasets.MNIST(root='/media/katy/Data/ICC/',
    #                                train=False,
    #                                transform=transforms)
    #
    # # define the data loaders
    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=BATCH_SIZE,
    #                           shuffle=True)
    #
    # valid_loader = DataLoader(dataset=valid_dataset,
    #                           batch_size=BATCH_SIZE,
    #                           shuffle=False)
    #
    # # Display data
    # # ROW_IMG = 10
    # # N_ROWS = 5
    # #
    # # fig = plt.figure()
    # # for index in range(1, ROW_IMG * N_ROWS + 1):
    # #     plt.subplot(N_ROWS, ROW_IMG, index)
    # #     plt.axis('off')
    # #     plt.imshow(train_dataset.data[index], cmap='gray_r')
    # #
    # # fig.suptitle('MNIST Dataset - preview')
    # # plt.show()
    #
    # torch.manual_seed(RANDOM_SEED)
    # model = LeNet5(N_CLASSES).to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # criterion = nn.CrossEntropyLoss()
    #
    # summary(model, input_size=(1, 32, 32), batch_size=BATCH_SIZE)
    #
    # model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)


# # Main that runs cholangio data
# if __name__ == '__main__':
#     N_CLASSES = 2
#
#     info_path = os.path.join(DATA_DIR, 'Labels', 'bin_RFS_all_tumors_zero.csv')
#     z_img_path = os.path.join(DATA_DIR, 'Images/Tumors', str(IMG_SIZE), 'Zero/')
#
#     info = pd.read_csv(info_path)
#
#     patnum = np.asarray(info['Pat_ID'])
#     unique_patnum = np.unique(patnum)
#
#     train_idx, test_idx = train_test_split(unique_patnum, train_size=SPLIT, random_state=RANDOM_SEED, shuffle=True)
#
#     train_dataset = CTBinSurvDataset(info, z_img_path, train_idx, IMG_SIZE)
#     test_dataset = CTBinSurvDataset(info, z_img_path, test_idx, IMG_SIZE)
#
#     train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
#     test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)
#
#     torch.manual_seed(RANDOM_SEED)
#     model = LeNet5(N_CLASSES).to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     criterion = nn.CrossEntropyLoss()
#
#     summary(model, input_size=(1, 32, 32), batch_size=BATCH_SIZE)
#
#     model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, test_loader, N_EPOCHS, DEVICE)
