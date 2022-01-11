import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch.cuda
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from hdfs_data_loading import *
from rfs_utils import *
import hdfs_config as args


class HDFSModel1(nn.Module):
    def __init__(self):
        super(HDFSModel1, self).__init__()
        # ImgIn shape = (?, 1, 221, 221)
        # Conv -> (?, 16, 72, 72)
        # Pool -> (?, 16, 36, 36)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=3),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        # Conv -> (?, 16, 16, 16)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=2),
            nn.SELU(),
            nn.Dropout(0.5)
        )
        # Conv -> (?, 8, 7, 7)
        # Pool -> (?, 8, 3, 3)
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # FC1 -> 8*3*3 inputs -> 8 outputs
        # FC2 -> 8 -> 3
        # Final -> 3 -> 1
        self.layer4 = nn.Sequential(
            nn.Linear(8*3*3, 8),
            nn.SELU(),
            nn.Linear(8, 3),
            nn.SELU(),
            nn.Linear(3, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        return x


def view_images(data_loader):
    # Function to view images that are loaded in in the correct way
    for X, _, _ in data_loader:

        # im = plt.imshow(X[0][0], cmap='gray', vmin=-100, vmax=300)
        # plt.savefig("../Data/Output/adjusted_img.png")

        # Function assumes batch size of 32
        # Number of columns to make in subplot
        N_COL = 8
        # Number of rows to make in subplot
        N_ROWS = 4

        # fig = plt.figure() # Uncomment if you want each batch to have their own figure
        for index in range(1, N_COL*N_ROWS+1):
            # Select subplot to put current image in
            plt.subplot(N_ROWS, N_COL, index)
            # Turn off cartesian plane
            plt.axis('off')
            # Display figure with correct grayscale range
            plt.imshow(X[index-1][0], cmap='gray', vmin=-100, vmax=300)

        # fig.suptitle('HDFS Dataset - preview')
        # Display the complete figure
        plt.show()


def train_epoch(model, device, dataloader, criterion, optimizer):
    # Function to train a deep learning model for some number of epochs
    # Set model to training mode so weights are updated
    model.train()
    # Initialize loss counter
    coxLoss = 0.0
    # Initialize c-index counter
    conInd = 0.0

    # use to confirm dataloader is passed correctly
    # view_images(dataloader)

    # Iterate over each batch in the data
    for X, t, e in dataloader:
        # X = CT image
        # t = time to event
        # e = event indicator

        # Convert all data to floats and store on CPU or GPU (if available)
        X, t, e = X.float().to(device), t.float().to(device), e.float().to(device)

        # Forward pass through model
        risk_pred = model(X)

        # Calculate loss and evaluation metrics
        train_loss = criterion(-risk_pred, t, e, model)
        coxLoss += train_loss.item() * t.size(0)

        train_ci = c_index(risk_pred, t, e)
        conInd += train_ci.item() * t.size(0)

        # Updating parameters based on forward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    return coxLoss, conInd


def valid_epoch(model, device, dataloader, criterion):
    # Function to run validation on a deep learning model
    # Set model to evaluation so weights are not updated
    model.eval()
    # Initialize loss counter
    coxLoss = 0.0
    # Initialize c-index counter
    conInd = 0.0

    # Iterate over each batch in the data
    for X, t, e in dataloader:
        # X = CT image
        # t = time to event
        # e = event indicator

        # Convert all data to floats and store on CPU or GPU (if available)
        X, t, e = X.float().to(device), t.float().to(device), e.float().to(device)

        # Pass data forward through trained model
        risk_pred = model(X)

        # Calculate loss and evaluation metrics
        val_loss = criterion(-risk_pred, t, e, model)
        coxLoss += val_loss.item() * t.size(0)

        val_ci = c_index(risk_pred, t, e)
        conInd += val_ci * t.size(0)

    return coxLoss, conInd


def kfold_train(data_info_path, data_img_path, k=None, seed=16):
    # K-fold cross validation setup
    splits = KFold(n_splits=k, shuffle=True, random_state=seed)
    foldperf = {}

    # Load info spreadsheet for use in separating data into folds
    info = pd.read_csv(data_info_path)
    # Load training dataset
    dataset = HDFSTumorDataset(data_info_path, data_img_path, args.ORIG_IMG_DIM)

    # Split data into k-folds and train/validate a model for each fold
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(info)))):
        # Output current fold number
        print('Fold {}'.format(fold + 1))
        # Select indices to use for train part of fold
        train_sampler = SubsetRandomSampler(train_idx)
        # Select indices to use for validation part of fold
        valid_sampler = SubsetRandomSampler(val_idx)

        # Setup dataloader for training portion of data
        train_loader = DataLoader(dataset, batch_size=args.BATCH_SIZE, sampler=train_sampler)
        # Setup dataloader for validation portion of data
        valid_loader = DataLoader(dataset, batch_size=args.BATCH_SIZE, sampler=valid_sampler, drop_last=True)

        # Create model to train for this fold
        model = HDFSModel1()
        # Save model to CPU or GPU (if available)
        model.to(device)
        # Set optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.LR)
        # Set loss function
        criterion = NegativeLogLikelihood(device)

        # Initialize dictionary to save evaluation metrics and trained model for each fold
        history = {'train_loss': [], 'valid_loss': [], 'train_cind': [], 'valid_cind': [], 'model': None}
        for epoch in range(args.EPOCHS):
            # Train the model
            train_loss, train_cind = train_epoch(model, device, train_loader, criterion, optimizer)
            # validate the model
            valid_loss, valid_cind = valid_epoch(model, device, valid_loader, criterion)

            # Get average metrics for this epoch
            train_loss = train_loss / len(train_loader.sampler)
            train_cind = train_cind / len(train_loader.sampler)
            valid_loss = valid_loss / len(valid_loader.sampler)
            valid_cind = valid_cind / len(valid_loader.sampler)

            # Display average metrics for this epoch
            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Validation Loss:{:.3f} "
                  "AVG Training C-index:{:.2f}  AVG Validation C-index:{:.2f} ".format(epoch + 1,
                                                                                         args.EPOCHS,
                                                                                         train_loss,
                                                                                         valid_loss,
                                                                                         train_cind,
                                                                                         valid_cind))
            # Store average metrics for this epoch
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_cind'].append(train_cind)
            history['valid_cind'].append(valid_cind)
        # END epoch loop

        # Store trained model for this fold
        history['model'] = model
        # Store data for this fold
        foldperf['fold{}'.format(fold+1)] = history
    # END k-fold loop

    # Calculate average performance of each fold (finding mean from all epochs)
    trainl_f, validl_f, trainc_f, validc_f = [], [], [], []
    for f in range(1,args.K+1):
        # Average Train loss for fold f
        trainl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        # Average validation loss for fold f
        validl_f.append(np.mean(foldperf['fold{}'.format(f)]['valid_loss']))
        # Average training c-index for fold f
        trainc_f.append(np.mean(foldperf['fold{}'.format(f)]['train_cind']))
        # Average validation c-index for fold f
        validc_f.append(np.mean(foldperf['fold{}'.format(f)]['valid_cind']))

    print('Performance of {} fold cross validation'.format(args.K))
    # Print the average loss and c-index for training and validation across all folds (Model performance)
    print("Average Training Loss: {:.3f} \t Average Validation Loss: {:.3f} \t "
          "Average Training C-Index: {:.2f} \t Average Validation C-Index: {:.2f}".format(np.mean(trainl_f),
                                                                                          np.mean(validl_f),
                                                                                          np.mean(trainc_f),
                                                                                          np.mean(validc_f)))
    # Find fold with best performance
    best_loss = np.amin(validl_f)
    best_cind = np.amax(validc_f)
    # fold_w_best_loss = np.where(validl_f == best_loss)
    # Get index of fold with best c-index
    fold_w_best_cind = np.where(validc_f == best_cind)
    # Get actual fold number (folds start at 1)
    best_fold = int(fold_w_best_cind[0]) + 1
    # Get model from the best fold
    best_model = foldperf['fold{}'.format(best_fold)]['model']

    print('breakpoint goes here')
    # Return model, loss, and c-ind from the best fold
    return best_model, best_loss, best_cind


def test_model(model, data_info_path, data_img_path, device):
    test_dataset = HDFSTumorDataset(data_info_path, data_img_path, args.ORIG_IMG_DIM)
    test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=True)

    model.to(device)
    model.eval()

    criterion = NegativeLogLikelihood(device)

    # Initialize dictionary to save evaluation metrics and trained model for each fold
    # history = {'test_loss': None, 'test_ci': None}

    test_loss, test_cind = valid_epoch(model, device, test_loader, criterion)

    # Get average metrics for test epoch
    test_loss = test_loss / len(test_loader.sampler)
    test_cind = test_cind / len(test_loader.sampler)

    # history['test_loss'] = test_loss
    # history['test_cind'] = test_cind

    print("Testing loss: {:.3f} \t Testing c-index: {:.2f}".format(test_loss, test_cind))

    return test_loss, test_cind


if __name__ == '__main__':
    # Preliminaries
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Random setup
    random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    np.random.seed(args.SEED)

    train_info_path = os.path.join(args.DATA_DIR, args.TRAIN_LABEL_FILE)
    test_info_path = os.path.join(args.DATA_DIR, args.TEST_LABEL_FILE)

    train_img_path = os.path.join(args.DATA_DIR, args.IMG_LOC_PATH, str(args.ORIG_IMG_DIM), 'train/')
    test_img_path = os.path.join(args.DATA_DIR, args.IMG_LOC_PATH, str(args.ORIG_IMG_DIM), 'test/')

    best_model, best_loss, best_cind = kfold_train(train_info_path, train_img_path, k=args.K, seed=args.SEED)
    torch.cuda.empty_cache()
    # TODO: run best_model through test loop
    test_loss, test_cind = test_model(best_model, test_info_path, test_img_path, device)

    # view_images(train_loader)

    print('breakpoint goes here')