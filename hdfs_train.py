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
    for X, _, _ in data_loader:

        # im = plt.imshow(X[0][0], cmap='gray', vmin=-100, vmax=300)
        # plt.savefig("../Data/Output/adjusted_img.png")

        ROW_IMG = 8
        N_ROWS = 4

        # fig = plt.figure()
        for index in range(1, ROW_IMG*N_ROWS+1):
            plt.subplot(N_ROWS, ROW_IMG, index)
            plt.axis('off')
            plt.imshow(X[index-1][0], cmap='gray', vmin=-100, vmax=300)

        # fig.suptitle('HDFS Dataset - preview')
        plt.show()


def train_epoch(model, device, dataloader, criterion, optimizer):

    model.train()
    coxLoss = 0.0
    conInd = 0.0

    # view_images(dataloader)

    for X, t, e in dataloader:
        # X = CT image
        # t = time to event
        # e = event indicator

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
    model.eval()
    coxLoss = 0.0
    conInd = 0.0

    for X, t, e in dataloader:
        X, t, e = X.float().to(device), t.float().to(device), e.float().to(device)
        risk_pred = model(X)

        val_loss = criterion(-risk_pred, t, e, model)
        coxLoss += val_loss.item() * t.size(0)

        val_ci = c_index(risk_pred, t, e)
        conInd += val_ci * t.size(0)

    return coxLoss, conInd


def kfold_train(data_info_path, data_img_path, k=None, seed=16):
    # K-fold cross validation setup
    splits = KFold(n_splits=k, shuffle=True, random_state=seed)
    foldperf = {}

    info = pd.read_csv(data_info_path)
    dataset = HDFSTumorDataset(data_info_path, data_img_path, args.ORIG_IMG_DIM)


    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(info)))):
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=args.BATCH_SIZE, sampler=train_sampler)
        valid_loader = DataLoader(dataset, batch_size=args.BATCH_SIZE, sampler=valid_sampler, drop_last=True)

        model = HDFSModel1()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.LR)
        criterion = NegativeLogLikelihood(device)

        history = {'train_loss': [], 'valid_loss': [], 'train_cind': [], 'valid_cind': []}
        for epoch in range(args.EPOCHS):
            train_loss, train_cind = train_epoch(model, device, train_loader, criterion, optimizer)
            valid_loss, valid_cind = valid_epoch(model, device, valid_loader, criterion)

            train_loss = train_loss / len(train_loader.sampler)
            train_cind = train_cind / len(train_loader.sampler)
            valid_loss = valid_loss / len(valid_loader.sampler)
            valid_cind = valid_cind / len(valid_loader.sampler)

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Validation Loss:{:.3f} "
                  "AVG Training C-index:{:.2f} % AVG Validation C-index:{:.2f} %".format(epoch + 1,
                                                                                         args.EPOCHS,
                                                                                         train_loss,
                                                                                         valid_loss,
                                                                                         train_cind,
                                                                                         valid_cind))
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_cind'].append(train_cind)
            history['valid_cind'].append(valid_cind)

        foldperf['fold{}'.format(fold+1)] = history

        print('breakpoint goes here')



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

    kfold_train(train_info_path, train_img_path, k=args.K, seed=args.SEED)

    # view_images(train_loader)


    print('breakpoint goes here')