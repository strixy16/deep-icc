import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import torch.cuda
from torch.utils.data import DataLoader, SubsetRandomSampler

from hdfs_data_loading import *
from rfs_utils import *
import hdfs_config as args


def view_images(data_loader):
    for X, _, _ in data_loader:
        ROW_IMG = 8
        N_ROWS = 4

        fig = plt.figure()
        for index in range(1, ROW_IMG*N_ROWS+1):
            plt.subplot(N_ROWS, ROW_IMG, index)
            plt.axis('off')
            plt.imshow(X[index-1][0], cmap='gray', vmin=-100, vmax=300)

        fig.suptitle('HDFS Dataset - preview')
        plt.show()


def train_epoch(model, device, dataloader, criterion, optimizer):

    model.train()
    coxLoss = 0.0
    conInd = 0.0

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

if __name__ == '__main__':
    # Preliminaries
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Random setup
    random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    np.random.seed(args.SEED)

    train_info = os.path.join(args.DATA_DIR, args.TRAIN_LABEL_FILE)
    test_info = os.path.join(args.DATA_DIR, args.TEST_LABEL_FILE)

    train_img_path = os.path.join(args.DATA_DIR, args.IMG_LOC_PATH, str(args.ORIG_IMG_DIM), 'train/')
    test_img_path = os.path.join(args.DATA_DIR, args.IMG_LOC_PATH, str(args.ORIG_IMG_DIM), 'test/')

    # K-fold cross validation setup
    splits = KFold(n_splits=args.K, shuffle=True, random_state=args.SEED)
    foldperf={}



    # view_images(train_loader)


    print('breakpoint goes here')