# File containing functions for loading HDFS datasets
import numpy as np
import os
import pandas as pd
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class HDFSTumorDataset(Dataset):
    # HDFSTumorDataset
    def __init__(self, info_file, img_dir, orig_img_dim, transform_list=None):
        """
        Initialize HDFSTumorDataset class
        Dataset for segmented CT tumor images with HDFS labels

        :param info_file: string, name of csv file with image files names and corresponding labels
        :param img_dir: string, path to folder containing image files listed in info_file
        :param orig_img_dim: int, original dimension of image after preprocessing
        :param transform_list: list, list of torchvision transforms to apply to images
        """

        self.info = pd.read_csv(info_file)
        self.img_fname = self.info['File']
        self.time_label = np.asarray(self.info['HDFS_Time'])
        self.event_label = np.asarray(self.info['HDFS_Code'])
        self.img_dir = img_dir
        self.orig_img_dim = orig_img_dim
        # Transformations to apply to image
        # Currently just transforming to Tensor
        # TODO: add resizing, normalization, etc. as needed
        # TODO: make this an input argument properly
        self.transform_list = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        # set up file path for loading
        img_path = os.path.join(self.img_dir, self.img_fname.iloc[idx])
        # Load in CT image as 1D array
        img = np.fromfile(img_path)
        # Reshape into 3D (channels, height, width)
        img = np.reshape(img, (self.orig_img_dim, self.orig_img_dim))

        if self.transform_list:
            # Apply transformations to image and convert to Tensor
            X_tensor = self.transform_list(img)

        # Get corresponding event and time label for slice
        e_tensor = torch.Tensor([self.event_label[idx]])
        t_tensor = torch.Tensor([self.time_label[idx]])

        return X_tensor, t_tensor, e_tensor


def load_hdfs_train(data_dir="../Data/",
                    label_file_name="Labels/HDFS_train_tumors.csv",
                    img_loc_path="Images/Labelled_Tumors/",
                    orig_img_dim=221,
                    batch_size=32,
                    kfold=False,
                    seed=16):
    """
    Load HDFS image and label data for the training set

    :param data_dir: string, path to Data directory containing Images and Labels folders
    :param label_file_name: string, path from Data directory to label file, including name of file
    :param img_loc_path: string, path to image directory containing folders for dimensions of images
    :param orig_img_dim: int, original image dimension from preprocessing
    :param batch_size: int, number of samples per training batch
    :param seed: int, random seed to set all seeds for shuffling
    :return: train_loader: DataLoader for HDFS training image dataset
    """
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Get path to images and labels
    info_path = os.path.join(data_dir, label_file_name)
    img_path = os.path.join(data_dir, img_loc_path, str(orig_img_dim), 'train/')

    # Set up data with custom Dataset class
    train_dataset = HDFSTumorDataset(info_path, img_path, orig_img_dim)

    # Set up DataLoader, shuffling data so slices from same patient are not next to each other
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    return train_loader


def load_hdfs_test(data_dir="../Data/",
                   label_file_name="Labels/HDFS_test_tumors.csv",
                   img_loc_path="Images/Labelled_Tumors/",
                   orig_img_dim=221,
                   batch_size=32,
                   seed=16):
    """
    Load HDFS image and label data for the testing set

    :param data_dir: string, path to Data directory containing Images and Labels folders
    :param label_file_name: string, path from Data directory to label file, including name of file
    :param img_loc_path: string, path to image directory containing folders for dimensions of images
    :param orig_img_dim: int, original image dimension from preprocessing
    :param batch_size: int, number of samples per testing batch
    :param seed: int, random seed to set all seeds for shuffling
    :return: test_loader: DataLoader for HDFS testing image dataset
    """
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Get path to images and labels
    info_path = os.path.join(data_dir, label_file_name)
    img_path = os.path.join(data_dir, img_loc_path, str(orig_img_dim), 'test/')

    # Set up data with custom Dataset class
    test_dataset = HDFSTumorDataset(info_path, img_path, orig_img_dim)

    # Set up DataLoader, shuffling data so slices from same patient are not next to each other
    # Dropping last to prevent a batch with no 0 events
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    return test_loader

