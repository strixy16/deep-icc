# Name: rfs_preprocessing.py
# Environment: Python 3.8
# Author: Katy Scott
# Last updated: February 28, 2021
# Data preprocessing and loading functions

from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import ndimage
import torch
from torch.utils.data import Dataset


class CTSurvDataset(Dataset):

    def __init__(self, info, img_path, idx, img_dim):
        """Initialize CTSurvDataset class
        Dataset for CT images used in survival prediction

        Args:
            info: pandas.Dataframe, read from CSV, contains image file names, patient ID, slice ID, and RFS time and event labels
                Column titles should be: File, Pat ID, Slice Num, RFS Code, RFS Time
            img_path: string, path to folder containing image files listed in info
            idx: list, indices to include in this dataset (ex. indices of training data)
            img_dim: int, dimension of images
        """

        self.info = info.iloc[idx, :]
        self.fname = np.asarray(self.info['File'])
        self.patid = np.asarray(self.info['Pat ID'])
        self.slice = np.asarray(self.info['Slice Num'])
        self.event = np.asarray(self.info['RFS Code'])
        self.time = np.asarray(self.info['RFS Time'])

        self.img_path = img_path
        self.dim = img_dim

        # TODO: introduce MinMaxScaler and/or Normalization

    def __getitem__(self, index):
        fname = self.fname[index]
        e_tensor = torch.Tensor([self.event[index]]).int()
        t_tensor = torch.Tensor([self.time[index]])

        # Load in CT bin image as numpy array
        img = np.fromfile(self.img_path + self.fname[index])
        # Reshape to a 3D array (channels, height, width)
        img = np.reshape(img, (1, self.dim, self.dim))

        X_tensor = torch.from_numpy(img)

        return X_tensor, t_tensor, e_tensor

    def __len__(self):
        return len(self.event)


def pat_train_test_split(pat_num, label, split_perc, seed=16):
    """
    Function to split data into training and testing, keeping slices from one patient in one class
    Args:
        pat_num - numpy array of patient numbers or ids to be split
        label - numpy array of binary labels for the data, indicating recurrence or non-recurrence (censoring)
        split_perc - float value < 1, percentage of data to put in training set, 1 - split_perc will be the testing size
        seed - seed for patient index shuffling
    Returns:
        sets - tuple of training and testing slice indices in a list
    """
    # Checking that split percentage is between 0 and 1 to print better error message
    if split_perc > 1.0 or split_perc < 0.0:
        print("Invalid split percentage. Must be between 0 and 1.")
        return -1

    # Separate out positive and negative labels to evenly distribute them between classes
    # Get index of slices with 0 and 1 label
    # z => zero, o => one
    z_idx = np.asarray(np.where(label == 0)).squeeze()
    o_idx = np.asarray(np.where(label == 1)).squeeze()

    # Get patient ids of 0 and 1 labels
    z_pats = pat_num[z_idx]
    o_pats = pat_num[o_idx]

    # Remove repeat patient ids (repeats are there because pat_nums has number for every slice)
    # u => unique
    uz_pats = np.unique(z_pats)
    uo_pats = np.unique(o_pats)

    np.random.seed(seed)
    # Shuffle patient index for splitting
    np.random.shuffle(uz_pats)
    np.random.shuffle(uo_pats)

    # Find index to split data at from input
    split_z = int(split_perc * len(uz_pats))
    split_o = int(split_perc * len(uo_pats))

    # Training patient set
    train_z_pat = uz_pats[:split_z]
    train_o_pat = uo_pats[:split_o]

    # Testing patient set
    test_z_pat = uz_pats[split_z:]
    test_o_pat = uo_pats[split_o:]

    # Training slice set for censored patients (rfs_code = 0)
    train_z_slice = []
    for pat in train_z_pat:
        tr_slice_z = np.asarray(np.where(pat_num == pat)).squeeze()
        if len(tr_slice_z.shape) == 0:
            tr_slice_z = np.expand_dims(tr_slice_z, axis=0)
        train_z_slice = np.concatenate((train_z_slice, tr_slice_z))

    # Training slice set for non-censored patients
    train_o_slice = []
    for pat in train_o_pat:
        tr_slice_o = np.asarray(np.where(pat_num == pat)).squeeze()
        if len(tr_slice_o.shape) == 0:
            tr_slice_o = np.expand_dims(tr_slice_o, axis=0)
        train_o_slice = np.concatenate((train_o_slice, tr_slice_o))

    # Testing slice set for censored patients (rfs_code = 0)
    test_z_slice = []
    for pat in test_z_pat:
        ts_slice_z = np.asarray(np.where(pat_num == pat)).squeeze()
        if len(ts_slice_z.shape) == 0:
            ts_slice_z = np.expand_dims(ts_slice_z, axis=0)
        test_z_slice = np.concatenate((test_z_slice, ts_slice_z))

    # Testing slice set for non-censored patients
    test_o_slice = []
    for pat in test_o_pat:
        ts_slice_o = np.asarray(np.where(pat_num == pat)).squeeze()
        if len(ts_slice_o.shape) == 0:
            ts_slice_o = np.expand_dims(ts_slice_o, axis=0)
        test_o_slice = np.concatenate((test_o_slice, ts_slice_o))

    # Combine censored and non-censored slice sets
    train_slice = np.concatenate((train_z_slice, train_o_slice)).astype(int)
    test_slice = np.concatenate((test_z_slice, test_o_slice)).astype(int)

    # Tuple of indices for training and testing slices
    sets = (train_slice, test_slice)
    return sets


def removeSmallScans(info, img_path, img_dim, thresh):
    """
    Filtering out CT scans with tumour pixel counts below a threshold.

    Args:
            info: pandas.Dataframe, read from CSV, contains image file names, patient ID, slice ID, and RFS time and event labels
                Column titles should be: File, Pat ID, Slice Num, RFS Code, RFS Time
            img_path: string, path to folder containing image files listed in info **with NaN backgrounds**
            img_dim: int, dimension of images
            thresh: int, amount of image that needs to be tumour pixels to be kept in

    """
    # Get list of image file names
    fnames = np.asarray(info['File'])
    # Initialize array to store the non-zero area of each image
    non_zeros = np.zeros(len(fnames))
    o_idx = list(range(len(fnames)))

    for idx, name in enumerate(fnames):
        img = np.fromfile(img_path + str(name))
        img = np.reshape(img, (img_dim, img_dim))

        # Changes all NaNs to zeros (so do you need the NaN background??)
        img[np.isnan(img)] = 0

        # Making a mask for the tumour vs. background??
        # Makes any non-background pixels = 1 (image is now binary)
        tmp = (img != 0).astype(float)
        # Fill in any holes that are within the tumour
        tmp = ndimage.binary_fill_holes(tmp).astype(float)
        # Convert the background back to NaN
        tmp[tmp == 0] = np.nan
        img = img * tmp

        binImg = (~np.isnan(img)).astype(int)
        binImg = ndimage.binary_fill_holes(binImg).astype(int)

        area = np.count_nonzero(binImg)
        non_zeros[idx] = area
        print(area)

    r_idx = np.asarray(np.where((non_zeros<thresh))).squeeze()
    n_idx = np.delete(o_idx, r_idx)
    print(len(r_idx))

    return n_idx


        # TODO: need the NaN images for this part to work, need to change that in rfs_train




