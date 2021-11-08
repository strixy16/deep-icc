# Name: rfs_preprocessing.py
# Environment: Python 3.8
# Author: Katy Scott
# Last updated: February 28, 2021
# Data preprocessing and loading functions


import numpy as np
import os
import pandas as pd
from scipy import ndimage
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CTSurvDataset(Dataset):

    def __init__(self, info, img_path, idx, img_dim, makeRGB=False):
        """Initialize CTSurvDataset class
        Dataset for labelled CT images used in survival prediction

        Args:
            info: pandas.Dataframe, read from CSV, contains image file names, patient ID, slice ID, and RFS time and event labels
                Column titles should be: File, Pat ID, Slice Num, RFS Code, RFS Time
            img_path: string, path to folder containing image files listed in info
            idx: list, indices to include in this dataset (ex. indices of training data)
            img_dim: int, dimension of images
            makeRGB: bool, whether to load images in and convert them to RGB 3 channel or leave as 1 channel
        """

        self.info = info.iloc[idx, :]
        self.fname = np.asarray(self.info['File'])
        self.patid = np.asarray(self.info['Pat_ID'])
        self.slice = np.asarray(self.info['Slice_Num'])
        self.event = np.asarray(self.info['RFS_Code'])
        self.time = np.asarray(self.info['RFS_Time'])

        self.img_path = img_path
        self.dim = img_dim
        self.rgb = makeRGB

        # TODO: introduce MinMaxScaler and/or Normalization
        # do I actually need these?

    def __getitem__(self, index):
        e_tensor = torch.Tensor([self.event[index]]).int()
        t_tensor = torch.Tensor([self.time[index]])

        # Load in CT bin image as numpy array
        img = np.fromfile(self.img_path + self.fname[index])

        # Normalize values to be between 0 and 1 (requires 2D input, so reshape is used)
        norm_img = normalize(np.reshape(img, (self.dim, self.dim)))

        # Some models expect an RGB image, so a 3 channel version of the CT image is generated here
        if self.rgb:
            rgb_img = gray2rgb(norm_img)
            rgb_tensor = torch.from_numpy(rgb_img)
            X_tensor = rgb_tensor.permute(2, 0, 1)

        else:
            # Reshape to a 3D array (channels, height, width)
            img_3D = np.reshape(norm_img, (1, self.dim, self.dim))

            # Convert from np array to Tensor
            X_tensor = torch.from_numpy(img_3D)

        # Adding fname so can figure out which slice this is
        # Making it a list so DataLoader works properly
        fname = self.fname[index]

        return X_tensor, t_tensor, e_tensor, fname

    def __len__(self):
        return len(self.event)


class GeneSurvDataset(Dataset):
    def __init__(self, info):
        """
        Initialize GeneSurvDataset class
        Dataset for labelled genetic data used in survival prediction

        Args:
            info: pandas.Dataframe, Contains ScoutID, RFS time of event (T), and event indicator (E) values,
            and binary genetic markers
        """
        self.info = info
        self.scoutid = np.asarray(self.info['ScoutID'])
        self.event = np.asarray(self.info['RFS_Code'])
        self.time = np.asarray(self.info['RFS'])

        gene_start = self.info.columns.get_loc('RFS') + 1
        self.genes = np.asarray(self.info.iloc[:, gene_start:])
        self.num_genes = self.genes.shape[1]

    def __getitem__(self, index):
        g_tensor = torch.from_numpy(self.genes[index])
        e_tensor = torch.Tensor([self.event[index]]).int()
        t_tensor = torch.Tensor([self.time[index]])

        # No fname returned in this class because the data is associated to scoutID's, not slice files

        return g_tensor, t_tensor, e_tensor

    def __len__(self):
        return len(self.event)


class CTGeneDataset(Dataset):

    def __init__(self, info, img_path, idx, img_dim, makeRGB=False):
        """
        Initialize CTGeneDataset class
        Dataset for labelled CT images and corresponding genetic data used in survival prediction

        Args:
            info: pandas.Dataframe, read from CSV, contains image file names, patient ID, slice ID, RFS time and event
                  labels, and binary genetic markers
                  Column titles should be: File, Pat ID, Slice Num, RFS Code, RFS Time, and gene names
            img_path: string, path to folder containing image files listed in info
            idx: list, indices to include in this dataset (ex. indices of training data)
            img_dim: int, dimension of images
            makeRGB: bool, whether to load CT images in and convert them to RGB 3 channel or leave as 1 channel
        """
        self.info = info.iloc[idx, :]
        self.fname = np.asarray(self.info['File'])
        self.patid = np.asarray(self.info['Pat_ID'])
        self.slice = np.asarray(self.info['Slice_Num'])
        self.event = np.asarray(self.info['RFS_Code'])
        self.time = np.asarray(self.info['RFS_Time'])

        self.img_path = img_path
        self.dim = img_dim

        # Find index of first gene in info
        gene_start = self.info.columns.get_loc('RFS_Time') + 1
        self.genes = np.asarray(self.info.iloc[:, gene_start:])
        self.num_genes = self.genes.shape[1]

    def __getitem__(self, index):
        e_tensor = torch.Tensor([self.event[index]]).int()
        t_tensor = torch.Tensor([self.time[index]])

        # Load in CT bin image as numpy array
        img = np.fromfile(self.img_path + self.fname[index])

        # Normalize values to be between 0 and 1 (requires 2D input, so reshape is used)
        norm_img = normalize(np.reshape(img, (self.dim, self.dim)))

        # Some models expect an RGB image, so a 3 channel version of the CT image is generated here
        if self.rgb:
            rgb_img = gray2rgb(norm_img)
            rgb_tensor = torch.from_numpy(rgb_img)
            X_tensor = rgb_tensor.permute(2, 0, 1)

        else:
            # Reshape to a 3D array (channels, height, width)
            img_3D = np.reshape(norm_img, (1, self.dim, self.dim))

            # Convert from np array to Tensor
            X_tensor = torch.from_numpy(img_3D)

        g_tensor = torch.from_numpy(self.genes[index])

        # Adding fname so can figure out which slice this is
        # Making it a list so DataLoader works properly
        fname = [self.fname[index]]

        return X_tensor, g_tensor, t_tensor, e_tensor, fname

    def __len__(self):
        return len(self.event)


def load_chol_tumor(data_dir="../Data/", imdim=256, scanthresh=300, split=0.8, batch_size=32, makeRGB=False,
                    valid=False, valid_split=0.2, seed=16):
    """
    Setting up data loading for cholangio tumour images and labels

    Args:
        data_dir: string, path to Data directory containing Images and Labels
        imdim: int, size of image to load
        scanthresh: int, threshold for removeSmallScans (number of tumour pixels required)
        split: float, value for hold out validation, size of train set
        batch_size: int, number of samples per batch
        valid: bool, whether to make a validation set or not
        valid_split: float, value for hold out validation, size of validation set (percentage of training)
        seed: int, random seed value

    Returns:
        train_loader: DataLoader for train set
        valid_laoder: DataLoader for valid set (if valid option set)
        test_loader: DataLoader for test set
    """
    # Get paths to images and labels
    info_path = os.path.join(data_dir, 'Labels', 'RFS_all_tumors_zero.csv')
    z_img_path = os.path.join(data_dir, 'Images/Tumors', str(imdim), 'Zero/')

    info = pd.read_csv(info_path)

    # Filter scans with mostly background in the image
    filtered_indices = removeSmallScans(info, z_img_path, imdim, scanthresh)
    filtered_info = info.iloc[filtered_indices]

    patnum = np.asarray(filtered_info['Pat_ID'])
    event = np.asarray(filtered_info['RFS_Code'])

    if valid:
        # Split data into train, validation, and test sets
        train_idx, valid_idx, test_idx = pat_train_test_split(patnum, event, split,
                                                              valid=valid, valid_split_perc=valid_split,
                                                              seed=seed)
        # Want to do testing with a single batch
        test_batch = len(test_idx)

        # Set up data with custom Dataset class
        train_dataset = CTSurvDataset(filtered_info, z_img_path, train_idx, imdim, makeRGB)
        valid_dataset = CTSurvDataset(filtered_info, z_img_path, valid_idx, imdim, makeRGB)
        test_dataset = CTSurvDataset(filtered_info, z_img_path, test_idx, imdim, makeRGB)

        # Setting up DataLoader for train, validation and test data
        # Shuffling data so slices from same patient are not passed in next to each other
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        # Dropping last to prevent a batch with no 0 events
        valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

        return train_loader, valid_loader, test_loader

    else:
        # Split data into train and test sets
        train_idx, test_idx = pat_train_test_split(patnum, event, split, seed=seed)

        # Want to do testing with a single batch
        test_batch = len(test_idx)

        # Set up data with custom Dataset class (in rfs_utils)
        train_dataset = CTSurvDataset(filtered_info, z_img_path, train_idx, imdim, makeRGB)
        test_dataset = CTSurvDataset(filtered_info, z_img_path, test_idx, imdim, makeRGB)

        # Setting up DataLoader for train and test data
        # Shuffling data so slices from same patient are not passed in next to each other
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

        return train_loader, test_loader


def load_chol_tumor_no_label(data_dir="../Data/", imdim=256, scanthresh=300, split=0.8, batch_size=32, makeRGB=False,
                             valid=False, valid_split=0.2, seed=16):
    # Get paths to images and labels
    info_path = os.path.join(data_dir, 'Labels', str(imdim), 'nl_all_tumors_zero.csv')
    z_img_path = os.path.join(data_dir, 'Images/Tumors', str(imdim), 'Zero/')

    info = pd.read_csv(info_path)

    # Filter scans with mostly background in the image
    filtered_indices = removeSmallScans(info, z_img_path, imdim, scanthresh)
    filtered_info = info.iloc[filtered_indices]

    patnum = np.asarray(filtered_info['Pat_ID'])
    unique_patnum = np.unique(patnum)

    if valid:
        raise Exception("This hasn't been implemented yet, please use test only")

    else:
        # Split data into train and test

        train_idx, test_idx = train_test_split(unique_patnum, train_size=split, random_state=seed, shuffle=True)

        # Set up data with custom Dataset class (in rfs_utils)
        train_dataset = CTSurvDataset(filtered_info, z_img_path, train_idx, imdim, makeRGB)
        test_dataset = CTSurvDataset(filtered_info, z_img_path, test_idx, imdim, makeRGB)

        # Setting up DataLoader for train and test data
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

        return train_loader, test_loader


def load_chol_tumor_w_gene(data_dir="../Data/", imdim=256, scanthresh=300, split=0.8, batch_size=32, valid=False, valid_split=0.2, seed=16):
    """
    Setting up data loading for cholangio tumour images, genetic info, and labels

    Args:
        data_dir: string, path to Data directory containing Images and Labels
        imdim: int, size of image to load
        scanthresh: int, threshold for removeSmallScans (number of tumour pixels required)
        split: float, value for hold out validation, size of train set
        batch_size: int, number of samples per batch
        valid: bool, whether to make a validation set or not
        valid_split: float, value for hold out validation, size of validation set (percentage of training)
        seed: int, random seed value

    Returns:
        train_loader: DataLoader for train set
        valid_loader: DataLoader for valid set (if valid option set)
        test_loader: DataLoader for test set
    """
    # Get paths to images and labels
    info_path = os.path.join(data_dir, 'Labels', str(imdim), 'RFS_gene_tumors_zero.csv')
    z_img_path = os.path.join(data_dir, 'Images/Tumors', str(imdim), 'Zero/')

    info = pd.read_csv(info_path)

    # Fixing columns with illegal characters in the name
    info.rename(columns={'CDKN2A.DEL': 'CDKN2A_DEL', 'TGF-Beta_Pathway': 'TGF_Beta_Pathway'}, inplace=True)

    # Filter scans with mostly background in the image
    filtered_indices = removeSmallScans(info, z_img_path, imdim, scanthresh)
    filtered_info = info.iloc[filtered_indices]

    patnum = np.asarray(filtered_info['Pat_ID'])
    event = np.asarray(filtered_info['RFS_Code'])

    if valid:
        # Split data into train, validation, and test sets
        train_idx, valid_idx, test_idx = pat_train_test_split(patnum, event, split,
                                                              valid=valid, valid_split_perc=valid_split,
                                                              seed=seed)

        # Want to do testing with a single batch
        # This is causing a problem with running out of GPU space
        test_batch = len(test_idx)

        # Set up data with custom Dataset class (in rfs_utils)
        train_dataset = CTGeneDataset(filtered_info, z_img_path, train_idx, imdim)
        valid_dataset = CTGeneDataset(filtered_info, z_img_path, valid_idx, imdim)
        test_dataset = CTGeneDataset(filtered_info, z_img_path, test_idx, imdim)

        # Setting up DataLoader for train, validation and test data
        # Shuffling data so slices from same patient are not passed in next to each other
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        # Dropping last to prevent a batch with no 0 events
        valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

        return train_loader, valid_loader, test_loader

    else:
        # Split data into train and test sets
        train_idx, test_idx = pat_train_test_split(patnum, event, split, seed=seed)

        # Want to do testing with a single batch
        # This is causing a problem with running out of GPU space
        test_batch = len(test_idx)

        # Set up data with custom Dataset class (in rfs_utils)
        train_dataset = CTGeneDataset(filtered_info, z_img_path, train_idx, imdim)
        test_dataset = CTGeneDataset(filtered_info, z_img_path, test_idx, imdim)

        # Setting up DataLoader for train and test data
        # Shuffling data so slices from same patient are not passed in next to each other
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=True)

        return train_loader, test_loader


def pat_train_test_split(pat_num, label, split_perc=0.1, valid=False, valid_split_perc=0.2, seed=16):
    """
    Function to split data into training and testing, keeping slices from one patient in one class
    Args:
        pat_num - numpy array of patient numbers or ids to be split
        label - numpy array of binary labels for the data, indicating recurrence or non-recurrence (censoring)
        split_perc - float value, between 0 and 1, percentage of data to put in training set, 1 - split_perc will be the testing size
        valid - boolean, whether to make a validation set or not
        valid_split_perc - float value, between 0 and 1, percentage of data to put in validation set, will be taken out of training set
        seed - seed for patient index shuffling
    Returns:
        sets - tuple of training and testing slice indices in a list (and validation if valid = 1)
    """
    # Checking that split percentage is between 0 and 1 to print better error message
    if split_perc > 1.0 or split_perc < 0.0:
        print("Invalid split percentage. Must be between 0 and 1.")
        return -1

    if valid:
        if valid_split_perc > 1.0 or valid_split_perc < 0.0:
            print("Invalid validation split percentage. Must be between 0 and 1.")
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

    if valid == 1:
        split_tr_z = int(valid_split_perc * len(train_z_pat))
        split_tr_o = int(valid_split_perc * len(train_o_pat))

        valid_z_pat = train_z_pat[:split_tr_z]
        valid_o_pat = train_o_pat[:split_tr_o]

        train_z_pat = train_z_pat[split_tr_z:]
        train_o_pat = train_o_pat[split_tr_o:]


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

    if valid == 1:
        # Validation slice set for censored patients (rfs_code = 0)
        valid_z_slice = []
        for pat in valid_z_pat:
            vl_slice_z = np.asarray(np.where(pat_num == pat)).squeeze()
            if len(vl_slice_z.shape) == 0:
                vl_slice_z = np.expand_dims(vl_slice_z, axis=0)
            valid_z_slice = np.concatenate((valid_z_slice, vl_slice_z))

        # Validation slice set for non-censored patients
        valid_o_slice = []
        for pat in valid_o_pat:
            vl_slice_o = np.asarray(np.where(pat_num == pat)).squeeze()
            if len(vl_slice_o.shape) == 0:
                vl_slice_o = np.expand_dims(vl_slice_o, axis=0)
            valid_o_slice = np.concatenate((valid_o_slice, vl_slice_o))

        valid_slice = np.concatenate((valid_z_slice, valid_o_slice)).astype(int)

    # Combine censored and non-censored slice sets
    train_slice = np.concatenate((train_z_slice, train_o_slice)).astype(int)
    test_slice = np.concatenate((test_z_slice, test_o_slice)).astype(int)

    if valid == 1:
        # Tuples of indices for training, validation, and testing slices
        sets = (train_slice, valid_slice, test_slice)
    else:
        # Tuple of indices for training and testing slices
        sets = (train_slice, test_slice)

    return sets


def removeSmallScans(info, img_path, img_dim, thresh):
    """
    Filtering out CT scans with tumour pixel counts below a threshold.

    Args:
            info: pandas.Dataframe, read from CSV, contains image file names, patient ID, slice ID, and RFS time and event labels
                Column titles should be: File, Pat ID, Slice Num, RFS Code, RFS Time
            img_path: string, path to folder containing image files listed in info
            img_dim: int, dimension of images
            thresh: int, amount of image that needs to be tumour pixels to be kept in

    """
    # Get list of image file names
    fnames = np.asarray(info['File'])
    # Initialize array to store the non-zero area of each image
    non_zeros = np.zeros(len(fnames))
    # original index list
    o_idx = list(range(len(fnames)))

    for idx, name in enumerate(fnames):
        img = np.fromfile(img_path + str(name))
        img = np.reshape(img, (img_dim, img_dim))

        # Making a mask for the tumour vs. background??
        # Makes any non-background pixels = 1 (image is now binary)
        binImg = (img != 0).astype(float)
        # Fill in any holes that are within the tumour
        binImg = ndimage.binary_fill_holes(binImg).astype(float)

        area = np.count_nonzero(binImg)
        non_zeros[idx] = area
        # print(area)

    # Indices to remove that have a total area of interest below the threshold
    r_idx = np.asarray(np.where((non_zeros < thresh))).squeeze()
    # New indices
    n_idx = np.delete(o_idx, r_idx)
    # print(len(r_idx))

    return n_idx


def torchNormalize(image):
    # Function to normalize the image based on requirements defined for ResNet, GoogLeNet, AlexNet, etc.
    # https://pytorch.org/hub/pytorch_vision_resnet/
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    out = norm(image)
    return out

