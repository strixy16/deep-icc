# Name: rfs_train.py
# Environment: Python 3.9
# Author: Katy Scott
# Last updated: June 24, 2021
# Contains Dataset class and other functions used in training

# TODO: C-index function
# TODO: move patient_data_split here maybe?

import numpy as np
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
        self.time = np.asarray([self.info['RFS Time']])

        self.img_path = img_path
        self.dim = img_dim

        # TODO: introduce MinMaxScaler and/or Normalization

    def __getitem__(self, index):
        fname = self.fname[index]
        e_tensor = torch.Tensor([self.event[index]]).int()
        t_tensor = torch.Tensor([self.time[index]])

        # Load in CT bin image as numpy array
        img = np.fromfile(self.img_path + self.fname[index])
        # Reshape to a 2D array
        img_2D = np.reshape(img, (self.dim, self.dim))

        X_tensor = torch.from_numpy(img_2D)

        return X_tensor, t_tensor, e_tensor

    def __len__(self):
        return len(self.event)





