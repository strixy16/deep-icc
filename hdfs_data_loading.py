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
    def __init__(self, info_file, img_dir, orig_img_dim, 
                transform_list=transforms.Compose([transforms.ToTensor()])):
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
        self.transform_list = transform_list

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        """
        Getter function

        :param: idx: int, index into the dataset
        :return: X_tensor: torch.Tensor, image data for indexed patient
        :return: t_tensor: torch.Tensor, time label for indexed patient
        :return: e_tensor: torch.Tensor, event label for indexed patient
        :return: fname: str, file name for indexed patient
        """
        # set up file path for loading
        img_path = os.path.join(self.img_dir, self.img_fname.iloc[idx])
        # Load in CT image as 1D array
        img = np.fromfile(img_path)
        # Reshape into 3D (channels, height, width)
        img = np.reshape(img, (self.orig_img_dim, self.orig_img_dim))

        # print(np.mean(img), np.std(img))
        # Apply transformations to image and convert to Tensor
        # transform_list = transforms.Compose([transforms.ToTensor()
        #                                ])
        X_tensor = self.transform_list(img)
        # Get corresponding event and time label for slice
        e_tensor = torch.Tensor([self.event_label[idx]])
        t_tensor = torch.Tensor([self.time_label[idx]])

        # Get file name associated with these data
        fname = self.img_fname.iloc[idx]

        return X_tensor, t_tensor, e_tensor, fname

