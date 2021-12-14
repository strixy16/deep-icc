# File containing functions for loading HDFS datasets
import numpy as np
import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class HDFSTumorDataset(Dataset):
    # HDFSTumorDataset
    def __init__(self, info_file, img_dir, img_dim, transform=None):
        """
        Initialize HDFSTumorDataset class
        Dataset for segmented CT tumor images with HDFS labels

        :param info_file: string, name of csv file with image files names and corresponding labels
        :param img_dir: string, path to folder containing image files listed in info_file
        :param transform: list, list of torchvision transforms to apply to images
        """

        self.info = pd.read_csv(info_file)
        self.img_fname = self.info['File']
        self.time_label = self.info['HDFS_Time']
        self.event_label = self.info['HDFS_Code']
        self.img_dir = img_dir
        self.img_dim = img_dim
        self.transform = transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        # set up file path for loading
        img_path = os.path.join(self.img_dir, self.img_fname.iloc[idx])
        # Load in CT image as 1D array
        img = np.fromfile(img_path)
        # Reshape into 3D (channels, height, width)
        img = np.reshape(img, (1, self.img_dim, self.img_dim))

        img = transforms.ToTensor()



# Need to use toTensor and resize
# transforms.Compose([tran])



