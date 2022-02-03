# Configuration file for use with HDFS training and testing

# Data settings
# Full path to data directory where images, labels, etc. are stored and output will be saved
DATA_DIR = '/Data/'
# Path from data_dir to label files
TRAIN_LABEL_FILE = 'Labels/HDFS_train_tumors.csv'
TEST_LABEL_FILE = 'Labels/HDFS_test_tumors.csv'
# Partial from data_dir to image files. From this directory select image dimension, then train and test folder
IMG_LOC_PATH = 'Images/Labelled_Tumors/'
# Dimension of CT image to load
ORIG_IMG_DIM = 221

MODEL_NAME = "HDFSModel2"

# C-index to use - if true, uses GHCI 
USE_GH = True

SEED = 16

OPTIM = 'Adam'
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.0003

# Regularization
# L1 regularization???
LOSS_WEIGHT_DECAY = 0.005
# L2 regularization
OPTIM_WEIGHT_DECAY = 0.0

# Validation
K = 5

# Mode to run HDFS_train in
TRAIN_MODE = False 
LOAD_MODE = True

# If just testing with existing model, put path to the model you wish to load here
LOAD_MODEL_PATH = '/Data/Output/HDFSModel2/2022_02_03_0134/k_cross_HDFSModel2.pt'


# Debugging
# If true, prevents output from saving
DEBUG = True