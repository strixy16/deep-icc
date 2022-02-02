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
EPOCHS = 15
LR = 0.0003

# Regularization
# L1 regularization???
LOSS_WEIGHT_DECAY = 0.025
# L2 regularization
OPTIM_WEIGHT_DECAY = 5e-05

# Validation
K = 5

# Debugging
# If true, prevents output from saving
DEBUG = False

