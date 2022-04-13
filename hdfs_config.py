# Configuration file for use with HDFS training and testing
from torchvision import transforms

# Data settings
# Variables to set up file paths properly
# Split percentage of data to use
TEST_PERC = 10
TRAIN_PERC = 100 - TEST_PERC

# Cancer types to include in analysis (acronyms separated by _ as in the directories)
CANCER_TYPES = "HCC_MCRC_ICC"

# Type of CT image to load
IMAGE_TYPE = "Tumor"
# Dimension of CT image to load
ORIG_IMG_DIM = 221
# Transforms to apply to image
TRANSFORM_LIST = transforms.Compose([transforms.ToTensor()])

# Full path to data directory where images, labels, etc. are stored and output will be saved
DATA_DIR = '/Data/'
# Path from data_dir to label files
TRAIN_LABEL_FILE = 'Labels/Tumor/' + CANCER_TYPES + '_HDFS_' + str(TRAIN_PERC) + '_' + str(TEST_PERC) + '_train_tumors.csv'
TEST_LABEL_FILE = 'Labels/Tumor/' + CANCER_TYPES + '_HDFS_' + str(TRAIN_PERC) + '_' + str(TEST_PERC) + '_test_tumors.csv'
# Partial from data_dir to image files. From this directory select image dimension, then train and test folder
IMG_LOC_PATH = 'Images/Labelled_Tumors/' + str(ORIG_IMG_DIM) + '/' + CANCER_TYPES + '_' + str(TRAIN_PERC) + '_' + str(TEST_PERC)

# Model to train from hdfs_models
MODEL_NAME = "HDFSModel2"

# C-index to use - if true, uses GHCI 
USE_GH = False

# Random seed to set
SEED = 16

# Optimizer to use for training
OPTIM = 'Adam'

# Number of data to use in each minibatch
BATCH_SIZE = 32

# Number of training epochs to run
EPOCHS = 20

# Initial learning rate
LR = 0.0003

# Regularization
# L1 regularization???
LOSS_WEIGHT_DECAY = 0.005
# L2 regularization
OPTIM_WEIGHT_DECAY = 0.000

# Validation
K = 5

# Mode to run HDFS_train in
# Train mode creates a model and trains it from scratch using k-fold validation and tests the best fold
TRAIN_MODE = True
# Load mode loads the trained model from LOAD_MODEL_PATH and runs it through testing
LOAD_MODE = False

# If just testing with existing model, put path to the model you wish to load here
LOAD_MODEL_PATH = '/Data/Output/HDFSModel2/2022_02_03_2048_train/k_cross_HDFSModel2.pt'

# Debugging
# If true, prevents output from saving
DEBUG = True