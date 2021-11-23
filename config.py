# Configuration file for use with rfs training

# Name of model type to build and train
# Options:
#   CNNs, take in images only (use for rfs_cttrain_v2.py)
#       - KT6Model
#       - DeepConvSurv
#       - SimpleCholangio
#       - ResNet18
#       - ResNet34
#       - LeNetCholangio
#   FCN, take in genes only (use for rfs_genetrain.py)
#       - DeepSurvGene
#   Multi-input models, combined CNN + FCN, take in both data types (use for rfs_fulltrain.py)
#       - CholClassifier18 (uses ResNet 18)
#       - CholClassifier34 (uses ResNet 34)
modelname = 'CholClassifier18'
# NOTE: IF USING RESNET OR CHOLCLASSIFIER, makeRGB MUST BE TRUE
makeRGB = True

# Data settings
# Full path to the Data directory where images, labels, etc. are stored and output will be saved
datadir = '/media/katy/Data/ICC/Data'
# Images
# Dimension of CT image to load
imdim = 256
# Threshold for number of tumor pixels to filter CT images through
scanthresh = 0

# Training variables
# Number of samples use for each training cycle
batchsize = 32
# Number of training epochs to run
epochs = 10
# Starting learning rate for training
learnrate = 0.0001
# Random seed for reproducibility
randseed = 16
# Fraction of data to use for training (ex. 0.8)
split = 0.8

# Validation settings
# Whether to use a validation set when training model
validation = False
# Fraction of training data to use for hold-out validation
valid_split = 0.2

# Output settings
# What to do with plots of evaluation values over model training.
# If false, will display instead of save.
saveplots = True
# Set this to disable saving output (e.g. plots, parameters). For use while testing scripts
testing = False


