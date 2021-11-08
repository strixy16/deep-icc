from sklearn.feature_extraction.image import extract_patches_2d

from rfs_preprocessing import *
from rfs_utils import *
from rfs_models import *
import config as args


def main():
    # Utilize GPUs for Tensor computations if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## DATA LOADING ##
    if args.validation:
        # Loading the tumor images, splitting into train/valid/test based on event indicator, and setting up DataLoader
        # objects for each
        train_loader, valid_loader, test_loader = load_chol_tumor_no_label(args.datadir, imdim=args.imdim,
                                                                  scanthresh=args.scanthresh, split=args.split,
                                                                  makeRGB=args.makeRGB,
                                                                  batch_size=args.batchsize, valid=True,
                                                                  valid_split=args.valid_split, seed=args.randseed)
    else:
        # Loading the tumor images, splitting into train/test based on event indicator, and setting up DataLoader
        # objects for each
        train_loader, test_loader = load_chol_tumor_no_label(args.datadir, imdim=args.imdim, scanthresh=args.scanthresh,
                                                    split=args.split, batch_size=args.batchsize, makeRGB=args.makeRGB,
                                                    valid=False, seed=args.randseed)

    ## DATA PREPROCESSING ##

    # TODO: need to make a separate Dataset class for unlabelled data OR add RFS_Code and RFS_Time to their spreadsheet
    #  and have it as NA
    # Think I need specific loader because need to do patches before

    for X, _, _, _ in train_loader:
        print('Image shape: {}'.format(X[0]))


if __name__ == '__main__':
    main()
