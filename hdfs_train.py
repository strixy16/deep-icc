import torch.cuda

from hdfs_data_loading import *
import hdfs_config as args

if __name__ == '__main__':
    train_loader = load_hdfs_train(data_dir=args.DATA_DIR,
                                   label_file_name=args.TRAIN_LABEL_FILE,
                                   img_loc_path=args.IMG_LOC_PATH,
                                   orig_img_dim=args.ORIG_IMG_DIM,
                                   )
    test_loader = load_hdfs_test(data_dir=args.DATA_DIR,
                                 label_file_name=args.TEST_LABEL_FILE,
                                 img_loc_path=args.IMG_LOC_PATH,
                                 orig_img_dim=args.ORIG_IMG_DIM
                                 )

    print('breakpoint goes here')