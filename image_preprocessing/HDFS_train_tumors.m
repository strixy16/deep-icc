function opt = HDFS_train_tumors
    HDFS_tumor_options = all_HDFS_tumors;
    opt.TestSize = HDFS_tumor_options.TestSize;
    
    test_perc = opt.TestSize * 100;
    train_perc = 100 - (test_perc);
    
    opt.BinLoc = HDFS_tumor_options.TrainDestination;
    opt.CSVname = strcat("../../HDFS/Labels/Tumor/HCC_MCRC_ICC_HDFS_", string(train_perc),"_", string(test_perc),"_train_tumors.csv");
    opt.CSV_header = HDFS_tumor_options.CSV_header;
    
    opt.Labels = HDFS_tumor_options.TrainLabels;
end