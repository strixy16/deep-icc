function opt = HDFS_train_liver
    HDFS_liver_options = all_HDFS_liver;
    opt.TestSize = HDFS_liver_options.TestSize;
    
    test_perc = opt.TestSize * 100;
    train_perc = 100 - (test_perc);
    
    opt.BinLoc = HDFS_liver_options.TrainDestination;
    opt.CSVname = strcat("../../HDFS/Labels/Liver/HCC_MCRC_ICC_HDFS_", string(train_perc),"_", string(test_perc),"_train_liver.csv");
    opt.CSV_header = HDFS_liver_options.CSV_header;
    
    opt.Labels = HDFS_liver_options.TrainLabels;


end