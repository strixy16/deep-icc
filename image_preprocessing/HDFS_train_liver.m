function opt = HDFS_train_liver
    HDFS_liver_options = all_HDFS_liver;
    
    opt.TestSize = HDFS_liver_options.TestSize;
    
    test_perc = opt.TestSize * 100;
    train_perc = 100 - (test_perc);

    opt.BinLoc = strcat("/Users/katyscott/Desktop/HDFS_Project/Data/Images/Labelled_Liver/412/HCC_MCRC_ICC_", ...
        string(train_perc),"_", string(test_perc),"/train/");
    opt.CSVname = strcat("/Users/katyscott/Desktop/HDFS_Project/Data/Labels/Liver/HCC_MCRC_ICC_HDFS_", ...
        string(train_perc),"_", string(test_perc),"_train_liver.csv");
    opt.CSV_header = {'File', 'Pat_ID', 'Slice_Num', 'HDFS_Code', 'HDFS_Time'};
    
    opt.Labels = strcat("/Users/katyscott/Desktop/HDFS_Project/Data/Labels/Liver/HCC_MCRC_ICC_HDFS_liver_", ...
        string(train_perc),"_", string(test_perc),"_train.xlsx");

end