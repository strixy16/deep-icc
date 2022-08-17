% Main script to run to set up train and test sets

main_config = all_HDFS_tumors;

train_config = HDFS_train_tumors;
test_config = HDFS_test_tumors;

if train_config.TestSize ~= test_config.TestSize
    error("Test holdout percentage must be the same in the train and test config files.")
elseif train_config.TestSize ~= main_config.TestSize
    error("Test holdout percentage must be the same in the train and main config files.")
elseif test_config.TestSize ~= main_config.TestSize
    error("Test holdout percentage must be the same in the test and main config files.")
end

train_test_split_HDFS(main_config);

createCSV_HDFS(train_config)
createCSV_HDFS(test_config)